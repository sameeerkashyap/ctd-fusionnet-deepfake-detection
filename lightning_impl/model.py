"""
Model definitions for CTD-FusionNet deepfake detection.
Contains all model architectures with fixes for stability issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism for combining RGB and noise features.
    Uses stable batched matrix multiplication for better numerical stability.
    """

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, q_in, k_in):
        """
        Args:
            q_in: Query tensor (B, D)
            k_in: Key tensor (B, D)

        Returns:
            Fused tensor (B, D) with attention applied
        """
        # Add batch and sequence dimensions for batched matrix multiplication
        q = self.q(q_in).unsqueeze(1)      # (B, 1, D)
        k = self.k(k_in).unsqueeze(1)      # (B, 1, D)
        v = self.v(k_in).unsqueeze(1)      # (B, 1, D)

        # Compute attention with proper scaling
        attn = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) / self.scale,  # (B, 1, 1)
            dim=-1
        )

        # Apply attention and remove sequence dimension
        out = torch.bmm(attn, v).squeeze(1)  # (B, D)
        return q_in + out  # Residual connection


class NoiseBranchCTD(nn.Module):
    """
    Noise branch that processes CTD residual images using EfficientNet-B0.
    """

    def __init__(self):
        super().__init__()
        # Feature extractor for noise images
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True
        )

        # Get the number of channels in the final feature map
        ch = self.backbone.feature_info[-1]['num_chs']

        # PRNU head for noise pattern analysis
        self.prnu_head = nn.Sequential(
            nn.Conv2d(ch, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, noise_img):
        """
        Args:
            noise_img: Noise image tensor (B, 3, H, W)

        Returns:
            feature_map: Feature map from backbone (B, C, H', W')
            prnu_map: PRNU pattern map (B, 1, H', W')
        """
        feature_map = self.backbone(noise_img)[-1]  # Final feature map
        prnu_map = self.prnu_head(feature_map)
        return feature_map, prnu_map


class FusionNetCTD(nn.Module):
    """
    Main fusion network combining RGB, noise, and spatial features.
    Uses ConvNeXt, EfficientNet, and MobileNetV3 (replacing Swin/ViT) with attention fusion.
    """

    def __init__(self, spsl_model_name="mobilenetv3_large_100", debug=False):
        super().__init__()
        self.debug = debug

        # RGB branch - ConvNeXt Tiny for high-level features
        self.rgb_branch = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            features_only=True
        )

        # Noise branch - EfficientNet B0 for noise analysis
        self.noise_branch = NoiseBranchCTD()

        # Spatial branch - MobileNetV3 for spatial reasoning (lighter than Swin/ViT)
        self.spsl_branch = timm.create_model(
            spsl_model_name,
            pretrained=True,
            num_classes=0  # Remove classification head
        )

        # Adaptive pooling to get fixed-size feature vectors
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Determine projection dimension from the spatial model
        # Run a dummy forward pass to get the actual output dimension
        with torch.no_grad():
            dummy_in = torch.zeros(1, 3, 224, 224)
            dummy_out = self.spsl_branch(dummy_in)
            spsl_features = dummy_out.shape[1]
            
        projection_dim = spsl_features  # Use the actual output dim as projection dim

        # Projection layers to common dimension
        rgb_channels = self.rgb_branch.feature_info[-1]['num_chs']
        noise_channels = self.noise_branch.backbone.feature_info[-1]['num_chs']
        # spsl_features is already determined above

        # Simplified projections (Linear only) to match checkpoints
        self.rgb_proj = nn.Linear(rgb_channels, projection_dim)
        self.noise_proj = nn.Linear(noise_channels, projection_dim)
        self.spsl_proj = nn.Linear(spsl_features, projection_dim)

        # Attention fusion
        self.attn = AttentionFusion(projection_dim)

        # Classification head (Simplified to match checkpoints)
        self.head = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),  # Concatenated features
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(projection_dim, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 2)  # Binary classification
        )
        
        if self.debug:
            print(f"[DEBUG] Model initialized with spsl_model={spsl_model_name}")
            print(f"[DEBUG] Projection dim: {projection_dim}")

    def forward(self, img, noise):
        """
        Args:
            img: RGB image tensor (B, 3, H, W)
            noise: CTD noise image tensor (B, 3, H, W)

        Returns:
            logits: Classification logits (B, 2)
        """
        if self.debug:
            print(f"[DEBUG] Input img shape: {img.shape}, noise shape: {noise.shape}")

        # RGB branch features
        rgb_map = self.rgb_branch(img)[-1]  # (B, C_rgb, H', W')
        rgb_vec = self.pool(rgb_map).flatten(1)  # (B, C_rgb)
        
        if self.debug:
            print(f"[DEBUG] RGB vec shape: {rgb_vec.shape}")

        # Noise branch features
        fmap, _ = self.noise_branch(noise)  # (B, C_noise, H', W')
        noise_vec = self.pool(fmap).flatten(1)  # (B, C_noise)
        
        if self.debug:
            print(f"[DEBUG] Noise vec shape: {noise_vec.shape}")

        # Spatial branch - resize input to 224x224 for Spatial Model
        swin_in = F.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
        spsl_vec = self.spsl_branch(swin_in)  # (B, spsl_features)
        
        if self.debug:
            print(f"[DEBUG] Spatial vec shape: {spsl_vec.shape}")

        # Project all features to common dimension
        rgb_vec = self.rgb_proj(rgb_vec)      # (B, projection_dim)
        noise_vec = self.noise_proj(noise_vec)  # (B, projection_dim)
        spsl_vec = self.spsl_proj(spsl_vec)    # (B, projection_dim)

        # Apply attention fusion to RGB and noise features
        rgb_attn = self.attn(rgb_vec, noise_vec)  # (B, projection_dim)

        # Concatenate all features
        fused = torch.cat([rgb_attn, noise_vec, spsl_vec], dim=1)  # (B, 1152)
        
        if self.debug:
            print(f"[DEBUG] Fused shape: {fused.shape}")

        # Classification
        out = self.head(fused)  # (B, 2)
        
        if self.debug:
            print(f"[DEBUG] Output shape: {out.shape}")
            
        return out
