
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleClassifier(nn.Module):
    """
    A simple classification model using ResNet18 backbone.
    No fancy fusion or noise branches.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_prob=0.5):
        super(SimpleClassifier, self).__init__()
        # Use ResNet18 as it is fast and effective for basic debugging
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        
        # Remove the original fc layer
        self.backbone.fc = nn.Identity()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # For binary classification, output a single logit
        self.classifier = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Classify
        return self.classifier(features)
