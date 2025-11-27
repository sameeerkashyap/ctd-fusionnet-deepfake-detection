"""
Transform Module
Contains image and noise transformation pipelines for deepfake detection.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size=224, augmentation_config=None, is_train=True):
    """
    Get augmentation transforms for training or validation.

    Args:
        img_size: Target image size (default: 224)
        augmentation_config: Dictionary containing augmentation parameters.
                           If only 'normalize' is provided, skips augmentations.
                           If None or contains other augmentation params, uses defaults.
        is_train: If True, return training transforms with augmentation.
                  If False, return validation transforms (no augmentation)

    Returns:
        Albumentations Compose transform
    """
    # Default augmentation parameters
    default_config = {
        'crop_scale': (0.75, 1.0),
        'horizontal_flip_p': 0.5,
        'gauss_noise_p': 0.15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'p': 0.4
        },
        'compression': {
            'quality_range': (70, 100),
            'p': 0.2
        },
        'normalize': {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }
    }

    # Merge with provided config
    if augmentation_config is not None:
        config = default_config.copy()
        config.update(augmentation_config)
        for key in config:
            if isinstance(config[key], dict) and key in augmentation_config:
                config[key].update(augmentation_config[key])
    else:
        config = default_config

    # Check if only normalize is specified (skip augmentations)
    skip_augmentations = (
        augmentation_config is not None and
        set(augmentation_config.keys()) == {'normalize'}
    )

    if is_train and not skip_augmentations:
        # Training with augmentations
        return A.Compose([
            A.Resize(img_size, img_size),
            A.RandomResizedCrop(size=(img_size, img_size), scale=config['crop_scale']),
            A.HorizontalFlip(p=config['horizontal_flip_p']),
            A.GaussNoise(p=config['gauss_noise_p']),
            A.ColorJitter(
                config['color_jitter']['brightness'],
                config['color_jitter']['contrast'],
                config['color_jitter']['saturation'],
                config['color_jitter']['hue'],
                p=config['color_jitter']['p']
            ),
            A.ImageCompression(
                quality_range=config['compression']['quality_range'],
                p=config['compression']['p']
            ),
            A.Normalize(mean=config['normalize']['mean'], std=config['normalize']['std']),
            ToTensorV2()
        ], additional_targets={"noise": "image"})
    else:
        # Validation transforms (or training without augmentations)
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=config['normalize']['mean'], std=config['normalize']['std']),
            ToTensorV2()
        ], additional_targets={"noise": "image"})
