"""
Lightning Classification Package

A basic image classification deep learning network using PyTorch Lightning
with comprehensive debugging statements.
"""

from .model import BasicImageClassifier
from .datamodule import ImageDataModule

__version__ = "1.0.0"
__all__ = ["BasicImageClassifier", "ImageDataModule"]
