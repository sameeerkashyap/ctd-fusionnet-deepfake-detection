
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleClassifier(nn.Module):
    """
    A simple classification model using ResNet18 backbone.
    No fancy fusion or noise branches.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleClassifier, self).__init__()
        # Use ResNet18 as it is fast and effective for basic debugging
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
