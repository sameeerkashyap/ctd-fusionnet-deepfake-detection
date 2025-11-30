import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from typing import Tuple, Dict, Any


class DeepfakeClassifier(pl.LightningModule):
    """
    Binary Deepfake Classification Network
    Classifies images as Real (0) or Fake (1)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING DEEPFAKE MODEL")
        print(f"{'='*60}")
        print(f"Number of classes: {num_classes}")
        print(f"Learning rate: {learning_rate}")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        # Input: 224x224 -> Pool1: 112 -> Pool2: 56 -> Pool3: 28 -> Pool4: 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, num_classes)
        
        # Metrics - Using Multiclass with 2 classes for simplicity with CrossEntropy
        task = "multiclass"
        
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)
        self.test_acc = Accuracy(task=task, num_classes=num_classes)
        
        self.train_precision = Precision(task=task, num_classes=num_classes, average='macro')
        self.val_precision = Precision(task=task, num_classes=num_classes, average='macro')
        
        self.train_recall = Recall(task=task, num_classes=num_classes, average='macro')
        self.val_recall = Recall(task=task, num_classes=num_classes, average='macro')
        
        self.train_f1 = F1Score(task=task, num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task=task, num_classes=num_classes, average='macro')
        
        print(f"Model initialized successfully")
        print(f"{'='*60}\n")
        
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        batch_size = x.size(0)
        
        if debug:
            print(f"Input: {x.shape}")
        
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # FC Block 1
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        
        # FC Block 2
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        
        # Output
        x = self.fc3(x)
        
        return x
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        
        # Debug first batch of epoch
        debug = (batch_idx == 0)
        if debug:
            print(f"\n[Train] Batch {batch_idx}: x={x.shape}, y={y.shape}")
        
        logits = self(x, debug=debug)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Metrics
        self.train_acc(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_f1(preds, y)
        
        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Metrics
        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        
        # Log
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
