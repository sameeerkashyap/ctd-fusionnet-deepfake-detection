"""
PyTorch Lightning Module for CTD-FusionNet.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
try:
    from .model import FusionNetCTD
except ImportError:
    try:
        from lightning_impl.model import FusionNetCTD
    except ImportError:
        from model import FusionNetCTD

class FusionNetLightning(pl.LightningModule):
    def __init__(self, 
                 learning_rate=1e-4, 
                 weight_decay=1e-4, 
                 num_epochs=10, 
                 spsl_model_name="mobilenetv3_large_100",
                 debug=False):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = FusionNetCTD(spsl_model_name=spsl_model_name, debug=debug)
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")
        
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")

    def forward(self, img, noise):
        return self.model(img, noise)

    def training_step(self, batch, batch_idx):
        img, noise, labels = batch
        outputs = self(img, noise)
        loss = self.criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)[:, 1]
        self.train_acc(probs, labels)
        self.train_auc(probs, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, noise, labels = batch
        outputs = self(img, noise)
        loss = self.criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)[:, 1]
        self.val_acc(probs, labels)
        self.val_auc(probs, labels)
        self.val_f1(probs, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        img, noise, labels = batch
        outputs = self(img, noise)
        loss = self.criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)[:, 1]
        self.test_acc(probs, labels)
        self.test_auc(probs, labels)
        self.test_f1(probs, labels)
        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', self.test_acc, on_epoch=True, prog_bar=True)
        self.log('test_auc', self.test_auc, on_epoch=True, prog_bar=True)
        self.log('test_f1', self.test_f1, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
