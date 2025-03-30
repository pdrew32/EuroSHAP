import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import pytorch_lightning as pl

class SatResNet(nn.Module):
    def __init__(self, num_classes: int=10):
        """
        Initialize a Resnet18-based model for satellite image classification

        Parameters
        ----------
        num_classes : int, optional
            the number of target classes (defaults to 10)
        pretrained : bool, optional
            If true, uses the pretrained resnet18 model.
        """
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        # replace the final fully connected layer
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
    

class LightningClassifier(pl.LightningModule):
    def __init__(self, model, lr=0.001, criterion=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, _):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer