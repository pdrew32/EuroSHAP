import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

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