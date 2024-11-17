import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the model
class SimpleMNist(nn.Sequential):
    def __init__(self,input_size,num_classes):
        """
        initialize convolutional layers and activation functions
        Args:
            input_size (int): The number of input features (1,28,28)
            hidden_size (int): The number of hidden units.
            num_classes (int): The number of output units.(10)
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0],out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc=nn.Linear(in_features=128*3*3,out_features=num_classes)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x
