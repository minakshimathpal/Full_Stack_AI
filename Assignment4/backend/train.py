import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from model import SimpleMNIST

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MNISTrainer:
    def __init__(self, input_size=(1,28,28), num_classes=10, batch_size=64, learning_rate=0.001, num_epochs=10):
        # Combined version
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = SimpleMNist(input_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_loader = self._get_data_loaders()

