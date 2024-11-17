import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from model import SimpleMNIST

# set device



class MNISTrainer:
    def __init__(self, input_size=(1,28,28), num_classes=10, batch_size=64, learning_rate=0.001, num_epochs=10):
        # Combined version
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize model, criterion, optimizer
        self.model = SimpleMNist(input_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize data loaders
        self.train_loader ,self.test_loader = self._get_data_loaders()

        # Initialize lists to store metrics
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        #create logs directory
        if not os.path.exists('logs'):
            os.makedirs('logs')

    def __get_data_loaders(self):
        """Initialize and return data loaders"""
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='./data', 
                                       train=True, 
                                       download=True, 
                                       transform=transform)
        test_dataset = datasets.MNIST(root='./data', 
                                      train=False, 
                                      download=True, 
                                      transform=transform)
        
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True)
        
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=self.batch_size, 
                                 shuffle=False)
        return train_loader, test_loader

