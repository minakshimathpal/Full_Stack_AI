import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple
import random
import numpy as np

def worker_init_fn(worker_id):
    """
    Initialize workers with unique seeds
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_data():
    train_transform = transforms.Compose([
        transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.98, 1.02)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST('data', train=False, transform=test_transform)

    dataloader_args = dict(
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn
    )

    train_loader = DataLoader(train_data, **dataloader_args)
    test_loader = DataLoader(test_data, **dataloader_args)

    return train_loader, test_loader
