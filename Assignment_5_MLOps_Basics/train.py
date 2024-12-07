import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
import random
import numpy as np
from torch.backends import cudnn
import matplotlib.pyplot as plt

# Determine absolute path for the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

def set_seeds(seed=42):
    """
    Set seeds for reproducibility across all required libraries and functions
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    """
    Initialize workers with unique seeds
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def visualize_augmentations(dataset, num_samples=5):
    """
    Visualize augmented images from the dataset
    Args:
        dataset: MNIST dataset with transformations
        num_samples: Number of samples to visualize (default=5)
    """
    # Original transformations
    original_transforms = transforms.Compose([       
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
        
    # Create a figure
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    # Get some random samples
    for i in range(num_samples):
        index = random.randint(0, len(dataset) - 1)
        # Get original image
        orig_dataset = datasets.MNIST('data', train=True, download=True, 
                                    transform=original_transforms)
        orig_img, _ = orig_dataset[index]
        
        # Get augmented image
        aug_img, _ = dataset[index]
        
        # Denormalize images for visualization
        orig_img = orig_img * 0.3081 + 0.1307
        aug_img = aug_img * 0.3081 + 0.1307
        
        # Plot original image
        axes[0, i].imshow(orig_img.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Plot augmented image
        axes[1, i].imshow(aug_img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    
    # Save the figure
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    # Save the augmented images
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'augmentation_samples.png'))
    plt.close()

def train():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST dataset
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
        
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transforms)
    
    # Visualize augmentations - just pass the dataset
    visualize_augmentations(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        worker_init_fn=worker_init_fn  # Add worker initialization
    )
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{timestamp}.pth"))
    
if __name__ == "__main__":
    train() 