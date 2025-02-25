import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import yaml

def calculate_mean_std(data_dir, batch_size, num_workers):
    """
    Calculate mean and std of the dataset
    Args:
        data_dir: Path to ILSVRC subset directory
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
    """
    train_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'train')
    
    # Only convert to tensor, no normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    # Calculate mean
    print("Calculating mean...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    
    # Calculate std
    print("Calculating std...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.view(3, 1))**2).mean(2).sum(0)
    
    std = torch.sqrt(std / total_images)
    
    return mean.tolist(), std.tolist()

def save_stats(mean, std, stats_file):
    """Save mean and std to a file"""
    with open(stats_file, 'w') as f:
        f.write(f"{mean}\n")  # writes list in Python format
        f.write(f"{std}\n")

if __name__ == '__main__':
    import argparse
    
    with open("/home/ubuntu/Assignment9/config.yml", "r") as file:
            config = yaml.safe_load(file)
        
    print("Calculate dataset mean and std")
    
    try:
        mean, std = calculate_mean_std(config['data_dir'], config['batch_size'], config['workers'])
        stats_file = os.path.join(config['data_dir'], 'dataset_stats.txt')
        save_stats(mean, std, stats_file)
        print("Mean and std calculated successfully")
    except Exception as e:
        print(f"Error calculating mean and std: {e}")
        
    print("\nDataset statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Statistics saved to: {stats_file}") 