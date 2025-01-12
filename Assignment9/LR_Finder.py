from torch_lr_finder import LRFinder
import torch
import torchvision
import torchvision.transforms as transforms
from model.model import ResNet50
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch
import gc
import yaml
import os

num_cores = os.cpu_count()
print(f"Available CPU cores: {num_cores}")
# Load the config file

torch.cuda.empty_cache()
gc.collect()
with open("/home/ubuntu/Assignment9/config.yml", "r") as file:
    config = yaml.safe_load(file)


def initialize_model(num_classes, model_path=config["initial_model_path"]):
    # Check if a saved model exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = ResNet50(num_classes=config["num_classes"])
        model.load_state_dict(torch.load(model_path))   
    else:
        print("No saved model found, initializing a new one.")
        model = ResNet50(num_classes=num_classes)
        torch.save(model.state_dict(), config["initial_model_path"])
        print(f"Initial model saved to: {config["initial_model_path"]}")
    
    return model

def find_lr(model,output_dir='lr_finder_plots'):
    
    print(f"Find LR with params: Start_lr: {config["start_lr"]}, End_lr: {config["end_lr"]}, Num_iter: {config["num_iter"]}")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    
    training_folder_name = '/home/ubuntu/Assignment9/data_subset/CLS-LOC/train_subset'
    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=True
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["start_lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    amp_config = {
    'device_type': 'cuda',
    'dtype': torch.float16,
    }
    grad_scaler = torch.cuda.amp.GradScaler()
    lr_finder = LRFinder(model, optimizer, criterion, device=device,amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
)
    lr_finder.range_test(train_loader, start_lr=config["start_lr"], end_lr=config["end_lr"], num_iter=config["num_iter"], step_mode="exp")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp and parameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'lr_finder_{timestamp}_start{config["start_lr"]}_end{config["end_lr"]}_iter{config["num_iter"]}.png'
    filepath = os.path.join(output_dir, filename)
    
    # Plot and save
    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)
    plt.title(f'Learning Rate Finder (iter: {config["num_iter"]})')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {filepath}")
    lr_finder.reset()

if __name__ == "__main__":
    model = initialize_model(num_classes=config["num_classes"])
    find_lr(model)
