import sys
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Now import the model
from backend.model.mnist_model import MNISTModel

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import requests
import json
import random
import torch.nn.functional as F
import logging
from datetime import datetime

# Set up logging
def setup_logger():
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f'{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will print to console too
        ]
    )
    return logging.getLogger(__name__)

# check cuda availability
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # setup logger
    logger = setup_logger()
    logger.info("Starting training...")

    # load data
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,),(0.3081,))])
    
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform),
        batch_size=64,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform),
        batch_size=64,
        shuffle=True
    )

    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters())
    
    logger.info("Model and data loaders initialized")

    # training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU/CPU
            data, target = data.to(device), target.to(device)            
            optimizer.zero_grad() # # Clear previous gradients
            output = model(data) # forward pass
             # check output shape
            loss = F.cross_entropy(output, target) # compute loss
            loss.backward() # backward pass
            optimizer.step() # update weights

            running_loss += loss.item() # accumulate loss 
            preds = output.argmax(dim=1, keepdim=True)# get predictions
             # check preds shape
            correct += preds.eq(target.view_as(preds)).sum().item() # count correct predictions
            total += target.size(0) # count total samples
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # Log training metrics
        logger.info(f'Epoch: {epoch+1}/{epochs}')
        logger.info(f'Training Loss: {epoch_loss:.4f}')
        logger.info(f'Training Accuracy: {epoch_acc:.2f}%')

        # Send training metrics to server
        metrics = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_acc
        }
        requests.post("http://localhost:5000/update_training_metrics", json=metrics)

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)    
                val_loss += loss.item()
                preds = output.argmax(dim=1,keepdim=True)
                correct += preds.eq(target.view_as(preds)).sum().item()
                total += target.size(0)
        
        # Calculate and log validation metrics
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * correct / total
        
        logger.info(f'Validation Loss: {val_loss:.4f}')
        logger.info(f'Validation Accuracy: {val_acc:.2f}%')
        logger.info('-' * 60)  # Separator line

        # Send validation metrics to server
        metrics = {
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }
        requests.post("http://localhost:5000/update_validation_metrics", json=metrics)
    logger.info('Training completed!')
    logger.info(f'Final Training Accuracy: {epoch_acc:.2f}%')
    logger.info(f'Final Validation Accuracy: {val_acc:.2f}%')

    # save model
    torch.save(model.state_dict(),"./model/mnist_cnn.pth")
    logger.info("Model saved")

    # generate predictions on random test images
    model.eval()
    indices = random.sample(range(len(test_loader.dataset)),10)
    
    logger.info("Generating predictions on random test images after training")
    predictions = []
    with torch.no_grad():
        for idx in indices:
            data, label = test_loader.dataset[idx]
            
            img_array = data.squeeze().numpy()
            img_array = (img_array * 0.3081) + 0.1307  # Reverse normalization
            img_array = img_array.tolist()  # Convert to list for JSON
            
            data = data.unsqueeze(0).to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).item()
            
            predictions.append({
                'index': int(idx),
                'true': int(label),
                'predicted': int(pred),
                'image_data': img_array
            })
            print(f"Sending prediction - True: {label}, Predicted: {pred}")  # Debug print

    # Send final results
    try:
        final_results_data = {'results': predictions}
        response = requests.post("http://localhost:5000/update_final_results", json=final_results_data)
        if response.status_code == 200:
            print("Final results sent successfully")
            try:
                print(f"Server response: {response.json()}")
            except requests.exceptions.JSONDecodeError:
                print(f"Raw server response: {response.text}")
        else:
            print(f"Error sending final results. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending final results: {str(e)}")

if __name__ == "__main__":
    train()