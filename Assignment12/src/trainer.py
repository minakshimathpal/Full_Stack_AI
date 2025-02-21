# script to train the transformer model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import numpy as np
import os, time
import argparse
from tqdm import tqdm
from model import GPT
from config import  GPTConfig
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import mlflow
import tiktoken

def select_device():
    """Select the best available device (CUDA, MPS, or CPU)."""
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return device

def set_seed(seed=1337):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_generation_config():
    """Get default configuration for text generation."""
    return {
        'num_return_sequences': 5,
        'max_length': 30
    }

# Use the functions
device = select_device()
set_seed()
gen_config = get_generation_config()

# STOP
num_return_sequences = gen_config['num_return_sequences']
max_length = gen_config['max_length']

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('/content/drive/MyDrive/Assignment12/data/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


model = GPT.from_pretrained('gpt2')
model.to(device)

def train(config, model, optimizer, scheduler, train_loader):  # _ instead of val_loader
    # Get device and set seed
    device = select_device()
    set_seed()
    
    # Initialize loss tracking
    best_loss = float('inf')
    train_losses = []
    
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("gpt-training")
    
    # Training Loop
    steps_per_epoch = len(train_loader.tokens) // (batches * no_of_tokens)
    print(f"Steps per epoch: {steps_per_epoch}")
    EPOCHS = 50

    # Start MLflow run
    with mlflow.start_run(run_name="gpt-training"):
        # Log parameters
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": batches,
            "sequence_length": no_of_tokens,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "scheduler_step_size": scheduler.step_size,
            "scheduler_gamma": scheduler.gamma,
            "device": device,
            "model_type": model.__class__.__name__,
            "total_params": sum(p.numel() for p in model.parameters())
        })

        for epoch in range(EPOCHS):
            model.train()
            train_loss_list = []
            
            # Training loop
            for step in range(steps_per_epoch):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
                
                train_loss_list.append(loss.item())
            
            # Calculate average training loss
            epoch_train_loss = sum(train_loss_list) / len(train_loss_list)
            train_losses.append(epoch_train_loss)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f}")
            
            # Step the scheduler
            scheduler.step()
            
            # Save if it's the best model
            if epoch_train_loss < best_loss:
                best_loss = epoch_train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_train_loss,
                }, "best_model.pth")
                mlflow.log_artifact("best_model.pth")
                print(f"Saved best model with loss: {epoch_train_loss:.4f}")
            
            # Early stopping
            if epoch_train_loss < 0.099999:
                print(f"Early stopping at epoch {epoch + 1} with loss {epoch_train_loss:.4f}")
                break

if __name__ == "__main__":
    batches, no_of_tokens = 16, 256
    train_loader = DataLoaderLite(B=batches, T=no_of_tokens)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    train(GPTConfig(), model, optimizer, scheduler, train_loader)