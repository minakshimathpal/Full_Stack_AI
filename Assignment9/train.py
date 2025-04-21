import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch.cuda.amp as amp
warnings.filterwarnings('ignore')
from datetime import datetime
from torchinfo import summary
import os
import time
import sys
from math import sqrt
from verify_dataset import verify_dataset_structure
from torch.cuda.amp import autocast, GradScaler
from model.model import ResNet50
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc
import yaml
torch.cuda.empty_cache()
gc.collect()
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

data_dir = '/home/ubuntu/raw_data/extracted/ILSVRC'
class ImageNetSubsetLoader:
    def __init__(self, config):
        """
        Initialize ImageNet subset data loader
        Args:
            data_dir: Path to ILSVRC subset directory
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        # Data directories
        train_dir = os.path.join(config["data_dir"], 'Data', 'CLS-LOC', 'train')
        val_dir = os.path.join(config["data_dir"], 'Data', 'CLS-LOC', 'val')
        print(train_dir)
        # # Load dataset statistics
        # stats_file = os.path.join(config["data_dir"], 'dataset_stats.txt')

        # with open(stats_file, 'r') as f:
        #     self.mean = eval(f.readline())
        #     self.std = eval(f.readline())
        
        # Create datasets
        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=self.get_transforms(train=True)
        )
        
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

        val_dataset = datasets.ImageFolder(
            val_dir,
            transform=self.get_transforms(train=False)
        )
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,            
            num_workers=config["workers"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["workers"],
            persistent_workers=True,
            pin_memory=True
        )
        
        self.num_classes = len(train_dataset.classes)

    def get_transforms(self, train=True):
            """Get data transforms using dataset statistics"""
            if train:
                return transforms.Compose([
                    transforms.RandomResizedCrop(224,interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
                ])     

class Trainer:
    def __init__(self, config):
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
       
        # Set up logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join('outputs', f'training_log_{timestamp}.txt')
        sys.stdout = Logger(self.log_file)
        
        with open("/home/ubuntu/Assignment9/config.yml", "r") as file:
            self.config = yaml.safe_load(file)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_loader = ImageNetSubsetLoader(
            self.config
        )
        
        self.train_loader = data_loader.train_loader
        self.val_loader = data_loader.val_loader
        self.num_classes = data_loader.num_classes
        
        # Initialize starting epoch and best accuracy
        self.start_epoch = 1
        self.best_acc = 0.0
        self.best_model_wts = None

        self.model = ResNet50(num_classes=self.num_classes)
        if config["initial_model_path"]:
            self.model.load_state_dict(torch.load(config["initial_model_path"]))
       
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            # Wrap model with DataParallel
            self.model = DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()      

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config["max_lr"]/config["div_factor"],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )

        total_steps = config["epochs"] * len(self.train_loader)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                            self.optimizer,
                                            max_lr=config["max_lr"],
                                            epochs= config["epochs"],
                                            steps_per_epoch=len(self.train_loader),
                                            base_momentum=config["base_momentum"],
                                            max_momentum=config["max_momentum"],
                                            cycle_momentum=True,
                                            total_steps=total_steps,
                                            pct_start=config["pct_start"],
                                            div_factor=config["div_factor"],
                                            final_div_factor=config["final_div_factor"],
                                            three_phase =True
                 )

        # Initialize mixed precision training
        self.scaler = amp.GradScaler()
        self.autocast = amp.autocast
        print("Using mixed precision training")
        
        # Resume from checkpoint if specified
        if config['resume']:           
            if os.path.isdir(config['checkpoint_dir']):                
                checkpoint_path = os.path.join(config['checkpoint_dir'], 'model_best.pth')                
                checkpoint = torch.load(checkpoint_path)

                # Handle DataParallel state dict
                if torch.cuda.device_count() > 1:
                    # If checkpoint was saved with DataParallel
                    if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # If checkpoint was saved without DataParallel
                        new_state_dict = {f'module.{k}': v for k, v in checkpoint['model_state_dict'].items()}
                        self.model.load_state_dict(new_state_dict)
                else:
                    # If using single GPU but checkpoint was saved with DataParallel
                    if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                        self.model.load_state_dict(new_state_dict)
                    else:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                # Load optimizer and scheduler states
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Set starting epoch and best accuracy
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_acc = checkpoint['best_acc']
                
                print(f"Loaded checkpoint '{config['checkpoint_dir']}/best_model.pth' (epoch {checkpoint['epoch']})")
                print(f"Previous best accuracy: {self.best_acc:.2f}%")        
            else:
                print(f"No checkpoint found at '{config['checkpoint_dir']}'")

        print("\nModel Summary:") 
        summary(self.model,input_size=(config['batch_size'], 3, 224, 224),
                device=self.device,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        print("\n")

    def train_one_epoch(self,epoch):
        
        self.model.train()
        start0 = time.time()
        running_loss = 0.0
        running_corrects = 0
        correct_top5 = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')

        for image, label in progress_bar:
            image, label = image.to(self.device), label.to(self.device)            
            self.optimizer.zero_grad()

            # Use autocast for mixed precision training
            with self.autocast():
                outputs  = self.model(image)
                _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(outputs, label)            
            # Scale loss and do backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item() * image.size(0)           
            running_corrects += torch.sum(preds == label.data)            
            total_samples += image.size(0)
           
            progress_bar.set_postfix({
                "running_corrects": f"{running_corrects}",
                "total_samples": f"{total_samples}",
                'loss': f"{running_loss / total_samples:.4f}",
                'accuracy': f"{(running_corrects / total_samples) * 100:.2f}%",
                
                })
               
        epoch_loss= running_loss / total_samples
        epoch_accuracy = (running_corrects / total_samples) * 100
        epoch_top5_accuracy = (correct_top5 / total_samples) * 100
        return epoch_loss, epoch_accuracy, epoch_top5_accuracy
            

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_top1_correct = 0
        running_top5_correct = 0
        total_samples = 0
        

        progress_bar = tqdm(self.val_loader, desc='Validating')
        with torch.no_grad(), autocast():
            for image, label in progress_bar:
                image, label = image.to(self.device), label.to(self.device)

                outputs = self.model(image)
                loss = self.loss_fn(outputs, label)


                # Calculate top-1 and top-5 accuracy
                _, top5_preds  = torch.topk(outputs, k=5, dim=1)
                labels = label.view(-1, 1).expand_as(top5_preds)  # Shape: batch_size x 5
                correct = top5_preds.eq(labels).sum().item()
                top1_preds = top5_preds[:, 0] 

                # Top-1 accuracy
                running_top1_correct += (top1_preds == label).sum().item()
        
                # Top-5 accuracy
                running_top5_correct += sum(label[i] in top5_preds[i] for i in range(len(label)))           
                
                running_loss += loss.item() * image.size(0)
                total_samples += image.size(0)
                
                # Running accuracies
                running_top1_acc = 100 * running_top1_correct / total_samples
                running_top5_acc = 100 * running_top5_correct / total_samples
                progress_bar.set_postfix({
                    'loss': f'{running_loss/total_samples:.4f}',
                    'running top1': f'{running_top1_acc:.2f}%',
                    'running top5': f'{running_top5_acc:.2f}%'
                })

        epoch_loss = running_loss / total_samples
        top1_acc = 100*running_top1_correct / total_samples
        top5_acc = 100*running_top5_correct / total_samples
        
        return epoch_loss, top1_acc, top5_acc

    def train(self):
        print(f"Training on {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Logging to: {self.log_file}")
            
        since = time.time()
        
        for epoch in range(self.start_epoch, self.config['epochs'] + 1):
            train_loss, train_acc, train_top5_acc = self.train_one_epoch(epoch)            
            # Validation phase
            val_loss, val_top1_acc, val_top5_acc = self.validate()        
            # Step the scheduler
            self.scheduler.step()

            # Print and log epoch summary
            summary = f"\nEpoch {epoch}/{self.config['epochs']}:\n"
            summary += f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%\n"
            summary += f"Val Loss: {val_loss:.4f} Top-1: {val_top1_acc:.2f}% Top-5: {val_top5_acc:.2f}%"
            summary += f"Train Top-1: {train_acc:.2f}%"
            summary += f"Val Top-1: {val_top1_acc:.2f}%"
            summary += f"Train Top-5: {train_top5_acc:.2f}%"
            summary += f"Val Top-5: {val_top5_acc:.2f}%"           

            print(summary)
            
            # Save best model based on top-1 accuracy
            is_best = val_top1_acc > self.best_acc
            if is_best:
                self.best_acc = val_top1_acc
                print(f'New best accuracy: Top-1 {self.best_acc:.2f}%')
            
            # Save checkpoint every epoch
            self.save_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_top1_acc=val_top1_acc,
                val_top5_acc=val_top5_acc,
                is_best=is_best
            )
            
        time_elapsed = time.time() - since
        final_summary = f'\nTraining completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s\n'
        final_summary += f'Best val accuracy: {self.best_acc:.2f}%'
        print(final_summary)
        
        # Load best model weights
        self.model.load_state_dict(torch.load(os.path.join(self.config['checkpoint_dir'], 'model_best.pth'))['model_state_dict'])
        return self.model

    def save_checkpoint(self, epoch, train_loss, train_acc, val_loss, val_top1_acc, val_top5_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_top1_acc': val_top1_acc,
            'val_top5_acc': val_top5_acc,
            'best_acc': self.best_acc,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch:03d}.pth'  # Zero-padded epoch number
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'model_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to: {best_path}")

if __name__ == "__main__":
    
    with open("/home/ubuntu/Assignment9/config.yml", "r") as file:
        config = yaml.safe_load(file)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Verify dataset before training
    # print("Verifying dataset structure...")
    # if not verify_dataset_structure(config["data_dir"]):
    #     print("Dataset verification failed. Please fix the issues before training.")
    #     exit(1)
    
    # Adjust batch size for multiple GPUs
    num_gpus = torch.cuda.device_count()
    effective_batch_size = config["batch_size"]
    
    if num_gpus > 1:
        # Scale batch size by number of GPUs
        config["batch_size"] = config["batch_size"] // num_gpus
        print(f"Scaling batch size to {config['batch_size']} per GPU "
              f"(effective batch size: {effective_batch_size})")

    trainer = Trainer(config)
    model = trainer.train()    