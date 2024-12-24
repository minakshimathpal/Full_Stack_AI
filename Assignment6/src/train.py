import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torchinfo import summary
import mlflow.pytorch
from model import Net
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import yaml
from tqdm import tqdm
from data_loader import load_data
from datetime import datetime
SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")   

def create_training_plots(train_losses, val_losses, train_accs, val_accs):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracies')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def train_epoch(model, device, train_loader, optimizer,scheduler):
    train_losses = []    
    train_acc = []
    
    model.train()
    pbar = tqdm(train_loader)
    running_loss=0.0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
                # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
   # Predict
        y_pred = model(data)
        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        l2_lambda = 0.000025
        l1_parameters = []
        for parameter in model.parameters():
            l1_parameters.append(parameter.view(-1))
        l1 = l2_lambda * model.compute_l1_loss(torch.cat(l1_parameters))
        loss += l1
        train_losses.append(loss)
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update pbar-tqdm
        
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc= f'LR={optimizer.param_groups[0]["lr"]} Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
    running_loss+=loss.item()    
    return running_loss/len(train_loader.dataset), 100. * correct/processed

def test(model, device, test_loader):  
    test_losses = []  
    test_acc = []

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_losses,accuracy

if __name__ == "__main__":        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
# Create model by passing entire config
model = Net().to(device)
# criterion = get_loss_function(config)

train_loader, test_loader = load_data()
EPOCHS=20
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
sch_dict = scheduler.state_dict()
sch_dict['total_steps'] = sch_dict['total_steps'] + EPOCHS * int(len(train_loader))
scheduler.load_state_dict(sch_dict)
losses = {"train": [], "val": []}
accuracy = {"train": [], "val": []}
for epoch in range(20):
    print(f"Epoch {epoch+1}")
    train_epoch_loss,train_epoch_accuracy=train_epoch(model,device,train_loader,optimizer,scheduler)
    val_epoch_loss,val_epoch_accuracy = test(model=model,test_loader=test_loader,device=device)
    
    losses["train"].append(train_epoch_loss)
    accuracy["train"].append(train_epoch_accuracy)

    losses["val"].append(val_epoch_loss)
    accuracy["val"].append(val_epoch_accuracy)
fig = create_training_plots(losses["train"], losses["val"], accuracy["train"], accuracy["val"])

if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)
plt.savefig(os.path.join(ARTIFACTS_DIR, "training_plots.png"))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{timestamp}.pth"))


