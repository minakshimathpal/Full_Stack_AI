import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch_lr_finder import LRFinder
from model import GPT, GPTConfig
from trainer import DataLoaderLite, select_device, set_seed

import json
import os
import mlflow
from datetime import datetime
import matplotlib.pyplot as plt

class DataLoaderLiteDataset(Dataset):
    def __init__(self, B, T, tokens):
        self.B = B
        self.T = T
        self.tokens = tokens
        # Calculate length based on number of complete batches possible
        self.length = (len(tokens) - 1) // (B * T)  # -1 for target shift

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Calculate start index for this batch
        start_idx = idx * self.B * self.T
        # Calculate end index (+1 for target shift)
        end_idx = start_idx + self.B * self.T + 1
        
        if end_idx > len(self.tokens):
            start_idx = 0
            end_idx = self.B * self.T + 1
            
        buf = self.tokens[start_idx:end_idx]
        # Reshape into a single batch
        x = buf[:-1].reshape(1, -1)  # shape: (1, B*T)
        y = buf[1:].reshape(1, -1)   # shape: (1, B*T)
        
        return x.squeeze(0), y.squeeze(0)  # Return flattened tensors

def custom_loss(outputs, targets):
    # Unpack the model outputs (our model returns (logits, loss))
    logits, pre_computed_loss = outputs
    if pre_computed_loss is not None:
        return pre_computed_loss
    
    # If no pre-computed loss, calculate it here
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

def find_best_lr(lr_finder):
    """Find the learning rate with the steepest negative gradient (optimal learning rate)"""
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    
    # Calculate gradients between consecutive points
    gradients = []
    gradient_lrs = []  # Store the corresponding learning rates
    
    for i in range(1, len(lrs)):
        # Calculate gradient between points i-1 and i
        gradient = (losses[i] - losses[i-1]) / (lrs[i] - lrs[i-1])
        gradients.append(gradient)
        # Store the learning rate at point i
        gradient_lrs.append(lrs[i])
    
    # Find the point with steepest negative gradient
    min_gradient_idx = gradients.index(min(gradients))
    
    # Return the corresponding learning rate
    return gradient_lrs[min_gradient_idx]  # Now we can directly use the index

def run_lr_finder(start_lr=1e-7, end_lr=10):
    # Set device
    device = select_device()
    set_seed(1337)

    # Set batch size and sequence length
    B, T = 8,64  # Batch size and sequence length
    
    # Initialize model with pretrained weights
    print("Loading pretrained GPT-2 model...")
    model = GPT.from_pretrained('gpt2')
    model = model.to(device)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), 
                     lr=start_lr,
                     weight_decay=0.1)
    
    # Create DataLoader
    data_loader_lite = DataLoaderLite(B=B, T=T)
    dataset = DataLoaderLiteDataset(B=B, T=T, tokens=data_loader_lite.tokens)
    train_loader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    print(f"Running LR finder from {start_lr:.2e} to {end_lr:.2e}")
    
    # Initialize LR Finder
    lr_finder = LRFinder(model, optimizer, custom_loss, device=device)
    
    try:
        # Run range test
        lr_finder.range_test(
            train_loader,
            num_iter=100,
            start_lr=start_lr,
            end_lr=end_lr,
            step_mode="exp",
            diverge_th=5,
        )
        
        # Find best learning rate
        suggested_lr = find_best_lr(lr_finder)
        
        # Get minimum loss
        min_loss = min(lr_finder.history['loss']) if lr_finder.history['loss'] else float('inf')
        
        # Create plot and save it
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_finder.plot(skip_start=10, ax=ax)
        plt.title('Learning Rate Finder')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        
        # Save the plot
        plot_path = f'lr_finder_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error during LR finding: {e}")
        print(f"Full error: {str(e)}")
        import traceback
        traceback.print_exc()
        suggested_lr = None
        min_loss = float('inf')
    finally:
        # Reset the model and optimizer
        lr_finder.reset()
    
    return suggested_lr, min_loss

def iterative_lr_search(iterations=3, initial_range=(1e-7, 1.0)):
    """
    Perform multiple iterations of LR finding, logging results to MLflow
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    current_start, current_end = initial_range
    best_lr = None
    best_loss = float('inf')
    
    # Create directory for results if it doesn't exist
    os.makedirs('lr_finder_results', exist_ok=True)
    
    # Set up MLflow experiment
    mlflow.set_experiment("LR_Finder_Experiment")
    
    # Start a parent run for the entire search
    with mlflow.start_run(run_name="LR_Search") as parent_run:
        mlflow.log_params({
            'total_iterations': iterations,
            'initial_lr_range_start': initial_range[0],
            'initial_lr_range_end': initial_range[1],
            'device': device
        })
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            
            # Start a child run for each iteration
            with mlflow.start_run(run_name=f"Iteration_{i+1}", nested=True):
                # Run LR finder
                suggested_lr, min_loss = run_lr_finder(current_start, current_end)
                
                # Update best_lr if this iteration has lower loss
                is_best = False
                if suggested_lr is not None and min_loss < best_loss:
                    best_loss = min_loss
                    best_lr = suggested_lr
                    is_best = True
                    print(f"New best LR found: {best_lr:.2e} (loss: {best_loss:.4f})")
                
                # Store results
                result = {
                    'iteration': i+1,
                    'start_lr': float(current_start),
                    'end_lr': float(current_end),
                    'suggested_lr': float(suggested_lr) if suggested_lr is not None else None,
                    'min_loss': float(min_loss),
                    'is_best': is_best,
                    'current_best_lr': float(best_lr) if best_lr is not None else None,
                    'current_best_loss': float(best_loss)
                }
                results.append(result)
                
                # Save iteration results to JSON
                results_path = os.path.join('lr_finder_results', f'lr_finder_results_iteration_{i+1}.json')
                with open(results_path, 'w') as f:
                    json.dump(result, f, indent=4)
                mlflow.log_artifact(results_path)
                
                # Log metrics to MLflow
                if suggested_lr is not None:
                    mlflow.log_metrics({
                        'suggested_lr': suggested_lr,
                        'min_loss': min_loss,
                        'current_best_lr': best_lr,
                        'current_best_loss': best_loss
                    })
                    print(f"Iteration {i+1} suggested LR: {suggested_lr:.2e} (loss: {min_loss:.4f})")
                    
                    # Update search range for next iteration
                    current_start = suggested_lr / 10
                    current_end = suggested_lr * 10
                else:
                    print("No valid learning rate found in this range")
                    current_end = current_start
                    current_start = current_start / 10
                
                # Log the plot if available
                plot_path = 'lr_finder_plot.png'
                if os.path.exists(plot_path):
                    # Copy plot to results directory with iteration number
                    new_plot_path = os.path.join('lr_finder_results', f'lr_finder_plot_iteration_{i+1}.png')
                    import shutil
                    shutil.copy2(plot_path, new_plot_path)
                    mlflow.log_artifact(new_plot_path)
        
        # Save final summary at the end
        final_summary = {
            'best_learning_rate': float(best_lr) if best_lr is not None else None,
            'best_loss': float(best_loss),
            'conservative_lr': float(best_lr/10) if best_lr is not None else None,
            'moderate_lr': float(best_lr/3) if best_lr is not None else None,
            'aggressive_lr': float(best_lr) if best_lr is not None else None,
            'total_iterations': iterations,
            'final_results': results
        }
        
        summary_path = os.path.join('lr_finder_results', 'lr_finder_final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=4)
        mlflow.log_artifact(summary_path)
    
    # Print summary with more details
    print("\nLR Finder Summary:")
    print("-" * 50)
    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Range: {result['start_lr']:.2e} - {result['end_lr']:.2e}")
        if result['suggested_lr'] is not None:
            print(f"  Suggested LR: {result['suggested_lr']:.2e}")
        else:
            print("  Suggested LR: None")
        if result['min_loss'] is not None:
            print(f"  Minimum Loss: {result['min_loss']:.4f}")
    
    print("\nBest Learning Rate Found:")
    if best_lr is not None:
        print(f"LR: {best_lr:.2e}")
        print(f"Loss: {best_loss:.4f}")
    else:
        print("No valid learning rate found")
    
    return best_lr

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set up MLflow tracking URI if needed
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    
    best_lr = iterative_lr_search(iterations=3)
    if best_lr:
        print(f"\nRecommended learning rate: {best_lr:.2e}")
        print("For fine-tuning pretrained GPT-2, consider using:")
        print(f"- Conservative: {best_lr/10:.2e}")
        print(f"- Moderate: {best_lr/3:.2e}")
        print(f"- Aggressive: {best_lr:.2e}")
    else:
        print("\nCould not find a suitable learning rate")
