import torch
from src.model import GPT
from huggingface_hub import HfApi, create_repo
from getpass import getpass
import os

def convert_checkpoint_to_model(checkpoint_path, save_path):
    try:
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("Initializing model...")
        model = GPT.from_pretrained('gpt2')
        
        # Get state dict from checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Load state dict into model
        model.load_state_dict(state_dict)
        
        # Save model properly
        print("Saving model in proper format...")
        torch.save(
            model.state_dict(),  # Save only state dict
            save_path,
            _use_new_zipfile_serialization=True
        )
        print(f"Model saved successfully to {save_path}")
        return True
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False

def upload_to_huggingface(model_path, repo_name="mathminakshi/custom_gpt2"):
    try:
        # Get Hugging Face token
        token = getpass("Enter your Hugging Face token: ")
        
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create or get repository
        repo_url = create_repo(
            repo_name,
            token=token,
            private=False,
            exist_ok=True
        )
        print(f"Repository ready at {repo_url}")
        
        # Upload file
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.pth",
            repo_id=repo_name,
            token=token
        )
        print(f"Model uploaded successfully to {repo_name}")
        return True
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        return False

if __name__ == "__main__":
    # Paths
    checkpoint_path = "/content/drive/MyDrive/Assignment12/best_model.pth"  # Your checkpoint path
    proper_model_path = "proper_model.pth"
    
    # Convert checkpoint to proper format
    print("Step 1: Converting checkpoint to proper format...")
    if convert_checkpoint_to_model(checkpoint_path, proper_model_path):
        print("\nStep 2: Uploading to Hugging Face...")
        if upload_to_huggingface(proper_model_path):
            print("\nProcess completed successfully!")
            print(f"Model is now available at: https://huggingface.co/mathminakshi/custom_gpt2")
        
        # Cleanup
        if os.path.exists(proper_model_path):
            os.remove(proper_model_path)
            print("\nTemporary files cleaned up")