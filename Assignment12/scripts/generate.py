import torch
from src.model import GPTModel
from src.dataset import TextDataset
from src.trainer import Trainer
from src.config import Config

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = TextDataset(Config.DATA_PATH, sequence_length=Config.SEQUENCE_LENGTH)
    Config.VOCAB_SIZE = dataset.vocab_size
    
    # Initialize model
    model = GPTModel(
        vocab_size=Config.VOCAB_SIZE,
        d_model=Config.D_MODEL,
        nhead=Config.N_HEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT
    ).to(device)
    
    # Load trained model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    
    # Create trainer
    trainer = Trainer(model, dataset, device)
    
    # Get input from user
    prompt = input("Enter your prompt: ")
    
    # Generate text
    generated_text = trainer.generate(
        prompt,
        max_length=Config.MAX_GENERATE_LENGTH,
        temperature=Config.TEMPERATURE
    )
    
    print(f"\nGenerated text:\n{prompt}{generated_text}")

if __name__ == "__main__":
    main() 