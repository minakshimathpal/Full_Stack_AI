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
    
    # Create trainer
    trainer = Trainer(model, dataset, device, learning_rate=Config.LEARNING_RATE)
    
    # Train model
    trainer.train(Config.EPOCHS, batch_size=Config.BATCH_SIZE)
    
    # Save model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    main() 