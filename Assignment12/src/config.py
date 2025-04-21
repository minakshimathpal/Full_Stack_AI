# class Config:
#     # Model parameters
#     VOCAB_SIZE = None  # Will be set based on dataset
#     D_MODEL = 256
#     N_HEAD = 8
#     NUM_LAYERS = 6
#     DIM_FEEDFORWARD = 1024
#     DROPOUT = 0.1
    
#     # Training parameters
#     BATCH_SIZE = 32
#     LEARNING_RATE = 0.001
#     EPOCHS = 50
#     SEQUENCE_LENGTH = 128
    
#     # Generation parameters
#     MAX_GENERATE_LENGTH = 50
#     TEMPERATURE = 0.8
    
#     # Paths
#     DATA_PATH = "data/input.txt"
#     MODEL_SAVE_PATH = "models/gpt_model.pth" 
from dataclasses import dataclass
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layers: int = 12 # number of layers
    n_heads: int = 12 # number of heads
    n_embed: int = 768 # embedding dimension