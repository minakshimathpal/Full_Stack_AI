import os
import time
from src.bpe import Tokenizer

def main(name="EnglishBPE", vocab_size=2000, verbose=False):  # Reduced vocab_size
    """
    Train a BPE tokenizer with smaller vocabulary to test compression ratio
    """
    os.makedirs("models", exist_ok=True)
    
    print("Loading training data...")
    with open("data/trainingdata.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text)} characters")

    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    t0 = time.time()
    
    tokenizer = Tokenizer()
    tokenizer.train(text, vocab_size=vocab_size, verbose=verbose)
    
    print(f"\nAchieved compression ratio: {tokenizer.compression_ratio}")
    
    model_path = os.path.join("models", f"{name}.model")
    tokenizer.save(model_path)
    
    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    # Try different vocab sizes
    for size in [5000]:
        print(f"\nTesting vocab_size = {size}")
        main(name=f"EnglishBPE_{size}", vocab_size=size)