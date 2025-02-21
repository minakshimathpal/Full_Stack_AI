import torch
import tiktoken
from model import GPT
import argparse

def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=40):
    # Get device
    device = next(model.parameters()).device
    
    # Tokenize prompt with special token handling
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(enc.encode(prompt, allowed_special={'<|endoftext|>'})).unsqueeze(0).to(device)
    
    # Get end token id
    end_token = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
    
    # Generate tokens
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model's output
            logits, _ = model(input_ids)
            
            # Get next token probabilities
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we generate EOS token
            if next_token.item() == end_token:
                break
    
    # Decode and return the generated text
    generated_tokens = input_ids[0].tolist()
    return enc.decode(generated_tokens)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate text using trained GPT model')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                      help='Input prompt for text generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=40,
                      help='Top-k sampling parameter')
    parser.add_argument('--model_path', type=str, default='proper_model.pth',
                      help='Path to the trained model checkpoint')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print("Loading model...")
        model = GPT.from_pretrained('gpt2')
        
        print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint)
        
        # Generate text
        print("\nGenerating text...")
        generated = generate_text(
            model,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        # Print results
        print("\nGeneration Results:")
        print("-" * 50)
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
        print(f"Generated: {generated}")
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"Error: Could not find model checkpoint at {args.model_path}")
    except Exception as e:
        print(f"Error during text generation: {str(e)}") 