import streamlit as st
import torch
import tiktoken
from src.model import GPT, GPTConfig

def load_model():
    """Load the trained GPT model."""
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load('final_best_model.pth', map_location='cpu')['model_state_dict'])
    model.eval()
    return model

@st.cache_resource  # Cache the model loading
def get_model():
    return load_model()

def generate_text(prompt, max_tokens=500, temperature=0.8, top_k=40):
    """Generate text based on the prompt."""
    # Encode the prompt
    enc = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    
    # Get cached model
    model = get_model()
    
    with torch.no_grad():
        output_sequence = []
        # Add a progress bar
        progress_bar = st.progress(0)
        
        for i in range(max_tokens):
            # Update progress
            progress_bar.progress(i / max_tokens)
            
            # Get predictions
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature
            l            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to output
            output_sequence.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we generate an EOS token
            if next_token.item() == 50256:
                break
    
    # Complete progress
    progress_bar.progress(1.0)
    
    # Decode and return the generated text
    generated_text = enc.decode(output_sequence)
    return prompt + generated_text

def main():
    st.title("GPT Text Generator")
    st.write("Enter a prompt to generate text. Generation will stop at EOS token or when max tokens is reached.")
    
    # Sidebar for parameters
    st.sidebar.header("Generation Parameters")
    max_tokens = st.sidebar.slider(
        "Max Tokens", 
        min_value=1, 
        max_value=1000, 
        value=100,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.8,
        help="Higher values make the output more random"
    )
    
    top_k = st.sidebar.slider(
        "Top-K", 
        min_value=1, 
        max_value=100, 
        value=40,
        help="Limits the number of tokens to choose from"
    )
    
    # Main area for input and output
    prompt = st.text_area(
        "Enter your prompt:",
        height=100,
        placeholder="Once upon a time..."
    )
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating text..."):
                generated_text = generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                st.write("### Generated Text:")
                st.write(generated_text)
        else:
            st.warning("Please enter a prompt first!")

if __name__ == "__main__":
    main() 