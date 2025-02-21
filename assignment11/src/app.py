import streamlit as st
from bpe import Tokenizer
import random
import colorsys

# Set page config
st.set_page_config(
    page_title="English BPE Tokenizer Visualizer",
    layout="wide"
)

# Load the trained tokenizer
@st.cache_resource
def load_tokenizer():
    tokenizer = Tokenizer()
    tokenizer.load("models/EnglishBPE_6999.model.model")
    return tokenizer

# Load example texts
@st.cache_data
def load_examples():
    try:
        with open("data/testdata1.txt", "r", encoding="utf-8") as f:
            example1 = f.read().strip()
        with open("data/testdata2.txt", "r", encoding="utf-8") as f:
            example2 = f.read().strip()
    except Exception as e:
        st.error(f"Error loading example texts: {str(e)}")
        # Fallback examples in case files can't be loaded
        
    return example1, example2

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + random.random() * 0.3
        value = 0.8 + random.random() * 0.2
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def process_text(text, tokenizer):
    try:
        # Get tokens
        tokens = tokenizer.encode(text)
        
        # Generate colors for visualization
        unique_tokens = list(set(tokens))
        colors = generate_distinct_colors(len(unique_tokens))
        token_colors = dict(zip(unique_tokens, colors))
        
        # Create HTML visualization
        html_parts = []
        decoded_tokens = [tokenizer.decode([token]) for token in tokens]
        
        for token, token_text in zip(tokens, decoded_tokens):
            color = token_colors[token]
            html_parts.append(f'<span style="background-color: {color}; padding: 0 2px; border-radius: 3px;" title="Token ID: {token}">{token_text}</span>')
        
        return ''.join(html_parts), tokens
    except Exception as e:
        return f"<span style='color: red'>Error processing text: {str(e)}</span>", None

def main():
    # Load tokenizer and examples
    tokenizer = load_tokenizer()
    example1, example2 = load_examples()

    # Title and description
    st.title("English BPE Tokenizer Visualizer")
    st.markdown("Enter text to see how it gets tokenized, with color-coded visualization")

    # Example selector
    example_option = st.selectbox(
        "Choose an example or enter your own text below:",
        ["Custom Input", "Example 1", "Example 2"]
    )

    # Text input
    if example_option == "Example 1":
        text = st.text_area("Enter Text", value=example1, height=100)
    elif example_option == "Example 2":
        text = st.text_area("Enter Text", value=example2, height=100)
    else:
        text = st.text_area("Enter Text", height=100)

    # Process button
    if st.button("Process Text") or text:
        if text.strip():
            # Create two columns for output
            col1, col2 = st.columns([2, 1])
            
            # Process the text
            visualization, tokens = process_text(text, tokenizer)
            
            with col1:
                st.subheader("Visualization")
                st.markdown(visualization, unsafe_allow_html=True)
            
            with col2:
                if tokens is not None:
                    st.subheader("Token Information")
                    st.write(f"Token count: {len(tokens)}")
                    st.write("Tokens:", tokens)
        else:
            st.warning("Please enter some text to process.")

if __name__ == "__main__":
    main()