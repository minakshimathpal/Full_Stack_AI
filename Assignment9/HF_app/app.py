import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
import sys
import os

# Add the parent directory to the Python path to access 'model' from 'HF_app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from model.py
from model.model import ResNet50

# Load the model
@st.cache_data
def load_class_names():
    try:
        with open("D:\TSAI\Full_Stack_AI\Assignment9\CLS-LOC\imagenet_classes.json", 'r', encoding='utf-8') as f:
            # Read the file content first
            content = f.read()
            # Try to clean the content of any control characters
            content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
            # Parse the cleaned content
            class_names = json.loads(content)
            return class_names
    except json.JSONDecodeError as e:
        st.error(f"Error loading class names: Invalid JSON format. {str(e)}")
        return {}
    except FileNotFoundError:
        st.error("Error: Class names file not found. Please check the file path.")
        return {}
    except Exception as e:
        st.error(f"Unexpected error loading class names: {str(e)}")
        return {}

# Load model
@st.cache_resource
def load_model():
    try:
        model = ResNet50(num_classes=1000)
        checkpoint = torch.load("D:\TSAI\Full_Stack_AI\Assignment9\checkpoints\model_best.pth", map_location=torch.device("cpu"))
        # Extract just the model state dict from the checkpoint
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            st.error("Invalid model checkpoint format")
            return None
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Streamlit app
st.title("Image Classification with ResNet50")
class_names = load_class_names()
model = load_model()

# Update the main section to handle None model
if model is None:
    st.error("Failed to load the model. Please check the model file.")
    st.stop()

# Create a container for the file uploader
upload_container = st.empty()

# File uploader inside the container
uploaded_file = upload_container.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Clear the upload container
    upload_container.empty()
    
    # Load the image first
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create a container for the entire content
    with st.container():
        # First row - Headers with reduced width
        st.markdown("""
            <style>
                div[data-testid="column"] {
                    display: flex;
                    flex-direction: column;
                    height: 400px;
                    justify-content: center;
                }
                div[data-testid="stImage"] {
                    height: 400px;
                    display: flex;
                    align-items: center;
                }
                div[data-testid="stTable"] {
                    height: 400px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                .header-row {
                    max-width: 800px;
                    margin: auto;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Wrap headers in a div with the header-row class
        st.markdown('<div class="header-row">', unsafe_allow_html=True)
        header_col1, header_col2 = st.columns(2)
        with header_col1:
            st.markdown("### Uploaded Image")
        with header_col2:
            st.markdown("### Predictions")
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Add a small space between headers and content
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Second row - Content (full width)
        content_col1, content_col2 = st.columns(2)
        
        # Column 1 - Image
        with content_col1:
            st.image(image, use_container_width=True)
        
        # Column 2 - Predictions
        with content_col2:
            # Process image and get predictions
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_prob, top5_idx = torch.topk(probabilities, 5)
                
                results = []
                for i in range(5):
                    class_id = top5_idx[i].item()
                    prob = top5_prob[i].item() * 100
                    class_name = class_names[str(class_id)]
                    results.append({
                        "Rank": i + 1,
                        "Class": class_name,
                        "Confidence": f"{prob:.2f}%"
                    })
                st.table(results)