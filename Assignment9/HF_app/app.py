import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now import from model.py
from model.model import ResNet50

# Load the model
@st.cache_data
def load_class_names():
    try:
        with open("CLS-LOC/imagenet_classes.json", 'r', encoding='utf-8') as f:
            content = f.read()
            content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
            class_names = json.loads(content)
            return class_names
    except Exception as e:
        st.error(f"Error loading class names: {str(e)}")
        return {}

# Load model
@st.cache_resource
def load_model():
    try:
        model = ResNet50(num_classes=1000)
        checkpoint = torch.load("checkpoints/model_best.pth", map_location=torch.device("cpu"))
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

# Initialize session state
if 'show_upload' not in st.session_state:
    st.session_state.show_upload = True

# Main content container
main_container = st.empty()

with main_container.container():
    if st.session_state.show_upload:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Uploaded Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("### Predictions")
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
            
            # Add the New Image button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("↻ New Image"):
                    main_container.empty()
                    st.session_state.show_upload = True
                    st.rerun()