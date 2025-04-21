import streamlit as st
import torch
import random
import time
import os
from PIL import Image
from custom_stable_diffusion import StableDiffusionConfig, StableDiffusionModels,ImageProcessor, generate_with_multiple_concepts,generate_with_multiple_concepts_and_color
import sys
import transformers
import diffusers
st.set_page_config(
    page_title="Butterfly Color Diffusion",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for models
if 'models' not in st.session_state:
    st.session_state.models = None
    st.session_state.config = None

# Add this near the top of your app.py
debug_mode = st.sidebar.checkbox("Debug Mode", value=True)

# Function to load models
@st.cache_resource
def load_models():
    if debug_mode:
        st.write("Debug: Starting model loading")
        st.write(f"Debug: Python version: {sys.version}")
        st.write(f"Debug: Torch version: {torch.__version__}")
        st.write(f"Debug: Transformers version: {transformers.__version__}")
        st.write(f"Debug: Diffusers version: {diffusers.__version__}")
    
    config = StableDiffusionConfig(
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=random.randint(1, 1000000),
        batch_size=1,
        device=None,
        max_length=77
    )
    models = StableDiffusionModels(config)
    image_processor = ImageProcessor(models, config)
    with st.spinner("Loading Stable Diffusion models... This may take a minute."):
        models.load_models()
        models.set_timesteps()
    
    if debug_mode:
        st.write(f"Debug: Models loaded successfully. Device: {config.device}")
    
    return models, config, image_processor

# Title and description
st.title("🦋 Butterfly Color Diffusion")
st.markdown("""
Generate beautiful butterfly images with Stable Diffusion and explore color guidance technology 
that enhances yellow tones. Compare standard image generation with color-guided generation to see 
how targeted color loss functions can transform your results.
""")

# Sidebar with controls
st.sidebar.title("Generation Settings")

# Common settings
prompt = st.sidebar.text_area(
    "Prompt", 
    value="A detailed photograph of a colorful monarch butterfly with orange and black wings, resting on a purple flower in a lush garden with sunlight",
    height=100
)

# Add concept selection dropdown
available_concepts = [
    "None (No concept)",
    "concept-art-2-1",
    "canna-lily-flowers102",
    "arcane-style-jv",
    "seismic-image",
    "azalea-flowers102",
    "photographic",
    "realistic",
    "detailed",
    "national-geographic",
    "macro-photography",
    "nature-photography"
]

selected_concept = st.sidebar.selectbox(
    "Select Concept Style",
    available_concepts,
    index=0,
    help="Choose a concept style to apply to your image. Select 'None' to use standard generation."
)

# Convert "None" selection to actual None value
concept_name = None if selected_concept == "None (No concept)" else selected_concept

steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=100, value=30, step=1)
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.1)
seed = st.sidebar.number_input("Seed (0 for random)", min_value=0, max_value=1000000, value=0, step=1)

# Color guidance settings
st.sidebar.markdown("---")
st.sidebar.subheader("Color Guidance Settings")
yellow_strength = st.sidebar.slider("Yellow Strength", min_value=0, max_value=500, value=200, step=10)
guidance_interval = st.sidebar.slider("Guidance Interval", min_value=1, max_value=10, value=5, step=1)

# Create two columns for the images
col1, col2 = st.columns(2)

with col1:
    st.subheader("Standard Stable Diffusion")
    standard_button = st.button("Generate Standard Image", use_container_width=True)

with col2:
    st.subheader("Color-Guided Stable Diffusion")
    color_button = st.button("Generate Color-Guided Image", use_container_width=True)

# Load models when needed
if standard_button or color_button:
    if st.session_state.models is None:
        st.session_state.models, st.session_state.config ,st.session_state.image_processor= load_models()
    
    # Update config with current settings
    st.session_state.config.num_inference_steps = steps
    st.session_state.config.guidance_scale = guidance_scale
    
    # Set seed
    if seed == 0:
        seed = random.randint(1, 1000000)
    st.session_state.config.seed = seed
    st.sidebar.write(f"Using seed: {seed}")

# Generate standard image
if standard_button:
    with col1:
        with st.spinner("Generating standard image..."):
            progress_bar = st.progress(0)
            start_time = time.time()
            
            # Call the generation function
            result = generate_with_multiple_concepts(
                models=st.session_state.models,
                config=st.session_state.config,
                image_processor=st.session_state.image_processor,
                prompt=prompt,
                concepts=[concept_name] if concept_name else [],
                output_dir="concept_images"
            )
            
            end_time = time.time()
            
            # Check if we got a valid image back
            if result is not None and hasattr(result, 'format'):
                # It's a PIL Image object
                image = result
            else:
                # Try to load the image from the expected output path
                try:
                    if concept_name:
                        image_path = f"concept_images/{concept_name}/{concept_name}.png"
                    else:
                        image_path = "concept_images/standard_image.png"
                    
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                    else:
                        st.error(f"Could not find generated image at {image_path}")
                        st.stop()  # Stop execution here
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    st.stop()  # Stop execution here
            
            caption = f"Standard Stable Diffusion"
            if concept_name:
                caption += f" with {concept_name} concept"
            st.image(image, caption=caption, use_container_width=True)
            st.write(f"Generation time: {end_time - start_time:.2f} seconds")

# Generate color-guided image
if color_button:
    with col2:
        with st.spinner("Generating color-guided image..."):
            progress_bar = st.progress(0)
            start_time = time.time()
            
            # Call the generation function
            result = generate_with_multiple_concepts_and_color(
                models=st.session_state.models,
                config=st.session_state.config,
                image_processor=st.session_state.image_processor,
                prompt=prompt,
                concepts=[concept_name] if concept_name else [],
                output_dir="concept_images",
                blue_loss_scale=0,
                yellow_loss_scale=yellow_strength                
            )
            
            end_time = time.time()
            
            # Check if we got a valid image back
            if result is not None and hasattr(result, 'format'):
                # It's a PIL Image object
                image = result
            else:
                # Try to load the image from the expected output path
                try:
                    if concept_name:
                        # Determine the filename based on color guidance
                        color_info = f"_yellow{yellow_strength}" if yellow_strength > 0 else ""
                        image_path = f"concept_images/{concept_name}/{concept_name}{color_info}.png"
                    else:
                        image_path = "concept_images/color_guided_image.png"
                    
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                    else:
                        st.error(f"Could not find generated image at {image_path}")
                        st.stop()  # Stop execution here
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    st.stop()  # Stop execution here
            
            caption = f"Color-Guided Stable Diffusion"
            if concept_name:
                caption += f" with {concept_name} concept"
            st.image(image, caption=caption, use_container_width=True)
            st.write(f"Generation time: {end_time - start_time:.2f} seconds")

# Explanation section
st.markdown("---")
st.header("How It Works")
st.markdown("""
### Standard Stable Diffusion
The standard approach uses text-to-image generation with classifier-free guidance to create images based on your prompt.

### Color-Guided Stable Diffusion
The color-guided approach adds a custom loss function during the diffusion process that encourages:
- Higher values in the red and green channels
- Lower values in the blue channel

This combination creates a yellow tone in the final image. The strength parameter controls how strongly this color guidance affects the generation process.

### Concept Styles
The concept styles use textual inversion embeddings to guide the image generation toward a particular artistic style or subject matter. These concepts have been trained on specific images and can dramatically change the look of your generated images.

### Technical Details
During each step of the diffusion process, we:
1. Calculate the predicted image at that step
2. Measure how far it is from our desired color profile
3. Calculate the gradient of this loss with respect to the latents
4. Adjust the latents to reduce the loss
5. Continue with the standard diffusion process

This approach allows for targeted control of specific visual attributes while maintaining the overall quality and coherence of the generated image.
""")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Stable Diffusion and Streamlit")
