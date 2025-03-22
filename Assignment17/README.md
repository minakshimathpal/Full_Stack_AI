---
title: Chromatic Diffusion Studio
emoji: 👀
colorFrom: blue
colorTo: blue
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
short_description: Generate beautiful images with Stable Diffusion an
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🦋 Butterfly Color Diffusion

A Hugging Face Space that demonstrates color-guided image generation with Stable Diffusion. This application allows you to generate beautiful butterfly images and explore how targeted color loss functions can transform your results.

## Features

- **Standard Stable Diffusion**: Generate images using the standard Stable Diffusion model
- **Color-Guided Generation**: Apply yellow color guidance to enhance specific tones in your images
- **Concept Styles**: Choose from various artistic concepts to guide the style of your generated images
- **Customizable Parameters**: Adjust inference steps, guidance scale, and color strength

## How It Works

### Standard Stable Diffusion

The standard approach uses text-to-image generation with classifier-free guidance to create images based on your prompt.

### Color-Guided Stable Diffusion

The color-guided approach adds a custom loss function during the diffusion process that encourages:
- Higher values in the red and green channels
- Lower values in the blue channel

This combination creates a yellow tone in the final image. The strength parameter controls how strongly this color guidance affects the generation process.

### Concept Styles

The concept styles use textual inversion embeddings to guide the image generation toward a particular artistic style or subject matter. These concepts have been trained on specific images and can dramatically change the look of your generated images.

## Technical Details

During each step of the diffusion process, we:
1. Calculate the predicted image at that step
2. Measure how far it is from our desired color profile
3. Calculate the gradient of this loss with respect to the latents
4. Adjust the latents to reduce the loss
5. Continue with the standard diffusion process

This approach allows for targeted control of specific visual attributes while maintaining the overall quality and coherence of the generated image.

## Usage

1. Enter a prompt describing the butterfly image you want to generate
2. Select a concept style (optional)
3. Adjust the inference steps, guidance scale, and seed
4. Set the yellow strength for color guidance
5. Click "Generate Standard Image" or "Generate Color-Guided Image"
6. Compare the results side by side

## Available Concept Styles

- concept-art-2-1
- canna-lily-flowers102
- arcane-style-jv
- seismic-image
- azalea-flowers102
- photographic
- realistic
- detailed
- national-geographic
- macro-photography
- nature-photography

## Requirements

This application uses:
- Stable Diffusion v1.4
- Hugging Face Diffusers
- Streamlit
- PyTorch
- Transformers

## Hugging Face Spaces Results

You can try out the application and view generated images on Hugging Face Spaces:

🔗 Live Demo: [Chromatic Diffusion Studio on Hugging Face Spaces](https://huggingface.co/spaces/mathminakshi/chromatic-diffusion-studio)
## Credits

Created by Minakshi Mathur as part of the ERA V3 course.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
