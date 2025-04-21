import math
import os
from base64 import b64encode
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login, hf_hub_download
from matplotlib import pyplot as plt
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging

class StableDiffusionConfig:
    """
    Configuration class for stable Diffusion parameters

    """
    def __init__(self, height: int=512,
                 width:int= 512,
                 num_inference_steps:int= 50,
                 guidance_scale:int=7.5,
                 seed:int=32,
                 batch_size:int=1,
                 device:str=None,
                 max_length:int=77):
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.batch_size = batch_size        
        self.max_length=max_length

        # set device
        if device is None:
            self.device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            if  "mps" ==self.device:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "TRUE"

            else:
                self.device=device

        self.generator= torch.manual_seed(self.seed)            

class StableDiffusionModels:
    """
    class to manage Stable Diffusion model components.  
    """    
    def __init__(self, config:StableDiffusionConfig):
        self.config=config
        self.vae= None
        self.tokenizer= None
        self.text_encoder= None
        self.unet= None
        self.scheduler= None

    def load_models(self, model_version:str="CompVis/stable-diffusion-v1-4"):
        """
        Load all the required models for stable diffusion.
        """
        try:
            # Add cache directory to ensure files are saved in a writable location
            cache_dir = "./model_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                model_version, 
                subfolder="vae",
                cache_dir=cache_dir,
                local_files_only=False
            )
            
            # Load tokenizer and text encoder with explicit cache directory
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=cache_dir,
                local_files_only=False
            )
            
            self.text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=cache_dir,
                local_files_only=False
            )

            # Load UNet
            self.unet = UNet2DConditionModel.from_pretrained(
                model_version, 
                subfolder="unet",
                cache_dir=cache_dir,
                local_files_only=False
            )
            
            # Load scheduler
            self.scheduler = LMSDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear", 
                num_train_timesteps=1000
            )
            
            # Move models to device
            self.vae = self.vae.to(self.config.device)
            self.text_encoder = self.text_encoder.to(self.config.device)
            self.unet = self.unet.to(self.config.device)
            
            print(f"Using device: {self.config.device}")
            return self
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            # Add more detailed error information
            import traceback
            traceback.print_exc()
            raise
    
    def set_timesteps(self, num_inference_steps:int=None):
        """
        Set the number of inference steps for the scheduler.
        """
        if num_inference_steps is None:
            num_inference_steps= self.config.num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps)

        # fix to ensure MPS compatibility
        self.scheduler.timesteps= self.scheduler.timesteps.to(torch.float32)
        return self

class ImageProcessor:
    """Class to handle image processing operations."""
    
    def __init__(self, models: StableDiffusionModels, config: StableDiffusionConfig):
        self.models = models
        self.config = config
    
    def pil_to_latent(self, input_im: Image.Image) -> torch.Tensor:
        """Convert a PIL image to latent space."""
        with torch.no_grad():
            # Scale to [-1, 1] and convert to tensor
            image_tensor = tfms.ToTensor()(input_im).unsqueeze(0).to(self.config.device) * 2 - 1
            # Encode to latent
            latent = self.models.vae.encode(image_tensor)
        return 0.18215 * latent.latent_dist.sample()
    
    def latents_to_pil(self, latents: torch.Tensor) -> List[Image.Image]:
        """Convert latents to PIL images."""
        # Scale latents
        latents = (1 / 0.18215) * latents
        
        with torch.no_grad():
            # Decode latents
            image = self.models.vae.decode(latents).sample
        
        # Process to PIL images
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        return pil_images

class TextEmbeddingProcessor:
    """Class to process and modify text embeddings."""   
    def __init__(self, models:StableDiffusionModels, config:StableDiffusionConfig,imageprocessor:ImageProcessor,prompt:str):
        self.models=models
        self.config=config
        self.token_emb_layer= models.text_encoder.text_model.embeddings.token_embedding
        self.pos_emb_layer= models.text_encoder.text_model.embeddings.position_embedding
        self.position_ids= models.text_encoder.text_model.embeddings.position_ids[:,:77]
        self.position_embeddings= self.pos_emb_layer(self.position_ids)
        self.imageprocessor = imageprocessor
        self.prompt=prompt

    def load_embedding(self, concept_name:str) -> Tuple[str, torch.Tensor]:
        """ Downlaod a textual inversion concept from hugging face"""
        try:
            # Download the file      
            file_path= hf_hub_download(
                repo_id=f"sd-concepts-library/{concept_name}",
                filename="learned_embeds.bin",
                repo_type="model"
            )
            # load the embedding
            embedding= torch.load(file_path)
            return embedding
        except Exception as e:
            print(f"Error downloading concept {concept_name}: {e}")
            return None, None
        
    def tokenize_text(self, prompt=None) -> Tuple[torch.Tensor, int]:
      """Tokenize text input."""
      if prompt is None:
          prompt = self.prompt
          
      if isinstance(prompt, str):            
          text_input = self.models.tokenizer(
              prompt, 
              padding="max_length", 
              truncation=True, 
              max_length=self.models.tokenizer.model_max_length, 
              return_tensors="pt"
          )
          position = text_input["input_ids"][0][4].item()  # Get the position of the concept token
      
      input_ids = text_input.input_ids.to(self.config.device)
      return input_ids, position
    
    def get_output_embeds(self,input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
        bsz, seq_len = input_embeddings.shape[:2]
        causal_attention_mask = self.models.text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

        # Getting the output embeddings involves calling the model with passing output_hidden_states=True
        # so that it doesn't just return the pooled final predictions:
        encoder_outputs = self.models.text_encoder.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=None, # We aren't using an attention mask so that can be None
            causal_attention_mask=causal_attention_mask.to(self.config.device),
            output_attentions=None,
            output_hidden_states=True, # We want the output embs not the final output
            return_dict=None,
        )

        # We're interested in the output hidden state only
        output = encoder_outputs[0]

        # There is a final layer norm we need to pass these through
        output = self.models.text_encoder.text_model.final_layer_norm(output)

        # And now they're ready!
        return output

    def generate_with_embs(self,text_embeddings,output_path=None, return_image=False):
        height = self.config.height                        # default height of Stable Diffusion
        width = self.config.width                         # default width of Stable Diffusion
        num_inference_steps = self.config.num_inference_steps            # Number of denoising steps
        guidance_scale = self.config.guidance_scale                # Scale for classifier-free guidance
        generator = torch.manual_seed(self.config.seed)   # Seed generator to create the inital latent noise
        batch_size = 1
         
        text_input= self.models.tokenizer(self.prompt, padding="max_length", truncation=True, max_length=self.models.tokenizer.model_max_length, return_tensors="pt") 
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.models.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.models.text_encoder(uncond_input.input_ids.to(self.config.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler
        self.models.set_timesteps(num_inference_steps)

        # Prep latents
        latents = torch.randn(
        (batch_size, self.models.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        )
        latents = latents.to(self.config.device)
        latents = latents * self.models.scheduler.init_noise_sigma

        # Loop
        for i, t in tqdm(enumerate(self.models.scheduler.timesteps), total=len(self.models.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.models.scheduler.sigmas[i]
            latent_model_input = self.models.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.models.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.models.scheduler.step(noise_pred, t, latents).prev_sample

        if output_path is not None:
          # Ensure the output directory exists
          os.makedirs(os.path.dirname(output_path), exist_ok=True)
          
          # Make sure the output path has a file extension
          if not os.path.splitext(output_path)[1]:
              output_path = output_path + ".png"
              
          self.imageprocessor.latents_to_pil(latents)[0].save(output_path)

        if return_image:    
            return self.imageprocessor.latents_to_pil(latents)[0] 
       
    def prepare_embeddings_with_concepts(self, prompt, concept_name:str=None, output_path:str=None) -> None:
      """Encode text input into embeddings and generate image with concept."""
      input_ids, position = self.tokenize_text(self.prompt)
      token_embeddings = self.token_emb_layer(input_ids)
      embeddings = self.load_embedding(concept_name)
      print(embeddings)
      if embeddings is not None:
          # embeddings = embeddings.to(self.config.device)
          replacement_token_embedding = embeddings[next(iter(embeddings.keys()))].to(self.config.device)
          
          # Get the position indices where the token appears
          position_indices = torch.where(input_ids[0] == position)[0]
          
          if len(position_indices) > 0:
              # Get the shape of a single token embedding
              single_token_shape = token_embeddings[0, position_indices[0]].shape
              
              # Replace the token embedding at the specified position
              if replacement_token_embedding.shape != single_token_shape:
                  print("Warning: Embedding dimensions don't match. This might not be the right embedding.")

              # Reshape if needed
              if replacement_token_embedding.shape[0] != single_token_shape[0]:
                  print(f"Reshaping embedding from {replacement_token_embedding.shape} to {single_token_shape}")
                  replacement_token_embedding = replacement_token_embedding[:single_token_shape[0]]
              
              # Correctly index and replace the token embedding
              for idx in position_indices:
                  token_embeddings[0, idx] = replacement_token_embedding.to(self.config.device)

              # Combine with pos embs
              input_embeddings = token_embeddings + self.position_embeddings
              modified_output_embeddings = self.get_output_embeds(input_embeddings)
              self.generate_with_embs(modified_output_embeddings, output_path=output_path)
          else:
              print(f"Token position {position} not found in input_ids")
      else:
          print(f"Failed to load concept: {concept_name}")

def generate_with_multiple_concepts(models, config, image_processor, prompt, concepts, output_dir="concept_images"):
    """
    Generate images using multiple concepts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If no concepts provided, generate a standard image
    if not concepts or len(concepts) == 0:
        print("No concepts provided, generating standard image")
        # Create a standard image without concepts
        # You'll need to implement this part based on your existing code
        # For now, return None
        return None
    
    # Process each concept
    for concept in concepts:
        if concept is None:
            continue
            
        print(f"Generating image for concept: {concept}")
        concepts_dir = os.path.join(output_dir, concept)
        os.makedirs(concepts_dir, exist_ok=True)
        
        output_path = os.path.join(concepts_dir, f"{concept}.png")
        
        text_processor = TextEmbeddingProcessor(models, config, image_processor, prompt)
        
        # Generate the image with the concept
        pil_image = text_processor.prepare_embeddings_with_concepts(prompt, concept_name=concept, output_path=output_path)
        print(f"Saved image to {output_path}")
        
        # Return the generated image
        return pil_image
    
    # If we get here (no valid concepts processed), return None
    return None

def channel_loss(images, channel_idx=2, target_value=0.9):
    """
    Calculate the mean absolute error between a specific color channel and a target value.
    
    Args:
        images (torch.Tensor): Batch of images with shape [batch_size, channels, height, width]
        channel_idx (int): Index of the color channel to target (0=R, 1=G, 2=B)
        target_value (float): Target value for the channel (0-1)
    
    Returns:
        torch.Tensor: Loss value
    """
    return torch.abs(images[:, channel_idx] - target_value).mean()

def blue_loss(images, target=0.9):
    """Make images more blue by increasing the blue channel"""
    return channel_loss(images, channel_idx=2, target_value=target)

def yellow_loss(images):
    """
    Make images more yellow by increasing red and green channels and decreasing blue
    Yellow = high R + high G + low B
    """
    red_high = channel_loss(images, channel_idx=0, target_value=0.9)
    green_high = channel_loss(images, channel_idx=1, target_value=0.9)
    blue_low = channel_loss(images, channel_idx=2, target_value=0.1)
    return (red_high + green_high + blue_low) / 3

def generate_with_concept_and_color(
    models,
    config,
    image_processor,
    prompt,
    concept_name,
    output_dir="concept_images",
    blue_loss_scale=0,
    yellow_loss_scale=400,
    guidance_interval=3  # Changed from 5 to 3 to apply more frequently
):
    """
    Generate images using a concept and color guidance, then save to specified directory
    """
    # Create output directory
    concept_dir = os.path.join(output_dir, f"{concept_name}")
    os.makedirs(concept_dir, exist_ok=True)
    
    # Define output path with color info in filename
    color_info = ""
    if blue_loss_scale > 0:
        color_info += f"_blue{blue_loss_scale}"
    if yellow_loss_scale > 0:
        color_info += f"_yellow{yellow_loss_scale}"
    
    output_path = os.path.join(concept_dir, f"{concept_name}{color_info}.png")
    
    # Create text processor
    text_processor = TextEmbeddingProcessor(models, config, image_processor, prompt)
    
    # Load concept embedding
    embeddings = text_processor.load_embedding(concept_name)
    
    if embeddings is None:
        print(f"Failed to load concept: {concept_name}")
        return
    
    # Process text with concept
    input_ids, position = text_processor.tokenize_text(prompt)
    token_embeddings = text_processor.token_emb_layer(input_ids)
    
    # Handle different embedding formats
    if isinstance(embeddings, dict):
        replacement_token_embedding = embeddings[next(iter(embeddings.keys()))].to(config.device)
    elif isinstance(embeddings, tuple) and len(embeddings) >= 2:
        replacement_token_embedding = embeddings[1].to(config.device)
    elif isinstance(embeddings, torch.Tensor):
        replacement_token_embedding = embeddings.to(config.device)
    else:
        print(f"Unsupported embedding format for concept: {concept_name}")
        return
    
    # Get the position indices where the token appears
    position_indices = torch.where(input_ids[0] == position)[0]
    
    if len(position_indices) == 0:
        print(f"Token position {position} not found in input_ids")
        return
    
    # Get the shape of a single token embedding
    single_token_shape = token_embeddings[0, position_indices[0]].shape
    
    # Reshape if needed
    if replacement_token_embedding.shape != single_token_shape:
        print("Warning: Embedding dimensions don't match. This might not be the right embedding.")
        if replacement_token_embedding.shape[0] != single_token_shape[0]:
            print(f"Reshaping embedding from {replacement_token_embedding.shape} to {single_token_shape}")
            replacement_token_embedding = replacement_token_embedding[:single_token_shape[0]]
    
    # Replace the token embedding at the specified position
    for idx in position_indices:
        token_embeddings[0, idx] = replacement_token_embedding.to(config.device)
    
    # Combine with position embeddings
    input_embeddings = token_embeddings + text_processor.position_embeddings
    text_embeddings = text_processor.get_output_embeds(input_embeddings)
    
    # Get uncond embeddings
    uncond_input = models.tokenizer(
        [""], padding="max_length", max_length=77, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = models.text_encoder(uncond_input.input_ids.to(config.device))[0]
    
    # Concatenate for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Set timesteps
    models.set_timesteps(config.num_inference_steps)
    
    # Prepare latents
    height = config.height
    width = config.width
    batch_size = config.batch_size
    
    # Create a generator on the same device as where the tensor will be created
    if "cuda" in str(config.device):
        generator = torch.Generator(device="cuda").manual_seed(config.seed)
    else:
        generator = torch.manual_seed(config.seed)
    
    latents = torch.randn(
        (batch_size, models.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=config.device
    )
    latents = latents * models.scheduler.init_noise_sigma
    
    # Define color loss functions
    def channel_loss(images, channel_idx=2, target_value=0.9):
        return torch.abs(images[:, channel_idx] - target_value).mean()
    
    def blue_loss(images, target=0.9):
        return channel_loss(images, channel_idx=2, target_value=target)
    
    def yellow_loss(images, red_target=0.95, green_target=0.95, blue_target=0.05):
      """
      Make images more yellow by increasing red and green channels and decreasing blue
      Yellow = high R + high G + low B
      
      Args:
          images: The image tensor
          red_target: Target value for red channel (higher = more red)
          green_target: Target value for green channel (higher = more green)
          blue_target: Target value for blue channel (lower = less blue)
      """
      red_high = torch.abs(images[:, 0] - red_target).mean()
      green_high = torch.abs(images[:, 1] - green_target).mean()
      blue_low = torch.abs(images[:, 2] - blue_target).mean()
      
      # Weight the blue channel more heavily to really reduce blue
      return (red_high + green_high + blue_low * 2) / 4
    
    # Denoising loop
    for i, t in tqdm(enumerate(models.scheduler.timesteps), total=len(models.scheduler.timesteps)):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = models.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Apply color guidance
        if (blue_loss_scale > 0 or yellow_loss_scale > 0) and i % guidance_interval == 0:
            # Get the current sigma value
            sigma = models.scheduler.sigmas[i]
            
            # Requires grad on the latents
            latents = latents.detach().requires_grad_()
            
            # Get the predicted x0 directly (like in the example code)
            latents_x0 = latents - sigma * noise_pred
            
            # Decode to image space
            denoised_images = models.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5
            
            # Calculate combined loss
            loss = 0
            if blue_loss_scale > 0:
                blue_loss_value = blue_loss(denoised_images) * blue_loss_scale
                loss += blue_loss_value
            
            if yellow_loss_scale > 0:
                yellow_loss_value = yellow_loss(denoised_images) * yellow_loss_scale
                loss += yellow_loss_value
            
            # Print loss occasionally
            if i % 10 == 0:
                print(f"Step {i}, Loss: {loss.item()}")
                if blue_loss_scale > 0 and yellow_loss_scale > 0:
                    print(f"  Blue loss: {blue_loss_value.item()}, Yellow loss: {yellow_loss_value.item()}")
            
            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents)[0]
            
            # Modify the latents based on this gradient (using sigma squared like in the example)
            latents = latents.detach() - cond_grad * sigma**2
        
        # Step with scheduler
        latents = models.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode the final image
    with torch.no_grad():
        decoded = models.vae.decode((1 / 0.18215) * latents).sample
    
    # Convert to PIL image
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")[0]
    pil_image = Image.fromarray(image)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image.save(output_path)
    print(f"Saved image to {output_path}")
    
    return pil_image

def generate_with_multiple_concepts_and_color(models, config, image_processor, prompt, concepts, output_dir="concept_images", blue_loss_scale=0, yellow_loss_scale=0):
    """
    Generate images using multiple concepts and color guidance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If no concepts provided, generate a standard image with color guidance
    if not concepts or len(concepts) == 0:
        print("No concepts provided, generating standard image with color guidance")
        # Create a standard image with color guidance but without concepts
        # You'll need to implement this part based on your existing code
        # For now, return None
        return None
    
    # Process each concept
    for concept in concepts:
        if concept is None:
            continue
            
        print(f"Generating image for concept: {concept} with color guidance")
        # Generate the image with the concept and color guidance
        pil_image = generate_with_concept_and_color(
            models=models,
            config=config,
            image_processor=image_processor,
            prompt=prompt,
            concept_name=concept,
            output_dir=output_dir,
            blue_loss_scale=blue_loss_scale,
            yellow_loss_scale=yellow_loss_scale
        )
        
        # Return the generated image
        return pil_image
    
    # If we get here (no valid concepts processed), return None
    return None

# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = StableDiffusionConfig(
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42,
        batch_size=1,
        device=None,
        max_length=77
    )
    if config.device is None:
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            if  "mps" ==config.device:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "TRUE"

            else:
                config.device=device
    # Load models
    models = StableDiffusionModels(config)
    models.load_models()
    models.set_timesteps()
    
    # Create image processor
    image_processor = ImageProcessor(models, config)
    
    # Define base prompt and concepts
    base_prompt = "A detailed photograph of a colorful monarch butterfly with orange and black wings, resting on a purple flower in a lush garden with sunlight"
    
    # List of concepts to use (these should be available in the Hugging Face sd-concepts-library)
    concepts = [
        "concept-art-2-1",
        "canna-lily-flowers102",
        "arcane-style-jv",
        "seismic-image",
        "azalea-flowers102"
    ]
    
    # Generate images for all concepts
    generate_with_multiple_concepts(
        models=models,
        config=config,
        image_processor=image_processor,
        prompt=base_prompt,
        concepts=concepts,
        output_dir="concept_images"
    )

    generate_with_multiple_concepts_and_color(
        models=models,
        config=config,
        image_processor=image_processor,
        prompt=base_prompt,
        concepts=concepts,
        output_dir="concept_images",
        blue_loss_scale=0,     # Set to 0 to disable blue guidance
        yellow_loss_scale=200  # Set to 0 to disable yellow guidance
    )
            
