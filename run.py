import torch
from diffusers import FluxPipeline
from pathlib import Path
import time
import random
import os

# Default configuration
DEFAULT_CONFIG = {
    # Model parameters
    "model_path": None,  # Required
    # Device and precision parameters
    "device": "cuda",  # "cuda", "cpu", or "mps"
    "dtype": "bfloat16",  # "float32", "float16", or "bfloat16"
    "enable_cpu_offload": False,
    "enable_vae_slicing": False,
    "enable_vae_tiling": False,
    # Generation parameters
    "prompt": None,  # Required
    "negative_prompt": None,
    "height": 768,
    "width": 1360,
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
    "seed": None,
    "max_sequence_length": 512,
    "num_images": 1,  # How many images to generate (one by one)
    # Output parameters
    "output_dir": "outputs",
    "filename": None,  # Optional custom filename
}


def generate_flux_images(
    model_path=None,
    device="cuda",
    dtype="bfloat16",
    enable_cpu_offload=False,
    enable_vae_slicing=False,
    enable_vae_tiling=False,
    prompt=None,
    negative_prompt=None,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.5,
    seed=None,
    max_sequence_length=512,
    num_images=1,
    output_dir="outputs",
    filename=None,
):
    """
    Generate images using Flux models

    Parameters
    ----------
    model_path : str
        Path to the local Flux model or HF model ID
    device : str, optional
        Device to run inference on: 'cuda', 'cpu', or 'mps'
    dtype : str, optional
        Model precision: 'float32', 'float16', or 'bfloat16'
    enable_cpu_offload : bool, optional
        Enable sequential CPU offloading to save VRAM
    enable_vae_slicing : bool, optional
        Enable VAE slicing to save memory
    enable_vae_tiling : bool, optional
        Enable VAE tiling to save memory and process larger images
    prompt : str
        Text prompt for image generation
    negative_prompt : str, optional
        Negative prompt for generation
    height : int, optional
        Height of generated image
    width : int, optional
        Width of generated image
    num_inference_steps : int, optional
        Number of denoising steps
    guidance_scale : float, optional
        Guidance scale
    seed : int, optional
        Random seed for generation
    max_sequence_length : int, optional
        Maximum sequence length for text processing
    num_images : int, optional
        Number of images to generate
    output_dir : str, optional
        Directory to save generated images
    filename : str, optional
        Custom filename for the generated image (without extension)

    Returns
    -------
    list
        Paths to generated images
    """
    # Validate required parameters
    if not model_path:
        raise ValueError("model_path is required")
    if not prompt:
        raise ValueError("prompt is required")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up dtype
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    # Load model - ensure model_path is a string
    print(f"Loading Flux model from {model_path}...")
    try:
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,  # Add this to ensure safetensors are used if available
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Model path: {model_path}, type: {type(model_path)}")
        raise

    # Configure pipeline
    if enable_cpu_offload:
        print("Enabling sequential CPU offload...")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    if enable_vae_slicing:
        print("Enabling VAE slicing...")
        pipe.enable_vae_slicing()

    if enable_vae_tiling:
        print("Enabling VAE tiling...")
        pipe.enable_vae_tiling()

    # Set up generator for reproducibility
    generator = None
    current_seed = seed
    if current_seed is not None:
        generator = torch.Generator(device if device != "mps" else "cpu").manual_seed(
            current_seed
        )

    # Generate images
    generated_image_paths = []
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...")
        start_time = time.time()

        try:
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=max_sequence_length,
            )

            # Save image with custom filename if provided
            if filename and num_images == 1:
                # Use the provided filename
                image_path = output_dir / f"{filename}.png"
            else:
                # Generate a default filename
                timestamp = int(time.time())
                seed_str = f"_seed{current_seed}" if current_seed is not None else ""
                if filename:
                    # If multiple images with a filename, append an index
                    image_path = output_dir / f"{filename}_{i+1}.png"
                else:
                    # Default filename with timestamp
                    image_path = output_dir / f"flux_{timestamp}{seed_str}_{i+1}.png"

            output.images[0].save(image_path)
            generated_image_paths.append(str(image_path))

            generation_time = time.time() - start_time
            print(f"Image saved to {image_path} (took {generation_time:.2f}s)")

        except Exception as e:
            print(f"Error generating image: {e}")
            continue

        # If generating multiple images with a seed, increment the seed
        if current_seed is not None and num_images > 1:
            current_seed += 1
            generator = torch.Generator(
                device if device != "mps" else "cpu"
            ).manual_seed(current_seed)

    return generated_image_paths


# Example usage
if __name__ == "__main__":
    # Example configuration for Flux
    dev_config = {
        "model_path": "black-forest-labs/FLUX.1-dev",
        "num_images": 1,
        "num_inference_steps": 50,
        "prompt": "a tiny astronaut hatching from an egg on the moon",
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "seed": 66,
        "enable_vae_tiling": True,
        "enable_cpu_offload": True,
        "filename": "flux_dev",
    }

    schnell_config = {
        "model_path": "black-forest-labs/FLUX.1-schnell",
        "num_images": 1,
        "num_inference_steps": 6,
        "prompt": "a tiny astronaut hatching from an egg on the moon",
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "seed": 66,
        "enable_vae_tiling": True,
        "enable_cpu_offload": True,
        "filename": "flux_schnell",
    }

    # Generate images
    generate_flux_images(**schnell_config)
