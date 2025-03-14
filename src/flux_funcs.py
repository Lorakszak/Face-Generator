import torch
from diffusers import FluxPipeline
from pathlib import Path
import time

# Global variable to store the loaded pipeline
_FLUX_PIPELINE = None


def get_flux_pipeline(
    model_path,
    device="cuda",
    dtype="bfloat16",
    enable_cpu_offload=False,
    enable_vae_slicing=False,
    enable_vae_tiling=False,
):
    """
    Get or create a Flux pipeline, reusing if already loaded

    Parameters
    ----------
    model_path : str
        Path to the local Flux model or HF model ID
    device : str
        Device to run inference on: 'cuda', 'cpu', or 'mps'
    dtype : str
        Model precision: 'float32', 'float16', or 'bfloat16'
    enable_cpu_offload : bool
        Enable sequential CPU offloading to save VRAM
    enable_vae_slicing : bool
        Enable VAE slicing to save memory
    enable_vae_tiling : bool
        Enable VAE tiling to save memory and process larger images

    Returns
    -------
    FluxPipeline
        The loaded Flux pipeline
    """
    global _FLUX_PIPELINE

    # If pipeline is already loaded with the same model, return it
    if _FLUX_PIPELINE is not None:
        return _FLUX_PIPELINE

    # Convert dtype string to torch dtype
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Load the model
    print(f"Loading Flux model from {model_path}...")
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        if enable_cpu_offload:
            print("Enabling sequential CPU offload...")
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    elif (
        device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        pipe = pipe.to("mps")
    else:
        print("Using CPU for inference (this will be slow)...")
        pipe = pipe.to("cpu")

    # Enable memory-saving features
    if enable_vae_slicing:
        print("Enabling VAE slicing...")
        pipe.enable_vae_slicing()

    if enable_vae_tiling:
        print("Enabling VAE tiling...")
        pipe.enable_vae_tiling()

    # Store the pipeline globally
    _FLUX_PIPELINE = pipe

    return pipe


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
        Maximum sequence length for text encoder
    num_images : int, optional
        Number of images to generate
    output_dir : str, optional
        Directory to save generated images
    filename : str, optional
        Custom filename for the generated image

    Returns
    -------
    list
        Paths to generated images
    """
    # Validate parameters
    if model_path is None:
        raise ValueError("model_path must be provided")
    if prompt is None:
        raise ValueError("prompt must be provided")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get the pipeline (reusing if already loaded)
    pipe = get_flux_pipeline(
        model_path=model_path,
        device=device,
        dtype=dtype,
        enable_cpu_offload=enable_cpu_offload,
        enable_vae_slicing=enable_vae_slicing,
        enable_vae_tiling=enable_vae_tiling,
    )

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


# Function to clear the pipeline from memory if needed
def clear_flux_pipeline():
    """Clear the loaded Flux pipeline from memory"""
    global _FLUX_PIPELINE
    if _FLUX_PIPELINE is not None:
        del _FLUX_PIPELINE
        _FLUX_PIPELINE = None
        # Force CUDA cache clear if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Flux pipeline cleared from memory")


# Example usage
if __name__ == "__main__":
    # Example configuration for Flux
    config = {
        "model_path": "black-forest-labs/FLUX.1-dev",
        "num_images": 1,
        "num_inference_steps": 50,
        "prompt": "a tiny astronaut hatching from an egg on the moon",
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "seed": 66,
        "enable_vae_tiling": True,
        "filename": "astronaut_egg_moon",
    }

    # Generate images
    generate_flux_images(**config)

    # Generate another image with the same model (will reuse the loaded pipeline)
    generate_flux_images(
        model_path=config["model_path"],
        prompt="a cat wearing a space helmet on mars",
        filename="space_cat",
    )

    # Clear the pipeline when done
    clear_flux_pipeline()
