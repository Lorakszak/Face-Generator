"""
Configuration for the face dataset generation.

This module contains the configuration for the face dataset generation.
As well as REGIONS, DISTINCTIVE_FEATURES and ACCESSORIES.
It also contains the DEFAULT_FLUX_CONFIG as an example configuration for the Flux pipeline.
"""

GENERATE_DATASET_CONFIG = {
    # Dataset parameters
    "num_faces": 10,
    "output_dir": "outputs",
    "save_metadata": True,
    # Model parameters
    "model_path": "black-forest-labs/FLUX.1-schnell",
    # Image parameters
    "height": 1024,
    "width": 1024,
    # Generation parameters
    "seed": None,  # Random seed (None for random seeds)
    "num_inference_steps": 5,
    "guidance_scale": 3.5,
    # Performance options
    "enable_cpu_offload": True,  # RTX4090 doesn't want to go BRRRRR without quantized model :(
    "enable_vae_slicing": False,
    "enable_vae_tiling": False,
}


# Define the standardized regions
REGIONS = [
    "African",
    "Anglo-Saxon",
    "Central_Asian",
    "Eastern_Asian",
    "European",
    "Far_Eastern",
    "Latin_Hispanic",
    "Middle_Eastern",
]

# Define distinctive features
DISTINCTIVE_FEATURES = [
    "freckles",
    "dimples",
    "wrinkles",
    "scar",
    "birthmark",
    "mole near lip",
    "mole on cheek",
    "mole on forehead or nose",
    "acne",
    "cleft chin",
    "gap between teeth",
    "tattooed",
    "beauty mark",
    "dark circles under eyes",
    "prominent cheekbones",
    "strong jawline",
    "large ears",
    "crooked nose",
]

# Define accessories
ACCESSORIES = [
    "round glasses",
    "square glasses",
    "rimless glasses",
    "sunglasses",
    "earrings",
    "nose ring",
    "lip piercing",
    "eyebrow piercing",
    "septum piercing",
    "necklace",
    "choker",
    "scarf",
    "headscarf",
    "turban",
    "baseball cap",
    "beanie",
    "headphones",
    "hair clips",
    "bandana",
    "tie",
    "bow tie",
    "pedant necklace",
    "bucket hat",
    "cowboy hat",
    "beret",
    "hood",
    "flower in hair",
    "brooch",
]


# Default FLUX pipeline configuration
DEFAULT_FLUX_CONFIG = {
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
