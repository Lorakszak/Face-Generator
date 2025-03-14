import random
import json
from pathlib import Path
from tqdm import tqdm

# Import our modules
from src.prompts import generate_diverse_face_prompts, generate_filename
from src.flux_funcs import generate_flux_images

from src.config import GENERATE_DATASET_CONFIG as CONFIG

if __name__ == "__main__":
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate diverse face prompts
    print(f"Generating {CONFIG['num_faces']} diverse face prompts...")
    prompts, metadata_list = generate_diverse_face_prompts(
        CONFIG["num_faces"], ensure_diversity=False
    )

    # Set up seed if provided
    current_seed = CONFIG["seed"]

    # Store results
    results = []

    # Generate images for each prompt
    for i, (prompt, metadata) in enumerate(
        tqdm(zip(prompts, metadata_list), total=len(prompts))
    ):
        print(f"\nGenerating face {i+1}/{CONFIG['num_faces']}:")
        print(
            f"Gender: {metadata['gender']}, Age: {metadata['age_group']}, Region: {metadata['ethnicity_region']}"
        )
        print(f"Prompt: {prompt}")

        # Generate a seed if not provided
        if current_seed is None:
            image_seed = random.randint(1, 1000000)
        else:
            image_seed = current_seed
            current_seed += 1

        # Generate the filename
        filename = generate_filename(metadata).replace(".jpg", "")

        # Configure Flux generation
        generation_config = {
            "model_path": CONFIG["model_path"],
            "prompt": prompt,
            "height": CONFIG["height"],
            "width": CONFIG["width"],
            "guidance_scale": CONFIG["guidance_scale"],
            "num_inference_steps": CONFIG["num_inference_steps"],
            "seed": image_seed,
            "output_dir": str(output_dir),
            "filename": filename,
            "enable_cpu_offload": CONFIG["enable_cpu_offload"],
            "enable_vae_slicing": CONFIG["enable_vae_slicing"],
            "enable_vae_tiling": CONFIG["enable_vae_tiling"],
        }

        # Generate the image
        try:
            image_paths = generate_flux_images(**generation_config)

            # Store result
            if image_paths:
                result = {
                    "path": image_paths[0],
                    "prompt": prompt,
                    "metadata": metadata,
                    "seed": image_seed,
                }
                results.append(result)

                print(f"Image saved as: {image_paths[0]}")
        except Exception as e:
            print(f"Error generating image: {e}")
            continue

    # Save metadata if requested
    if CONFIG["save_metadata"] and results:
        metadata_path = output_dir / "face_dataset_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetadata saved to {metadata_path}")

    print(
        f"\nGenerated {len(results)} face images out of {CONFIG['num_faces']} requested"
    )
    print(f"Images saved to {output_dir}")
