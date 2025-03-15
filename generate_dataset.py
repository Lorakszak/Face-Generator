import random
import json
from pathlib import Path
from tqdm import tqdm

# Import our modules
from src.prompts import generate_diverse_face_prompts, generate_filename
from src.flux_funcs import generate_flux_images, clear_flux_pipeline

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

    # Track new faces count
    new_faces_count = 0

    # Metadata path
    metadata_path = output_dir / "face_dataset_metadata.json"

    # Load existing metadata if it exists
    if metadata_path.exists() and CONFIG["save_metadata"]:
        try:
            with open(metadata_path, "r") as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing face metadata entries")
        except Exception as e:
            print(f"Error loading existing metadata: {e}")
            existing_results = []
    else:
        existing_results = []

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
        filename = generate_filename(metadata)

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
            "filename": filename.replace(
                ".png", ""
            ),  # Remove .png as it's added by generate_flux_images
            "enable_cpu_offload": CONFIG["enable_cpu_offload"],
            "enable_vae_slicing": CONFIG["enable_vae_slicing"],
            "enable_vae_tiling": CONFIG["enable_vae_tiling"],
        }

        # Generate the image
        try:
            image_paths = generate_flux_images(**generation_config)

            # If image was generated successfully
            if image_paths:
                # Create result entry
                result = {
                    "path": image_paths[0],
                    "prompt": prompt,
                    "metadata": metadata,
                    "seed": image_seed,
                }

                # Append to existing results
                existing_results.append(result)

                # Increment new faces count
                new_faces_count += 1

                # Save metadata after each successful generation
                if CONFIG["save_metadata"]:
                    with open(metadata_path, "w") as f:
                        json.dump(existing_results, f, indent=2)

                print(f"Image saved as: {image_paths[0]}")
                print(f"Metadata updated ({len(existing_results)} total entries)")

        except Exception as e:
            print(f"Error generating image: {e}")
            continue

    # Final summary
    print(
        f"\nGenerated {new_faces_count} new face images out of {CONFIG['num_faces']} requested"
    )
    print(f"Total dataset size: {len(existing_results)} images")
    print(f"Images saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")

    # Clear the pipeline when done
    clear_flux_pipeline()
