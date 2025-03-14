import random
import os
import uuid

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


def generate_face_prompt(
    gender=None,
    age_group=None,
    ethnicity_region=None,
    image_type="portrait photograph",
    style="sharp and detailed high quality professional photography",
    background="plain background",
    lighting="natural lighting",
    expression="natural expression",
):
    """
    Generate a prompt for diverse face generation.

    Parameters
    ----------
    gender : str, optional
        Gender of the face ('male' or 'female'). If None, randomly selected.
    age_group : str, optional
        Age group ('child', 'young_adult', 'adult', 'middle_aged', 'senior'). If None, randomly selected.
    ethnicity_region : str, optional
        Ethnicity region from REGIONS list. If None, randomly selected.
    image_type : str, optional
        Type of image (e.g., "portrait photograph", "headshot")
    style : str, optional
        Style of the image
    background : str, optional
        Background description
    lighting : str, optional
        Lighting description
    expression : str, optional
        Facial expression

    Returns
    -------
    str
        Generated prompt for face generation
    dict
        Metadata about the generated face prompt
    """
    # Define possible values for randomization
    genders = ["male", "female"]

    # Define age group ranges
    age_group_ranges = {
        "child": (5, 14),
        "young_adult": (15, 30),
        "adult": (30, 40),
        "middle_aged": (40, 55),
        "senior": (60, 90),  # Max age set to 90
    }

    # Ethnicity features based on region
    ethnicity_features = {
        "African": "African features",
        "Anglo-Saxon": "Anglo-Saxon features",
        "Central_Asian": "Central Asian features",
        "Eastern_Asian": "Eastern Asian features",
        "European": "European features",
        "Far_Eastern": "Far Eastern features",
        "Latin_Hispanic": "Latin/Hispanic features",
        "Middle_Eastern": "Middle Eastern features",
    }

    # Randomly select if not provided
    if gender is None:
        gender = random.choice(genders)

    if age_group is None:
        # Apply the distribution: 20% children, 80% adults
        # For adults: distribute among young adults, adults, middle-aged, and seniors
        if random.random() < 0.2:  # 20% chance for children
            age_group = "child"
        else:
            adult_groups = ["young_adult", "adult", "middle_aged", "senior"]
            age_group = random.choice(adult_groups)

    if ethnicity_region is None:
        # Randomly select an ethnicity region
        ethnicity_region = random.choice(REGIONS)

    # Generate a specific age within the range for the selected age group
    min_age, max_age = age_group_ranges[age_group]
    specific_age = random.randint(min_age, max_age)

    # Create age description with specific age
    if age_group == "child":
        age_description = f"child ({specific_age} years old)"
    elif age_group == "young_adult":
        age_description = f"young adult ({specific_age} years old)"
    elif age_group == "adult":
        age_description = f"adult ({specific_age} years old)"
    elif age_group == "middle_aged":
        age_description = f"middle-aged adult ({specific_age} years old)"
    else:  # senior
        age_description = f"senior ({specific_age} years old)"

    # Build the prompt
    prompt_parts = [
        f"{image_type} of a {gender}",
        age_description,
        f"with {ethnicity_features[ethnicity_region]}",
        background,
        lighting,
        expression,
        style,
    ]

    prompt = ", ".join(prompt_parts)

    # Create metadata
    metadata = {
        "gender": gender,
        "age_group": age_group,
        "specific_age": specific_age,
        "ethnicity_region": ethnicity_region,
    }

    return prompt, metadata


def generate_diverse_face_prompts(count, ensure_diversity=True):
    """
    Generate a diverse set of face prompts.

    Parameters
    ----------
    count : int
        Number of prompts to generate
    ensure_diversity : bool, optional
        If True, ensures diversity across gender, age, and ethnicity

    Returns
    -------
    list
        List of generated prompts
    list
        List of metadata dictionaries for each prompt
    """
    prompts = []
    metadata_list = []

    if ensure_diversity and count >= 8:
        # Ensure gender balance
        genders = ["male", "female"]

        # Ensure age group distribution
        # 20% children (split between genders)
        # 80% adults (split between age groups and genders)
        child_count = max(1, int(count * 0.2))
        adult_count = count - child_count
        young_adult_count = int(adult_count / 3)
        middle_aged_count = int(adult_count / 3)
        senior_count = adult_count - young_adult_count - middle_aged_count

        # Create a distribution plan
        distribution = []

        # Add children
        for i in range(child_count):
            gender = "male" if i < child_count / 2 else "female"
            ethnicity_region = REGIONS[i % len(REGIONS)]
            distribution.append(
                {
                    "gender": gender,
                    "age_group": "child",
                    "ethnicity_region": ethnicity_region,
                }
            )

        # Add young adults
        for i in range(young_adult_count):
            gender = "male" if i < young_adult_count / 2 else "female"
            ethnicity_region = REGIONS[(i + child_count) % len(REGIONS)]
            distribution.append(
                {
                    "gender": gender,
                    "age_group": "young_adult",
                    "ethnicity_region": ethnicity_region,
                }
            )

        # Add middle-aged adults
        for i in range(middle_aged_count):
            gender = "male" if i < middle_aged_count / 2 else "female"
            ethnicity_region = REGIONS[
                (i + child_count + young_adult_count) % len(REGIONS)
            ]
            distribution.append(
                {
                    "gender": gender,
                    "age_group": "middle_aged",
                    "ethnicity_region": ethnicity_region,
                }
            )

        # Add seniors
        for i in range(senior_count):
            gender = "male" if i < senior_count / 2 else "female"
            ethnicity_region = REGIONS[
                (i + child_count + young_adult_count + middle_aged_count) % len(REGIONS)
            ]
            distribution.append(
                {
                    "gender": gender,
                    "age_group": "senior",
                    "ethnicity_region": ethnicity_region,
                }
            )

        # Shuffle the distribution
        random.shuffle(distribution)

        # Generate prompts based on the distribution
        for item in distribution:
            gender = item["gender"]
            age_group = item["age_group"]
            ethnicity_region = item["ethnicity_region"]

            prompt, metadata = generate_face_prompt(
                gender=gender, age_group=age_group, ethnicity_region=ethnicity_region
            )
            prompts.append(prompt)
            metadata_list.append(metadata)

        # If we need more prompts than our distribution plan
        remaining = count - len(distribution)
        for _ in range(remaining):
            prompt, metadata = generate_face_prompt()
            prompts.append(prompt)
            metadata_list.append(metadata)
    else:
        # Simple random generation
        for _ in range(count):
            prompt, metadata = generate_face_prompt()
            prompts.append(prompt)
            metadata_list.append(metadata)

    return prompts, metadata_list


def generate_filename(metadata):
    """
    Generate a filename based on metadata following the structure:
    <gender>_<region>_<random_id>.jpg

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing gender and ethnicity_region

    Returns
    -------
    str
        Generated filename
    """
    gender = metadata["gender"]
    # Convert region to lowercase and replace / with _
    region = metadata["ethnicity_region"].lower().replace("/", "_").replace("-", "_")
    # Generate a random ID (using first 8 characters of a UUID)
    random_id = str(uuid.uuid4())[:8]

    return f"{gender}_{region}_{random_id}.jpg"


# Example usage
if __name__ == "__main__":
    # Generate a single face prompt
    prompt, metadata = generate_face_prompt()
    filename = generate_filename(metadata)
    print(f"Single prompt: {prompt}")
    print(f"Metadata: {metadata}")
    print(f"Filename: {filename}")

    # Generate 10 diverse face prompts
    prompts, metadata_list = generate_diverse_face_prompts(10)
    print("\nGenerated 10 diverse face prompts:")
    for i, (prompt, metadata) in enumerate(zip(prompts, metadata_list)):
        filename = generate_filename(metadata)
        print(f"\n{i+1}. {prompt}")
        print(f"   Metadata: {metadata}")
        print(f"   Filename: {filename}")
