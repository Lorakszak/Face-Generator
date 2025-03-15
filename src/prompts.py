import random
import os
import uuid

# Define the standardized regions
from src.config import REGIONS, DISTINCTIVE_FEATURES, ACCESSORIES


def generate_face_prompt(
    gender=None,
    age_group=None,
    ethnicity_region=None,
    image_type="portrait photograph",
    style="realistic, detailed, high quality, professional",
    background="plain background",
    lighting="professional lighting",
    expression="neutral expression",
    distinctive_features_chance=0.15,
    multiple_distinctive_features=False,
    min_num_of_distinctive_features=1,
    max_num_of_distinctive_features=3,
    accessories_chance=0.15,
    multiple_accessories=False,
    min_num_of_accessories=1,
    max_num_of_accessories=2,
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
    distinctive_features_chance : float, optional
        Chance (0.0-1.0) to include distinctive features
    multiple_distinctive_features : bool, optional
        Whether to allow multiple distinctive features
    min_num_of_distinctive_features : int, optional
        Minimum number of distinctive features if multiple is True
    max_num_of_distinctive_features : int, optional
        Maximum number of distinctive features if multiple is True
    accessories_chance : float, optional
        Chance (0.0-1.0) to include accessories
    multiple_accessories : bool, optional
        Whether to allow multiple accessories
    min_num_of_accessories : int, optional
        Minimum number of accessories if multiple is True
    max_num_of_accessories : int, optional
        Maximum number of accessories if multiple is True

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
        "middle_aged": (40, 50),
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
        # Apply the distribution: child (15%), young adult (25%), adult (25%),
        # middle_aged (20%), senior (15%)
        random_value = random.random()
        if random_value < 0.15:  # 15% chance for children
            age_group = "child"
        elif random_value < 0.40:  # 25% chance for young adults (0.15 + 0.25)
            age_group = "young_adult"
        elif random_value < 0.65:  # 25% chance for adults (0.40 + 0.25)
            age_group = "adult"
        elif random_value < 0.85:  # 20% chance for middle-aged (0.65 + 0.20)
            age_group = "middle_aged"
        else:  # 15% chance for seniors (remaining)
            age_group = "senior"

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

    # Determine distinctive features
    selected_distinctive_features = []
    if random.random() < distinctive_features_chance and DISTINCTIVE_FEATURES:
        if (
            multiple_distinctive_features
            and len(DISTINCTIVE_FEATURES) >= min_num_of_distinctive_features
        ):
            # Choose a random number of features between min and max
            num_features = min(
                random.randint(
                    min_num_of_distinctive_features, max_num_of_distinctive_features
                ),
                len(DISTINCTIVE_FEATURES),  # Can't select more than available
            )
            # Sample without replacement
            selected_distinctive_features = random.sample(
                DISTINCTIVE_FEATURES, num_features
            )
        else:
            # Just choose one feature
            selected_distinctive_features = [random.choice(DISTINCTIVE_FEATURES)]

    # Determine accessories
    selected_accessories = []
    if random.random() < accessories_chance and ACCESSORIES:
        if multiple_accessories and len(ACCESSORIES) >= min_num_of_accessories:
            # Choose a random number of accessories between min and max
            num_accessories = min(
                random.randint(min_num_of_accessories, max_num_of_accessories),
                len(ACCESSORIES),  # Can't select more than available
            )
            # Sample without replacement
            selected_accessories = random.sample(ACCESSORIES, num_accessories)
        else:
            # Just choose one accessory
            selected_accessories = [random.choice(ACCESSORIES)]

    # Build the prompt with the new template
    prompt = f"{image_type} of a {gender}, {age_description} with {ethnicity_features[ethnicity_region]}, {style}"

    # Add distinctive features if present
    if selected_distinctive_features:
        features_text = ", ".join(selected_distinctive_features)
        prompt += f", {features_text}"

    # Add accessories if present
    if selected_accessories:
        accessories_text = ", ".join(selected_accessories)
        prompt += f", {accessories_text}"

    # Add remaining elements
    prompt += f", {background}, {lighting}, {expression}"

    # Create metadata
    metadata = {
        "gender": gender,
        "age_group": age_group,
        "specific_age": specific_age,
        "ethnicity_region": ethnicity_region,
        "distinctive_features": (
            selected_distinctive_features if selected_distinctive_features else []
        ),
        "accessories": selected_accessories if selected_accessories else [],
    }

    return prompt, metadata


def generate_diverse_face_prompts(
    count,
    ensure_diversity=True,
    distinctive_features_chance=0.15,
    multiple_distinctive_features=False,
    min_num_of_distinctive_features=1,
    max_num_of_distinctive_features=3,
    accessories_chance=0.15,
    multiple_accessories=False,
    min_num_of_accessories=1,
    max_num_of_accessories=2,
):
    """
    Generate a diverse set of face prompts.

    Parameters
    ----------
    count : int
        Number of prompts to generate
    ensure_diversity : bool, optional
        If True, ensures diversity across gender, age, and ethnicity
    distinctive_features_chance : float, optional
        Chance (0.0-1.0) to include distinctive features
    multiple_distinctive_features : bool, optional
        Whether to allow multiple distinctive features
    min_num_of_distinctive_features : int, optional
        Minimum number of distinctive features if multiple is True
    max_num_of_distinctive_features : int, optional
        Maximum number of distinctive features if multiple is True
    accessories_chance : float, optional
        Chance (0.0-1.0) to include accessories
    multiple_accessories : bool, optional
        Whether to allow multiple accessories
    min_num_of_accessories : int, optional
        Minimum number of accessories if multiple is True
    max_num_of_accessories : int, optional
        Maximum number of accessories if multiple is True

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
        # Ensure age group distribution
        # 20% children (split between genders)
        # 80% adults (split between age groups and genders)
        child_count = max(1, int(count * 0.2))
        adult_count = count - child_count
        young_adult_count = int(adult_count / 4)
        adult_count_middle = int(adult_count / 4)
        middle_aged_count = int(adult_count / 4)
        senior_count = (
            adult_count - young_adult_count - adult_count_middle - middle_aged_count
        )

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

        # Add adults
        for i in range(adult_count_middle):
            gender = "male" if i < adult_count_middle / 2 else "female"
            ethnicity_region = REGIONS[
                (i + child_count + young_adult_count) % len(REGIONS)
            ]
            distribution.append(
                {
                    "gender": gender,
                    "age_group": "adult",
                    "ethnicity_region": ethnicity_region,
                }
            )

        # Add middle-aged adults
        for i in range(middle_aged_count):
            gender = "male" if i < middle_aged_count / 2 else "female"
            ethnicity_region = REGIONS[
                (i + child_count + young_adult_count + adult_count_middle)
                % len(REGIONS)
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
                (
                    i
                    + child_count
                    + young_adult_count
                    + adult_count_middle
                    + middle_aged_count
                )
                % len(REGIONS)
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
                gender=gender,
                age_group=age_group,
                ethnicity_region=ethnicity_region,
                distinctive_features_chance=distinctive_features_chance,
                multiple_distinctive_features=multiple_distinctive_features,
                min_num_of_distinctive_features=min_num_of_distinctive_features,
                max_num_of_distinctive_features=max_num_of_distinctive_features,
                accessories_chance=accessories_chance,
                multiple_accessories=multiple_accessories,
                min_num_of_accessories=min_num_of_accessories,
                max_num_of_accessories=max_num_of_accessories,
            )
            prompts.append(prompt)
            metadata_list.append(metadata)

        # If we need more prompts than our distribution plan
        remaining = count - len(distribution)
        for _ in range(remaining):
            prompt, metadata = generate_face_prompt(
                distinctive_features_chance=distinctive_features_chance,
                multiple_distinctive_features=multiple_distinctive_features,
                min_num_of_distinctive_features=min_num_of_distinctive_features,
                max_num_of_distinctive_features=max_num_of_distinctive_features,
                accessories_chance=accessories_chance,
                multiple_accessories=multiple_accessories,
                min_num_of_accessories=min_num_of_accessories,
                max_num_of_accessories=max_num_of_accessories,
            )
            prompts.append(prompt)
            metadata_list.append(metadata)
    else:
        # Simple random generation
        for _ in range(count):
            prompt, metadata = generate_face_prompt(
                distinctive_features_chance=distinctive_features_chance,
                multiple_distinctive_features=multiple_distinctive_features,
                min_num_of_distinctive_features=min_num_of_distinctive_features,
                max_num_of_distinctive_features=max_num_of_distinctive_features,
                accessories_chance=accessories_chance,
                multiple_accessories=multiple_accessories,
                min_num_of_accessories=min_num_of_accessories,
                max_num_of_accessories=max_num_of_accessories,
            )
            prompts.append(prompt)
            metadata_list.append(metadata)

    return prompts, metadata_list


def generate_filename(metadata):
    """
    Generate a filename based on metadata following the structure:
    <gender>_<region>_<age>_<random_id>.png

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing gender, ethnicity_region, and specific_age

    Returns
    -------
    str
        Generated filename
    """
    gender = metadata["gender"]
    # Convert region to lowercase and replace / with _
    region = metadata["ethnicity_region"].lower().replace("/", "_").replace("-", "_")
    # Get the specific age if available, otherwise use age_group
    age = str(metadata.get("specific_age", metadata["age_group"]))
    # Generate a random ID (using first 8 characters of a UUID)
    random_id = str(uuid.uuid4())[:8]

    return f"{gender}_{region}_{age}_{random_id}.png"
