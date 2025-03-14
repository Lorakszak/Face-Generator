# Face-Generator
Project serving the sole purpose of human face generation, intended to use for generating synthetic faces for **World Memory Championship** Names&Faces discipline.
Utilizes Flux-dev or Flux-schnell with focus on diversity when it comes to gender, age, ethnicity, distinctive features and accessories.


## Setup
You need Python 3 installed (Python 3.10 is recommended as it was tested with this version). The setup script will work with other Python 3 versions, but some features may not work as expected.

```bash
./setup.sh
```

Flux-dev or Flux-schnell model will be downloaded automatically when the `generate_dataset.py` script is run using given model.


## Usage
Before running the script, you need to configure the generation parameters in `src/config.py`. The file contains a `GENERATE_DATASET_CONFIG` dictionary that you should modify according to your needs. An example configuration is already provided.

```bash
python generate_dataset.py
```

## TODO
- [x] Add more detailed prompt
- [x] Add setup.sh and requirements.txt for quick setup
- [x] Fill in the README nicely and with more details
- [x] Apply numpy style docstrings to the code