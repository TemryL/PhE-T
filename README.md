## Setup Instructions

### Cloning the Repository
First, clone the repository using the following command:
```bash
git clone --recurse-submodules -j8 git@github.com:repo  #TODO
```

### Installing Dependencies
Install the required dependencies with Python 3.11:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Setting Environment Variables
Set your environment variables:
```bash
export UKB_FOLDER=...
export PROJECT_ID=...
```

### Creating Raw Data
Follow these steps to create raw data. More details can be found in the [UKB-Tools README](https://github.com/TemryL/UKB-Tools/blob/main/README.md):
```bash
cd UKB-Tools
python commands/get_newest_baskets.py $UKB_FOLDER $PROJECT_ID ../data/ukb_fields.txt ../data/field_to_basket.json
python commands/create_data.py $UKB_FOLDER ../data/field_to_basket.json ../data/raw_data.csv
```

### Preprocessing Raw Data
Return to the root folder and preprocess the raw data with the following command:
```bash
python preprocess.py data/raw_data.csv configs/preprocess_cfg.py data/preprocessed_data.csv
```