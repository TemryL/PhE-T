# PhE-T: A Transformer-based Approach for Phenotype Representation and Multi-task Disease Risk Prediction

## Setup Instructions

### Cloning the Repository

First, clone the repository using the following command:

```bash
git clone --recurse-submodules -j8 https://github.com/TemryL/PhE-T.git
```

### Installing Dependencies

Install the required dependencies with Python 3.12:

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
python commands/create_data.py $UKB_FOLDER ../data/field_to_basket.json ../data/raw.csv
```

### Preprocessing Raw Data

Return to the root folder and preprocess the raw data with the following command:

```bash
python preprocess.py data/raw.csv configs/preprocess_cfg.py data/preprocessed.csv
```

### Generating Splits

To split the dataset in train/val/test sets, run the following command:

```bash
python scripts/split.py --data_path data/preprocessed.csv --val_size 10000 --test_size 10000 --save_dir data/
```

### Preparing Spirometry Data

Prepare the spirometry data by running the following notebooks: [notebooks/prepare_spiro.ipynb](notebooks/prepare_spiro.ipynb).

## Training

```bash
python train.py --model='phet' \
--nb_epochs=10 \
--nb_gpus=1 \
--nb_nodes=1 \
--nb_workers=20 \
--pin_memory \
--config='configs/train_phet_cfg.py' \
--run_name='v0'
```

```bash
python train.py --model='as-resnet' \
--nb_epochs=10 \
--nb_gpus=1 \
--nb_nodes=1 \
--nb_workers=20 \
--pin_memory \
--config='configs/train_as-resnet_cfg.py' \
--run_name='v0'
```

```bash
python train.py --model='as-phet' \
--nb_epochs=10 \
--nb_gpus=1 \
--nb_nodes=1 \
--nb_workers=20 \
--pin_memory \
--config='configs/train_as-phet_cfg.py' \
--run_name='v0'
```

## Prediction

```bash
python predict.py --model='phet' \
--ckpt_path=ckpts/PhE-T/v0/best-epoch=3-step=3842.ckpt \
--out_dir=scores/phet \
--nb_workers=8 \
--config='configs/train_phet_cfg.py'
```

```bash
python predict.py --model='as-phet' \
--ckpt_path=ckpts/AsthmaPhE-T/v0/best-epoch=6-step=5030.ckpt \
--out_dir=scores/as-phet \
--nb_workers=8 \
--config='configs/train_as-phet_cfg.py'
```

## Evaluate

```bash
python scripts/generate_results.py scores/phet results/phet
python scripts/generate_results.py scores/as-phet results/as-phet
```
