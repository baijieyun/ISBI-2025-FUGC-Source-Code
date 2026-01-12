# Fetal_BCP - Fetal Ultrasound Grand Challenge 

This repository contains the implementation for training models for the Fetal Ultrasound Grand Challenge.

## Project Structure

```
.
├── Configs/            # Configuration files
├── Datasets/          # Dataset processing and augmentation
├── Models/            # Model architectures
├── Utils/            # Utility functions
├── Fetal_BCP.py      # Main training script
├── download.ipynb    # Dataset download and preprocessing notebook
└── requirements.txt  # Project dependencies
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download and preprocess the dataset:

   - Run all cells in `download.ipynb` to download and prepare the dataset
   - The processed data will be stored in the `data_processed/fugc/` directory

## Configuration

The main configuration file is `Configs/cf_fugc.yml`. Key settings include:

- Dataset paths and preprocessing
- Training parameters (batch size, learning rate, etc.)
- Model architecture settings
- Augmentation parameters

To modify data augmentation settings, edit the `get_dataset_without_full_label()` function in `Datasets/create_dataset.py`.

## Training

To train the model:

```bash
python3 Fetal_BCP.py --config_yml Configs/cf_fugc.yml
```

The training checkpoints will be saved in the directory specified by `data.save_folder` in the config file. For each fold, the following files will be saved:

- `pre_train_best.pth`: Best model weights from pre-training phase
- `pre_train_optim.pth`: Optimizer state from pre-training phase
- `self_train_best.pth`: Best model weights from self-training phase
- `self_train_optim.pth`: Optimizer state from self-training phase