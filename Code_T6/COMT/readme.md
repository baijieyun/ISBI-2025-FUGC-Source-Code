# COMT

## File composition
- COMT: COMT: Co-training mean teachers semi-supervised training framework for cervical segmentation
- post: Prediction post-processing code to generate pseudo labels for high-quality predictions
- retrain: Supervised retraining using high quality pseudo labels, stronger enhancements, and smaller models

## usage
- COMT:
  - 1. Modify the path and experiment information in the config/config.toml file
  - 2. python train.py

- post:
  - 1. Place the trained unet16 weights inside the post folder
  - 2. Modify the path information of the main function inside predict.py and the pseudo-label save folder store_dir, and finally run python predict.py
  - 3. [Optional] Modify the internal path information of process.py to visualize post-processing to predict false labels

- retrain:
  - 1. The pseudo-label after post-processing is used as the label of the unlabeled image of the original dataset
  - 2. Modify the path and experiment information in the config/config.toml file
  - 3. python retrain.py