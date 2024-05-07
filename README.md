# cs529-project3
Neural Networks

## Setup
Since the training data is over a GB in size, it needs to be manually placed into this project. All that is missing is data\raw\train which can be retrieved from Google Drive.

This project requires a conda installation on your system. First setup the environment:
```bash
conda env create -f environment.yml
conda activate cs529_proj3
```
Run the feature extraction scripts to populate `/data`:
```bash
python ./src/feature/feature_extraction.py
python ./src/feature/spectrogram_extraction_manual.py
```
Run different models:
```bash
python ./src/model/train_cnn_testing.py
```

## File Manifest
The critical files are described here:
- `feature_model.py`: Traditional Neural Network Model based on features extracted from the audio files using librosa library.
- `spectrogram_extraction_manual.py`: Loads in audio data and geenrates spectrogram images of 10 different types The spectrograms are organized and saved into `data/processed/spectrograms/`
- `train_cnn_testing.py`: Main module which loads data into memory, transforms, trains a CNN given a set of hyper-parameters, and graphs the results.
- `convolutional_neural_net.py`: Inherits the PyTorch neural network module and defines a dynamically built convolutional neural network dependant on the hyper-parameters set in `train_cnn_testing.py`. Also includes methods for fitting and evaluating the model.
- `transfer_model.py`: Transfer learning model for audio classification using the VGGish model as a backbone. Contains the VGGishTransferModel class which is a PyTorch Lightning module that uses the VGGish model as a backbone. The model is trained on the given audio data and the predictions are saved to a Kaggle submission file when run as a script.

### Full Project Structure
Project tree with description and contributions made on each source file:
```bash
CS529-PROJECT2:
C:.
├───data/
│   ├───preprocessed/: Holds pickles of feature extracted data frames
│   ├───processed/
│   │   └───feature_extracted/: Holds pickles of feature extracted data frames
│   │   └───spectrograms/: Contains all types spectrograms for test and train
│   │
│   └───raw/
│       └───train/
│       └───test/
│
├───docs/
│   └───results/: CSV files of testing results
│       Music classification with Neural Networks.pdf
│
├───figures/: Contains all figures
│
├───notebooks/
│   └───images/: Plots generated from notebooks
│       mlp_batch_result_plots.ipynb: (CALVIN) Plots for MLP model
│       mlp_epoch_result_plots.ipynb: (CALVIN) Plots for MLP model
│       mlp_lr_result_plots.ipynb: (CALVIN) Plots for MLP model
│       mlp_transfer_result_plots.ipynb: (CALVIN) Plots for transfer learning model
│
├───src/
│   ├───data/:
│   │       pickle_data.py: (NICK)
│   ├───feature/:
│   │       custom_transformers.py: (NICK)
│   │       feature_extraction.py: (NICK)
│   │       spectrogram_experimentation.py: (CALVIN)
│   │       spectrogram_extraction_manual.py: (CALVIN)
│   │       spectrogram_extraction_pipeline.py: (CALVIN)
│   ├───feature_model/: 
│   │       feature_data_loader.py: (NICK)
│   │       feature_model_cv.py: (NICK)
│   │       feature_model.py: (NICK)
│   ├───model/:
│   │       convolutional_neural_net.py: (CALVIN)
│   │       device.py: (CALVIN)
│   │       train_cnn_testing.py: (CALVIN)
│   │       view_transformation.py: (CALVIN)
│   └───transfer_model/:
│           audio_data_module.py: (NICK)
│           transfer_model.py: (NICK)
│       utils.py: (NICK) Utility functions for other files
│       __init__.py: (NICK) Python template file
│
└───trained_model/: Trained models from train_cnn_testing.py
```

## Contributions
Nick Livingstone:
- Setup

Calvin Stahoviak:
- Setup 

## Kaggle
Kaggle Score: 0.89 (1st)

Team Name: Nick & Calvin

Date Run: 5/5/2024

# Brainstorming
- Generating spectrograms:
    - Generate using MFCC components
    - Generate using Chromogram components
    - More @ librosa (https://librosa.org/doc/0.10.1/feature.html)
    - Consider generating multiple spectrograms and have each be a different channel 

- Data Augmentation:
    - Translation
    - Cropping

- CNN Development:
    - Write a multi channel CNN, similar to rgb channel but for different freq bands?

- Hyperparameter Optimization
    - Consider uptuna or hyperopt instead of ray tune

- Post analysis 
    - Gradcam
    - Run SOTA tests for model comparison