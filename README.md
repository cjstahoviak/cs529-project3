# cs529-project3
Neural Networks

## Setup
Since the training data is over a GB in size, it needs to be manually placed into this project. All that is missing is data\raw\train which can be retrieved from Google Drive.

Next step requires a conda installation on your system. First setup the environment:
```bash
conda env create -f environment.yml
conda activate cs529_proj3
```


## File Manifest
Project tree with description and contributions made on each source file.
```bash
```

## Contributions
Nick Livingstone:
- Setup

Calvin Stahoviak:
- Setup 

## Kaggle
Kaggle Score: 

Team Name: Nick & Calvin

Date Run: 

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