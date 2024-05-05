import os
import sys
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import parallel_config
from PIL import Image

# Import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

parallel_config(n_jobs=-2)
WINDOW_SIZE = 1024  # Number of samples in each frame
HOP_LENGTH = int(WINDOW_SIZE / 2)  # Number of samples between successive frames

# File Paths
train_fpath = Path("data/processed/train_data.pkl").resolve()
output_dir = Path(
    "data/processed/spectrograms/amplitude_" + str(WINDOW_SIZE) + "/train"
).resolve()

# Load data
train_df = pd.read_pickle(train_fpath)

# Define the target and feature columns
y_train = train_df["target"]
# Access with loc so it returns a DataFrame
X_train = train_df.loc[:, ["audio"]]

# Find the shortest audio clip and crop all clips to that length
min_audio_length = min(X_train["audio"].apply(len))
X_train["audio"] = X_train["audio"].apply(lambda x: x[:min_audio_length])

genre_count = {}  # Dictionary to keep track of the count for each genre
image_sizes = []

sampling_rate = 22050  # Standard sampling rate for audio data
n_fft = WINDOW_SIZE  # Number of samples to use for each FFT

for idx, audio_clip in enumerate(X_train["audio"]):
    genre_label = y_train.iloc[idx]
    # Initialize or increment the genre count
    if genre_label not in genre_count:
        genre_count[genre_label] = 0
    else:
        genre_count[genre_label] += 1

    output_genre_dir = output_dir / genre_label
    output_genre_dir.mkdir(parents=True, exist_ok=True)

    # Compute the Short-Time Fourier Transform (STFT) of the audio
    stft_matrix = librosa.stft(
        audio_clip, n_fft=n_fft, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH
    )

    # Convert the magnitude of the STFT matrix to decibel (dB) scale
    spectrogram_db = librosa.amplitude_to_db(abs(stft_matrix), ref=np.max)

    # TODO: Consider log-amplitude and log-frequency plots

    # Normalize the spectrogram for image saving: Scale between 0 and 255
    img_array = np.flipud(
        spectrogram_db
    )  # Flip the array to make the lower frequencies appear at the bottom of the image
    img_normalized = (
        255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())
    )
    img_normalized = img_normalized.astype(np.uint8)  # Convert to unsigned byte format

    # Generate filename
    filename = f"{genre_label}_{genre_count[genre_label]:06}_spectrogram.png"
    spectrogram_path = output_genre_dir / filename

    # Save the spectrogram image using PIL
    img = Image.fromarray(img_normalized)
    img.save(spectrogram_path)
    image_sizes.append(img.size)

if image_sizes:
    max_size = max(image_sizes)
    min_size = min(image_sizes)
    print(f"Largest Image Size: {max_size}")
    print(f"Smallest Image Size: {min_size}")
else:
    print("No images were processed.")
