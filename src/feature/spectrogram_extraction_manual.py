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
HOP_LENGTH = WINDOW_SIZE // 2  # Number of samples between successive frames

# File Paths
train_fpath = Path("data/processed/train_data.pkl").resolve()
test_fpath = Path("data/processed/test_data.pkl").resolve()
output_dir = Path("data/processed/spectrograms").resolve()

# Load data
train_df = pd.read_pickle(train_fpath)
test_df = pd.read_pickle(test_fpath)

# Define the target and feature columns
y_train = train_df["target"]
# Access with loc so it returns a DataFrame
X_train = train_df.loc[:, ["audio"]]

y_test = test_df["target"]
X_test = test_df.loc[:, ["audio"]]

# Find the shortest audio clip and crop all clips to that length
min_audio_length = min(
    min(X_train["audio"].apply(len)), min(X_test["audio"].apply(len))
)
X_train["audio"] = X_train["audio"].apply(lambda x: x[:min_audio_length])
X_test["audio"] = X_test["audio"].apply(lambda x: x[:min_audio_length])

genre_count = {}  # Dictionary to keep track of the count for each genre
min_max_image_size = [(10000, 10000), (0, 0)]
progress_counter = 0

sampling_rate = 22050  # Standard sampling rate for audio data
n_fft = WINDOW_SIZE  # Number of samples to use for each FFT

for X_df, y_df, subset in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
    genre_count = {}  # Dictionary to keep track of the count for each genre
    for idx, audio_clip in enumerate(X_df["audio"]):
        genre_label = y_df.iloc[idx]
        # Initialize or increment the genre count
        if genre_label not in genre_count:
            genre_count[genre_label] = 0
        else:
            genre_count[genre_label] += 1

        # Compute the Short-Time Fourier Transform (STFT) of the audio
        stft_matrix = librosa.stft(
            audio_clip, n_fft=n_fft, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH
        )

        # Compute all the features
        mfccs = librosa.feature.mfcc(
            y=audio_clip,
            sr=sampling_rate,
            n_mfcc=13,
            n_fft=n_fft,
            win_length=WINDOW_SIZE,
            hop_length=HOP_LENGTH,
        )
        melspectrogram = librosa.feature.melspectrogram(
            y=audio_clip, sr=sampling_rate, n_mels=128, fmax=8000
        )
        chroma_stft = librosa.feature.chroma_stft(
            y=audio_clip,
            sr=sampling_rate,
            n_chroma=12,
            n_fft=n_fft,
            win_length=WINDOW_SIZE,
            hop_length=HOP_LENGTH,
        )
        chroma_cqt = librosa.feature.chroma_cqt(
            y=audio_clip, sr=sampling_rate, n_chroma=12, hop_length=HOP_LENGTH
        )
        tempogram = librosa.feature.tempogram(
            y=audio_clip,
            sr=sampling_rate,
            win_length=WINDOW_SIZE,
            hop_length=HOP_LENGTH,
        )
        tonnetz = librosa.feature.tonnetz(y=audio_clip, sr=sampling_rate)
        spectrogram_db = librosa.amplitude_to_db(abs(stft_matrix), ref=np.max)
        power_spectrogram = librosa.power_to_db(np.abs(stft_matrix) ** 2, ref=np.max)
        phase_spectrogram = np.angle(stft_matrix)
        cqt_spectrogram = np.abs(librosa.cqt(y=audio_clip, sr=sampling_rate))

        # Create a list of all features and their names
        features = [
            (mfccs, "mfccs"),
            (melspectrogram, "melspectrogram"),
            (chroma_stft, "chroma_stft"),
            (chroma_cqt, "chroma_cqt"),
            (tempogram, "tempogram"),
            (tonnetz, "tonnetz"),
            (spectrogram_db, "spectrogram_db"),
            (power_spectrogram, "power_spectrogram"),
            (phase_spectrogram, "phase_spectrogram"),
            (cqt_spectrogram, "cqt_spectrogram"),
        ]

        for feature, feature_name in features:
            feature_name = feature_name + "_" + str(WINDOW_SIZE)

            # Normalize the spectrogram for image saving: Scale between 0 and 255
            img_array = np.flipud(
                feature
            )  # Flip the array to make the lower frequencies appear at the bottom of the image
            img_normalized = (
                255
                * (img_array - img_array.min())
                / (img_array.max() - img_array.min())
            )
            img_normalized = img_normalized.astype(
                np.uint8
            )  # Convert to unsigned byte format

            # Generate filename
            (output_dir / subset / feature_name / genre_label).mkdir(
                parents=True, exist_ok=True
            )
            filename = f"{genre_label}_{genre_count[genre_label]:06}_spectrogram.png"
            spectrogram_path = (
                output_dir / subset / feature_name / genre_label / filename
            )

            # Save the spectrogram image using PIL
            img = Image.fromarray(img_normalized)
            img.save(spectrogram_path)
            if img.size < min_max_image_size[0]:
                min_max_image_size[0] = img.size
            if img.size > min_max_image_size[1]:
                min_max_image_size[1] = img.size

        # Update the progress counter
        progress_counter += 1
        if progress_counter % 10 == 0:
            print(f"Processed {progress_counter} audio clips.")

if min_max_image_size:
    print(f"Largest Image Size: {min_max_image_size[1]}")
    print(f"Smallest Image Size: {min_max_image_size[0]}")
else:
    print("No images were processed.")
