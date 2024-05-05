import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

WINDOW_SIZE = 1024  # Number of samples in each frame
HOP_LENGTH = int(WINDOW_SIZE / 2)  # Number of samples between successive frames

sampling_rate = 22050  # Standard sampling rate for audio data
n_fft = WINDOW_SIZE  # Number of samples to use for each FFT

# Load an audio file
audio_path = Path("data/raw/train/blues/blues.00000.au").resolve()
y, sr = librosa.load(audio_path, sr=None)

# Print sampling rate and audio duration
print(f"Sampling rate: {sr}")
print(f"Audio duration: {len(y) / sr} seconds")

# Compute the Short-Time Fourier Transform (STFT) of the audio
stft_matrix = librosa.stft(
    y, n_fft=n_fft, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH
)

# Compute all the features
mfccs = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=13, n_fft=n_fft, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH
)
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
spectrogram_db = librosa.amplitude_to_db(abs(stft_matrix), ref=np.max)
power_spectrogram = librosa.power_to_db(
    np.abs(stft_matrix) ** 2, ref=np.max
)  # np.abs(stft_matrix)**2
phase_spectrogram = np.angle(
    librosa.stft(y=y, n_fft=n_fft, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH)
)
cqt_spectrogram = np.abs(librosa.cqt(y=y, sr=sr))
chroma = librosa.feature.chroma_stft(
    y=y, sr=sr, n_chroma=12, n_fft=n_fft, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH
)
tempogram = librosa.feature.tempogram(
    y=y, sr=sr, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH
)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

# Displaying the MFCCs:
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(melspectrogram, x_axis='time', sr=sr)
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()

# List of features and their titles
features = [
    (mfccs, "MFCCs", None),
    (melspectrogram, "Mel Spectrogram", "mel"),
    (spectrogram_db, "Spectrogram (dB)", "linear"),
    (power_spectrogram, "Power Spectrogram", "log"),
    (phase_spectrogram, "Phase Spectrogram", "linear"),
    (cqt_spectrogram, "CQT Spectrogram", "cqt_hz"),
    (chroma, "Chroma Features", "chroma"),
    (tempogram, "Tempogram", "tempo"),
    (tonnetz, "Tonnetz", None),
]

# Plotting setup
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

# Loop to plot each feature with the appropriate axis labels
for ax, (feature, title, axis) in zip(axes, features):
    if axis:
        img = librosa.display.specshow(
            feature, ax=ax, x_axis="time", y_axis=axis, sr=sr, hop_length=HOP_LENGTH
        )
    else:
        img = librosa.display.specshow(
            feature, ax=ax, x_axis="time", sr=sr, hop_length=HOP_LENGTH
        )
    ax.set_title(title, fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.tight_layout(pad=3.0)
plt.savefig("./figures/spectrogram_features_sample.png")  # Save the plot as an image
