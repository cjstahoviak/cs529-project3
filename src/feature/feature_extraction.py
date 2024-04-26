import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import parallel_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.feature.custom_transformers import ElementwiseSummaryStats, LibrosaTransformer

parallel_config(n_jobs=-2)

# Window sizes to generate features for
WIN_SIZES = [8192]

# File Paths
train_fpath = Path("data/processed/train_data.pkl").resolve()
test_fpath = Path("data/processed/test_data.pkl").resolve()
dest_dir = Path("data/processed/feature_extracted").resolve()

# Create the directory if it doesn't exist
dest_dir.parent.mkdir(parents=True, exist_ok=True)

# Load data
train_df = pd.read_pickle(train_fpath)
test_df = pd.read_pickle(test_fpath)

y_train = train_df["target"]
# Access with loc so it returns a DataFrame
X_train = train_df.loc[:, ["audio"]]
X_test = test_df.loc[:, ["audio"]]

# Librosa features to extract
librosa_features = [
    "mfcc",
    "chroma_stft",
    "chroma_cens",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_contrast",
    "zero_crossing_rate",
    "tonnetz",
    "tempogram_ratio",
]

pipe = Pipeline(
    [
        (
            "librosa_features",
            ColumnTransformer(
                [
                    (feature, LibrosaTransformer(feature=feature), ["audio"])
                    for feature in librosa_features
                ]
            ),
        ),
        ("stats", ElementwiseSummaryStats()),
    ]
)

pipe.set_output(transform="pandas")

for win_size in WIN_SIZES:
    print("Extracting features for window size:", win_size)

    params = {
        "librosa_features__mfcc__n_mfcc": 20,
        "librosa_features__mfcc__n_fft": win_size,
        "librosa_features__chroma_stft__n_fft": win_size,
        "librosa_features__spectral_centroid__n_fft": win_size,
        "librosa_features__spectral_bandwidth__n_fft": win_size,
        "librosa_features__spectral_rolloff__n_fft": win_size,
        "librosa_features__spectral_contrast__n_fft": win_size,
        "librosa_features__zero_crossing_rate__frame_length": win_size,
    }

    pipe.set_params(**params)

    print("Transforming train data...")
    time_start = datetime.now()
    train_df = pipe.fit_transform(X_train)
    time_end = datetime.now()
    print(f"Completed in: {time_end - time_start}")

    print("Transforming test data...")
    time_start = datetime.now()
    test_df = pipe.transform(X_test)
    time_end = datetime.now()
    print(f"Completed in: {time_end - time_start}")

    print("Saving features...")
    train_df["target"] = y_train
    pd.DataFrame(train_df).to_pickle(dest_dir / f"train_features_{win_size}.pkl")
    pd.DataFrame(test_df).to_pickle(dest_dir / f"test_features_{win_size}.pkl")
    print("")
