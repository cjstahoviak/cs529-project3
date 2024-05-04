from pathlib import Path
from typing import List

import pandas as pd


def load_feature_data(
    data_folder_path: Path, win_sizes: List[int] = [512, 1024, 2048, 4096, 8192]
):
    train_dict = {}
    test_dict = {}

    y_train = None

    for win_size in win_sizes:
        train_df: pd.DataFrame = pd.read_pickle(
            data_folder_path / f"train_features_{win_size}.pkl"
        )
        X_test: pd.DataFrame = pd.read_pickle(
            data_folder_path / f"test_features_{win_size}.pkl"
        )

        X_train = train_df.drop(columns=["target"], level=0)

        train_dict[win_size] = X_train
        test_dict[win_size] = X_test

        if y_train is None:
            y_train = train_df["target"].values.ravel()

    X_train = pd.concat(
        train_dict.values(),
        axis=1,
        keys=win_sizes,
        names=["win_size", "feature", "stat"],
    )
    X_test = pd.concat(
        test_dict.values(),
        axis=1,
        keys=win_sizes,
        names=["win_size", "feature", "stat"],
    )

    return X_train, y_train, X_test
