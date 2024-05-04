from pathlib import Path
from typing import List

import lightning as L
import pandas as pd
import torch
import torch.utils.data as data
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


class LibrosaFeaturesDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_folder_path: Path,
        win_sizes: List[int] = [512, 1024, 2048, 4096, 8192],
        full_train: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()

        self.data_folder_path = data_folder_path
        self.win_sizes = win_sizes
        self.full_train = full_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = StandardScaler()
        self.oe = OrdinalEncoder()

    @staticmethod
    def _create_tensor_dataset(X, y=None):
        # Dynamic dataset creation to handle both train and unlabeled data
        if y is not None:
            return data.TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y.flatten(), dtype=torch.long),
            )

        return data.TensorDataset(torch.tensor(X, dtype=torch.float32))

    def setup(self, stage=None):
        X, y, X_kaggle = load_feature_data(self.data_folder_path, self.win_sizes)

        self._X = X
        self._y = y
        self._X_kaggle = X_kaggle
        self._num_features = X.shape[1]

        y = self.oe.fit_transform(y.reshape(-1, 1))  # Ordinal encoding
        self._num_classes = len(self.oe.categories_[0])

        # Data preparation
        if self.full_train:
            X_train = self.scaler.fit_transform(X)
            y_train = y
            self.test_dataloader = None
            self.val_dataloader = None
        else:
            # Split into train/test/val if not using full dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_val = self.scaler.transform(X_val)

            self.test_dataset = self._create_tensor_dataset(X_test, y_test)
            self.val_dataset = self._create_tensor_dataset(X_val, y_val)

        X_kaggle_scaled = self.scaler.transform(X_kaggle)

        self.train_dataset = self._create_tensor_dataset(X_train, y_train)
        self.predict_dataset = self._create_tensor_dataset(X_kaggle_scaled)

    def dataloader(self, ds: Dataset, **kwargs):
        """Taken and modified from pytorch_lightning LightningDataModule

        https://github.com/Lightning-AI/pytorch-lightning/blob/0f12271d7feeacb6fbe5d70d2ce057da4a04d8b4/src/lightning/pytorch/core/datamodule.py#L113
        """
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            **kwargs,
        )

    def train_dataloader(self):
        return self.dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self.dataloader(self.predict_dataset)

    def inverse_transform(self, y):
        """Inverse transform the ordinal encoded target"""
        # There's probably a more conventional way to do this
        return self.oe.inverse_transform(y)

    @property
    def predict_data_indices(self):
        return self._X_kaggle.index

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_features(self):
        return self._num_features


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
