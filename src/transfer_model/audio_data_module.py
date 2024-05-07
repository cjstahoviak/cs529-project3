"""A module for handling audio data for the transfer model
"""

from collections import defaultdict
from pathlib import Path
from typing import Union

import lightning as L
import torch
import torch.utils
import torch.utils.data
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder


def empty_str(x):
    return ""


def load_audio(path):
    """
    Loads an audio file from the given path and returns the waveform.
    """
    waveform, _ = torchaudio.load(path)
    return waveform


class MultiSampleDatasetFolder(DatasetFolder):
    """
    A custom class that extends the `DatasetFolder` class to handle multiple examples per sample.
    Assumes a transformation that returns a multiple examples per sample.

    This class overrides the `__getitem__` method to return a tuple of examples, target, and num_examples.
    """

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # This transform is assumed to create multiple examples per sample
        # e.g. multiple frames per a single audio file
        examples = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        num_examples = len(examples)
        targets = torch.tensor([target] * num_examples)

        return examples, targets, num_examples


def collate_fn(batch):
    """Collate function for combining multiple examples per sasample into a single tensor batch.
    e.g. multiple frames generated per audio sample.
    """

    # Unzip the batch
    examples, targets, num_examples = zip(*batch)

    # Stack the examples into a single tensor
    examples = torch.cat(examples, dim=0)
    targets = torch.cat(targets, dim=0)
    # targets = torch.tensor(targets, dtype=torch.float)
    return examples, targets, num_examples


class UnlabeledDatasetFolder(MultiSampleDatasetFolder):
    """
    A custom class that extends the `MultiStampleDatasetFolder` class to handle unknown classes.

    This class overrides the `find_classes` method to return a single class label "unknown"
    and a default class index of "unknown" for all images in the dataset.
    """

    def find_classes(self, directory: Union[str, Path]):

        class_to_idx = defaultdict(empty_str)
        class_to_idx[""] = 0
        return ([""], class_to_idx)


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_folder_path: Path,
        batch_size: int = 64,
        num_workers: int = 4,
        validation_size: float = 0.2,
        transform=None,
    ):
        """
        LightningDataModule for handling audio data in a PyTorch Lightning project.

        Args:
            data_folder_path (Path): The path to the folder containing the audio data.
            batch_size (int, optional): The batch size for data loading. Defaults to 64.
            num_workers (int, optional): The number of workers for data loading. Defaults to 4.
            validation_size (float, optional): The proportion of the dataset to use for validation. Defaults to 0.2.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
                Defaults to None.
        """

        super().__init__()
        self.data_folder_path = data_folder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.transform = transform

    def setup(self, stage=None):
        self.full_dataset = MultiSampleDatasetFolder(
            self.data_folder_path / "train",
            loader=load_audio,
            extensions=("au"),
            transform=self.transform,
        )

        # Split train set into train and validation
        targets = self.full_dataset.targets
        n_samples = len(self.full_dataset)

        train_idx, val_idx = train_test_split(
            range(n_samples),
            stratify=targets,
            test_size=self.validation_size,
            shuffle=True,
            random_state=4,
        )

        self.train_set = torch.utils.data.Subset(self.full_dataset, train_idx)
        self.val_set = torch.utils.data.Subset(self.full_dataset, val_idx)

        # Unknown kaggle dataset
        self.predict_set = UnlabeledDatasetFolder(
            self.data_folder_path / "test",
            loader=load_audio,
            extensions=("au"),
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
