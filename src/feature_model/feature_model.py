"""
Traditional Neural Network Model based on features extracted from the audio files using librosa library.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils
from feature_data_loader import LibrosaFeaturesDataModule
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score
from torch import nn


class FeatureNetwork(L.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()

        hidden_size = (input_size + num_classes) // 2

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear_relu_stack(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        accuracy = accuracy_score(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())
        self.log("test_accuracy", accuracy)

    def predict_step(self, batch, batch_idx):
        X = batch[0]
        y_pred = self(X)

        self.log
        return y_pred.argmax(dim=1).cpu().numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":

    # Constants
    FULL_DATASET = True
    DATA_FPATH = "../../data/preprocessed"

    # Setup MLFlow logger
    mlf_logger = MLFlowLogger(experiment_name="/experiment", tracking_uri="databricks")
    mlf_logger.experiment.log_param(
        key="FULL_DATASET", value=FULL_DATASET, run_id=mlf_logger.run_id
    )

    torch.set_float32_matmul_precision("medium")

    # Load the data
    data_folder = Path(DATA_FPATH)
    if not data_folder.is_absolute():
        data_folder = data_folder.resolve()

    train_loader = LibrosaFeaturesDataModule(
        data_folder, full_train=FULL_DATASET, batch_size=64, num_workers=4
    )

    # Manually prepare and setup the data
    # because we need information from the data loader
    train_loader.prepare_data()
    train_loader.setup()

    # Initialize the model
    model = FeatureNetwork(train_loader.num_features, train_loader.num_classes)

    # Train the model
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        log_every_n_steps=5,
        logger=mlf_logger,
        default_root_dir="checkpoints/",
    )

    # Train and test the model
    trainer.fit(model, datamodule=train_loader)

    # Test the model
    if not FULL_DATASET:
        trainer.test(model, datamodule=train_loader)

    # Predict on Kaggle test data
    y_pred = trainer.predict(model, datamodule=train_loader)

    y_pred = np.concatenate(y_pred)
    y_pred = train_loader.inverse_transform(y_pred.reshape(-1, 1))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    index = train_loader.predict_data_indices
    test_results = pd.DataFrame({"class": y_pred.flatten()}, index=index)
    test_results.index.name = "id"

    # Save kaggle test results
    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle_submission_fname = Path(tmpdir) / f"kaggle_submission_{timestamp}.csv"
        test_results.to_csv(kaggle_submission_fname)
        mlf_logger.experiment.log_artifact(
            local_path=kaggle_submission_fname, run_id=mlf_logger.run_id
        )
