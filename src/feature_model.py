"""
Traditional Neural Network Model based on features extracted from the audio files using librosa library.
"""

from datetime import datetime
import tempfile
import pandas as pd
import torch
from torch import nn
from pathlib import Path
import numpy as np

import torch.utils
import torch.utils.data as data
from feature_data_loader import load_feature_data
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import lightning as L

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
        self.log("train_loss", loss, on_epoch=True)
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

def create_tensor_dataset(X, y=None):
    if y is not None:
        return data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y.flatten(), dtype=torch.long)
        )
    
    return data.TensorDataset(torch.tensor(X, dtype=torch.float32))

if __name__ == "__main__":

    from pytorch_lightning.loggers import MLFlowLogger
    mlf_logger = MLFlowLogger(
        experiment_name="/experiment",
        tracking_uri="databricks"
    )
    
    torch.set_float32_matmul_precision('medium')

    data_folder = Path("../data/preprocessed").resolve()
    X, y, X_kaggle = load_feature_data(data_folder)

    # Split the data
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tv_idx, test_idx = next(cv.split(X, y))
    X_tv, y_tv = X.iloc[tv_idx], y[tv_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    train_idx, val_idx = next(cv.split(X_tv, y_tv))
    X_train, y_train = X_tv.iloc[train_idx], y_tv[train_idx]
    X_val, y_val = X_tv.iloc[val_idx], y_tv[val_idx]

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    X_kaggle_scaled = scaler.transform(X_kaggle)

    # Encode the target variable
    oe = OrdinalEncoder()
    y_train = oe.fit_transform(y_train.reshape(-1, 1))
    y_test = oe.transform(y_test.reshape(-1, 1))
    y_val = oe.transform(y_val.reshape(-1, 1))
    n_classes = oe.categories_[0].shape[0]
    n_features = X_train.shape[1]

    train_loader = L.LightningDataModule.from_datasets(
        train_dataset=create_tensor_dataset(X_train, y_train), 
        test_dataset=create_tensor_dataset(X_test, y_test),
        val_dataset=create_tensor_dataset(X_val, y_val),
        predict_dataset=create_tensor_dataset(X_kaggle_scaled),
        batch_size=64,
        num_workers=4
    )

    # Initialize the model
    model = FeatureNetwork(n_features, n_classes)
    
    # Train the model
    trainer = L.Trainer(max_epochs=50, accelerator="gpu", log_every_n_steps=5, logger=mlf_logger)
    trainer.fit(model, datamodule=train_loader)

    # Test the model
    trainer.test(model, datamodule=train_loader)

    # Predict on Kaggle test data
    y_pred = trainer.predict(model, datamodule=train_loader)
    
    y_pred = np.concatenate(y_pred)
    y_pred = oe.inverse_transform(y_pred.reshape(-1, 1))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_results = pd.DataFrame({"class": y_pred.flatten()}, index=X_kaggle.index)
    test_results.index.name = "id"

    # Save kaggle test results
    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle_submission_fname = (
            Path(tmpdir) / f"kaggle_submission_{timestamp}.csv"
        )
        test_results.to_csv(kaggle_submission_fname)
        mlf_logger.experiment.log_artifact(local_path = kaggle_submission_fname, run_id = mlf_logger.run_id)


