import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import lightning as L
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from feature_data_loader import load_feature_data
from lightning.pytorch.loggers import MLFlowLogger
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from feature_model import FeatureNetwork

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "preprocessed"
MLFLOW_EXPERIMENT = "feature_model"


def make_dataloader(ds, batch_size: int, num_workers: int = 4):
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, persistent_workers=True
    )


def run_cv(hparams):
    torch.set_float32_matmul_precision("medium")
    le = LabelEncoder()
    scaler = StandardScaler()
    X, y, X_kaggle = load_feature_data(DATA_PATH)
    y = le.fit_transform(y.ravel())  # Ordinal encoding

    kaggle_data = TensorDataset(torch.tensor(X_kaggle.to_numpy(), dtype=torch.float32))

    cv = StratifiedShuffleSplit(n_splits=hparams.folds, test_size=0.2, random_state=42)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f"/{MLFLOW_EXPERIMENT}")
    validation_accuracies = []
    with mlflow.start_run(run_name=f"{timestamp}") as parent_run:
        mlflow.log_params(hparams.__dict__)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y.flatten())):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True) as child_run:
                print(f"Running fold {fold}")
                mlflow.log_params(hparams.__dict__)
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                train_set = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train.flatten(), dtype=torch.long),
                )

                val_set = TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val.flatten(), dtype=torch.long),
                )

                model = FeatureNetwork(1932, 10, hparams.lr)
                mlf_logger = MLFlowLogger(
                    experiment_name=f"/{MLFLOW_EXPERIMENT}",
                    tracking_uri="databricks",
                    run_id=child_run.info.run_id,
                )

                trainer = L.Trainer(
                    max_epochs=hparams.epochs,
                    accelerator="gpu",
                    log_every_n_steps=5,
                    logger=mlf_logger,
                    default_root_dir="checkpoints/",
                    num_sanity_val_steps=0,
                )

                t_start = datetime.now()
                trainer.fit(
                    model,
                    train_dataloaders=make_dataloader(train_set, hparams.batch_size),
                    val_dataloaders=make_dataloader(val_set, hparams.batch_size),
                )
                t_end = datetime.now()
                mlflow.log_metric("training_time", (t_end - t_start).total_seconds())

                # Predict and log final validation results
                valid_y_pred = trainer.predict(
                    model, dataloaders=make_dataloader(val_set, hparams.batch_size)
                )
                # print(valid_y_pred)
                y_val = le.inverse_transform(y_val)
                valid_y_pred = le.inverse_transform(np.concatenate(valid_y_pred))
                valid_accuracy = accuracy_score(y_val, valid_y_pred)
                cm = confusion_matrix(y_val, valid_y_pred, normalize="true")
                cmd = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
                fig, ax = plt.subplots(figsize=(10, 10))
                cmd.plot(ax=ax)
                plt.tight_layout()
                mlflow.log_metric("final_val_accuracy", valid_accuracy)
                validation_accuracies.append(valid_accuracy)
                mlflow.log_figure(cmd.figure_, f"confusion_matrix.png")

                # Predict on Kaggle test data
                # kaggle_transformed = scaler.transform(X_kaggle)
                # kaggle_data = TensorDataset(torch.tensor(kaggle_transformed, dtype=torch.float32))

                # y_pred = trainer.predict(
                #     model,
                #     dataloaders=make_dataloader(kaggle_data, hparams.batch_size)
                # )

                # y_pred = np.concatenate(y_pred)
                # y_pred = le.inverse_transform(y_pred)

                # test_results = pd.DataFrame({"class": y_pred.flatten()}, index=X_kaggle.index)
                # test_results.index.name = "id"

                # # Save kaggle test results
                # parent_run_name = parent_run.info.run_id
                # with tempfile.TemporaryDirectory() as tmpdir:
                #     kaggle_submission_fname = Path(tmpdir) / f"kaggle_{parent_run_name}_fold_{fold}.csv"
                #     test_results.to_csv(kaggle_submission_fname)
                #     mlf_logger.experiment.log_artifact(
                #         local_path=kaggle_submission_fname, run_id=mlf_logger.run_id
                #     )

        mean_validation_accuracy = np.mean(validation_accuracies)
        mlflow.log_metric("mean_validation_accuracy", mean_validation_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fabric MNIST K-Fold Cross Validation Example"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="number of folds for k-fold cross validation",
    )
    hparams = parser.parse_args()

    run_cv(hparams)
