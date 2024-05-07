"""Transfer learning model for audio classification using the VGGish model as a backbone.
Contains the VGGishTransferModel class which is a PyTorch Lightning module that uses the VGGish model as a backbone.
The model is trained on the given audio data and the predictions are saved to a Kaggle submission file when
run as a script.
"""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import lightning as L
import mlflow
import numpy as np
import pandas as pd
import torch
import torchaudio
from audio_data_module import AudioDataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch import nn
from torchaudio.prototype.pipelines import VGGISH


class VGGishTransferModel(L.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3):
        """Transfer learning model for audio classification using the VGGish model as a backbone.

        Args:
            num_classes (int): Number of target classes, used to determine the size of the output layer.
        """
        super().__init__()

        backbone = VGGISH.get_model()

        # Select just the features network from the VGGish model
        embeding_layers = list(backbone.children())[:-1]
        self.vggish = nn.Sequential(*embeding_layers)

        # Freeze the backbone
        self.vggish.eval()
        for param in self.vggish.parameters():
            param.requires_grad = False

        # Get the number of features output by VGGish
        num_features = 12_288  #  should be constant 128 x 96 = 12,288

        hidden_size = (num_features + num_classes) // 2

        # Use transfer learning to train a classifier on top of the backbone
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # Get the embeddings from the VGGish model
        embeddings = self.vggish(x)

        # Pass the embeddings through the classifier
        logits = self.linear_relu_stack(embeddings)

        return logits

    def calc_y_pred(self, logits: torch.Tensor, num_examples: List[int]):
        """
        Calculate the predicted labels for each original sample by taking the majority vote of the predictions for each.
        Each original sample may have multiple predictions due to multiple frames per audio file. The majority vote is
        taken as the predicted label for the original sample. e.g. if 3 frames are predicted as class 1 and 2 frames are
        predicted as class 2, the predicted label for the original sample is class 1.

        Args:
            logits (torch.Tensor): sum(num_examples) x num_classes tensor containing the predicted logits for each frame.
            num_examples (List[int]): A list containing the number of examples for each original sample.

        Returns:
            torch.Tensor: The predicted labels for each original sample.
        """
        # Group the predictions for each original sample
        y_pred_grouped = torch.split(logits, num_examples)

        # Determine the highest probability class for each prediction
        y_pred = [preds.argmax(dim=1) for preds in y_pred_grouped]

        # Take the majority vote for each original sample
        y_pred = [torch.mode(preds)[0] for preds in y_pred]

        # Convert the list of tensors to a single tensor
        y_pred = torch.tensor(y_pred, dtype=torch.float, device=logits.device)
        return y_pred

    def training_step(self, batch, batch_idx):
        X, y, num_examples = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=X.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, num_examples = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, batch_size=X.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        X, y, num_examples = batch
        logits = self(X)
        y_pred = self.calc_y_pred(logits, num_examples)
        accuracy = accuracy_score(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())
        self.log("test_accuracy", accuracy)

    def predict_step(self, batch, batch_idx):
        X, y, num_examples = batch
        y_pred = self(X)
        y_pred = self.calc_y_pred(y_pred, num_examples)
        return y_pred.cpu().numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def vgg_preprocessing(waveform):
    """
    Preprocesses the given waveform for input to VGGISH model.

    Args:
        waveform (torch.Tensor): The input waveform to be preprocessed.

    Returns:
        torch.Tensor: The preprocessed waveform.

    """
    input_proc = VGGISH.get_input_processor()
    input_sr = VGGISH.sample_rate
    waveform = torchaudio.functional.resample(waveform, 22050, input_sr)
    waveform = waveform.squeeze()
    waveform = input_proc(waveform)
    return waveform


def y_pred_to_labels(y_pred, dataset):
    """
    Convert the predicted labels to class labels using the given dataset.
    """
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    y_pred_labeled = np.array([idx_to_class[i] for i in y_pred])
    return y_pred_labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Model Run Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )

    hparams = parser.parse_args()

    # Constants
    DATA_FPATH = Path(__file__).parents[2].joinpath("data/raw")

    # Setup MLFlow logger
    mlf_logger = MLFlowLogger(experiment_name="/experiment", tracking_uri="databricks")

    # Set the default precision for matmul operations
    torch.set_float32_matmul_precision("medium")

    # Load the data
    data_folder = Path(DATA_FPATH)
    if not data_folder.is_absolute():
        data_folder = data_folder.resolve()

    input_proc = VGGISH.get_input_processor()

    train_loader = AudioDataModule(
        data_folder,
        batch_size=hparams.batch_size,
        num_workers=4,
        transform=vgg_preprocessing,
        validation_size=0.05,
    )

    # Initialize the model
    model = VGGishTransferModel(10, hparams.lr)

    print(model)

    # Train the model
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        log_every_n_steps=5,
        logger=mlf_logger,
        default_root_dir="checkpoints/",
        callbacks=[EarlyStopping(monitor="val_loss")],
        num_sanity_val_steps=0,
    )

    mlflow.log_params(hparams.__dict__)

    # Train the model
    trainer.fit(model, datamodule=train_loader)

    # make prediction on validation set
    y_pred = trainer.predict(model, dataloaders=train_loader.val_dataloader())
    y_pred = np.concatenate(y_pred)

    y_pred_labeled = y_pred_to_labels(y_pred, train_loader.full_dataset)

    # Get the original validation targets
    y_val = np.array(
        [
            train_loader.full_dataset.targets[i]
            for i in train_loader.val_dataloader().dataset.indices
        ]
    )
    y_val = y_pred_to_labels(y_val, train_loader.full_dataset)

    # Calculate validation accuracy and confusion matrix
    validation_accuracy = accuracy_score(y_val, y_pred_labeled)
    cm = confusion_matrix(y_val, y_pred_labeled, normalize="true")
    cmd = ConfusionMatrixDisplay(cm, display_labels=train_loader.full_dataset.classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    cmd.plot(ax=ax)
    plt.tight_layout()
    mlf_logger.experiment.log_metric(
        key="final_val_accuracy", value=validation_accuracy, run_id=mlf_logger.run_id
    )
    mlf_logger.experiment.log_figure(
        figure=cmd.figure_,
        artifact_file=f"confusion_matrix.png",
        run_id=mlf_logger.run_id,
    )

    # Predict on Kaggle test data
    y_pred = trainer.predict(model, datamodule=train_loader)

    y_pred = np.concatenate(y_pred)
    # Convert the predictions to class labels using the train_loader
    dataset = train_loader.full_dataset
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    y_pred = [idx_to_class[i] for i in y_pred]

    # Format the predictions for submission to Kaggle
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    index = [sample[0] for sample in train_loader.predict_set.samples]
    test_results = pd.DataFrame({"class": y_pred}, index=index)
    test_results.index.name = "id"

    # Save kaggle test results
    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle_submission_fname = Path(tmpdir) / f"kaggle_submission_{timestamp}.csv"
        test_results.to_csv(kaggle_submission_fname)
        mlf_logger.experiment.log_artifact(
            local_path=kaggle_submission_fname, run_id=mlf_logger.run_id
        )
