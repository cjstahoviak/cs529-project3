import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import lightning as L
import numpy as np
import pandas as pd
import torch
import torchaudio
from audio_data_module import AudioDataModule
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score
from torch import nn
from torchaudio.prototype.pipelines import VGGISH


class VGGishTransferModel(L.LightningModule):
    def __init__(self, num_classes):
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
            nn.Dropout(0.6),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_size, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Get the embeddings from the VGGish model
        embeddings = self.vggish(x)

        # Pass the embeddings through the classifier
        logits = self.linear_relu_stack(embeddings)

        return logits

    def calc_y_pred(self, logits: torch.Tensor, num_examples: List[int]):
        # Group the predictions for each original sample
        y_pred_grouped = torch.split(logits, num_examples)

        # Determine the highest probability class for each prediction
        y_pred = [preds.argmax(dim=1) for preds in y_pred_grouped]

        # Take the majority vote for each original sample
        y_pred = [torch.mode(preds)[0] for preds in y_pred]
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
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def vgg_preprocessing(waveform):
    input_proc = VGGISH.get_input_processor()
    input_sr = VGGISH.sample_rate
    waveform = torchaudio.functional.resample(waveform, 22050, input_sr)
    waveform = waveform.squeeze()
    waveform = input_proc(waveform)
    return waveform


if __name__ == "__main__":
    # Constants
    DATA_FPATH = "../../data/raw"

    # Setup MLFlow logger
    mlf_logger = MLFlowLogger(
        experiment_name="/experiment", tracking_uri="databricks", log_model=True
    )

    # Set the default precision for matmul operations
    torch.set_float32_matmul_precision("medium")

    # Load the data
    data_folder = Path(DATA_FPATH)
    if not data_folder.is_absolute():
        data_folder = data_folder.resolve()

    input_proc = VGGISH.get_input_processor()

    train_loader = AudioDataModule(
        data_folder, batch_size=16, num_workers=4, transform=vgg_preprocessing
    )

    # Initialize the model
    model = VGGishTransferModel(10)

    print(model)

    # Train the model
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        log_every_n_steps=5,
        logger=mlf_logger,
        default_root_dir="checkpoints/",
    )

    # Train the model
    trainer.fit(model, datamodule=train_loader)

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
