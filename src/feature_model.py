"""
Traditional Neural Network Model based on features extracted from the audio files using librosa library.
"""

import os
import torch
from torch import nn
from pathlib import Path
from feature_data_loader import load_feature_data
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class FeatureNetwork(nn.Module):
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

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    data_folder = Path("../data/preprocessed").resolve()
    X, y, X_kaggle = load_feature_data(data_folder)

    # Split the data
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(cv.split(X, y))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode the target variable
    oe = OrdinalEncoder()
    y_train = oe.fit_transform(y_train.reshape(-1, 1))
    n_classes = oe.categories_[0].shape[0]
    n_features = X_train.shape[1]

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.flatten(), dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Initialize the model
    model = FeatureNetwork(n_features, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    NUM_EPOCHS = 1_000
    for t in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if t % 100 == 0:
            print(f"Epoch {t}, Loss: {loss.item()}")

    model.eval()

    # Predict on the test data
    with torch.no_grad():
        logits = model(X_test)
    
        y_test_pred = torch.argmax(logits, dim=1)
        y_test_pred = oe.inverse_transform(y_test_pred.cpu().numpy().reshape(-1, 1))

    score = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {score}")


