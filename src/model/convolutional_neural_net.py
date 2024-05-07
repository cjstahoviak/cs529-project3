import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ConvolutionalNeuralNet(ImageClassificationBase):
    def __init__(
        self,
        input_channels,
        input_size,
        num_classes=10,
        kernel_size=3,
        stride=1,
        padding=1,
        final_dim=(4, 4),
    ):
        super().__init__()

        layers = []
        current_channels = input_channels
        current_width, current_height = input_size
        final_dim = (8, 8)

        # Dynamically add conv, relu, and max pooling layers
        while (
            current_width > final_dim[0] and current_height > final_dim[1]
        ):  # Arbitrary condition, can be adjusted
            out_channels = current_channels * 2  # Increase channels in each layer
            kernel_size = 3
            stride = 1
            padding = 1

            # Adding Conv2D layer
            layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size, stride, padding)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))  # Pool with window 2x2, stride 2

            # Update current parameters
            current_channels = out_channels
            current_width = (current_width + 2 * padding - kernel_size) // stride + 1
            current_height = (current_height + 2 * padding - kernel_size) // stride + 1
            current_width //= 2  # size reduction due to pooling
            current_height //= 2  # size reduction due to pooling

        # Flatten layer
        layers.append(nn.Flatten())

        # Fully connected layers
        fc_size = current_channels * current_width * current_height
        print("Fully connected layer size:", fc_size)

        layers.append(nn.Linear(fc_size, fc_size // 2))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(fc_size // 2, 512))
        layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(512, num_classes))

        # Combine all layers into a single sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, xb):
        return self.network(xb)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(
    epochs,
    lr,
    model,
    train_loader,
    val_loader,
    opt_func=torch.optim.SGD,
    lr_decay_factor=0.1,
    lr_decay_step_size=10,
    patience=5,
    min_delta=0.001,
):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor
    )
    best_loss = float("inf")
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Adjust the learning rate
        scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lr"] = scheduler.get_last_lr()[0]
        model.epoch_end(epoch, result)
        history.append(result)

        # Check early stopping criteria
        val_loss = result["val_loss"]
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            model.load_state_dict(best_model)
            break

    return history
