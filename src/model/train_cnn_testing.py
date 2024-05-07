import sys
from multiprocessing import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

# Import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.model.convolutional_neural_net import ConvolutionalNeuralNet, evaluate, fit
from src.model.device import DeviceDataLoader, get_default_device, to_device


# Gets the mean and standard deviation of the dataset
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def main():

    # Data parameters
    random_seed = 42
    batch_size = 128
    num_workers = 0
    resize_dim = (256, 256)
    spectrogram_type = "spectrogram_db_1024"

    # Fit parameters
    opt_func = optim.Adam  # try other optimizers
    lr = 0.001
    num_epochs = 50
    lr_decay_factor = 0.0000
    lr_decay_step_size = 100
    patience = 5
    min_delta = 0.001

    # CNN parameters
    input_channels = 1
    kernel_size = 4
    stride = 1
    padding = 1
    final_dim = (4, 4)

    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    train_dir = Path("data/processed/spectrograms/train/" + spectrogram_type).resolve()

    # Define a transform without normalization
    pre_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=input_channels),
            transforms.Resize(resize_dim),  # Resize smaller edge to 128
            transforms.ToTensor(),
        ]
    )

    # Load data and split
    pre_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=pre_transform,
    )
    val_size = pre_dataset.__len__() // 5
    train_size = len(pre_dataset) - val_size
    pre_train_dataset, _ = random_split(pre_dataset, [train_size, val_size])

    # Create data loaders
    pre_train_dataloader = DataLoader(
        pre_train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_mean, train_std = get_mean_std(pre_train_dataloader)
    print("Mean:", train_mean)
    print("Std:", train_std)

    # Final transform with normalization
    final_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=input_channels),
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]
    )

    # Load normalized data and split
    dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=final_transform,
    )
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create final data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size * 2, num_workers=num_workers, pin_memory=True
    )

    # Define cnn
    cnn_model = ConvolutionalNeuralNet(
        input_channels=input_channels,
        input_size=resize_dim,
        num_classes=len(dataset.classes),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        final_dim=final_dim,
    )
    # print(cnn)

    # Test the model with a batch of data
    for images, labels in train_dataloader:
        print("images.shape:", images.shape)
        out = cnn_model(images)
        print("out.shape:", out.shape)
        print("out[0]:", out[0])
        break

    # Move model to the GPU if available
    device = get_default_device()
    train_dataloader = DeviceDataLoader(train_dataloader, device)
    val_dataloader = DeviceDataLoader(val_dataloader, device)
    to_device(cnn_model, device)
    print(device)

    # Instantiate the model on the GPU
    cnn_model = to_device(
        ConvolutionalNeuralNet(input_channels=input_channels, input_size=resize_dim),
        device,
    )

    # Evaluate the model on the validation set without training
    print(evaluate(cnn_model, val_dataloader))

    # Train the model
    history = fit(
        num_epochs,
        lr,
        cnn_model,
        train_dataloader,
        val_dataloader,
        opt_func,
        lr_decay_factor=lr_decay_factor,
        lr_decay_step_size=lr_decay_step_size,
        patience=patience,
        min_delta=min_delta,
    )

    # Save the model
    model_name = (
        "cnn_model_"
        + str(num_epochs)
        + "_epochs_"
        + str(lr)
        + "_lr_"
        + str(batch_size)
        + "_batch"
        + ".pth"
    )
    torch.save(cnn_model.state_dict(), "./trained_model/" + model_name)

    # Data preparation
    accuracies = [x["val_acc"] for x in history]
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    lr_hist = [x.get("lr") for x in history]

    # Define the parameters using the variables
    params = {
        "Batch Size": batch_size,
        "Resize Dimensions": resize_dim,
        "Optimizer Function": opt_func.__name__,  # Using the name of the optimizer function
        "Learning Rate": lr,
        "Number of Epochs": num_epochs,
        "Learning Rate Decay Factor": lr_decay_factor,
        "Learning Rate Decay Step Size": lr_decay_step_size,
        "Input Channels": input_channels,
        "Kernel Size": kernel_size,
        "Stride": stride,
        "Padding": padding,
        "Final Dimensions": final_dim,
    }

    # Create parameter string from dictionary
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))  # Two subplots in one column

    # Plot accuracy on the first subplot
    ax1.plot(accuracies, "-x")
    # ax1.plot(lr_hist, "-rx")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs. No. of epochs")

    # Plot loss on the second subplot
    ax2.plot(train_losses, "-bx")
    ax2.plot(val_losses, "-rx")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(["Training", "Validation"])
    ax2.set_title("Loss vs. No. of epochs")

    # Add a text box with parameters to the figure
    # fig.text(0.05, 0.05, param_text, fontsize=12, ha='left', verticalalignment='bottom')

    # Adjust layout and save the figure
    plt.tight_layout()
    figure_name = (
        "_batch-"
        + str(batch_size)
        + "_resize-"
        + str(resize_dim[0])
        + "_spectrogram-"
        + spectrogram_type
        + "_optfunc-"
        + opt_func.__name__
        + "_lr-"
        + str(lr)
        + "_epochs-"
        + str(num_epochs)
        + "_lr_decay_factor-"
        + str(lr_decay_factor)
        + "_lr_decay_step_size-"
        + str(lr_decay_step_size)
        + "_channels-"
        + str(input_channels)
        + "_kernelsize-"
        + str(kernel_size)
        + "_stride-"
        + str(stride)
        + "_padding-"
        + str(padding)
        + "_finaldim-"
        + str(final_dim[0])
    )
    plt.savefig("./figures/search/" + "lossAndAccuracy_" + figure_name + ".png")

    # Make confusion matrix
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Get predictions
    cnn_model.eval()
    all_preds = []
    all_labels = []
    for images, labels in val_dataloader:
        preds = cnn_model(images)
        all_preds.append(preds)
        all_labels.append(labels)

    # Stack all the predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Get the predicted class
    _, predicted = torch.max(all_preds, 1)

    # Get the confusion matrix
    cm = confusion_matrix(all_labels.cpu(), predicted.cpu())
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=dataset.classes,
        yticklabels=dataset.classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("./figures/search/" + "confusionMatrix_" + figure_name + ".png")


if __name__ == "__main__":
    main()
