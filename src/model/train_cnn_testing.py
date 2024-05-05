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

    # Prepare parameters for data loading
    random_seed = 42
    torch.manual_seed(random_seed)
    batch_size = 32
    num_workers = 0

    # Define a transform without normalization
    pre_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),  # Resize smaller edge to 128
            transforms.ToTensor(),
        ]
    )

    # Load data and split
    pre_dataset = torchvision.datasets.ImageFolder(
        root="./data/processed/spectrograms/amplitude_1024/train",
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
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]
    )

    # Load normalized data and split
    dataset = torchvision.datasets.ImageFolder(
        root="./data/processed/spectrograms/amplitude_1024/train",
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
    cnn_model = ConvolutionalNeuralNet()
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
    cnn_model = to_device(ConvolutionalNeuralNet(), device)

    # Evaluate the model on the validation set without training
    print(evaluate(cnn_model, val_dataloader))

    # Train the model
    num_epochs = 10
    opt_func = optim.Adam
    lr = 0.001
    history = fit(num_epochs, lr, cnn_model, train_dataloader, val_dataloader, opt_func)

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

    # Plot the accuracy over time
    accuracies = [x["val_acc"] for x in history]
    plt.plot(accuracies, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. No. of epochs")
    plt.savefig("./figures/accuracy_vs_epochs.png")

    # Plot the loss over time
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    plt.clf()
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")
    plt.savefig("./figures/loss_vs_epochs.png")


if __name__ == "__main__":
    main()
