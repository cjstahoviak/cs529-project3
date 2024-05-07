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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

# Import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.model.convolutional_neural_net import ConvolutionalNeuralNet, evaluate, fit
from src.model.device import DeviceDataLoader, get_default_device, to_device


def main():
    # Final transform with normalization
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(0.3213, 0.1939),
        ]
    )

    # Load an image
    image_path = Path(
        "data/processed/spectrograms/train/spectrogram_db_1024/blues/blues_000000_spectrogram.png"
    ).resolve()
    original_image = Image.open(image_path)
    original_image.save("./figures/before_transform.jpg")

    # Apply the transformation
    transformed_image = transform(original_image)

    # Convert transformed tensor back to PIL Image for display/saving
    transformed_image_pil = transforms.ToPILImage()(transformed_image)
    transformed_image_pil.save("./figures/after_transform.jpg")

    # Create padding and labels
    padding = 10  # Padding around each image
    label_space = 30  # Space for labels

    # Create a new image with padding and label space
    total_width = original_image.width + transformed_image_pil.width + 3 * padding
    total_height = (
        max(original_image.height, transformed_image_pil.height) + padding + label_space
    )
    combined_image = Image.new(
        "RGB", (total_width, total_height), (255, 255, 255)
    )  # white background

    # Insert the images into the combined image
    combined_image.paste(original_image, (padding, padding + label_space))
    combined_image.paste(
        transformed_image_pil,
        (2 * padding + original_image.width, padding + label_space),
    )

    # Create draw object to add text
    draw = ImageDraw.Draw(combined_image)

    # Optionally load a font. Fonts might need to be adjusted depending on your system
    # font = ImageFont.truetype("arial.ttf", size=16)  # Uncomment and adjust path/font as necessary
    font = ImageFont.load_default()  # Default font

    # Add labels
    draw.text((padding, 5), "Original", font=font, fill="black")
    draw.text(
        (2 * padding + original_image.width, 5), "Transformed", font=font, fill="black"
    )

    # # Save the combined image
    # combined_image.save('combined_image_with_labels.jpg')

    # # Optionally, show the combined image
    # combined_image.show()


if __name__ == "__main__":
    main()
