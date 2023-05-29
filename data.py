import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class RacingDataset(Dataset):
    """
    Custom dataset class for racing data.
    """
    def __init__(self, csv_file, frames_dir, transform=None):
        """
        Initialize the RacingDataset.
        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            frames_dir (str): Directory containing the racing frames.
            transform (torchvision.transforms.Compose, optional):
                Optional transformations to be applied to the images. Defaults to None.
        """
        self.data = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the image (as a numpy array), in_race, speed, position, and lap.
        """
        row = self.data.iloc[idx]
        frame_file = row['frame_file']
        frame_path = os.path.join(self.frames_dir, frame_file)

        # Load image
        desired_size = (224, 224)

        # Open the image file
        with Image.open(frame_path) as img:
            # Convert the image to RGB and resize
            image = img.convert('RGB').resize(desired_size, Image.ANTIALIAS)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Convert the image to a numpy array
        image = np.array(image)

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0

        # Transpose the image so that channels come first
        image = np.transpose(image, (2, 0, 1))

        # Extract target variables
        in_race = row['in_race']
        speed = row['speed']
        position = row['position']
        lap = row['lap']
        gear = row['gear']

        # Handle missing values and apply specific transformations
        if np.isnan(position):
            position = 0
        if np.isnan(lap):
            lap = 0
        if np.isnan(speed):
            speed = 0
        if np.isnan(gear):
            gear = 0
        elif gear == 'N':
            gear = 1
        elif gear == 'R':
            gear = 2
        else:
            gear = int(gear) + 2

        # Normalize speed to the range [0, 1]
        speed /= 500.0

        return image, in_race, speed, position, lap, gear


def get_loaders(batch_size):
    """
    Create and return the data loaders for training and validation.
    Args:
        batch_size (int): Batch size for the data loaders.
    Returns:
        tuple: A tuple containing the training data loader and validation data loader.
    """
    # Path and parameters
    dataset_file = 'data/dataset.csv'
    frames_dir = ''
    train_val_split = 0.8

    # Load dataset
    dataset = RacingDataset(dataset_file, frames_dir)

    # Calculate split sizes
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define transformations for data augmentation (if desired)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transformations to the datasets
    train_dataset.transform = transform
    val_dataset.transform = transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Print dataset sizes
    print('Train dataset size:', len(train_dataset))
    print('Validation dataset size:', len(val_dataset))

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_loaders(16)
    for batch in train_loader:
        image, in_race, speed, position, lap, gear = batch
        print(image.shape)
        print(in_race)
        print(speed)
        print(position)
        print(lap)
        print(gear)
        break
        # Process the batch
