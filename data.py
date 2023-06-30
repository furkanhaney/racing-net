import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class RacingDataset(Dataset):
    def __init__(self, csv_file, frames_dir, split, transform=None):
        data = pd.read_csv(csv_file)
        self.data = data[data['split'] == split]
        self.frames_dir = frames_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frame_path = os.path.join(self.frames_dir, row['frame_file'])

        with Image.open(frame_path) as img:
            image = img.convert('RGB').resize((224, 224), Image.LANCZOS)

        if self.transform:
            image = self.transform(image)

        image = np.array(image).astype(np.float32) / 255.0
        # image = np.transpose(image, (2, 0, 1))

        in_race, speed, position, lap, gear = row['in_race'], row['speed'], row['position'], row['lap'], row['gear']

        if pd.isna(position):
            position = 0
        if pd.isna(lap):
            lap = 0
        if pd.isna(speed):
            speed = 0
        if pd.isna(gear):
            gear = 0
        elif gear == 'N':
            gear = 1
        elif gear == 'R':
            gear = 2
        else:
            gear = int(gear) + 2

        speed /= 500.0

        return image, in_race, speed, position, lap, gear


def get_loaders(csv_file, frames_dir, batch_size, transform=None):
    imageTransform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) if transform is None else transform
    datasets = {
        split: RacingDataset(csv_file, frames_dir, split, imageTransform)
        for split in ['train', 'valid', 'test']
    }
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        for split, dataset in datasets.items()
    }
    return loaders


def speed_test(loader, loader_name):
    start_time = time.time()

    for _ in tqdm(loader, desc=loader_name):
        pass

    elapsed_time = time.time() - start_time
    print(f"{loader_name} Speed Test: Iterated through {len(loader.dataset)} samples in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    batch_size = 16
    loaders = get_loaders('data/dataset.csv', 'dataset/frames', batch_size)

    for split, loader in loaders.items():
        speed_test(loader, f"{split.capitalize()} Loader")
