import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
import zipfile


class MapData(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __getitem__(self, index):
        # Load image and mask
        image = cv2.imread(self.image_paths[index])
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

        # Preprocess image and mask
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask

    def __len__(self):
        return len(self.image_paths)


class MapDataLoader:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data_loader(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.fetch_data()
        self.process_data()
        dataset = self.create_dataset()
        return dataset

    def fetch_data(self):
        # Download the dataset zip file from a URL
        url = "https://example.com/dataset.zip"
        file_name = os.path.join(self.data_dir, "dataset.zip")
        urllib.request.urlretrieve(url, file_name)

        # Extract the dataset files from the zip file
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

    def process_data(self):
        # Load and preprocess the data
        for subset in ["train", "val", "test"]:
            image_dir = os.path.join(self.data_dir, subset, "images")
            mask_dir = os.path.join(self.data_dir, subset, "masks")
            image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
            mask_paths = [os.path.join(mask_dir, f.replace(".jpg", "_mask.png")) for f in os.listdir(image_dir)]
            for image_path, mask_path in zip(image_paths, mask_paths):
                # Preprocess image and mask
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image.transpose(2, 0, 1))
                mask = torch.from_numpy(mask.astype(np.int64))

                # Save the preprocessed image and mask
                save_dir = os.path.join(self.data_dir, "processed", subset)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".pt"))
                torch.save((image, mask), save_path)