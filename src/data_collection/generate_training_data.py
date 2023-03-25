import os
import numpy as np
import torch
from data_collection.data_loader import MapData, MapDataLoader

def run(data_dir, batch_size):
    # Initialize data loader
    data_loader = MapDataLoader(data_dir, batch_size=batch_size)
    dataset = data_loader.prepare_data_loader()

    # Split dataset into train and validation sets
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Save train and validation sets
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for i, (image, mask) in enumerate(train_set):
        save_path = os.path.join(train_dir, f"image_{i}.pt")
        torch.save((image, mask), save_path)
    for i, (image, mask) in enumerate(val_set):
        save_path = os.path.join(val_dir, f"image_{i}.pt")
        torch.save((image, mask), save_path)

    print(f"Generated {len(train_set)} training samples and {len(val_set)} validation samples.")

if __name__ == '__main__':
    data_dir = "data/satellite"
    batch_size = 16
    run(data_dir, batch_size)