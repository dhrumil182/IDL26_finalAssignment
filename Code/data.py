"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split

def get_loaders(data, data_path, batch_size, val_split=0.1, seed=42):
    d_path = Path(data_path) / f"{data}.pt" # Fix: Use data instead of {data}_data.pt
    data_dict = torch.load(d_path, weights_only=True)

    total_samples = data_dict['train_images'].shape[0]

    train_labels = data_dict['train_labels']

    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    full_train_dataset = TensorDataset(data_dict['train_images'], train_labels)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
