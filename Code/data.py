"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
from torchvision import transforms

TRAIN_AUGMENTATION = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])


class AugmentedSubset(Dataset):
    """Applies a transform to a Subset's images at fetch time, without
    touching the underlying (shared) train/val tensors."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label


def get_loaders(data, data_path, batch_size, val_split=0.1, seed=42, augment=False):
    d_path = Path(data_path) / f"{data}.pt" # Fix: Use data instead of {data}_data.pt
    data_dict = torch.load(d_path, weights_only=True)

    total_samples = data_dict['train_images'].shape[0]
    num_channels = data_dict['train_images'].shape[1]
    input_size = data_dict['train_images'].shape[-1]  # assumes square images (H == W)

    train_labels = data_dict['train_labels'].squeeze(1)
    test_labels = data_dict['test_labels'].squeeze(1)
    num_classes = int(torch.max(torch.cat([train_labels, test_labels])).item()) + 1

    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    full_train_dataset = TensorDataset(data_dict['train_images'], train_labels)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    if augment:
        train_dataset = AugmentedSubset(train_dataset, TRAIN_AUGMENTATION)

    test_dataset = TensorDataset(data_dict['test_images'], test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_channels, num_classes, input_size
