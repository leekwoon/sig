import numpy as np
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


class MaskedCIFARDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        masks: Optional[np.ndarray] = None,
        fill_value: str = 'mean',
    ):
        self.base_dataset = base_dataset
        self.masks = masks
        self.fill_value = fill_value
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[idx]).float()
            
            if self.fill_value == 'mean':
                preserved = image * mask
                sum_preserved = preserved.sum()
                count_preserved = mask.sum()
                fill = sum_preserved / (count_preserved + 1e-8)
            elif self.fill_value == 'zero':
                fill = 0.0
            else:
                fill = float(self.fill_value)
            
            image = image * mask + fill * (1 - mask)
        
        return image, label


def load_cifar10_datasets(
    dataset_path: str = './data',
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    mean: List[float] = [0.4914, 0.4822, 0.4465],
    std: List[float] = [0.2470, 0.2435, 0.2616],
    random_flip: bool = True,
    download: bool = True,
    val_only: bool = False,
) -> Tuple[Optional[DataLoader], DataLoader]:
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    test_dataset = CIFAR10(
        root=dataset_path,
        train=False,
        download=download,
        transform=test_transform,
    )
    
    if val_only:
        print(f"CIFAR-10 Dataset statistics (val only):")
        print(f"  Test images: {len(test_dataset)}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return None, test_loader
    
    if random_flip:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    train_dataset = CIFAR10(
        root=dataset_path,
        train=True,
        download=download,
        transform=train_transform,
    )
    
    print(f"CIFAR-10 Dataset statistics:")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Test images: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, test_loader


def load_cifar100_datasets(
    dataset_path: str = './data',
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    mean: List[float] = [0.5071, 0.4867, 0.4408],
    std: List[float] = [0.2675, 0.2565, 0.2761],
    random_flip: bool = True,
    download: bool = True,
    val_only: bool = False,
) -> Tuple[Optional[DataLoader], DataLoader]:
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    test_dataset = CIFAR100(
        root=dataset_path,
        train=False,
        download=download,
        transform=test_transform,
    )
    
    if val_only:
        print(f"CIFAR-100 Dataset statistics (val only):")
        print(f"  Test images: {len(test_dataset)}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return None, test_loader
    
    if random_flip:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    train_dataset = CIFAR100(
        root=dataset_path,
        train=True,
        download=download,
        transform=train_transform,
    )
    
    print(f"CIFAR-100 Dataset statistics:")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Test images: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, test_loader


def get_cifar_base_dataset(
    dataset_name: str,
    dataset_path: str,
    train: bool,
    image_size: int,
    mean: List[float],
    std: List[float],
    download: bool = True,
) -> Dataset:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    if dataset_name == 'cifar10':
        return CIFAR10(
            root=dataset_path,
            train=train,
            download=download,
            transform=transform,
        )
    elif dataset_name == 'cifar100':
        return CIFAR100(
            root=dataset_path,
            train=train,
            download=download,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_masked_dataloader(
    base_dataset: Dataset,
    masks: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    fill_value: str = 'mean',
) -> DataLoader:
    masked_dataset = MaskedCIFARDataset(
        base_dataset=base_dataset,
        masks=masks,
        fill_value=fill_value,
    )
    
    return DataLoader(
        masked_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
