import numpy as np
from typing import Optional, List, Tuple, Dict, Callable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def generate_masks_from_attributions(
    attributions: np.ndarray,
    ratio: float,
    keep_most_salient: bool = True,
) -> np.ndarray:
    batch_size = attributions.shape[0]
    
    if len(attributions.shape) == 4:
        attr_2d = np.abs(attributions).sum(axis=1)
    else:
        attr_2d = np.abs(attributions)
    
    flat_attr = attr_2d.reshape(batch_size, -1)
    num_pixels = flat_attr.shape[1]
    k = int(num_pixels * ratio)
    
    masks = np.zeros((batch_size, num_pixels), dtype=np.float32)
    
    for i in range(batch_size):
        if keep_most_salient:
            indices = np.argsort(flat_attr[i])[-k:]
        else:
            indices = np.argsort(flat_attr[i])[:k]
        masks[i, indices] = 1.0
    
    H, W = attr_2d.shape[1], attr_2d.shape[2]
    masks = masks.reshape(batch_size, 1, H, W)
    masks = np.broadcast_to(masks, (batch_size, 3, H, W)).copy()
    
    return masks


def train_model_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_epochs: int = 10,
    lr: float = 0.01,
    lr_decay_epochs: List[int] = [5, 7],
    lr_decay_factor: float = 0.1,
    verbose: bool = True,
) -> Tuple[float, List[float]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=lr_decay_epochs, 
        gamma=lr_decay_factor
    )
    
    epoch_accuracies = []
    
    for epoch in range(num_epochs):
        train_loss = train_model_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        epoch_accuracies.append(test_acc)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        scheduler.step()
    
    final_acc = evaluate_model(model, test_loader, device)
    return final_acc, epoch_accuracies


def compute_roar_score(
    model_fn: Callable[[], nn.Module],
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_attributions: np.ndarray,
    ratios: Optional[List[float]] = None,
    device: str = 'cuda',
    batch_size: int = 128,
    num_workers: int = 4,
    num_epochs: int = 10,
    lr: float = 0.01,
    lr_decay_epochs: List[int] = [5, 7],
    fill_value: str = 'mean',
    verbose: bool = True,
) -> Dict:
    from cleanig.dataset.cifar_dataset import MaskedCIFARDataset
    
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    most_salient_accs = []
    least_salient_accs = []
    diff_roar_scores = []
    
    for ratio in ratios:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing ratio: {ratio:.0%}")
            print(f"{'='*60}")
        
        if verbose:
            print(f"\nTraining with MOST salient {ratio:.0%} pixels retained...")
        
        masks_most = generate_masks_from_attributions(
            train_attributions, ratio, keep_most_salient=True
        )
        
        masked_train_most = MaskedCIFARDataset(
            base_dataset=train_dataset,
            masks=masks_most,
            fill_value=fill_value,
        )
        
        train_loader_most = DataLoader(
            masked_train_most,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        model_most = model_fn()
        acc_most, _ = train_and_evaluate(
            model_most, train_loader_most, test_loader, device,
            num_epochs=num_epochs, lr=lr, lr_decay_epochs=lr_decay_epochs,
            verbose=verbose
        )
        most_salient_accs.append(acc_most)
        
        if verbose:
            print(f"\nTraining with LEAST salient {ratio:.0%} pixels retained...")
        
        masks_least = generate_masks_from_attributions(
            train_attributions, ratio, keep_most_salient=False
        )
        
        masked_train_least = MaskedCIFARDataset(
            base_dataset=train_dataset,
            masks=masks_least,
            fill_value=fill_value,
        )
        
        train_loader_least = DataLoader(
            masked_train_least,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        model_least = model_fn()
        acc_least, _ = train_and_evaluate(
            model_least, train_loader_least, test_loader, device,
            num_epochs=num_epochs, lr=lr, lr_decay_epochs=lr_decay_epochs,
            verbose=verbose
        )
        least_salient_accs.append(acc_least)
        
        diff = acc_most - acc_least
        diff_roar_scores.append(diff)
        
        if verbose:
            print(f"\nRatio {ratio:.0%}: Most={acc_most:.4f}, Least={acc_least:.4f}, Diff={diff:.4f}")
    
    auc = np.trapz(diff_roar_scores, ratios)
    
    return {
        'ratios': ratios,
        'most_salient_accs': most_salient_accs,
        'least_salient_accs': least_salient_accs,
        'diff_roar_scores': diff_roar_scores,
        'diff_roar_auc': auc,
    }


def compute_random_baseline_roar(
    model_fn: Callable[[], nn.Module],
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_train_samples: int,
    image_shape: Tuple[int, int, int],
    ratios: Optional[List[float]] = None,
    device: str = 'cuda',
    batch_size: int = 128,
    num_workers: int = 4,
    num_epochs: int = 10,
    lr: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    np.random.seed(seed)
    
    C, H, W = image_shape
    random_attributions = np.random.randn(num_train_samples, C, H, W).astype(np.float32)
    
    return compute_roar_score(
        model_fn=model_fn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_attributions=random_attributions,
        ratios=ratios,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        num_epochs=num_epochs,
        lr=lr,
        verbose=verbose,
    )
