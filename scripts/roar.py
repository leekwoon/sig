#!/usr/bin/env python
"""
ROAR (RemOve-And-Retrain) Evaluation

Based on: "Beyond Single Path Integrated Gradients for Reliable Input Attribution 
          via Randomized Path Sampling"

Methodology:
1. Remove top k% most important pixels from training images
2. Replace with average pixel value of remaining pixels
3. Retrain model from scratch on modified training set
4. Test on UNMODIFIED test set
5. Lower accuracy = better attribution (correctly identifies important features)
"""
import os
import json
import sys
import hydra
import argparse
import numpy as np
from functools import partial
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cleanig.utils import set_seed, preprocess
from cleanig.explainer import (
    IGExplainer,
    AGIExplainer,
    BIGExplainer,
    GIGExplainer,
    EIGExplainer,
    MIGExplainer,
    SpectralIGExplainer,
    GradInputExplainer,
    IG2Explainer,
    SAMPExplainer,
)
from cleanig.vae_wrapper import create_vae
from cleanig.dataset.cifar_dataset import get_cifar_base_dataset, MaskedCIFARDataset
from cleanig.metric.roar import generate_masks_from_attributions


ROAR_DEFAULTS = {
    'base_epochs': 200,
    'base_lr': 0.1,
    'roar_epochs': 100,
    'roar_lr': 3e-4,
    'ratios': [0.1, 0.3, 0.5, 0.7, 0.9],
    'fill_value': 'mean',
    'use_cache': True,
    'cache_dir': 'results/roar_cache',
    'verbose': True,
    'cifar_mean': [0.4914, 0.4822, 0.4465],
    'cifar_std': [0.2023, 0.1994, 0.2010],
}


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


def train_base_classifier(model, train_loader, test_loader, device, num_epochs=200, lr=0.1, save_path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'acc': f'{100.*correct/total:.1f}%'})

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'accuracy': best_acc,
                }, save_path)

        scheduler.step()

    print(f'Best accuracy: {best_acc:.2f}%')
    return best_acc


def train_roar_model(model, train_loader, test_loader, device, num_epochs=100, lr=3e-4, verbose=True):
    """Train model using Adam optimizer as per the paper."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, disable=not verbose)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'acc': f'{100.*correct/total:.1f}%'})

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        history.append(test_acc)

        if (epoch + 1) % 20 == 0 and verbose:
            print(f'Epoch {epoch+1}: Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc

    return best_acc, history


def create_explainer(args, model, preprocess_fn, undo_preprocess_fn, num_classes, mean, std):
    if args.explainer_name == 'random':
        return None

    if args.explainer_name == 'ig':
        return IGExplainer(
            model=model, baseline_method=args.baseline_method, num_steps=args.num_steps,
            device=args.device, exp_obj=args.exp_obj, preprocess_fn=preprocess_fn
        )
    elif args.explainer_name == 'gig':
        return GIGExplainer(
            model=model, baseline_method=args.baseline_method, num_steps=args.num_steps,
            device=args.device, exp_obj=args.exp_obj, preprocess_fn=preprocess_fn,
            fraction=args.fraction
        )
    elif args.explainer_name == 'big':
        return BIGExplainer(
            model=model, num_steps=args.num_steps, device=args.device,
            exp_obj=args.exp_obj, preprocess_fn=preprocess_fn,
            undo_preprocess_fn=undo_preprocess_fn,
            max_sigma=args.max_sigma, sqrt=args.sqrt
        )
    elif args.explainer_name == 'agi':
        return AGIExplainer(
            model=model, device=args.device, exp_obj=args.exp_obj, preprocess_fn=preprocess_fn,
            num_classes=num_classes, num_neg_cls=args.num_neg_cls,
            step_size=args.step_size, max_iter=args.max_iter,
            mean=mean, std=std
        )
    elif args.explainer_name == 'eig':
        vae = create_vae(args.vae_type, preprocess_fn, undo_preprocess_fn, args.device)
        return EIGExplainer(
            model=model, vae=vae, baseline_method=args.baseline_method, num_steps=args.num_steps,
            device=args.device, exp_obj=args.exp_obj, preprocess_fn=preprocess_fn
        )
    elif args.explainer_name == 'mig':
        vae = create_vae(args.vae_type, preprocess_fn, undo_preprocess_fn, args.device)
        return MIGExplainer(
            model=model, vae=vae, baseline_method=args.baseline_method, num_steps=args.num_steps,
            device=args.device, exp_obj=args.exp_obj, preprocess_fn=preprocess_fn,
            alpha=args.alpha, max_iterations=args.max_iterations,
            epsilon=args.epsilon
        )
    elif args.explainer_name == 'spectral_ig':
        return SpectralIGExplainer(
            model=model, baseline_method=args.baseline_method, num_steps=args.num_steps,
            device=args.device, exp_obj=args.exp_obj, preprocess_fn=preprocess_fn,
            overlap=args.overlap
        )
    elif args.explainer_name == 'grad_input':
        return GradInputExplainer(
            model=model, device=args.device, exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn, baseline_method=args.baseline_method
        )
    elif args.explainer_name == 'ig2':
        return IG2Explainer(
            model=model, device=args.device, exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn, steps=args.steps,
            step_size=args.step_size,
            reference_mode=args.reference_mode
        )
    elif args.explainer_name == 'samp':
        if args.direction == 'both':
            line_types = ['deletion', 'insertion']
        else:
            line_types = [args.direction]
        return SAMPExplainer(
            model=model, device=args.device, exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn, step=args.step_size, n_frag=args.n_frag,
            klen=args.klen, ksig=args.ksig, momen=args.momentum,
            line_types=line_types, reduction=args.reduction,
            insertion_baseline=args.insertion_baseline
        )
    else:
        raise ValueError(f"Unknown explainer: {args.explainer_name}")


def generate_attributions(model, dataset, explainer, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_attrs = []
    for images, labels in tqdm(loader, desc='Generating attributions'):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred_labels = model(images).argmax(dim=1)

        attrs = explainer.get_attributions(images, labels=pred_labels)
        all_attrs.append(attrs.cpu().numpy())

    return np.concatenate(all_attrs, axis=0)


def compute_roar_score(
    model_fn,
    train_dataset,
    test_dataset,
    train_attributions,
    ratios,
    device,
    batch_size=128,
    num_workers=4,
    num_epochs=100,
    lr=3e-4,
    fill_value='mean',
    verbose=True,
):
    """
    Compute ROAR scores.
    
    For each ratio, remove top k% most important pixels from training images,
    retrain from scratch, and evaluate on UNMODIFIED test set.
    
    Lower accuracy = better attribution method.
    """
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    roar_accs = []

    for ratio in ratios:
        if verbose:
            print(f"\n{'='*60}")
            print(f"ROAR: Removing top {ratio:.0%} most important pixels")
            print(f"{'='*60}")

        masks = generate_masks_from_attributions(train_attributions, 1.0 - ratio, keep_most_salient=False)
        masked_train = MaskedCIFARDataset(base_dataset=train_dataset, masks=masks, fill_value=fill_value)
        train_loader = DataLoader(
            masked_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )

        model = model_fn()
        acc, _ = train_roar_model(
            model, train_loader, test_loader, device,
            num_epochs=num_epochs, lr=lr, verbose=verbose
        )
        roar_accs.append(acc)

        if verbose:
            print(f"Ratio {ratio:.0%}: Test Accuracy = {acc:.2f}%")

    auc = np.trapz(roar_accs, ratios)

    return {
        'ratios': ratios,
        'roar_accs': roar_accs,
        'roar_auc': auc,
    }


def plot_results(results, save_path, dataset_name, explainer_name):
    ratios = results['ratios']

    fig, ax = plt.subplots(figsize=(8, 6))

    auc = results['roar_auc']
    ax.plot(ratios, results['roar_accs'], '-o', color='blue', markersize=6, linewidth=2)
    ax.set_xlabel('Removal Ratio (Top k% pixels removed)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'ROAR - {dataset_name}/{explainer_name}\n(AUC={auc:.2f}, Lower is better)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    for i, (r, acc) in enumerate(zip(ratios, results['roar_accs'])):
        ax.annotate(f'{acc:.1f}%', (r, acc), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_roar_args():
    parser = argparse.ArgumentParser(description='ROAR arguments (parsed before Hydra)')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--base_epochs', type=int, default=ROAR_DEFAULTS['base_epochs'])
    parser.add_argument('--base_lr', type=float, default=ROAR_DEFAULTS['base_lr'])
    parser.add_argument('--roar_epochs', type=int, default=ROAR_DEFAULTS['roar_epochs'])
    parser.add_argument('--roar_lr', type=float, default=ROAR_DEFAULTS['roar_lr'])
    parser.add_argument('--ratios', type=float, nargs='+', default=ROAR_DEFAULTS['ratios'])
    parser.add_argument('--fill_value', type=str, default=ROAR_DEFAULTS['fill_value'])
    parser.add_argument('--use_cache', action='store_true', default=ROAR_DEFAULTS['use_cache'])
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--cache_dir', type=str, default=ROAR_DEFAULTS['cache_dir'])
    parser.add_argument('--verbose', action='store_true', default=ROAR_DEFAULTS['verbose'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    args, remaining = parser.parse_known_args()
    if args.no_cache:
        args.use_cache = False
    return args, remaining


roar_args, remaining_argv = parse_roar_args()


@hydra.main(config_path="../configs", config_name="gig", version_base=None)
def pipeline(args):
    set_seed(args.seed)

    dataset_name = roar_args.dataset_name
    explainer_name = args.explainer_name
    num_classes = 10 if dataset_name == 'cifar10' else 100

    if roar_args.save_dir is None:
        save_dir = f"results/benchmark_roar/{dataset_name}/{explainer_name}"
    else:
        save_dir = roar_args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    cache_dir = os.path.join(roar_args.cache_dir, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    mean = ROAR_DEFAULTS['cifar_mean']
    std = ROAR_DEFAULTS['cifar_std']

    train_dataset = get_cifar_base_dataset(
        dataset_name, roar_args.dataset_path, True, 32, mean, std
    )
    test_dataset = get_cifar_base_dataset(
        dataset_name, roar_args.dataset_path, False, 32, mean, std
    )

    print(f"\n{'='*60}")
    print(f"ROAR Benchmark: {dataset_name.upper()} / {explainer_name.upper()}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Ratios: {roar_args.ratios}")
    print(f"Epochs: {roar_args.roar_epochs}, LR: {roar_args.roar_lr}")
    print(f"{'='*60}\n")

    base_model_path = os.path.join(cache_dir, 'base_model.pt')

    if not os.path.exists(base_model_path):
        print("Training base classifier...")
        train_loader = DataLoader(
            train_dataset, batch_size=roar_args.batch_size, shuffle=True,
            num_workers=roar_args.num_workers, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=roar_args.batch_size, shuffle=False,
            num_workers=roar_args.num_workers, pin_memory=True
        )

        model = PreActResNet18(num_classes=num_classes)
        train_base_classifier(
            model, train_loader, test_loader, args.device,
            num_epochs=roar_args.base_epochs, lr=roar_args.base_lr, save_path=base_model_path
        )

    print(f"\nLoading base model from {base_model_path}")
    model = PreActResNet18(num_classes=num_classes).to(args.device)
    ckpt = torch.load(base_model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    base_accuracy = ckpt['accuracy']
    print(f"Base model accuracy: {base_accuracy:.2f}%")

    preprocess_fn = partial(preprocess, mean=mean, std=std)

    def undo_preprocess_fn(x):
        m = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
        s = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
        return x * s + m

    attr_cache_path = os.path.join(cache_dir, f'attrs_{explainer_name}.npy')

    if roar_args.use_cache and os.path.exists(attr_cache_path):
        print(f"Loading cached attributions from {attr_cache_path}")
        train_attributions = np.load(attr_cache_path)
    else:
        print(f"Generating attributions for {explainer_name}...")

        if explainer_name == 'random':
            np.random.seed(args.seed)
            train_attributions = np.random.randn(len(train_dataset), 3, 32, 32).astype(np.float32)
        else:
            explainer = create_explainer(args, model, preprocess_fn, undo_preprocess_fn, num_classes, mean, std)
            
            if explainer_name == 'ig2':
                ref_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
                reference_images = next(iter(ref_loader))[0].to(args.device)
                explainer.set_reference_bank(reference_images)
            
            train_attributions = generate_attributions(
                model, train_dataset, explainer, args.device,
                batch_size=64 if explainer_name in ['ig', 'gig', 'eig', 'mig', 'spectral_ig'] else 128
            )

        if roar_args.use_cache:
            np.save(attr_cache_path, train_attributions)
            print(f"Attributions saved to {attr_cache_path}")

    print(f"Attributions shape: {train_attributions.shape}")

    print("\nRunning ROAR evaluation...")

    def model_fn():
        return PreActResNet18(num_classes=num_classes)

    results = compute_roar_score(
        model_fn=model_fn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_attributions=train_attributions,
        ratios=list(roar_args.ratios),
        device=args.device,
        batch_size=roar_args.batch_size,
        num_workers=roar_args.num_workers,
        num_epochs=roar_args.roar_epochs,
        lr=roar_args.roar_lr,
        fill_value=roar_args.fill_value,
        verbose=roar_args.verbose,
    )

    print("\n" + "="*60)
    print("FINAL RESULTS (Lower accuracy = Better attribution)")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Explainer: {explainer_name}")
    print(f"Base Model Accuracy: {base_accuracy:.2f}%")
    print(f"ROAR AUC: {results['roar_auc']:.2f} (lower is better)")
    print("\nPer-ratio results:")
    print(f"{'Ratio':<15} {'Test Accuracy':<15}")
    print("-" * 30)
    for i, ratio in enumerate(results['ratios']):
        print(f"{ratio:<15.0%} {results['roar_accs'][i]:<15.2f}%")

    output = {
        'dataset': dataset_name,
        'explainer': explainer_name,
        'num_samples': len(train_dataset),
        'base_model_accuracy': base_accuracy,
        'roar_auc': results['roar_auc'],
        'ratios': results['ratios'],
        'roar_accs': results['roar_accs'],
        'explainer_config': OmegaConf.to_container(args, resolve=True),
        'roar_config': {
            'base_epochs': roar_args.base_epochs,
            'base_lr': roar_args.base_lr,
            'roar_epochs': roar_args.roar_epochs,
            'roar_lr': roar_args.roar_lr,
            'ratios': roar_args.ratios,
            'fill_value': roar_args.fill_value,
        },
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(save_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_path = os.path.join(save_dir, 'roar.png')
    plot_results(results, plot_path, dataset_name, explainer_name)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0]] + remaining_argv
    pipeline()
