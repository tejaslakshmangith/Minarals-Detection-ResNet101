"""Training script for rock/sediment/mineral image classification (ResNet-101)."""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Allow imports from sibling package when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from models.resnet101 import get_model, SmartMineResNet18


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def build_transforms(use_randaugment: bool = False):
    """Return (train_transform, val_transform) as a tuple.

    Args:
        use_randaugment: When ``True``, append a ``RandAugment`` step to the
            training pipeline for automatic augmentation policy selection.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_ops = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]
    if use_randaugment:
        train_ops.append(transforms.RandAugment())
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
    train_transform = transforms.Compose(train_ops)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_transform, val_transform


def make_weighted_sampler(dataset: torch.utils.data.Dataset) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that balances classes in the dataset.

    This is useful when one class dominates the training set (e.g., Baryte)."""
    # Extract the target labels from the dataset (supports ImageFolder and Subset)
    if isinstance(dataset, torch.utils.data.Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = list(dataset.targets)

    class_counts = Counter(targets)
    weights = [1.0 / class_counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Mixup helper
# ---------------------------------------------------------------------------

def mixup_data(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    """Apply Mixup to a batch and return mixed inputs and both label sets.

    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lam) where *lam* is the
        mixing coefficient drawn from Beta(alpha, alpha).
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Compute the Mixup loss as a convex combination of two CE losses."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    training: bool,
    mixup_alpha: float = 0.0,
    clip_grad_norm: float = 0.0,
):
    """Run one full epoch and return (avg_loss, accuracy).

    Args:
        optimizer: Required when *training* is ``True``; pass ``None`` for
            validation/evaluation passes where no parameter updates occur.
        mixup_alpha: Alpha parameter for Mixup augmentation. Set to ``0`` to
            disable Mixup (default).
        clip_grad_norm: Max norm for gradient clipping. Set to ``0`` to
            disable clipping (default).
    """
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    use_mixup = training and mixup_alpha > 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()
                if use_mixup:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
                with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                    outputs = model(images)
                    if use_mixup:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
            else:
                with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            # For Mixup batches use the primary label (labels_a) for accuracy tracking
            primary_labels = labels_a if use_mixup else labels
            correct += predicted.eq(primary_labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path: str) -> None:
    """Save a two-panel loss + accuracy training curve figure."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, val_accs, label="Val Accuracy")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SmartMine ResNet-101 model")
    parser.add_argument("--data_dir", type=str, default="ai-model/dataset_balanced",
                        help="Root dataset directory (must contain train/ and val/ sub-dirs)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Number of classes (0 to infer from dataset)",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="If >0, limit the number of training samples (useful for quick experiments)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="If >0, limit the number of validation samples (useful for quick experiments)",
    )
    parser.add_argument("--save_path", type=str,
                        default="ai-model/models/resnet101_mineral.pth")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Use a weighted sampler to balance class frequencies during training.",
    )
    parser.add_argument("--fast", action="store_true", help="Use ResNet18 (5x faster)")
    parser.add_argument(
        "--unfreeze_layer1",
        action="store_true",
        help="Unfreeze ResNet-101 layer1 for finer mineral-specific feature tuning.",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.2,
        help="Alpha for Mixup augmentation (0 to disable, default: 0.2).",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping (0 to disable, default: 1.0).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of linear warmup epochs before cosine annealing (default: 5).",
    )
    parser.add_argument(
        "--randaugment",
        action="store_true",
        help="Append RandAugment to the training augmentation pipeline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ResNet-101 on CPU is very memory-heavy; cap batch size to avoid OOM.
    # (User can still override by using --fast or a smaller --batch_size.)
    if device == "cpu" and (not args.fast) and args.batch_size > 4:
        print(
            f"WARNING: ResNet101 on CPU may OOM with batch_size={args.batch_size}. "
            "Auto-setting --batch_size to 4. (Use --fast for quicker training.)"
        )
        args.batch_size = 4

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_transform, val_transform = build_transforms(use_randaugment=args.randaugment)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    raw_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    raw_val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    class_names = raw_train_dataset.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    # Optional subset for quick iteration/debugging
    train_dataset = raw_train_dataset
    val_dataset = raw_val_dataset
    if args.max_train_samples > 0 and args.max_train_samples < len(raw_train_dataset):
        train_dataset = torch.utils.data.Subset(raw_train_dataset, list(range(args.max_train_samples)))
    if args.max_val_samples > 0 and args.max_val_samples < len(raw_val_dataset):
        val_dataset = torch.utils.data.Subset(raw_val_dataset, list(range(args.max_val_samples)))

    if args.num_classes <= 0:
        args.num_classes = len(class_names)
    elif args.num_classes != len(class_names):
        print(
            f"WARNING: --num_classes ({args.num_classes}) does not match dataset "
            f"classes ({len(class_names)}). Using {len(class_names)} instead."
        )
        args.num_classes = len(class_names)

    if args.balance:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
        print("Using weighted sampler to balance class frequencies during training.")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    if args.fast:
        model = SmartMineResNet18(num_classes=args.num_classes).to(device)
        args.save_path = args.save_path.replace("resnet101", "resnet18")
    else:
        model = get_model(
            num_classes=args.num_classes,
            device=device,
            unfreeze_layer1=args.unfreeze_layer1,
        )
    print(f"Model: {type(model).__name__}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Warmup for the first `warmup_epochs` epochs, then cosine annealing
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    cosine_epochs = max(args.epochs - warmup_epochs, 1)
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        print(f"LR schedule: {warmup_epochs}-epoch linear warmup → cosine annealing")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    try:
        from torch.cuda.amp import GradScaler  # type: ignore

        scaler = GradScaler() if device == "cuda" else None
    except Exception:
        scaler = None

    mixup_alpha = args.mixup_alpha if not args.fast else 0.0
    clip_grad_norm = args.clip_grad_norm

    if mixup_alpha > 0:
        print(f"Mixup enabled (alpha={mixup_alpha})")
    if clip_grad_norm > 0:
        print(f"Gradient clipping enabled (max_norm={clip_grad_norm})")

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            training=True, mixup_alpha=mixup_alpha, clip_grad_norm=clip_grad_norm,
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion, None, scaler, device, training=False,
        )

        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {vl_loss:>8.4f}  {vl_acc:>7.4f}  {elapsed:>5.1f}s")

        # Save best checkpoint (by val accuracy for classification)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_val_loss = vl_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "val_acc": vl_acc,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "model_type": "resnet18" if args.fast else "resnet101",
            }, args.save_path)
            print(f"  >> Best model saved (val_acc={vl_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience}).")
                break

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    plot_curves(train_losses, val_losses, train_accs, val_accs, "training_curves.png")

    print("\n=== Training Summary ===")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Best Val Acc  : {max(val_accs):.4f}")
    print(f"  Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
