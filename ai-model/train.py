"""Training script for rock/sediment/mineral image classification (ResNet-101)."""

import argparse
import copy
import os
import random
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
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Allow imports from sibling package when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from models.resnet101 import get_model, SmartMineResNet18
from models.sam import SAM


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for hard-example mining in imbalanced mineral datasets.

    Args:
        gamma: Focusing parameter (default: 2.0).  Higher values give more
            weight to hard misclassified examples.
        label_smoothing: Label smoothing coefficient (default: 0.0).
        reduction: ``"mean"`` or ``"sum"`` (default: ``"mean"``).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        # Compute cross-entropy with optional label smoothing
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


# ---------------------------------------------------------------------------
# MixUp / CutMix helpers
# ---------------------------------------------------------------------------

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed inputs, pairs of targets, and mixing coefficient lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the mixed loss for a MixUp batch."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)

    # Sample bounding box
    cut_rat = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[index], lam


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model weights for smoother inference.

    Args:
        model: The model to track.
        decay: EMA decay factor (default: 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model weights."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def build_transforms(use_randaugment: bool = False):
    """Return (train_transform, val_transform) as a tuple.

    Args:
        use_randaugment: When ``True``, append RandAugment to the train
            pipeline for additional automatic augmentation diversity.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_aug = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]

    if use_randaugment:
        train_aug.append(transforms.RandAugment(num_ops=2, magnitude=9))

    train_aug += [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]

    train_transform = transforms.Compose(train_aug)

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
# Layer-wise Learning Rate Decay
# ---------------------------------------------------------------------------

def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    decay: float = 0.9,
) -> list[dict]:
    """Build parameter groups with layer-wise learning rate decay.

    Layers closer to the input use a smaller learning rate by ``decay``
    per layer group, encouraging stable feature reuse from ImageNet
    pretraining while the head adapts faster.

    Args:
        model: ``SmartMineResNet101`` instance.
        base_lr: Learning rate for the classification head and top-most block.
        decay: Multiplicative decay applied per layer group going deeper.

    Returns:
        A list of ``{"params": ..., "lr": ...}`` dicts for the optimiser.
    """
    # Define groups from deepest (highest lr) to shallowest (lowest lr)
    layer_groups = [
        ("fc", list(model.fc.parameters())),
        ("layer4", list(model.layer4.parameters())),
        ("layer3", list(model.layer3.parameters())),
        ("layer2", list(model.layer2.parameters())),
    ]

    param_groups = []
    current_lr = base_lr
    for name, params in layer_groups:
        trainable = [p for p in params if p.requires_grad]
        if trainable:
            param_groups.append({"params": trainable, "lr": current_lr})
        current_lr *= decay

    return param_groups


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
    use_mixup: bool = False,
    use_cutmix: bool = False,
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    grad_clip: float = 0.0,
    ema: "EMA | None" = None,
    use_sam: bool = False,
):
    """Run one full epoch and return (avg_loss, accuracy).

    Args:
        optimizer: Required when *training* is ``True``; pass ``None`` for
            validation/evaluation passes where no parameter updates occur.
        use_mixup: Apply MixUp augmentation during training.
        use_cutmix: Apply CutMix augmentation during training (takes
            precedence over MixUp when both are enabled by random coin-flip).
        mixup_alpha: Alpha for MixUp Beta distribution.
        cutmix_alpha: Alpha for CutMix Beta distribution.
        grad_clip: If > 0, clip gradient L2-norm to this value.
        ema: Optional EMA tracker to update after each batch.
        use_sam: Use the SAM two-step optimiser update.
    """
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                # Choose augmentation strategy for this batch.
                # When both flags are set, randomly select one per batch.
                if use_mixup and use_cutmix:
                    do_cutmix = random.random() < 0.5
                    do_mixup = not do_cutmix
                else:
                    do_mixup = use_mixup
                    do_cutmix = use_cutmix

                if do_cutmix:
                    images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
                elif do_mixup:
                    images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                else:
                    y_a, y_b, lam = labels, labels, 1.0

                use_mixed = do_mixup or do_cutmix

                if use_sam:
                    # SAM first step
                    optimizer.zero_grad()
                    with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                        outputs = model(images)
                        loss = (
                            mixup_criterion(criterion, outputs, y_a, y_b, lam)
                            if use_mixed
                            else criterion(outputs, labels)
                        )
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    # SAM second step
                    with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                        outputs = model(images)
                        loss = (
                            mixup_criterion(criterion, outputs, y_a, y_b, lam)
                            if use_mixed
                            else criterion(outputs, labels)
                        )
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            [p for group in optimizer.param_groups for p in group["params"]],
                            grad_clip,
                        )
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                        outputs = model(images)
                        loss = (
                            mixup_criterion(criterion, outputs, y_a, y_b, lam)
                            if use_mixed
                            else criterion(outputs, labels)
                        )
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if grad_clip > 0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                [p for group in optimizer.param_groups for p in group["params"]],
                                grad_clip,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(
                                [p for group in optimizer.param_groups for p in group["params"]],
                                grad_clip,
                            )
                        optimizer.step()

                if ema is not None:
                    ema.update(model)
            else:
                with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
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
    parser.add_argument("--epochs", type=int, default=40)
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

    # ---- Accuracy improvement flags ----
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        help="Use Focal Loss instead of Cross-Entropy to focus on hard examples.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.2,
        # Increased from 0.1 to 0.2 to improve calibration and reduce over-confidence
        # on visually-similar mineral classes (e.g. Baryte vs Calcite).
        help="Label smoothing coefficient (default: 0.2).",
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Apply MixUp data augmentation during training.",
    )
    parser.add_argument(
        "--cutmix",
        action="store_true",
        help="Apply CutMix data augmentation during training.",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
        help="Alpha for MixUp Beta distribution (default: 0.4).",
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=1.0,
        help="Alpha for CutMix Beta distribution (default: 1.0).",
    )
    parser.add_argument(
        "--randaugment",
        action="store_true",
        help="Append RandAugment to the training augmentation pipeline.",
    )
    parser.add_argument(
        "--sam",
        action="store_true",
        help="Use SAM (Sharpness Aware Minimisation) optimiser.",
    )
    parser.add_argument(
        "--sam_rho",
        type=float,
        default=0.05,
        help="SAM neighbourhood size rho (default: 0.05).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="L2 weight decay (default: 5e-4).",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping L2 norm threshold (0 to disable, default: 1.0).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of linear LR warm-up epochs before cosine annealing (default: 5).",
    )
    parser.add_argument(
        "--llrd",
        action="store_true",
        help="Use Layer-wise Learning Rate Decay (LLRD) for per-layer LR scaling.",
    )
    parser.add_argument(
        "--llrd_decay",
        type=float,
        default=0.9,
        help="Multiplicative LR decay per layer group for LLRD (default: 0.9).",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Maintain an EMA copy of the model; save EMA weights as best checkpoint.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="Decay coefficient for EMA (default: 0.999).",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        help="Stochastic Depth (DropPath) drop probability for ResNet-101 (default: 0.1).",
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
            drop_path_rate=args.drop_path_rate,
        )
    print(f"Model: {type(model).__name__}")

    if args.focal_loss:
        criterion = FocalLoss(gamma=2.0, label_smoothing=args.label_smoothing)
        print(f"Loss: FocalLoss(gamma=2.0, label_smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"Loss: CrossEntropyLoss(label_smoothing={args.label_smoothing})")

    # Build parameter groups (LLRD or flat)
    if args.llrd and not args.fast:
        param_groups = build_llrd_param_groups(model, args.lr, decay=args.llrd_decay)
        print(f"LLRD enabled: {len(param_groups)} layer groups, decay={args.llrd_decay}")
    else:
        param_groups = [
            {"params": list(filter(lambda p: p.requires_grad, model.parameters())), "lr": args.lr}
        ]

    base_opt_kwargs = {"weight_decay": args.weight_decay}

    if args.sam:
        optimizer = SAM(
            param_groups,
            base_optimizer=torch.optim.Adam,
            rho=args.sam_rho,
            **base_opt_kwargs,
        )
        print(f"Optimiser: SAM(Adam, rho={args.sam_rho}, weight_decay={args.weight_decay})")
    else:
        optimizer = torch.optim.Adam(param_groups, **base_opt_kwargs)
        print(f"Optimiser: Adam(weight_decay={args.weight_decay})")

    # LR scheduler: optional linear warm-up then cosine annealing
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
        print(f"Scheduler: LinearWarmup({args.warmup_epochs} epochs) -> CosineAnnealingLR")
    else:
        scheduler = cosine_scheduler
        print("Scheduler: CosineAnnealingLR")

    try:
        from torch.cuda.amp import GradScaler  # type: ignore

        scaler = GradScaler() if device == "cuda" else None
    except Exception:
        scaler = None

    # Optional EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema else None
    if ema:
        print(f"EMA enabled (decay={args.ema_decay})")

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    aug_desc = []
    if args.mixup:
        aug_desc.append(f"MixUp(alpha={args.mixup_alpha})")
    if args.cutmix:
        aug_desc.append(f"CutMix(alpha={args.cutmix_alpha})")
    if aug_desc:
        print(f"Batch augmentation: {', '.join(aug_desc)}")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            training=True,
            use_mixup=args.mixup,
            use_cutmix=args.cutmix,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            grad_clip=args.grad_clip,
            ema=ema,
            use_sam=args.sam,
        )

        # Evaluate using EMA model if available
        eval_model = ema.shadow if ema is not None else model
        vl_loss, vl_acc = run_epoch(
            eval_model, val_loader, criterion, None, scaler, device, training=False
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
            save_state = ema.state_dict() if ema is not None else model.state_dict()
            torch.save({
                "epoch": epoch,
                "val_acc": vl_acc,
                "state_dict": save_state,
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
