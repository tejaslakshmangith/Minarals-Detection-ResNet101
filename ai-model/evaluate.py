"""Evaluation script for rock/sediment/mineral classification (ResNet-101)."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow imports from sibling package when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from models.resnet101 import SmartMineResNet101


def load_checkpoint(model_path: str, device: str):
    """Load a saved checkpoint and return (model, class_names)."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = SmartMineResNet101(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names


def build_test_transform():
    """Standard ImageNet normalisation transform for inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def evaluate(model, loader, device):
    """Run the model over *loader* and collect predictions + true labels."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names: list, save_path: str) -> None:
    """Save a seaborn heatmap of *cm* to *save_path*."""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def print_per_class_metrics(true_labels, pred_labels, class_names):
    """Print per-class accuracy and F1-score for each mineral class."""
    print("\nPer-Class Metrics:")
    print(f"  {'Class':<30}  {'Accuracy':>8}  {'F1-Score':>8}  {'Correct':>7}  {'Total':>6}")
    print("  " + "-" * 70)

    for idx, name in enumerate(class_names):
        mask = true_labels == idx
        if mask.sum() == 0:
            continue
        class_correct = (pred_labels[mask] == idx).sum()
        class_total = mask.sum()
        class_acc = class_correct / class_total
        # Per-class F1: treat this class as positive, all others as negative
        binary_true = (true_labels == idx).astype(int)
        binary_pred = (pred_labels == idx).astype(int)
        class_f1 = f1_score(binary_true, binary_pred, zero_division=0)
        print(f"  {name:<30}  {class_acc:>8.4f}  {class_f1:>8.4f}  {class_correct:>7}  {class_total:>6}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SmartMine ResNet-101 model")
    parser.add_argument("--model_path", type=str,
                        default="ai-model/models/resnet101_rocks.pth")
    parser.add_argument("--data_dir", type=str, default="ai-model/dataset/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--confusion_matrix_path",
        type=str,
        default="confusion_matrix.png",
        help="File path to save the confusion matrix image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model, class_names = load_checkpoint(args.model_path, device)
    print(f"Classes: {class_names}")

    # Dataset & loader
    test_dataset = datasets.ImageFolder(args.data_dir, transform=build_test_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Run evaluation
    true_labels, pred_labels = evaluate(model, test_loader, device)

    # Overall accuracy
    accuracy = (true_labels == pred_labels).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Macro / weighted F1
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    print(f"Macro    F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

    # Per-class accuracy & F1
    print_per_class_metrics(true_labels, pred_labels, class_names)

    # Full sklearn classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, class_names, save_path=args.confusion_matrix_path)


if __name__ == "__main__":
    main()
