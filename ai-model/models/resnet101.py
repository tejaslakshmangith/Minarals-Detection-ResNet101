"""ResNet model definitions for rock/sediment/mineral classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class SmartMineResNet18(nn.Module):
    """ResNet-18 - 5x faster than ResNet-101, same interface."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        b = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in b.parameters():
            p.requires_grad = False
        for p in b.layer2.parameters():
            p.requires_grad = True
        for p in b.layer3.parameters():
            p.requires_grad = True
        for p in b.layer4.parameters():
            p.requires_grad = True
        self.conv1, self.bn1, self.relu = b.conv1, b.bn1, b.relu
        self.maxpool = b.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = b.layer1, b.layer2, b.layer3, b.layer4
        self.avgpool = b.avgpool
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


class SmartMineResNet101(nn.Module):
    """ResNet-101 with a custom classification head for mine-safety detection.

    Pretrained ImageNet weights are used for transfer learning. By default
    ``layer2``, ``layer3``, ``layer4``, and the custom ``fc`` head are trained;
    ``layer1`` can optionally be unfrozen for finer mineral-specific features.

    Args:
        num_classes: Number of output classes (default: 4).
        unfreeze_layer1: When ``True``, ``layer1`` is also unfrozen so that
            low-level texture features can be fine-tuned for mineral images.
    """

    def __init__(self, num_classes: int = 4, unfreeze_layer1: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._unfreeze_layer1 = unfreeze_layer1

        # Load pretrained backbone
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Freeze all layers first, then selectively unfreeze for fine-tuning
        for param in backbone.parameters():
            param.requires_grad = False
        if unfreeze_layer1:
            for param in backbone.layer1.parameters():
                param.requires_grad = True
        for param in backbone.layer2.parameters():
            param.requires_grad = True
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Keep all backbone modules except the original fc
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Custom classification head: 2048 -> 1024 -> 512 -> num_classes
        # BatchNorm and reduced dropout (0.25) for more stable training and
        # better feature flow compared to the original 2048->512 head.
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def __repr__(self) -> str:  # noqa: D105
        unfrozen = "layer1+layer2+layer3+layer4" if self._unfreeze_layer1 else "layer2+layer3+layer4"
        return (
            f"SmartMineResNet101(\n"
            f"  backbone=ResNet-101 (pretrained, {unfrozen} unfrozen)\n"
            f"  fc=Linear(2048->1024)->BN->ReLU->Dropout(0.25)"
            f"->Linear(1024->512)->BN->ReLU->Dropout(0.25)->Linear(512->{self.num_classes})\n"
            f"  num_classes={self.num_classes}\n"
            f")"
        )


def get_model(
    num_classes: int = 4,
    device: str = "cpu",
    unfreeze_layer1: bool = False,
) -> SmartMineResNet101:
    """Factory function that returns a ``SmartMineResNet101`` on *device*.

    Args:
        num_classes: Number of output classes.
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
        unfreeze_layer1: When ``True``, ``layer1`` is also unfrozen during
            fine-tuning to capture mineral-specific low-level features.

    Returns:
        Model moved to the requested device.
    """
    model = SmartMineResNet101(num_classes=num_classes, unfreeze_layer1=unfreeze_layer1)
    model = model.to(device)
    return model
