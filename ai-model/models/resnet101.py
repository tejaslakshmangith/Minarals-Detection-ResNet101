"""ResNet model definitions for rock/sediment/mineral classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularisation.

    During training each sample is independently dropped (replaced by a
    zero tensor) with probability *drop_prob*.  At test time the full
    residual is passed through unchanged.

    Args:
        drop_prob: Probability of dropping the residual branch (default: 0.0).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape: (batch, 1, 1, 1) so the same mask is applied to all spatial positions.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob).to(x.dtype)
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


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

    Pretrained ImageNet weights are used for transfer learning. All layers
    except ``layer2``, ``layer3``, ``layer4``, and the custom ``fc`` head
    are frozen by default (controllable via ``llrd`` parameter).

    Args:
        num_classes: Number of output classes (default: 4).
        drop_path_rate: Stochastic Depth drop probability applied after the
            backbone's spatial feature extraction (default: 0.1).
    """

    def __init__(self, num_classes: int = 4, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Freeze early layers; unfreeze layer2, layer3, layer4 for mineral-specific features
        for param in backbone.parameters():
            param.requires_grad = False
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

        # Stochastic Depth applied to the feature map before the head
        self.drop_path = DropPath(drop_prob=drop_path_rate)

        # Custom classification head: 2048 -> 512 -> num_classes
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
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
        # Apply stochastic depth to the flattened feature vector
        x = self.drop_path(x)
        x = self.fc(x)
        return x

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SmartMineResNet101(\n"
            f"  backbone=ResNet-101 (pretrained, layer2+layer3+layer4 unfrozen)\n"
            f"  drop_path={self.drop_path.drop_prob}\n"
            f"  fc=Linear(2048->512)->ReLU->Dropout(0.4)->Linear(512->{self.num_classes})\n"
            f"  num_classes={self.num_classes}\n"
            f")"
        )


def get_model(
    num_classes: int = 4,
    device: str = "cpu",
    drop_path_rate: float = 0.1,
) -> SmartMineResNet101:
    """Factory function that returns a ``SmartMineResNet101`` on *device*.

    Args:
        num_classes: Number of output classes.
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
        drop_path_rate: Stochastic Depth drop probability (default: 0.1).

    Returns:
        Model moved to the requested device.
    """
    model = SmartMineResNet101(num_classes=num_classes, drop_path_rate=drop_path_rate)
    model = model.to(device)
    return model
