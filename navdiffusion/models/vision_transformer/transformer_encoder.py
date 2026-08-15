from typing import Callable, Optional

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import ResNet50_Weights, resnet50


class VisualEncoder(nn.Module):
    """Encode one RGB image while retaining coarse spatial information."""

    def __init__(
        self,
        obs_encoder: str = "efficientnet-b0",
        embedding_dim: int = 128,
        input_channels: int = 3,
        pretrained: bool = False,
        replace_batch_norm: bool = True,
        spatial_pool_size: int = 4,
        spatial_channels: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if spatial_pool_size < 1:
            raise ValueError("spatial_pool_size must be positive")
        if spatial_channels < 1:
            raise ValueError("spatial_channels must be positive")

        self.embedding_dim = int(embedding_dim)
        self.obs_encoder_type = obs_encoder

        if obs_encoder.startswith("efficientnet-"):
            if pretrained:
                self.obs_encoder = EfficientNet.from_pretrained(
                    obs_encoder, in_channels=input_channels
                )
            else:
                self.obs_encoder = EfficientNet.from_name(
                    obs_encoder, in_channels=input_channels
                )
            feature_channels = self.obs_encoder._fc.in_features
        elif obs_encoder == "resnet50":
            if pretrained and input_channels != 3:
                raise ValueError("Pretrained ResNet-50 requires three input channels")
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
            if input_channels != 3:
                backbone.conv1 = nn.Conv2d(
                    input_channels,
                    backbone.conv1.out_channels,
                    kernel_size=backbone.conv1.kernel_size,
                    stride=backbone.conv1.stride,
                    padding=backbone.conv1.padding,
                    bias=False,
                )
            feature_channels = backbone.fc.in_features
            # Keep the convolutional feature map; the original avgpool/fc would
            # discard the left/right layout that is important for steering.
            self.obs_encoder = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError("Unsupported visual encoder: {}".format(obs_encoder))

        if replace_batch_norm:
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)

        spatial_groups = _valid_group_count(spatial_channels)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, spatial_channels, kernel_size=1, bias=False),
            nn.GroupNorm(spatial_groups, spatial_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((spatial_pool_size, spatial_pool_size)),
        )
        flattened_features = spatial_channels * spatial_pool_size * spatial_pool_size
        self.compress_obs_enc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_features, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Mish(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 5 and imgs.shape[1] == 1:
            imgs = imgs[:, 0]
        if imgs.ndim != 4:
            raise ValueError(
                "Expected images shaped [B, C, H, W], got {}".format(imgs.shape)
            )

        if self.obs_encoder_type.startswith("efficientnet-"):
            feature_map = self.obs_encoder.extract_features(imgs)
        else:
            feature_map = self.obs_encoder(imgs)
        return self.compress_obs_enc(self.spatial_encoder(feature_map))


# Backward-compatible import name.
VisualGoalTransformer = VisualEncoder


def _valid_group_count(num_channels: int, features_per_group: int = 16) -> int:
    """Choose a valid GroupNorm group count near the requested group size."""
    num_groups = max(1, num_channels // features_per_group)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return num_groups


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """Replace BatchNorm2d with valid GroupNorm layers, preserving affine values."""

    def make_group_norm(batch_norm: nn.BatchNorm2d) -> nn.GroupNorm:
        group_norm = nn.GroupNorm(
            num_groups=_valid_group_count(
                batch_norm.num_features, features_per_group=features_per_group
            ),
            num_channels=batch_norm.num_features,
            eps=batch_norm.eps,
            affine=batch_norm.affine,
        )
        if batch_norm.affine:
            with torch.no_grad():
                group_norm.weight.copy_(batch_norm.weight)
                group_norm.bias.copy_(batch_norm.bias)
        return group_norm

    return replace_submodules(
        root_module=root_module,
        predicate=lambda module: isinstance(module, nn.BatchNorm2d),
        func=make_group_norm,
    )


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace every matching descendant module in place."""
    if predicate(root_module):
        return func(root_module)

    matching_paths = [
        name.split(".")
        for name, module in root_module.named_modules(remove_duplicate=True)
        if predicate(module)
    ]
    for *parent_path, child_name in matching_paths:
        parent_module = root_module
        if parent_path:
            parent_module = root_module.get_submodule(".".join(parent_path))
        source_module = (
            parent_module[int(child_name)]
            if isinstance(parent_module, nn.Sequential)
            else getattr(parent_module, child_name)
        )
        target_module = func(source_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(child_name)] = target_module
        else:
            setattr(parent_module, child_name, target_module)

    remaining = [module for module in root_module.modules() if predicate(module)]
    if remaining:
        raise RuntimeError("Failed to replace every matching submodule")
    return root_module
