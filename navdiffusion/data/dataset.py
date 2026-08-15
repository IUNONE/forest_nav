import random
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class NavigationDataset(Dataset):
    """Single-RGB-frame navigation dataset backed by the converted HDF5 file."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        split: str,
    ):
        self.split = split
        self.h5_file_path = data_config["h5_file_path"]
        # Keep compatibility with the original misspelled configuration key.
        self.transform = data_config.get("transform", data_config.get("tranform", {}))
        self.len_traj_pred = int(model_config["len_traj_pred"])

        with h5py.File(self.h5_file_path, "r") as h5_file:
            all_tokens = sorted(list(h5_file.keys()))
            self.tokens = self._split_tokens(all_tokens, h5_file, data_config, split)

            if (
                "trajectory_min_xy" in h5_file.attrs
                and "trajectory_max_xy" in h5_file.attrs
            ):
                trajectory_min = np.asarray(
                    h5_file.attrs["trajectory_min_xy"], dtype=np.float32
                )
                trajectory_max = np.asarray(
                    h5_file.attrs["trajectory_max_xy"], dtype=np.float32
                )
                configured_min = np.asarray(
                    [model_config["traj_norm_x"][0], model_config["traj_norm_y"][0]],
                    dtype=np.float32,
                )
                configured_max = np.asarray(
                    [model_config["traj_norm_x"][1], model_config["traj_norm_y"][1]],
                    dtype=np.float32,
                )
                if np.any(trajectory_min < configured_min) or np.any(
                    trajectory_max > configured_max
                ):
                    raise ValueError(
                        "Trajectory range {} -> {} exceeds configured normalization "
                        "range {} -> {}".format(
                            trajectory_min.tolist(),
                            trajectory_max.tolist(),
                            configured_min.tolist(),
                            configured_max.tolist(),
                        )
                    )

            if self.tokens:
                first_group = h5_file[self.tokens[0]]
                image_shape = first_group["image"].shape
                trajectory_shape = first_group["future_waypoint_local"].shape
                if len(image_shape) != 3 or image_shape[-1] != 3:
                    raise ValueError(
                        "Expected RGB images shaped [H, W, 3], got {}".format(
                            image_shape
                        )
                    )
                if len(trajectory_shape) != 2 or trajectory_shape[1] < 2:
                    raise ValueError(
                        "Expected trajectories shaped [N, >=2], got {}".format(
                            trajectory_shape
                        )
                    )
                if trajectory_shape[0] < self.len_traj_pred:
                    raise ValueError(
                        "Dataset has {} trajectory points but model requests {}".format(
                            trajectory_shape[0], self.len_traj_pred
                        )
                    )

        print(
            "Loaded {} {} samples from {}".format(
                len(self.tokens), self.split, self.h5_file_path
            )
        )

    @staticmethod
    def _split_tokens(
        tokens: List[str],
        h5_file: h5py.File,
        data_config: Dict[str, Any],
        split: str,
    ) -> List[str]:
        if not bool(data_config.get("use_validation", False)):
            return tokens if split == "train" else []
        if split not in ("train", "val"):
            return tokens
        val_fraction = float(data_config.get("val_fraction", 0.1))
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("data_params.val_fraction must be between 0 and 1")
        if len(tokens) < 2:
            return tokens

        seed = int(data_config.get("split_seed", 42))
        split_by_bag = bool(data_config.get("split_by_bag", True))
        units: Dict[str, List[str]] = {}
        if split_by_bag:
            for token in tokens:
                source_bag = h5_file[token].attrs.get("source_bag", "")
                if isinstance(source_bag, bytes):
                    source_bag = source_bag.decode("utf-8")
                units.setdefault(str(source_bag), []).append(token)

        # A single bag cannot produce a non-empty bag-level train and val split.
        if not split_by_bag or len(units) < 2:
            units = {token: [token] for token in tokens}

        unit_names = sorted(units)
        configured_val_units = set(data_config.get("val_bags", []))
        if configured_val_units:
            unknown_units = configured_val_units.difference(unit_names)
            if unknown_units:
                raise ValueError(
                    "Configured validation bags are absent from the dataset: {}".format(
                        sorted(unknown_units)
                    )
                )
            if configured_val_units == set(unit_names):
                raise ValueError("Validation bags cannot contain every source bag")
            val_units = configured_val_units
        else:
            random.Random(seed).shuffle(unit_names)
            val_unit_count = max(1, int(round(len(unit_names) * val_fraction)))
            val_unit_count = min(val_unit_count, len(unit_names) - 1)
            val_units = set(unit_names[:val_unit_count])
        selected = [
            token
            for unit_name in unit_names
            if (unit_name in val_units) == (split == "val")
            for token in units[unit_name]
        ]
        return sorted(selected)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        token = self.tokens[index]
        with h5py.File(self.h5_file_path, "r", swmr=True) as h5_file:
            image = h5_file[token]["image"][:]
            future_waypoints = h5_file[token]["future_waypoint_local"][
                : self.len_traj_pred, :2
            ]

        image = image.astype(np.float32, copy=False)
        if image.max(initial=0.0) > 1.0:
            image = image / 255.0
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        image_tensor = self._transform(image_tensor)

        return (
            image_tensor,
            torch.as_tensor(future_waypoints, dtype=torch.float32),
        )

    def _transform(self, image: torch.Tensor) -> torch.Tensor:
        transform_list = []
        if "resize" in self.transform:
            width, height = self.transform["resize"]["w_h"]
            transform_list.append(transforms.Resize((height, width)))
        if "norm" in self.transform:
            transform_list.append(
                transforms.Normalize(
                    self.transform["norm"]["mean"],
                    self.transform["norm"]["std"],
                )
            )
        if not transform_list:
            return image
        return transforms.Compose(transform_list)(image)
