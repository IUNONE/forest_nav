#!/usr/bin/env python3
"""Infer navigation trajectories from an MCAP and write them to a new MCAP.

The source MCAP is copied message-for-message.  For each sampled RGB image,
this script adds seven ``nav_msgs/msg/Path`` messages to
``/planned_trajectory_opt1`` through ``/planned_trajectory_opt7``.  Each
candidate uses a different deterministic random seed for diffusion noise.
The ground-truth future trajectory is written to
``/planned_trajectory_final``.  Path timestamps are snapped to the nearest
odometry timestamps so the result can be rendered together with the recorded
odometry in Rerun.

Run from the repository root with the repository on PYTHONPATH, for example:

    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    python -B scripts/inference_add_topic.py \
        --input /path/to/input.mcap \
        --output /path/to/output_with_trajectory.mcap \
        --checkpoint results/resnet50_spatial/epoch=999-step=4000.ckpt

For a directory, use ``--input_dir`` and ``--output_dir``.  All ``.mcap``
files are processed sequentially, preserving their relative paths and file
names under the output directory.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from torchvision import transforms
from tqdm import tqdm

try:
    from mcap.reader import make_reader
    from mcap.writer import Writer as McapWriter
    from mcap_ros2._dynamic import serialize_dynamic
    from mcap_ros2.decoder import DecoderFactory
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "MCAP support is missing. Install it in the forest_nav environment with:\n"
        "  conda run -n forest_nav pip install mcap mcap-ros2-support"
    ) from exc

from navdiffusion import NavDiffsionLightning
from scripts.data_process.mcap_to_h5 import decode_ros_image, resize_rgb


NANOSECONDS_PER_SECOND = 1_000_000_000
DEFAULT_IMAGE_TOPIC = "/robot1/D435i_front/color/image_raw"
DEFAULT_ODOM_TOPIC = "/Odometry"
PLANNED_TOPICS = ("/planned_trajectory_final",) + tuple(
    f"/planned_trajectory_opt{index}" for index in range(1, 8)
)
PATH_SCHEMA_NAME = "nav_msgs/msg/Path"
PATH_MESSAGE_DEFINITION = """std_msgs/Header header
geometry_msgs/PoseStamped[] poses
===
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
===
MSG: geometry_msgs/PoseStamped
std_msgs/Header header
geometry_msgs/Pose pose
===
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
===
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
===
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""


@dataclass(frozen=True)
class OdometrySeries:
    """Odometry pose samples keyed by MCAP log timestamps."""

    timestamps_ns: np.ndarray
    positions_xy: np.ndarray
    yaws: np.ndarray
    root_frame: str
    body_frame: str


@dataclass(frozen=True)
class InferenceResources:
    """Model and preprocessing resources shared by multiple MCAP files."""

    image_transform: dict[str, Any]
    image_size: int
    trajectory_points: int
    model: NavDiffsionLightning
    device: torch.device


def clean_frame_id(frame_id: object) -> str:
    """Normalize ROS frame IDs without changing their semantic name."""
    return "/".join(part for part in str(frame_id).strip().split("/") if part)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract planar yaw from an xyzw quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Odometry contains an invalid orientation quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def load_odometry(mcap_path: Path, odom_topic: str) -> OdometrySeries:
    """Load odometry using MCAP log timestamps for exact Rerun alignment."""
    timestamps: list[int] = []
    positions: list[tuple[float, float]] = []
    yaws: list[float] = []
    root_frame: str | None = None
    body_frame: str | None = None

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, record, odometry in reader.iter_decoded_messages(
            topics=[odom_topic]
        ):
            position = odometry.pose.pose.position
            orientation = odometry.pose.pose.orientation
            timestamp_ns = int(record.log_time)
            values = (
                timestamp_ns,
                float(position.x),
                float(position.y),
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            if not np.isfinite(values).all():
                continue
            current_root = clean_frame_id(odometry.header.frame_id)
            current_body = clean_frame_id(odometry.child_frame_id)
            if root_frame is None:
                root_frame = current_root
                body_frame = current_body
            elif current_root != root_frame or current_body != body_frame:
                raise ValueError(
                    "Odometry frame changed from {} -> {} to {} -> {}".format(
                        root_frame,
                        body_frame,
                        current_root,
                        current_body,
                    )
                )
            timestamps.append(timestamp_ns)
            positions.append((float(position.x), float(position.y)))
            yaws.append(
                quaternion_to_yaw(
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                )
            )

    if len(timestamps) < 2:
        raise ValueError(
            "Topic {!r} in {} contains fewer than two valid odometry messages".format(
                odom_topic, mcap_path
            )
        )

    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    position_array = np.asarray(positions, dtype=np.float64)
    yaw_array = np.asarray(yaws, dtype=np.float64)
    order = np.argsort(timestamp_array, kind="stable")
    timestamp_array = timestamp_array[order]
    position_array = position_array[order]
    yaw_array = yaw_array[order]

    # Keep the last sample for duplicate log timestamps.
    _, reverse_indices = np.unique(timestamp_array[::-1], return_index=True)
    keep = np.sort(len(timestamp_array) - 1 - reverse_indices)
    timestamp_array = timestamp_array[keep]
    position_array = position_array[keep]
    yaw_array = np.unwrap(yaw_array[keep])
    if len(timestamp_array) < 2 or np.any(np.diff(timestamp_array) <= 0):
        raise ValueError("Odometry timestamps are not strictly increasing")

    return OdometrySeries(
        timestamps_ns=timestamp_array,
        positions_xy=position_array,
        yaws=yaw_array,
        root_frame=root_frame or "",
        body_frame=body_frame or "",
    )


def nearest_timestamp_index(timestamps_ns: np.ndarray, timestamp_ns: int) -> int:
    """Return the index of the odometry sample nearest to a timestamp."""
    right = int(np.searchsorted(timestamps_ns, timestamp_ns, side="left"))
    if right <= 0:
        return 0
    if right >= len(timestamps_ns):
        return len(timestamps_ns) - 1
    left = right - 1
    if timestamp_ns - int(timestamps_ns[left]) <= int(
        timestamps_ns[right]
    ) - timestamp_ns:
        return left
    return right


def future_ground_truth(
    odometry: OdometrySeries,
    current_index: int,
    trajectory_points: int,
    interval_seconds: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return local-frame GT points and their nearest real odometry times."""
    timestamps = odometry.timestamps_ns
    current_timestamp = int(timestamps[current_index])
    desired_times = current_timestamp + np.rint(
        interval_seconds
        * NANOSECONDS_PER_SECOND
        * np.arange(1, trajectory_points + 1, dtype=np.float64)
    ).astype(np.int64)
    if desired_times[-1] > int(timestamps[-1]):
        return None

    right = np.searchsorted(timestamps, desired_times, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(timestamps[right] - desired_times) < np.abs(
        desired_times - timestamps[left]
    )
    future_indices = np.where(choose_right, right, left).astype(np.int64)
    future_indices = np.maximum(future_indices, current_index + 1)
    if np.any(future_indices >= len(timestamps)):
        return None

    current_position = odometry.positions_xy[current_index]
    # Keep the model/data convention of fixed 0.1 s (or configured) future
    # points, while using the nearest real odometry timestamps on the Path
    # poses for synchronization.
    future_positions = np.column_stack(
        [
            np.interp(desired_times, timestamps, odometry.positions_xy[:, axis])
            for axis in range(2)
        ]
    )
    delta = future_positions - current_position[None, :]
    current_yaw = float(odometry.yaws[current_index])
    cos_yaw = math.cos(current_yaw)
    sin_yaw = math.sin(current_yaw)
    local_trajectory = np.column_stack(
        (
            cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1],
            -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1],
        )
    ).astype(np.float32)
    if not np.isfinite(local_trajectory).all():
        return None
    return local_trajectory, timestamps[future_indices].copy()


def prepare_image(
    image_message: Any,
    transform_config: dict[str, Any],
    image_size: int,
    resize_mode: str,
) -> torch.Tensor:
    """Decode an MCAP RGB image with the same preprocessing as training."""
    image = resize_rgb(
        decode_ros_image(image_message),
        size=image_size,
        mode=resize_mode,
    ).astype(np.float32, copy=False)
    image_tensor = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1)))
    norm = transform_config.get("norm")
    if norm is not None:
        image_tensor = transforms.Normalize(norm["mean"], norm["std"])(image_tensor)
    return image_tensor


def seed_torch(seed: int) -> None:
    """Reset all relevant RNGs before one diffusion sample."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_message(timestamp_ns: int) -> SimpleNamespace:
    """Build a ROS builtin_interfaces/Time-like object."""
    seconds, nanoseconds = divmod(int(timestamp_ns), NANOSECONDS_PER_SECOND)
    return SimpleNamespace(sec=int(seconds), nanosec=int(nanoseconds))


def make_path_message(
    points_xy: np.ndarray,
    current_timestamp_ns: int,
    waypoint_timestamps_ns: np.ndarray,
    frame_id: str,
) -> SimpleNamespace:
    """Build a nav_msgs/msg/Path compatible dynamic message."""
    current_time = time_message(current_timestamp_ns)
    header = SimpleNamespace(stamp=current_time, frame_id=frame_id)
    poses = []
    for point, waypoint_timestamp_ns in zip(points_xy, waypoint_timestamps_ns):
        pose_header = SimpleNamespace(
            stamp=time_message(int(waypoint_timestamp_ns)),
            frame_id=frame_id,
        )
        poses.append(
            SimpleNamespace(
                header=pose_header,
                pose=SimpleNamespace(
                    position=SimpleNamespace(
                        x=float(point[0]),
                        y=float(point[1]),
                        z=0.0,
                    ),
                    orientation=SimpleNamespace(
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        w=1.0,
                    ),
                ),
            )
        )
    return SimpleNamespace(header=header, poses=poses)


def resolve_checkpoint(
    checkpoint: Path | None,
    checkpoint_dir: Path,
) -> Path:
    """Resolve an explicit checkpoint or the latest checkpoint in a directory."""
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint
    candidates = sorted(checkpoint_dir.expanduser().resolve().glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}; pass --checkpoint explicitly"
        )
    return candidates[-1]


def validate_runtime_args(args: argparse.Namespace) -> None:
    """Validate options that apply independently to each input MCAP."""
    if args.sample_interval < 0.0:
        raise ValueError("--sample-interval must be non-negative")
    if args.trajectory_interval <= 0.0:
        raise ValueError("--trajectory-interval must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if not 1 <= args.candidate_batch_size <= len(PLANNED_TOPICS) - 1:
        raise ValueError("--candidate-batch-size must be between 1 and 7")


def load_inference_resources(args: argparse.Namespace) -> InferenceResources:
    """Load configuration and checkpoint once for single- or directory-mode."""
    config_path = args.config_path.expanduser().resolve()
    with config_path.open("r") as stream:
        config = yaml.safe_load(stream)
    data_config = config["data_params"]
    model_config = config["model_params"]
    image_transform = data_config.get(
        "transform", data_config.get("tranform", {})
    )
    resize_config = image_transform.get("resize", {}).get("w_h", [224, 224])
    if len(resize_config) != 2 or int(resize_config[0]) != int(resize_config[1]):
        raise ValueError(
            "The inference preprocessing currently requires a square resize, "
            f"got {resize_config}"
        )
    image_size = int(resize_config[0])
    trajectory_points = int(model_config["len_traj_pred"])
    if trajectory_points < 1:
        raise ValueError("model_params.len_traj_pred must be positive")

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    print(f"Loading checkpoint: {checkpoint_path}")
    model = NavDiffsionLightning.load_from_checkpoint(
        str(checkpoint_path), map_location=device
    )
    configured_steps = model_config.get("noise_pred_net", {}).get(
        "num_inference_steps"
    )
    if configured_steps is not None:
        model.num_inference_steps = int(configured_steps)
    model.eval().to(device)
    print(
        f"Model device: {device}; diffusion inference steps: "
        f"{model.num_inference_steps}"
    )
    return InferenceResources(
        image_transform=image_transform,
        image_size=image_size,
        trajectory_points=trajectory_points,
        model=model,
        device=device,
    )


_DIRECTORY_WORKER_ARGS: argparse.Namespace | None = None
_DIRECTORY_WORKER_RESOURCES: InferenceResources | None = None


def initialize_directory_worker(
    args_values: dict[str, Any],
    worker_threads: int,
) -> None:
    """Load one model per CPU worker and cap its intra-op thread count."""
    global _DIRECTORY_WORKER_ARGS, _DIRECTORY_WORKER_RESOURCES
    if worker_threads > 0:
        torch.set_num_threads(worker_threads)
    _DIRECTORY_WORKER_ARGS = argparse.Namespace(**args_values)
    _DIRECTORY_WORKER_RESOURCES = load_inference_resources(_DIRECTORY_WORKER_ARGS)


def process_directory_file_worker(
    input_path: Path,
    output_path: Path,
    progress_position: int,
) -> tuple[Path, Counter]:
    """Process one directory-mode MCAP inside a worker process."""
    if _DIRECTORY_WORKER_ARGS is None or _DIRECTORY_WORKER_RESOURCES is None:
        raise RuntimeError("Directory worker was not initialized")
    file_args = argparse.Namespace(**vars(_DIRECTORY_WORKER_ARGS))
    file_args.input = input_path
    file_args.output = output_path
    file_args.progress_position = progress_position
    return input_path, infer_and_write(file_args, _DIRECTORY_WORKER_RESOURCES)


def register_source_message(
    writer: McapWriter,
    schema: Any,
    channel: Any,
    message: Any,
    schema_ids: dict[int, int],
    channel_ids: dict[int, int],
) -> None:
    """Copy one raw source message and lazily recreate its schema/channel."""
    source_schema_id = int(channel.schema_id)
    if source_schema_id == 0:
        output_schema_id = 0
    else:
        if schema is None:
            raise ValueError(
                f"Channel {channel.topic} references missing schema {source_schema_id}"
            )
        output_schema_id = schema_ids.get(source_schema_id, 0)
        if output_schema_id == 0:
            output_schema_id = writer.register_schema(
                name=schema.name,
                encoding=schema.encoding,
                data=schema.data,
            )
            schema_ids[source_schema_id] = output_schema_id

    output_channel_id = channel_ids.get(int(channel.id))
    if output_channel_id is None:
        output_channel_id = writer.register_channel(
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            schema_id=output_schema_id,
            metadata=dict(channel.metadata),
        )
        channel_ids[int(channel.id)] = output_channel_id

    writer.add_message(
        channel_id=output_channel_id,
        log_time=int(message.log_time),
        publish_time=int(message.publish_time),
        sequence=int(message.sequence),
        data=message.data,
    )


def infer_and_write(
    args: argparse.Namespace,
    resources: InferenceResources | None = None,
) -> Counter:
    """Copy the MCAP and append synchronized prediction/GT Path messages."""
    validate_runtime_args(args)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output MCAP paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input MCAP does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} (use --overwrite to replace it)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    odometry = load_odometry(input_path, args.odom_topic)
    if resources is None:
        resources = load_inference_resources(args)
    image_transform = resources.image_transform
    image_size = resources.image_size
    trajectory_points = resources.trajectory_points
    model = resources.model
    device = resources.device
    print(
        f"Odometry: {odometry.root_frame} -> {odometry.body_frame}; "
        f"samples: {len(odometry.timestamps_ns)}"
    )

    path_schema_data = PATH_MESSAGE_DEFINITION.encode("utf-8")
    path_encoder = serialize_dynamic(
        PATH_SCHEMA_NAME, PATH_MESSAGE_DEFINITION
    )[PATH_SCHEMA_NAME]
    stats: Counter = Counter()
    last_odom_index: int | None = None
    seed_stride = len(PLANNED_TOPICS) + 1

    # Check for collisions before truncating/replacing the requested output.
    with input_path.open("rb") as summary_stream:
        summary_reader = make_reader(summary_stream)
        summary = summary_reader.get_summary()
        existing_topics = {
            channel.topic
            for channel in (summary.channels.values() if summary else [])
        }
        image_message_total: int | None = None
        if summary is None:
            image_message_total = 0
            for _, channel, _ in summary_reader.iter_messages(
                log_time_order=False
            ):
                existing_topics.add(channel.topic)
                if channel.topic == args.image_topic:
                    image_message_total += 1
        elif summary.statistics is not None:
            image_channel_ids = {
                int(channel.id)
                for channel in summary.channels.values()
                if channel.topic == args.image_topic
            }
            image_message_total = sum(
                summary.statistics.channel_message_counts.get(channel_id, 0)
                for channel_id in image_channel_ids
            )
    overlap = existing_topics.intersection(PLANNED_TOPICS)
    if overlap:
        raise ValueError(
            "Input MCAP already contains planned trajectory topics: "
            + ", ".join(sorted(overlap))
        )
    if output_path.exists():
        output_path.unlink()

    with input_path.open("rb") as input_stream, output_path.open("wb") as output_stream:
        input_reader = make_reader(input_stream)

        writer = McapWriter(output_stream)
        writer.start(profile="ros2", library="forest_nav/inference_add_topic")
        path_schema_id = writer.register_schema(
            name=PATH_SCHEMA_NAME,
            encoding="ros2msg",
            data=path_schema_data,
        )
        path_channel_ids = {
            topic: writer.register_channel(
                topic=topic,
                message_encoding="cdr",
                schema_id=path_schema_id,
            )
            for topic in PLANNED_TOPICS
        }
        source_schema_ids: dict[int, int] = {}
        source_channel_ids: dict[int, int] = {}
        image_decoder_factory = DecoderFactory()
        next_image_ns = -math.inf
        progress = tqdm(
            total=image_message_total,
            desc=f"{input_path.name} samples",
            unit="sample",
            dynamic_ncols=True,
            position=getattr(args, "progress_position", 0),
        )

        try:
            for schema, channel, message in input_reader.iter_messages(
                log_time_order=True
            ):
                if channel.topic in PLANNED_TOPICS:
                    raise ValueError(
                        "Input MCAP contains a planned trajectory topic: "
                        + channel.topic
                    )
                register_source_message(
                    writer,
                    schema,
                    channel,
                    message,
                    source_schema_ids,
                    source_channel_ids,
                )
                stats["copied_messages"] += 1
                if channel.topic != args.image_topic:
                    continue
                stats["image_messages"] += 1
                progress.update(1)
                if (
                    args.max_samples is not None
                    and stats["written_samples"] >= args.max_samples
                ):
                    stats["skipped_max_samples"] += 1
                    continue
                image_timestamp_ns = int(message.log_time)
                if image_timestamp_ns < next_image_ns:
                    stats["skipped_sample_interval"] += 1
                    continue
                next_image_ns = image_timestamp_ns + int(
                    args.sample_interval * NANOSECONDS_PER_SECOND
                )
                current_odom_index = nearest_timestamp_index(
                    odometry.timestamps_ns, image_timestamp_ns
                )
                if last_odom_index == current_odom_index:
                    stats["skipped_duplicate_odom"] += 1
                    continue
                current_timestamp_ns = int(
                    odometry.timestamps_ns[current_odom_index]
                )
                gt_result = future_ground_truth(
                    odometry,
                    current_odom_index,
                    trajectory_points,
                    args.trajectory_interval,
                )
                if gt_result is None:
                    stats["skipped_incomplete_future"] += 1
                    continue
                gt_trajectory, waypoint_timestamps_ns = gt_result

                try:
                    decoder = image_decoder_factory.decoder_for(
                        channel.message_encoding, schema
                    )
                    if decoder is None:
                        raise ValueError(
                            f"No ROS2 decoder for image topic {args.image_topic}"
                        )
                    image_message = decoder(message.data)
                    image_tensor = prepare_image(
                        image_message,
                        image_transform,
                        image_size,
                        args.resize_mode,
                    ).unsqueeze(0).to(device)
                except Exception as exc:
                    stats["image_decode_error"] += 1
                    print(
                        f"Skipping image at {image_timestamp_ns}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                predictions: list[np.ndarray] = []
                with torch.inference_mode():
                    # The seven candidates share the same image condition, so
                    # denoise them as a batch instead of running seven separate
                    # forward passes. Each row receives its own seeded initial
                    # noise, preserving the multi-candidate behavior.
                    image_condition = model.transformer_encoder(image_tensor)
                    for batch_start in range(
                        0, len(PLANNED_TOPICS) - 1, args.candidate_batch_size
                    ):
                        option_indices = range(
                            batch_start + 1,
                            min(
                                batch_start + args.candidate_batch_size,
                                len(PLANNED_TOPICS) - 1,
                            )
                            + 1,
                        )
                        initial_noises = []
                        for option_index in option_indices:
                            seed = (
                                args.seed_base
                                + stats["written_samples"] * seed_stride
                                + option_index
                            )
                            seed_torch(seed)
                            initial_noises.append(
                                torch.randn(
                                    (1, trajectory_points, 2),
                                    device=device,
                                    dtype=image_tensor.dtype,
                                )
                            )
                        candidate_count = len(initial_noises)
                        candidate_images = image_tensor.expand(
                            candidate_count, -1, -1, -1
                        ).contiguous()
                        candidate_condition = image_condition.expand(
                            candidate_count, -1
                        ).contiguous()
                        initial_noise = torch.cat(initial_noises, dim=0)
                        prediction = model.predict(
                            (candidate_images, None),
                            initial_noise=initial_noise,
                            image_cond=candidate_condition,
                        )
                        prediction_np = prediction.detach().cpu().numpy()
                        expected_shape = (
                            candidate_count,
                            trajectory_points,
                            2,
                        )
                        if prediction_np.shape != expected_shape:
                            raise ValueError(
                                "Model prediction has shape {}, expected {}".format(
                                    prediction_np.shape, expected_shape
                                )
                            )
                        predictions.extend(
                            prediction_np.astype(np.float32)
                        )

                paths = [*predictions, gt_trajectory]
                for topic, points in zip(PLANNED_TOPICS[1:], paths[:-1]):
                    path_message = make_path_message(
                        points,
                        current_timestamp_ns,
                        waypoint_timestamps_ns,
                        odometry.body_frame,
                    )
                    writer.add_message(
                        channel_id=path_channel_ids[topic],
                        log_time=current_timestamp_ns,
                        publish_time=current_timestamp_ns,
                        sequence=stats[f"sequence:{topic}"],
                        data=path_encoder(path_message),
                    )
                    stats[f"sequence:{topic}"] += 1
                final_message = make_path_message(
                    gt_trajectory,
                    current_timestamp_ns,
                    waypoint_timestamps_ns,
                    odometry.body_frame,
                )
                final_topic = PLANNED_TOPICS[0]
                writer.add_message(
                    channel_id=path_channel_ids[final_topic],
                    log_time=current_timestamp_ns,
                    publish_time=current_timestamp_ns,
                    sequence=stats[f"sequence:{final_topic}"],
                    data=path_encoder(final_message),
                )
                stats[f"sequence:{final_topic}"] += 1
                stats["written_samples"] += 1
                last_odom_index = current_odom_index
        finally:
            progress.close()
            writer.finish()

    print(f"Wrote new MCAP: {output_path}")
    for key, value in sorted(stats.items()):
        if not key.startswith("sequence:"):
            print(f"  {key}: {value}")
    return stats


def path_is_relative_to(path: Path, root: Path) -> bool:
    """Return whether ``path`` is inside ``root`` after both are resolved."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def discover_mcap_files(
    input_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, list[Path]]:
    """Find input MCAPs recursively while ignoring an output subtree."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path exists but is not a directory: {output_dir}"
        )
    if input_dir == output_dir:
        raise ValueError("--input_dir and --output_dir must be different")

    output_is_inside_input = path_is_relative_to(output_dir, input_dir)
    input_paths = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".mcap":
            continue
        if output_is_inside_input and path_is_relative_to(path, output_dir):
            continue
        input_paths.append(path.resolve())
    input_paths.sort()
    if not input_paths:
        raise FileNotFoundError(f"No .mcap files found under {input_dir}")
    return input_dir, output_dir, input_paths


def infer_directory(args: argparse.Namespace) -> None:
    """Process every MCAP below ``args.input_dir`` in deterministic order."""
    input_dir, output_dir, input_paths = discover_mcap_files(
        args.input_dir, args.output_dir
    )
    print(
        f"Found {len(input_paths)} MCAP file(s) under {input_dir}; "
        f"output directory: {output_dir}"
    )
    total_stats: Counter = Counter()
    failures: list[tuple[Path, Exception]] = []

    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if args.worker_threads < 0:
        raise ValueError("--worker-threads must be non-negative")
    requested_workers = args.num_workers
    num_workers = min(requested_workers, len(input_paths))
    using_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if num_workers > 1 and using_cuda:
        raise ValueError(
            "Multiple MCAP workers are disabled on one CUDA device; use "
            "--num-workers 1 and increase --candidate-batch-size instead"
        )

    tasks = []
    for index, input_path in enumerate(input_paths):
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        print(f"\n[{index + 1}/{len(input_paths)}] {relative_path} -> {output_path}")
        tasks.append((input_path, output_path, index))

    if num_workers == 1:
        # Loading the checkpoint once is important in serial directory mode.
        resources = load_inference_resources(args)
        for input_path, output_path, progress_position in tasks:
            file_args = argparse.Namespace(**vars(args))
            file_args.input = input_path
            file_args.output = output_path
            file_args.progress_position = 0
            try:
                total_stats.update(infer_and_write(file_args, resources))
            except Exception as exc:
                failures.append((input_path, exc))
                print(f"Failed: {input_path}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
    else:
        worker_threads = args.worker_threads or max(
            1, (os.cpu_count() or 1) // num_workers
        )
        print(
            f"Parallel MCAP workers: {num_workers}; "
            f"PyTorch threads per worker: {worker_threads}"
        )
        worker_args = vars(args).copy()
        worker_futures = []
        process_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=process_context,
            initializer=initialize_directory_worker,
            initargs=(worker_args, worker_threads),
        ) as executor:
            for input_path, output_path, progress_position in tasks:
                worker_futures.append(
                    (
                        input_path,
                        executor.submit(
                            process_directory_file_worker,
                            input_path,
                            output_path,
                            progress_position,
                        ),
                    )
                )
            for input_path, future in worker_futures:
                try:
                    _, stats = future.result()
                    total_stats.update(stats)
                except Exception as exc:
                    failures.append((input_path, exc))
                    print(f"Failed: {input_path}: {exc}", file=sys.stderr)
                    if not args.continue_on_error:
                        raise

    print("\nDirectory inference summary:")
    print(f"  processed_files: {len(input_paths) - len(failures)}")
    print(f"  failed_files: {len(failures)}")
    for key, value in sorted(total_stats.items()):
        if not key.startswith("sequence:"):
            print(f"  {key}: {value}")
    if failures:
        failed_names = ", ".join(str(path) for path, _ in failures)
        raise RuntimeError(f"Inference failed for: {failed_names}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an MCAP and add seven seeded diffusion trajectories plus "
            "an odometry-aligned ground-truth Path."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Input MCAP")
    input_group.add_argument(
        "--input_dir",
        "--input-dir",
        dest="input_dir",
        type=Path,
        help="Input directory; recursively process all .mcap files",
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path, help="New output MCAP")
    output_group.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        help="Output directory for directory mode; relative paths are preserved",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Model checkpoint; defaults to the latest *.ckpt in --checkpoint-dir",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/resnet50_spatial"),
    )
    parser.add_argument(
        "--config-path", type=Path, default=Path("config/lightning.yaml")
    )
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed; each option and sample derives a different seed",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Minimum seconds between inferred RGB samples; use 0 for every image",
    )
    parser.add_argument("--trajectory-interval", type=float, default=0.1)
    parser.add_argument(
        "--resize-mode",
        choices=("letterbox", "stretch", "center_crop"),
        default="letterbox",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=7,
        help=(
            "Number of seeded trajectory candidates denoised together; "
            "lower this if memory is limited (default: 7)"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Directory-mode MCAP worker processes; use 1 on a single GPU, "
            "2-4 can help on CPU (default: 1)"
        ),
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=0,
        help=(
            "PyTorch CPU threads per parallel worker; 0 splits available "
            "CPU threads automatically"
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In directory mode, continue processing after a failed MCAP",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if (args.input is None) != (args.output is None):
        parser.error("--input and --output must be used together")
    if args.input is not None:
        infer_and_write(args)
    else:
        infer_directory(args)


if __name__ == "__main__":
    main()
