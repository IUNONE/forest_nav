#!/usr/bin/env python3
"""Convert ROS 2 MCAP recordings into the forest navigation HDF5 format.

Each output group contains one 224x224 RGB image and 32 future local-frame
waypoints at 0.1 second intervals.  A sample is discarded if any odometry
segment between the image and the end of its trajectory exceeds the configured
velocity limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "MCAP support is missing. Install it in the forest_nav environment with:\n"
        "  conda run -n forest_nav pip install mcap mcap-ros2-support"
    ) from exc


DEFAULT_IMAGE_TOPIC = "/robot1/D435i_front/color/image_raw"
DEFAULT_ODOM_TOPIC = "/Odometry"


@dataclass(frozen=True)
class OdometrySeries:
    timestamps: np.ndarray
    positions_xy: np.ndarray
    yaws: np.ndarray
    reference_frame: str
    body_frame: str


def message_timestamp(header: object, log_time_ns: int) -> float:
    """Return the ROS header time, falling back to MCAP log time if unset."""
    stamp = header.stamp
    timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return timestamp if timestamp > 0.0 else float(log_time_ns) * 1e-9


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract the planar yaw angle from a quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        return math.nan
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def discover_mcap_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mcap":
            raise ValueError("Input file must have the .mcap extension")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError("Input path does not exist: {}".format(input_path))

    paths = sorted(input_path.rglob("*.mcap"))
    if not paths:
        raise FileNotFoundError("No .mcap files found below {}".format(input_path))
    return paths


def bag_name(mcap_path: Path) -> str:
    """Use the rosbag directory name, including the file stem when necessary."""
    parent_name = mcap_path.parent.name
    if parent_name and mcap_path.stem.startswith(parent_name + "_"):
        return parent_name
    return mcap_path.stem


def load_odometry(mcap_path: Path, odom_topic: str) -> OdometrySeries:
    timestamps: List[float] = []
    positions: List[Tuple[float, float]] = []
    yaws: List[float] = []
    reference_frame: Optional[str] = None
    body_frame: Optional[str] = None

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, record, odom in reader.iter_decoded_messages(topics=[odom_topic]):
            timestamp = message_timestamp(odom.header, record.log_time)
            position = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            yaw = quaternion_to_yaw(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
            values = (
                timestamp,
                position.x,
                position.y,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
                yaw,
            )
            if not np.all(np.isfinite(values)):
                continue
            message_reference_frame = str(odom.header.frame_id)
            message_body_frame = str(odom.child_frame_id)
            if reference_frame is None:
                reference_frame = message_reference_frame
                body_frame = message_body_frame
            elif (
                message_reference_frame != reference_frame
                or message_body_frame != body_frame
            ):
                raise ValueError(
                    "Odometry frame changed from {} -> {} to {} -> {}".format(
                        reference_frame,
                        body_frame,
                        message_reference_frame,
                        message_body_frame,
                    )
                )
            timestamps.append(timestamp)
            positions.append((position.x, position.y))
            yaws.append(yaw)

    if len(timestamps) < 2:
        raise ValueError(
            "Topic {!r} in {} contains fewer than two valid messages".format(
                odom_topic, mcap_path
            )
        )

    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    positions_array = np.asarray(positions, dtype=np.float64)
    yaws_array = np.asarray(yaws, dtype=np.float64)

    order = np.argsort(timestamps_array, kind="stable")
    timestamps_array = timestamps_array[order]
    positions_array = positions_array[order]
    yaws_array = yaws_array[order]

    # Keep the last message for duplicate timestamps so all intervals are > 0.
    reverse_unique_indices = np.unique(timestamps_array[::-1], return_index=True)[1]
    keep = np.sort(len(timestamps_array) - 1 - reverse_unique_indices)
    timestamps_array = timestamps_array[keep]
    positions_array = positions_array[keep]
    yaws_array = np.unwrap(yaws_array[keep])

    if len(timestamps_array) < 2:
        raise ValueError("Odometry timestamps are not strictly increasing")

    return OdometrySeries(
        timestamps_array,
        positions_array,
        yaws_array,
        reference_frame or "",
        body_frame or "",
    )


def local_future_trajectory(
    odometry: OdometrySeries,
    image_timestamp: float,
    horizon: float,
    interval: float,
    max_velocity: float,
    max_odom_gap: float,
) -> Tuple[Optional[np.ndarray], str]:
    """Interpolate a fixed-rate trajectory and validate every source segment."""
    trajectory_length = int(round(horizon / interval))
    future_times = image_timestamp + interval * np.arange(
        1, trajectory_length + 1, dtype=np.float64
    )
    end_timestamp = float(future_times[-1])
    timestamps = odometry.timestamps

    if image_timestamp < timestamps[0] or end_timestamp > timestamps[-1]:
        return None, "outside_odometry_range"

    # Segment i spans timestamps[i] -> timestamps[i + 1].  Include every
    # segment touched by [image_timestamp, end_timestamp].
    first_segment = max(
        0, int(np.searchsorted(timestamps, image_timestamp, side="right")) - 1
    )
    last_segment = min(
        len(timestamps) - 2,
        int(np.searchsorted(timestamps, end_timestamp, side="left")) - 1,
    )
    if last_segment < first_segment:
        return None, "outside_odometry_range"

    segment_dt = np.diff(timestamps[first_segment : last_segment + 2])
    segment_distance = np.linalg.norm(
        np.diff(odometry.positions_xy[first_segment : last_segment + 2], axis=0),
        axis=1,
    )
    if (
        np.any(~np.isfinite(segment_dt))
        or np.any(segment_dt <= 0.0)
        or np.any(segment_dt > max_odom_gap)
    ):
        return None, "odometry_gap"

    segment_velocity = segment_distance / segment_dt
    if np.any(~np.isfinite(segment_velocity)) or np.any(
        segment_velocity > max_velocity
    ):
        return None, "velocity"

    current_x = np.interp(image_timestamp, timestamps, odometry.positions_xy[:, 0])
    current_y = np.interp(image_timestamp, timestamps, odometry.positions_xy[:, 1])
    current_yaw = np.interp(image_timestamp, timestamps, odometry.yaws)
    future_x = np.interp(future_times, timestamps, odometry.positions_xy[:, 0])
    future_y = np.interp(future_times, timestamps, odometry.positions_xy[:, 1])

    delta_x = future_x - current_x
    delta_y = future_y - current_y
    cos_yaw = math.cos(current_yaw)
    sin_yaw = math.sin(current_yaw)
    trajectory = np.column_stack(
        (
            cos_yaw * delta_x + sin_yaw * delta_y,
            -sin_yaw * delta_x + cos_yaw * delta_y,
        )
    ).astype(np.float32)
    if not np.all(np.isfinite(trajectory)):
        return None, "non_finite_trajectory"
    return trajectory, "valid"


def decode_ros_image(image_message: object) -> np.ndarray:
    """Decode common 8-bit sensor_msgs/Image encodings into RGB."""
    encoding = image_message.encoding.lower()
    encoding_channels = {
        "rgb8": (3, None),
        "bgr8": (3, cv2.COLOR_BGR2RGB),
        "rgba8": (4, cv2.COLOR_RGBA2RGB),
        "bgra8": (4, cv2.COLOR_BGRA2RGB),
        "mono8": (1, cv2.COLOR_GRAY2RGB),
    }
    if encoding not in encoding_channels:
        raise ValueError("Unsupported sensor_msgs/Image encoding: {}".format(encoding))

    channels, conversion = encoding_channels[encoding]
    height = int(image_message.height)
    width = int(image_message.width)
    step = int(image_message.step)
    expected_row_bytes = width * channels
    if height <= 0 or width <= 0 or step < expected_row_bytes:
        raise ValueError(
            "Invalid image dimensions: height={}, width={}, step={}".format(
                height, width, step
            )
        )

    raw = np.frombuffer(image_message.data, dtype=np.uint8)
    expected_size = height * step
    if raw.size < expected_size:
        raise ValueError(
            "Image data is truncated: expected {} bytes, got {}".format(
                expected_size, raw.size
            )
        )
    packed = raw[:expected_size].reshape(height, step)[:, :expected_row_bytes]
    image = packed.reshape(height, width, channels)
    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    return image


def resize_rgb(image: np.ndarray, size: int, mode: str) -> np.ndarray:
    """Resize an RGB image to a square without changing its channel order."""
    height, width = image.shape[:2]
    if mode == "center_crop":
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        image = image[top : top + side, left : left + side]
    elif mode == "letterbox":
        scale = min(float(size) / width, float(size) / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=interpolation
        )
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        top = (size - resized_height) // 2
        left = (size - resized_width) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return canvas
    elif mode != "stretch":
        raise ValueError("Unknown resize mode: {}".format(mode))

    interpolation = cv2.INTER_AREA if min(height, width) > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def image_messages(
    mcap_path: Path, image_topic: str
) -> Iterator[Tuple[int, float, object]]:
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, record, image in reader.iter_decoded_messages(topics=[image_topic]):
            yield record.log_time, message_timestamp(image.header, record.log_time), image


def write_sample(
    output_file: h5py.File,
    group_key: str,
    source_bag: str,
    source_mcap: Path,
    image_timestamp: float,
    image: np.ndarray,
    trajectory: np.ndarray,
    odometry: OdometrySeries,
) -> None:
    group = output_file.create_group(group_key)
    group.attrs["source_bag"] = source_bag
    group.attrs["source_mcap"] = str(source_mcap)
    group.attrs["image_timestamp"] = image_timestamp
    group.attrs["odometry_reference_frame"] = odometry.reference_frame
    group.attrs["trajectory_frame"] = "{}_yaw_aligned".format(odometry.body_frame)
    group.create_dataset(
        "image",
        data=image,
        dtype=np.uint8,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    group.create_dataset("future_waypoint_local", data=trajectory, dtype=np.float32)


def initialise_output(output_file: h5py.File, args: argparse.Namespace) -> None:
    expected = {
        "format": "forest_nav_rgb_trajectory_v2",
        "image_height": args.image_size,
        "image_width": args.image_size,
        "image_channels": 3,
        "trajectory_horizon_s": args.horizon,
        "trajectory_interval_s": args.trajectory_interval,
        "trajectory_points": int(round(args.horizon / args.trajectory_interval)),
        "max_velocity_mps": args.max_velocity,
        "max_odom_gap_s": args.max_odom_gap,
        "observation_sample_interval_s": args.sample_interval,
        "image_topic": args.image_topic,
        "odometry_topic": args.odom_topic,
        "resize_mode": args.resize_mode,
        "trajectory_frame_semantics": "odometry child frame at image time, yaw-only",
        "trajectory_axis_convention": "+x forward, +y left, metres",
    }
    if "format" in output_file.attrs:
        for key, value in expected.items():
            if key not in output_file.attrs or output_file.attrs[key] != value:
                raise ValueError(
                    "Existing output has incompatible {!r}: {!r} != {!r}".format(
                        key, output_file.attrs.get(key), value
                    )
                )
        return
    for key, value in expected.items():
        output_file.attrs[key] = value


def convert(args: argparse.Namespace) -> Counter:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    mcap_paths = discover_mcap_files(input_path)
    if output_path in mcap_paths:
        raise ValueError("Output path cannot also be an input MCAP file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()
    if output_path.exists() and not args.append:
        raise FileExistsError(
            "Output already exists: {} (use --append or --overwrite)".format(
                output_path
            )
        )

    stats: Counter = Counter()
    per_bag_stats = {}
    trajectory_min = np.asarray([math.inf, math.inf], dtype=np.float64)
    trajectory_max = np.asarray([-math.inf, -math.inf], dtype=np.float64)
    file_mode = "a" if args.append else "w"
    with h5py.File(str(output_path), file_mode) as output_file:
        initialise_output(output_file, args)
        existing_keys = set(output_file.keys())
        if "trajectory_min_xy" in output_file.attrs:
            trajectory_min = np.asarray(
                output_file.attrs["trajectory_min_xy"], dtype=np.float64
            )
        if "trajectory_max_xy" in output_file.attrs:
            trajectory_max = np.asarray(
                output_file.attrs["trajectory_max_xy"], dtype=np.float64
            )
        if existing_keys and (
            "trajectory_min_xy" not in output_file.attrs
            or "trajectory_max_xy" not in output_file.attrs
        ):
            for existing_key in existing_keys:
                existing_trajectory = output_file[existing_key][
                    "future_waypoint_local"
                ][:, :2]
                trajectory_min = np.minimum(
                    trajectory_min, existing_trajectory.min(axis=0)
                )
                trajectory_max = np.maximum(
                    trajectory_max, existing_trajectory.max(axis=0)
                )

        for mcap_index, mcap_path in enumerate(mcap_paths, start=1):
            source_bag = bag_name(mcap_path)
            print(
                "[{}/{}] Reading odometry: {}".format(
                    mcap_index, len(mcap_paths), mcap_path
                ),
                flush=True,
            )
            try:
                odometry = load_odometry(mcap_path, args.odom_topic)
            except Exception as exc:
                stats["bag_error"] += 1
                per_bag_stats[str(mcap_path)] = {
                    "bag_error": 1,
                    "error": str(exc),
                }
                print("  Skipping bag: {}".format(exc), file=sys.stderr, flush=True)
                continue

            next_sample_timestamp = -math.inf
            bag_written = 0
            bag_stats: Counter = Counter()
            try:
                messages = image_messages(mcap_path, args.image_topic)
                for _, image_timestamp, image_message in messages:
                    bag_stats["image_messages"] += 1
                    if image_timestamp + 1e-9 < next_sample_timestamp:
                        bag_stats["sample_interval"] += 1
                        continue
                    next_sample_timestamp = image_timestamp + args.sample_interval

                    timestamp_ns = int(round(image_timestamp * 1e9))
                    group_key = "{}_{:019d}".format(source_bag, timestamp_ns)
                    if group_key in existing_keys:
                        bag_stats["already_exists"] += 1
                        continue

                    trajectory, reason = local_future_trajectory(
                        odometry=odometry,
                        image_timestamp=image_timestamp,
                        horizon=args.horizon,
                        interval=args.trajectory_interval,
                        max_velocity=args.max_velocity,
                        max_odom_gap=args.max_odom_gap,
                    )
                    if trajectory is None:
                        bag_stats[reason] += 1
                        continue

                    try:
                        image = resize_rgb(
                            decode_ros_image(image_message),
                            size=args.image_size,
                            mode=args.resize_mode,
                        )
                    except ValueError as exc:
                        bag_stats["invalid_image"] += 1
                        if bag_stats["invalid_image"] == 1:
                            print("  Invalid image: {}".format(exc), file=sys.stderr)
                        continue

                    write_sample(
                        output_file=output_file,
                        group_key=group_key,
                        source_bag=source_bag,
                        source_mcap=mcap_path,
                        image_timestamp=image_timestamp,
                        image=image,
                        trajectory=trajectory,
                        odometry=odometry,
                    )
                    trajectory_min = np.minimum(
                        trajectory_min, trajectory.min(axis=0)
                    )
                    trajectory_max = np.maximum(
                        trajectory_max, trajectory.max(axis=0)
                    )
                    existing_keys.add(group_key)
                    bag_stats["written"] += 1
                    bag_written += 1
                    stats["written"] += 1

                    if args.flush_every and stats["written"] % args.flush_every == 0:
                        output_file.flush()
                    if args.max_samples and stats["written"] >= args.max_samples:
                        break
            except Exception as exc:
                bag_stats["bag_error"] += 1
                print("  Error while reading images: {}".format(exc), file=sys.stderr)

            stats.update(
                {key: value for key, value in bag_stats.items() if key != "written"}
            )
            per_bag_stats[str(mcap_path)] = {
                key: int(value) for key, value in sorted(bag_stats.items())
            }
            print(
                "  wrote {} samples; skipped velocity={}, gaps={}, range={}".format(
                    bag_written,
                    bag_stats["velocity"],
                    bag_stats["odometry_gap"],
                    bag_stats["outside_odometry_range"],
                ),
                flush=True,
            )
            if args.max_samples and stats["written"] >= args.max_samples:
                break

        output_file.attrs["sample_count"] = len(output_file)
        if np.all(np.isfinite(trajectory_min)) and np.all(
            np.isfinite(trajectory_max)
        ):
            output_file.attrs["trajectory_min_xy"] = trajectory_min
            output_file.attrs["trajectory_max_xy"] = trajectory_max
        output_file.attrs["last_conversion_stats_json"] = json.dumps(
            per_bag_stats, ensure_ascii=False, sort_keys=True
        )
        output_file.flush()

    print("Output: {}".format(output_path))
    print("Samples in this run: {}".format(stats["written"]))
    print(
        "Skipped: velocity={}, odometry_gap={}, outside_range={}, invalid_image={}, bag_error={}".format(
            stats["velocity"],
            stats["odometry_gap"],
            stats["outside_odometry_range"],
            stats["invalid_image"],
            stats["bag_error"],
        )
    )
    return stats


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 224x224 RGB observations and fixed-rate local trajectories "
            "from ROS 2 MCAP bags."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="MCAP file, rosbag directory, or root containing rosbag directories",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 file")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--resize-mode",
        choices=("center_crop", "letterbox", "stretch"),
        default="letterbox",
    )
    parser.add_argument(
        "--sample-interval",
        type=positive_float,
        default=0.0333,
        help="Minimum time between extracted RGB observations in seconds",
    )
    parser.add_argument("--horizon", type=positive_float, default=3.2)
    parser.add_argument(
        "--trajectory-interval", type=positive_float, default=0.1
    )
    parser.add_argument("--max-velocity", type=positive_float, default=2.0)
    parser.add_argument(
        "--max-odom-gap",
        type=positive_float,
        default=0.25,
        help="Reject trajectories crossing a larger odometry time gap",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after this many new samples; 0 means unlimited",
    )
    parser.add_argument("--flush-every", type=int, default=100)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--append", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    trajectory_points = args.horizon / args.trajectory_interval
    if not math.isclose(trajectory_points, round(trajectory_points), abs_tol=1e-8):
        raise ValueError("--horizon must be an integer multiple of --trajectory-interval")
    if args.image_size <= 0:
        raise ValueError("--image-size must be greater than zero")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")
    if args.flush_every < 0:
        raise ValueError("--flush-every cannot be negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        convert(args)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
