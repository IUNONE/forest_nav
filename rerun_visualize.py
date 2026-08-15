#!/usr/bin/env python3
"""Add predictions and a robot model to ROS 2 MCAP files for Rerun.

The input MCAP is never modified in place. Every original schema, channel,
message, attachment, and metadata record is copied into a new MCAP, with a
``/robot_description`` URDF, its static joint transforms, and eight additional
``sensor_msgs/msg/PointCloud2`` topics written at 10 Hz by default:

* ``/planned_trajectory_final``
* ``/planned_trajectory_opt1`` ... ``/planned_trajectory_opt7``

PointCloud2 is used deliberately: Rerun converts it directly to Points3D,
whereas nav_msgs/Path is currently only available through schema reflection.
The stored points are in the odometry reference frame (``camera_init`` in this
dataset), transformed from the model's +x-forward/+y-left local coordinates.
The bundled URDF is rooted at the recorded ``body`` frame, so Rerun can place
the robot in the same transform tree as the point clouds.

The seven ``opt`` trajectories share the configured fast diffusion iteration
count and use seven deterministic initial-noise seeds.  The ``final`` topic is
an offline oracle result: the same seeds are sampled with the full diffusion
iteration count, and the candidate with the lowest mean displacement error to
the future odometry trajectory is selected.  When valid future odometry is not
available, seed 1 is used as a fallback so output remains close to 10 Hz.

Run ``python rerun_visualize.py view --input OUTPUT.mcap`` from an environment
with a current Rerun SDK to open the robot-centered point-cloud/RGB layout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

FINAL_TOPIC = "/planned_trajectory_final"
OPT_TOPICS = tuple("/planned_trajectory_opt{}".format(i) for i in range(1, 8))
PLANNED_TOPICS = (FINAL_TOPIC,) + OPT_TOPICS
POINTCLOUD_SCHEMA_NAME = "sensor_msgs/msg/PointCloud2"
TF_SCHEMA_NAME = "tf2_msgs/msg/TFMessage"
STRING_SCHEMA_NAME = "std_msgs/msg/String"
STRING_SCHEMA_DATA = b"string data\n"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
DEFAULT_IMAGE_TOPIC = "/robot1/D435i_front/color/image_raw"
DEFAULT_ODOM_TOPIC = "/Odometry"
DEFAULT_ROBOT_FRAME = "body"
DEFAULT_ROBOT_URDF = (
    Path(__file__).resolve().parent
    / "assets"
    / "diablo_original"
    / "urdf"
    / "diablo_forest_nav.urdf"
)
POINTFIELD_FLOAT32 = 7


def load_conversion_dependencies() -> None:
    """Load training/MCAP dependencies only for the conversion command.

    Keeping these imports lazy allows the ``view`` command to run from a small,
    modern Rerun environment instead of the Python 3.8 training environment.
    """

    global np, torch, yaml
    global make_reader, Schema, CompressionType, Writer
    global serialize_dynamic, DecoderFactory
    global OdometrySeries, decode_ros_image, discover_mcap_files
    global load_odometry, local_future_trajectory, message_timestamp, resize_rgb
    global NavDiffsionLightning

    import numpy as np_module
    import torch as torch_module
    import yaml as yaml_module
    from mcap.reader import make_reader as make_reader_function
    from mcap.records import Schema as Schema_class
    from mcap.writer import CompressionType as CompressionType_class
    from mcap.writer import Writer as Writer_class
    from mcap_ros2._dynamic import serialize_dynamic as serialize_dynamic_function
    from mcap_ros2.decoder import DecoderFactory as DecoderFactory_class

    from mcap_to_h5 import (
        OdometrySeries as OdometrySeries_class,
        decode_ros_image as decode_ros_image_function,
        discover_mcap_files as discover_mcap_files_function,
        load_odometry as load_odometry_function,
        local_future_trajectory as local_future_trajectory_function,
        message_timestamp as message_timestamp_function,
        resize_rgb as resize_rgb_function,
    )
    from navdiffusion import NavDiffsionLightning as NavDiffsionLightning_class

    np = np_module
    torch = torch_module
    yaml = yaml_module
    make_reader = make_reader_function
    Schema = Schema_class
    CompressionType = CompressionType_class
    Writer = Writer_class
    serialize_dynamic = serialize_dynamic_function
    DecoderFactory = DecoderFactory_class
    OdometrySeries = OdometrySeries_class
    decode_ros_image = decode_ros_image_function
    discover_mcap_files = discover_mcap_files_function
    load_odometry = load_odometry_function
    local_future_trajectory = local_future_trajectory_function
    message_timestamp = message_timestamp_function
    resize_rgb = resize_rgb_function
    NavDiffsionLightning = NavDiffsionLightning_class


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_seeds(value: str) -> Tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(seeds) != 7:
        raise argparse.ArgumentTypeError("exactly seven seeds are required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("the seven seeds must be unique")
    return seeds


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix != ".ckpt":
            raise ValueError("Checkpoint file must have the .ckpt extension")
        return path
    if not path.is_dir():
        raise FileNotFoundError("Checkpoint path does not exist: {}".format(path))
    checkpoints = list(path.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError("No .ckpt files found in {}".format(path))
    return max(checkpoints, key=lambda checkpoint: checkpoint.stat().st_mtime_ns)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_outputs(input_path: Path, output_path: Path) -> List[Tuple[Path, Path]]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    sources = discover_mcap_files(input_path)

    if input_path.is_file():
        destination = (
            output_path
            if output_path.suffix.lower() == ".mcap"
            else output_path / "{}_planned.mcap".format(input_path.stem)
        )
        return [(input_path, destination)]

    if output_path.suffix.lower() == ".mcap":
        raise ValueError("--output must be a directory when --input is a directory")
    pairs = []
    for source in sources:
        relative = source.relative_to(input_path)
        destination = (
            output_path
            / relative.parent
            / "{}_planned.mcap".format(source.stem)
        )
        pairs.append((source, destination))
    return pairs


def load_model(checkpoint: Path, device: torch.device) -> NavDiffsionLightning:
    print("Loading checkpoint: {}".format(checkpoint), flush=True)
    model = NavDiffsionLightning.load_from_checkpoint(
        str(checkpoint), map_location=device
    )
    model.to(device)
    model.eval()
    model.freeze()
    return model


def load_preprocessing(config_path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
    with config_path.expanduser().open("r") as stream:
        config = yaml.safe_load(stream)
    transform = config["data_params"].get(
        "transform", config["data_params"].get("tranform", {})
    )
    width, height = transform.get("resize", {}).get("w_h", [224, 224])
    if int(width) != int(height):
        raise ValueError("rerun_visualize.py currently requires a square image size")
    norm = transform.get("norm", {})
    mean = np.asarray(norm.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.asarray(norm.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0.0):
        raise ValueError("Image normalization must contain three valid mean/std values")
    return int(width), mean, std


def preprocess_image(
    image_message: object,
    image_size: int,
    resize_mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    image = resize_rgb(
        decode_ros_image(image_message), size=image_size, mode=resize_mode
    ).astype(np.float32)
    image = image / 255.0
    image = (image - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def seeded_noise(
    seeds: Sequence[int],
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    samples = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        samples.append(
            torch.randn(
                (horizon, 2),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    return torch.stack(samples, dim=0)


def denoise_candidates(
    model: NavDiffsionLightning,
    image_condition: torch.Tensor,
    initial_noise: torch.Tensor,
    inference_steps: int,
) -> torch.Tensor:
    sample = initial_noise.clone()
    condition = image_condition.expand(sample.shape[0], -1)
    model.noise_scheduler.set_timesteps(inference_steps, device=sample.device)
    for timestep in model.noise_scheduler.timesteps:
        noise_prediction = model.noise_pred_net(
            sample=sample,
            timestep=timestep,
            global_cond=condition,
        )
        sample = model.noise_scheduler.step(
            model_output=noise_prediction,
            timestep=timestep,
            sample=sample,
        ).prev_sample
    return (
        (sample + 1.0)
        / 2.0
        * (model.traj_norm_higher - model.traj_norm_lower)
        + model.traj_norm_lower
    )


def predict_candidates(
    model: NavDiffsionLightning,
    image: torch.Tensor,
    seeds: Sequence[int],
    opt_steps: int,
    final_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        image_condition = model.transformer_encoder(image)
        noise = seeded_noise(
            seeds=seeds,
            horizon=model.len_traj_pred,
            device=image.device,
            dtype=image_condition.dtype,
        )
        opt_candidates = denoise_candidates(
            model, image_condition, noise, inference_steps=opt_steps
        )
        final_candidates = denoise_candidates(
            model, image_condition, noise, inference_steps=final_steps
        )
        return (
            opt_candidates.detach().cpu().numpy().astype(np.float32),
            final_candidates.detach().cpu().numpy().astype(np.float32),
        )


def pose_at(odometry: OdometrySeries, timestamp: float) -> Optional[Tuple[float, float, float]]:
    if timestamp < odometry.timestamps[0] or timestamp > odometry.timestamps[-1]:
        return None
    x = float(np.interp(timestamp, odometry.timestamps, odometry.positions_xy[:, 0]))
    y = float(np.interp(timestamp, odometry.timestamps, odometry.positions_xy[:, 1]))
    yaw = float(np.interp(timestamp, odometry.timestamps, odometry.yaws))
    if not np.all(np.isfinite([x, y, yaw])):
        return None
    return x, y, yaw


def local_to_reference(
    local_trajectory: np.ndarray, current_pose: Tuple[float, float, float]
) -> np.ndarray:
    current_x, current_y, current_yaw = current_pose
    cos_yaw = math.cos(current_yaw)
    sin_yaw = math.sin(current_yaw)
    global_x = (
        current_x
        + cos_yaw * local_trajectory[:, 0]
        - sin_yaw * local_trajectory[:, 1]
    )
    global_y = (
        current_y
        + sin_yaw * local_trajectory[:, 0]
        + cos_yaw * local_trajectory[:, 1]
    )
    return np.column_stack(
        (global_x, global_y, np.zeros_like(global_x))
    ).astype(np.float32)


def ros_stamp(timestamp: float) -> SimpleNamespace:
    timestamp_ns = int(round(timestamp * 1e9))
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return SimpleNamespace(sec=int(seconds), nanosec=int(nanoseconds))


def pointcloud_message(
    points_xyz: np.ndarray, frame_id: str, timestamp: float
) -> SimpleNamespace:
    points = np.asarray(points_xyz, dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape [N, 3]")
    if not np.all(np.isfinite(points)):
        raise ValueError("Point cloud contains non-finite values")
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=POINTFIELD_FLOAT32, count=1),
        SimpleNamespace(name="y", offset=4, datatype=POINTFIELD_FLOAT32, count=1),
        SimpleNamespace(name="z", offset=8, datatype=POINTFIELD_FLOAT32, count=1),
    ]
    point_step = 12
    return SimpleNamespace(
        header=SimpleNamespace(stamp=ros_stamp(timestamp), frame_id=frame_id),
        height=1,
        width=int(points.shape[0]),
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * int(points.shape[0]),
        data=points.tobytes(order="C"),
        is_dense=True,
    )


def parse_vector_attribute(value: Optional[str], length: int, label: str) -> Tuple[float, ...]:
    text = value if value is not None else " ".join("0" for _ in range(length))
    try:
        parsed = tuple(float(item) for item in text.split())
    except ValueError as exc:
        raise ValueError("Invalid {} in robot URDF: {!r}".format(label, text)) from exc
    if len(parsed) != length or not all(math.isfinite(item) for item in parsed):
        raise ValueError("Invalid {} in robot URDF: {!r}".format(label, text))
    return parsed


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Return the ROS quaternion for a URDF fixed-axis roll/pitch/yaw."""

    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def load_robot_description(
    urdf_path: Path, expected_root_frame: str
) -> Tuple[str, List[Tuple[str, str, Tuple[float, ...], Tuple[float, ...]]]]:
    """Load a URDF, validate its tree, and resolve asset paths for MCAP use."""

    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError("Robot URDF does not exist: {}".format(urdf_path))
    try:
        tree = ElementTree.parse(str(urdf_path))
    except ElementTree.ParseError as exc:
        raise ValueError("Invalid robot URDF {}: {}".format(urdf_path, exc)) from exc
    robot = tree.getroot()
    if robot.tag != "robot":
        raise ValueError("URDF root element must be <robot>: {}".format(urdf_path))

    links = {element.get("name") for element in robot.findall("link")}
    if None in links or not links:
        raise ValueError("URDF contains an unnamed link or no links: {}".format(urdf_path))
    child_links = set()
    transforms = []
    for joint in robot.findall("joint"):
        joint_name = joint.get("name", "<unnamed>")
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError("URDF joint {} has no parent/child".format(joint_name))
        parent = parent_element.get("link")
        child = child_element.get("link")
        if parent not in links or child not in links:
            raise ValueError("URDF joint {} references an unknown link".format(joint_name))
        if child in child_links:
            raise ValueError("URDF link {} has more than one parent".format(child))
        child_links.add(child)
        origin = joint.find("origin")
        xyz = parse_vector_attribute(
            origin.get("xyz") if origin is not None else None,
            3,
            "{} xyz".format(joint_name),
        )
        rpy = parse_vector_attribute(
            origin.get("rpy") if origin is not None else None,
            3,
            "{} rpy".format(joint_name),
        )
        transforms.append((parent, child, xyz, quaternion_from_rpy(*rpy)))

    root_links = links.difference(child_links)
    if root_links != {expected_root_frame}:
        raise ValueError(
            "Robot URDF root must be {!r}, found {}".format(
                expected_root_frame, sorted(root_links)
            )
        )

    for element in list(robot.iter("mesh")) + list(robot.iter("texture")):
        filename = element.get("filename")
        if not filename:
            continue
        if "://" in filename:
            continue
        asset_path = (urdf_path.parent / filename).resolve()
        if not asset_path.is_file():
            raise FileNotFoundError(
                "URDF asset does not exist: {} (from {})".format(asset_path, filename)
            )
        # A robot_description string has no source-directory context. Absolute
        # file URIs make its mesh dependencies resolvable by Rerun on this host.
        element.set("filename", asset_path.as_uri())

    xml_bytes = ElementTree.tostring(robot, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8"), transforms


def transform_stamped_message(
    parent_frame: str,
    child_frame: str,
    translation: Sequence[float],
    quaternion_xyzw: Sequence[float],
    timestamp: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=ros_stamp(timestamp), frame_id=parent_frame),
        child_frame_id=child_frame,
        transform=SimpleNamespace(
            translation=SimpleNamespace(
                x=float(translation[0]),
                y=float(translation[1]),
                z=float(translation[2]),
            ),
            rotation=SimpleNamespace(
                x=float(quaternion_xyzw[0]),
                y=float(quaternion_xyzw[1]),
                z=float(quaternion_xyzw[2]),
                w=float(quaternion_xyzw[3]),
            ),
        ),
    )


def robot_tf_message(
    transforms: Sequence[Tuple[str, str, Sequence[float], Sequence[float]]],
    timestamp: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        transforms=[
            transform_stamped_message(parent, child, xyz, quaternion, timestamp)
            for parent, child, xyz, quaternion in transforms
        ]
    )


def find_schema(summary: object, name: str) -> Schema:
    matches = [schema for schema in summary.schemas.values() if schema.name == name]
    if not matches:
        raise ValueError("Input MCAP does not contain required schema {}".format(name))
    first = matches[0]
    for schema in matches[1:]:
        if schema.encoding != first.encoding or schema.data != first.data:
            raise ValueError("Input MCAP has conflicting definitions for {}".format(name))
    return first


def find_topic_channel(summary: object, topic: str, schema_name: str) -> object:
    matches = []
    for channel in summary.channels.values():
        schema = summary.schemas.get(channel.schema_id)
        if channel.topic == topic and schema is not None and schema.name == schema_name:
            matches.append(channel)
    if not matches:
        raise ValueError(
            "Input MCAP does not contain {} with schema {}".format(topic, schema_name)
        )
    return min(matches, key=lambda channel: channel.id)


def register_original_records(
    writer: Writer, summary: object
) -> Tuple[Dict[int, int], Dict[int, int]]:
    schema_ids: Dict[int, int] = {}
    for old_id, schema in sorted(summary.schemas.items()):
        schema_ids[old_id] = writer.register_schema(
            name=schema.name,
            encoding=schema.encoding,
            data=schema.data,
        )

    channel_ids: Dict[int, int] = {}
    for old_id, channel in sorted(summary.channels.items()):
        channel_ids[old_id] = writer.register_channel(
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            schema_id=schema_ids.get(channel.schema_id, 0),
            metadata=dict(channel.metadata),
        )
    return schema_ids, channel_ids


def generated_qos_metadata(summary: object) -> Dict[str, str]:
    for channel in summary.channels.values():
        if channel.topic == "/path":
            return dict(channel.metadata)
    return {}


def copy_auxiliary_records(reader: object, writer: Writer) -> None:
    for metadata in reader.iter_metadata():
        writer.add_metadata(name=metadata.name, data=dict(metadata.metadata))
    for attachment in reader.iter_attachments():
        writer.add_attachment(
            create_time=attachment.create_time,
            log_time=attachment.log_time,
            name=attachment.name,
            media_type=attachment.media_type,
            data=attachment.data,
        )


def should_sample(timestamp: float, next_timestamp: Optional[float], period: float) -> Tuple[bool, float]:
    if next_timestamp is None:
        return True, timestamp + period
    if timestamp + 1e-9 < next_timestamp:
        return False, next_timestamp
    while next_timestamp <= timestamp + 1e-9:
        next_timestamp += period
    return True, next_timestamp


def augment_one_mcap(
    source: Path,
    destination: Path,
    model: NavDiffsionLightning,
    device: torch.device,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> Counter:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Input MCAP cannot be modified in place")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(
            "Output already exists: {} (use --overwrite)".format(destination)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        if args.overwrite:
            partial.unlink()
        else:
            raise FileExistsError(
                "Partial output already exists: {} (use --overwrite)".format(partial)
            )

    print("Reading odometry: {}".format(source), flush=True)
    odometry = load_odometry(source, args.odom_topic)
    frame_id = odometry.reference_frame
    if not frame_id:
        raise ValueError("Odometry reference frame is empty in {}".format(source))
    robot_description, robot_transforms = load_robot_description(
        args.robot_urdf, expected_root_frame=args.robot_frame
    )

    compression = {
        "none": CompressionType.NONE,
        "lz4": CompressionType.LZ4,
        "zstd": CompressionType.ZSTD,
    }[args.compression]
    stats: Counter = Counter()
    best_seed_counts: Counter = Counter()
    oracle_error_sum = 0.0
    next_inference_timestamp: Optional[float] = None
    decoder_factory = DecoderFactory()

    try:
        with source.open("rb") as input_stream, partial.open("wb") as output_stream:
            reader = make_reader(input_stream)
            summary = reader.get_summary()
            if summary is None:
                raise ValueError("Input MCAP must contain a summary/index")
            existing_topics = {channel.topic for channel in summary.channels.values()}
            duplicates = sorted(existing_topics.intersection(PLANNED_TOPICS))
            if duplicates:
                raise ValueError(
                    "Input already contains planned topics: {}".format(duplicates)
                )
            if args.robot_description_topic in existing_topics:
                raise ValueError(
                    "Input already contains {}".format(args.robot_description_topic)
                )

            pointcloud_schema = find_schema(summary, POINTCLOUD_SCHEMA_NAME)
            encoder_map = serialize_dynamic(
                pointcloud_schema.name, pointcloud_schema.data.decode()
            )
            pointcloud_encoder: Callable[[object], bytes] = encoder_map[
                pointcloud_schema.name
            ]
            tf_schema = find_schema(summary, TF_SCHEMA_NAME)
            tf_encoder_map = serialize_dynamic(tf_schema.name, tf_schema.data.decode())
            tf_encoder: Callable[[object], bytes] = tf_encoder_map[tf_schema.name]
            tf_static_channel = find_topic_channel(
                summary, "/tf_static", TF_SCHEMA_NAME
            )
            string_encoder_map = serialize_dynamic(
                STRING_SCHEMA_NAME, STRING_SCHEMA_DATA.decode()
            )
            string_encoder: Callable[[object], bytes] = string_encoder_map[
                STRING_SCHEMA_NAME
            ]

            header = reader.get_header()
            writer = Writer(
                output=output_stream,
                chunk_size=args.chunk_size_mb * 1024 * 1024,
                compression=compression,
                enable_crcs=True,
            )
            writer.start(
                profile=header.profile or "ros2",
                library="forest_nav/rerun_visualize.py",
            )
            schema_ids, channel_ids = register_original_records(writer, summary)
            qos_metadata = generated_qos_metadata(summary)
            string_schema_id = writer.register_schema(
                name=STRING_SCHEMA_NAME,
                encoding="ros2msg",
                data=STRING_SCHEMA_DATA,
            )
            generated_channel_ids = {
                topic: writer.register_channel(
                    topic=topic,
                    message_encoding="cdr",
                    schema_id=schema_ids[pointcloud_schema.id],
                    metadata=qos_metadata,
                )
                for topic in PLANNED_TOPICS
            }
            robot_description_channel_id = writer.register_channel(
                topic=args.robot_description_topic,
                message_encoding="cdr",
                schema_id=string_schema_id,
                metadata=qos_metadata,
            )
            tf_static_channel_id = channel_ids[tf_static_channel.id]
            copy_auxiliary_records(reader, writer)

            robot_records_written = False
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                if not robot_records_written:
                    record_timestamp = message.log_time / 1e9
                    writer.add_message(
                        channel_id=robot_description_channel_id,
                        log_time=message.log_time,
                        publish_time=message.publish_time,
                        sequence=0,
                        data=string_encoder(SimpleNamespace(data=robot_description)),
                    )
                    writer.add_message(
                        channel_id=tf_static_channel_id,
                        log_time=message.log_time,
                        publish_time=message.publish_time,
                        sequence=0,
                        data=tf_encoder(
                            robot_tf_message(robot_transforms, record_timestamp)
                        ),
                    )
                    stats["robot_description_messages"] += 1
                    stats["robot_static_tf_messages"] += 1
                    robot_records_written = True
                writer.add_message(
                    channel_id=channel_ids[channel.id],
                    log_time=message.log_time,
                    publish_time=message.publish_time,
                    sequence=message.sequence,
                    data=message.data,
                )
                stats["original_messages"] += 1
                if channel.topic != args.image_topic:
                    continue

                decoder = decoder_factory.decoder_for(
                    channel.message_encoding, schema
                )
                if decoder is None:
                    stats["image_decode_unavailable"] += 1
                    continue
                image_message = decoder(message.data)
                image_timestamp = message_timestamp(image_message.header, message.log_time)
                sample, next_inference_timestamp = should_sample(
                    timestamp=image_timestamp,
                    next_timestamp=next_inference_timestamp,
                    period=1.0 / args.inference_hz,
                )
                if not sample:
                    continue
                if args.max_frames and stats["planned_frames"] >= args.max_frames:
                    continue

                current_pose = pose_at(odometry, image_timestamp)
                if current_pose is None:
                    stats["missing_current_odometry"] += 1
                    continue
                try:
                    image_tensor = preprocess_image(
                        image_message=image_message,
                        image_size=image_size,
                        resize_mode=args.resize_mode,
                        mean=mean,
                        std=std,
                        device=device,
                    )
                    opt_candidates, final_candidates = predict_candidates(
                        model=model,
                        image=image_tensor,
                        seeds=args.seeds,
                        opt_steps=args.opt_steps,
                        final_steps=args.final_steps,
                    )
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(
                        "Inference failed at {:.9f}s in {}: {}".format(
                            image_timestamp, source, exc
                        )
                    ) from exc

                ground_truth, ground_truth_reason = local_future_trajectory(
                    odometry=odometry,
                    image_timestamp=image_timestamp,
                    horizon=model.len_traj_pred * args.trajectory_interval,
                    interval=args.trajectory_interval,
                    max_velocity=args.max_velocity,
                    max_odom_gap=args.max_odom_gap,
                )
                if ground_truth is not None:
                    candidate_errors = np.linalg.norm(
                        final_candidates - ground_truth[None, :, :], axis=-1
                    ).mean(axis=-1)
                    best_index = int(np.argmin(candidate_errors))
                    oracle_error_sum += float(candidate_errors[best_index])
                    stats["oracle_selected"] += 1
                else:
                    best_index = 0
                    stats["oracle_fallback_{}".format(ground_truth_reason)] += 1
                best_seed_counts[str(args.seeds[best_index])] += 1

                topic_trajectories = {
                    FINAL_TOPIC: final_candidates[best_index],
                }
                topic_trajectories.update(
                    {
                        topic: opt_candidates[index]
                        for index, topic in enumerate(OPT_TOPICS)
                    }
                )
                for topic in PLANNED_TOPICS:
                    points = local_to_reference(
                        topic_trajectories[topic], current_pose=current_pose
                    )
                    encoded = pointcloud_encoder(
                        pointcloud_message(
                            points_xyz=points,
                            frame_id=frame_id,
                            timestamp=image_timestamp,
                        )
                    )
                    writer.add_message(
                        channel_id=generated_channel_ids[topic],
                        log_time=message.log_time,
                        publish_time=message.publish_time,
                        sequence=stats["planned_frames"],
                        data=encoded,
                    )
                    stats["planned_messages"] += 1

                stats["planned_frames"] += 1
                if stats["planned_frames"] % 100 == 0:
                    print(
                        "  inferred {} frames ({:.1f}s ROS time)".format(
                            stats["planned_frames"], image_timestamp
                        ),
                        flush=True,
                    )

            if not robot_records_written:
                raise ValueError("Input MCAP contains no messages: {}".format(source))
            mean_oracle_error = (
                oracle_error_sum / stats["oracle_selected"]
                if stats["oracle_selected"]
                else math.nan
            )
            writer.add_metadata(
                name="forest_nav_planned_trajectories",
                data={
                    "checkpoint": str(args.resolved_checkpoint),
                    "topics": json.dumps(PLANNED_TOPICS),
                    "message_type": POINTCLOUD_SCHEMA_NAME,
                    "frame_id": frame_id,
                    "frequency_hz": str(args.inference_hz),
                    "opt_steps": str(args.opt_steps),
                    "final_steps": str(args.final_steps),
                    "seeds": json.dumps(args.seeds),
                    "final_selection": "minimum mean L2 error to filtered future odometry",
                    "oracle_mean_error_m": str(mean_oracle_error),
                    "best_seed_counts": json.dumps(best_seed_counts, sort_keys=True),
                    "statistics": json.dumps(stats, sort_keys=True),
                    "robot_urdf": str(args.robot_urdf),
                    "robot_description_topic": args.robot_description_topic,
                    "robot_root_frame": args.robot_frame,
                    "robot_static_joint_count": str(len(robot_transforms)),
                    "rerun_view_target_frame": args.robot_frame,
                },
            )
            writer.finish()

        if destination.exists():
            destination.unlink()
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    print(
        "Output: {} (frames={}, messages={}, oracle={}, fallbacks={})".format(
            destination,
            stats["planned_frames"],
            stats["planned_messages"],
            stats["oracle_selected"],
            stats["planned_frames"] - stats["oracle_selected"],
        ),
        flush=True,
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add forest-nav inference trajectories to ROS 2 MCAP files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="MCAP file or directory tree containing MCAP files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output MCAP file, or output root when input is a directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/resnet50_spatial"),
        help="Checkpoint file or directory; newest checkpoint is selected",
    )
    parser.add_argument("--config", type=Path, default=Path("config/lightning.yaml"))
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=DEFAULT_ROBOT_URDF,
        help="URDF embedded in /robot_description; relative mesh paths are resolved",
    )
    parser.add_argument(
        "--robot-description-topic",
        default=ROBOT_DESCRIPTION_TOPIC,
        help="ROS 2 string topic used by Rerun's URDF MCAP decoder",
    )
    parser.add_argument(
        "--robot-frame",
        default=DEFAULT_ROBOT_FRAME,
        help="Expected URDF root and Rerun target frame",
    )
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--inference-hz", type=positive_float, default=10.0)
    parser.add_argument("--trajectory-interval", type=positive_float, default=0.1)
    parser.add_argument("--opt-steps", type=positive_int, default=4)
    parser.add_argument(
        "--final-steps",
        type=int,
        default=0,
        help="0 uses the inference iteration count stored in the checkpoint",
    )
    parser.add_argument(
        "--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4,5,6")
    )
    parser.add_argument(
        "--resize-mode",
        choices=("center_crop", "letterbox", "stretch"),
        default="letterbox",
    )
    parser.add_argument("--max-velocity", type=positive_float, default=2.0)
    parser.add_argument("--max-odom-gap", type=positive_float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--compression", choices=("zstd", "lz4", "none"), default="zstd"
    )
    parser.add_argument("--chunk-size-mb", type=positive_int, default=8)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum inferred frames per MCAP; 0 means unlimited",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, model: NavDiffsionLightning) -> None:
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if args.final_steps == 0:
        args.final_steps = int(model.num_inference_steps)
    if args.final_steps <= 0:
        raise ValueError("--final-steps must be positive, or 0 for checkpoint default")
    train_steps = int(model.noise_scheduler.config.num_train_timesteps)
    if args.opt_steps > train_steps or args.final_steps > train_steps:
        raise ValueError(
            "Diffusion inference steps cannot exceed {} training steps".format(
                train_steps
            )
        )
    if args.opt_steps >= args.final_steps:
        raise ValueError("--opt-steps must be smaller than --final-steps")


def conversion_main(argv: Optional[Sequence[str]] = None) -> int:
    load_conversion_dependencies()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.resolved_checkpoint = resolve_checkpoint(args.checkpoint)
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        model = load_model(args.resolved_checkpoint, device=device)
        validate_args(args, model=model)
        image_size, mean, std = load_preprocessing(args.config)
        pairs = resolve_outputs(args.input, args.output)
        print(
            "Device: {}; bags: {}; opt_steps: {}; final_steps: {}; seeds: {}".format(
                device,
                len(pairs),
                args.opt_steps,
                args.final_steps,
                args.seeds,
            ),
            flush=True,
        )
        for index, (source, destination) in enumerate(pairs, start=1):
            print(
                "[{}/{}] {} -> {}".format(
                    index, len(pairs), source, destination
                ),
                flush=True,
            )
            augment_one_mcap(
                source=source,
                destination=destination,
                model=model,
                device=device,
                image_size=image_size,
                mean=mean,
                std=std,
                args=args,
            )
    except (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


def rerun_version_tuple(version: str) -> Tuple[int, int, int]:
    values = []
    for component in version.split(".")[:3]:
        digits = "".join(character for character in component if character.isdigit())
        values.append(int(digits) if digits else 0)
    while len(values) < 3:
        values.append(0)
    return tuple(values)  # type: ignore[return-value]


def build_view_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open one augmented MCAP in a robot-centered Rerun layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Augmented MCAP file")
    parser.add_argument("--robot-frame", default=DEFAULT_ROBOT_FRAME)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument(
        "--application-id",
        default="forest_nav_robot_centered",
        help="Rerun application ID for the recording and blueprint",
    )
    parser.add_argument(
        "--blueprint-output",
        type=Path,
        help="Also save the robot-centered blueprint to this .rbl file",
    )
    return parser


def view_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_view_parser()
    args = parser.parse_args(argv)
    mcap_path = args.input.expanduser().resolve()
    if not mcap_path.is_file() or mcap_path.suffix.lower() != ".mcap":
        parser.error("--input must be one existing .mcap file")

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        parser.error(
            "The view command requires rerun-sdk>=0.36 in a Python >=3.10 "
            "environment: {}".format(exc)
        )
    if rerun_version_tuple(rr.__version__) < (0, 36, 0):
        parser.error(
            "rerun-sdk>=0.36 is required for named target frames; found {}".format(
                rr.__version__
            )
        )

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Robot-centered point cloud",
                origin="/",
                contents=["/**"],
                spatial_information=rrb.SpatialInformation(
                    target_frame=args.robot_frame,
                    show_axes=True,
                    show_bounding_box=False,
                ),
                eye_controls=rrb.EyeControls3D(
                    position=(-4.0, -4.0, 3.0),
                    look_target=(0.0, 0.0, 0.0),
                    eye_up=(0.0, 0.0, 1.0),
                    speed=4.0,
                ),
            ),
            rrb.Spatial2DView(
                name="Front RGB",
                origin=args.image_topic,
            ),
        ),
        collapse_panels=True,
    )
    if args.blueprint_output is not None:
        blueprint_path = args.blueprint_output.expanduser().resolve()
        if blueprint_path.suffix.lower() != ".rbl":
            parser.error("--blueprint-output must end in .rbl")
        blueprint_path.parent.mkdir(parents=True, exist_ok=True)
        blueprint.save(args.application_id, str(blueprint_path))
        print("Saved Rerun blueprint: {}".format(blueprint_path), flush=True)

    with rr.RecordingStream(args.application_id) as recording:
        recording.spawn()
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log_file_from_path(mcap_path)
        recording.flush()
    print(
        "Opened {} with target frame {!r}".format(mcap_path, args.robot_frame),
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "view":
        return view_main(arguments[1:])
    return conversion_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
