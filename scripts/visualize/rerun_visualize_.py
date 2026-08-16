import argparse
import bisect
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rosbags.highlevel import AnyReader


NANOSECONDS_PER_SECOND = 1_000_000_000
COLOR_HISTORY = np.array([66, 165, 245], dtype=np.uint8)
COLOR_FUTURE = np.array([255, 167, 38], dtype=np.uint8)
COLOR_CURRENT = np.array([239, 83, 80], dtype=np.uint8)
COLOR_FULL_PATH = np.array([120, 132, 158], dtype=np.uint8)
COLOR_VX = np.array([41, 182, 246], dtype=np.uint8)
COLOR_VY = np.array([255, 112, 67], dtype=np.uint8)


@dataclass(frozen=True)
class TopicSelection:
    """保存参与可视化的 ROS topic 选择结果."""

    pointcloud: str
    rgb: str
    depth: str
    odometry: str
    color_camera_info: str | None
    tf: str | None
    tf_static: str | None


@dataclass(frozen=True)
class CameraCalibration:
    """保存相机内参及图像坐标系信息."""

    frame_id: str
    width: int
    height: int
    image_from_camera: np.ndarray
    synthetic: bool


@dataclass(frozen=True)
class TransformRecord:
    """保存一条已转换为矩阵形式的 TF 记录."""

    timestamp_ns: int
    parent: str
    child: str
    parent_from_child: np.ndarray
    is_static: bool


@dataclass(frozen=True)
class ImageDescription:
    """保存 ROS 图像消息的基本编码信息."""

    frame_id: str
    width: int
    height: int
    encoding: str


@dataclass(frozen=True)
class PointCloudDescription:
    """保存 ROS 点云消息的基本字段信息."""

    frame_id: str
    fields: tuple[str, ...]
    point_step: int
    point_count: int


@dataclass(frozen=True)
class OdometrySeries:
    """保存按记录时间排序的里程计轨迹和速度."""

    timestamps_ns: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    root_frame: str
    child_frame: str
    twist_velocities_xy: np.ndarray
    display_velocities_xy: np.ndarray
    velocity_source: str


@dataclass(frozen=True)
class BagInspection:
    """保存预扫描得到的 bag 结构和数据质量信息."""

    bag_path: Path
    start_ns: int
    end_ns: int
    topics: TopicSelection
    topic_types: dict[str, str]
    topic_counts: dict[str, int]
    sampled_frames: dict[str, str]
    transforms: tuple[TransformRecord, ...]
    odometry: OdometrySeries
    calibration: CameraCalibration
    rgb_description: ImageDescription
    depth_description: ImageDescription
    pointcloud_description: PointCloudDescription


class FrameResolver:
    """维护规范化后的静态和动态 TF 图并执行时刻查询."""

    def __init__(self) -> None:
        """初始化空的 TF 图."""
        self._static_edges: dict[tuple[str, str], np.ndarray] = {}
        self._dynamic_records: dict[
            tuple[str, str], list[tuple[int, np.ndarray]]
        ] = defaultdict(list)
        self._dynamic_edges: dict[
            tuple[str, str], tuple[np.ndarray, np.ndarray]
        ] = {}

    def add_static(
        self,
        parent: str,
        child: str,
        parent_from_child: np.ndarray,
    ) -> None:
        """添加一条静态 TF 并拒绝自环和冲突定义."""
        if parent == child:
            raise ValueError(f"静态 TF 形成自环: {parent} -> {child}")
        key = (parent, child)
        existing = self._static_edges.get(key)
        if existing is not None and not np.allclose(
            existing, parent_from_child, atol=1e-6
        ):
            raise ValueError(f"静态 TF 存在冲突定义: {parent} -> {child}")
        self._static_edges[key] = np.asarray(parent_from_child, dtype=np.float64)

    def add_dynamic(
        self,
        timestamp_ns: int,
        parent: str,
        child: str,
        parent_from_child: np.ndarray,
    ) -> None:
        """添加一条带记录时间的动态 TF."""
        if parent == child:
            raise ValueError(f"动态 TF 形成自环: {parent} -> {child}")
        self._dynamic_records[(parent, child)].append(
            (timestamp_ns, np.asarray(parent_from_child, dtype=np.float64))
        )

    def finalize(self) -> None:
        """排序动态 TF 并构建适合二分查询的连续数组."""
        self._dynamic_edges.clear()
        for edge, records in self._dynamic_records.items():
            records.sort(key=lambda item: item[0])
            timestamps = np.asarray([item[0] for item in records], dtype=np.int64)
            matrices = np.stack([item[1] for item in records])
            self._dynamic_edges[edge] = timestamps, matrices

    def edge_keys(self) -> set[tuple[str, str]]:
        """返回当前 TF 图中的有向边集合."""
        return set(self._static_edges) | set(self._dynamic_records)

    def static_edge_keys(self) -> set[tuple[str, str]]:
        """返回当前 TF 图中的静态有向边集合."""
        return set(self._static_edges)

    def lookup(
        self,
        target: str,
        source: str,
        timestamp_ns: int,
    ) -> np.ndarray | None:
        """查询指定时刻从 source 坐标到 target 坐标的变换矩阵."""
        if target == source:
            return np.eye(4, dtype=np.float64)
        adjacency: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for (parent, child), matrix in self._static_edges.items():
            adjacency[child].append((parent, matrix))
            adjacency[parent].append((child, rigid_inverse(matrix)))
        for (parent, child), series in self._dynamic_edges.items():
            matrix = sample_transform(series, timestamp_ns)
            adjacency[child].append((parent, matrix))
            adjacency[parent].append((child, rigid_inverse(matrix)))
        queue: deque[tuple[str, np.ndarray]] = deque(
            [(source, np.eye(4, dtype=np.float64))]
        )
        visited = {source}
        while queue:
            current, current_from_source = queue.popleft()
            for neighbor, neighbor_from_current in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                neighbor_from_source = neighbor_from_current @ current_from_source
                if neighbor == target:
                    return neighbor_from_source
                visited.add(neighbor)
                queue.append((neighbor, neighbor_from_source))
        return None

    def components(self) -> list[set[str]]:
        """返回忽略边方向后的 TF 连通分量."""
        adjacency: dict[str, set[str]] = defaultdict(set)
        for parent, child in self.edge_keys():
            adjacency[parent].add(child)
            adjacency[child].add(parent)
        components: list[set[str]] = []
        unseen = set(adjacency)
        while unseen:
            seed = min(unseen)
            component = {seed}
            queue = deque([seed])
            unseen.remove(seed)
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return sorted(components, key=lambda item: (-len(item), sorted(item)))


def clean_frame_id(frame_id: str) -> str:
    """清理 frame id 的空白和多余路径分隔符."""
    stripped = str(frame_id).strip()
    return "/".join(part for part in stripped.split("/") if part)


def canonical_frame_id(frame_id: str, aliases: dict[str, str]) -> str:
    """递归应用 frame alias 并检测 alias 环."""
    current = clean_frame_id(frame_id)
    visited: set[str] = set()
    while current in aliases:
        if current in visited:
            cycle = " -> ".join([*sorted(visited), current])
            raise ValueError(f"frame alias 存在环: {cycle}")
        visited.add(current)
        current = clean_frame_id(aliases[current])
    return current


def parse_frame_aliases(values: Sequence[str]) -> dict[str, str]:
    """解析 source=target 形式的 frame alias 参数."""
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"无效 frame alias: {value}. 应使用 source=target")
        source, target = value.split("=", maxsplit=1)
        source = clean_frame_id(source)
        target = clean_frame_id(target)
        if not source or not target:
            raise ValueError(f"无效 frame alias: {value}")
        aliases[source] = target
    for source in tuple(aliases):
        canonical_frame_id(source, aliases)
    return aliases


def quaternion_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """将 xyzw 四元数转换为齐次旋转矩阵."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"无效四元数: {quaternion.tolist()}")
    x, y, z, w = quaternion / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return matrix


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵稳定地转换为 xyzw 四元数."""
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def make_transform(
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """根据平移和四元数构造齐次变换矩阵."""
    matrix = quaternion_to_matrix(quaternion_xyzw)
    matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return matrix


def rigid_inverse(matrix: np.ndarray) -> np.ndarray:
    """计算刚体齐次变换的高效逆矩阵."""
    matrix = np.asarray(matrix, dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def sample_transform(
    series: tuple[np.ndarray, np.ndarray],
    timestamp_ns: int,
) -> np.ndarray:
    """使用不晚于查询时刻的最近 TF 并处理边界外推."""
    timestamps, matrices = series
    index = int(np.searchsorted(timestamps, timestamp_ns, side="right") - 1)
    index = min(max(index, 0), len(timestamps) - 1)
    return matrices[index]


def ros_time_to_ns(stamp: Any) -> int:
    """将 ROS builtin time 转换为纳秒整数."""
    return int(stamp.sec) * NANOSECONDS_PER_SECOND + int(stamp.nanosec)


def transform_from_ros_message(transform: Any) -> np.ndarray:
    """将 geometry Transform 消息转换为齐次矩阵."""
    translation = transform.translation
    rotation = transform.rotation
    return make_transform(
        [translation.x, translation.y, translation.z],
        [rotation.x, rotation.y, rotation.z, rotation.w],
    )


def pose_from_ros_message(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    """从 geometry Pose 消息提取位置和 xyzw 四元数."""
    position = np.array(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    quaternion = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quaternion)
    if not math.isfinite(float(norm)) or norm < 1e-12:
        raise ValueError("Odometry 包含无效姿态四元数")
    return position, quaternion / norm


def smooth_columns(values: np.ndarray, window: int = 5) -> np.ndarray:
    """使用边缘保持的移动平均平滑多列时间序列."""
    if len(values) < 3 or window <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    width = min(window, len(values))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    padding = width // 2
    kernel = np.ones(width, dtype=np.float64) / width
    result = np.empty_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        padded = np.pad(values[:, column], padding, mode="edge")
        result[:, column] = np.convolve(padded, kernel, mode="valid")
    return result


def build_odometry_series(records: list[tuple[int, Any]]) -> OdometrySeries:
    """构建轨迹数组并在 twist 无效时从位姿推导速度."""
    if len(records) < 2:
        raise ValueError("至少需要两条 Odometry 消息才能构建轨迹")
    records.sort(key=lambda item: item[0])
    timestamps: list[int] = []
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    twists: list[list[float]] = []
    root_frames: Counter[str] = Counter()
    child_frames: Counter[str] = Counter()
    for timestamp_ns, message in records:
        position, quaternion = pose_from_ros_message(message.pose.pose)
        timestamps.append(timestamp_ns)
        positions.append(position)
        quaternions.append(quaternion)
        twists.append(
            [message.twist.twist.linear.x, message.twist.twist.linear.y]
        )
        root_frames[clean_frame_id(message.header.frame_id)] += 1
        child_frames[clean_frame_id(message.child_frame_id)] += 1
    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    position_array = np.stack(positions)
    quaternion_array = np.stack(quaternions)
    twist_array = np.asarray(twists, dtype=np.float64)
    elapsed = (timestamp_array - timestamp_array[0]).astype(np.float64) / 1e9
    displacement = float(np.linalg.norm(position_array[-1, :2] - position_array[0, :2]))
    twist_peak = float(np.max(np.linalg.norm(twist_array, axis=1)))
    if displacement > 0.02 and twist_peak < 1e-5:
        derived = np.column_stack(
            [
                np.gradient(position_array[:, 0], elapsed),
                np.gradient(position_array[:, 1], elapsed),
            ]
        )
        display_velocity = smooth_columns(derived, window=5)
        velocity_source = "pose-derived world frame"
    else:
        display_velocity = twist_array
        velocity_source = "odometry twist"
    return OdometrySeries(
        timestamps_ns=timestamp_array,
        positions=position_array,
        quaternions_xyzw=quaternion_array,
        root_frame=root_frames.most_common(1)[0][0],
        child_frame=child_frames.most_common(1)[0][0],
        twist_velocities_xy=twist_array,
        display_velocities_xy=display_velocity,
        velocity_source=velocity_source,
    )


def score_topic(topic: str, preferred_tokens: Sequence[str]) -> int:
    """根据名称 token 对候选 topic 进行确定性评分."""
    lowered = topic.lower()
    score = 0
    for index, token in enumerate(preferred_tokens):
        if token.lower() in lowered:
            score += 100 - index
    return score


def choose_topic(
    candidates: Sequence[str],
    explicit: str | None,
    label: str,
    preferred_tokens: Sequence[str],
) -> str:
    """选择显式 topic 或评分最高的自动候选."""
    if explicit is not None:
        if explicit not in candidates:
            available = ", ".join(candidates) or "无"
            raise ValueError(f"{label} topic 不存在: {explicit}. 候选: {available}")
        return explicit
    if not candidates:
        raise ValueError(f"bag 中未找到 {label} topic")
    return max(
        sorted(candidates),
        key=lambda topic: score_topic(topic, preferred_tokens),
    )


def choose_optional_topic(
    candidates: Sequence[str],
    explicit: str | None,
    preferred_tokens: Sequence[str],
) -> str | None:
    """选择可缺省的 topic 并验证显式参数."""
    if explicit is not None:
        if explicit not in candidates:
            available = ", ".join(candidates) or "无"
            raise ValueError(f"topic 不存在: {explicit}. 候选: {available}")
        return explicit
    if not candidates:
        return None
    return max(
        sorted(candidates),
        key=lambda topic: score_topic(topic, preferred_tokens),
    )


def discover_topics(
    connections: Sequence[Any],
    arguments: argparse.Namespace,
) -> TopicSelection:
    """根据 ROS 消息类型和名称发现可视化 topic."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for connection in connections:
        by_type[connection.msgtype].append(connection.topic)
    pointcloud = choose_topic(
        by_type["sensor_msgs/msg/PointCloud2"],
        arguments.pointcloud_topic,
        "PointCloud2",
        [
            "/livox/lidar_pointcloud2",
            "livox/lidar_pointcloud2",
            "/cloud_registered",
            "registered",
            "laser_map",
        ],
    )
    image_topics = by_type["sensor_msgs/msg/Image"]
    rgb_candidates = [
        topic
        for topic in image_topics
        if "depth" not in topic.lower()
    ]
    depth_candidates = [
        topic
        for topic in image_topics
        if "depth" in topic.lower()
    ]
    rgb = choose_topic(
        rgb_candidates,
        arguments.rgb_topic,
        "RGB image",
        ["color/image_raw", "rgb", "color", "image_raw"],
    )
    depth = choose_topic(
        depth_candidates,
        arguments.depth_topic,
        "depth image",
        ["depth/image_rect_raw", "depth/image_raw", "depth"],
    )
    odometry = choose_topic(
        by_type["nav_msgs/msg/Odometry"],
        arguments.odom_topic,
        "Odometry",
        ["/odometry", "/odom"],
    )
    camera_infos = by_type["sensor_msgs/msg/CameraInfo"]
    color_info_candidates = [
        topic
        for topic in camera_infos
        if "depth" not in topic.lower()
    ]
    color_camera_info = choose_optional_topic(
        color_info_candidates,
        arguments.color_info_topic,
        ["color/camera_info", "rgb", "color"],
    )
    tf = choose_optional_topic(
        by_type["tf2_msgs/msg/TFMessage"],
        arguments.tf_topic,
        ["/tf"],
    )
    tf_static_candidates = [
        topic
        for topic in by_type["tf2_msgs/msg/TFMessage"]
        if "static" in topic.lower()
    ]
    if tf is not None and "static" in tf.lower():
        dynamic_candidates = [
            topic
            for topic in by_type["tf2_msgs/msg/TFMessage"]
            if "static" not in topic.lower()
        ]
        tf = choose_optional_topic(dynamic_candidates, arguments.tf_topic, ["/tf"])
    tf_static = choose_optional_topic(
        tf_static_candidates,
        arguments.tf_static_topic,
        ["/tf_static"],
    )
    return TopicSelection(
        pointcloud=pointcloud,
        rgb=rgb,
        depth=depth,
        odometry=odometry,
        color_camera_info=color_camera_info,
        tf=tf,
        tf_static=tf_static,
    )


def first_message(reader: AnyReader, connection: Any) -> tuple[int, Any]:
    """读取并反序列化指定 connection 的第一条消息."""
    item = next(reader.messages(connections=[connection]), None)
    if item is None:
        raise ValueError(f"topic 没有消息: {connection.topic}")
    _, timestamp_ns, rawdata = item
    return timestamp_ns, reader.deserialize(rawdata, connection.msgtype)


def camera_calibration_from_message(message: Any) -> CameraCalibration:
    """从 CameraInfo 消息提取 Rerun Pinhole 所需内参."""
    intrinsic = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("CameraInfo 包含无效焦距")
    return CameraCalibration(
        frame_id=clean_frame_id(message.header.frame_id),
        width=int(message.width),
        height=int(message.height),
        image_from_camera=intrinsic,
        synthetic=False,
    )


def synthetic_camera_calibration(message: Any) -> CameraCalibration:
    """在缺少 CameraInfo 时生成九十度水平视场的近似内参."""
    width = int(message.width)
    height = int(message.height)
    focal = width / 2.0
    intrinsic = np.array(
        [
            [focal, 0.0, (width - 1.0) / 2.0],
            [0.0, focal, (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CameraCalibration(
        frame_id=clean_frame_id(message.header.frame_id),
        width=width,
        height=height,
        image_from_camera=intrinsic,
        synthetic=True,
    )


def inspect_bag(path: Path, arguments: argparse.Namespace) -> BagInspection:
    """预扫描 bag 并提取 topic, TF, 相机和里程计信息."""
    if not path.exists():
        raise FileNotFoundError(f"bag 路径不存在: {path}")
    with AnyReader([path]) as reader:
        topics = discover_topics(reader.connections, arguments)
        connection_by_topic = {
            connection.topic: connection for connection in reader.connections
        }
        topic_types = {
            connection.topic: connection.msgtype for connection in reader.connections
        }
        topic_counts = {
            connection.topic: int(connection.msgcount)
            for connection in reader.connections
        }
        sampled_frames: dict[str, str] = {}
        sample_topics = {
            connection.topic
            for connection in reader.connections
            if connection.msgtype
            in {
                "sensor_msgs/msg/PointCloud2",
                "sensor_msgs/msg/Image",
                "sensor_msgs/msg/CameraInfo",
                "sensor_msgs/msg/Imu",
            }
            and connection.msgcount > 0
        }
        samples: dict[str, Any] = {}
        sample_timestamps: dict[str, int] = {}
        for topic in sorted(sample_topics):
            timestamp_ns, message = first_message(reader, connection_by_topic[topic])
            samples[topic] = message
            sample_timestamps[topic] = timestamp_ns
            sampled_frames[topic] = clean_frame_id(message.header.frame_id)
        rgb_message = samples[topics.rgb]
        depth_message = samples[topics.depth]
        pointcloud_message = samples[topics.pointcloud]
        if topics.color_camera_info is not None:
            calibration = camera_calibration_from_message(
                samples[topics.color_camera_info]
            )
        else:
            calibration = synthetic_camera_calibration(rgb_message)
        transform_records: list[TransformRecord] = []
        odometry_records: list[tuple[int, Any]] = []
        full_scan_topics = {topics.odometry}
        if topics.tf is not None:
            full_scan_topics.add(topics.tf)
        if topics.tf_static is not None:
            full_scan_topics.add(topics.tf_static)
        full_scan_connections = [
            connection_by_topic[topic] for topic in full_scan_topics
        ]
        for connection, timestamp_ns, rawdata in reader.messages(
            connections=full_scan_connections
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == topics.odometry:
                odometry_records.append((timestamp_ns, message))
                continue
            is_static = connection.topic == topics.tf_static
            for transform in message.transforms:
                transform_records.append(
                    TransformRecord(
                        timestamp_ns=timestamp_ns,
                        parent=clean_frame_id(transform.header.frame_id),
                        child=clean_frame_id(transform.child_frame_id),
                        parent_from_child=transform_from_ros_message(
                            transform.transform
                        ),
                        is_static=is_static,
                    )
                )
        odometry = build_odometry_series(odometry_records)
        visualization_start_ns = min(
            sample_timestamps[topics.pointcloud],
            sample_timestamps[topics.rgb],
            sample_timestamps[topics.depth],
            int(odometry.timestamps_ns[0]),
        )
        return BagInspection(
            bag_path=path,
            start_ns=visualization_start_ns,
            end_ns=int(reader.end_time),
            topics=topics,
            topic_types=topic_types,
            topic_counts=topic_counts,
            sampled_frames=sampled_frames,
            transforms=tuple(transform_records),
            odometry=odometry,
            calibration=calibration,
            rgb_description=ImageDescription(
                frame_id=clean_frame_id(rgb_message.header.frame_id),
                width=int(rgb_message.width),
                height=int(rgb_message.height),
                encoding=str(rgb_message.encoding),
            ),
            depth_description=ImageDescription(
                frame_id=clean_frame_id(depth_message.header.frame_id),
                width=int(depth_message.width),
                height=int(depth_message.height),
                encoding=str(depth_message.encoding),
            ),
            pointcloud_description=PointCloudDescription(
                frame_id=clean_frame_id(pointcloud_message.header.frame_id),
                fields=tuple(field.name for field in pointcloud_message.fields),
                point_step=int(pointcloud_message.point_step),
                point_count=int(pointcloud_message.width)
                * int(pointcloud_message.height),
            ),
        )


def automatic_aliases(
    inspection: BagInspection,
    explicit_aliases: dict[str, str],
) -> dict[str, str]:
    """为常见的孤立机器人主体 frame 生成保守 alias."""
    aliases = dict(explicit_aliases)
    observed_frames = set(inspection.sampled_frames.values())
    graph_frames = {
        frame
        for transform in inspection.transforms
        for frame in (transform.parent, transform.child)
    }
    graph_frames.update(
        [inspection.odometry.root_frame, inspection.odometry.child_frame]
    )
    child = inspection.odometry.child_frame
    for candidate in ("base_link", "base_footprint", "livox_frame"):
        if (
            candidate in observed_frames
            and candidate not in graph_frames
            and candidate != child
            and candidate not in aliases
        ):
            aliases[candidate] = child
    for source in tuple(aliases):
        canonical_frame_id(source, aliases)
    return aliases


def build_frame_resolver(
    inspection: BagInspection,
    aliases: dict[str, str],
    arguments: argparse.Namespace,
) -> tuple[FrameResolver, str | None, tuple[str, str] | None]:
    """构建 TF 解析器并按需添加相机到机器人主体的桥接."""
    resolver = FrameResolver()
    for transform in inspection.transforms:
        parent = canonical_frame_id(transform.parent, aliases)
        child = canonical_frame_id(transform.child, aliases)
        if parent == child:
            continue
        if transform.is_static:
            resolver.add_static(parent, child, transform.parent_from_child)
        else:
            resolver.add_dynamic(
                transform.timestamp_ns,
                parent,
                child,
                transform.parent_from_child,
            )
    odometry = inspection.odometry
    root = canonical_frame_id(odometry.root_frame, aliases)
    child = canonical_frame_id(odometry.child_frame, aliases)
    if (root, child) not in resolver.edge_keys():
        for timestamp_ns, position, quaternion in zip(
            odometry.timestamps_ns,
            odometry.positions,
            odometry.quaternions_xyzw,
        ):
            resolver.add_dynamic(
                int(timestamp_ns),
                root,
                child,
                make_transform(position, quaternion),
            )
    resolver.finalize()
    camera_frame = canonical_frame_id(inspection.calibration.frame_id, aliases)
    synthetic_bridge: tuple[str, str] | None = None
    if resolver.lookup(root, camera_frame, inspection.start_ns) is None:
        if arguments.strict_tf:
            raise ValueError(
                f"相机 frame {camera_frame} 与根 frame {root} 不连通"
            )
        camera_component = next(
            (
                component
                for component in resolver.components()
                if camera_frame in component
            ),
            {camera_frame},
        )
        static_parents = {
            child_frame: parent_frame
            for parent_frame, child_frame in resolver.static_edge_keys()
            if child_frame in camera_component and parent_frame in camera_component
        }
        camera_root = camera_frame
        visited: set[str] = set()
        while camera_root in static_parents:
            if camera_root in visited:
                raise ValueError("相机静态 TF 链包含环")
            visited.add(camera_root)
            camera_root = static_parents[camera_root]
        camera_parent = canonical_frame_id(
            arguments.camera_parent or child,
            aliases,
        )
        extrinsic_values = arguments.camera_extrinsic or [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        bridge_matrix = make_transform(
            extrinsic_values[:3], extrinsic_values[3:]
        )
        resolver.add_static(camera_parent, camera_root, bridge_matrix)
        resolver.finalize()
        synthetic_bridge = camera_parent, camera_root
    return resolver, camera_frame, synthetic_bridge


def point_field_dtype(datatype: int, big_endian: bool) -> np.dtype[Any]:
    """将 PointField datatype 映射为带字节序的 NumPy dtype."""
    dtype_by_code = {
        1: np.dtype("i1"),
        2: np.dtype("u1"),
        3: np.dtype("i2"),
        4: np.dtype("u2"),
        5: np.dtype("i4"),
        6: np.dtype("u4"),
        7: np.dtype("f4"),
        8: np.dtype("f8"),
    }
    if datatype not in dtype_by_code:
        raise ValueError(f"不支持的 PointField datatype: {datatype}")
    byte_order = ">" if big_endian else "<"
    return dtype_by_code[datatype].newbyteorder(byte_order)


def structured_pointcloud(message: Any) -> np.ndarray:
    """将 PointCloud2 数据映射为保留字段 offset 的结构化数组."""
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    for field in message.fields:
        if not field.name or field.name in names:
            continue
        dtype = point_field_dtype(int(field.datatype), bool(message.is_bigendian))
        count = int(field.count)
        names.append(str(field.name))
        formats.append(dtype if count == 1 else (dtype, (count,)))
        offsets.append(int(field.offset))
    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(message.point_step),
        }
    )
    data = np.ascontiguousarray(message.data, dtype=np.uint8)
    expected_bytes = int(message.row_step) * int(message.height)
    if data.nbytes < expected_bytes:
        raise ValueError(
            f"PointCloud2 data 过短: {data.nbytes} < {expected_bytes}"
        )
    cloud = np.ndarray(
        shape=(int(message.height), int(message.width)),
        dtype=dtype,
        buffer=data,
        strides=(int(message.row_step), int(message.point_step)),
    )
    return cloud.reshape(-1)


def scalar_colormap(values: np.ndarray) -> np.ndarray:
    """将标量值映射为高对比度的连续 RGB 颜色."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        normalized = np.zeros_like(values)
    else:
        lower, upper = np.percentile(finite, [1.0, 99.0])
        if upper - lower < 1e-9:
            lower = float(np.min(finite))
            upper = lower + 1.0
        normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    anchors = np.array(
        [
            [48, 18, 112],
            [31, 119, 180],
            [38, 188, 143],
            [242, 208, 65],
            [213, 62, 79],
        ],
        dtype=np.float64,
    )
    scaled = normalized * (len(anchors) - 1)
    left = np.floor(scaled).astype(np.int64)
    right = np.minimum(left + 1, len(anchors) - 1)
    fraction = (scaled - left)[:, None]
    colors = anchors[left] * (1.0 - fraction) + anchors[right] * fraction
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def packed_color(values: np.ndarray) -> np.ndarray:
    """将 PCL packed rgb 或 rgba 字段转换为 uint8 RGB."""
    contiguous = np.ascontiguousarray(values)
    if contiguous.dtype.kind == "f" and contiguous.dtype.itemsize == 4:
        packed = contiguous.view(np.uint32)
    else:
        packed = contiguous.astype(np.uint32, copy=False)
    red = ((packed >> 16) & 0xFF).astype(np.uint8)
    green = ((packed >> 8) & 0xFF).astype(np.uint8)
    blue = (packed & 0xFF).astype(np.uint8)
    return np.column_stack([red, green, blue])


def decode_pointcloud(
    message: Any,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """解码 PointCloud2 并过滤无效点及执行确定性降采样."""
    cloud = structured_pointcloud(message)
    required = {"x", "y", "z"}
    if not required.issubset(cloud.dtype.names or ()):
        raise ValueError("PointCloud2 缺少 x, y, z 字段")
    positions = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(
        np.float32,
        copy=False,
    )
    valid_indices = np.flatnonzero(np.isfinite(positions).all(axis=1))[::stride]
    positions = positions[valid_indices]
    names = set(cloud.dtype.names or ())
    if "rgb" in names:
        colors = packed_color(cloud["rgb"])[valid_indices]
    elif "rgba" in names:
        colors = packed_color(cloud["rgba"])[valid_indices]
    elif "intensity" in names:
        colors = scalar_colormap(
            np.asarray(cloud["intensity"])[valid_indices]
        )
    else:
        colors = scalar_colormap(positions[:, 2])
    return positions, colors


def transform_points(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """将三维点批量变换到目标坐标系."""
    rotation = target_from_source[:3, :3]
    translation = target_from_source[:3, 3]
    transformed = np.einsum(
        "nj,ij->ni",
        points,
        rotation,
        optimize=False,
    ) + translation
    if not np.isfinite(transformed).all():
        raise ValueError("点云坐标变换产生非有限数值")
    return transformed


def image_encoding_spec(encoding: str) -> tuple[np.dtype[Any], int, str]:
    """返回 ROS Image 编码对应的 dtype, 通道数和颜色顺序."""
    normalized = encoding.strip().lower()
    specifications: dict[str, tuple[np.dtype[Any], int, str]] = {
        "rgb8": (np.dtype("u1"), 3, "rgb"),
        "bgr8": (np.dtype("u1"), 3, "bgr"),
        "rgba8": (np.dtype("u1"), 4, "rgba"),
        "bgra8": (np.dtype("u1"), 4, "bgra"),
        "mono8": (np.dtype("u1"), 1, "mono"),
        "8uc1": (np.dtype("u1"), 1, "mono"),
        "8uc3": (np.dtype("u1"), 3, "rgb"),
        "mono16": (np.dtype("u2"), 1, "mono"),
        "16uc1": (np.dtype("u2"), 1, "depth_mm"),
        "32fc1": (np.dtype("f4"), 1, "depth_m"),
    }
    if normalized not in specifications:
        supported = ", ".join(sorted(specifications))
        raise ValueError(
            f"不支持的 ROS Image encoding: {encoding}. 支持: {supported}"
        )
    return specifications[normalized]


def decode_ros_image(message: Any) -> tuple[np.ndarray, str]:
    """按照 step 和字节序解码 ROS Image 消息."""
    base_dtype, channels, semantic = image_encoding_spec(str(message.encoding))
    item_size = base_dtype.itemsize
    row_payload = int(message.width) * channels * item_size
    if int(message.step) < row_payload:
        raise ValueError(
            f"Image step 过小: {message.step} < {row_payload}"
        )
    data = np.ascontiguousarray(message.data, dtype=np.uint8)
    expected_bytes = int(message.step) * int(message.height)
    if data.nbytes < expected_bytes:
        raise ValueError(f"Image data 过短: {data.nbytes} < {expected_bytes}")
    rows = data[:expected_bytes].reshape(int(message.height), int(message.step))
    payload = np.ascontiguousarray(rows[:, :row_payload])
    if item_size > 1:
        byte_order = ">" if bool(message.is_bigendian) else "<"
        dtype = base_dtype.newbyteorder(byte_order)
    else:
        dtype = base_dtype
    image = payload.reshape(-1).view(dtype)
    shape = (int(message.height), int(message.width))
    if channels > 1:
        shape += (channels,)
    image = image.reshape(shape)
    if item_size > 1:
        image = image.astype(base_dtype, copy=False)
    if semantic == "bgr":
        image = image[..., ::-1].copy()
        semantic = "rgb"
    elif semantic == "bgra":
        image = image[..., [2, 1, 0, 3]].copy()
        semantic = "rgba"
    return image, semantic


def build_blueprint() -> rrb.Blueprint:
    """构建左侧三维视图和右侧传感器及状态面板布局."""
    pointcloud_view = rrb.Spatial3DView(
        name="Point cloud + 1 s history + 2 s future",
        origin="/world",
        contents=[
            "/world/pointcloud",
            "/world/trajectory/**",
            "/world/camera/**",
        ],
        line_grid=True,
    )
    rgb_view = rrb.Spatial2DView(
        name="RGB",
        origin="/world/camera/rgb",
        contents=["/world/camera/rgb"],
    )
    depth_view = rrb.Spatial2DView(
        name="Depth",
        origin="/sensors/depth",
        contents=["/sensors/depth"],
    )
    top_down_view = rrb.Spatial2DView(
        name="Top-down trajectory",
        origin="/dashboard/top_down",
        contents=["/dashboard/top_down/**"],
    )
    velocity_view = rrb.TimeSeriesView(
        name="vx / vy",
        origin="/dashboard/velocity",
        contents=["/dashboard/velocity/**"],
        plot_legend=rrb.PlotLegend(corner="RightTop", visible=True),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            pointcloud_view,
            rrb.Vertical(
                rrb.Horizontal(rgb_view, depth_view, column_shares=[1, 1]),
                rrb.Horizontal(
                    top_down_view,
                    velocity_view,
                    column_shares=[1, 1],
                ),
                row_shares=[1, 1],
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )


def log_static_scene(
    inspection: BagInspection,
    velocity_source: str,
) -> None:
    """记录坐标系, 相机模型, 全局轨迹和曲线样式."""
    rr.log("world", rr.ViewCoordinates.FLU, static=True)
    calibration = inspection.calibration
    rr.log(
        "world/camera",
        rr.Pinhole(
            image_from_camera=calibration.image_from_camera,
            resolution=[calibration.width, calibration.height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.65,
        ),
        static=True,
    )
    inverse_intrinsic = np.linalg.inv(calibration.image_from_camera)
    pixels = np.array(
        [
            [0.0, 0.0, 1.0],
            [calibration.width - 1.0, 0.0, 1.0],
            [calibration.width - 1.0, calibration.height - 1.0, 1.0],
            [0.0, calibration.height - 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    corners = (inverse_intrinsic @ pixels.T).T * 0.65
    origin = np.zeros((1, 3), dtype=np.float64)
    frustum_strips = [
        np.vstack([origin, corner[None, :]]) for corner in corners
    ]
    frustum_strips.append(np.vstack([corners, corners[0]]))
    rr.log(
        "world/camera/frustum",
        rr.LineStrips3D(
            frustum_strips,
            colors=[[38, 198, 218]],
            radii=0.012,
            show_labels=False,
        ),
        static=True,
    )
    full_xy = inspection.odometry.positions[:, :2]
    rr.log(
        "dashboard/top_down/full_trajectory",
        rr.LineStrips2D(
            [full_xy],
            colors=[COLOR_FULL_PATH],
            radii=0.018,
            labels=["Full trajectory"],
            show_labels=False,
        ),
        static=True,
    )
    velocity_suffix = (
        "world derived" if velocity_source.startswith("pose-derived") else "twist"
    )
    rr.log(
        "dashboard/velocity/vx",
        rr.SeriesLines(
            colors=[COLOR_VX],
            names=[f"vx [m/s] ({velocity_suffix})"],
            widths=[2.0],
        ),
        static=True,
    )
    rr.log(
        "dashboard/velocity/vy",
        rr.SeriesLines(
            colors=[COLOR_VY],
            names=[f"vy [m/s] ({velocity_suffix})"],
            widths=[2.0],
        ),
        static=True,
    )


def trajectory_window(
    odometry: OdometrySeries,
    timestamp_ns: int,
    history_seconds: float,
    future_seconds: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """截取当前时刻的一秒历史和两秒未来轨迹窗口."""
    timestamps = odometry.timestamps_ns
    index = int(np.searchsorted(timestamps, timestamp_ns, side="left"))
    index = min(max(index, 0), len(timestamps) - 1)
    if index > 0 and abs(int(timestamps[index - 1]) - timestamp_ns) < abs(
        int(timestamps[index]) - timestamp_ns
    ):
        index -= 1
    history_start = bisect.bisect_left(
        timestamps,
        timestamp_ns - int(history_seconds * NANOSECONDS_PER_SECOND),
    )
    future_end = bisect.bisect_right(
        timestamps,
        timestamp_ns + int(future_seconds * NANOSECONDS_PER_SECOND),
    )
    history = odometry.positions[history_start : index + 1]
    future = odometry.positions[index:future_end]
    return history, future, index


def log_trajectory_state(
    odometry: OdometrySeries,
    timestamp_ns: int,
    history_seconds: float,
    future_seconds: float,
) -> None:
    """记录三维和俯视轨迹窗口以及当前速度."""
    history, future, index = trajectory_window(
        odometry,
        timestamp_ns,
        history_seconds,
        future_seconds,
    )
    current = odometry.positions[index]
    rotation = quaternion_to_matrix(odometry.quaternions_xyzw[index])[:3, :3]
    heading = rotation[:, 0] * 0.55
    rr.log(
        "world/trajectory/history_1s",
        rr.LineStrips3D(
            [history],
            colors=[COLOR_HISTORY],
            radii=0.035,
            labels=[f"History {history_seconds:g} s"],
            show_labels=False,
        ),
    )
    rr.log(
        "world/trajectory/future_2s",
        rr.LineStrips3D(
            [future],
            colors=[COLOR_FUTURE],
            radii=0.035,
            labels=[f"Future {future_seconds:g} s"],
            show_labels=False,
        ),
    )
    rr.log(
        "world/trajectory/robot",
        rr.Points3D(
            [current],
            colors=[COLOR_CURRENT],
            radii=0.11,
            labels=["Robot"],
            show_labels=False,
        ),
    )
    rr.log(
        "world/trajectory/heading",
        rr.Arrows3D(
            origins=[current],
            vectors=[heading],
            colors=[COLOR_CURRENT],
            radii=0.025,
        ),
    )
    rr.log(
        "dashboard/top_down/history_1s",
        rr.LineStrips2D(
            [history[:, :2]],
            colors=[COLOR_HISTORY],
            radii=0.035,
            labels=[f"History {history_seconds:g} s"],
            show_labels=False,
        ),
    )
    rr.log(
        "dashboard/top_down/future_2s",
        rr.LineStrips2D(
            [future[:, :2]],
            colors=[COLOR_FUTURE],
            radii=0.035,
            labels=[f"Future {future_seconds:g} s"],
            show_labels=False,
        ),
    )
    rr.log(
        "dashboard/top_down/robot",
        rr.Points2D(
            [current[:2]],
            colors=[COLOR_CURRENT],
            radii=0.10,
            labels=["Robot"],
            show_labels=False,
        ),
    )
    rr.log(
        "dashboard/top_down/heading",
        rr.Arrows2D(
            origins=[current[:2]],
            vectors=[heading[:2]],
            colors=[COLOR_CURRENT],
            radii=0.025,
        ),
    )
    velocity = odometry.display_velocities_xy[index]
    rr.log("dashboard/velocity/vx", rr.Scalars(float(velocity[0])))
    rr.log("dashboard/velocity/vy", rr.Scalars(float(velocity[1])))


def log_camera_pose(
    resolver: FrameResolver,
    root_frame: str,
    camera_frame: str,
    timestamp_ns: int,
) -> bool:
    """解析并记录相机 optical frame 在世界坐标中的姿态."""
    world_from_camera = resolver.lookup(root_frame, camera_frame, timestamp_ns)
    if world_from_camera is None:
        return False
    rr.log(
        "world/camera",
        rr.Transform3D(
            translation=world_from_camera[:3, 3],
            quaternion=rr.Quaternion(
                xyzw=matrix_to_quaternion(world_from_camera)
            ),
            axis_length=0.25,
        ),
    )
    return True


def render_bag(
    inspection: BagInspection,
    resolver: FrameResolver,
    aliases: dict[str, str],
    camera_frame: str,
    arguments: argparse.Namespace,
) -> None:
    """按 bag 记录时间流式写入 Rerun 数据和布局."""
    blueprint = build_blueprint()
    recording_id = f"{inspection.bag_path.stem}-{inspection.start_ns}"
    rr.init(
        arguments.application_id,
        recording_id=recording_id,
        spawn=arguments.save is None,
        default_blueprint=blueprint,
    )
    if arguments.save is not None:
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(arguments.save, default_blueprint=blueprint)
    log_static_scene(inspection, inspection.odometry.velocity_source)
    root_frame = canonical_frame_id(inspection.odometry.root_frame, aliases)
    start_ns = inspection.start_ns + int(
        arguments.start_seconds * NANOSECONDS_PER_SECOND
    )
    end_ns = inspection.end_ns
    if arguments.duration_seconds is not None:
        end_ns = min(
            end_ns,
            start_ns
            + int(arguments.duration_seconds * NANOSECONDS_PER_SECOND),
        )
    selected_topics = {
        inspection.topics.pointcloud,
        inspection.topics.rgb,
        inspection.topics.depth,
        inspection.topics.odometry,
    }
    counters: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    with AnyReader([inspection.bag_path]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in selected_topics
        ]
        for connection, timestamp_ns, rawdata in reader.messages(
            connections=connections,
            start=start_ns,
            stop=end_ns,
        ):
            counters[connection.topic] += 1
            if connection.topic in {
                inspection.topics.rgb,
                inspection.topics.depth,
            } and (counters[connection.topic] - 1) % arguments.image_stride:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            rr.set_time(
                "bag_time",
                duration=(timestamp_ns - inspection.start_ns) / 1e9,
            )
            if connection.topic == inspection.topics.odometry:
                log_trajectory_state(
                    inspection.odometry,
                    timestamp_ns,
                    arguments.history_seconds,
                    arguments.future_seconds,
                )
            elif connection.topic == inspection.topics.pointcloud:
                points, colors = decode_pointcloud(
                    message,
                    arguments.point_stride,
                )
                cloud_frame = canonical_frame_id(message.header.frame_id, aliases)
                world_from_cloud = resolver.lookup(
                    root_frame,
                    cloud_frame,
                    timestamp_ns,
                )
                if world_from_cloud is None:
                    warnings[f"pointcloud:{cloud_frame}"] += 1
                else:
                    points = transform_points(points, world_from_cloud)
                    rr.log(
                        "world/pointcloud",
                        rr.Points3D(
                            points,
                            colors=colors,
                            radii=arguments.point_radius,
                        ),
                    )
            elif connection.topic == inspection.topics.rgb:
                image, semantic = decode_ros_image(message)
                current_camera_frame = canonical_frame_id(
                    message.header.frame_id,
                    aliases,
                )
                if current_camera_frame != camera_frame:
                    warnings[
                        f"rgb_frame_changed:{current_camera_frame}"
                    ] += 1
                if not log_camera_pose(
                    resolver,
                    root_frame,
                    current_camera_frame,
                    timestamp_ns,
                ):
                    warnings[f"camera:{current_camera_frame}"] += 1
                image_archetype: Any = rr.Image(image)
                if arguments.jpeg_quality > 0 and semantic in {
                    "rgb",
                    "rgba",
                    "mono",
                }:
                    image_archetype = image_archetype.compress(
                        jpeg_quality=arguments.jpeg_quality
                    )
                rr.log("world/camera/rgb", image_archetype)
            elif connection.topic == inspection.topics.depth:
                depth, semantic = decode_ros_image(message)
                if semantic not in {"depth_mm", "depth_m", "mono"}:
                    raise ValueError(
                        f"depth topic 使用了非深度编码: {message.encoding}"
                    )
                meter = arguments.depth_meter
                if meter is None:
                    meter = 1.0 if semantic == "depth_m" else 1000.0
                rr.log(
                    "sensors/depth",
                    rr.DepthImage(
                        depth,
                        meter=meter,
                        colormap="Turbo",
                    ),
                )
    rr.disconnect()
    print("\nRerun 写入完成")
    for topic in sorted(counters):
        print(f"  {topic}: {counters[topic]} messages")
    if warnings:
        print("\n未解析 frame 警告")
        for key, count in warnings.items():
            print(f"  {key}: {count}")
    if arguments.save is not None:
        print(f"\nRRD: {arguments.save.resolve()}")


def print_inspection_report(
    inspection: BagInspection,
    aliases: dict[str, str],
    resolver: FrameResolver,
    synthetic_bridge: tuple[str, str] | None,
) -> None:
    """打印 topic 选择和 TF 修复的可审计报告."""
    duration = (inspection.end_ns - inspection.start_ns) / 1e9
    print("MCAP inspection")
    print(f"  path: {inspection.bag_path.resolve()}")
    print(f"  duration: {duration:.3f} s")
    print("\nSelected topics")
    for label, topic in (
        ("pointcloud", inspection.topics.pointcloud),
        ("rgb", inspection.topics.rgb),
        ("depth", inspection.topics.depth),
        ("odometry", inspection.topics.odometry),
        ("color camera info", inspection.topics.color_camera_info),
        ("tf", inspection.topics.tf),
        ("tf_static", inspection.topics.tf_static),
    ):
        count = inspection.topic_counts.get(topic, 0) if topic else 0
        print(f"  {label}: {topic or 'missing'} ({count})")
    print("\nSampled sensor frames")
    for topic, frame in sorted(inspection.sampled_frames.items()):
        print(f"  {topic}: {frame or '<empty>'}")
    print("\nFrame normalization")
    if aliases:
        for source, target in sorted(aliases.items()):
            print(f"  alias: {source} -> {target}")
    else:
        print("  aliases: none")
    if synthetic_bridge is not None:
        parent, child = synthetic_bridge
        print(
            "  synthetic camera bridge: "
            f"{parent} -> {child} with configured extrinsic"
        )
    else:
        print("  synthetic camera bridge: not needed")
    print("\nCanonical TF components")
    for index, component in enumerate(resolver.components(), start=1):
        print(f"  component {index}: {', '.join(sorted(component))}")
    pointcloud = inspection.pointcloud_description
    print("\nPointCloud2")
    print(
        f"  frame={pointcloud.frame_id}, points={pointcloud.point_count}, "
        f"point_step={pointcloud.point_step}, fields={pointcloud.fields}"
    )
    print("\nImages")
    print(
        f"  rgb: {inspection.rgb_description.width}x"
        f"{inspection.rgb_description.height} "
        f"{inspection.rgb_description.encoding}"
    )
    print(
        f"  depth: {inspection.depth_description.width}x"
        f"{inspection.depth_description.height} "
        f"{inspection.depth_description.encoding}"
    )
    if inspection.calibration.synthetic:
        print("  camera intrinsics: synthetic 90 degree horizontal FOV")
    else:
        print("  camera intrinsics: CameraInfo")
    odometry = inspection.odometry
    displacement = np.linalg.norm(
        odometry.positions[-1, :2] - odometry.positions[0, :2]
    )
    twist_peak = np.max(np.linalg.norm(odometry.twist_velocities_xy, axis=1))
    print("\nOdometry")
    print(f"  frames: {odometry.root_frame} -> {odometry.child_frame}")
    print(f"  samples: {len(odometry.timestamps_ns)}")
    print(f"  planar displacement: {displacement:.3f} m")
    print(f"  twist peak: {twist_peak:.6f} m/s")
    print(f"  displayed velocity: {odometry.velocity_source}")


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ROS 2 MCAP point cloud, RGB-D, odometry, and TF in Rerun."
        )
    )
    parser.add_argument(
        "bag",
        nargs="?",
        type=Path,
        help="MCAP file or rosbag2 directory for backward compatibility",
    )
    parser.add_argument(
        "--bag-name",
        type=Path,
        help="Bag name under rosbags or an explicit bag path",
    )
    parser.add_argument("--pointcloud-topic")
    parser.add_argument("--rgb-topic")
    parser.add_argument("--depth-topic")
    parser.add_argument("--odom-topic")
    parser.add_argument("--color-info-topic")
    parser.add_argument("--tf-topic")
    parser.add_argument("--tf-static-topic")
    parser.add_argument(
        "--frame-alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Repeatable frame normalization rule",
    )
    parser.add_argument(
        "--camera-parent",
        help="Parent frame for a missing camera TF bridge",
    )
    parser.add_argument(
        "--camera-extrinsic",
        type=float,
        nargs=7,
        metavar=("TX", "TY", "TZ", "QX", "QY", "QZ", "QW"),
        help=(
            "parent_from_camera_root extrinsic used only when the camera tree "
            "is disconnected"
        ),
    )
    parser.add_argument(
        "--strict-tf",
        action="store_true",
        help="Fail instead of adding a synthetic identity camera bridge",
    )
    parser.add_argument("--history-seconds", type=float, default=1.0)
    parser.add_argument("--future-seconds", type=float, default=2.0)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=float, default=0.025)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--depth-meter",
        type=float,
        help="Raw depth value corresponding to one meter",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument(
        "--save",
        type=Path,
        help="Save to an RRD file instead of spawning the viewer",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print topic and TF diagnostics without launching Rerun",
    )
    parser.add_argument(
        "--application-id",
        default="forest_nav_mcap",
    )
    return parser


def resolve_bag_path(arguments: argparse.Namespace) -> Path:
    """解析 bag-name 或兼容的位置路径并返回实际 bag 路径."""
    if arguments.bag_name is not None and arguments.bag is not None:
        raise ValueError("--bag-name 不能与位置 bag 路径同时使用")
    requested = arguments.bag_name or arguments.bag
    if requested is None:
        return Path("rosbags/test_tf")
    if requested.exists():
        return requested
    bag_under_root = Path("rosbags") / requested
    if bag_under_root.exists():
        return bag_under_root
    return requested


def validate_arguments(arguments: argparse.Namespace) -> None:
    """验证会影响索引和时间范围的数值参数."""
    if arguments.history_seconds <= 0.0:
        raise ValueError("--history-seconds 必须大于零")
    if arguments.future_seconds <= 0.0:
        raise ValueError("--future-seconds 必须大于零")
    if arguments.point_stride < 1:
        raise ValueError("--point-stride 必须大于等于一")
    if arguments.image_stride < 1:
        raise ValueError("--image-stride 必须大于等于一")
    if arguments.point_radius <= 0.0:
        raise ValueError("--point-radius 必须大于零")
    if not 0 <= arguments.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须位于零到一百")
    if arguments.start_seconds < 0.0:
        raise ValueError("--start-seconds 不能为负数")
    if arguments.duration_seconds is not None and arguments.duration_seconds <= 0.0:
        raise ValueError("--duration-seconds 必须大于零")
    if arguments.depth_meter is not None and arguments.depth_meter <= 0.0:
        raise ValueError("--depth-meter 必须大于零")


def run(arguments: argparse.Namespace) -> None:
    """执行 bag 预扫描, TF 修复和 Rerun 写入流程."""
    validate_arguments(arguments)
    bag_path = resolve_bag_path(arguments)
    inspection = inspect_bag(bag_path, arguments)
    explicit_aliases = parse_frame_aliases(arguments.frame_alias)
    aliases = automatic_aliases(inspection, explicit_aliases)
    resolver, camera_frame, synthetic_bridge = build_frame_resolver(
        inspection,
        aliases,
        arguments,
    )
    if camera_frame is None:
        raise ValueError("无法确定 RGB camera frame")
    print_inspection_report(
        inspection,
        aliases,
        resolver,
        synthetic_bridge,
    )
    if arguments.inspect_only:
        return
    render_bag(
        inspection,
        resolver,
        aliases,
        camera_frame,
        arguments,
    )


def main() -> int:
    """解析命令行并返回适合 shell 的退出状态."""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        run(arguments)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
