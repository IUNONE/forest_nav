# MCAP 训练数据转换

转换器从 ROS 2 MCAP 中读取：

- RGB：`/robot1/D435i_front/color/image_raw`
- 里程计：`/Odometry`

默认每 `0.0333 s` 最多提取一个当前 RGB 帧，使用 letterbox 保留完整视野并缩放到
`224×224`。轨迹按照
消息的 `header.stamp` 同步，在当前车体坐标系内插值得到
`t+0.1, t+0.2, ..., t+3.2 s` 共 32 个 `(x, y)` 点。

## 转换全部数据

```bash
conda activate forest_nav
python mcap_to_h5.py \
  --input /data/rosbag2 \
  --output /data/zhangshenghong/datasets/forest_nav_rgb_trajectory.h5
```

输出文件已配置在 `config/lightning.yaml` 中。转换过程中会逐个 bag 打印保留和过滤的
样本数。输出已存在时，使用 `--append` 继续转换，或者显式使用 `--overwrite` 重建。

## 异常过滤

每个样本会检查从当前 RGB 时间到未来 `3.2 s` 所涉及的全部原始 Odometry 相邻段。
速度由平面位移除以真实时间差计算；任一段速度大于 `2 m/s` 时，整条样本会被过滤，
从而保持输出轨迹固定为 32 个等间隔点。跨越大于 `0.25 s` 的 Odometry 断档、时间范围
不足或非有限数据的样本同样不会写入。

阈值均可调整，例如：

```bash
python mcap_to_h5.py \
  --input /data/rosbag2/exp2 \
  --output /tmp/exp2_check.h5 \
  --max-velocity 2.0 \
  --max-odom-gap 0.25 \
  --max-samples 100
```

## HDF5 格式

每个样本是一个 group：

- `image`: `(224, 224, 3)`, `uint8`, RGB
- `future_waypoint_local`: `(32, 2)`, `float32`, 当前车体坐标系下的米制 `(x, y)`
- group attributes: 原始 bag、MCAP 路径和图像时间戳
- `trajectory_frame`: `body_yaw_aligned`，即图像时刻的机器人二维局部坐标系

文件根 attributes 记录图像尺寸、轨迹 horizon/interval、速度阈值、坐标轴语义、
实际轨迹范围、逐 bag 过滤统计、topic 和样本总数。
训练时 `NavigationDataset` 会将 RGB 转成 `(3, 224, 224)` 浮点 tensor。当前配置关闭
validation split，六个 bag 中通过过滤的约 1902 条样本全部用于训练。

由于录制数据不包含 `body -> D435i_front_link` 外参，可视化脚本只分别显示 RGB 与
body local-frame 轨迹，不把轨迹伪精确地投影到图像上。

## 用 Rerun 查看规划轨迹

可视化环境沿用 Rerun 0.24。首次使用时创建独立环境：

```bash
conda env create -f rerun_env.yaml
```

打开一个包含传感器、TF、Odometry 和规划轨迹的 MCAP 或 rosbag2 目录：

```bash
conda run -n rerun python rerun_visualize.py \
  --bag-name /data/zhangshenghong/datasets/rosbag2_planned/exp2/exp2_0_planned.mcap
```

脚本识别 `/planned_trajectory_final` 和 `/planned_trajectory_opt1` 至
`/planned_trajectory_opt7`，支持 `sensor_msgs/msg/PointCloud2` 和 `nav_msgs/msg/Path`。
`final` 显示为亮黄色粗线，七条 `opt` 使用不同暗色细线。

左侧三维窗口以 Odometry child frame（通常为 `body`）为固定坐标系。机器人 URDF 从
`assets/diablo_original/urdf/diablo_forest_nav.urdf` 加载并保持在原点，点云、历史/未来
轨迹、规划轨迹和相机模型均逐帧转换到该坐标系。可用 `--robot-frame` 或
`--robot-urdf` 覆盖默认值；右侧 RGB、深度、俯视轨迹和速度面板保留原布局。
