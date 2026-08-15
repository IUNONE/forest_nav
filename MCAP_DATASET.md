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

## 写入推理轨迹并用 Rerun 查看

训练完成后，可将六个原始 MCAP 流式复制到新目录，并以 10 Hz 写入8个规划话题：

```bash
python rerun_visualize.py \
  --input /data/rosbag2 \
  --output /data/zhangshenghong/datasets/rosbag2_planned \
  --checkpoint results/resnet50_spatial
```

输入文件不会被原地修改。目录输入会生成例如
`rosbag2_planned/exp2/exp2_0_planned.mcap` 的文件。输出保留所有原始 schema、channel、
message、attachment 和 metadata，因此需要预留接近原始六个 bag 总大小的额外磁盘空间。

转换器还会写入 `/robot_description`（`std_msgs/msg/String`）以及一条新增的
`/tf_static` 消息。机器人模型来自
`assets/diablo_original/urdf/diablo_forest_nav.urdf`：它在上游模型基础上增加了
`body -> base_link` 零偏移固定关节，并包含其余14条关节固定姿态。原始 bag 已有动态
`camera_init -> body`，所以机器人、地图点云和规划轨迹位于同一 TF 树中。

新增话题为：

- `/planned_trajectory_final`
- `/planned_trajectory_opt1` 至 `/planned_trajectory_opt7`

消息采用 `sensor_msgs/msg/PointCloud2`，每条消息包含32个轨迹点。Rerun 可将该标准消息
自动显示为 `Points3D`；`nav_msgs/Path` 当前不能被 Rerun 自动转换为空间轨迹。点坐标已经
从图像时刻的 `body` 局部坐标系转换到实际 odometry reference frame `camera_init`。

默认 `opt1` 至 `opt7` 使用 seed `0..6` 和4步快速扩散。`final` 使用 checkpoint 配置中的
完整10步扩散，并在相同7个 seed 的候选中选择与未来 odometry 真值平均点误差最小的一条。
因此 `final` 是带真值信息的离线 oracle 可视化结果，不能用作在线部署方法。在 odometry
真值无效或不足3.2秒时，脚本回退到第一个 full-step 候选，并在 MCAP metadata 中记录原因。

Rerun 0.36 需要 Python 3.10，不能安装到训练使用的 Python 3.8 环境。首次使用时创建独立
环境：

```bash
conda env create -f rerun_env.yaml
```

然后用脚本的 `view` 子命令打开一个转换后的 bag：

```bash
conda run -n rerun_viz python rerun_visualize.py view \
  --input /data/zhangshenghong/datasets/rosbag2_planned/exp2/exp2_0_planned.mcap
```

该布局左侧为3D点云/轨迹/URDF，target frame 固定为 `body`；机器人保持在视图原点，
地图和点云相对机器人移动。右侧显示前视 RGB。这里固定的是空间参考系，不是简单锁定
相机屏幕坐标，因此旋转时的方向关系仍然正确。

可以通过 `--opt-steps`、`--final-steps`、`--seeds`、`--device` 和 `--max-frames`
调整推理行为。`--max-frames` 只限制推理帧数，原始 MCAP 消息仍会被完整复制。
