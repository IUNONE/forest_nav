# Forest Nav

Forest Nav 是一个基于 RGB 观测和 diffusion policy 的机器人导航轨迹预测项目. 项目从 ROS 2
MCAP 或已有 HDF5 数据中构造训练样本, 使用视觉编码器和扩散模型预测机器人局部坐标系下的
未来轨迹, 并提供训练, 数据处理和可视化脚本.

所有命令默认在项目根目录下执行.

## 目录结构

```text
README.md
config/
└── lightning.yaml
navdiffusion/
└── 模型和数据集实现
scripts/
├── inference_add_topic.py
├── train/
│   └── train.py
├── visualize/
│   ├── h5_data_visulize.py
│   ├── h5_result_visulize.py
│   ├── rerun_visualize.py
│   └── rerun_visualize_.py
└── data_process/
    ├── h5_data_converter.py
    ├── h5_data_converter_batch.py
    └── mcap_to_h5.py
```

## 环境

- `env.yml`: 训练和 HDF5/MCAP 数据处理环境.
- `rerun_env.yaml`: Rerun MCAP 可视化环境.

训练和数据处理脚本使用 `forest_nav` 环境, Rerun 可视化使用 `rerun` 环境.

## 通用运行约定

- 在项目根目录执行命令, 以保证 `config/lightning.yaml` 和相对路径能够正确解析.
- 训练和 HDF5/MCAP 数据处理使用 `forest_nav` 环境.
- Rerun 可视化使用 `rerun` 环境.
- 在项目根目录执行下面的命令, 并先将当前项目加入 `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

- 脚本统一使用 `python -B scripts/.../xxx.py` 直接运行.
- 配置文件位于 `config/lightning.yaml`, 训练 checkpoint 默认写入
  `results/resnet50_spatial`.

## train

### 训练导航扩散模型

```bash
conda activate forest_nav
python -B scripts/train/train.py
```

训练脚本读取 `config/lightning.yaml`, 使用其中的 HDF5 数据集, batch size, 优化器,
训练轮数和 checkpoint 配置. 脚本还会初始化 Weights and Biases logger, 因此运行前需要
准备对应的 W&B 登录和项目配置.

### 从 MCAP 推理并写入新 MCAP

`inference_add_topic.py` 会复制输入 MCAP 的原有消息, 并生成一个新的 MCAP. 对每个采样的
RGB 帧, 它使用不同 seed 的 diffusion noise 写入 `/planned_trajectory_opt1` 至
`/planned_trajectory_opt7`, 同时把里程计对齐的未来 GT 轨迹写入
`/planned_trajectory_final`. 轨迹消息类型为 `nav_msgs/msg/Path`, 使用 Odometry 的
`child_frame_id` 作为 frame, waypoint 时间戳对齐到已有 Odometry 时间戳.

```bash
conda activate forest_nav
python -B scripts/inference_add_topic.py \
  --input /data/rosbag2/exp2/exp2_0.mcap \
  --output /data/rosbag2/exp2/exp2_0_planned.mcap \
  --checkpoint results/resnet50_spatial/epoch=999-step=4000.ckpt
```

如果输入目录下有多个实验子目录，可以递归批量处理。下面的命令会处理
`/data/rosbag2/exp1` 至 `exp6` 下的所有 `.mcap` 文件，并在输出目录中保留相同的
子目录和文件名；`metadata.yaml` 会自动忽略：

```bash
python -B scripts/inference_add_topic.py \
  --input_dir /data/rosbag2 \
  --output_dir /data/rosbag2_with_planned \
  --checkpoint results/resnet50_spatial/epoch=999-step=4000.ckpt
```

默认每 0.1 秒处理一帧 RGB; 使用 `--sample-interval 0` 可处理每一帧. 可用
`--seed-base` 修改七条候选轨迹的随机种子起点, 用 `--max-samples` 先限制每个 MCAP
的测试数量. 目录模式默认按文件名顺序顺序处理；如需单个文件失败后继续，可加
`--continue-on-error`.

## visualize

### 可视化 HDF5 数据集

```bash
conda activate forest_nav
python -B scripts/visualize/h5_data_visulize.py \
  --h5_path /data/zhangshenghong/datasets/forest_nav_rgb_trajectory.h5
```

该脚本逐样本显示 RGB 图像和局部坐标系下的未来轨迹.

### 使用 checkpoint 推理并可视化轨迹

```bash
conda activate forest_nav
python -B scripts/visualize/h5_result_visulize.py \
  --h5_path /data/zhangshenghong/datasets/forest_nav_rgb_trajectory.h5 \
  --config_path config/lightning.yaml \
  --ckpt_dir results/resnet50_spatial
```

该脚本从 HDF5 图像输入执行模型推理, 然后在窗口中对比真实轨迹和预测轨迹. 它不会读取
MCAP, 也不会向 MCAP 写入 `/planned_trajectory`.

没有显示器时, 可关闭交互窗口并将所有样本的对比图保存为 PNG:

```bash
python -B scripts/visualize/h5_result_visulize.py \
  --h5_path /data/zhangshenghong/datasets/forest_nav_rgb_trajectory.h5 \
  --config_path config/lightning.yaml \
  --ckpt_dir results/resnet50_spatial \
  --png_output true \
  --png_output_dir results/h5_result_png
```

不指定 `--png_output_dir` 时, 默认保存到 `results/h5_result_png`. 每张图包含 RGB 图像,
真实轨迹, 预测轨迹和该样本的 L2 误差.

### 使用 Rerun 查看 MCAP

```bash
conda run -n rerun python -B scripts/visualize/rerun_visualize.py \
  --bag-name /data/zhangshenghong/datasets/rosbag2_planned/exp2/exp2_0_planned.mcap
```

`rerun_visualize.py` 读取已有 MCAP 中的传感器, TF, Odometry 和规划轨迹, 并将内容写入
Rerun. `--save` 参数保存的是 RRD 文件, 不是 MCAP.

`rerun_visualize_.py` 是旧版 Rerun 可视化脚本, 用于保留历史运行方式. 新的规划轨迹识别和
TF 处理逻辑请使用 `rerun_visualize.py`.

## data_process

### 将传统 HDF5 数据转换为导航格式

处理单个 HDF5 文件:

```bash
conda activate forest_nav
python -B scripts/data_process/h5_data_converter.py \
  --h5_path /path/to/data.h5 \
  --dataset_name scenario1 \
  --n_future 32
```

批量处理目录下的 `*/data.h5` 文件:

```bash
conda activate forest_nav
python -B scripts/data_process/h5_data_converter_batch.py \
  --dir /path/to/scenarios \
  --n_future 32
```

### MCAP 训练数据转换

本节说明 MCAP 数据集转换和规划轨迹可视化流程.

转换器从 ROS 2 MCAP 中读取:

- RGB: `/robot1/D435i_front/color/image_raw`
- 里程计: `/Odometry`

默认每 `0.0333 s` 最多提取一个当前 RGB 帧, 使用 letterbox 保留完整视野并缩放到
`224x224`. 轨迹按照消息的 `header.stamp` 同步, 在当前车体坐标系内插值得到
`t+0.1, t+0.2, ..., t+3.2 s` 共 32 个 `(x, y)` 点.

#### 转换全部数据

```bash
conda activate forest_nav
python -B scripts/data_process/mcap_to_h5.py \
  --input /data/rosbag2 \
  --output /data/zhangshenghong/datasets/forest_nav_rgb_trajectory.h5
```

输出文件已配置在 `config/lightning.yaml` 中. 转换过程中会逐个 bag 打印保留和过滤的
样本数. 输出已存在时, 使用 `--append` 继续转换, 或者显式使用 `--overwrite` 重建.

#### 异常过滤

每个样本会检查从当前 RGB 时间到未来 `3.2 s` 所涉及的全部原始 Odometry 相邻段.
速度由平面位移除以真实时间差计算; 任一段速度大于 `2 m/s` 时, 整条样本会被过滤,
从而保持输出轨迹固定为 32 个等间隔点. 跨越大于 `0.25 s` 的 Odometry 断档, 时间范围
不足或非有限数据的样本同样不会写入.

阈值均可调整, 例如:

```bash
python -B scripts/data_process/mcap_to_h5.py \
  --input /data/rosbag2/exp2 \
  --output /tmp/exp2_check.h5 \
  --max-velocity 2.0 \
  --max-odom-gap 0.25 \
  --max-samples 100
```

#### HDF5 格式

每个样本是一个 group:

- `image`: `(224, 224, 3)`, `uint8`, RGB
- `future_waypoint_local`: `(32, 2)`, `float32`, 当前车体坐标系下的米制 `(x, y)`
- group attributes: 原始 bag, MCAP 路径和图像时间戳
- `trajectory_frame`: `body_yaw_aligned`, 即图像时刻的机器人二维局部坐标系

文件根 attributes 记录图像尺寸, 轨迹 horizon/interval, 速度阈值, 坐标轴语义,
实际轨迹范围, 逐 bag 过滤统计, topic 和样本总数. 训练时 `NavigationDataset` 会将 RGB
转成 `(3, 224, 224)` 浮点 tensor. 当前配置训练 loader 使用全部有效样本, 并从同一份训练数据中
固定抽取 10% 作为 validation loader; 因此 validation 样本会与训练样本重叠. 验证阶段使用
diffusion 推理生成完整轨迹, 记录原始米制坐标上的平均 waypoint 欧氏 L2 误差
`val/l2_error`.

`num_train_timesteps` 是训练时的噪声时间步数, `num_inference_steps` 是每次 diffusion 推理
反向去噪的迭代次数. 当前配置分别为 100 和 100, 因此每次推理默认执行 100 次去噪迭代.

由于录制数据不包含 `body -> D435i_front_link` 外参, 可视化脚本只分别显示 RGB 与
body local-frame 轨迹, 不把轨迹伪精确地投影到图像上.

#### 用 Rerun 查看规划轨迹

可视化环境沿用 Rerun 0.24. 首次使用时创建独立环境:

```bash
conda env create -f rerun_env.yaml
```

打开一个包含传感器, TF, Odometry 和规划轨迹的 MCAP 或 rosbag2 目录:

```bash
conda run -n rerun python -B scripts/visualize/rerun_visualize.py \
  --bag-name /data/zhangshenghong/datasets/rosbag2_planned/exp2/exp2_0_planned.mcap
```

脚本识别 `/planned_trajectory_final` 和 `/planned_trajectory_opt1` 至
`/planned_trajectory_opt7`, 支持 `sensor_msgs/msg/PointCloud2` 和 `nav_msgs/msg/Path`.
`final` 显示为亮黄色粗线, 七条 `opt` 使用不同暗色细线.

左侧三维窗口以 Odometry child frame, 通常为 `body`, 为固定坐标系, 坐标约定为 ROS FLU
（`+x` 前方, `+y` 左方, `+z` 上方）. 机器人 URDF 从
`assets/diablo_original/urdf/diablo_forest_nav.urdf` 加载并保持在原点, 点云, 历史/未来
轨迹, 规划轨迹和相机模型均逐帧转换到该坐标系. 可用 `--robot-frame` 或 `--robot-urdf`
覆盖默认值; 右侧 RGB, 深度, 俯视轨迹和速度面板保留原布局.

## 当前脚本范围

`h5_result_visulize.py` 对 HDF5 样本执行推理并可视化, `inference_add_topic.py` 将模型
推理结果和 GT 写入新的 MCAP, `rerun_visualize.py` 读取这些规划轨迹并生成 Rerun 输出.
