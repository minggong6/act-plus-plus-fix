# 数据集格式说明

本项目使用 HDF5 格式存储机器人操作数据。每个 HDF5 文件代表一个 episode（回合）。

## 目录结构

数据集目录通常包含多个 `episode_*.hdf5` 文件：

```text
dataset_dir/
├── episode_0.hdf5
├── episode_1.hdf5
├── episode_2.hdf5
...
└── episode_N.hdf5
```

## HDF5 文件结构

每个 `episode_*.hdf5` 文件的内部结构如下（以文件树形式展示）：

```text
/
├── attrs                       # 根节点属性
│   ├── sim                     # [bool] 是否为仿真数据 (可选)
│   └── compress                # [bool] 是否包含压缩图像 (可选, 实机数据通常为 True)
│
├── action                      # [Dataset] 动作序列
│                               # Shape: (T, Action_Dim)
│                               # 类型: float64
│                               # 说明: 包含机械臂的目标位置。对于双臂 ALOHA 通常是 14 (7+7)。
│
├── base_action                 # [Dataset] 移动底盘动作 (可选，通常仅实机数据包含)
│                               # Shape: (T, 2)
│                               # 类型: float64
│                               # 说明: [linear_vel, angular_vel]
│
├── observations                # [Group] 观测数据组
│   │
│   ├── qpos                    # [Dataset] 关节位置
│   │                           # Shape: (T, 14)
│   │                           # 类型: float64
│   │                           # 说明: 当前机械臂关节的实际位置。
│   │
│   ├── qvel                    # [Dataset] 关节速度
│   │                           # Shape: (T, 14)
│   │                           # 类型: float64
│   │                           # 说明: 当前机械臂关节的实际速度。
│   │
│   ├── effort                  # [Dataset] 关节力矩 (可选，通常仅实机数据包含)
│   │                           # Shape: (T, 14)
│   │                           # 类型: float64
│   │                           # 说明: 当前机械臂关节的力矩/电流。
│   │
│   └── images                  # [Group] 图像数据组
│       ├── <camera_name_1>     # [Dataset] 相机 1 图像
│       │                       # 未压缩: Shape (T, H, W, 3), uint8, RGB格式
│       │                       # 压缩:   Shape (T, Padded_Size), uint8 (JPEG 编码)
│       │
│       ├── <camera_name_2>     # [Dataset] 相机 2 图像
│       │                       # ...
│       │
│       └── ...                 # 其他相机 (取决于 constants.py 中的配置)
```

### 字段详细说明

*   **T**: 时间步长 (Timesteps)，即一个 episode 的总帧数。
*   **compress**: 如果此属性存在且为 `True`，则 `images` 下的数据存储为变长的 JPEG 字节流（为了对齐会进行 Padding），读取时需要使用 `cv2.imdecode` 进行解码。
*   **camera_names**: 常见的相机名称包括：
    *   **仿真**: `top`, `wrist`
    *   **实机**: `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`
