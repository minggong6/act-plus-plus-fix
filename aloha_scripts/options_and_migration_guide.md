# Aloha 脚本指南与机器人迁移教程

本文档旨在介绍 `aloha_scripts/` 目录中各个文件的作用，并提供分步指南，说明如何迁移此代码库以支持其他类型的机械臂。

## 1. 文件结构概览

`aloha_scripts/` 文件夹包含了真实机器人执行、数据收集和遥操作的核心逻辑。它基于 ALOHA 系统（用于双臂遥操作的低成本开源硬件系统）。

### 核心组件 (Core Components)
*   **`constants.py`**: 中央配置文件。它包含：
    *   任务定义（数据集路径、剧集长度）。
    *   硬件常量（相机名称、关节限制）。
    *   运动学映射函数（例如，将 Master 手柄状态映射到 Puppet 机械臂状态）。
*   **`real_env.py`**: 与 `dm_env` 兼容的主环境类 (`RealEnv`)。
    *   它处理机器人（默认为 Interbotix ViperX）和相机的初始化。  ## "相机要改成gemini335的库"
    *   它构建观测字典 (`qpos`, `qvel`, `images`)。
*   **`fr5_env.py`**: 专门用于 FR5 机器人的替代环境。
    *   在使用 FR5 硬件时，它可以作为 `RealEnv` 的直接替代品。
*   **`fr5_robot.py`**: FR5 机器人的底层驱动封装。
    *   它将特定于供应商的 SDK（来自 `Robot.py` 的 RPC 调用）抽象为标准接口（`get_joint_state`, `servo_j`, `move_gripper`）。
*   **`record_episodes.py`**: 数据收集的主要脚本。
    *   它运行“开幕式 (Opening Ceremony)”（同步 Master 和 Puppet 机器人）。
    *   它循环捕获观测结果并将它们保存到 HDF5 文件中以供训练。

### 工具与辅助脚本 (Utilities & Helpers)
*   **`robot_utils.py`**: 包含辅助类和函数：
    *   `ImageRecorder`: 来自 Realsense 相机的多线程图像捕获。### 改成gemini 335
    *   `Recorder`: 记录关节状态。   关节数目缩小至7个关节
    *   `setup_master_bot` / `setup_puppet_bot`: PID 和操作模式配置（主要用于 Interbotix）。
*   **`dynamixel_client.py`**: 用于控制 Dynamixel 电机（通常用于夹爪或自定义执行器）的客户端。  fr5不需要这个，我们需要一个基于fr5的sdk的控制
*   **`visualize_episodes.py`**: 用于检查记录的 HDF5 数据集的工具（可视化图像流和关节轨迹）。
*   **`one_side_teleop.py`**: 一个简化的脚本，用于在不记录数据的情况下测试单臂对的遥操作。用于调试硬件连接。
*   **`sleep.py`**: 一个实用脚本，用于将机器人置于安全/睡眠姿势。
*   **`waypoint_control.py`** & **`example_waypoint_pid.py`**: 用于测试简单 PID 轨迹跟踪（非神经网络控制）的脚本。

---

## 2. 机器人迁移指南

如果你想更换其他机械臂（例如 UR5, Kinova, Aubo 或自定义机械臂），请按照以下步骤将其集成到 ACT/ALOHA 框架中。

### 步骤 1：创建底层驱动封装
你需要编写一个类，将你的机器人特定 SDK 抽象为训练代码所期望的接口。
创建一个新文件，例如 `my_robot.py`。

**要求：**
你的类必须至少实现以下方法：
```python
class MyRobot:
    def __init__(self, ip_address, ...):
        # 初始化与机器人 SDK 的连接
        pass

    def get_joint_state(self):
        """
        返回:
            joint_pos (np.array): 6 个关节角度，单位：弧度 (RADIANS)
            joint_vel (np.array): 6 个关节速度，单位：弧度/秒 (RADIANS/SEC)
        """
        pass

    def servo_j(self, joint_pos_rad, duration=0.01):
        """
        发送关节位置指令以立即执行（伺服模式）。
        参数:
            joint_pos_rad: 目标关节位置
            duration: 预期执行时间（控制循环的时间步长 dt）
        """
        pass

    def get_gripper_state(self):
        """
        返回:
            float: 归一化的夹爪位置 [0, 1] (0=关闭, 1=打开)
        """
        pass

    def move_gripper(self, pos_normalized):
        """
        参数:
            pos_normalized: 目标位置 [0, 1]
        """
        pass
```
*参考 `aloha_scripts/fr5_robot.py` 作为真实示例。*

### 步骤 2：创建一个新的环境类
创建一个新的环境文件，例如 `my_robot_env.py`（或调整 `real_env.py`）。

1.  **初始化 (Init)**: 实例化左右臂的 `MyRobot` 类。
2.  **获取 Qpos/Qvel**: 实现 `get_qpos()` 和 `get_qvel()`。
    *   **关键点**: 标准格式必须是 `[左臂关节(6), 左夹爪(1), 右臂关节(6), 右夹爪(1)]`。
    *   确保单位一致（手臂使用弧度，夹爪使用 0-1 归一化值）。
3.  **图像 (Images)**: 如果你使用的是 Realsense 相机，可以使用 `robot_utils.py` 中的 `ImageRecorder`，或者实现你自己的图像捕获逻辑。

```python
from my_robot import MyRobot
from robot_utils import ImageRecorder

class MyRobotEnv:
    def __init__(self):
        self.puppet_bot_left = MyRobot(ip="192.168.1.100")
        self.puppet_bot_right = MyRobot(ip="192.168.1.101")
        self.image_recorder = ImageRecorder(init_node=False)

    def get_qpos(self):
        l_q, _ = self.puppet_bot_left.get_joint_state()
        r_q, _ = self.puppet_bot_right.get_joint_state()
        l_g = self.puppet_bot_left.get_gripper_state()
        r_g = self.puppet_bot_right.get_gripper_state()
        return np.concatenate([l_q, [l_g], r_q, [r_g]])
    
    # ... 实现 get_qvel, get_observation, reset ...
```

### 步骤 3：更新常量和配置
修改 `aloha_scripts/constants.py`:

1.  **关节限制**: 如果你的机器人有不同的物理限制，或者“Master”机器人（遥操作输入）与“Puppet”机器人的映射关系不同，你需要定义映射函数。
    *   例如：`MASTER2PUPPET_JOINT_FN`。如果你的 Master 是 Interbotix 而 Puppet 是 UR5，你可能需要缩放输入值。
2.  **夹爪归一化**: 确保 0 代表关闭，1 代表打开。如果你的原始夹爪数值是 0-255，请在 `constants.py` 中添加归一化函数。
3.  **任务配置**: 在 `TASK_CONFIGS` 中添加你的新任务条目，并确保使用正确的相机名称。

### 步骤 4：适配数据收集脚本 (`record_episodes.py`)
`record_episodes.py` 脚本通常硬编码了“开幕式”（将机器人移动到开始姿态）。

1.  **导入你的环境**: 将 `from real_env import make_real_env` 改为使用你新的 `MyRobotEnv`。
2.  **修改 `opening_ceremony`**:
    *   现有代码假定了 Interbotix 的动力学特性（设置操作模式、重启电机）。
    *   将其替换为你机器人的特定启动流程（例如，`robot.unlock_brakes()`, `robot.go_to_home()`）。
3.  **Master-Puppet 循环**: 确保 `get_action(master_bot)` 返回的指令与你的 `MyRobot.servo_j` 兼容。

### 步骤 5：Master 手臂的考量
如果你仍然使用 Interbotix 手臂作为“Master”（控制器），你可以保留 `master_bot` 的逻辑。如果你要更改输入设备（例如改为 VR 控制器或定制的外骨骼）：
1.  你需要一个 `get_action()` 函数来从你的输入设备读取数据。
2.  它应该返回 Puppet 机器人的目标关节位置。
