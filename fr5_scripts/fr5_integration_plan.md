# ACT 项目 FR5 机器人集成方案

本文档概述了将 FR5 机器人集成到 ACT (Action Chunking with Transformers) 代码库中的计划，以便在 FR5 硬件上执行策略。

## 1. 概述

目标是使用提供的 `Robot.py` SDK 替换现有的 Interbotix 机器人控制逻辑（在 `aloha_scripts/real_env.py` 中使用），改为 FR5 机器人控制逻辑。我们将创建一个新的环境类 `FR5RealEnv`，它模仿 `RealEnv` 的接口，但与 FR5 机器人进行通信。
fr5是使用sdk和网路通信的，不要需要使用ros，
相机是usb通信的gemini335，其中关键的sdk和工具我已在‘fr5_scripts/Robot.py’，
fr5_scripts/testcamera.py给出部分代码用例
相机需要使用480*640的格式
你需要参考aloha_scripts里的代码给出怎么真机操纵他实现act模仿学习，单独在fr5_scripts文件夹里实现，新做一套像aloha的代码

FR5 机器人的确切 IP 地址。192.168.58.2

## 2. 前置条件

*   **硬件**: 1台 FR5 机器人（假设为单臂设置）及其控制器。
*   **网络**: 工作站必须能够 ping 通 FR5 控制器。这个没问题
*   **软件**:
    *   `Robot.py`: FR5 Python SDK（已提供）。
    *   现有的 ACT 代码库。




1.  `aloha_scripts/fr5_robot.py`: `Robot.py` 的封装类，提供与 ACT 要求兼容的简化接口。
2.  `aloha_scripts/fr5_env.py`: 主要的环境类 `FR5RealEnv`，实现 `get_qpos`、`get_qvel`、`step` 和 `get_images` 方法。

## 4. 实现细节

### 4.1. `FR5Robot` 类 (`aloha_scripts/fr5_robot.py`)

此类将封装来自 `Robot.py` 的底层 `RPC` 调用。

*   **初始化**:
    *   使用 `RPC(ip)` 连接到机器人。
    *   如有必要，初始化夹爪配置。
*   **方法**:
    *   `get_joint_state()`: 返回当前的关节位置（弧度）和速度。
        *   *注意*: FR5 SDK 可能返回角度，因此需要转换为弧度。
    *   `servo_joint_command(target_pos, duration)`: 发送 `ServoJ` 命令以平滑地将关节移动到 `target_pos`。
    *   `set_gripper_position(pos)`: 发送 `MoveGripper` 命令。
        *   需要将归一化的动作 [0, 1] 映射到 FR5 夹爪范围（例如 0-100 或 0-255）。


### 5.1. `Robot.py` 的封装

```python
import numpy as np
from Robot import RPC  # 假设 Robot.py 在 python 路径中

class FR5Robot:
    def __init__(self, ip):
        self.rpc = RPC(ip)
        self.rpc.connect_to_robot()
        # 确保机器人处于正确模式（例如，已使能）
        
    def get_joint_positions(self):
        # 如果可能，直接访问机器人状态以提高速度，或使用 GetActualJointPosDegree
        # 将角度转换为弧度
        # 返回形状为 (6,) 的 numpy 数组
        pass

    def servo_j(self, joint_positions):
        # 将弧度转换为角度
        # 调用 self.rpc.ServoJ(...)
        pass
        
    def move_gripper(self, position):
        # 将 0-1 映射到夹爪范围
        # 调用 self.rpc.MoveGripper(...)
        pass
```
