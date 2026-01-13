# FR5 常量配置
import numpy as np

# 机器人 IP
FR5_IP = "192.168.58.2"

# 控制频率
DT = 0.02  # 50Hz

# 夹爪
# 0: Open, 100: Closed (通常 FR5 定义)
# 但是 ACT 期望 0-1.
# ALOHA: 0: Close, 1: Open.
# 我们需要映射.
# FR5 MoveGripper: 0 (Closed) -> 100 (Open) 通常? 
# 这里假设 FR5: 0=Closed, 100=Open.
# 所以 ACT 0->0, 1->100.

# 相机
CAMERA_NAMES = ['top', 'wrist']
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 机器人初始化
# J1, J2, J3, J4, J5, J6 (弧度)
# 对应 [0, -90, 0, -90, 0, 0] 度
START_JOINTS = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

# 相机序列号 (用实际序列号替换)
CAMERA_SERIALS = {
    'wrist': None, # 例如 "CK5678..."
    'top': None    # 例如 "AY1234..."
}
