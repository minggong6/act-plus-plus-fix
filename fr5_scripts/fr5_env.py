import time
import numpy as np
import collections
import dm_env
import sys
import os

# Ensure local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from constants import FR5_IP, DT, START_JOINTS, CAMERA_SERIALS
from fr5_robot import FR5Robot
from fr5_camera import GeminiCamera

class FR5RealEnv:
    """
    用于真实 FR5 机器人操作的环境。
    将特定的 FR5 SDK + Orbbec SDK 适配到 ACT 环境接口。
    """
    def __init__(self, setup_robot=True, setup_camera=True):
        # 1. 初始化机器人
        self.robot = None
        if setup_robot:
            print("[FR5RealEnv] 正在配置机器人...")
            self.robot = FR5Robot(ip=FR5_IP)
        
        # 2. 初始化相机
        # 暂时假设单相机 ('wrist' 或 'top' 或两者都有?)
        # 集成计划说 "camera is usb communication gemini335" (单数还是复数?)
        # testcamera.py 只显示了 1 个设备。
        # 让我们假设 1 个相机，映射到 'wrist'（单臂标准）
        # 或者如果有的话支持 2 个相机。
        # 我将默认实现对 1 个相机 'wrist' 的支持。
        self.cameras = {}
        if setup_camera:
            print("[FR5RealEnv] 正在配置相机...")
            try:
                # 基于常量初始化相机
                for cam_name, cam_serial in CAMERA_SERIALS.items():
                    if cam_serial is None and cam_name != 'wrist': 
                         # 跳过未定义的相机，除非是默认的 'wrist'，因为可能会自动识别第一个
                         continue
                         
                    print(f"[FR5RealEnv] 正在初始化 {cam_name} 相机...")
                    self.cameras[cam_name] = GeminiCamera(
                        name=cam_name, 
                        serial_num=cam_serial
                    )
                
                # 如果列表中没有定义相机但代码依赖默认值
                if not self.cameras:
                     print("[FR5RealEnv] 未配置特定相机。默认使用单个 'wrist' 相机。")
                     self.cameras['wrist'] = GeminiCamera(name='wrist_cam')

            except Exception as e:
                print(f"[FR5RealEnv] 相机配置失败: {e}")

    def get_qpos(self):
        """
        Return: np.array (7,) -> [j1...j6 (rad), gripper (0-1)]
        """
        if not self.robot:
            return np.zeros(7)
        
        joint_pos, _ = self.robot.get_joint_state() # (6,)
        gripper_pos = self.robot.get_gripper_state() # Scalar 0-1
        
        return np.concatenate([joint_pos, [gripper_pos]])

    def get_qvel(self):
        """
        Return: np.array (7,) -> [v1...v6 (rad/s), gripper_vel (0)]
        """
        if not self.robot:
            return np.zeros(7)

        _, joint_vel = self.robot.get_joint_state() # (6,)
        # Gripper velocity not easily available/needed, return 0
        return np.concatenate([joint_vel, [0.0]])

    def get_images(self):
        """
        Return: dict {name: image(H,W,3)}
        """
        image_dict = {}
        for name, cam in self.cameras.items():
            image_dict[name] = cam.get_image()
        return image_dict

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['images'] = self.get_images()
        # obs['effort'] = ... # Optional
        return obs

    def get_reward(self):
        return 0

    def reset(self):
        # 重置时物理移动机器人以确保起始状态一致
        if self.robot:
             print("[FR5RealEnv] 正在重置机器人到起始位置...")
             self.robot.servo_j(np.array(START_JOINTS), duration=2.0)
             self.robot.move_gripper(1.0) # 打开夹爪
             time.sleep(2.0) # 等待移动完成 (servo_j 不阻塞但我们想在这里等待)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        """
        action: (7,) [j1...j6, gripper]
        """
        start_time = time.time()
        
        if self.robot:
            # 1. Parse Action
            # Assumes action is 7-dim
            joint_action = action[:6]
            gripper_action = action[6] # 0-1
            
            # 2. Send Commands
            # ServoJ for smooth motion
            self.robot.servo_j(joint_action, duration=DT)
            
            # Move Gripper
            # (Optimization: check if value changed significantly to avoid spamming gripper cmds)
            self.robot.move_gripper(gripper_action)

        # 3. Maintain Frequency (Sync)
        # We want to wait until DT has passed since start_time
        # However, getting observation also takes time.
        # Usually: Send Action -> Sleep Remainder -> Get Obs
        
        elapsed = time.time() - start_time
        sleep_time = max(0, DT - elapsed)
        time.sleep(sleep_time)

        # 4. Get Observation (New state after action)
        obs = self.get_observation()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

