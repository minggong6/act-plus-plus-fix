import time
import numpy as np
from fr5_robot import FR5Robot
from fr5_camera import GeminiCamera
from fr5_env import FR5RealEnv

# 1. 测试 FR5Robot 单独功能
def test_robot():
    print("\n=== 测试 FR5Robot 机械臂接口 ===")
    robot = FR5Robot()
    print("当前关节状态 (弧度):", robot.get_joint_state()[0])
    print("当前关节速度 (rad/s):", robot.get_joint_state()[1])
    print("当前夹爪开度 (0-1):", robot.get_gripper_state())
    print("测试 ServoJ: 让J1+0.1rad (小幅度)")
    qpos, _ = robot.get_joint_state()
    qpos[0] += 0.1
    robot.servo_j(qpos)
    time.sleep(1)
    print("测试夹爪开合: 先开后关")
    robot.move_gripper(1.0)
    time.sleep(1)
    robot.move_gripper(0.0)
    time.sleep(1)
    print("FR5Robot 单元测试完成\n")

# 2. 测试 GeminiCamera 单独功能
def test_camera():
    print("\n=== 测试 GeminiCamera 相机接口 ===")
    cam = GeminiCamera()
    img = cam.get_image()
    print("相机采集到的图像 shape:", img.shape)
    import cv2
    cv2.imwrite("test_gemini_rgb.png", img[..., ::-1]) # 保存为BGR
    print("已保存 test_gemini_rgb.png\n")
    cam.close()

# 3. 测试 FR5RealEnv 环境整体功能
def test_env():
    print("\n=== 测试 FR5RealEnv 环境整体接口 ===")
    env = FR5RealEnv(setup_robot=True, setup_camera=True)
    obs = env.get_observation()
    print("观测 qpos:", obs['qpos'])
    print("观测 qvel:", obs['qvel'])
    print("观测 images keys:", list(obs['images'].keys()))
    for k, v in obs['images'].items():
        print(f"  {k} shape: {v.shape}")
    print("执行 step: 让J1+0.1rad, 夹爪开")
    action = obs['qpos'].copy()
    action[0] += 0.1
    action[-1] = 1.0
    ts = env.step(action)
    print("step 后观测 qpos:", ts.observation['qpos'])
    print("FR5RealEnv 测试完成\n")

if __name__ == "__main__":
    print("==== FR5 全链路测试脚本 ====")
    test_robot()
    test_camera()
    test_env()
    print("==== 所有测试完成 ====")
