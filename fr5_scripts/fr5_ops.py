import argparse
import time
import sys
import os

# 确保当前目录在 sys.path 中，以便能导入 Robot.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from fr5_scripts.Robot import RPC
except ImportError:
    print("Error: Could not import 'RPC' from 'Robot.py'. Please ensure Robot.py is in the same directory.")
    sys.exit(1)

class FR5:
    def __init__(self, ip="192.168.58.2"):
        self.ip = ip
        print(f"Connecting to robot at {ip}...")
        try:
            self.robot = RPC(self.ip)
            # 给一点时间让后台线程建立连接
            time.sleep(1)
            print("RPC object created.")
        except Exception as e:
            print(f"Failed to create RPC object: {e}")
            sys.exit(1)

    def init_robot(self):
        """
        机械臂初始化：
        1. 清除错误
        2. 使能机械臂
        3. 切换到运动模式（非拖动示教）
        """
        print(">>> [Function] init_robot")
        
        # 1. Reset Errors
        print("  Executing ResetAllError()...")
        ret = self.robot.ResetAllError()
        print(f"  Result: {ret}")
        
        # 2. Enable Robot (1: Enable)
        print("  Executing RobotEnable(1)...")
        ret = self.robot.RobotEnable(1)
        print(f"  Result: {ret}")
        print("  Waiting 3s for enable...")
        time.sleep(3) 

        # 3. Ensure Drag Mode is OFF (0)
        print("  Executing DragTeachSwitch(0) to exit drag mode...")
        ret = self.robot.DragTeachSwitch(0)
        print(f"  Result: {ret}")
        
    def init_gripper(self):
        """
        夹爪初始化：Activate the gripper
        """
        print(">>> [Function] init_gripper")
        
        # ActGripper(index, action) -> index=1, action=1 (activate)
        index = 1
        act = 1 
        print(f"  Executing ActGripper({index}, {act})...")
        ret = self.robot.ActGripper(index, act)
        print(f"  Result: {ret}")
        print("  Waiting 2s for activation...")
        time.sleep(2) 

    def open_gripper(self):
        print(">>> [Function] open_gripper")
        # MoveGripper(index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque)
        # pos=100 usually means fully open
        print("  Moving gripper to 100 (Open)...")
        self.robot.MoveGripper(1, 100, 50, 50, 10000, 1, 1, 0, 0, 0)
        time.sleep(1)

    def close_gripper(self):
        print(">>> [Function] close_gripper")
        # pos=0 usually means fully closed
        print("  Moving gripper to 0 (Closed)...")
        self.robot.MoveGripper(1, 0, 50, 50, 10000, 1, 1, 0, 0, 0)
        time.sleep(1)

    def reset(self):
        """
        机械臂自动复位：Move to a safe home position
        """
        print(">>> [Function] reset (Move to Home)")
        
        # Define a safe home position (Joint angles in degrees)
        # [J1, J2, J3, J4, J5, J6]
        # J2=-90, J4=-90 is a standard "ready" pose for many 6-DOF robots
        home_joints = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]
        
        print(f"  Target Joint Position: {home_joints}")
        print("  Executing MoveJ...")
        
        # MoveJ(joint_pos, tool, user, vel=...)
        # Note: desc_pos uses default values in wrapper if omitted
        ret = self.robot.MoveJ(home_joints, 0, 0, vel=30.0)
        print(f"  Result: {ret}")
        
        if ret == 0:
            print("  Command sent. Waiting 5s for movement...")
            time.sleep(5)
            print("  Movement period ended.")
        else:
            print("  Failed to send MoveJ command.")

def main():
    parser = argparse.ArgumentParser(description="FR5 Robot Control Script using New Python SDK")
    parser.add_argument('--ip', type=str, default="192.168.58.2", help="Robot IP address")
    parser.add_argument('--action', type=str, required=True, 
                        choices=['init_robot', 'init_gripper', 'reset', 'test_all'],
                        help="Action: init_robot, init_gripper, reset, or test_all")

    args = parser.parse_args()

    fr5 = FR5(ip=args.ip)

    if args.action == 'init_robot':
        fr5.init_robot()
    elif args.action == 'init_gripper':
        fr5.init_gripper()
        print("  Testing gripper open/close loop...")
        fr5.open_gripper()
        fr5.close_gripper()
        fr5.open_gripper()
    elif args.action == 'reset':
        fr5.reset()
    elif args.action == 'test_all':
        print("=== Starting Full Test Sequence ===")
        fr5.init_robot()
        time.sleep(1)
        
        try:
            fr5.init_gripper()
            fr5.open_gripper()
            fr5.close_gripper()
            fr5.open_gripper()
        except Exception as e:
            print(f"Gripper test failed (might be optional): {e}")

        time.sleep(1)
        fr5.reset()
        print("=== Test Sequence Complete ===")

if __name__ == "__main__":
    main()
