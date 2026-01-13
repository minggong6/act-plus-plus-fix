
import time
import os
import sys

# Ensure Robot.py is Importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Robot import RPC

class FR5Robot:
    def __init__(self, ip="192.168.58.2"):
        self.ip = ip
        print(f"[FR5Robot] 正在连接到 {ip}...")
        self.rpc = RPC(ip)
        self.last_gripper_target = None
        time.sleep(1) # 等待连接
        
        # 初始化机器人到良好状态
        self._init_robot()
        
    def _init_robot(self):
        # 1. 使能
        self.rpc.RobotEnable(1)
        time.sleep(1.0)
        # 2. 确保退出拖拽模式
        self.rpc.DragTeachSwitch(0)
        # 3. 重置错误
        self.rpc.ResetAllError()
        # 4. 激活夹爪
        self.rpc.ActGripper(1, 1)
        time.sleep(1.0)
        print("[FR5Robot] 机器人初始化完成.")

    def get_joint_state(self):
        """
        Returns:
            qpos (np.array): [rad] (6,)
            qvel (np.array): [rad/s] (6,)
        """
        # Read Position (Radians)
        ret = self.rpc.GetActualJointPosRadian()
        if isinstance(ret, int):
            # Error occurred (ret is error code)
            pos = [0.0] * 6
        else:
            # ret is tuple: (error_code, [p1, p2...])
            error, pos = ret
            if error != 0:
                pos = [0.0] * 6
        
        # Read Velocity (Degrees -> Radians)
        # GetActualJointSpeedsDegree always returns tuple (0, [v...]) in current Robot.py
        ret_val = self.rpc.GetActualJointSpeedsDegree()
        if isinstance(ret_val, int):
             vel_deg = [0.0] * 6
        else:
             _, vel_deg = ret_val
        
        return np.array(pos), np.deg2rad(np.array(vel_deg))

    def servo_j(self, joint_rad, duration=0.02):
        """
        Send ServoJ command.
        joint_rad: [rad]
        duration: [s] time to reach target (cmdT)
        """
        # Convert to Degrees
        joint_deg = np.rad2deg(joint_rad).tolist()
        
        # Axis pos (placeholder)
        axis_pos = [0.0, 0.0]
        
        # ServoJ(joint_pos, axisPos, acc, vel, cmdT, filterT, gain)
        # We only really care about position and time
        self.rpc.ServoJ(joint_deg, axis_pos, cmdT=duration)

    def move_gripper(self, width):
        """
        width: 0.0 (Closed) to 1.0 (Open)
        FR5: 0 (Closed) -> 100 (Open)
        """
        # Map 0-1 to 0-100
        target_pos = int(width * 100)
        target_pos = max(0, min(100, target_pos))
        
        # Optimization: only send command if target changed
        if self.last_gripper_target is not None and abs(target_pos - self.last_gripper_target) < 1:
            return
            
        self.last_gripper_target = target_pos
        
        # MoveGripper(index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque)
        # index=1
        self.rpc.MoveGripper(1, target_pos, 50, 50, 1000, 0, 1, 0, 0, 0) # Non-blocking (block=0)
        
    def get_gripper_state(self):
        """
        Returns: 
           width (0-1)
        """
        # Robot.py usually has struct for gripper
        # self.rpc.robot_state_pkg.gripper_position (byte)
        # Value is 0-100
        g_pos = self.rpc.robot_state_pkg.gripper_position
        return g_pos / 100.0
