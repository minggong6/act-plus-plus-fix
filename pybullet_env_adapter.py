import numpy as np
import pybullet as p
import collections

from scipy.spatial.transform import Rotation as R
import os
import sys
import cv2

# Try to import reward function, assuming sys.path is set correctly by the caller

# 定位到 pybullet/stage_1/FR_Gym 文件夹
fr_gym_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pybullet', 'stage_1', 'FR_Gym')

# 将该路径加入系统路径
if fr_gym_folder not in sys.path:
    sys.path.append(fr_gym_folder)

# 现在可以直接导入 reward 模块
try:
    from reward import grasp_reward
except (ImportError, ModuleNotFoundError) as e:
    print("fuckyou")
    def grasp_reward(env):
        return 0, {}

# Define TimeStep namedtuple to match dm_control
TimeStep = collections.namedtuple('TimeStep', ['step_type', 'reward', 'discount', 'observation'])

def capture_wrist_camera(p_client,
                         robot_id,
                         get_gripper_position_fn=None,
                         orientation_link_index=7,
                         width=640,
                         height=480,
                         fov=120,
                         near=0.01,
                         far=3.0,
                         back_offset=0.15,
                         up_offset=0.15,
                         forward_look_distance=0.3,
                         forward_axis=1,
                         up_axis=2,
                         renderer=None):
    """
    Universal function: Capture image from a camera placed near the gripper.
    """
    try:
        # 1) Get gripper center position
        if get_gripper_position_fn is not None:
            gripper_centre_pos = np.array(get_gripper_position_fn(), dtype=np.float64)
        else:
            Gripper_pos = p_client.getLinkState(robot_id, 6)[0]
            relative_position = np.array([0, 0, 0.169]) 
            rotation = R.from_quat(p_client.getLinkState(robot_id, orientation_link_index)[1])
            rotated_relative_position = rotation.apply(relative_position)
            gripper_centre_pos = np.array(Gripper_pos) + rotated_relative_position

        # 2) Get gripper orientation
        gripper_quat = p_client.getLinkState(robot_id, orientation_link_index)[1]
        rot_matrix = np.array(p_client.getMatrixFromQuaternion(gripper_quat)).reshape(3, 3)

        forward_dir = rot_matrix[:, 2]
        up_dir = -rot_matrix[:, 1]

        # 3) Calculate camera position and target
        cam_pos = gripper_centre_pos - forward_dir * back_offset + up_dir * up_offset
        cam_target = gripper_centre_pos + forward_dir * forward_look_distance

        # 4) Compute view and projection matrices
        view_matrix = p_client.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), up_dir.tolist())
        projection_matrix = p_client.computeProjectionMatrixFOV(fov, float(width) / float(height), near, far)

        # 5) Get image
        rend = renderer if renderer is not None else getattr(p_client, 'ER_BULLET_HARDWARE_OPENGL', None)
        img_tuple = p_client.getCameraImage(width, height, view_matrix, projection_matrix, renderer=rend)

        rgba = img_tuple[2]
        try:
            img_arr = np.frombuffer(rgba, dtype=np.uint8)
        except Exception:
            img_arr = np.array(rgba, dtype=np.uint8)
        try:
            img_arr = img_arr.reshape((height, width, -1))
        except Exception:
            return np.zeros((height, width, 4), dtype=np.uint8)

        return img_arr.astype(np.uint8)
    except Exception as e:
        print(f"Camera capture failed: {e}")
        return np.zeros((height, width, 4), dtype=np.uint8)

def capture_static_camera(p_client,
                          cam_pos,
                          cam_target,
                          up_vector=[0, 0, 1],
                          width=640,
                          height=480,
                          fov=60,
                          near=0.01,
                          far=10.0,
                          renderer=None):
    """
    Capture image from a static camera.
    """
    try:
        view_matrix = p_client.computeViewMatrix(cam_pos, cam_target, up_vector)
        projection_matrix = p_client.computeProjectionMatrixFOV(fov, float(width) / float(height), near, far)
        
        rend = renderer if renderer is not None else getattr(p_client, 'ER_BULLET_HARDWARE_OPENGL', None)
        img_tuple = p_client.getCameraImage(width, height, view_matrix, projection_matrix, renderer=rend)
        
        rgba = img_tuple[2]
        try:
            img_arr = np.frombuffer(rgba, dtype=np.uint8)
        except Exception:
            img_arr = np.array(rgba, dtype=np.uint8)
        try:
            img_arr = img_arr.reshape((height, width, -1))
        except Exception:
            return np.zeros((height, width, 4), dtype=np.uint8)

        return img_arr.astype(np.uint8)
    except Exception as e:
        print(f"Static camera capture failed: {e}")
        return np.zeros((height, width, 4), dtype=np.uint8)

class PyBulletACTAdapter:
    def __init__(self, env):
        self.env = env
        self.task = self 
        self.max_reward = 1 
        self._physics = self 

        # Disable PyBullet synthetic camera preview to avoid conflict/flickering
        if self.env.p.getConnectionInfo()['connectionMethod'] == p.GUI:
            self.env.p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self.env.p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self.env.p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    def reset(self):
        obs, info = self.env.reset()
        return self._get_dm_control_timestep(obs)

    def step(self, action):
        # action is 7-dim: 6 arm joints + 1 gripper
        # print(f"DEBUG: PyBullet Adapter Step Action: {action}")
        arm_action = action[:6] 
        gripper_action = action[6] 

        # Gripper logic: 0 is closed, 1 is open (assuming >0.5 is open command)
        # Adjust 0.075 based on your robot's max gripper width
        real_gripper_pos = 0.075 if gripper_action > 0.5 else 0.0
        
        # Execute control
        # Joints 1-6 for arm
        p.setJointMotorControlArray(self.env.fr5, [1, 2, 3, 4, 5, 6], p.POSITION_CONTROL, targetPositions=arm_action)
        # Joints 8, 9 for gripper (assuming these are the indices)
        p.setJointMotorControlArray(self.env.fr5, [8, 9], p.POSITION_CONTROL, targetPositions=[real_gripper_pos, real_gripper_pos])
        
        # Step simulation
        for _ in range(20): 
            self.env.p.stepSimulation()

        reward, info = grasp_reward(self.env)
        
        # Normalize reward for ACT evaluation (1.0 for success, 0.0 otherwise)
        # ACT expects reward == env_max_reward (which is 1) for success
        is_success = info.get('is_success', False)
        dm_reward = 1.0 if is_success else 0.0
        
        # Debug print to monitor reward and success status
        # print(f"Step reward: {reward}, Success: {is_success}, DM Reward: {dm_reward}")
        
        return self._get_dm_control_timestep(None, reward=dm_reward)

    def _get_dm_control_timestep(self, raw_obs, reward=0):
        # Get joint states
        joint_angles = []
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.env.fr5, i)
            joint_angles.append(joint_info[0]) 
        
        # Get gripper state
        gripper_state = p.getJointState(self.env.fr5, 8)[0]
        norm_gripper = gripper_state / 0.075 
        
        qpos = np.array(joint_angles + [norm_gripper], dtype=np.float32)
        
        # Get images
        # Assuming the model expects 'angle' camera
        # We need to check if get_gripper_position exists in env, otherwise pass None
        get_gripper_pos = getattr(self.env, 'get_gripper_position', None)
        wrist_img_rgba = capture_wrist_camera(self.env.p, self.env.fr5, get_gripper_position_fn=get_gripper_pos)
        wrist_img_rgb = wrist_img_rgba[:, :, :3]

        # Capture Top Camera
        # Positioned above the workspace, looking down
        top_cam_pos = [0.3, 0.5, 1.0] 
        top_cam_target = [0.3, 0.5, 0.0]
        top_img_rgba = capture_static_camera(self.env.p, top_cam_pos, top_cam_target, up_vector=[-1, 0, 0])
        top_img_rgb = top_img_rgba[:, :, :3]

        # Visualization using OpenCV (Independent windows)
        try:
            # Convert RGB to BGR for OpenCV
            wrist_bgr = cv2.cvtColor(wrist_img_rgb, cv2.COLOR_RGB2BGR)
            top_bgr = cv2.cvtColor(top_img_rgb, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("Wrist Camera", wrist_bgr)
            cv2.imshow("Top Camera", top_bgr)
            cv2.waitKey(1)
        except Exception:
            pass

        observation = {
            'qpos': qpos,
            'images': {
                'wrist': wrist_img_rgb,
                'top': top_img_rgb
            } 
        }

        return TimeStep(step_type=None, reward=reward, discount=1.0, observation=observation)

    def render(self, height=480, width=640, camera_id=None):
        if camera_id == 'top':
             top_cam_pos = [0.3, 0.5, 1.0] 
             top_cam_target = [0.3, 0.5, 0.0]
             top_img_rgba = capture_static_camera(self.env.p, top_cam_pos, top_cam_target, up_vector=[-1, 0, 0], width=width, height=height)
             return top_img_rgba[:, :, :3]
        
        get_gripper_pos = getattr(self.env, 'get_gripper_position', None)
        wrist_img_rgba = capture_wrist_camera(self.env.p, self.env.fr5, get_gripper_position_fn=get_gripper_pos, width=width, height=height)
        return wrist_img_rgba[:, :, :3]
