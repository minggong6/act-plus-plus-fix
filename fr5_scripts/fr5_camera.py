import sys
import os
import time
import numpy as np
import cv2

# Add current dir to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Use provided utils for connection and path setup
from orbbecsdk_utils import add_path_pyorbbecsdk, connect_device, color_frame_to_bgr_img
add_path_pyorbbecsdk()
from pyorbbecsdk import *

class GeminiCamera:
    def __init__(self, name='gemini_wrist', serial_num=None, width=640, height=480):
        self.name = name
        self.serial_num = serial_num
        self.width = width
        self.height = height
        self.pipeline = None
        self.device = None
        
        self._init_camera()

    def _init_camera(self):
        print(f"[{self.name}] 正在连接奥比中光 Gemini 相机 (序列号: {self.serial_num})...")
        success, self.device = connect_device(self.serial_num)
        if not success:
            raise RuntimeError(f"无法连接到相机: {self.name} (序列号: {self.serial_num})")

        self.pipeline = Pipeline(self.device)
        config = Config()

        # Configure Color Stream
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            # Try to find matching profile
            # MJPG is usually preferred for USB bandwidth
            color_profile = profile_list.get_video_stream_profile(self.width, self.height, OBFormat.MJPG, 30)
            config.enable_stream(color_profile)
        except OBError as e:
            print(f"[{self.name}] 配置配置文件时出错: {e}")
            # Fallback (Auto)
            # config.enable_stream(profile_list.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30))
            raise e

        # 如果不需要可以通过跳过深度流来节省带宽 (ACT 主要使用 RGB)
        # 如果以后需要，取消注释:
        # depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # depth_profile = depth_profile_list.get_video_stream_profile(self.width, self.height, OBFormat.Y16, 30)
        # config.enable_stream(depth_profile)
        
        self.pipeline.start(config)
        print(f"[{self.name}] 管道已启动. {self.width}x{self.height}")

    def get_image(self):
        """
        返回 RGB 图像 (H, W, 3) np.uint8
        """
        # 最多等待 100ms
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            print(f"[{self.name}] 警告: 未接收到帧!")
            # 失败是返回黑色图像？还是重试？
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Helper returns BGR
        bgr_image = color_frame_to_bgr_img(color_frame)
        
        if bgr_image is None:
             return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Resize if necessary (though we requested specific size)
        if bgr_image.shape[0] != self.height or bgr_image.shape[1] != self.width:
             bgr_image = cv2.resize(bgr_image, (self.width, self.height))

        # Convert to RGB for ACT
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def close(self):
        if self.pipeline:
            self.pipeline.stop()
        if self.device:
            # self.device.close() # PyOrbbecSDK might not need explicit close if obj deleted
            pass
