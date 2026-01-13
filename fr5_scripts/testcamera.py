import time
import sys
import os

# 导入阿凯写的Orbbec工具库
# 确保orbbec_utils.py跟你目前所执行的脚本在同一级目录下
from orbbecsdk_utils import *
# 添加Python Path
add_path_pyorbbecsdk()

from pyorbbecsdk import *

# 设置日志等级为ERROR
# 这样不会频繁的打印日志信息
ctx = Context()
ctx.set_logger_level(OBLogLevel.ERROR)

# 查询设备列表
device_list = ctx.query_devices()
# 获取设备个数
device_num = device_list.get_count()

if device_num == 0:
    print("[ERROR]没有设备连接")
else:
    print(f"检测到{device_num}个设备")
    # 获取特定索引下的设备序列号
    serial_num = device_list.get_device_serial_number_by_index(0)
    print(f"设备序列号为: {serial_num}")

connect_device()
time.sleep(1)