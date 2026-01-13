import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def check_dataset_variance(dataset_dir, num_episodes_to_check=5):
    """
    检查数据集的前几个 Episode，对比它们的图像和动作是否相同。
    """
    print(f"正在检查数据集目录: {dataset_dir}")
    
    first_images = []
    all_actions = []
    
    for i in range(num_episodes_to_check):
        # 尝试两种文件名格式
        file_path_1 = os.path.join(dataset_dir, f'episode_{i}.hdf5')
        file_path_2 = os.path.join(dataset_dir, f'episode_{i:05d}.hdf5') # 补零格式
        
        if os.path.exists(file_path_1):
            file_path = file_path_1
        elif os.path.exists(file_path_2):
            file_path = file_path_2
        else:
            print(f"文件不存在: {file_path_1} 或 {file_path_2}")
            continue
            
        with h5py.File(file_path, 'r') as root:
            # 1. 获取第一帧图像 (假设相机名为 'top' 或 'angle')
            # 根据你的 constants.py，相机名可能是 'top' 或 'wrist'
            if 'images' in root['observations']:
                img_group = root['observations']['images']
                cam_name = list(img_group.keys())[0] # 取第一个相机
                first_frame = img_group[cam_name][0] # 取第0步的图像
                first_images.append(first_frame)
                print(f"Episode {i}: 获取到相机 {cam_name} 的第一帧图像，Shape: {first_frame.shape}, Type: {first_frame.dtype}, Max: {np.max(first_frame)}")
            
            # 2. 获取动作序列
            actions = root['action'][:]
            all_actions.append(actions)
            print(f"Episode {i}: 动作序列长度: {len(actions)}")
            print(f"Episode {i}: 前5步动作:\n{actions[:5]}")

    # 对比分析
    if len(first_images) >= 2:
        diff_img = np.abs(first_images[0].astype(float) - first_images[1].astype(float))
        mean_diff = np.mean(diff_img)
        print(f"\n=== 图像差异分析 (Ep0 vs Ep1) ===")
        print(f"像素平均差异值: {mean_diff:.4f}")
        if mean_diff < 1.0:
            print("警告: 两个 Episode 的初始图像几乎完全相同！这表明目标物体位置可能没有变化。")
        else:
            print("图像存在差异，目标位置可能已改变。")

    if len(all_actions) >= 2:
        # 截断到最短长度进行比较
        min_len = min(len(all_actions[0]), len(all_actions[1]))
        # 只比较前100步或者最短长度
        compare_len = min(100, min_len)
        
        diff_action = np.abs(all_actions[0][:compare_len] - all_actions[1][:compare_len]) 
        mean_action_diff = np.mean(diff_action)
        print(f"\n=== 动作差异分析 (Ep0 vs Ep1) ===")
        print(f"动作平均差异值: {mean_action_diff:.4f}")
        if mean_action_diff < 1e-4:
            print("警告: 两个 Episode 的动作轨迹几乎完全相同！数据集缺乏多样性。")
        else:
            print("动作轨迹存在差异，数据具有多样性。")

if __name__ == '__main__':
    # 默认指向你的数据集路径
    DATA_DIR = '/home/dianrobot/PycharmProjects/act-plus-plus-fixed/data_set/sim_fr5_task'
    check_dataset_variance(DATA_DIR)
