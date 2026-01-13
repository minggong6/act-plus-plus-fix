# 模仿学习机械臂“固定轨迹”问题排查分析报告

## 1. 问题描述
在 ACT (Action Chunking with Transformers) 模仿学习评估过程中，出现以下现象：
*   **Episode 1**: 机械臂成功完成任务（或按照预期轨迹运动）。
*   **Episode 2+**: 目标物体位置发生变化（预期），但机械臂仍然严格重复 Episode 1 的运动轨迹，导致抓取失败。
*   **表现特征**: 模型表现出“盲目”的特性，似乎忽略了当前的视觉观测（Visual Observation）。

## 2. 潜在原因分析 (Root Cause Analysis)

### 2.1. 观测数据“冻结” (Observation Freeze) - **高风险**
这是最常见的原因。虽然环境重置了，但传给模型的图像数据可能没有更新。
*   **现象**: 在 Episode 2 中，模型接收到的 `image` 仍然是 Episode 1 结束时或开始时的画面。
*   **原因**: PyBullet 的相机渲染器（Renderer）可能卡死，或者代码中获取图像的变量没有在循环内正确刷新。
*   **后果**: 如果模型看到的画面没变，它输出的动作自然也不会变。

### 2.2. 训练数据集缺乏多样性 (Dataset Bias) - **高风险**
*   **现象**: 训练集中的所有轨迹，目标物体都在同一个位置（或变化极小）。
*   **原因**: 数据采集阶段脚本设置问题，导致 `reset()` 时没有随机化目标位置。
*   **后果**: 模型学会了 $f(image) = fixed\_trajectory$。因为它发现不管图像怎么变（或者图像根本没变），最优解都是那条固定的轨迹。模型退化为“过拟合”的开环控制器。

### 2.3. 坐标系或归一化错误 (Normalization/Frame Issues)
*   **现象**: 动作空间定义有问题。
*   **原因**: 如果训练使用的是绝对坐标，而测试时目标移动了，但模型输出的绝对坐标还是原来的区域。
*   **后果**: 机械臂去抓原来的位置。

### 2.4. 环境重置逻辑失效 (Reset Failure)
*   **现象**: 代码调用了 `env.reset()`，但物理仿真中目标物体并没有移动。
*   **原因**: `Fr5_env.py` 中的 `reset` 函数随机化逻辑未生效，或者随机种子（Seed）被固定了。

---

## 3. 排查步骤与解决方案

### 步骤 1: 验证数据集多样性 (Check Dataset Variance)
我们需要确认喂给模型的数据本身是包含不同目标位置的。如果数据本身就是重复的，模型不可能学会泛化。

**操作**: 运行提供的 `check_dataset_variance.py` 脚本。
*   **检查点**: 查看不同 Episode 间，第一帧图像是否不同？查看 `qpos`（机械臂起始位置）是否有微小变化？查看 `action`（轨迹）是否不同？

### 步骤 2: 验证评估时的视觉输入 (Verify Evaluation Inputs)
我们需要确认在 `imitate_episodes.py` 运行时，模型真的“看”到了新的位置。

**操作**: 在 `imitate_episodes.py` 的评估循环中保存图像。
*   **修改代码**:
    ```python
    # 在 get_image 后添加保存逻辑
    if t == 0: # 只看每个 Episode 的第一帧
        import cv2
        # 假设 curr_image 是 (channel, height, width)
        img_to_save = curr_image[0].permute(1, 2, 0).cpu().numpy() 
        cv2.imwrite(f'debug_eval_ep{rollout_id}_start.png', img_to_save * 255)
    ```
*   **判定**: 打开生成的图片，对比 `debug_eval_ep0_start.png` 和 `debug_eval_ep1_start.png`。如果两张图里的杯子位置一模一样，说明环境重置有问题；如果图一模一样且连噪声都一样，说明相机捕捉代码有问题。

### 步骤 3: 检查环境重置代码 (Check Environment Reset)
检查 `pybullet/stage_1/FR_Gym/Fr5_env.py`。

*   **代码审查**:
    ```python
    def reset(self, seed=None, options=None):
        # ...
        self.goalx = np.random.uniform(0.1, 0.35, 1)[0] # 确认这里是否真的随机了
        # ...
        p.resetBasePositionAndOrientation(self.target, ...)
    ```
*   **注意**: 如果在 `imitate_episodes.py` 开头设置了 `set_seed(0)`，且 `env.reset()` 内部使用了受该种子控制的随机生成器，可能会导致每次运行生成的随机序列是一样的。但在同一个脚本的连续 `rollout` 中，通常应该会变。

### 步骤 4: 检查 Temporal Aggregation (时间聚合)
虽然可能性较小，但如果 `all_time_actions` 缓存没有在 Episode 之间清空，旧的动作可能会干扰新的 Episode。

*   **检查**: 确认 `imitate_episodes.py` 中，在 `for rollout_id in range(num_rollouts):` 循环内部，是否重置了 `all_time_actions`。
    *   *现状*: 代码中 `all_time_actions` 是在 `if temporal_agg:` 块内初始化的，看起来是在 `rollout_id` 循环内部（如果 `imitate_episodes.py` 结构正确的话）。需要确认它是否在 `rollout` 循环内被重新置零。

## 4. 结论
建议优先执行 **步骤 1** 和 **步骤 2**。
1. 如果数据集全是重复轨迹 -> **重新采集数据**。
2. 如果评估时图像没变 -> **修复相机代码或环境重置逻辑**。
3. 如果图像变了，数据也有多样性，但还是走老路 -> **模型欠拟合或训练参数问题（如权重衰减过大，忽略了图像特征）**。
