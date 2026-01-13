import h5py
import numpy as np

# 假设您的 HDF5 文件名为 'your_file.hdf5'
file_name = 'episode_1.hdf5'

try:
    # 1. 打开文件 (只读模式 'r')
    # 使用 'with' 语句可以确保文件在操作完成后自动关闭
    with h5py.File(file_name, 'r') as f:
        print(f"--- 成功打开文件：{file_name} ---")

        # 2. 查看根目录下的所有键 (Groups 和 Datasets 的名称)
        print("\n🔑 文件根目录下的键 (Groups 和 Datasets):")
        # f.keys() 返回一个字典键视图对象，通常转换为 list 打印
        print(list(f.keys()))


        # 3. 遍历并打印所有对象
        # 这是一个递归函数，用于探索文件的层次结构
        def print_hdf5_item(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                # 打印 Dataset 的名称、形状和数据类型
                print(f"📊 Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")

                # 也可以打印 Dataset 的属性 (Attributes)
                if obj.attrs:
                    print(f"   - Attributes: {list(obj.attrs.keys())}")
            # print(f"Name: {name}, Type: {type(obj)}") # 打印所有对象的类型


        print("\n🔍 文件内容结构:")
        f.visititems(print_hdf5_item)  # visititems 遍历文件中的所有对象

        # 4. 读取特定的 Dataset
        # 假设文件有一个名为 'data/image_data' 的 Dataset
        dataset_path = '/data/image_data'

        # 检查 Dataset 是否存在
        if dataset_path in f:
            data = f[dataset_path]

            # 读取并打印 Dataset 的信息
            print(f"\n✅ Dataset '{dataset_path}' 的信息:")
            print(f"   - 形状 (Shape): {data.shape}")
            print(f"   - 数据类型 (Dtype): {data.dtype}")

            # 读取数据。使用 [:] 可以将 HDF5 数据集加载为 NumPy 数组
            # **注意：对于非常大的数据集，不要一次性加载所有数据！**
            # 如果数据集太大，可以只加载部分数据，例如： data[0:10]
            if np.prod(data.shape) < 1000000000000:  # 假设总元素少于 100 时才打印
                print(f"   - 前 5 个元素: {data[:5]}")
            else:
                print("   - 数据集太大，未打印全部内容。")

        else:
            print(f"\n❌ Dataset '{dataset_path}' 未找到。")

except FileNotFoundError:
    print(f"\n🚨 错误：文件未找到。请检查文件名 '{file_name}' 是否正确。")
except Exception as e:
    print(f"\n❌ 发生错误: {e}")# python
