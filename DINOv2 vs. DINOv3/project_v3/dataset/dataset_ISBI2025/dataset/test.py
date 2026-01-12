import h5py
import numpy as np

# 随便找一个转换好的 h5 文件
path = '/home/ubuntu/db/FUGC_revision/dataset/dataset_ISBI2025/dataset/oursdataPath/labeled/0001.h5' # 替换为你的实际路径

with h5py.File(path, 'r') as f:
    label = f['label'][:]
    unique_vals = np.unique(label)
    print(f"标签中包含的值: {unique_vals}")
    print(f"最大值: {unique_vals.max()}")
    
    if unique_vals.max() == 2:
        print("结论: nclass 必须设为 3")
    elif unique_vals.max() == 1:
        print("结论: nclass 可以设为 2")