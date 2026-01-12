import os
import glob
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始数据的根目录
SRC_ROOT = "val" 
SRC_LABELED_IMG = os.path.join(SRC_ROOT, "labeled_data/images")
SRC_LABELED_MASK = os.path.join(SRC_ROOT, "labeled_data/labels")
SRC_UNLABELED_IMG = os.path.join(SRC_ROOT, "unlabeled_data/images")

# 2. 输出目录
DST_ROOT = "oursdataPath1"

# 3. 输出的子文件夹名称
DIR_NAME_LABELED = "labeled"
DIR_NAME_UNLABELED = "unlabeled_data"

# 4. 确保输出目录存在
os.makedirs(os.path.join(DST_ROOT, DIR_NAME_LABELED), exist_ok=True)
os.makedirs(os.path.join(DST_ROOT, DIR_NAME_UNLABELED), exist_ok=True)
# ===========================================

def save_to_h5(img_path, mask_path, save_full_path):
    """
    将图像和掩膜保存为 H5 文件 (针对多分类任务，保留原始像素值)
    """
    try:
        # 1. 读取图片 (转为 RGB)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # 2. 读取标签 (如果有)
        label_np = None
        if mask_path:
            # [关键] 标签转为灰度 'L' 模式，保留 0, 1, 2...
            # 不要用 '1' 模式，否则非黑即白
            mask = Image.open(mask_path).convert('L') 
            label_np = np.array(mask)
            
            # [关键修改]
            # 既然你的标签已经是 0, 1, 2，这里千万不要除以 255！
            # 只需要确保它是整数类型即可
            label_np = label_np.astype('uint8')

            # [安全检查] 打印一下看看有没有奇怪的值 (调试用，只打印警告)
            unique_values = np.unique(label_np)
            if unique_values.max() > 2:
                # 如果发现大于2的值，可能是数据有问题，或者是 0-255 的数据
                print(f"\n⚠️ 警告: 文件 {os.path.basename(mask_path)} 包含标签值: {unique_values}")
                print("   如果这是 0-255 的掩膜，请取消注释代码里的归一化逻辑。")
                print("   如果这是多分类任务(>3类)，请忽略此警告。")

        # 3. 写入 H5
        with h5py.File(save_full_path, 'w') as f:
            f.create_dataset('image', data=img_np, dtype='uint8')
            if label_np is not None:
                f.create_dataset('label', data=label_np, dtype='uint8')
                
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def process_data():
    # --- 处理有标签数据 (Labeled) ---
    print(f"1. 正在转换有标签数据 (Labels: 0, 1, 2) -> {DIR_NAME_LABELED}/ ...")
    labeled_imgs = sorted(glob.glob(os.path.join(SRC_LABELED_IMG, "*.png")))
    train_lines = []

    for img_path in tqdm(labeled_imgs):
        filename = os.path.basename(img_path)
        file_id = os.path.splitext(filename)[0]
        
        mask_path = os.path.join(SRC_LABELED_MASK, filename)
        if not os.path.exists(mask_path):
            print(f"  [跳过] 找不到标签: {filename}")
            continue

        # 保存 H5
        save_name = f"{file_id}.h5"
        save_path = os.path.join(DST_ROOT, DIR_NAME_LABELED, save_name)
        save_to_h5(img_path, mask_path, save_path)
        
        # 添加到列表 (注意：通常 dataset 会自动加 .h5，所以这里只存 ID)
        # 格式: labeled/case_001
        train_lines.append(os.path.join(DIR_NAME_LABELED, f"{file_id}.h5"))

    # --- 处理无标签数据 (Unlabeled) ---
    print(f"2. 正在转换无标签数据 -> {DIR_NAME_UNLABELED}/ ...")
    unlabeled_imgs = sorted(glob.glob(os.path.join(SRC_UNLABELED_IMG, "*.png")))
    unlabeled_lines = []

    for img_path in tqdm(unlabeled_imgs):
        filename = os.path.basename(img_path)
        file_id = os.path.splitext(filename)[0]
        
        save_name = f"{file_id}.h5"
        save_path = os.path.join(DST_ROOT, DIR_NAME_UNLABELED, save_name)
        save_to_h5(img_path, None, save_path)
        
        # 格式: unlabeled_data/case_001
        unlabeled_lines.append(os.path.join(DIR_NAME_UNLABELED, f"{file_id}.h5"))

    # --- 保存 TXT 列表 ---
    print("3. 生成列表文件...")
    
    with open(os.path.join(DST_ROOT, "train.txt"), 'w') as f:
        for line in train_lines:
            f.write(f"{line.replace(os.sep, '/')}\n")
            
    with open(os.path.join(DST_ROOT, "unlabeled.txt"), 'w') as f:
        for line in unlabeled_lines:
            f.write(f"{line.replace(os.sep, '/')}\n")

    print(f"\n✅ 处理完成!")
    print(f" - 列表文件已生成在: {DST_ROOT}")

if __name__ == "__main__":
    process_data()