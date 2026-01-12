import cv2
import os


def convert_rgb_to_gray(input_folder, output_folder):
    # 确保输出文件夹存在，如果不存在则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历输入文件夹中的文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # 读取RGB图像
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # 将RGB图像转换为灰度图像
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 检查灰度图像的像素值范围
                gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
                # 生成输出文件名
                output_filename = os.path.join(output_folder, filename)
                # 保存灰度图像
                cv2.imwrite(output_filename, gray_img)


# 调用函数，将输入文件夹中的RGB PNG图像转换为灰度图像并保存到输出文件夹
input_folder = "./pseudo_test2"
output_folder = "./labels_gray_test2"
convert_rgb_to_gray(input_folder, output_folder)
