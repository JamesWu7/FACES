from mtcnn import MTCNN
import cv2
import os
import numpy as np
from pathlib import Path


def batch_detect_faces_mtcnn(input_dir, output_dir):
    """
    使用MTCNN批量检测并截取人脸
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # 初始化MTCNN检测器
    detector = MTCNN()

    # 统计信息
    total_images = 0
    total_faces = 0

    # 处理每张图像
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            total_images += 1
            # print(f"处理: {image_file.name}")

            try:
                # 读取图像
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"  无法读取图像: {image_file.name}")
                    continue

                # 转换颜色空间
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 检测人脸
                results = detector.detect_faces(image_rgb)

                faces_count = 0
                result =results[0]
                # 获取人脸边界框
                x, y, width, height = result['box']
                height= int(height*1.15)

                # 确保边界框在图像范围内并添加扩展
                padding = 25
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(image.shape[1] - x, width + 2 * padding)
                height = min(image.shape[0] - y, height + 2 * padding)

                # 截取人脸区域
                face = image[y:y + height+25, x:x + width]

                # 保存人脸图像
                if face.size > 0:
                    faces_count += 1
                    total_faces += 1
                    face_filename = output_path / f"{image_file.stem}.jpg"
                    cv2.imwrite(str(face_filename), face)
                else:
                    print('没有找到是：',image_file.name)


                # print(f"  检测到 {faces_count} 张人脸")

            except Exception as e:
                print(f"  处理 {image_file.name} 时出错: {e}")

    print(f"\n处理完成！")
    print(f"总共处理图像: {total_images} 张")
    print(f"总共检测到人脸: {total_faces} 张")
    print(f"输出目录: {output_dir}")


# 使用示例
if __name__ == "__main__":
    input_directory = r"C:\Users\86195\PycharmProjects\disease_2(final_9898)\2\data"
    output_directory = r"C:\Users\86195\PycharmProjects\disease_2(final_9898)\2\crop_oy"

    batch_detect_faces_mtcnn(input_directory, output_directory)