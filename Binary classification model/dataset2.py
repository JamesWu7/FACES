import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

warnings.filterwarnings("ignore")

try:
    import dlib
except ImportError:
    dlib = None


class AlignmentTransform:
    def __init__(self, max_rotation=10, max_scale=0.1, max_translation=0.1):
        self.max_rotation = max_rotation
        self.max_scale = max_scale
        self.max_translation = max_translation

    def __call__(self, image):
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        image = F.rotate(image, angle)
        scale = 1 + random.uniform(-self.max_scale, self.max_scale)
        new_size = [int(dim * scale) for dim in image.size]
        return F.resize(image, new_size)


class IlluminationVariation:
    def __init__(self, brightness_range=0.3, contrast_range=0.3):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image):
        image = F.adjust_brightness(image, random.uniform(1 - self.brightness_range, 1 + self.brightness_range))
        return F.adjust_contrast(image, random.uniform(1 - self.contrast_range, 1 + self.contrast_range))


class QualityDegradation:
    def __init__(self, blur_prob=0.2, noise_prob=0.2, compression_prob=0.1):
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.compression_prob = compression_prob

    def __call__(self, image):
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
        if random.random() < self.noise_prob:
            img_array = np.array(image)
            noise = np.random.normal(0, random.randint(5, 20), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        return image


class FaceNeckLandmarkCrop:
    """基于 dlib 68 点保留五官和下巴，弱化脖颈区域。"""

    def __init__(self, predictor_path=None):
        self.predictor_path = self._resolve_predictor_path(predictor_path)
        self.available = dlib is not None and self.predictor_path is not None
        self._warned = False
        if self.available:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(self.predictor_path))
        else:
            self.detector = None
            self.predictor = None

    def _resolve_predictor_path(self, predictor_path):
        candidates = []
        if predictor_path:
            candidates.append(Path(predictor_path))
        env_path = os.environ.get("DLIB_LANDMARK_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat")
        for path in candidates:
            if path.exists():
                return path
        return None

    def _warn_once(self, message):
        if not self._warned:
            print(message)
            self._warned = True

    def __call__(self, image):
        if not self.available:
            self._warn_once("警告：dlib 或 68 点模型未就绪，已跳过ROI裁剪。")
            return image
        image_np = np.array(image)
        detections = self.detector(image_np, 1)
        if not detections:
            return image
        face = max(detections, key=lambda rect: rect.width() * rect.height())
        shape = self.predictor(image_np, face)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        jaw = pts[0:17]
        brow = pts[17:27]
        eyes = pts[36:48]
        nose = pts[27:36]
        mouth = pts[48:68]
        face_core = np.vstack([brow, eyes, nose, mouth, jaw[4:13]])

        x_min, x_max = face_core[:, 0].min(), face_core[:, 0].max()
        upper_anchor = min(brow[:, 1].min(), eyes[:, 1].min())
        chin_y = jaw[:, 1].max()
        mouth_bottom = mouth[:, 1].max()

        face_width = max(1, x_max - x_min)
        face_height = max(1, chin_y - upper_anchor)
        chin_extension = max(int(face_height * 0.10), int((chin_y - mouth_bottom) * 0.65))
        side_margin = int(face_width * 0.10)
        top_margin = int(face_height * 0.18)

        crop_x1 = max(0, int(x_min - side_margin))
        crop_x2 = min(image.width, int(x_max + side_margin))
        crop_y1 = max(0, int(upper_anchor - top_margin))
        crop_y2 = min(image.height, int(chin_y + chin_extension))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return image
        return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None, roi_cropper=None):
        self.image_dir = image_dir
        self.transform = transform
        self.roi_cropper = roi_cropper
        self.all_jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(image_dir, f))]
        if not self.all_jpg_files:
            raise FileNotFoundError(f"图片目录 {image_dir} 中未找到JPG图片")

        label_df = pd.read_excel(label_path).dropna(subset=['ID', '病因大类'])
        if 'ID' not in label_df.columns or '病因大类' not in label_df.columns:
            raise ValueError("Excel必须包含'ID'和'病因大类'列")
        id_to_label = dict(zip(label_df['ID'].astype(str), label_df['病因大类']))

        self.image_paths, self.raw_labels = [], []
        for img_name in self.all_jpg_files:
            img_id = os.path.splitext(img_name)[0]
            if img_id in id_to_label:
                self.image_paths.append(os.path.join(image_dir, img_name))
                self.raw_labels.append(id_to_label[img_id])

        binary_label_map = {
            'HC': 0,
            'CMS (Chiari malformation-associated scoliosis)': 1,
            'AIS (Adolescent Idiopathic Scoliosis)': 1,
            'SS (Syndromic Scoliosis)': 1,
        }
        self.labels = []
        for label in self.raw_labels:
            if label not in binary_label_map:
                raise ValueError(f"未知病因类别: {label}")
            self.labels.append(binary_label_map[label])

        self.classes = ['无病(HC)', '有病']
        self.class_to_idx = {'无病(HC)': 0, '有病': 1}
        print("数据加载完成：")
        print(f"  - 总JPG数量：{len(self.all_jpg_files)} | 有效样本数：{len(self.image_paths)}")
        print(f"  - 二分类任务 | 类别映射：{self.class_to_idx}")
        if self.roi_cropper is not None:
            print("  - 已启用dlib 68关键点ROI裁剪（聚焦五官+下巴）")
        num_healthy = self.labels.count(0)
        num_disease = self.labels.count(1)
        print(f"    · 无病(HC)：{num_healthy} 个（占比 {num_healthy / len(self.labels) * 100:.1f}%）")
        print(f"    · 有病：{num_disease} 个（占比 {num_disease / len(self.labels) * 100:.1f}%）")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample_path = self.image_paths[idx]
        try:
            image = Image.open(sample_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"读取图片 {sample_path} 失败：{str(e)}")
        if self.roi_cropper is not None:
            image = self.roi_cropper(image)
        if self.transform:
            image = self.transform(image)
        sample_id = os.path.splitext(os.path.basename(sample_path))[0]
        return image, torch.tensor(self.labels[idx], dtype=torch.float32), sample_id


def get_transform(train=True):
    if train:
        transform_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            AlignmentTransform(max_rotation=8, max_scale=0.1, max_translation=0.1),
            IlluminationVariation(brightness_range=0.2, contrast_range=0.2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            QualityDegradation(blur_prob=0.15, noise_prob=0.15),
            transforms.RandomResizedCrop(size=(210, 140), scale=(0.85, 1.1), ratio=(0.82, 1.1)),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [transforms.Resize((210, 140)), transforms.ToTensor()]
    return transforms.Compose(transform_list)


def get_data_loaders(image_dir, label_path, batch_size=32, train_split=0.7, random_seed=42, save_split_info=True,
                     output_dir="./split_info", predictor_path=None):
    roi_cropper = FaceNeckLandmarkCrop(predictor_path=predictor_path)
    full_dataset = CustomImageDataset(image_dir=image_dir, label_path=label_path, transform=get_transform(train=True), roi_cropper=roi_cropper)

    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=1 - train_split,
        stratify=full_dataset.labels,
        random_state=random_seed,
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = get_transform(train=False)

    train_labels = [full_dataset.labels[i] for i in train_indices]
    val_labels = [full_dataset.labels[i] for i in val_indices]
    train_class_dist, val_class_dist = {}, {}
    for cls_idx, cls_name in enumerate(full_dataset.classes):
        train_count = train_labels.count(cls_idx)
        val_count = val_labels.count(cls_idx)
        train_class_dist[cls_name] = {"count": train_count, "percentage": round(train_count / len(train_labels) * 100, 1)}
        val_class_dist[cls_name] = {"count": val_count, "percentage": round(val_count / len(val_labels) * 100, 1)}

    print(f"\n分层划分结果（训练集{train_split * 100:.0f}% | 验证集{(1 - train_split) * 100:.0f}%）：")
    print("  训练集类别分布：")
    for cls_name, info in train_class_dist.items():
        print(f"    · {cls_name}：{info['count']}个（占比{info['percentage']}%）")
    print("  验证集类别分布：")
    for cls_name, info in val_class_dist.items():
        print(f"    · {cls_name}：{info['count']}个（占比{info['percentage']}%）")

    if save_split_info:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"data_split_{timestamp}_seed{random_seed}.json")
        split_info = {
            "dataset_info": {"name": os.path.basename(image_dir), "total_samples": len(full_dataset), "num_classes": len(full_dataset.classes), "classes": full_dataset.classes},
            "split_config": {"train_split": train_split, "val_split": 1 - train_split, "batch_size": batch_size, "random_seed": random_seed, "split_method": "stratified_split"},
            "split_results": {
                "train_set": {"num_samples": len(train_indices), "class_distribution": train_class_dist, "indices": train_indices},
                "val_set": {"num_samples": len(val_indices), "class_distribution": val_class_dist, "indices": val_indices},
            },
            "file_paths": {"image_dir": image_dir, "label_path": label_path, "split_info_file": json_path},
            "timestamp": timestamp,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"\n划分信息已保存到：{json_path}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    image_dir = r"C:\Users\86195\PycharmProjects\disease_2(final_9898)\2\deformity_data"
    label_path = r"C:\Users\86195\PycharmProjects\disease_2(final_9898)\2\label_list_final_0316.xlsx"
    train_loader, val_loader = get_data_loaders(image_dir=image_dir, label_path=label_path, batch_size=6, train_split=0.7, random_seed=42)
    print(f"\n训练集批次：{len(train_loader)} | 验证集批次：{len(val_loader)}")
    for _, labels, _ in train_loader:
        print(labels)
        print("unique labels:", labels.unique())
        break
    for img, labels, sample_ids in train_loader:
        print(f"图片形状：{img.shape} | 标签形状：{labels.shape}")
        print(f"  - 单批次图片形状：{img.shape}（batch_size×C×H×W）")
        print(f"  - 单批次标签示例：{labels.tolist()}")
        print(f"  - 单批次样本ID示例：{list(sample_ids)}")
        break
