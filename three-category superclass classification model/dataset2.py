import os
import pandas as pd
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision.transforms.functional as F
import random
import numpy as np
import json
from datetime import datetime
import copy


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.class_to_idx = {
            'SS (Syndromic Scoliosis)': 0,
            'CMS (Chiari malformation-associated scoliosis)': 1,
            'AIS (Adolescent Idiopathic Scoliosis)': 2,
        }
        self.classes = ['SS', 'CMS', 'AIS']

        self.all_jpg_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(image_dir, f))
        ]
        if not self.all_jpg_files:
            raise FileNotFoundError(f"图片目录 {image_dir} 中未找到JPG图片")

        self.label_df = pd.read_excel(label_path).dropna(subset=['ID', '病因大类'])
        self.id_to_label = dict(zip(self.label_df['ID'].astype(str), self.label_df['病因大类']))

        self.image_paths = []
        self.raw_labels = []
        skipped = 0
        for img_name in self.all_jpg_files:
            img_id = os.path.splitext(img_name)[0]
            if img_id in self.id_to_label:
                raw_label = self.id_to_label[img_id]
                if raw_label in self.class_to_idx:
                    self.image_paths.append(os.path.join(image_dir, img_name))
                    self.raw_labels.append(raw_label)
                else:
                    skipped += 1

        if skipped > 0:
            print(f"  - 已跳过 {skipped} 个不属于三分类的样本")

        self.labels = [self.class_to_idx[lbl] for lbl in self.raw_labels]

        print(f"数据加载完成：总JPG={len(self.all_jpg_files)} | 有效样本={len(self.image_paths)}")
        for cls_name, cls_idx in self.class_to_idx.items():
            count = self.labels.count(cls_idx)
            print(f"  · {cls_name}（{cls_idx}）：{count} 个（{count/len(self.labels)*100:.1f}%）")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


def get_transform(train=True):
    """数据变换：训练集使用增强，验证集只做标准化"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.RandomAffine(
                degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15),
                shear=5
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


def get_tta_transform():
    """Test Time Augmentation 变换"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])


class SubsetWithTransform(Dataset):
    """支持独立 transform 的 Subset 封装"""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = Image.open(self.dataset.image_paths[real_idx]).convert('RGB')
        label = self.dataset.labels[real_idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(image_dir, label_path, batch_size=32, train_split=0.85,
                     random_seed=42, save_split_info=True, output_dir="./split_info"):
    """
    获取分层划分的训练/验证 DataLoader
    使用 WeightedRandomSampler 隐式过采样，三类均衡，每 epoch ~10500 样本
    返回：train_loader, val_loader, class_weights
    """
    full_dataset = CustomImageDataset(
        image_dir=image_dir, label_path=label_path, transform=None
    )

    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=1 - train_split,
        stratify=full_dataset.labels,
        random_state=random_seed
    )

    train_dataset = SubsetWithTransform(full_dataset, train_indices, get_transform(train=True))
    val_dataset   = SubsetWithTransform(full_dataset, val_indices,   get_transform(train=False))

    train_labels = [full_dataset.labels[i] for i in train_indices]
    val_labels   = [full_dataset.labels[i] for i in val_indices]

    train_class_dist, val_class_dist = {}, {}
    for cls_idx, cls_name in enumerate(full_dataset.classes):
        tc = train_labels.count(cls_idx)
        vc = val_labels.count(cls_idx)
        train_class_dist[cls_name] = {"count": tc, "percentage": round(tc/len(train_labels)*100, 1)}
        val_class_dist[cls_name]   = {"count": vc, "percentage": round(vc/len(val_labels)*100, 1)}

    print(f"\n分层划分（训练{train_split*100:.0f}% | 验证{(1-train_split)*100:.0f}%）：")
    for cls_name, info in train_class_dist.items():
        print(f"  训练 {cls_name}：{info['count']}个（{info['percentage']}%）")
    for cls_name, info in val_class_dist.items():
        print(f"  验证 {cls_name}：{info['count']}个（{info['percentage']}%）")

    if save_split_info:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"data_split_{timestamp}_seed{random_seed}.json")
        split_info = {
            "split_config": {"train_split": train_split, "random_seed": random_seed},
            "split_results": {
                "train_set": {"num_samples": len(train_indices), "class_distribution": train_class_dist},
                "val_set":   {"num_samples": len(val_indices),   "class_distribution": val_class_dist}
            },
            "timestamp": timestamp
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"划分信息已保存到：{json_path}")

    # 动态计算类别权重（用于 CrossEntropyLoss）
    num_classes = len(full_dataset.classes)
    class_counts = torch.zeros(num_classes)
    for lbl in train_labels:
        class_counts[lbl] += 1
    class_weights = class_counts.sum() / (num_classes * class_counts)
    print(f"动态类别权重: {[round(w, 4) for w in class_weights.tolist()]}")

    # ---- WeightedRandomSampler：隐式过采样，三类均衡 ----
    # ResNet18 可训练参数约 1100万
    # 三类均衡：每类采样 3500 次，总计 10500 次/epoch（约原训练集的 8.6 倍）
    # 少数类（AIS=216）被重复采样约 16 次，多数类（SS=691）被采样约 5 次
    samples_per_class = 3500
    per_class_weight = torch.tensor(
        [samples_per_class / class_counts[i].item() for i in range(num_classes)],
        dtype=torch.float
    )
    sample_weights = torch.tensor(
        [per_class_weight[full_dataset.labels[i]].item() for i in train_indices],
        dtype=torch.float
    )
    total_samples = samples_per_class * num_classes  # 10500
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )
    print(f"\nWeightedRandomSampler 配置：")
    print(f"  每类目标采样数: {samples_per_class}")
    print(f"  每epoch总采样数: {total_samples}（原 {len(train_indices)} 的 {total_samples/len(train_indices):.1f}x）")
    print(f"  各类采样权重: SS={per_class_weight[0]:.2f}, CMS={per_class_weight[1]:.2f}, AIS={per_class_weight[2]:.2f}")

    # sampler 与 shuffle 互斥，不能同时使用
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    image_dir  = r"C:\Users\86195\PycharmProjects\disease_33\deformity_data"
    label_path = r"C:\Users\86195\PycharmProjects\disease_33\label_list_final_0316.xlsx"
    train_loader, val_loader, class_weights = get_data_loaders(
        image_dir=image_dir, label_path=label_path,
        batch_size=8, train_split=0.85, random_seed=42
    )
    print(f"训练批次：{len(train_loader)} | 验证批次：{len(val_loader)}")
    for img, labels in train_loader:
        print(f"图片形状：{img.shape} | 标签：{labels.tolist()}")
        break
