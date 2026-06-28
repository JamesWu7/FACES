import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from collections import Counter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as F

# ──────────────────────────────────────────────────────────────
# 病因大类 → 小类 映射（与 Excel 保持一致）
# ──────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "AIS (Adolescent Idiopathic Scoliosis)": ["AIS"],
    "CMS (Chiari malformation-associated scoliosis)": ["CM-I (Chiari malformation type I)"],
    "SS (Syndromic Scoliosis)": [
        "AMC (Arthrogryposis Multiplex Congenita)",
        "EDS (Ehlers\u2013Danlos Syndrome)",
        "FSS (Freeman-Sheldon syndrome)",
        "GSD (Gorham-Stout disease)",
        "MFS (Marfan syndrome)",
        "NF-1 (Neurofibromatosis type 1)",
        "Osteochondrodysplasia",
        "Osteogenesis imperfecta",
        "Other Syndrome",
        "PWS (Prader-Willi syndrome)",
        "SGS (Shprintzen-Goldberg syndrome)",
    ],
}

# 大类标签顺序（固定，保证可复现）
COARSE_CLASSES = sorted(CATEGORY_MAP.keys())   # 3类
# SS 下的细类顺序（固定）
FINE_CLASSES_SS = sorted(CATEGORY_MAP["SS (Syndromic Scoliosis)"])  # SS细类，自动随 CATEGORY_MAP 更新


def _small_to_coarse(small_label: str) -> str:
    """给定小类名，返回其所属大类名"""
    for coarse, smalls in CATEGORY_MAP.items():
        if small_label in smalls:
            return coarse
    raise ValueError(f"未知小类: {small_label}")


# ──────────────────────────────────────────────────────────────
# 数据增强
# ──────────────────────────────────────────────────────────────
def get_transform(train: bool = True):
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])


def get_strong_transform():
    """针对少样本类（<30张）的强增强"""
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.3)),
    ])


# ──────────────────────────────────────────────────────────────
# 数据集基类：读取所有样本，保存大类/小类原始标签
# ──────────────────────────────────────────────────────────────
class HierarchicalDataset(Dataset):
    """
    mode='coarse' : 标签为大类索引（0/1/2，对应 COARSE_CLASSES）
    mode='fine_ss': 仅包含 SS 大类样本，标签为 SS 内细类索引
    """

    def __init__(
        self,
        image_dir: str,
        label_path: str,
        mode: str = "coarse",       # 'coarse' 或 'fine_ss'
        transform=None,
        verbose: bool = True,
    ):
        assert mode in ("coarse", "fine_ss"), "mode 必须为 'coarse' 或 'fine_ss'"
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.verbose = verbose

        # 读取 Excel
        df = pd.read_excel(label_path).dropna(subset=["ID", "病因大类", "病因小类"])
        df["ID"] = df["ID"].astype(str).str.strip()
        df["病因大类"] = df["病因大类"].astype(str).str.strip()
        df["病因小类"] = df["病因小类"].astype(str).str.strip()
        id_to_small  = dict(zip(df["ID"], df["病因小类"]))
        id_to_coarse = dict(zip(df["ID"], df["病因大类"]))

        # 扫描图片
        all_jpgs = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(image_dir, f))
        ]
        if not all_jpgs:
            raise FileNotFoundError(f"图片目录 {image_dir} 中未找到 JPG 图片")

        # 过滤：只保留 Excel 中有标签的图片
        self.image_paths: list = []
        self.coarse_raw:  list = []   # 大类原始字符串
        self.small_raw:   list = []   # 小类原始字符串

        for img_name in all_jpgs:
            img_id = os.path.splitext(img_name)[0]
            if img_id in id_to_small:
                coarse_name = id_to_coarse[img_id]
                small_name  = id_to_small[img_id]

                # fine_ss 模式只保留 SS 大类
                if mode == "fine_ss" and coarse_name != "SS (Syndromic Scoliosis)":
                    continue

                self.image_paths.append(os.path.join(image_dir, img_name))
                self.coarse_raw.append(coarse_name)
                self.small_raw.append(small_name)

        if not self.image_paths:
            raise ValueError(f"mode='{mode}' 下无有效样本")

        # 构建标签索引
        if mode == "coarse":
            self.classes = COARSE_CLASSES
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.labels = [self.class_to_idx[c] for c in self.coarse_raw]
        else:  # fine_ss
            self.classes = FINE_CLASSES_SS
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            # 小类标签（EDS 编码问题：做模糊匹配）
            self.labels = []
            for s in self.small_raw:
                matched = None
                for cls in self.classes:
                    if cls.split("(")[0].strip() == s.split("(")[0].strip():
                        matched = cls
                        break
                if matched is None:
                    matched = s  # 直接用，可能报 KeyError -> 好调试
                self.labels.append(self.class_to_idx[matched])

        # 强增强（仅 fine_ss 训练时使用）
        self._strong_tfm = get_strong_transform() if (mode == "fine_ss" and transform is not None) else None

        if verbose:
            print(f"[HierarchicalDataset] mode={mode}")
            print(f"  总样本数: {len(self.image_paths)}  | 类别数: {len(self.classes)}")
            cnt = Counter(self.labels)
            for i, cls in enumerate(self.classes):
                n = cnt.get(i, 0)
                print(f"  [{i}] {cls}: {n} 张 ({n/len(self.labels)*100:.1f}%)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # fine_ss 模式下，少样本类（标签对应样本数<30）使用强增强
        if self.transform:
            if hasattr(self, '_strong_tfm') and self._strong_tfm is not None:
                from collections import Counter as _C
                cnt = _C(self.labels)
                if cnt.get(self.labels[idx], 99) < 30:
                    image = self._strong_tfm(image)
                else:
                    image = self.transform(image)
            else:
                image = self.transform(image)
        return image, self.labels[idx]


# ──────────────────────────────────────────────────────────────
# 工具函数：创建 DataLoader
# ──────────────────────────────────────────────────────────────
def get_hierarchical_loaders(
    image_dir: str,
    label_path: str,
    mode: str = "coarse",
    batch_size: int = 16,
    train_split: float = 0.85,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    save_split_info: bool = True,
    output_dir: str = "./split_info",
    verbose: bool = True,
):
    """
    返回 (train_loader, val_loader, num_classes, class_names)
    """
    train_ds_full = HierarchicalDataset(
        image_dir, label_path, mode=mode,
        transform=get_transform(train=True), verbose=verbose
    )
    val_ds_full = HierarchicalDataset(
        image_dir, label_path, mode=mode,
        transform=get_transform(train=False), verbose=False
    )

    train_idx, val_idx = train_test_split(
        range(len(train_ds_full)),
        test_size=1 - train_split,
        stratify=train_ds_full.labels,
        random_state=random_seed,
    )

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full,   val_idx)

    train_labels = [train_ds_full.labels[i] for i in train_idx]
    val_labels   = [val_ds_full.labels[i]   for i in val_idx]

    train_cnt = Counter(train_labels)
    val_cnt = Counter(val_labels)
    train_ratio = len(train_idx) / (len(train_idx) + len(val_idx))
    val_ratio = len(val_idx) / (len(train_idx) + len(val_idx))

    if verbose:
        print(f"\n  划分结果: 训练={len(train_idx)} | 验证={len(val_idx)}")
        print(f"  划分比例: 训练={train_ratio:.2%} | 验证={val_ratio:.2%}")
        print("\n  训练集各病因小类分布:")
        for i, cls in enumerate(train_ds_full.classes):
            n = train_cnt.get(i, 0)
            pct = (n / len(train_idx) * 100) if len(train_idx) else 0.0
            print(f"    [{i}] {cls}: {n} 张 ({pct:.1f}%)")

        print("\n  验证集各病因小类分布:")
        for i, cls in enumerate(train_ds_full.classes):
            n = val_cnt.get(i, 0)
            pct = (n / len(val_idx) * 100) if len(val_idx) else 0.0
            print(f"    [{i}] {cls}: {n} 张 ({pct:.1f}%)")

    if save_split_info:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        info = {
            "mode": mode,
            "classes": train_ds_full.classes,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "train_distribution": {
                train_ds_full.classes[i]: int(train_cnt.get(i, 0))
                for i in range(len(train_ds_full.classes))
            },
            "val_distribution": {
                train_ds_full.classes[i]: int(val_cnt.get(i, 0))
                for i in range(len(train_ds_full.classes))
            },
            "seed": random_seed,
            "timestamp": ts,
        }
        with open(os.path.join(output_dir, f"split_{mode}_{ts}_seed{random_seed}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, len(train_ds_full.classes), train_ds_full.classes


if __name__ == "__main__":
    image_dir  = r"C:\Users\86195\PycharmProjects\disease_7150\deformity_data"
    label_path = r"C:\Users\86195\PycharmProjects\disease_7150\label_list_final_0316.xlsx"

    print("=== Stage 1: 大类数据集 ===")
    tl, vl, nc, cls = get_hierarchical_loaders(
        image_dir, label_path, mode="coarse", batch_size=8, verbose=True
    )
    print(f"  类别数={nc}, 训练批次={len(tl)}, 验证批次={len(vl)}")

    print("\n=== Stage 2: SS 细类数据集 ===")
    tl2, vl2, nc2, cls2 = get_hierarchical_loaders(
        image_dir, label_path, mode="fine_ss", batch_size=8, verbose=True
    )
    print(f"  类别数={nc2}, 训练批次={len(tl2)}, 验证批次={len(vl2)}")
