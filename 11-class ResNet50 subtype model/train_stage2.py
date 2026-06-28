"""
Stage 2: SS细类分类器训练 (仅使用SS大类样本，类别数自动读取)
用法: python train_stage2.py
"""
import argparse, json, os, random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from dataset_hierarchical import FINE_CLASSES_SS, get_hierarchical_loaders
from model2 import ImageClassifier
from VisualizeAC import visualize_activations



def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def compute_auc(labels, probs, nc):
    try:
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

def compute_pr_auc(labels, probs, nc):
    try:
        y_true = label_binarize(labels, classes=list(range(nc)))
        if y_true.shape[1] != nc or probs.shape[1] != nc:
            return float("nan")
        return float(average_precision_score(y_true, probs, average="macro"))
    except Exception:
        return float("nan")

def compute_specificity(labels, preds, nc):
    try:
        cm = confusion_matrix(labels, preds, labels=list(range(nc)))
        total = cm.sum()
        specificities = []
        for i in range(nc):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp
            denom = tn + fp
            if denom > 0:
                specificities.append(tn / denom)
        if not specificities:
            return float("nan")
        return float(np.mean(specificities))
    except Exception:
        return float("nan")

def compute_metrics(labels, probs, nc) -> Dict[str, float]:
    if len(labels) == 0 or probs.size == 0:
        return {
            "accuracy": float("nan"),
            "f1_score": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "specificity": float("nan"),
        }

    preds = np.argmax(probs, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_score": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "roc_auc": compute_auc(labels, probs, nc),
        "pr_auc": compute_pr_auc(labels, probs, nc),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "specificity": compute_specificity(labels, preds, nc),
    }
    return metrics

def compute_class_weights(train_loader, nc, device):
    subset = train_loader.dataset
    labs = [subset.dataset.labels[i] for i in subset.indices]
    cnt = Counter(labs)
    counts = torch.zeros(nc, dtype=torch.float32)
    for l, c in cnt.items():
        if 0 <= int(l) < nc: counts[int(l)] = float(c)
    counts[counts == 0] = 1.0
    w = 1.0 / torch.sqrt(counts)
    return (w / w.sum()).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
            loss = criterion(outputs.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            if random.random() < 0.5:
                lam = float(torch.distributions.Beta(0.4, 0.4).sample())
                idx = torch.randperm(images.size(0), device=device)
                outputs = model(lam * images + (1-lam) * images[idx])
                loss = lam * criterion(outputs, labels) + (1-lam) * criterion(outputs, labels[idx])
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    if total == 0: return float("nan"), 0.0
    return total_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device, nc):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_probs.append(torch.softmax(outputs, 1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if total == 0: return float("nan"), 0.0, np.array([]), np.array([])
    return (total_loss/total, 100.0*correct/total,
            np.concatenate(all_probs, 0), np.concatenate(all_labels, 0))


def plot_single_metric_curve(values, metric_name_cn, metric_name_en, path):
    ep = range(1, len(values) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(ep, values, color="#1565C0", linewidth=2.2)
    plt.title(f"{metric_name_cn} 曲线")
    plt.xlabel("训练轮数")
    plt.ylabel(metric_name_cn)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{metric_name_en} 曲线已保存: {path}")


def plot_metric_curves(metric_histories, output_dir):
    metric_name_map = {
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "recall": "召回率",
        "precision": "精确率",
        "specificity": "特异度",
    }
    filename_map = {
        "f1_score": "f1_score_curve.png",
        "roc_auc": "roc_auc_curve.png",
        "pr_auc": "pr_auc_curve.png",
        "recall": "recall_curve.png",
        "precision": "precision_curve.png",
        "specificity": "specificity_curve.png",
    }
    for key, values in metric_histories.items():
        plot_single_metric_curve(
            values=values,
            metric_name_cn=metric_name_map[key],
            metric_name_en=key,
            path=os.path.join(output_dir, filename_map[key])
        )


# 11个病因小类的固定颜色，与第二张图（柱状图）中的颜色对应
CLASS_COLORS = [
    "#4E79A7",  # NF-1
    "#F28E2B",  # MFS
    "#E15759",  # AMC
    "#76B7B2",  # Osteochondrodysplasia
    "#59A14F",  # EDS
    "#EDC948",  # Other Syndrome
    "#B07AA1",  # Osteogenesis imperfecta
    "#FF9DA7",  # FSS
    "#9C755F",  # PWS
    "#BAB0AC",  # SGS
    "#D37295",  # GSD
]
# FINE_CLASSES_SS 排序后的顺序:
# [0] AMC, [1] EDS, [2] FSS, [3] GSD, [4] MFS,
# [5] NF-1, [6] Osteochondrodysplasia, [7] Osteogenesis imperfecta,
# [8] Other Syndrome, [9] PWS, [10] SGS
CLASS_COLOR_MAP = {
    0: "#E15759",   # AMC
    1: "#59A14F",   # EDS
    2: "#FF9DA7",   # FSS
    3: "#D37295",   # GSD
    4: "#F28E2B",   # MFS
    5: "#4E79A7",   # NF-1
    6: "#76B7B2",   # Osteochondrodysplasia
    7: "#B07AA1",   # Osteogenesis imperfecta
    8: "#EDC948",   # Other Syndrome
    9: "#9C755F",   # PWS
    10: "#BAB0AC",  # SGS
}


def _short_name(full_name):
    """Extract short name like 'AMC', 'EDS' from full class name."""
    if "(" in full_name:
        part = full_name.split("(")[0].strip()
        if len(part) <= 5:
            return part
        part2 = full_name.split("(")[1].rstrip(")")
        if len(part2) <= 20:
            return part2
        return part
    return full_name if len(full_name) <= 20 else full_name[:17] + "..."


def plot_roc_curves(labels, probs, class_names, save_path):
    """Plot per-class ROC curves (Sensitivity vs 1-Specificity)."""
    nc = len(class_names)
    y_true = label_binarize(labels, classes=list(range(nc)))
    if y_true.shape[1] != nc:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(nc):
        fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
        auc_val = roc_auc_score(y_true[:, i], probs[:, i])
        short = _short_name(class_names[i])
        color = CLASS_COLOR_MAP.get(i, f"C{i}")
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{short} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 – Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存: {save_path}")


def plot_pr_curves(labels, probs, class_names, save_path):
    """Plot per-class Precision-Recall curves."""
    nc = len(class_names)
    y_true = label_binarize(labels, classes=list(range(nc)))
    if y_true.shape[1] != nc:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(nc):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        ap_val = average_precision_score(y_true[:, i], probs[:, i])
        short = _short_name(class_names[i])
        color = CLASS_COLOR_MAP.get(i, f"C{i}")
        ax.plot(recall_vals, precision_vals, color=color, linewidth=1.8,
                label=f"{short} (AUPRC = {ap_val:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR 曲线已保存: {save_path}")


def generate_gradcam_all(model, loaders, device, class_names, save_dir):
    """Generate Grad-CAM for ALL images in the dataset. Each image saved individually."""
    ensure_dir(save_dir)
    model.eval()

    _MEAN = np.array([0.485, 0.456, 0.406])
    _STD = np.array([0.229, 0.224, 0.225])

    activations, gradients = {}, {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    target_layer = model.layer4
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    count = 0
    try:
        for loader in loaders:
            for images, labels in loader:
                images = images.to(device)
                labels_np = labels.numpy()

                for idx in range(images.size(0)):
                    single_img = images[idx:idx+1]
                    single_img.requires_grad_(True)
                    model.zero_grad()
                    output = model(single_img)
                    pred_class = output.argmax(dim=1).item()
                    output[0, pred_class].backward()

                    act = activations['value'][0]
                    grad = gradients['value'][0]
                    weights = grad.mean(dim=(1, 2), keepdim=True)
                    cam = (weights * act).sum(dim=0)
                    cam = torch.relu(cam)
                    cam = cam - cam.min()
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    cam_np = cam.cpu().numpy()
                    cam_resized = cv2.resize(cam_np, (224, 224))

                    img_np = single_img[0].detach().cpu().numpy().transpose(1, 2, 0)
                    img_np = img_np * _STD + _MEAN
                    img_np = np.clip(img_np, 0, 1)

                    true_label = int(labels_np[idx])
                    true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
                    pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)

                    short_true = true_name.split("(")[0].strip() if "(" in true_name else true_name
                    class_dir = os.path.join(save_dir, short_true)
                    ensure_dir(class_dir)

                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    overlay = 0.5 * img_np + 0.5 * heatmap
                    overlay = np.clip(overlay, 0, 1)

                    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                    axes[0].imshow(img_np)
                    axes[0].set_title("Original", fontsize=9)
                    axes[0].axis('off')
                    axes[1].imshow(overlay)
                    axes[1].set_title(f"Pred: {pred_name.split('(')[0].strip()}", fontsize=9)
                    axes[1].axis('off')
                    plt.subplots_adjust(wspace=0.05)
                    fname = f"sample_{count:04d}_true_{short_true}_pred_{pred_name.split('(')[0].strip()}.png"
                    plt.savefig(os.path.join(class_dir, fname), dpi=150, bbox_inches='tight', pad_inches=0.05)
                    plt.close()
                    count += 1
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    print(f"Grad-CAM: 已为 {count} 张图片生成热力图，保存在: {save_dir}")


def plot_performance_summary(metrics_dict, save_path):
    """Plot horizontal bar chart of key performance metrics (purple style)."""
    display_order = [
        ("Accuracy", "accuracy"),
        ("Recall (Sensitivity)", "recall"),
        ("Specificity", "specificity"),
        ("Precision (PPV)", "precision"),
        ("F1-score", "f1_score"),
    ]
    names, values = [], []
    for display_name, key in display_order:
        v = metrics_dict.get(key, float("nan"))
        if v != v:
            v = 0.0
        names.append(display_name)
        values.append(v * 100.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color="#B39DDB", edgecolor="#7E57C2", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Performance (%)", fontsize=10)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va='center', ha='left', fontsize=9, fontweight='bold')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能指标汇总图已保存: {save_path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练/验证损失与准确率曲线已保存: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    short_names = [n.split("(")[1].rstrip(")") if "(" in n else n for n in class_names]
    short_names = [s if len(s) <= 20 else s[:17] + "..." for s in short_names]

    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm.astype(float) / row_sums * 100.0

    nc = len(class_names)
    fig, ax = plt.subplots(figsize=(max(9, nc * 0.95), max(7, nc * 0.8)))

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Purples', vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row\nnormalized (%)", rotation=0, labelpad=40, va='center')

    ax.set_xticks(np.arange(nc))
    ax.set_yticks(np.arange(nc))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)

    for i in range(nc):
        for j in range(nc):
            value = cm_norm[i, j]
            text_color = "white" if value > 60 else "black"
            ax.text(j, i, f"{value:.2f}", ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(np.arange(nc + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nc + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    fig.text(0.5, -0.02, "Values are percentages (%). Diagonal: correct predictions.",
             ha='center', fontsize=9, style='italic')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


@dataclass(frozen=True)
class Stage2Config:
    image_dir:   str
    label_path:  str
    num_classes: int   = len(FINE_CLASSES_SS)       # SS细类数量，自动随数据集类别列表更新
    batch_size:  int   = 16
    lr:          float = 3e-4
    epochs:      int   = 150
    train_split: float = 0.85
    seed:        int   = 51
    num_workers: int   = 0
    use_amp:     bool  = False
    early_stop_patience:  int   = 35
    early_stop_min_delta: float = 0.01
    visualize:            bool  = True
    visualize_interval:   int   = 20
    visualize_max_layers: int   = 6
    results_dir: str   = "results_stage2"


def run_stage2(cfg: Stage2Config) -> str:
    device = get_device()
    seed_everything(cfg.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(cfg.results_dir, f"stage2_{ts}_seed{cfg.seed}"))
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    ensure_dir(os.path.join(run_dir, "split_info"))
    ensure_dir(os.path.join(run_dir, "activation_maps"))
    write_json(os.path.join(run_dir, "config.json"), asdict(cfg) | {"device": str(device)})

    print(f"\n=== Stage 2: SS细类分类器训练 ({len(FINE_CLASSES_SS)}类) ===")
    print("注意: 仅使用病因大类=SS的样本，已剔除AIS和CMS")
    train_loader, val_loader, nc, class_names = get_hierarchical_loaders(
        image_dir=cfg.image_dir, label_path=cfg.label_path, mode="fine_ss",
        batch_size=cfg.batch_size, train_split=cfg.train_split, random_seed=cfg.seed,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        save_split_info=True, output_dir=os.path.join(run_dir, "split_info"), verbose=True,
    )
    if cfg.num_classes != nc:
        print(f"配置类别数 {cfg.num_classes} 与数据集类别数 {nc} 不一致，自动使用数据集类别数 {nc}")
    num_classes = nc
    print(f"SS细类顺序: {class_names}")

    model = ImageClassifier(num_classes=num_classes).to(device)
    info = model.get_param_count()
    print(f"模型参数: 总={info['总参数量']}, 可训练={info['可训练参数量']}")

    cw = compute_class_weights(train_loader, num_classes, device)
    print(f"类别权重: {cw.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    bb_params = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
    hd_params = [p for n, p in model.named_parameters() if "fc" in n     and p.requires_grad]
    optimizer = optim.Adam([
        {"params": bb_params, "lr": cfg.lr * 0.1},
        {"params": hd_params, "lr": cfg.lr * 1.0},
    ], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    val_imgs = None
    if cfg.visualize:
        val_imgs, _ = next(iter(val_loader))
        val_imgs = val_imgs.to(device)

    tl_hist, vl_hist, ta_hist, va_hist = [], [], [], []
    metric_histories = {"f1_score": [], "roc_auc": [], "pr_auc": [], "recall": [], "precision": [], "specificity": []}
    best_acc, best_auc, best_ep = 0.0, 0.0, -1
    no_imp, es_best = 0, float("-inf")

    for epoch in range(cfg.epochs):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            cfg.use_amp and device.type == "cuda")
        v_loss, v_acc, v_probs, v_labs = validate(
            model, val_loader, criterion, device, num_classes)
        scheduler.step()

        val_metrics = compute_metrics(v_labs, v_probs, num_classes)
        tl_hist.append(t_loss); vl_hist.append(v_loss)
        ta_hist.append(t_acc); va_hist.append(v_acc)
        for key in metric_histories:
            metric_histories[key].append(val_metrics[key])

        def fmt(v):
            return f"{v:.4f}" if v == v else "N/A"

        print(
            f"[S2] Ep[{epoch+1}/{cfg.epochs}] "
            f"TrLoss={t_loss:.4f} TrAcc={t_acc:.2f}% "
            f"VaLoss={v_loss:.4f} VaAcc={v_acc:.2f}% "
            f"F1={fmt(val_metrics['f1_score'])} "
            f"ROC-AUC={fmt(val_metrics['roc_auc'])} "
            f"PR-AUC={fmt(val_metrics['pr_auc'])} "
            f"Recall={fmt(val_metrics['recall'])} "
            f"Precision={fmt(val_metrics['precision'])} "
            f"Specificity={fmt(val_metrics['specificity'])}"
        )

        if cfg.visualize and val_imgs is not None:
            if epoch == 0 or (epoch+1) % cfg.visualize_interval == 0:
                model.eval()
                acts = {}
                def _hook(nm):
                    def _fn(m, i, o): acts[nm] = o
                    return _fn
                hooks, hooked = [], 0
                for nm, mod in model.named_modules():
                    if isinstance(mod, nn.Conv2d) and hooked < cfg.visualize_max_layers:
                        hooks.append(mod.register_forward_hook(_hook(nm))); hooked += 1
                with torch.no_grad(): model(val_imgs)
                for h in hooks: h.remove()
                visualize_activations(acts, epoch,
                                      base_dir=os.path.join(run_dir, "activation_maps"))
                model.train()

        current_auc = val_metrics["roc_auc"]
        auc_s = f"{current_auc:.4f}" if current_auc == current_auc else "N/A"
        if v_acc > best_acc:
            best_acc, best_auc, best_ep = v_acc, current_auc, epoch+1
            torch.save(model.state_dict(), os.path.join(run_dir, "checkpoints", "best.pth"))
            print(f"  -> 保存最佳模型 (VaAcc={best_acc:.2f}%, ROC-AUC={auc_s})")
        torch.save(model.state_dict(), os.path.join(run_dir, "checkpoints", "last.pth"))

        if (v_acc - es_best) >= cfg.early_stop_min_delta:
            es_best, no_imp = v_acc, 0
        else:
            no_imp += 1
        if no_imp >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}"); break

    plot_metric_curves(metric_histories, run_dir)
    plot_training_curves(
        train_losses=tl_hist,
        val_losses=vl_hist,
        train_accs=ta_hist,
        val_accs=va_hist,
        save_path=os.path.join(run_dir, "training_curves.png")
    )

    _, _, vp_f, vl_f = validate(model, val_loader, criterion, device, num_classes)
    final_metrics = compute_metrics(vl_f, vp_f, num_classes)
    final_preds = np.argmax(vp_f, 1)
    report = classification_report(vl_f, final_preds, target_names=class_names, digits=4)
    print("\n[Stage2] Classification Report:\n" + report)
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(vl_f, final_preds, labels=list(range(num_classes)))
    plot_confusion_matrix(cm, class_names, os.path.join(run_dir, "confusion_matrix.png"))
    np.savetxt(os.path.join(run_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    try:
        plot_roc_curves(vl_f, vp_f, class_names, os.path.join(run_dir, "roc_curves.png"))
    except Exception as e:
        print(f"ROC曲线绘制失败: {e}")
    try:
        plot_pr_curves(vl_f, vp_f, class_names, os.path.join(run_dir, "pr_curves.png"))
    except Exception as e:
        print(f"PR曲线绘制失败: {e}")

    try:
        plot_performance_summary(final_metrics, os.path.join(run_dir, "performance_summary.png"))
    except Exception as e:
        print(f"性能指标汇总图绘制失败: {e}")

    try:
        generate_gradcam_all(model, [train_loader, val_loader], device, class_names,
                             save_dir=os.path.join(run_dir, "gradcam"))
    except Exception as e:
        print(f"Grad-CAM绘制失败: {e}")

    write_json(os.path.join(run_dir, "metrics.json"), {
        "best": {"epoch": best_ep, "val_acc": best_acc, "val_roc_auc": best_auc},
        "final": {
            "f1_score": float(final_metrics["f1_score"]) if final_metrics["f1_score"] == final_metrics["f1_score"] else None,
            "roc_auc": float(final_metrics["roc_auc"]) if final_metrics["roc_auc"] == final_metrics["roc_auc"] else None,
            "pr_auc": float(final_metrics["pr_auc"]) if final_metrics["pr_auc"] == final_metrics["pr_auc"] else None,
            "recall": float(final_metrics["recall"]),
            "precision": float(final_metrics["precision"]),
            "specificity": float(final_metrics["specificity"]) if final_metrics["specificity"] == final_metrics["specificity"] else None,
        },
        "class_names": class_names,
        "train_loss": [float(x) for x in tl_hist],
        "val_loss": [float(x) for x in vl_hist],
        "train_acc": [float(x) for x in ta_hist],
        "val_acc": [float(x) for x in va_hist],
        "val_f1_score": [float(x) if x == x else None for x in metric_histories["f1_score"]],
        "val_roc_auc": [float(x) if x == x else None for x in metric_histories["roc_auc"]],
        "val_pr_auc": [float(x) if x == x else None for x in metric_histories["pr_auc"]],
        "val_recall": [float(x) if x == x else None for x in metric_histories["recall"]],
        "val_precision": [float(x) if x == x else None for x in metric_histories["precision"]],
        "val_specificity": [float(x) if x == x else None for x in metric_histories["specificity"]],
    })

    def fmt(v):
        return f"{v:.4f}" if v == v else "N/A"

    print(f"\n=== Stage 2 完成 ===")
    print(f"最佳验证准确率: {best_acc:.2f}%  ROC-AUC: {fmt(best_auc)}  Epoch: {best_ep}")
    print("最终验证集指标:")
    print(f"  F1-Score : {fmt(final_metrics['f1_score'])}")
    print(f"  ROC-AUC  : {fmt(final_metrics['roc_auc'])}")
    print(f"  PR-AUC   : {fmt(final_metrics['pr_auc'])}")
    print(f"  召回率    : {fmt(final_metrics['recall'])}")
    print(f"  精确率    : {fmt(final_metrics['precision'])}")
    print(f"  特异度    : {fmt(final_metrics['specificity'])}")
    print(f"结果目录: {run_dir}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",  default=r"C:\Users\86195\PycharmProjects\disease_7150\deformity_data")
    p.add_argument("--label_path", default=r"C:\Users\86195\PycharmProjects\disease_7150\label_list_final_0316.xlsx")
    p.add_argument("--num_classes", type=int,   default=len(FINE_CLASSES_SS))
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--train_split", type=float, default=0.85)
    p.add_argument("--seed",        type=int,   default=51)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--use_amp",     action="store_true")
    p.add_argument("--early_stop_patience",  type=int,   default=25)
    p.add_argument("--early_stop_min_delta", type=float, default=0.05)
    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--visualize_interval",   type=int,   default=20)
    p.add_argument("--results_dir", default="results_stage2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Stage2Config(
        image_dir=args.image_dir, label_path=args.label_path,
        num_classes=args.num_classes, batch_size=args.batch_size,
        lr=args.lr, epochs=args.epochs, train_split=args.train_split,
        seed=args.seed, num_workers=args.num_workers, use_amp=args.use_amp,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        visualize=args.visualize, visualize_interval=args.visualize_interval,
        results_dir=args.results_dir,
    )
    run_stage2(cfg)
