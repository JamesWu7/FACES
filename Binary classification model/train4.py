## 添加了训练曲线合保存数据到json文件
import os
import copy
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix

from dataset2 import get_data_loaders, CustomImageDataset, get_transform, FaceNeckLandmarkCrop
from model3 import ImageClassifier
from VisualizeAC import visualize_activations
from GradCAM import generate_all_gradcam


# --- 绘制训练曲线的函数 ---
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path="training_curves.png"):
    """绘制训练和验证的loss和accuracy曲线"""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存至 {save_path}")


# --- 保存训练结果到JSON文件 ---
def save_training_results(train_losses, val_losses, train_accs, val_accs, best_val_acc,
                          save_path="training_results.json"):
    """保存训练结果到JSON文件"""
    results = {
        "training_info": {
            "total_epochs": len(train_losses),
            "best_validation_accuracy": best_val_acc,
            "final_training_accuracy": train_accs[-1],
            "final_validation_accuracy": val_accs[-1],
            "final_training_loss": train_losses[-1],
            "final_validation_loss": val_losses[-1],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "epoch_details": [
            {
                "epoch": epoch + 1,
                "train_loss": float(train_losses[epoch]),
                "val_loss": float(val_losses[epoch]),
                "train_acc": float(train_accs[epoch]),
                "val_acc": float(val_accs[epoch])
            }
            for epoch in range(len(train_losses))
        ]
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"训练结果已保存至 {save_path}")


import numpy as np


def bootstrap_ci(labels, probs, metric_fn, n_bootstraps=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    labels_arr, probs_arr = np.array(labels), np.array(probs)
    n = len(labels_arr)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels_arr[idx])) < 2:
            continue
        vals.append(metric_fn(labels_arr[idx], probs_arr[idx]))
    vals = np.array(vals)
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 + ci) / 2 * 100)
    return lo, hi


def plot_roc_pr_curves(all_labels, all_probs, save_path):
    labels = np.array(all_labels)
    probs = np.array(all_probs)

    color_hc = '#7ab648'
    color_scoliosis = '#e8a0a0'

    # Scoliosis as positive (label=1)
    fpr_sc, tpr_sc, _ = roc_curve(labels, probs)
    roc_auc_sc = auc(fpr_sc, tpr_sc)

    # HC as positive (label=0): flip labels and probs
    fpr_hc, tpr_hc, _ = roc_curve(1 - labels, 1 - probs)
    roc_auc_hc = auc(fpr_hc, tpr_hc)

    # PR curve - Scoliosis as positive
    prec_sc, rec_sc, _ = precision_recall_curve(labels, probs)
    pr_auc_sc = auc(rec_sc, prec_sc)

    # PR curve - HC as positive
    prec_hc, rec_hc, _ = precision_recall_curve(1 - labels, 1 - probs)
    pr_auc_hc = auc(rec_hc, prec_hc)

    # Bootstrap CI for overall (Scoliosis positive)
    roc_lo, roc_hi = bootstrap_ci(labels, probs, lambda y, p: roc_auc_score(y, p))
    pr_lo, pr_hi = bootstrap_ci(labels, probs, lambda y, p: auc(*precision_recall_curve(y, p)[1::-1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr_sc, tpr_sc, color=color_scoliosis, linewidth=2, label=f'Scoliosis (AUC={roc_auc_sc:.3f})')
    ax1.plot(fpr_hc, tpr_hc, color=color_hc, linewidth=2, label=f'HC (AUC={roc_auc_hc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.05])
    ax1.set_xlabel('1 - Specificity', fontsize=12)
    ax1.set_ylabel('Sensitivity', fontsize=12)
    ax1.set_title('ROC curve (Validation set)', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.text(0.45, 0.35, f'Mean AUC = {(roc_auc_sc + roc_auc_hc) / 2:.3f}', fontsize=10, transform=ax1.transAxes)
    ax1.text(0.45, 0.27, f'(95% CI, {roc_lo:.3f}\u2013{roc_hi:.3f})', fontsize=9, transform=ax1.transAxes)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(rec_sc, prec_sc, color=color_scoliosis, linewidth=2, label=f'Scoliosis (AUC={pr_auc_sc:.3f})')
    ax2.plot(rec_hc, prec_hc, color=color_hc, linewidth=2, label=f'HC (AUC={pr_auc_hc:.3f})')
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('PR curve (Validation set)', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.text(0.45, 0.35, f'Mean PR-AUC = {(pr_auc_sc + pr_auc_hc) / 2:.3f}', fontsize=10, transform=ax2.transAxes)
    ax2.text(0.45, 0.27, f'(95% CI, {pr_lo:.3f}\u2013{pr_hi:.3f})', fontsize=9, transform=ax2.transAxes)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC/PR curves saved: {save_path}")


def plot_confusion_matrix(all_labels, all_probs, save_path, threshold=0.5):
    labels = np.array(all_labels).astype(int)
    preds = (np.array(all_probs) >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity_hc = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity_scoliosis = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tn + tp) / (tn + fp + fn + tp)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    colors = [['#d6e4f0', '#f0f0f0'], ['#f0f0f0', '#b8cfe0']]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=colors[i][j], edgecolor='gray', linewidth=1.5))
            ax.text(j + 0.5, 1.5 - i, str(cm[i][j]), ha='center', va='center', fontsize=22, fontweight='bold')

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['HC', 'Scoliosis'], fontsize=12)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Scoliosis', 'HC'], fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Predicted', fontsize=13, labelpad=10)
    ax.set_ylabel('True', fontsize=13, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(length=0)

    right_x = 2.15
    ax.text(right_x, 1.7, 'Sensitivity (Recall)', fontsize=10, va='center')
    ax.text(right_x, 1.4, f'HC: {sensitivity_hc * 100:.2f}%', fontsize=10, va='center')
    ax.text(right_x, 1.1, f'Scoliosis: {sensitivity_scoliosis * 100:.2f}%', fontsize=10, va='center')
    ax.text(right_x, 0.5, f'Specificity: {specificity * 100:.2f}%', fontsize=10, va='center')
    ax.text(1.0, -0.15, f'Accuracy = {accuracy * 100:.2f}%', ha='center', va='center', fontsize=12, transform=ax.transData)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")



def plot_performance_metrics(all_labels, all_probs, save_path, threshold=0.5):
    labels = np.array(all_labels).astype(int)
    preds = (np.array(all_probs) >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0

    metric_names = ['F1-score', 'Precision (PPV)', 'Specificity', 'Recall (Sensitivity)', 'Accuracy']
    metric_values = [f1, precision, specificity, recall, accuracy]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(metric_names, metric_values, color='#b0a4c7', edgecolor='none', height=0.6)

    ax.set_xlim(0, 105)
    ax.set_xlabel('Performance (%)', fontsize=11)
    ax.set_title('Performance Metrics (Validation set)', fontsize=13, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10)

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance metrics chart saved: {save_path}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
BASE_DIR = Path(__file__).resolve().parent
ACTIVATION_MAPS_ROOT = str(BASE_DIR / "all_activation_maps")
GRADCAM_ROOT = str(BASE_DIR / "all_gradcam")


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, sample_ids in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    return running_loss / total, 100. * correct / total


def validate(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, sample_ids in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    val_loss = running_loss / total
    val_acc = 100. * correct / total
    val_auc = roc_auc_score(
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy()
    )

    return val_loss, val_acc, val_auc


def build_all_data_loader(image_dir, label_path):
    full_dataset = CustomImageDataset(
        image_dir=image_dir,
        label_path=label_path,
        transform=get_transform(train=False),
        roi_cropper=FaceNeckLandmarkCrop()
    )
    return DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


def generate_all_activation_maps(model, data_loader, seed):
    print("正在为所有样本生成激活图...")
    model.eval()

    root_dir = os.path.join(ACTIVATION_MAPS_ROOT, f"seed_{seed}")
    os.makedirs(root_dir, exist_ok=True)

    activations = {}

    def hook_fn(name):
        def hook(module, inputs, output):
            activations[name] = output.detach().cpu()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for images, labels, sample_ids in data_loader:
            images = images.to(DEVICE)
            model(images)

            for sample_idx, sample_name in enumerate(sample_ids):
                sample_dir = os.path.join(root_dir, sample_name)
                sample_activations = {
                    layer_name: activation[sample_idx:sample_idx + 1].clone()
                    for layer_name, activation in activations.items()
                }
                visualize_activations(sample_activations, sample_dir, sample_name=sample_name)

    for hook in hooks:
        hook.remove()

    print(f"所有样本的激活图已保存至 {root_dir}")


def main(seed=51):
    os.makedirs(str(BASE_DIR / "results"), exist_ok=True)
    os.makedirs(ACTIVATION_MAPS_ROOT, exist_ok=True)
    os.makedirs(GRADCAM_ROOT, exist_ok=True)

    print("加载数据集...")
    image_dir = str(BASE_DIR / "deformity_data")
    label_path = str(BASE_DIR / "label_list_final_0316.xlsx")
    train_loader, val_loader = get_data_loaders(
        image_dir=image_dir,
        label_path=label_path,
        batch_size=BATCH_SIZE,
        train_split=0.85,
        random_seed=seed
    )
    all_data_loader = build_all_data_loader(image_dir, label_path)

    print("初始化模型...")
    model = ImageClassifier(input_channels=3, dropout_rate=0.35).to(DEVICE)
    pos_weight = torch.tensor([1168 / 1407], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"开始训练（使用{DEVICE}）...")
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), str(BASE_DIR / "results" / f"best_auc_model_seed{seed}.pth"))
            print(f"✔ 保存最佳模型（AUC = {best_val_auc:.4f}）")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    model.load_state_dict(best_model_state)

    # Plot ROC and PR curves using best model on validation set
    model.eval()
    final_probs = []
    final_labels = []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            final_probs.append(probs.cpu())
            final_labels.append(labels.cpu())
    final_probs = torch.cat(final_probs).numpy().flatten()
    final_labels = torch.cat(final_labels).numpy().flatten()
    plot_roc_pr_curves(
        final_labels, final_probs,
        save_path=str(BASE_DIR / "results" / f"ROC_PR_curves_seed{seed}.png")
    )
    plot_confusion_matrix(
        final_labels, final_probs,
        save_path=str(BASE_DIR / "results" / f"confusion_matrix_seed{seed}.png")
    )
    plot_performance_metrics(
        final_labels, final_probs,
        save_path=str(BASE_DIR / "results" / f"performance_metrics_seed{seed}.png")
    )

    print("Generating Grad-CAM for all samples...")
    generate_all_gradcam(
        model=model,
        data_loader=all_data_loader,
        device=DEVICE,
        output_root=GRADCAM_ROOT,
        seed=seed,
        target_layer=model.conv_block3,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    print("已跳过普通激活图生成，仅生成Grad-CAM图。")

    print(f"训练结束！最佳验证准确率：{best_val_acc:.2f}%")

    print("正在生成训练曲线和保存结果...")
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_path=str(BASE_DIR / "results" / f"Val_Loss_{val_loss:.4f}_Val_Acc_{val_acc:.2f}%training_curves.png")
    )
    save_training_results(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        best_val_acc,
        save_path=str(BASE_DIR / "results" / f"Val_Loss_{val_loss:.4f}_Val_Acc_{val_acc:.2f}%training_results.json")
    )

    print("\n=== 训练结果摘要 ===")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最终训练准确率: {train_accs[-1]:.2f}%")
    print(f"最终验证准确率: {val_accs[-1]:.2f}%")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {val_losses[-1]:.4f}")
    print("===================")


if __name__ == "__main__":
    seeds = [99, 64, 65, 69, 53, 123]
    for i in range(6):
        print(f"这是第{i + 1}次循环的结果")
        main(seeds[i])
