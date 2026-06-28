## ResNet18 最终优化版：人脸疾病三分类（SS / CMS / AIS）
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from dataset2 import get_data_loaders, CustomImageDataset, SubsetWithTransform, get_transform
from model3 import ImageClassifier
from VisualizeAC import visualize_activations
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ['SS', 'CMS', 'AIS']
NUM_CLASSES = 3


def mixup_data(x, y, alpha=0.3, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    all_probs_np = torch.cat(all_probs).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    try:
        val_auc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro')
    except ValueError:
        val_auc = 0.0
    return val_loss, val_acc, val_auc


def train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=0.3):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        if np.random.random() < 0.5:
            images, la, lb, lam = mixup_data(images, labels, mixup_alpha, device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, la, lb, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100. * correct / total


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至 {save_path}")


def plot_confusion_matrix(model, val_loader, device, save_path):
    """训练结束后生成内部验证混淆矩阵图（行归一化百分比 + 样本数）"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / row_sums * 100

    class_counts = row_sums.flatten()
    pred_counts = cm.sum(axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_percent, cmap='Blues', vmin=0, vmax=100)

    for i in range(3):
        for j in range(3):
            count = cm[i, j]
            pct = cm_percent[i, j]
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha='center', va='center', fontsize=11,
                    color='white' if pct > 60 else 'black')

    y_labels = [f"{CLASS_NAMES[i]}\n(n={int(class_counts[i])})" for i in range(3)]
    x_labels = [f"{CLASS_NAMES[i]}\n(n={int(pred_counts[i])})" for i in range(3)]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title('Internal Validation Confusion Matrix', fontsize=13, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Row\nnormalized (%)', fontsize=10, rotation=0, labelpad=40, va='center')

    ax.text(0.5, -0.18, 'Values are number of cases (%)',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"内部验证混淆矩阵已保存至 {save_path}")



def plot_roc_and_pr_curves(model, val_loader, device, save_path):
    """绘制 One-vs-Rest ROC 曲线和 PR 曲线（AIS=橙色, CMS=蓝色, SS=绿色）"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    colors = {'AIS': '#FF8C00', 'CMS': '#1F77B4', 'SS': '#2CA02C'}
    class_order = [2, 1, 0]  # AIS, CMS, SS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：One-vs-Rest ROC curves
    for cls_idx in class_order:
        cls_name = CLASS_NAMES[cls_idx]
        y_true_bin = (all_labels == cls_idx).astype(int)
        y_score = all_probs[:, cls_idx]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[cls_name], linewidth=2,
                 label=f"{cls_name} (AUC = {roc_auc_val:.3f})")

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('1 \u2013 Specificity', fontsize=12)
    ax1.set_ylabel('Sensitivity', fontsize=12)
    ax1.set_title('One-vs-rest ROC curves (validation set)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.tick_params(labelsize=10)

    # 右图：One-vs-Rest PR curves
    for cls_idx in class_order:
        cls_name = CLASS_NAMES[cls_idx]
        y_true_bin = (all_labels == cls_idx).astype(int)
        y_score = all_probs[:, cls_idx]
        precision_arr, recall_arr, _ = precision_recall_curve(y_true_bin, y_score)
        ap = average_precision_score(y_true_bin, y_score)
        ax2.plot(recall_arr, precision_arr, color=colors[cls_name], linewidth=2,
                 label=f"{cls_name} (AUPRC = {ap:.3f})")

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('One-vs-rest PR curves (validation set)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC 和 PR 曲线已保存至 {save_path}")


def plot_performance_bar(model, val_loader, device, save_path):
    """绘制内部验证性能指标水平条形图（Accuracy, Recall, Specificity, Precision, F1-score）"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算 macro 平均指标
    precision_vals, recall_vals, f1_vals, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], average='macro', zero_division=0
    )

    # Accuracy
    accuracy = np.mean(all_preds == all_labels) * 100

    # Macro Recall (Sensitivity)
    recall = recall_vals * 100

    # Macro Specificity: 对每个类取 TN/(TN+FP) 然后求平均
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    specificities = []
    for i in range(3):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    specificity = np.mean(specificities) * 100

    # Macro Precision (PPV)
    precision = precision_vals * 100

    # Macro F1-score
    f1 = f1_vals * 100

    # 指标名称和数值（从下往上排列，与参考图一致）
    metrics_names = ['F1-score', 'Precision (PPV)', 'Specificity', 'Recall (Sensitivity)', 'Accuracy']
    metrics_values = [f1, precision, specificity, recall, accuracy]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = range(len(metrics_names))
    bars = ax.barh(y_pos, metrics_values, color='#9B89B3', edgecolor='none', height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_names, fontsize=11)
    ax.set_xlim(0, 110)
    ax.set_xlabel('Performance (%)', fontsize=11)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xticks(range(0, 101, 20))

    # 在条形右侧标注百分比数值
    for bar, val in zip(bars, metrics_values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', ha='left', fontsize=11)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能指标条形图已保存至 {save_path}")


def generate_gradcam_all(model, image_dir, label_path, device, output_dir):
    """为数据集中每张图片生成 Grad-CAM 热力图并保存"""
    os.makedirs(output_dir, exist_ok=True)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                   and os.path.isfile(os.path.join(image_dir, f))]

    model.eval()
    target_layer = model.layer4[-1].conv2

    count = 0
    for img_name in sorted(image_files):
        img_path = os.path.join(image_dir, img_name)
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        input_tensor = val_transform(pil_img).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1.0
        output.backward(gradient=one_hot)

        handle_fwd.remove()
        handle_bwd.remove()

        grad = gradients[0].detach()
        act = activations[0].detach()

        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        orig_img = pil_img.resize((224, 224))
        orig_np = np.array(orig_img).astype(np.float32)

        overlay = 0.5 * orig_np + 0.5 * heatmap.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        save_name = os.path.splitext(img_name)[0] + '_gradcam.png'
        save_path = os.path.join(output_dir, save_name)
        Image.fromarray(overlay).save(save_path)
        count += 1

    print(f"Grad-CAM 已为 {count} 张图片生成，保存至 {output_dir}")

def save_training_results(train_losses, val_losses, train_accs, val_accs,
                          best_val_acc, best_val_auc, save_path):
    results = {
        "training_info": {
            "total_epochs": len(train_losses),
            "best_validation_accuracy": best_val_acc,
            "best_validation_auc": best_val_auc,
            "final_train_acc": train_accs[-1],
            "final_val_acc": val_accs[-1],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "epoch_details": [
            {"epoch": e + 1, "train_loss": float(train_losses[e]),
             "val_loss": float(val_losses[e]), "train_acc": float(train_accs[e]),
             "val_acc": float(val_accs[e])}
            for e in range(len(train_losses))
        ]
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"训练结果已保存至 {save_path}")


# ==================== 超参数 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR_SHALLOW   = 5e-6   # conv1/bn1/layer1/layer2：极小 lr
LR_DEEP      = 5e-5   # layer3/layer4：小 lr
LR_HEAD      = 3e-4   # 分类头：正常 lr
WEIGHT_DECAY = 1e-3
EPOCHS = 150
WARMUP_EPOCHS = 8
MIXUP_ALPHA = 0.3
VISUALIZE_INTERVAL = 20
EARLY_STOP_PAT = 25
# ================================================


def main(seed=51):
    os.makedirs("results", exist_ok=True)

    print("加载数据集...")
    IMAGE_DIR  = r"C:\Users\86195\PycharmProjects\disease_3(final_8111)\deformity_data"
    LABEL_PATH = r"C:\Users\86195\PycharmProjects\disease_3(final_8111)\label_list_final_0316.xlsx"

    # class_weights 由 get_data_loaders 返回，但 WeightedRandomSampler 已做三类均衡
    # 不再传入 CrossEntropyLoss，避免双重纠偏
    train_loader, val_loader, _ = get_data_loaders(
        image_dir=IMAGE_DIR, label_path=LABEL_PATH,
        batch_size=BATCH_SIZE, train_split=0.85, random_seed=seed
    )

    print("初始化模型（ResNet18 全层微调，分层学习率）...")
    model = ImageClassifier(dropout_rate=0.5).to(DEVICE)
    print(model.get_param_count())

    # WeightedRandomSampler 已做三类均衡，CrossEntropyLoss 无需 class_weights
    # label_smoothing=0.1 抑制过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 三组分层学习率
    shallow_params, deep_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if name.startswith('classifier'):
            head_params.append(param)
        elif any(name.startswith(p) for p in ['conv1', 'bn1', 'layer1', 'layer2']):
            shallow_params.append(param)
        else:
            deep_params.append(param)

    optimizer = optim.AdamW([
        {'params': shallow_params, 'lr': LR_SHALLOW, 'weight_decay': WEIGHT_DECAY},
        {'params': deep_params,    'lr': LR_DEEP,    'weight_decay': WEIGHT_DECAY},
        {'params': head_params,    'lr': LR_HEAD,    'weight_decay': WEIGHT_DECAY * 2},
    ])

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    val_images_vis, _ = next(iter(val_loader))
    val_images_vis = val_images_vis.to(DEVICE)

    best_val_acc = 0.0
    best_val_auc = 0.0
    early_stop_counter = 0

    print(f"开始训练（{DEVICE}）...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, MIXUP_ALPHA
        )
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr_head = optimizer.param_groups[2]['lr']
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
            f"Val AUC: {val_auc:.4f}  LR(head): {lr_head:.2e}"
        )

        # 激活图可视化（保留原有功能）
        if (epoch + 1) % VISUALIZE_INTERVAL == 0 or epoch == 0:
            print("正在生成激活图...")
            model.eval()
            activations = {}

            def hook_fn(name):
                def hook(module, inp, out):
                    activations[name] = out
                return hook

            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            with torch.no_grad():
                model(val_images_vis)
            for h in hooks:
                h.remove()
            visualize_activations(activations, epoch)

        # 早停基于 val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), f"results/best_acc_model_seed{seed}.pth")
            print(f"  [Best ACC] {best_val_acc:.2f}%  AUC={val_auc:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PAT:
                print(f"  早停：val_acc 连续 {EARLY_STOP_PAT} 轮未提升")
                break

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"results/best_auc_model_seed{seed}.pth")
            print(f"  [Best AUC] {best_val_auc:.4f}  Acc={val_acc:.2f}%")

    safe_name = f"ValAcc{best_val_acc:.2f}_AUC{best_val_auc:.4f}_seed{seed}"
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=f"results/{safe_name}_training_curves.png"
    )
    plot_confusion_matrix(
        model, val_loader, DEVICE,
        save_path=f"results/{safe_name}_confusion_matrix.png"
    )
    plot_roc_and_pr_curves(
        model, val_loader, DEVICE,
        save_path=f"results/{safe_name}_roc_pr_curves.png"
    )
    plot_performance_bar(
        model, val_loader, DEVICE,
        save_path=f"results/{safe_name}_performance_bar.png"
    )
    generate_gradcam_all(
        model, IMAGE_DIR, LABEL_PATH, DEVICE,
        output_dir=f"results/{safe_name}_gradcam"
    )
    save_training_results(
        train_losses, val_losses, train_accs, val_accs, best_val_acc, best_val_auc,
        save_path=f"results/{safe_name}_training_results.json"
    )

    print("\n=== 训练结果摘要 ===")
    print("骨干网络：ResNet18（全层微调，分层学习率，WeightedRandomSampler）")
    print(f"实际训练轮数: {len(train_losses)}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最佳验证AUC:    {best_val_auc:.4f}")
    print(f"最终训练准确率: {train_accs[-1]:.2f}%")
    print(f"最终验证准确率: {val_accs[-1]:.2f}%")
    print("===================")


if __name__ == "__main__":
    seeds = [99, 64, 65, 69, 53, 123]
    for i in range(6):
        print("\n" + "=" * 50)
        print(f"第 {i + 1} 次循环，seed={seeds[i]}")
        print("=" * 50)
        main(seeds[i])
