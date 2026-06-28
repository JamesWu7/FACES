import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from dataset_hierarchical import FINE_CLASSES_SS
from model2 import ImageClassifier


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def load_model(weights_path: str, device: torch.device):
    num_classes = len(FINE_CLASSES_SS)
    model = ImageClassifier(num_classes=num_classes)
    state = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"当前数据集包含 {num_classes} 个SS病因小类，模型也按 {num_classes} 类初始化。"
        ) from exc
    model.to(device)
    model.eval()
    print(f"已加载模型: {weights_path}")
    print(f"当前SS病因小类数量: {num_classes}")
    return model


def predict_probabilities(model, image_path: str, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        if logits.shape[1] != len(FINE_CLASSES_SS):
            raise ValueError(
                f"模型输出类别数为 {logits.shape[1]}，但当前SS病因小类数量为 {len(FINE_CLASSES_SS)}。"
                "请确认使用的是按当前11类数据训练得到的权重。"
            )
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    results = [
        {
            "class_name": class_name,
            "probability": float(prob),
            "rad_score": float(prob),
        }
        for class_name, prob in zip(FINE_CLASSES_SS, probs)
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def save_probability_text(results, image_path: str, out_path: str):
    lines = [
        f"图片: {image_path}",
        f"SS病因小类数量: {len(FINE_CLASSES_SS)}",
        "病因小类概率预测结果（rad-score）:",
        "",
    ]
    for item in results:
        lines.append(f"{item['class_name']}: rad-score={item['rad_score']:.4f} ({item['probability'] * 100:.2f}%)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"文字版rad-score结果已保存: {out_path}")


def save_probability_json(results, image_path: str, out_path: str):
    payload = {
        "image_path": image_path,
        "num_classes": len(FINE_CLASSES_SS),
        "class_order": list(FINE_CLASSES_SS),
        "predictions": results,
        "rad_scores": [
            {"class_name": item["class_name"], "rad_score": item["rad_score"]}
            for item in results
        ],
        "top1_class": results[0]["class_name"],
        "top1_probability": results[0]["probability"],
        "top1_rad_score": results[0]["rad_score"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"JSON概率/rad-score结果已保存: {out_path}")


def save_probability_bar_chart(results, out_path: str):
    class_names = [item["class_name"] for item in results]
    probabilities = [item["probability"] for item in results]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(class_names)), probabilities, color="#4C72B0", edgecolor="#1F2A44")
    plt.xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    plt.ylabel("Rad-score")
    plt.xlabel("SS disease subclasses")
    plt.title("Rad-score of Each SS Disease Subclass")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)

    for bar, prob in zip(bars, probabilities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(prob + 0.015, 0.98),
            f"{prob:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"柱状图已保存: {out_path}")


def run_inference(image_path: str, weights_path: str, output_dir: str):
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_model(weights_path, device)
    results = predict_probabilities(model, image_path, device)

    print("\n=== 该图片属于各病因小类的概率（rad-score） ===")
    for item in results:
        print(f"{item['class_name']}: rad-score={item['rad_score']:.4f} ({item['probability'] * 100:.2f}%)")

    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    save_probability_text(results, image_path, os.path.join(output_dir, f"{image_stem}_probabilities.txt"))
    save_probability_json(results, image_path, os.path.join(output_dir, f"{image_stem}_probabilities.json"))
    save_probability_bar_chart(results, os.path.join(output_dir, f"{image_stem}_probabilities_bar.png"))

    print(f"\nTop-1 预测: {results[0]['class_name']} | 概率: {results[0]['probability']:.4f} | rad-score: {results[0]['rad_score']:.4f}")
    print(f"结果目录: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="输出单张图片属于所有SS病因小类的概率（rad-score）与柱状图")
    p.add_argument("--image_path", default=r"C:\Users\86195\PycharmProjects\disease_7150\deformity_data\457.jpg", help="待验证图片路径")
    p.add_argument("--weights_path", default=r"C:\Users\86195\PycharmProjects\disease_7150\results_stage2\stage2_20260524_010616_seed51\checkpoints\best.pth", help="stage2 best.pth 权重路径")
    p.add_argument("--output_dir", default="single_image_probs", help="输出目录")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        image_path=args.image_path,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
    )
