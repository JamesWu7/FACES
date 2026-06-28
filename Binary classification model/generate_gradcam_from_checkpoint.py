import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset2 import CustomImageDataset, get_transform
from model3 import ImageClassifier
from GradCAM import generate_all_gradcam


BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent


def find_default_model_path(results_dir):
    candidates = sorted(
        Path(results_dir).glob("best_auc_model_seed*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"未在 {results_dir} 下找到 best_auc_model_seed*.pth")
    return candidates[0]


def load_model(model_path):
    model = ImageClassifier(input_channels=3, dropout_rate=0.35).to(DEVICE)
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_gradcam_loader(image_dir, label_path, batch_size):
    dataset = CustomImageDataset(
        image_dir=str(image_dir),
        label_path=str(label_path),
        transform=get_transform(train=False),
        roi_cropper=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser(description="使用已训练模型为全部样本生成 Grad-CAM 图")
    parser.add_argument("--image_dir", default=str(BASE_DIR / "deformity_data"), help="图片文件夹")
    parser.add_argument("--label_path", default=str(BASE_DIR / "label_list_final_0316.xlsx"), help="标签Excel路径")
    parser.add_argument("--model_path", default=None, help="模型权重路径，默认读取 results 下最新 best_auc_model_seed*.pth")
    parser.add_argument("--output_dir", default=str(BASE_DIR / "all_gradcam_from_checkpoint"), help="Grad-CAM输出目录")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--seed_name", default="checkpoint", help="输出子文件夹名称")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else find_default_model_path(BASE_DIR / "results")
    print(f"使用模型: {model_path}")
    print(f"使用设备: {DEVICE}")

    model = load_model(model_path)
    data_loader = build_gradcam_loader(args.image_dir, args.label_path, args.batch_size)

    generate_all_gradcam(
        model=model,
        data_loader=data_loader,
        device=DEVICE,
        output_root=args.output_dir,
        seed=args.seed_name,
        target_layer=model.cbam,
        mean=None,
        std=None,
    )


if __name__ == "__main__":
    main()
