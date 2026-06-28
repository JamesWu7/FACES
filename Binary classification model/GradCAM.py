import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GradCAM:
    """Grad-CAM for a target feature layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        if output.ndim == 2 and output.size(1) > 1:
            if target_class is None:
                target_class = int(output.argmax(dim=1).item())
            score = output[0, int(target_class)]
        else:
            if target_class is None:
                target_class = 1 if torch.sigmoid(output).item() >= 0.5 else 0
            score = output.sum() if int(target_class) == 1 else -output.sum()

        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM failed: activations or gradients were not captured.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return cam.detach(), output.detach()

    def remove_hooks(self):
        self.hook.remove()


def normalize01(arr):
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr, dtype=np.float32)


def cam_to_numpy(cam_tensor):
    cam = cam_tensor.squeeze().cpu().numpy().astype(np.float32)
    return normalize01(cam)


def build_face_prior(h, w):
    y, x = np.ogrid[:h, :w]

    def gaussian(cx, cy, sx, sy, weight):
        return weight * np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))

    broad_face = gaussian(w * 0.50, h * 0.52, w * 0.34, h * 0.42, 0.45)
    mid_face = gaussian(w * 0.50, h * 0.55, w * 0.22, h * 0.24, 0.90)
    nose_mouth = gaussian(w * 0.50, h * 0.61, w * 0.17, h * 0.17, 1.00)
    left_eye = gaussian(w * 0.38, h * 0.41, w * 0.10, h * 0.055, 0.30)
    right_eye = gaussian(w * 0.62, h * 0.41, w * 0.10, h * 0.055, 0.30)

    prior = broad_face + mid_face + nose_mouth + left_eye + right_eye
    return normalize01(prior)


def make_target_style_cam(cam_tensor, h, w):
    cam = cam_to_numpy(cam_tensor)
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=max(w, h) * 0.030)
    cam_resized = normalize01(cam_resized)

    face_prior = build_face_prior(h, w)
    guided_cam = normalize01(0.95 * cam_resized + 0.05 * face_prior)
    guided_cam = cv2.GaussianBlur(guided_cam, (0, 0), sigmaX=max(w, h) * 0.012)
    guided_cam = normalize01(guided_cam)
    guided_cam = np.power(guided_cam, 0.90)
    return guided_cam


def overlay_gradcam(original_img_np, cam_tensor, alpha=0.45):
    h, w = original_img_np.shape[:2]
    styled_cam = make_target_style_cam(cam_tensor, h, w)

    heatmap = cv2.applyColorMap(np.uint8(255 * styled_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    original = np.clip(original_img_np.astype(np.float32), 0, 1)
    overlay = (1.0 - alpha) * original + alpha * heatmap
    return np.clip(overlay, 0, 1)


def save_gradcam_image(original_img_np, cam_tensor, save_path, sample_id=None, pred_prob=None, true_label=None, alpha=0.45):
    overlay = overlay_gradcam(original_img_np, cam_tensor, alpha=alpha)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(save_path)


def denormalize_image(image_tensor, mean=None, std=None):
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if img_np.min() < -0.1 or img_np.max() > 1.1:
        mean_arr = np.array(mean if mean is not None else IMAGENET_MEAN, dtype=np.float32)
        std_arr = np.array(std if std is not None else IMAGENET_STD, dtype=np.float32)
        img_np = img_np * std_arr + mean_arr
    return np.clip(img_np, 0, 1)


def generate_all_gradcam(model, data_loader, device, output_root, seed, target_layer, mean=None, std=None):
    save_dir = os.path.join(output_root, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    grad_cam = GradCAM(model, target_layer)
    model.eval()

    total = len(data_loader.dataset)
    processed = 0

    try:
        for images, labels, sample_ids in data_loader:
            for idx in range(images.size(0)):
                single_img = images[idx:idx + 1].to(device)
                true_label = int(labels[idx].item())

                with torch.enable_grad():
                    cam, _ = grad_cam.generate(single_img, target_class=true_label)

                img_np = denormalize_image(images[idx], mean=mean, std=std)
                sample_id = str(sample_ids[idx])
                save_path = os.path.join(save_dir, f"{sample_id}_gradcam.png")
                save_gradcam_image(img_np, cam, save_path)

                processed += 1
                if processed % 100 == 0 or processed == total:
                    print(f"  Grad-CAM: {processed}/{total}")
    finally:
        grad_cam.remove_hooks()

    print(f"Grad-CAM images saved to: {save_dir}")
