from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from .config import (
    BINARY_CLASSES,
    BINARY_WEIGHTS,
    DLIB_LANDMARK_MODEL,
    MODEL_VERSIONS,
    SS_DISPLAY_THRESHOLD,
    SS_SUBTYPE_CLASSES,
    SUBTYPE_WEIGHTS,
    SUPERCLASS_CLASSES,
    SUPERCLASS_FULL_NAMES,
    SUPERCLASS_WEIGHTS,
)
from .models import BinaryCNNCBAM, ResNet18Etiology, ResNet50SSSubtype
from .preprocess import BINARY_TRANSFORM, RESNET_TRANSFORM, FaceCropper


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _ranked_results(class_names, probabilities):
    items = [
        {
            "class_name": str(class_name),
            "probability": float(probability),
            "probability_percent": float(probability) * 100.0,
        }
        for class_name, probability in zip(class_names, probabilities)
    ]
    return sorted(items, key=lambda item: item["probability"], reverse=True)


class FACESPredictor:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cropper = FaceCropper(DLIB_LANDMARK_MODEL)
        self.binary_model = self._load_binary_model()
        self.etiology_model = self._load_etiology_model()
        self.subtype_model = self._load_subtype_model()

    def _load_binary_model(self):
        if not BINARY_WEIGHTS.exists():
            raise FileNotFoundError(f"Missing binary model weights: {BINARY_WEIGHTS}")
        model = BinaryCNNCBAM(input_channels=3, dropout_rate=0.35)
        model.load_state_dict(_torch_load(BINARY_WEIGHTS, self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_etiology_model(self):
        if not SUPERCLASS_WEIGHTS.exists():
            raise FileNotFoundError(f"Missing etiology model weights: {SUPERCLASS_WEIGHTS}")
        model = ResNet18Etiology(dropout_rate=0.5)
        model.load_state_dict(_torch_load(SUPERCLASS_WEIGHTS, self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_subtype_model(self):
        if not SUBTYPE_WEIGHTS.exists():
            raise FileNotFoundError(f"Missing SS subtype model weights: {SUBTYPE_WEIGHTS}")
        model = ResNet50SSSubtype(num_classes=len(SS_SUBTYPE_CLASSES), dropout_rate=0.5)
        model.load_state_dict(_torch_load(SUBTYPE_WEIGHTS, self.device))
        model.to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        if image is None:
            raise ValueError("请先上传一张正面人脸照片。")

        warnings = []
        crop_result = self.cropper.crop(image)
        if crop_result.warning:
            warnings.append(crop_result.warning)
        face_image = crop_result.image.convert("RGB")

        binary_tensor = BINARY_TRANSFORM(face_image).unsqueeze(0).to(self.device)
        resnet_tensor = RESNET_TRANSFORM(face_image).unsqueeze(0).to(self.device)

        disease_logit = self.binary_model(binary_tensor)
        disease_prob = torch.sigmoid(disease_logit).flatten()[0].item()
        binary_probs = [1.0 - disease_prob, disease_prob]

        etiology_logits = self.etiology_model(resnet_tensor)
        etiology_probs = torch.softmax(etiology_logits, dim=1).flatten().cpu().tolist()

        subtype_logits = self.subtype_model(resnet_tensor)
        subtype_probs = torch.softmax(subtype_logits, dim=1).flatten().cpu().tolist()

        binary = _ranked_results(BINARY_CLASSES, binary_probs)
        etiology = _ranked_results(SUPERCLASS_CLASSES, etiology_probs)
        ss_subtypes = _ranked_results(SS_SUBTYPE_CLASSES, subtype_probs)

        etiology_top1 = etiology[0]["class_name"]
        ss_probability = next(item["probability"] for item in etiology if item["class_name"] == "SS")
        show_ss_subtypes = etiology_top1 == "SS" or ss_probability >= SS_DISPLAY_THRESHOLD
        if not show_ss_subtypes:
            warnings.append(
                "SS 不是当前三分类最高支持类别，SS 11类细分结果仅作为补充参考。"
            )

        return {
            "cropped_face": face_image,
            "face_crop_method": crop_result.method,
            "binary": binary,
            "etiology": etiology,
            "etiology_full_names": SUPERCLASS_FULL_NAMES,
            "ss_subtypes": ss_subtypes,
            "show_ss_subtypes": show_ss_subtypes,
            "warnings": warnings,
            "model_versions": MODEL_VERSIONS,
            "device": str(self.device),
            "top_summary": {
                "screening_result": binary[0]["class_name"],
                "screening_probability": binary[0]["probability"],
                "etiology_top1": etiology_top1,
                "etiology_probability": etiology[0]["probability"],
                "ss_subtype_top3": ss_subtypes[:3],
            },
        }


@lru_cache(maxsize=1)
def get_predictor() -> FACESPredictor:
    return FACESPredictor()


def predict_faces(image: Image.Image) -> dict:
    return get_predictor().predict(image)
