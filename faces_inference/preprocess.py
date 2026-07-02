from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms


try:
    import dlib
except ImportError:  # pragma: no cover - depends on local installation
    dlib = None


try:
    import cv2
except ImportError:  # pragma: no cover - depends on local installation
    cv2 = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BINARY_TRANSFORM = transforms.Compose([
    transforms.Resize((210, 140)),
    transforms.ToTensor(),
])

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@dataclass
class FaceCropResult:
    image: Image.Image
    method: str
    warning: str | None = None


class FaceCropper:
    def __init__(self, predictor_path: Path):
        self.predictor_path = Path(predictor_path)
        self.dlib_available = dlib is not None and self.predictor_path.exists()
        self.cv2_available = cv2 is not None
        self._dlib_detector = None
        self._dlib_predictor = None
        self._haar = None

        if self.dlib_available:
            self._dlib_detector = dlib.get_frontal_face_detector()
            self._dlib_predictor = dlib.shape_predictor(str(self.predictor_path))
        elif self.cv2_available:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self._haar = cv2.CascadeClassifier(str(cascade_path))

    def crop(self, image: Image.Image) -> FaceCropResult:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.dlib_available:
            cropped = self._crop_with_dlib(image)
            if cropped is not None:
                return FaceCropResult(cropped, method="dlib_68_landmarks")

        if self._haar is not None:
            cropped = self._crop_with_haar(image)
            if cropped is not None:
                return FaceCropResult(
                    cropped,
                    method="opencv_haar",
                    warning="dlib/68点模型不可用，已使用 OpenCV Haar 人脸检测作为回退。",
                )

        raise ValueError("未检测到可靠正脸，请上传清晰、无遮挡、正面的人脸照片。")

    def _crop_with_dlib(self, image: Image.Image) -> Image.Image | None:
        image_np = np.array(image)
        detections = self._dlib_detector(image_np, 1)
        if not detections:
            return None

        face = max(detections, key=lambda rect: rect.width() * rect.height())
        shape = self._dlib_predictor(image_np, face)
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
            return None
        return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def _crop_with_haar(self, image: Image.Image) -> Image.Image | None:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        side_margin = int(w * 0.18)
        top_margin = int(h * 0.18)
        bottom_margin = int(h * 0.18)
        x1 = max(0, x - side_margin)
        y1 = max(0, y - top_margin)
        x2 = min(image.width, x + w + side_margin)
        y2 = min(image.height, y + h + bottom_margin)
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))
