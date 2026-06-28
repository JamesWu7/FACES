from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LANDMARK_COUNT = 68
BOUNDARY_FRACTIONS = np.array([
    (0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5),
    (1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.5),
], dtype=np.float32)


@dataclass
class DetectedFace:
    sample_id: str
    image: np.ndarray
    points: np.ndarray


@dataclass
class NormalizedFace:
    sample_id: str
    image: np.ndarray
    points: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按病因大类生成经典平均脸（dlib + similarity transform + Delaunay）。")
    parser.add_argument("--images-dir", type=Path, default=Path("pingjunlian_2"))
    parser.add_argument("--labels-file", type=Path, default=Path("pingjunlian_2.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("average_faces_by_subclass_dlib"))
    parser.add_argument("--predictor-path", type=Path, default=Path("shape_predictor_68_face_landmarks.dat"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(600, 600))
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r'[\\/:*?"<>|]+', "_", str(name).strip())) or "unknown"


def load_labels(labels_file: Path, sheet_name: str | int) -> pd.DataFrame:
    df = pd.read_excel(labels_file, sheet_name=sheet_name)
    cols = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    if len(cols) < 3:
        raise ValueError("标签文件至少需要三列有效数据：ID、病因大类、病因小类")
    df = df[cols[:3]].copy()
    df.columns = ["ID", "病因大类", "病因小类"]
    df = df.dropna(subset=["ID", "病因大类"])
    df["ID"] = df["ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["病因大类"] = df["病因大类"].astype(str).str.strip()
    return df[["ID", "病因大类"]]


def find_image_path(images_dir: Path, sample_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        for file_name in (f"{sample_id}{suffix}", f"{sample_id}{suffix.upper()}"):
            path = images_dir / file_name
            if path.exists():
                return path
    for path in sorted(images_dir.glob(f"{sample_id}.*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255.0


def boundary_points(image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    return BOUNDARY_FRACTIONS * np.array([w - 1, h - 1], dtype=np.float32)


def eye_corner_dst(image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    return np.array([(0.3 * w, h / 3.0), (0.7 * w, h / 3.0)], dtype=np.float32)


def similarity_transform(in_points: np.ndarray, out_points: np.ndarray) -> np.ndarray:
    s60 = math.sin(math.radians(60))
    c60 = math.cos(math.radians(60))
    in_pts = in_points.astype(np.float32).tolist()
    out_pts = out_points.astype(np.float32).tolist()

    xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]
    xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    in_pts.append([xin, yin])
    out_pts.append([xout, yout])
    return cv2.getAffineTransform(np.float32(in_pts), np.float32(out_pts))


def detect_landmarks(image: np.ndarray, detector, predictor) -> np.ndarray | None:
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 1)
    if not faces:
        return None
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    shape = predictor(gray, face)
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(LANDMARK_COUNT)], dtype=np.float32)


def prepare_detected_face(sample_id: str, image_path: Path, detector, predictor) -> DetectedFace | None:
    image = load_rgb_image(image_path)
    points = detect_landmarks(image, detector, predictor)
    if points is None:
        return None
    return DetectedFace(sample_id, image, points)


def normalize_face(face: DetectedFace, image_size: tuple[int, int], dst_eye_corners: np.ndarray, boundary: np.ndarray) -> NormalizedFace:
    tform = similarity_transform(face.points[[36, 45]], dst_eye_corners)
    image = cv2.warpAffine(face.image, tform, image_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    points = cv2.transform(face.points.reshape(1, -1, 2), tform).reshape(-1, 2)
    points = np.vstack([points, boundary])
    return NormalizedFace(face.sample_id, image, points)


def rect_contains(rect: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    x, y, w, h = rect
    return x <= point[0] < x + w and y <= point[1] < y + h


def calculate_delaunay_triangles(rect: tuple[int, int, int, int], points: np.ndarray) -> list[tuple[int, int, int]]:
    subdiv = cv2.Subdiv2D(rect)
    for point in points:
        subdiv.insert((float(point[0]), float(point[1])))
    triangles = []
    seen = set()
    for t in subdiv.getTriangleList():
        tri = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if not all(rect_contains(rect, p) for p in tri):
            continue
        idx = []
        for p in tri:
            dist = np.linalg.norm(points - np.array(p, dtype=np.float32), axis=1)
            k = int(np.argmin(dist))
            if dist[k] > 1.0:
                idx = []
                break
            idx.append(k)
        if len(idx) == 3 and len(set(idx)) == 3:
            item = tuple(idx)
            if item not in seen:
                seen.add(item)
                triangles.append(item)
    return triangles


def warp_triangle(src: np.ndarray, dst: np.ndarray, src_tri: np.ndarray, dst_tri: np.ndarray) -> None:
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))
    if min(r1[2], r1[3], r2[2], r2[3]) <= 0:
        return
    t1 = src_tri - np.array([r1[0], r1[1]], dtype=np.float32)
    t2 = dst_tri - np.array([r2[0], r2[1]], dtype=np.float32)
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(np.round(t2)), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)
    src_rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    warp_mat = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
    warped = cv2.warpAffine(src_rect, warp_mat, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    region = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    region *= 1.0 - mask
    region += warped * mask


def should_preserve_spots(subclass: str) -> bool:
    normalized = subclass.strip().lower().replace("_", "-").replace(" ", "")
    return normalized in {"nf-1", "nf1", "neurofibromatosistype1"} or "nf-1" in normalized or "nf1" in normalized


def extract_spot_mask(image: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[..., 0] / 255.0
    a_channel = lab[..., 1] / 255.0
    b_channel = lab[..., 2] / 255.0

    local_mean = cv2.GaussianBlur(l_channel, (0, 0), 6)
    local_detail = np.clip(local_mean - l_channel, 0.0, None)
    fine_detail = np.clip(
        cv2.GaussianBlur(local_detail, (0, 0), 0.8) - cv2.GaussianBlur(local_detail, (0, 0), 2.2),
        0.0,
        None,
    )
    warm_bias = np.clip((a_channel - 0.53) * 1.7 + (b_channel - 0.53) * 0.9, 0.0, 1.0)

    spot_mask = fine_detail * (0.42 + 0.58 * warm_bias)
    return cv2.GaussianBlur(spot_mask, (0, 0), 0.85)


def create_average_face(faces: list[NormalizedFace], image_size: tuple[int, int], preserve_spots: bool = False) -> np.ndarray:
    avg_points = np.mean([face.points for face in faces], axis=0).astype(np.float32)
    rect = (0, 0, image_size[0], image_size[1])
    triangles = calculate_delaunay_triangles(rect, avg_points)
    output = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)
    spot_sum = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    spot_peak = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    for face in faces:
        warped = np.zeros_like(output)
        for i, j, k in triangles:
            warp_triangle(face.image, warped, face.points[[i, j, k]], avg_points[[i, j, k]])
        output += warped

        if preserve_spots:
            spot_mask = extract_spot_mask(warped)
            spot_sum += spot_mask
            spot_peak = np.maximum(spot_peak, spot_mask)

    average = output / len(faces)
    if preserve_spots:
        mean_spot = spot_sum / len(faces)
        reinforced_spot = np.maximum(mean_spot * 1.95, spot_peak * 0.82)
        reinforced_spot = cv2.GaussianBlur(reinforced_spot, (0, 0), 0.70)
        spot_strength = np.clip((reinforced_spot - 0.0055) / 0.040, 0.0, 1.0)
        darken = 0.33 * spot_strength[..., None]
        warm_tint = 0.115 * spot_strength
        average *= 1.0 - darken
        average[..., 0] *= 1.0 - warm_tint * 0.38
        average[..., 1] *= 1.0 - warm_tint * 0.20

    return np.clip(np.rint(average * 255.0), 0, 255).astype(np.uint8)


def generate_average_faces(images_dir: Path, labels_file: Path, output_dir: Path, predictor_path: Path, sheet_name: str | int, image_size: tuple[int, int]) -> None:
    if not predictor_path.exists():
        raise FileNotFoundError(f"未找到 dlib 68 点模型文件：{predictor_path}")
    labels = load_labels(labels_file, sheet_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    boundary = boundary_points(image_size)
    dst_eye_corners = eye_corner_dst(image_size)
    missing_ids: list[str] = []
    failed_ids: list[str] = []
    generated_count = 0

    for subclass, group in labels.groupby("病因大类"):
        detected_faces: list[DetectedFace] = []
        for sample_id in group["ID"]:
            image_path = find_image_path(images_dir, sample_id)
            if image_path is None:
                missing_ids.append(sample_id)
                continue
            face = prepare_detected_face(sample_id, image_path, detector, predictor)
            if face is None:
                failed_ids.append(sample_id)
                continue
            detected_faces.append(face)
        if len(detected_faces) < 2:
            print(f"跳过病因大类 {subclass}：有效图片不足 2 张")
            continue

        normalized_faces = [normalize_face(face, image_size, dst_eye_corners, boundary) for face in detected_faces]
        average_face = create_average_face(
            normalized_faces,
            image_size,
            preserve_spots=should_preserve_spots(str(subclass)),
        )
        output_path = output_dir / f"{sanitize_name(subclass)}_average.jpg"
        Image.fromarray(average_face).save(output_path)
        generated_count += 1

        print(f"已完成病因大类 {subclass}：使用 {len(detected_faces)} 张图片生成 1 张平均脸")

    print(f"总共生成 {generated_count} 张平均脸，输出目录：{output_dir}")
    if missing_ids:
        print("以下 ID 未找到对应图片：")
        for sample_id in missing_ids:
            print(f"  - {sample_id}")
    if failed_ids:
        print("以下 ID 未检测到有效 68 点关键点，已跳过：")
        for sample_id in failed_ids:
            print(f"  - {sample_id}")


def main() -> None:
    args = parse_args()
    generate_average_faces(args.images_dir, args.labels_file, args.output_dir, args.predictor_path, args.sheet_name, tuple(args.image_size))


if __name__ == "__main__":
    main()
