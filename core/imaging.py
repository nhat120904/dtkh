"""Tiện ích xử lý ảnh (Pillow + OpenCV + NumPy)."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from core.config import THUMB_SIZE


class ImageError(ValueError):
    """Ảnh không đọc được."""


def load_image(data: bytes) -> Image.Image:
    """Đọc ảnh từ bytes, xoay đúng chiều theo EXIF (ảnh điện thoại) và chuyển về RGB."""
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("Không đọc được ảnh. Hãy dùng ảnh JPG hoặc PNG.") from exc


def image_digest(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_bytes(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def make_thumbnail_b64(image: Image.Image, size: int = THUMB_SIZE) -> str:
    thumb = ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    return base64.b64encode(pil_to_bytes(thumb, quality=72)).decode("ascii")


def b64_to_data_uri(b64: str, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{b64}"


def preprocess_teachable(image: Image.Image, size: int) -> np.ndarray:
    """Tiền xử lý giống hệt code mẫu của Teachable Machine:
    cắt giữa + resize về size x size, chuẩn hóa về [-1, 1], thêm chiều batch."""
    fitted = ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(fitted, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr[np.newaxis, ...]


# --------------------------------------------------------------------------- OpenCV features
def _gray(image: Image.Image, size: int | None = None) -> np.ndarray:
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash 64-bit – dùng để nhận ra ảnh gần giống nhau."""
    gray = cv2.resize(_gray(image), (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = gray[:, 1:] > gray[:, :-1]
    return int("".join("1" if v else "0" for v in diff.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def color_histogram(image: Image.Image) -> np.ndarray:
    hsv = cv2.cvtColor(np.asarray(image.convert("RGB").resize((128, 128))), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Tương quan histogram, đưa về [0, 1]."""
    return float((cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) + 1) / 2)


@dataclass
class QualityReport:
    brightness: float
    sharpness: float
    tips: list[str]


def assess_quality(image: Image.Image) -> QualityReport:
    """Đánh giá nhanh chất lượng ảnh bằng OpenCV để gợi ý chụp lại."""
    gray = _gray(image, 256)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blown_out = float((gray >= 250).mean())  # tỉ lệ điểm ảnh bị cháy sáng
    tips = []
    if brightness < 60:
        tips.append("Ảnh hơi tối – hãy bật thêm đèn hoặc ra chỗ sáng hơn.")
    elif blown_out > 0.35 and brightness > 225:
        tips.append("Ảnh bị chói, cháy sáng – tránh chụp ngược sáng hoặc dưới đèn quá gắt.")
    if sharpness < 60:
        tips.append("Ảnh bị mờ – giữ chắc tay và lấy nét vào dụng cụ.")
    return QualityReport(brightness, sharpness, tips)
