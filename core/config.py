"""Cấu hình chung: đường dẫn, hằng số, ngưỡng mặc định.

Có thể ghi đè một số giá trị bằng biến môi trường:
  XUONG_MODEL_PATH    – đường dẫn tới file model (.h5/.keras) hoặc thư mục SavedModel
  XUONG_LABELS_PATH   – đường dẫn tới labels.txt
  XUONG_PROFILES_DIR  – thư mục lưu hồ sơ học sinh (JSON)
  XUONG_CONFIDENCE_THRESHOLD – ngưỡng tin cậy trong khoảng 0.0–1.0 (mặc định 0.60)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
SAMPLES_DIR = ASSETS_DIR / "samples"
SAMPLES_REAL_DIR = ASSETS_DIR / "samples_real"  # ảnh chụp thật (giấy phép tự do) để thử nhận dạng
MODELS_DIR = ROOT_DIR / "models"

TOOLS_PATH = DATA_DIR / "tools.json"
QUIZ_BANK_PATH = DATA_DIR / "quiz_bank.json"
SAMPLE_PROFILE_PATH = DATA_DIR / "sample_profile.json"
CSS_PATH = ASSETS_DIR / "style.css"

PROFILES_DIR = Path(os.environ.get("XUONG_PROFILES_DIR", ROOT_DIR / "user_data" / "profiles"))

APP_NAME = "AI Xưởng Cơ Khí"
APP_SUBJECT = "Công nghệ 8"

# Các lớp mà model Teachable Machine mặc định cần nhận dạng (cũng là các lớp của chế độ demo)
DEFAULT_AI_TOOLS = ["bua", "kim", "tua_vit", "co_le", "cua_tay", "giua", "thuoc_la"]

# Nhận dạng
def _confidence_threshold() -> float:
    value = os.environ.get("XUONG_CONFIDENCE_THRESHOLD", "0.60").strip()
    try:
        threshold = float(value)
    except ValueError:
        return 0.60
    return threshold if 0.0 <= threshold <= 1.0 else 0.60


DEFAULT_CONFIDENCE_THRESHOLD = _confidence_threshold()
DEFAULT_TOP_K = 3
TM_INPUT_SIZE = 224  # Teachable Machine (MobileNet) dùng ảnh 224x224

# Lưu trữ
HISTORY_LIMIT = 80
XP_LOG_LIMIT = 200
THUMB_SIZE = 128


def env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None
