"""Bộ nhận dạng dụng cụ.

* TeachableMachineClassifier – chạy model Google Teachable Machine (Keras .h5 / .keras / SavedModel).
* DemoClassifier            – dùng khi CHƯA có model: so khớp ảnh với ảnh mẫu bằng OpenCV.
                              Đây KHÔNG phải AI thật, chỉ để học sinh làm quen với ứng dụng.

create_classifier() tự chọn bộ phù hợp và trả về trạng thái để giao diện hiển thị.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from core.config import DEFAULT_AI_TOOLS, MODELS_DIR, SAMPLES_DIR, TM_INPUT_SIZE, env_path
from core.imaging import color_histogram, dhash, hamming, histogram_similarity, preprocess_teachable
from core.knowledge import KnowledgeBase


class ModelLoadError(RuntimeError):
    """Không nạp được model (thiếu TensorFlow, file hỏng, nhãn không khớp…)."""


@dataclass
class Prediction:
    label: str            # nhãn gốc của model
    confidence: float     # 0..1
    tool_id: str | None   # id dụng cụ trong kho kiến thức (None nếu không khớp)
    is_background: bool = False


@dataclass
class PredictionResult:
    predictions: list[Prediction]   # đã sắp xếp giảm dần theo độ tin cậy
    mode: str                       # "model" | "demo"
    elapsed_ms: float
    note: str = ""

    @property
    def best(self) -> Prediction:
        return self.predictions[0]

    def top(self, k: int) -> list[Prediction]:
        return self.predictions[:k]


@dataclass
class ModelStatus:
    mode: str                       # "model" | "demo"
    title: str
    message: str
    model_path: str | None = None
    labels: list[str] = field(default_factory=list)
    error: str | None = None
    input_size: int = TM_INPUT_SIZE

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"


class Classifier(Protocol):
    mode: str
    labels: list[str]

    def predict(self, image: Image.Image) -> PredictionResult: ...


# --------------------------------------------------------------------------- helpers
def clean_label(line: str) -> str:
    """'0 Búa' -> 'Búa' (Teachable Machine ghi số thứ tự trước tên lớp)."""
    return re.sub(r"^\s*\d+\s+", "", line).strip()


def load_labels(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    labels = [clean_label(line) for line in lines if line.strip()]
    if not labels:
        raise ModelLoadError(f"File nhãn {Path(path).name} đang trống.")
    return labels


def to_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.all(scores >= 0) and np.all(scores <= 1.0001) and abs(scores.sum() - 1) < 0.02:
        return scores / scores.sum()
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def _make_predictions(labels: list[str], probs: np.ndarray, kb: KnowledgeBase) -> list[Prediction]:
    preds = []
    for label, p in zip(labels, probs):
        tool = kb.match_label(label)
        preds.append(Prediction(label=label, confidence=float(p), tool_id=tool.id if tool else None,
                                is_background=kb.is_background_label(label)))
    preds.sort(key=lambda p: p.confidence, reverse=True)
    return preds


# --------------------------------------------------------------------------- locating model files
def find_model_files(models_dir: Path = MODELS_DIR) -> tuple[Path, Path | None] | None:
    """Tìm (file model, file nhãn). Trả về None nếu chưa có model."""
    models_dir = Path(models_dir)
    candidates: list[Path] = []
    if (env_model := env_path("XUONG_MODEL_PATH")) is not None:
        candidates.append(env_model)
    candidates += [models_dir / "keras_model.h5", models_dir / "model.keras", models_dir / "model.savedmodel"]
    if models_dir.exists():
        candidates += sorted(models_dir.glob("*.h5")) + sorted(models_dir.glob("*.keras"))
        candidates += sorted(p.parent for p in models_dir.glob("*/saved_model.pb") if not p.parent.name.startswith("_"))

    model = next((p for p in candidates
                  if (p.is_file() and p.suffix in (".h5", ".keras")) or (p / "saved_model.pb").is_file()), None)
    if model is None:
        return None

    label_candidates = [env_path("XUONG_LABELS_PATH"), model.parent / "labels.txt", models_dir / "labels.txt"]
    if model.is_dir():
        label_candidates.insert(1, model / "labels.txt")
    labels = next((p for p in label_candidates if p is not None and p.is_file()), None)
    return model, labels


def model_signature(models_dir: Path = MODELS_DIR) -> tuple:
    """Dấu vân tay của file model hiện tại – đổi khi file model/nhãn thay đổi (để nạp lại cache)."""
    found = find_model_files(models_dir)
    if not found:
        return ("demo",)
    sig = []
    for path in found:
        if path is None:
            sig.append(None)
        elif path.is_dir():
            sig.append((str(path), max((f.stat().st_mtime for f in path.rglob("*") if f.is_file()), default=0)))
        else:
            st = path.stat()
            sig.append((str(path), st.st_mtime, st.st_size))
    return tuple(sig)


# --------------------------------------------------------------------------- loading Keras / TF models
def _input_size(model) -> int:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    try:
        size = int(shape[1])
        return size if size > 0 else TM_INPUT_SIZE
    except (TypeError, ValueError, IndexError):
        return TM_INPUT_SIZE


def _compat_objects(keras_module) -> dict:
    """File .h5 của Teachable Machine lưu tham số 'groups' cho DepthwiseConv2D,
    Keras mới không nhận tham số này -> bỏ đi khi nạp."""
    base = keras_module.layers.DepthwiseConv2D

    class DepthwiseConv2DCompat(base):
        def __init__(self, *args, groups=None, **kwargs):  # noqa: ARG002
            super().__init__(*args, **kwargs)

    return {"DepthwiseConv2D": DepthwiseConv2DCompat}


def _strip_depthwise_groups(node) -> bool:
    """Xóa khóa 'groups' trong cấu hình DepthwiseConv2D (đệ quy). Trả về True nếu có sửa."""
    changed = False
    if isinstance(node, dict):
        if node.get("class_name") == "DepthwiseConv2D" and isinstance(node.get("config"), dict):
            changed = node["config"].pop("groups", None) is not None
        for value in node.values():
            changed = _strip_depthwise_groups(value) or changed
    elif isinstance(node, list):
        for value in node:
            changed = _strip_depthwise_groups(value) or changed
    return changed


def _sanitized_h5(path: Path) -> Path | None:
    """Tạo bản sao tạm của file .h5 đã bỏ tham số 'groups' – cách chắc chắn nhất để
    Keras 3 mở được file Teachable Machine (Keras 3 có thể bỏ qua custom_objects)."""
    import h5py

    with h5py.File(path, "r") as fh:
        raw = fh.attrs.get("model_config")
    if raw is None:
        return None
    config = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    if not _strip_depthwise_groups(config):
        return None
    copy = Path(tempfile.mkdtemp(prefix="xuong-model-")) / path.name
    shutil.copyfile(path, copy)
    with h5py.File(copy, "r+") as fh:
        fh.attrs["model_config"] = json.dumps(config)
    return copy


def _load_with_tf_keras(path: Path):
    import tf_keras  # Keras 2 – tương thích tốt nhất với file của Teachable Machine

    model = tf_keras.models.load_model(str(path), compile=False, custom_objects=_compat_objects(tf_keras))
    return (lambda x: np.asarray(model(x, training=False))), _input_size(model)


def _load_with_keras3(path: Path):
    import keras

    if path.is_dir():
        layer = keras.layers.TFSMLayer(str(path), call_endpoint="serving_default")

        def run(x):
            out = layer(x)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            return np.asarray(out)

        return run, TM_INPUT_SIZE
    model = keras.models.load_model(str(path), compile=False, custom_objects=_compat_objects(keras))
    return (lambda x: np.asarray(model(x, training=False))), _input_size(model)


def _load_with_tf_signature(path: Path):
    import tensorflow as tf

    if not path.is_dir():
        raise ModelLoadError("Không phải thư mục SavedModel")
    loaded = tf.saved_model.load(str(path))
    signature = loaded.signatures["serving_default"]
    _, kwargs = signature.structured_input_signature
    input_name, spec = next(iter(kwargs.items()))
    size = int(spec.shape[1]) if spec.shape.rank == 4 and spec.shape[1] else TM_INPUT_SIZE

    def run(x):
        out = signature(**{input_name: tf.constant(x)})
        return np.asarray(next(iter(out.values())))

    return run, size


def load_model_fn(path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    try:
        import tensorflow  # noqa: F401
    except Exception as exc:  # ImportError hoặc lỗi thư viện hệ thống
        raise ModelLoadError(
            "Chưa cài được TensorFlow nên chưa chạy được model. "
            "Hãy chạy: pip install -r requirements.txt"
        ) from exc

    loaders = [_load_with_tf_keras, _load_with_keras3]
    if path.is_dir():
        loaders.append(_load_with_tf_signature)
    errors = []
    sanitized = None
    if path.suffix == ".h5":
        try:
            sanitized = _sanitized_h5(path)
        except Exception as exc:  # file hỏng: để các bộ nạp báo lỗi chi tiết
            errors.append(f"• h5: {type(exc).__name__}: {str(exc)[:200]}")
    try:
        for loader in loaders:
            try:
                return loader(sanitized or path)
            except ImportError:
                continue
            except Exception as exc:  # thử cách nạp tiếp theo
                errors.append(f"• {loader.__name__.removeprefix('_load_with_')}: {type(exc).__name__}: {str(exc)[:300]}")
    finally:
        if sanitized is not None:
            shutil.rmtree(sanitized.parent, ignore_errors=True)
    raise ModelLoadError("Không mở được file model:\n" + "\n".join(errors))


# --------------------------------------------------------------------------- classifiers
class TeachableMachineClassifier:
    mode = "model"

    def __init__(self, model_path: Path, labels_path: Path, kb: KnowledgeBase):
        self.kb = kb
        self.model_path = Path(model_path)
        self.labels = load_labels(labels_path)
        self._run, self.input_size = load_model_fn(self.model_path)
        self._lock = threading.Lock()
        # Chạy thử một lần: phát hiện sớm lỗi và làm "nóng" model
        probe = self._raw_predict(np.zeros((1, self.input_size, self.input_size, 3), dtype=np.float32))
        if probe.size != len(self.labels):
            raise ModelLoadError(
                f"Model có {probe.size} lớp nhưng labels.txt có {len(self.labels)} nhãn. "
                "Hãy dùng đúng cặp keras_model.h5 + labels.txt tải cùng lúc từ Teachable Machine."
            )

    def _raw_predict(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            return np.asarray(self._run(batch)).reshape(-1)

    def predict(self, image: Image.Image) -> PredictionResult:
        start = time.perf_counter()
        probs = to_probabilities(self._raw_predict(preprocess_teachable(image, self.input_size)))
        elapsed = (time.perf_counter() - start) * 1000
        return PredictionResult(_make_predictions(self.labels, probs, self.kb), self.mode, elapsed)


class DemoClassifier:
    """So khớp ảnh với bộ ảnh mẫu bằng đặc trưng OpenCV (dHash + histogram màu).

    * Ảnh mẫu (hoặc gần giống) -> trả kết quả đúng với độ tin cậy cao.
    * Ảnh khác -> độ tin cậy luôn thấp (< 50%) để giao diện nhắc "chưa chắc chắn".
    """

    mode = "demo"
    MAX_UNSURE_CONFIDENCE = 0.48

    def __init__(self, kb: KnowledgeBase, samples_dir: Path = SAMPLES_DIR, tool_ids: list[str] | None = None):
        self.kb = kb
        self.tool_ids = [t for t in (tool_ids or DEFAULT_AI_TOOLS) if kb.get(t)]
        self.labels = [kb.get(t).name for t in self.tool_ids]
        self._refs: list[tuple[int, int, np.ndarray]] = []  # (tool index, dhash, histogram)
        for idx, tool_id in enumerate(self.tool_ids):
            for path in sorted(Path(samples_dir).glob(f"{tool_id}_*.png")):
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    self._refs.append((idx, dhash(rgb), color_histogram(rgb)))

    def predict(self, image: Image.Image) -> PredictionResult:
        start = time.perf_counter()
        n = len(self.tool_ids)
        h, hist = dhash(image), color_histogram(image)
        sim = np.zeros(n)
        best_dist = np.full(n, 64)
        for idx, ref_hash, ref_hist in self._refs:
            dist = hamming(h, ref_hash)
            score = 0.7 * (1 - dist / 64) + 0.3 * histogram_similarity(hist, ref_hist)
            sim[idx] = max(sim[idx], score)
            best_dist[idx] = min(best_dist[idx], dist)

        winner = int(np.argmin(best_dist)) if self._refs else 0
        if self._refs and best_dist[winner] <= 8:
            # Gần như trùng ảnh mẫu
            top = 0.97 - best_dist[winner] * 0.01
            rest = np.delete(sim, winner)
            rest = np.exp(rest * 6) / np.exp(rest * 6).sum() * (1 - top) if rest.size else rest
            probs = np.insert(rest, winner, top)
            note = "Đây là ảnh minh họa trong ứng dụng."
        else:
            probs = np.exp(sim * 8)
            probs = probs / probs.sum()
            if probs.max() > self.MAX_UNSURE_CONFIDENCE:
                # Trộn với phân phối đều để độ tin cậy lớn nhất không vượt ngưỡng
                a = (self.MAX_UNSURE_CONFIDENCE - 1 / n) / (probs.max() - 1 / n)
                probs = a * probs + (1 - a) / n
            note = "Ảnh này chưa khớp với các ảnh minh họa. Hãy chọn thẻ “Ảnh mẫu” để thử."
        elapsed = (time.perf_counter() - start) * 1000
        return PredictionResult(_make_predictions(self.labels, probs, self.kb), self.mode, elapsed, note)


def create_classifier(kb: KnowledgeBase, models_dir: Path = MODELS_DIR) -> tuple[Classifier, ModelStatus]:
    found = find_model_files(models_dir)
    if found is None:
        return DemoClassifier(kb), ModelStatus(
            mode="demo",
            title="AI đang ở chế độ trải nghiệm",
            message="Kết quả nhận dạng dùng ảnh minh họa. Các bài học và trò chơi vẫn hoạt động bình thường.",
            labels=[kb.get(t).name for t in DEFAULT_AI_TOOLS],
        )
    model_path, labels_path = found
    if labels_path is None:
        return DemoClassifier(kb), ModelStatus(
            mode="demo", title="AI đang ở chế độ trải nghiệm",
            message="Kết quả nhận dạng dùng ảnh minh họa. Các bài học và trò chơi vẫn hoạt động bình thường.",
            model_path=str(model_path), error="Hãy đặt labels.txt (tải cùng model) vào thư mục models/.",
        )
    try:
        clf = TeachableMachineClassifier(model_path, labels_path, kb)
    except ModelLoadError as exc:
        return DemoClassifier(kb), ModelStatus(
            mode="demo", title="AI đang ở chế độ trải nghiệm",
            message="Kết quả nhận dạng dùng ảnh minh họa. Các bài học và trò chơi vẫn hoạt động bình thường.",
            model_path=str(model_path), error=str(exc),
        )
    unmatched = [lb for lb in clf.labels if not kb.match_label(lb) and not kb.is_background_label(lb)]
    message = "Chọn “Nhận dạng” để bắt đầu khám phá dụng cụ."
    if unmatched:
        message = "AI đã sẵn sàng. Một số tên dụng cụ có thể chưa khớp với bài học."
    return clf, ModelStatus(mode="model", title="AI đã sẵn sàng", message=message,
                            model_path=str(model_path), labels=clf.labels, input_size=clf.input_size)


def load_model_info(model_path: str | Path | None) -> dict | None:
    """Đọc model_info.json (do scripts/train_model.py tạo) nằm cạnh file model, nếu có."""
    if not model_path:
        return None
    path = Path(model_path)
    info_path = (path if path.is_dir() else path.parent) / "model_info.json"
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- installing models from the UI
def _backup_existing(models_dir: Path) -> Path | None:
    existing = [p for p in models_dir.iterdir()
                if p.name in ("keras_model.h5", "model.keras", "labels.txt", "model.savedmodel", "model_info.json")
                or p.suffix in (".h5", ".keras")]
    if not existing:
        return None
    backup = models_dir / f"_backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir()
    for p in existing:
        shutil.move(str(p), backup / p.name)
    return backup


def install_model_zip(data: bytes, models_dir: Path = MODELS_DIR) -> list[str]:
    """Cài model từ file .zip tải về từ Teachable Machine (Keras hoặc SavedModel).
    Model cũ (nếu có) được chuyển vào thư mục models/_backup_*/ chứ không bị xóa."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ModelLoadError("File tải lên không phải file .zip hợp lệ.") from exc
    names = [n for n in archive.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
    labels = [n for n in names if Path(n).name == "labels.txt"]
    keras_files = [n for n in names if Path(n).suffix in (".h5", ".keras")]
    savedmodel = [n for n in names if Path(n).name == "saved_model.pb"]
    if not labels or not (keras_files or savedmodel):
        raise ModelLoadError("File zip cần có labels.txt và keras_model.h5 (hoặc thư mục model.savedmodel).")

    models_dir.mkdir(parents=True, exist_ok=True)
    _backup_existing(models_dir)
    written = []
    (models_dir / "labels.txt").write_bytes(archive.read(labels[0]))
    written.append("labels.txt")
    if keras_files:
        target = "keras_model.h5" if keras_files[0].endswith(".h5") else "model.keras"
        (models_dir / target).write_bytes(archive.read(keras_files[0]))
        written.append(target)
    else:
        root = str(Path(savedmodel[0]).parent)
        prefix = "" if root == "." else root + "/"
        dest = models_dir / "model.savedmodel"
        for name in names:
            if name.startswith(prefix) and Path(name).name != "labels.txt":
                rel = Path(name[len(prefix):])
                if ".." in rel.parts or rel.is_absolute():
                    continue  # chống ghi file ra ngoài thư mục
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(archive.read(name))
        written.append("model.savedmodel/")
    return written


def install_model_files(model_bytes: bytes, model_filename: str, labels_bytes: bytes,
                        models_dir: Path = MODELS_DIR) -> list[str]:
    suffix = Path(model_filename).suffix.lower()
    if suffix not in (".h5", ".keras"):
        raise ModelLoadError("File model phải có đuôi .h5 hoặc .keras.")
    models_dir.mkdir(parents=True, exist_ok=True)
    _backup_existing(models_dir)
    target = "keras_model.h5" if suffix == ".h5" else "model.keras"
    (models_dir / target).write_bytes(model_bytes)
    (models_dir / "labels.txt").write_bytes(labels_bytes)
    return [target, "labels.txt"]
