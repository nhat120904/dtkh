"""Huấn luyện model nhận dạng dụng cụ ngay trên máy (thay cho Teachable Machine).

Cách làm giống Teachable Machine: học chuyển giao từ MobileNetV2 (đã học trên ImageNet),
ảnh 224x224 chuẩn hóa về [-1, 1]. Kết quả tương thích 100% với ứng dụng.

Dữ liệu:  dataset/<thư mục lớp>/*.jpg
  * tên thư mục là id dụng cụ trong kho kiến thức ("bua", "kim", "tua_vit"…) hoặc "nen" (lớp nền);
  * cũng có thể đặt tên tự do, vd "Ê tô" – tên thư mục được dùng làm nhãn.
  Muốn AI nhận tốt dụng cụ của lớp mình: chụp thêm ảnh thật bỏ vào đúng thư mục rồi chạy lại.

Kết quả:  models/keras_model.h5, models/labels.txt, models/model_info.json (độ chính xác, ma trận nhầm lẫn…)

Chạy:     python scripts/train_model.py            (lần đầu cần mạng để tải trọng số MobileNetV2 ~14 MB)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tf_keras  # noqa: E402
from PIL import Image  # noqa: E402

from core.config import MODELS_DIR, SAMPLES_DIR  # noqa: E402
from core.knowledge import KnowledgeBase  # noqa: E402

IMG = 224
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Thứ tự lớp ưu tiên (giữ ổn định giữa các lần huấn luyện)
PREFERRED_ORDER = ["bua", "kim", "tua_vit", "co_le", "mo_let", "cua_tay", "giua", "nen"]


def class_label(folder: str, kb: KnowledgeBase) -> str:
    if folder == "nen":
        return "Nền"
    tool = kb.get(folder)
    return tool.name if tool else folder


def list_dataset(root: Path) -> dict[str, list[Path]]:
    classes = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if files:
            classes[d.name] = files
    order = [c for c in PREFERRED_ORDER if c in classes] + sorted(c for c in classes if c not in PREFERRED_ORDER)
    return {c: classes[c] for c in order}


def split(data: dict[str, list[Path]], val: float, test: float, seed: int):
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    for idx, files in enumerate(data.values()):
        files = files[:]
        rng.shuffle(files)
        n_test, n_val = max(1, int(len(files) * test)), max(1, int(len(files) * val))
        parts["test"] += [(f, idx) for f in files[:n_test]]
        parts["val"] += [(f, idx) for f in files[n_test:n_test + n_val]]
        parts["train"] += [(f, idx) for f in files[n_test + n_val:]]
    return parts


# ----------------------------------------------------------------------------- tf.data pipeline
def _load(path, label, n_classes):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    return img, tf.one_hot(label, n_classes)


def _center_fit(img):
    """Cắt hình vuông ở giữa rồi thu về 224 – giống ImageOps.fit của ứng dụng."""
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    side = tf.minimum(h, w)
    img = tf.image.crop_to_bounding_box(img, (h - side) // 2, (w - side) // 2, side, side)
    return tf.image.resize(img, (IMG, IMG), method="area")


def _augment(img):
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    side = tf.cast(tf.cast(tf.minimum(h, w), tf.float32) * tf.random.uniform([], 0.6, 1.0), tf.int32)
    img = tf.image.random_crop(img, tf.stack([side, side, 3]))
    img = tf.image.resize(img, (IMG, IMG), method="area")
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))  # dụng cụ có thể nằm theo mọi hướng
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 40.0)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.6, 1.4)
    img = tf.image.random_hue(img, 0.04)
    # Mô phỏng webcam độ phân giải thấp / hơi mờ
    if tf.random.uniform([]) < 0.25:
        small = tf.random.uniform([], 64, 140, dtype=tf.int32)
        img = tf.image.resize(tf.image.resize(img, tf.stack([small, small])), (IMG, IMG))
    return tf.clip_by_value(img, 0.0, 255.0)


def make_dataset(items, n_classes, training: bool, batch: int):
    paths = [str(p) for p, _ in items]
    labels = [lab for _, lab in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(items), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _load(p, y, n_classes), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: ((_augment(x) if training else _center_fit(x)) / 127.5 - 1.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------- model
def build_model(n_classes: int):
    base = tf_keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf_keras.Input((IMG, IMG, 3), name="image")
    x = base(inputs, training=False)  # giữ BatchNorm ở chế độ suy luận khi tinh chỉnh
    x = tf_keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = tf_keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf_keras.layers.Dense(n_classes, activation="softmax", name="probs",
                                    kernel_regularizer=tf_keras.regularizers.l2(1e-4))(x)
    return tf_keras.Model(inputs, outputs, name="xuong_co_khi_classifier"), base


def export_flat(model, n_classes: int):
    """Dựng lại model dạng "phẳng" (không lồng MobileNetV2 như một lớp con) và chép trọng số.
    File .h5 dạng này mở được bằng cả tf-keras lẫn Keras 3."""
    base = tf_keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights=None)
    x = tf_keras.layers.GlobalAveragePooling2D(name="pool")(base.output)
    x = tf_keras.layers.Dropout(0.3, name="dropout")(x)
    out = tf_keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    flat = tf_keras.Model(base.input, out, name="xuong_co_khi_classifier")
    flat.set_weights(model.get_weights())
    probe = np.random.default_rng(0).uniform(-1, 1, (2, IMG, IMG, 3)).astype("float32")
    if not np.allclose(model(probe, training=False), flat(probe, training=False), atol=1e-5):
        raise RuntimeError("Xuất model thất bại: kết quả model phẳng khác model gốc")
    return flat


def class_weights(items, n_classes):
    counts = np.bincount([lab for _, lab in items], minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


# ----------------------------------------------------------------------------- evaluation through the app
def evaluate_with_app(model_dir: Path, test_items, labels: list[str], threshold: float) -> dict:
    """Đánh giá bằng ĐÚNG đường xử lý của ứng dụng (core.model + PIL), không phải tf.data."""
    from core.model import create_classifier

    kb = KnowledgeBase.load()
    clf, status = create_classifier(kb, models_dir=model_dir)
    if status.mode != "model":
        raise RuntimeError(f"Ứng dụng không mở được model vừa huấn luyện: {status.error}")
    n = len(labels)
    confusion = np.zeros((n, n), dtype=int)
    confident = confident_correct = 0
    for path, true in test_items:
        with Image.open(path) as im:
            result = clf.predict(im.convert("RGB"))
        pred = labels.index(result.best.label)
        confusion[true, pred] += 1
        if result.best.confidence >= threshold:
            confident += 1
            confident_correct += int(pred == true)
    total = int(confusion.sum())
    per_class = {}
    for i, lab in enumerate(labels):
        tp = confusion[i, i]
        per_class[lab] = {
            "precision": round(float(tp / max(confusion[:, i].sum(), 1)), 3),
            "recall": round(float(tp / max(confusion[i].sum(), 1)), 3),
            "support": int(confusion[i].sum()),
        }
    return {
        "accuracy": round(float(np.trace(confusion) / max(total, 1)), 4),
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "threshold": threshold,
        "confident_share": round(confident / max(total, 1), 4),
        "confident_accuracy": round(confident_correct / max(confident, 1), 4),
        "test_images": total,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(MODELS_DIR))
    ap.add_argument("--epochs-head", type=int, default=14)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--finetune-layers", type=int, default=40, help="số lớp cuối của MobileNetV2 được tinh chỉnh")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-samples", action="store_true", help="không trộn ảnh mẫu minh họa vào tập huấn luyện")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf_keras.utils.set_random_seed(args.seed)
    kb = KnowledgeBase.load()
    data = list_dataset(Path(args.data))
    if len(data) < 2:
        print(f"Cần ít nhất 2 thư mục lớp có ảnh trong {args.data}")
        return 1
    folders = list(data)
    labels = [class_label(f, kb) for f in folders]
    n = len(labels)
    parts = split(data, val=0.15, test=0.15, seed=args.seed)
    if not args.no_samples:  # ảnh minh họa của ứng dụng -> để tab "Ảnh mẫu" cũng nhận đúng
        for f in SAMPLES_DIR.glob("*.png"):
            tool_id = f.stem.rsplit("_", 1)[0]
            if tool_id in folders:
                parts["train"] += [(f, folders.index(tool_id))] * 3
    for name, items in parts.items():
        counts = np.bincount([lab for _, lab in items], minlength=n)
        print(f"{name:5}: {len(items):5} ảnh  " + "  ".join(f"{labels[i]}={c}" for i, c in enumerate(counts)))

    train = make_dataset(parts["train"], n, True, args.batch)
    val = make_dataset(parts["val"], n, False, args.batch)
    model, base = build_model(n)
    loss = tf_keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    weights = class_weights(parts["train"], n)
    stop = tf_keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
    started = time.time()

    print("\n▶ Giai đoạn 1: huấn luyện phần phân loại (MobileNetV2 đóng băng)")
    model.compile(tf_keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs_head, class_weight=weights, callbacks=[stop], verbose=2)

    print(f"\n▶ Giai đoạn 2: tinh chỉnh {args.finetune_layers} lớp cuối của MobileNetV2")
    base.trainable = True
    for layer in base.layers[:-args.finetune_layers]:
        layer.trainable = False
    model.compile(tf_keras.optimizers.Adam(2e-5), loss=loss, metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs_finetune, class_weight=weights,
              callbacks=[tf_keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)],
              verbose=2)
    _, val_acc = model.evaluate(val, verbose=0)

    # Lưu vào thư mục tạm, đánh giá bằng đúng code của ứng dụng rồi mới chép vào models/
    tmp = Path(tempfile.mkdtemp(prefix="xuong-train-"))
    export_flat(model, n).save(tmp / "keras_model.h5", include_optimizer=False)
    (tmp / "labels.txt").write_text("".join(f"{i} {lab}\n" for i, lab in enumerate(labels)), encoding="utf-8")
    print("\n▶ Đánh giá trên tập kiểm tra (ảnh chưa từng dùng để huấn luyện), qua đường xử lý của ứng dụng…")
    report = evaluate_with_app(tmp, parts["test"], labels, threshold=0.6)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("keras_model.h5", "labels.txt"):
        if (out / name).exists():
            backup = out / f"_backup_{datetime.now():%Y%m%d_%H%M%S}"
            backup.mkdir(exist_ok=True)
            shutil.move(str(out / name), backup / name)
        shutil.move(str(tmp / name), out / name)
    shutil.rmtree(tmp, ignore_errors=True)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "base_model": "MobileNetV2 (ImageNet) + học chuyển giao",
        "input_size": IMG,
        "labels": labels,
        "folders": folders,
        "images": {s: len(items) for s, items in parts.items()},
        "per_class_images": {labels[i]: len(files) for i, files in enumerate(data.values())},
        "val_accuracy": round(float(val_acc), 4),
        "test": report,
        "train_minutes": round((time.time() - started) / 60, 1),
    }
    (out / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n✓ Đã lưu model vào {out}/keras_model.h5 ({(out / 'keras_model.h5').stat().st_size / 1e6:.1f} MB)")
    print(f"  Độ chính xác kiểm tra: {report['accuracy'] * 100:.1f}% trên {report['test_images']} ảnh")
    print(f"  Khi AI chắc chắn (≥60%): chiếm {report['confident_share'] * 100:.0f}% số ảnh, "
          f"đúng {report['confident_accuracy'] * 100:.1f}%")
    for lab, m in report["per_class"].items():
        print(f"   {lab:10} precision={m['precision']:.2f} recall={m['recall']:.2f} (n={m['support']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
