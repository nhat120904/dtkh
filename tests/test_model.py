"""Kiểm thử bộ nhận dạng: chế độ demo và nạp model kiểu Teachable Machine (.h5)."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from core.config import SAMPLES_DIR
from core.imaging import ImageError, assess_quality, load_image, preprocess_teachable
from core.model import (
    DemoClassifier, ModelLoadError, clean_label, create_classifier, find_model_files, install_model_zip,
    model_signature, to_probabilities,
)

LABELS = ["Búa", "Kìm", "Tua vít", "Cờ lê", "Cưa tay", "Giũa", "Nền"]


# --------------------------------------------------------------------------- helpers
def test_clean_label():
    assert clean_label("0 Búa") == "Búa"
    assert clean_label("12  Tua vít ") == "Tua vít"
    assert clean_label("Cờ lê") == "Cờ lê"


def test_to_probabilities():
    assert np.allclose(to_probabilities(np.array([0.2, 0.8])), [0.2, 0.8])
    probs = to_probabilities(np.array([2.0, -1.0, 0.5]))  # logits -> softmax
    assert abs(probs.sum() - 1) < 1e-9 and probs.argmax() == 0


def test_preprocess_matches_teachable_machine():
    arr = preprocess_teachable(Image.new("RGB", (640, 480), (255, 0, 127)), 224)
    assert arr.shape == (1, 224, 224, 3) and arr.dtype == np.float32
    assert np.isclose(arr.max(), 1.0) and np.isclose(arr.min(), -1.0)


def test_load_image_handles_png_alpha_and_garbage():
    buf = io.BytesIO()
    Image.new("RGBA", (50, 40), (0, 0, 0, 0)).save(buf, "PNG")
    img = load_image(buf.getvalue())
    assert img.mode == "RGB" and img.size == (50, 40)
    with pytest.raises(ImageError):
        load_image(b"not an image")


def test_quality_tips():
    assert assess_quality(Image.new("RGB", (300, 300), (10, 10, 10))).tips  # tối + mờ


# --------------------------------------------------------------------------- demo mode
def test_demo_mode_when_no_model(kb, tmp_path):
    clf, status = create_classifier(kb, models_dir=tmp_path)
    assert status.is_demo and isinstance(clf, DemoClassifier)
    assert model_signature(tmp_path) == ("demo",)


@pytest.mark.parametrize("path", sorted(SAMPLES_DIR.glob("*.png")), ids=lambda p: p.stem)
def test_demo_recognises_samples(kb, path):
    clf = DemoClassifier(kb)
    result = clf.predict(Image.open(path).convert("RGB"))
    assert result.best.tool_id == path.stem.rsplit("_", 1)[0]
    assert result.best.confidence > 0.9
    assert abs(sum(p.confidence for p in result.predictions) - 1) < 1e-6


def test_demo_is_unsure_on_other_images(kb):
    clf = DemoClassifier(kb)
    rng = np.random.default_rng(0)
    for _ in range(5):
        img = Image.fromarray((rng.random((240, 320, 3)) * 255).astype("uint8"))
        assert clf.predict(img).best.confidence < 0.5


def test_missing_labels_falls_back_to_demo(kb, tmp_path):
    (tmp_path / "keras_model.h5").write_bytes(b"fake")
    clf, status = create_classifier(kb, models_dir=tmp_path)
    assert status.is_demo and status.error


def test_broken_model_falls_back_to_demo(kb, tmp_path):
    (tmp_path / "keras_model.h5").write_bytes(b"this is not a real h5 file")
    (tmp_path / "labels.txt").write_text("0 Búa\n1 Kìm\n", encoding="utf-8")
    clf, status = create_classifier(kb, models_dir=tmp_path)
    assert status.is_demo and status.error and status.model_path


# --------------------------------------------------------------------------- Teachable Machine model
tf_keras = pytest.importorskip("tf_keras", reason="Cần TensorFlow + tf-keras để kiểm thử model thật")


def _make_teachable_machine_like_model(path, n_classes):
    """Mô phỏng cấu trúc file keras_model.h5 của Teachable Machine:
    Sequential lồng nhau (khối MobileNet có DepthwiseConv2D + đầu phân loại Dense softmax), lưu bằng Keras 2."""
    layers = tf_keras.layers
    tf_keras.utils.set_random_seed(0)
    feature = tf_keras.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),
        layers.Conv2D(8, 3, strides=4, padding="same", activation="relu"),
        layers.DepthwiseConv2D(3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
    ], name="sequential_1")
    head = tf_keras.Sequential([layers.Dense(16, activation="relu"), layers.Dense(n_classes, activation="softmax")],
                               name="sequential_3")
    model = tf_keras.Sequential([feature, head])
    model.build((None, 224, 224, 3))
    model.save(path)
    return model


@pytest.fixture(scope="module")
def tm_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tm_model")
    _make_teachable_machine_like_model(str(d / "keras_model.h5"), len(LABELS))
    (d / "labels.txt").write_text("".join(f"{i} {lb}\n" for i, lb in enumerate(LABELS)), encoding="utf-8")
    return d


def test_h5_contains_groups_quirk(tm_dir):
    """File phải chứa tham số 'groups' của DepthwiseConv2D giống file thật của Teachable Machine."""
    import h5py
    with h5py.File(tm_dir / "keras_model.h5") as f:
        config = f.attrs["model_config"]
    assert '"groups"' in (config.decode() if isinstance(config, bytes) else config)


def test_teachable_machine_model_loads_and_predicts(kb, tm_dir):
    clf, status = create_classifier(kb, models_dir=tm_dir)
    assert status.mode == "model", status.error
    assert clf.labels == LABELS and status.input_size == 224
    result = clf.predict(Image.open(SAMPLES_DIR / "bua_1.png").convert("RGB"))
    assert len(result.predictions) == len(LABELS)
    assert abs(sum(p.confidence for p in result.predictions) - 1) < 1e-4
    by_label = {p.label: p for p in result.predictions}
    assert by_label["Cờ lê"].tool_id == "co_le" and by_label["Nền"].is_background


def test_keras3_loader_handles_groups_quirk(tm_dir):
    """Máy không cài tf-keras: Keras 3 vẫn mở được nhờ bản sao .h5 đã làm sạch tham số 'groups'."""
    from core.model import _load_with_keras3, _sanitized_h5
    with pytest.raises(Exception, match="groups"):
        _load_with_keras3(tm_dir / "keras_model.h5")  # đúng lỗi người dùng hay gặp
    clean = _sanitized_h5(tm_dir / "keras_model.h5")
    run, size = _load_with_keras3(clean)
    out = run(np.zeros((1, size, size, 3), dtype=np.float32))
    assert out.shape == (1, len(LABELS))


def test_both_loaders_agree(tm_dir):
    from core.model import _load_with_keras3, _load_with_tf_keras, _sanitized_h5
    x = preprocess_teachable(Image.open(SAMPLES_DIR / "kim_1.png").convert("RGB"), 224)
    a, _ = _load_with_tf_keras(_sanitized_h5(tm_dir / "keras_model.h5"))
    b, _ = _load_with_keras3(_sanitized_h5(tm_dir / "keras_model.h5"))
    assert np.allclose(a(x), b(x), atol=1e-5)


def test_label_count_mismatch_is_reported(kb, tm_dir, tmp_path):
    (tmp_path / "keras_model.h5").write_bytes((tm_dir / "keras_model.h5").read_bytes())
    (tmp_path / "labels.txt").write_text("0 Búa\n1 Kìm\n", encoding="utf-8")
    clf, status = create_classifier(kb, models_dir=tmp_path)
    assert status.is_demo and "labels.txt" in status.error


def test_install_from_teachable_machine_zip(kb, tm_dir, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(tm_dir / "keras_model.h5", "keras_model.h5")
        zf.write(tm_dir / "labels.txt", "labels.txt")
    (tmp_path / "keras_model.h5").write_bytes(b"old model")  # model cũ phải được sao lưu
    written = install_model_zip(buf.getvalue(), tmp_path)
    assert set(written) == {"keras_model.h5", "labels.txt"}
    assert list(tmp_path.glob("_backup_*/keras_model.h5"))
    assert find_model_files(tmp_path)[0].name == "keras_model.h5"
    _, status = create_classifier(kb, models_dir=tmp_path)
    assert status.mode == "model"


def test_install_rejects_bad_zip(tmp_path):
    with pytest.raises(ModelLoadError):
        install_model_zip(b"nope", tmp_path)
