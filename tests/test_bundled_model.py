"""Kiểm thử model đi kèm ứng dụng (models/keras_model.h5) – bỏ qua nếu chưa có model."""

import pytest
from PIL import Image

from core.config import MODELS_DIR, SAMPLES_DIR, SAMPLES_REAL_DIR
from core.model import create_classifier, find_model_files, load_model_info

pytestmark = pytest.mark.skipif(find_model_files(MODELS_DIR) is None, reason="Chưa có model trong models/")


@pytest.fixture(scope="module")
def bundled(kb):
    clf, status = create_classifier(kb, models_dir=MODELS_DIR)
    assert status.mode == "model", status.error
    return clf


def test_every_label_maps_to_knowledge_base(kb, bundled):
    for label in bundled.labels:
        assert kb.match_label(label) or kb.is_background_label(label), label


@pytest.mark.parametrize("path", sorted(SAMPLES_DIR.glob("*.png")), ids=lambda p: p.stem)
def test_recognises_app_samples(bundled, path):
    result = bundled.predict(Image.open(path).convert("RGB"))
    assert result.best.tool_id == path.stem.rsplit("_", 1)[0]


def test_model_info_reports_good_accuracy():
    info = load_model_info(find_model_files(MODELS_DIR)[0])
    assert info is not None, "thiếu models/model_info.json"
    assert info["test"]["accuracy"] >= 0.85


@pytest.mark.parametrize("path", sorted(SAMPLES_REAL_DIR.glob("*.jpg")), ids=lambda p: p.stem)
def test_recognises_real_photo_samples(bundled, path):
    """Ảnh chụp thật (không dùng khi huấn luyện) phải được nhận đúng và đủ chắc chắn."""
    result = bundled.predict(Image.open(path).convert("RGB"))
    assert result.best.tool_id == path.stem.rsplit("_", 1)[0]
    assert result.best.confidence >= 0.6
