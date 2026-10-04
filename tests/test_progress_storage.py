import json

from core import progress
from core.gamification import BadgeContext, evaluate_badges, level_for
from core.model import Prediction, PredictionResult
from core.storage import ProfileStore, slugify


def _result(tool_id, label, conf, mode="model"):
    preds = [Prediction(label, conf, tool_id), Prediction("Khác", 1 - conf, None)]
    return PredictionResult(preds, mode, 5.0)


def test_level_progression():
    assert level_for(0).level.number == 1
    assert level_for(99).level.number == 1
    lp = level_for(175)
    assert lp.level.number == 2 and lp.xp_to_next == 75 and 0 < lp.ratio < 1
    top = level_for(99999)
    assert top.next_level is None and top.ratio == 1.0


def test_confident_scan_awards_xp_and_discovery():
    p = progress.new_profile("An", "an")
    out = progress.record_scan(p, _result("bua", "Búa", 0.9), source="upload", thumb_b64=None, threshold=0.6, top_k=3)
    assert out.confident and out.new_tool and out.xp == 30
    assert "bua" in p["discovered"] and p["xp"] == 30
    again = progress.record_scan(p, _result("bua", "Búa", 0.9), source="upload", thumb_b64=None, threshold=0.6, top_k=3)
    assert again.xp == 10 and not again.new_tool


def test_unsure_scan_gives_no_xp():
    p = progress.new_profile("An", "an")
    out = progress.record_scan(p, _result("bua", "Búa", 0.4), source="camera", thumb_b64=None, threshold=0.6, top_k=3)
    assert not out.confident and out.xp == 0 and out.note
    assert p["history"][0]["tool_id"] is None and p["stats"]["scans_total"] == 1


def test_background_is_never_confident():
    p = progress.new_profile("An", "an")
    result = PredictionResult([Prediction("Nền", 0.99, None, is_background=True)], "model", 1.0)
    assert not progress.record_scan(p, result, source="upload", thumb_b64=None, threshold=0.6, top_k=3).confident


def test_daily_xp_limit_per_tool():
    p = progress.new_profile("An", "an")
    xps = [progress.record_scan(p, _result("kim", "Kìm", 0.95), source="upload", thumb_b64=None,
                                threshold=0.6, top_k=3).xp for _ in range(7)]
    assert xps == [30, 10, 10, 10, 10, 0, 0]


def test_feedback_xp_only_once():
    p = progress.new_profile("An", "an")
    entry = progress.record_scan(p, _result("kim", "Kìm", 0.95), source="upload", thumb_b64=None,
                                 threshold=0.6, top_k=3).entry
    assert progress.set_scan_feedback(p, entry["id"], False, "co_le") == 2
    assert progress.set_scan_feedback(p, entry["id"], True) == 0
    assert p["history"][0]["feedback"]["correct"] is True


def test_read_tool_once():
    p = progress.new_profile("An", "an")
    assert progress.mark_tool_read(p, "bua") == 5
    assert progress.mark_tool_read(p, "bua") == 0


def test_quiz_record_and_badges():
    p = progress.new_profile("An", "an")
    for i in range(5):
        progress.record_answer(p, True, i + 1, 10)
    attempt = progress.record_quiz(p, topic="an_toan", total=5, correct=5, xp_earned=50, duration_s=60, best_streak=5)
    assert attempt["bonus"] == 30 and attempt["score"] == 10.0
    earned = {b.id for b in evaluate_badges(p, BadgeContext(("bua",), 15))}
    assert {"quiz_first", "quiz_perfect", "safety_star", "streak_5"} <= earned
    assert "collector" not in earned


def test_store_roundtrip(tmp_path):
    store = ProfileStore(tmp_path)
    a = store.create("Minh Anh 8A2")
    b = store.create("Minh Anh 8A2")
    assert a["id"] == "minh-anh-8a2" and b["id"] == "minh-anh-8a2-2"
    progress.add_xp(a, 42, "test")
    store.save(a)
    loaded = store.load(a["id"])
    assert loaded["xp"] == 42 and loaded["name"] == "Minh Anh 8A2"
    assert [p["id"] for p in store.list()][0] in (a["id"], b["id"])
    store.delete(b["id"])
    assert not store.exists(b["id"])


def test_store_handles_corrupt_and_old_files(tmp_path):
    store = ProfileStore(tmp_path)
    (tmp_path / "hong.json").write_text("{not json", encoding="utf-8")
    assert store.load("hong") is None
    assert list(tmp_path.glob("hong.corrupt-*"))
    (tmp_path / "cu.json").write_text(json.dumps({"name": "Cũ", "xp": 7}), encoding="utf-8")
    old = store.load("cu")
    assert old["xp"] == 7 and old["history"] == [] and old["stats"]["scans_total"] == 0


def test_slugify():
    assert slugify("Nguyễn Văn Đức") == "nguyen-van-duc"
    assert slugify("!!!") == "hoc-sinh"


def test_sample_profile_is_valid():
    from core.config import SAMPLE_PROFILE_PATH
    data = progress.ensure_schema(json.loads(SAMPLE_PROFILE_PATH.read_text(encoding="utf-8")))
    assert data["history"] and data["quiz_attempts"] and data["xp"] > 0
