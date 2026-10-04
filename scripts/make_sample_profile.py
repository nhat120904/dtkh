"""Tạo data/sample_profile.json – hồ sơ học sinh mẫu để xem trước trang Tiến độ.

Dùng đúng các hàm của ứng dụng (DemoClassifier + core.progress) nên dữ liệu luôn khớp định dạng.
Chạy:  python scripts/make_sample_profile.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from core import progress  # noqa: E402
from core.config import SAMPLE_PROFILE_PATH, SAMPLES_DIR  # noqa: E402
from core.gamification import BadgeContext, evaluate_badges  # noqa: E402
from core.imaging import make_thumbnail_b64  # noqa: E402
from core.knowledge import KnowledgeBase  # noqa: E402
from core.model import DemoClassifier  # noqa: E402
from core.quiz import build_quiz, load_bank, score_answer  # noqa: E402


def main() -> int:
    rng = random.Random(8)
    kb = KnowledgeBase.load()
    clf = DemoClassifier(kb)
    profile = progress.new_profile("Học sinh mẫu", "hoc-sinh-mau")
    start = datetime(2026, 9, 14, 8, 30)
    clock = [start]

    def tick(minutes: int) -> str:
        clock[0] += timedelta(minutes=minutes)
        return clock[0].isoformat(timespec="seconds")

    # Lượt nhận dạng: ảnh mẫu (chắc chắn) xen lẫn một ảnh "khó" (chưa chắc)
    samples = sorted(SAMPLES_DIR.glob("*.png"))
    for path in rng.sample(samples, 9):
        image = Image.open(path).convert("RGB")
        result = clf.predict(image)
        tool = kb.get(result.best.tool_id)
        outcome = progress.record_scan(profile, result, source=rng.choice(["upload", "camera", "sample"]),
                                       thumb_b64=make_thumbnail_b64(image), threshold=0.6, top_k=5,
                                       tool_name=tool.name if tool else "")
        outcome.entry["time"] = tick(rng.randint(20, 400))
        if rng.random() < 0.6:
            progress.set_scan_feedback(profile, outcome.entry["id"], True)
    noisy = Image.effect_noise((320, 320), 60).convert("RGB")
    outcome = progress.record_scan(profile, clf.predict(noisy), source="camera", thumb_b64=make_thumbnail_b64(noisy),
                                   threshold=0.6, top_k=5)
    outcome.entry["time"] = tick(90)
    profile["history"].sort(key=lambda h: h["time"], reverse=True)

    for tool_id in ["bua", "co_le", "tua_vit", "kim", "cua_tay", "giua", "e_to"]:
        progress.mark_tool_read(profile, tool_id, kb.get(tool_id).name)

    # Các bài quiz với điểm tăng dần
    bank = load_bank()
    for topic, skill in [("ten", 0.55), ("cong_dung", 0.6), ("an_toan", 0.7), ("all", 0.8), ("an_toan", 1.0)]:
        questions = build_quiz(kb, bank, topic, 10, seed=rng.randrange(10_000))
        streak = best = xp = correct = 0
        for q in questions:
            choice = q.answer if rng.random() < skill else (q.answer + 1) % len(q.options)
            outcome = score_answer(q, choice, streak)
            streak, best = outcome.streak, max(best, outcome.streak)
            xp += outcome.xp
            correct += outcome.correct
            progress.record_answer(profile, outcome.correct, outcome.streak, outcome.xp)
        attempt = progress.record_quiz(profile, topic=topic, total=len(questions), correct=correct, xp_earned=xp,
                                       duration_s=rng.uniform(90, 260), best_streak=best)
        attempt["time"] = tick(rng.randint(600, 2400))

    ctx = BadgeContext(ai_tool_ids=tuple(clf.tool_ids), total_tools=len(kb))
    for badge in evaluate_badges(profile, ctx):
        profile["badges"][badge.id] = clock[0].isoformat(timespec="seconds")
    for entry in profile["xp_log"]:
        entry["time"] = clock[0].isoformat(timespec="seconds")
    profile["created_at"] = start.isoformat(timespec="seconds")
    profile["updated_at"] = clock[0].isoformat(timespec="seconds")

    SAMPLE_PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ {SAMPLE_PROFILE_PATH.relative_to(ROOT)}: {profile['xp']} XP, {len(profile['history'])} lượt nhận dạng, "
          f"{len(profile['quiz_attempts'])} bài quiz, {len(profile['badges'])} huy hiệu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
