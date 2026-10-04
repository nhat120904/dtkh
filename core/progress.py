"""Thao tác trên hồ sơ học sinh (dict JSON): cộng XP, lịch sử nhận dạng, kết quả quiz.

Các hàm ở đây là hàm thuần (không phụ thuộc Streamlit) để dễ kiểm thử.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from core.config import HISTORY_LIMIT, XP_LOG_LIMIT
from core.gamification import (
    PERFECT_MIN_QUESTIONS,
    SCAN_XP_DAILY_LIMIT_PER_TOOL,
    XP_FEEDBACK,
    XP_QUIZ_PERFECT,
    XP_READ_TOOL,
    XP_SCAN,
    XP_SCAN_NEW_TOOL,
)
from core.model import PredictionResult

SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_profile(name: str, profile_id: str) -> dict:
    ts = now_iso()
    return {
        "version": SCHEMA_VERSION,
        "id": profile_id,
        "name": name,
        "created_at": ts,
        "updated_at": ts,
        "xp": 0,
        "discovered": {},        # tool_id -> thời điểm khám phá lần đầu
        "read_tools": [],        # các bài đã đọc trong kho kiến thức
        "history": [],           # lịch sử nhận dạng (mới nhất trước)
        "quiz_attempts": [],     # kết quả các bài quiz
        "badges": {},            # badge_id -> thời điểm đạt
        "best_streak": 0,
        "xp_log": [],
        "stats": {"scans_total": 0, "scans_confident": 0, "quiz_questions": 0, "quiz_correct": 0},
    }


def ensure_schema(profile: dict) -> dict:
    """Bổ sung các trường còn thiếu (hồ sơ cũ / do người dùng sửa tay)."""
    base = new_profile(profile.get("name", "Học sinh"), profile.get("id", "hoc-sinh"))
    for key, value in base.items():
        profile.setdefault(key, value)
    for key, value in base["stats"].items():
        profile["stats"].setdefault(key, value)
    profile["xp"] = max(0, int(profile.get("xp", 0)))
    return profile


def add_xp(profile: dict, amount: int, reason: str) -> int:
    if amount <= 0:
        return 0
    profile["xp"] += amount
    profile["xp_log"].insert(0, {"time": now_iso(), "amount": amount, "reason": reason})
    del profile["xp_log"][XP_LOG_LIMIT:]
    return amount


# ----------------------------------------------------------------------------- scans
@dataclass
class ScanOutcome:
    entry: dict
    xp: int
    confident: bool
    new_tool: bool
    note: str = ""


def is_confident(result: PredictionResult, threshold: float) -> bool:
    best = result.best
    return bool(best.tool_id) and not best.is_background and best.confidence >= threshold


def record_scan(profile: dict, result: PredictionResult, *, source: str, thumb_b64: str | None,
                threshold: float, top_k: int, tool_name: str = "") -> ScanOutcome:
    best = result.best
    confident = is_confident(result, threshold)
    new_tool = confident and best.tool_id not in profile["discovered"]
    xp, note = 0, ""
    if confident:
        today = datetime.now().date().isoformat()
        earned_today = sum(1 for h in profile["history"]
                           if h.get("tool_id") == best.tool_id and h.get("xp", 0) > 0 and h["time"].startswith(today))
        if earned_today >= SCAN_XP_DAILY_LIMIT_PER_TOOL:
            note = "Hôm nay em đã nhận đủ XP cho dụng cụ này – hãy thử dụng cụ khác nhé!"
        else:
            xp = XP_SCAN + (XP_SCAN_NEW_TOOL if new_tool else 0)
        if new_tool:
            profile["discovered"][best.tool_id] = now_iso()
    else:
        note = "AI chưa đủ chắc chắn nên chưa cộng XP."

    entry = {
        "id": uuid.uuid4().hex[:10],
        "time": now_iso(),
        "tool_id": best.tool_id if confident else None,
        "label": best.label,
        "confidence": round(best.confidence, 4),
        "confident": confident,
        "mode": result.mode,
        "source": source,
        "top": [{"label": p.label, "tool_id": p.tool_id, "confidence": round(p.confidence, 4)}
                for p in result.top(top_k)],
        "thumb": thumb_b64,
        "xp": xp,
        "feedback": None,
    }
    profile["history"].insert(0, entry)
    del profile["history"][HISTORY_LIMIT:]
    profile["stats"]["scans_total"] += 1
    profile["stats"]["scans_confident"] += int(confident)
    reason = f"Nhận dạng {tool_name or best.label}" + (" (dụng cụ mới!)" if new_tool else "")
    add_xp(profile, xp, reason)
    return ScanOutcome(entry, xp, confident, new_tool, note)


def find_scan(profile: dict, entry_id: str) -> dict | None:
    return next((h for h in profile["history"] if h["id"] == entry_id), None)


def set_scan_feedback(profile: dict, entry_id: str, correct: bool, corrected_tool_id: str | None = None) -> int:
    """Học sinh xác nhận kết quả đúng/sai. Lần đầu phản hồi được +XP."""
    entry = find_scan(profile, entry_id)
    if entry is None:
        return 0
    first_time = entry.get("feedback") is None
    entry["feedback"] = {"correct": bool(correct), "tool_id": None if correct else corrected_tool_id,
                         "time": now_iso()}
    return add_xp(profile, XP_FEEDBACK, "Phản hồi kết quả nhận dạng") if first_time else 0


def clear_history(profile: dict) -> None:
    profile["history"] = []


# ----------------------------------------------------------------------------- knowledge
def mark_tool_read(profile: dict, tool_id: str, tool_name: str = "") -> int:
    if tool_id in profile["read_tools"]:
        return 0
    profile["read_tools"].append(tool_id)
    return add_xp(profile, XP_READ_TOOL, f"Đọc bài: {tool_name or tool_id}")


# ----------------------------------------------------------------------------- quiz
def record_answer(profile: dict, correct: bool, streak: int, xp: int) -> None:
    profile["stats"]["quiz_questions"] += 1
    profile["stats"]["quiz_correct"] += int(correct)
    profile["best_streak"] = max(profile["best_streak"], streak)
    add_xp(profile, xp, "Trả lời đúng câu hỏi quiz")


def record_quiz(profile: dict, *, topic: str, total: int, correct: int, xp_earned: int,
                duration_s: float, best_streak: int) -> dict:
    """Lưu kết quả bài quiz. Trả về bản ghi (kèm XP thưởng nếu đúng hết)."""
    bonus = XP_QUIZ_PERFECT if total >= PERFECT_MIN_QUESTIONS and correct == total else 0
    add_xp(profile, bonus, "Thưởng bài quiz đúng 100%")
    attempt = {
        "time": now_iso(), "topic": topic, "total": total, "correct": correct,
        "score": round(10 * correct / total, 1) if total else 0.0,
        "xp": xp_earned + bonus, "bonus": bonus, "duration_s": round(duration_s, 1), "best_streak": best_streak,
    }
    profile["quiz_attempts"].append(attempt)
    return attempt


# ----------------------------------------------------------------------------- summary
def summarize(profile: dict) -> dict:
    stats = profile["stats"]
    attempts = profile["quiz_attempts"]
    accuracy = stats["quiz_correct"] / stats["quiz_questions"] if stats["quiz_questions"] else None
    return {
        "xp": profile["xp"],
        "scans_total": stats["scans_total"],
        "scans_confident": stats["scans_confident"],
        "discovered": len(profile["discovered"]),
        "read": len(profile["read_tools"]),
        "quizzes": len(attempts),
        "accuracy": accuracy,
        "best_score": max((a["score"] for a in attempts), default=None),
        "badges": len(profile["badges"]),
        "best_streak": profile["best_streak"],
    }
