"""Luật chơi: điểm XP, cấp độ, huy hiệu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ----------------------------------------------------------------------------- XP
XP_SCAN = 10                 # nhận dạng thành công (AI đủ chắc chắn)
XP_SCAN_NEW_TOOL = 20        # thưởng thêm khi khám phá dụng cụ mới
XP_FEEDBACK = 2              # xác nhận/sửa kết quả nhận dạng
XP_READ_TOOL = 5             # đọc bài của một dụng cụ (1 lần/dụng cụ)
XP_QUIZ_CORRECT = 10         # mỗi câu đúng
XP_QUIZ_STREAK_BONUS = 5     # thưởng thêm từ câu đúng thứ 3 liên tiếp
XP_QUIZ_PERFECT = 30         # đúng hết bài (>= 5 câu)
SCAN_XP_DAILY_LIMIT_PER_TOOL = 5  # chống "cày" XP bằng cách quét mãi một dụng cụ
PERFECT_MIN_QUESTIONS = 5


# ----------------------------------------------------------------------------- Levels
@dataclass(frozen=True)
class Level:
    number: int
    title: str
    icon: str
    min_xp: int


LEVELS = [
    Level(1, "Học việc", "🔩", 0),
    Level(2, "Thợ tập sự", "🔧", 100),
    Level(3, "Thợ phụ", "🪛", 250),
    Level(4, "Thợ lành nghề", "⚙️", 450),
    Level(5, "Thợ cả", "🛠️", 700),
    Level(6, "Kỹ sư nhí", "🏗️", 1000),
    Level(7, "Bậc thầy cơ khí", "🏆", 1400),
]


@dataclass(frozen=True)
class LevelProgress:
    level: Level
    next_level: Level | None
    xp: int

    @property
    def xp_into_level(self) -> int:
        return self.xp - self.level.min_xp

    @property
    def xp_needed(self) -> int:
        return self.next_level.min_xp - self.level.min_xp if self.next_level else 0

    @property
    def xp_to_next(self) -> int:
        return self.next_level.min_xp - self.xp if self.next_level else 0

    @property
    def ratio(self) -> float:
        if not self.next_level:
            return 1.0
        return max(0.0, min(1.0, self.xp_into_level / self.xp_needed))


def level_for(xp: int) -> LevelProgress:
    xp = max(0, int(xp))
    current = LEVELS[0]
    for lvl in LEVELS:
        if xp >= lvl.min_xp:
            current = lvl
    idx = LEVELS.index(current)
    nxt = LEVELS[idx + 1] if idx + 1 < len(LEVELS) else None
    return LevelProgress(current, nxt, xp)


# ----------------------------------------------------------------------------- Badges
@dataclass(frozen=True)
class BadgeContext:
    ai_tool_ids: tuple[str, ...]   # các dụng cụ AI hiện tại nhận dạng được
    total_tools: int               # số dụng cụ trong kho kiến thức


@dataclass(frozen=True)
class Badge:
    id: str
    name: str
    icon: str
    description: str
    check: Callable[[dict, BadgeContext], bool]


def _perfect(attempt: dict) -> bool:
    return attempt["total"] >= PERFECT_MIN_QUESTIONS and attempt["correct"] == attempt["total"]


BADGES: list[Badge] = [
    Badge("first_scan", "Mắt thần nhí", "🔍", "Nhận dạng thành công dụng cụ đầu tiên",
          lambda p, c: p["stats"]["scans_confident"] >= 1),
    Badge("collector", "Nhà sưu tầm", "🧰", "Khám phá đủ tất cả dụng cụ mà AI nhận dạng được",
          lambda p, c: bool(c.ai_tool_ids) and set(c.ai_tool_ids) <= set(p["discovered"])),
    Badge("scanner_20", "Chuyên gia quét", "📸", "Nhận dạng thành công 20 lần",
          lambda p, c: p["stats"]["scans_confident"] >= 20),
    Badge("reader_5", "Mọt sách xưởng", "📖", "Đọc 5 bài trong kho kiến thức",
          lambda p, c: len(p["read_tools"]) >= 5),
    Badge("reader_all", "Bách khoa cơ khí", "🎓", "Đọc hết toàn bộ kho kiến thức",
          lambda p, c: c.total_tools > 0 and len(p["read_tools"]) >= c.total_tools),
    Badge("quiz_first", "Khởi động", "🚀", "Hoàn thành bài quiz đầu tiên",
          lambda p, c: len(p["quiz_attempts"]) >= 1),
    Badge("quiz_perfect", "Điểm 10 tròn trĩnh", "💯", "Đúng tất cả các câu của một bài quiz (từ 5 câu)",
          lambda p, c: any(_perfect(a) for a in p["quiz_attempts"])),
    Badge("safety_star", "Ngôi sao an toàn", "🦺", "Đạt 100% bài quiz chủ đề An toàn",
          lambda p, c: any(_perfect(a) and a["topic"] == "an_toan" for a in p["quiz_attempts"])),
    Badge("streak_5", "Chuỗi lửa", "🔥", "Trả lời đúng 5 câu liên tiếp",
          lambda p, c: p["best_streak"] >= 5),
    Badge("feedback_5", "Thầy giáo của AI", "🧑‍🏫", "Xác nhận hoặc sửa kết quả nhận dạng 5 lần",
          lambda p, c: sum(1 for h in p["history"] if h.get("feedback")) >= 5),
    Badge("level_5", "Thợ cả", "🛠️", "Đạt cấp 5",
          lambda p, c: level_for(p["xp"]).level.number >= 5),
]
BADGES_BY_ID = {b.id: b for b in BADGES}


def evaluate_badges(profile: dict, ctx: BadgeContext) -> list[Badge]:
    """Trả về các huy hiệu vừa đạt (chưa có trong hồ sơ)."""
    owned = profile.get("badges", {})
    return [b for b in BADGES if b.id not in owned and b.check(profile, ctx)]
