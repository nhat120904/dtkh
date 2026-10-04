"""Ngân hàng câu hỏi + sinh câu hỏi tự động từ kho kiến thức."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

from core.config import QUIZ_BANK_PATH
from core.gamification import XP_QUIZ_CORRECT, XP_QUIZ_STREAK_BONUS
from core.knowledge import KnowledgeBase, Tool

TOPICS = {
    "all": "Tổng hợp",
    "ten": "Tên dụng cụ",
    "cong_dung": "Công dụng",
    "an_toan": "An toàn",
}
STREAK_BONUS_FROM = 3

# Các cặp dụng cụ có công dụng gần giống nhau – không dùng làm phương án nhiễu của nhau
_CONFUSABLE = [{"co_le", "mo_let"}]


@dataclass(frozen=True)
class Question:
    id: str
    topic: str
    prompt: str
    options: tuple[str, ...]
    answer: int
    explanation: str = ""
    tool_id: str | None = None
    show_icon: bool = False   # câu “nhìn hình đoán tên”: hiện hình lớn của dụng cụ
    source: str = "bank"

    @property
    def correct_option(self) -> str:
        return self.options[self.answer]

    def shuffled(self, rng: random.Random) -> "Question":
        order = list(range(len(self.options)))
        rng.shuffle(order)
        return replace(self, options=tuple(self.options[i] for i in order), answer=order.index(self.answer))


def load_bank(path: Path = QUIZ_BANK_PATH) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for q in raw["questions"]:
        if q["topic"] not in TOPICS or not 0 <= q["answer"] < len(q["options"]):
            raise ValueError(f"Câu hỏi {q.get('id')} không hợp lệ")
        questions.append(Question(
            id=q["id"], topic=q["topic"], prompt=q["question"], options=tuple(q["options"]),
            answer=q["answer"], explanation=q.get("explanation", ""), tool_id=q.get("tool"),
        ))
    return questions


def _confusable(a: str, b: str) -> bool:
    return any({a, b} <= pair for pair in _CONFUSABLE)


def _distractors(tool: Tool, kb: KnowledgeBase, rng: random.Random, k: int = 3,
                 prefer_same_group: bool = False, other_groups_only: bool = False) -> list[Tool]:
    others = [t for t in kb.all() if t.id != tool.id and not _confusable(t.id, tool.id)]
    if other_groups_only:
        others = [t for t in others if t.group != tool.group]
    rng.shuffle(others)
    if prefer_same_group:
        others.sort(key=lambda t: t.group != tool.group)
    return others[:k]


def generate_questions(kb: KnowledgeBase, rng: random.Random) -> list[Question]:
    """Sinh câu hỏi từ dữ liệu dụng cụ: nhìn hình đoán tên, công dụng, nhóm, lưu ý an toàn."""
    groups = kb.groups()
    out: list[Question] = []
    for tool in kb.all():
        group = kb.group(tool.group)

        names = [tool.name] + [t.name for t in _distractors(tool, kb, rng, prefer_same_group=True)]
        out.append(Question(
            id=f"auto-icon-{tool.id}", topic="ten", prompt="Nhìn hình và cho biết: đây là dụng cụ gì?",
            options=tuple(names), answer=0, tool_id=tool.id, show_icon=True, source="auto",
            explanation=f"Đây là {tool.name.lower()} ({tool.english}) – {tool.summary}.",
        ))

        names = [tool.name] + [t.name for t in _distractors(tool, kb, rng)]
        out.append(Question(
            id=f"auto-name-{tool.id}", topic="ten", prompt=f"Dụng cụ nào {tool.summary}?",
            options=tuple(names), answer=0, tool_id=tool.id, source="auto",
            explanation=f"{tool.name} {tool.summary}.",
        ))

        uses = [tool.summary_sentence] + [t.summary_sentence for t in _distractors(tool, kb, rng, other_groups_only=True)]
        out.append(Question(
            id=f"auto-use-{tool.id}", topic="cong_dung", prompt=f"{tool.name} được dùng để làm gì?",
            options=tuple(uses), answer=0, tool_id=tool.id, source="auto",
            explanation=f"{tool.name} {tool.summary}. {tool.description}",
        ))

        group_names = [group.name] + [g.name for g in groups if g.id != group.id]
        out.append(Question(
            id=f"auto-group-{tool.id}", topic="cong_dung", prompt=f"{tool.name} thuộc nhóm dụng cụ nào?",
            options=tuple(group_names), answer=0, tool_id=tool.id, source="auto",
            explanation=f"{tool.name} thuộc nhóm {group.name.lower()}: {group.description[:1].lower()}{group.description[1:]}",
        ))

        if tool.key_safety:
            names = [tool.name] + [t.name for t in _distractors(tool, kb, rng)]
            out.append(Question(
                id=f"auto-safety-{tool.id}", topic="an_toan",
                prompt=f"Lưu ý an toàn sau đây dành cho dụng cụ nào?\n\n“{tool.key_safety}”",
                options=tuple(names), answer=0, tool_id=tool.id, source="auto",
                explanation=f"Đây là lưu ý khi dùng {tool.name.lower()}. " + " ".join(f"• {s}." for s in tool.safety[:2]),
            ))
    return out


def build_quiz(kb: KnowledgeBase, bank: list[Question], topic: str = "all", n: int = 10,
               seed: int | None = None) -> list[Question]:
    """Chọn n câu (trộn câu soạn sẵn và câu tự sinh), xáo trộn thứ tự đáp án."""
    if topic not in TOPICS:
        raise ValueError(f"Chủ đề không hợp lệ: {topic}")
    rng = random.Random(seed)
    pool = {q.id: q for q in bank + generate_questions(kb, rng)}
    candidates = [q for q in pool.values() if topic == "all" or q.topic == topic]
    rng.shuffle(candidates)
    # Ưu tiên câu soạn sẵn (có giải thích kĩ) chiếm khoảng một nửa bài
    bank_qs = [q for q in candidates if q.source == "bank"]
    auto_qs = [q for q in candidates if q.source == "auto"]
    half = min(len(bank_qs), (n + 1) // 2)
    chosen = bank_qs[:half] + auto_qs[: n - half]
    if len(chosen) < n:
        rest = [q for q in candidates if q not in chosen]
        chosen += rest[: n - len(chosen)]
    rng.shuffle(chosen)
    return [q.shuffled(rng) for q in chosen]


@dataclass(frozen=True)
class AnswerOutcome:
    correct: bool
    xp: int
    streak: int


def score_answer(question: Question, choice: int, streak: int) -> AnswerOutcome:
    if choice == question.answer:
        streak += 1
        bonus = XP_QUIZ_STREAK_BONUS if streak >= STREAK_BONUS_FROM else 0
        return AnswerOutcome(True, XP_QUIZ_CORRECT + bonus, streak)
    return AnswerOutcome(False, 0, 0)
