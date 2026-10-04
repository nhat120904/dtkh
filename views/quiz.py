"""Trang Quiz: chọn chủ đề → trả lời từng câu (có giải thích) → tổng kết điểm, XP."""

from __future__ import annotations

import random
import time

import streamlit as st

from core import progress
from core.gamification import XP_QUIZ_CORRECT, XP_QUIZ_PERFECT, XP_QUIZ_STREAK_BONUS
from core.quiz import STREAK_BONUS_FROM, TOPICS, build_quiz, score_answer
from ui import state
from ui.components import (
    answered_options, chip, esc, explain_box, page_header, question_card, quiz_progress, show,
)

TOPIC_INFO = {
    "all": ("🎲", "var(--yellow)", "Trộn đủ mọi loại câu hỏi."),
    "ten": ("🔤", "var(--blue-2)", "Nhìn hình, đọc mô tả và gọi đúng tên dụng cụ."),
    "cong_dung": ("🎯", "var(--teal-2)", "Dụng cụ dùng để làm gì, thuộc nhóm nào."),
    "an_toan": ("🦺", "var(--orange-2)", "Những quy tắc giúp em an toàn trong xưởng."),
}


# ----------------------------------------------------------------------------- state transitions
def _start(topic: str, count: int) -> None:
    questions = build_quiz(state.get_kb(), state.get_bank(), topic, count, seed=random.randrange(1 << 30))
    st.session_state.quiz = {
        "stage": "play", "topic": topic, "questions": questions, "index": 0,
        "answers": [], "streak": 0, "best_streak": 0, "xp": 0, "started": time.time(),
    }


def _answer(choice: int) -> None:
    quiz = st.session_state.quiz
    if len(quiz["answers"]) > quiz["index"]:
        return  # đã trả lời câu này rồi (bấm đúp)
    question = quiz["questions"][quiz["index"]]
    outcome = score_answer(question, choice, quiz["streak"])
    quiz["answers"].append({"choice": choice, "correct": outcome.correct, "xp": outcome.xp})
    quiz["streak"] = outcome.streak
    quiz["best_streak"] = max(quiz["best_streak"], outcome.streak)
    quiz["xp"] += outcome.xp
    with state.update_profile() as profile:
        progress.record_answer(profile, outcome.correct, outcome.streak, outcome.xp)


def _next() -> None:
    quiz = st.session_state.quiz
    quiz["index"] += 1
    if quiz["index"] >= len(quiz["questions"]):
        correct = sum(a["correct"] for a in quiz["answers"])
        with state.update_profile() as profile:
            quiz["attempt"] = progress.record_quiz(
                profile, topic=quiz["topic"], total=len(quiz["questions"]), correct=correct,
                xp_earned=quiz["xp"], duration_s=time.time() - quiz["started"], best_streak=quiz["best_streak"],
            )
        quiz["stage"] = "done"


def _reset() -> None:
    st.session_state.pop("quiz", None)


# ----------------------------------------------------------------------------- screens
def _pick_topic(topic: str) -> None:
    st.session_state.quiz_topic = topic


def _setup() -> None:
    profile = state.current_profile()
    selected = st.session_state.setdefault("quiz_topic", "all")
    st.session_state.setdefault("quiz_count", 10)

    cols = st.columns(4)
    for col, (key, label) in zip(cols, TOPICS.items()):
        icon, bg, desc = TOPIC_INFO[key]
        best = max((a["score"] for a in profile["quiz_attempts"] if a["topic"] == key), default=None)
        best_txt = chip(f"Kỉ lục {best:g}/10", "teal") if best is not None else chip("Chưa làm")
        is_on = key == selected
        style = "min-height:222px;background:var(--yellow-2);box-shadow:6px 6px 0 var(--ink)" if is_on else "min-height:222px"
        with col:
            show(f'<div class="xw-card" style="{style}"><div class="xw-feature"><div class="ic" style="background:{bg}">{icon}</div></div>'
                 f'<div class="xw-title" style="font-size:1.35rem;margin-top:.7rem">{esc(label)}</div>'
                 f'<div class="xw-muted" style="font-size:.88rem;margin-bottom:.6rem">{esc(desc)}</div>{best_txt}</div>')
            st.button("✓ Đang chọn" if is_on else "Chọn chủ đề", key=f"pick_{key}", width="stretch",
                      type="primary" if is_on else "secondary", on_click=_pick_topic, args=(key,))

    st.write("")
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    count = c1.pills("Số câu hỏi", [5, 10, 15], format_func=lambda n: f"{n} câu", key="quiz_count") or 10
    c2.button("🚀 Bắt đầu!", type="primary", width="stretch", on_click=_start, args=(selected, count))

    st.write("")
    show(f'<div class="xw-banner"><div class="ic">⚡</div><div><b>Cách tính XP</b>'
         f"<p>Mỗi câu đúng +{XP_QUIZ_CORRECT} XP · từ câu đúng thứ {STREAK_BONUS_FROM} liên tiếp thưởng thêm "
         f"+{XP_QUIZ_STREAK_BONUS} XP/câu 🔥 · đúng hết bài (từ 5 câu) thưởng +{XP_QUIZ_PERFECT} XP.</p></div></div>")


def _play() -> None:
    quiz = st.session_state.quiz
    questions = quiz["questions"]
    idx = quiz["index"]
    question = questions[idx]
    answered = len(quiz["answers"]) > idx

    top_l, top_r = st.columns([3, 2], vertical_alignment="center")
    with top_l:
        show(quiz_progress(len(questions), idx, [a["correct"] for a in quiz["answers"]]))
    with top_r:
        flame = f"🔥 x{quiz['streak']}" if quiz["streak"] >= 2 else "🔥 –"
        xp_txt = f"⚡ +{quiz['xp']} XP"
        show(f'<div class="xw-chips" style="justify-content:flex-end">{chip(flame, "orange")}{chip(xp_txt, "yellow")}</div>')

    show(question_card(question.prompt, idx, len(questions), TOPICS[question.topic], question.tool_id, question.show_icon))
    st.write("")

    if not answered:
        with st.container(key="quiz_options"):
            cols = st.columns(2)
            for i, option in enumerate(question.options):
                cols[i % 2].button(f"{'ABCD'[i]}.  {option}", key=f"q{idx}_opt{i}", width="stretch",
                                   on_click=_answer, args=(i,))
    else:
        ans = quiz["answers"][idx]
        show(answered_options(question.options, question.answer, ans["choice"]))
        show(explain_box(ans["correct"], ans["xp"], question.explanation, question.correct_option))
        st.write("")
        last = idx + 1 >= len(questions)
        st.button("🏁 Xem kết quả" if last else "Câu tiếp theo ➜", type="primary", width="stretch", on_click=_next)

    st.write("")
    st.button("✖ Dừng bài quiz", type="tertiary", on_click=_reset)


def _done() -> None:
    quiz = st.session_state.quiz
    attempt = quiz["attempt"]
    total, correct = attempt["total"], attempt["correct"]
    ratio = correct / total if total else 0
    if ratio == 1:
        msg = "Xuất sắc! Em là thợ cả tương lai! 🏆"
    elif ratio >= 0.8:
        msg = "Rất giỏi! Chỉ còn chút xíu nữa thôi 💪"
    elif ratio >= 0.5:
        msg = "Khá lắm! Ôn lại kho kiến thức rồi thử lần nữa nhé 📚"
    else:
        msg = "Đừng nản! Đọc lại bài học rồi làm lại, em sẽ tiến bộ 🌱"

    left, right = st.columns([2, 3], gap="large")
    with left:
        chips = [
            chip(f"✅ {correct}/{total} câu đúng", "teal"),
            chip(f"⚡ +{attempt['xp']} XP", "yellow"),
            chip(f"🔥 Chuỗi {attempt['best_streak']}", "orange"),
            chip(f"⏱ {attempt['duration_s']:.0f}s"),
        ]
        if attempt["bonus"]:
            chips.append(chip(f"🎁 Thưởng +{attempt['bonus']} XP", "orange"))
        show(f'<div class="xw-card xw-score"><div class="xw-kicker">{esc(TOPICS[attempt["topic"]])}</div>'
             f'<div class="big">{attempt["score"]:g}<small>/10</small></div><div class="msg">{esc(msg)}</div>'
             f'<div class="xw-chips" style="justify-content:center;margin-top:.8rem">{"".join(chips)}</div></div>')
        st.write("")
        st.button("🔁 Làm bài mới cùng chủ đề", type="primary", width="stretch",
                  on_click=_start, args=(quiz["topic"], total))
        st.button("🗂️ Chọn chủ đề khác", width="stretch", on_click=_reset)
    with right:
        st.subheader("📝 Xem lại bài làm")
        for i, (q, a) in enumerate(zip(quiz["questions"], quiz["answers"])):
            icon = "✅" if a["correct"] else "❌"
            with st.expander(f"{icon} Câu {i + 1}: {q.prompt.splitlines()[0][:80]}", expanded=not a["correct"]):
                show(answered_options(q.options, q.answer, a["choice"]))
                if q.explanation:
                    st.caption(f"💡 {q.explanation}")


def render() -> None:
    show(page_header("03", "Thử thách Quiz",
                     "Kiểm tra hiểu biết về tên gọi, công dụng và an toàn khi dùng dụng cụ cơ khí."))
    stage = st.session_state.get("quiz", {}).get("stage", "setup")
    if stage == "play":
        _play()
    elif stage == "done":
        _done()
    else:
        _setup()
