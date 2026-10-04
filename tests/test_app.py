"""Kiểm thử giao diện bằng streamlit.testing (chạy app không cần trình duyệt)."""

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

TIMEOUT = 60


def run_view(view_name: str) -> None:
    """Chạy một trang riêng lẻ (AppTest.from_function cần hàm tự chứa)."""
    import importlib

    import streamlit as st

    from ui import state
    from ui.components import inject_css
    from views import guide, home, library, progress_page, quiz, recognize

    modules = {"home": home, "recognize": recognize, "library": library, "quiz": quiz,
               "progress": progress_page, "guide": guide}
    state.PAGES = {k: st.Page(m.render, title=k, url_path=k) for k, m in modules.items()}
    inject_css()
    importlib.import_module(f"views.{view_name}").render()
    state.flush_notifications()


def view_app(view: str) -> AppTest:
    at = AppTest.from_function(run_view, args=(view,), default_timeout=TIMEOUT)
    return at


def by_label(widgets, label: str):
    return next(w for w in widgets if w.label == label)


def test_full_app_home_runs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.session_state["profile_id"]


@pytest.mark.parametrize("view", ["recognize", "library", "quiz", "progress_page", "guide"])
def test_each_page_renders(view):
    at = view_app(view).run()
    assert not at.exception, at.exception


def test_recognize_sample_awards_xp():
    at = view_app("recognize")
    at.session_state["scan_source"] = "sample"
    at.run()
    assert not at.exception
    at.button(key="sample_samples_bua_1").click().run()
    assert not at.exception
    scan = at.session_state["current_scan"]
    assert scan["result"].best.tool_id == "bua"
    profile = at.session_state["profile"]
    assert profile["history"][0]["tool_id"] == "bua" and profile["xp"] >= 10

    # Bấm lại cùng ảnh: không cộng XP lần nữa
    xp = profile["xp"]
    at.button(key="sample_samples_bua_1").click().run()
    assert at.session_state["profile"]["xp"] == xp

    # Phản hồi "Đúng rồi"
    entry_id = scan["entry_id"]
    at.button(key=f"fb_ok_{entry_id}").click().run()
    assert at.session_state["profile"]["history"][0]["feedback"]["correct"] is True


def test_quiz_full_flow():
    at = view_app("quiz").run()
    at.button(key="pick_an_toan").click().run()
    by_label(at.button, "🚀 Bắt đầu!").click().run()
    quiz = at.session_state["quiz"]
    total = len(quiz["questions"])
    assert quiz["stage"] == "play" and total == 10
    assert all(q.topic == "an_toan" for q in quiz["questions"])

    for idx in range(total):
        answer = at.session_state["quiz"]["questions"][idx].answer
        at.button(key=f"q{idx}_opt{answer}").click().run()
        assert not at.exception
        label = "🏁 Xem kết quả" if idx == total - 1 else "Câu tiếp theo ➜"
        by_label(at.button, label).click().run()

    quiz = at.session_state["quiz"]
    assert quiz["stage"] == "done"
    attempt = quiz["attempt"]
    assert attempt["correct"] == total and attempt["bonus"] == 30
    profile = at.session_state["profile"]
    assert {"quiz_first", "quiz_perfect", "safety_star"} <= set(profile["badges"])


def test_progress_with_sample_profile():
    at = view_app("progress_page").run()
    by_label(at.button, "🧪 Nạp hồ sơ mẫu để xem thử").click().run()
    assert not at.exception
    assert at.session_state["profile"]["quiz_attempts"]
