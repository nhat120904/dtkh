"""Quản lý trạng thái phiên Streamlit: tài nguyên dùng chung, hồ sơ hiện tại, thông báo XP/huy hiệu."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

from core.config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_TOP_K
from core.gamification import BadgeContext, evaluate_badges, level_for
from core.knowledge import KnowledgeBase
from core.model import Classifier, ModelStatus, create_classifier, model_signature
from core.progress import now_iso
from core.quiz import Question, load_bank
from core.storage import ProfileStore

DEFAULT_PROFILE_NAME = "Học sinh"

# Các trang (st.Page) – app.py gán vào để các trang tạo liên kết qua lại
PAGES: dict = {}


def page(name: str):
    return PAGES.get(name)


# ----------------------------------------------------------------------------- shared resources
@st.cache_resource
def get_kb() -> KnowledgeBase:
    return KnowledgeBase.load()


@st.cache_resource
def get_bank() -> list[Question]:
    return load_bank()


@st.cache_resource
def get_store() -> ProfileStore:
    return ProfileStore()


@st.cache_resource(show_spinner="🤖 Đang khởi động AI… (lần đầu có thể mất một lúc)")
def _load_classifier(signature: tuple) -> tuple[Classifier, ModelStatus]:  # noqa: ARG001 – khóa cache
    return create_classifier(get_kb())


def get_classifier() -> tuple[Classifier, ModelStatus]:
    """Bộ nhận dạng dùng chung; tự nạp lại khi file model trong models/ thay đổi."""
    return _load_classifier(model_signature())


def reload_classifier() -> None:
    _load_classifier.clear()


def ai_tool_ids() -> tuple[str, ...]:
    """Các dụng cụ trong kho kiến thức mà bộ nhận dạng hiện tại có lớp tương ứng."""
    kb = get_kb()
    clf, _ = get_classifier()
    ids = [t.id for t in (kb.match_label(label) for label in clf.labels) if t]
    return tuple(dict.fromkeys(ids))


def badge_context() -> BadgeContext:
    return BadgeContext(ai_tool_ids=ai_tool_ids(), total_tools=len(get_kb()))


# ----------------------------------------------------------------------------- settings
def threshold() -> float:
    return DEFAULT_CONFIDENCE_THRESHOLD


def top_k() -> int:
    return DEFAULT_TOP_K


# ----------------------------------------------------------------------------- profile
def current_profile() -> dict:
    store = get_store()
    profile = st.session_state.get("profile")
    wanted = st.session_state.get("profile_id")
    if profile is not None and profile["id"] == wanted:
        return profile
    profile = store.load(wanted) if wanted else None
    if profile is None:
        existing = store.list()
        profile = store.load(existing[0]["id"]) if existing else None
        profile = profile or store.create(DEFAULT_PROFILE_NAME)
    st.session_state.profile = profile
    st.session_state.profile_id = profile["id"]
    return profile


def switch_profile(profile_id: str) -> None:
    st.session_state.profile_id = profile_id
    st.session_state.pop("profile", None)
    for key in ("scan_cache", "quiz", "live_last"):
        st.session_state.pop(key, None)


@contextmanager
def update_profile() -> Iterator[dict]:
    """Sửa hồ sơ rồi tự động: xét huy hiệu, lưu file, báo XP / lên cấp / huy hiệu mới.

        with state.update_profile() as profile:
            progress.mark_tool_read(profile, "bua")
    """
    profile = current_profile()
    xp_before = profile["xp"]
    level_before = level_for(xp_before).level.number
    yield profile
    new_badges = evaluate_badges(profile, badge_context())
    for badge in new_badges:
        profile["badges"][badge.id] = now_iso()
    get_store().save(profile)

    gained = profile["xp"] - xp_before
    if gained > 0:
        notify(f"+{gained} XP", "⚡")
    for badge in new_badges:
        notify(f"{badge.icon} Huy hiệu mới: {badge.name}!", "🏅")
        st.session_state.setdefault("new_badges", set()).add(badge.id)
    level_after = level_for(profile["xp"]).level
    if level_after.number > level_before:
        notify(f"{level_after.icon} Lên cấp {level_after.number}: {level_after.title}!", "🎉")
    if new_badges or level_after.number > level_before:
        st.session_state["_celebrate"] = True


# ----------------------------------------------------------------------------- notifications
def notify(message: str, icon: str = "✨") -> None:
    st.session_state.setdefault("_toasts", []).append((message, icon))


def flush_notifications() -> None:
    """Hiện các thông báo đang chờ (gọi ở đầu và cuối mỗi lần chạy app)."""
    for message, icon in st.session_state.pop("_toasts", []):
        st.toast(message, icon=icon)
    if st.session_state.pop("_celebrate", False):
        st.balloons()
