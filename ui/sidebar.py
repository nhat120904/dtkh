"""Thanh bên: hồ sơ học sinh và trạng thái nhận dạng."""

from __future__ import annotations

import streamlit as st

from core.gamification import level_for
from core.storage import clean_name
from ui import state
from ui.components import banner, brand, player_card, show


def _on_profile_change() -> None:
    state.switch_profile(st.session_state["profile_select"])


def _create_profile() -> None:
    name = clean_name(st.session_state.get("new_profile_name", ""))
    if not name:
        state.notify("Hãy nhập tên trước khi tạo hồ sơ", "✏️")
        return
    profile = state.get_store().create(name)
    state.switch_profile(profile["id"])
    st.session_state["new_profile_name"] = ""
    state.notify(f"Chào mừng {name} tới xưởng!", "👋")


def render_sidebar() -> None:
    profile = state.current_profile()
    store = state.get_store()

    with st.sidebar:
        show(brand())
        show(player_card(profile["name"], level_for(profile["xp"])))

        profiles = store.list()
        names = {p["id"]: p["name"] for p in profiles}
        names.setdefault(profile["id"], profile["name"])
        st.session_state["profile_select"] = profile["id"]
        st.selectbox("👤 Hồ sơ học sinh", list(names), format_func=lambda pid: names.get(pid, pid),
                     key="profile_select", on_change=_on_profile_change)
        with st.popover("➕ Thêm học sinh", width="stretch"):
            st.text_input("Tên hoặc biệt danh", key="new_profile_name", max_chars=40,
                          placeholder="vd: Minh Anh 8A2")
            st.button("Tạo hồ sơ", type="primary", on_click=_create_profile, width="stretch")

        st.write("")
        _, status = state.get_classifier()
        if status.is_demo:
            show(banner("demo", "🧪", "AI đang ở chế độ trải nghiệm", "Thử nhận dạng bằng ảnh minh họa."))
        else:
            show(banner("ok", "🤖", "AI đã sẵn sàng", "Có thể bắt đầu nhận dạng dụng cụ."))
