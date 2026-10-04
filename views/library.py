"""Trang Kho kiến thức: tra cứu theo nhóm, xem bài học chi tiết từng dụng cụ."""

from __future__ import annotations

import streamlit as st

from core.gamification import XP_READ_TOOL
from ui import state
from ui.components import empty_state, page_header, show, tool_card, xp_bar
from ui.dialogs import tool_dialog

COLUMNS = 3


def render() -> None:
    kb = state.get_kb()
    profile = state.current_profile()
    ai_ids = set(state.ai_tool_ids())

    show(page_header("02", "Kho kiến thức",
                     f"{len(kb)} dụng cụ cơ khí trong chương trình Công nghệ 8. "
                     f"Mỗi bài học mới hoàn thành được <b>+{XP_READ_TOOL} XP</b>."))

    c1, c2 = st.columns([3, 4], vertical_alignment="bottom")
    query = c1.text_input("🔎 Tìm dụng cụ", placeholder="vd: cắt, vặn vít, kẹp, đo…", key="lib_query")
    groups = {"all": "Tất cả"} | {g.id: f"{g.icon} {g.short}" for g in kb.groups()}
    group = c2.pills("Nhóm dụng cụ", list(groups), format_func=groups.get, default="all", key="lib_group") or "all"

    read = len(profile["read_tools"])
    show(f'<div style="display:flex;gap:.8rem;align-items:center;margin:.6rem 0 1rem"><b style="white-space:nowrap">📖 Đã học {read}/{len(kb)}</b>'
         f'<div style="flex:1">{xp_bar(read / len(kb))}</div></div>')

    tools = kb.search(query)
    if group != "all":
        tools = [t for t in tools if t.group == group]
    st.write("")
    if not tools:
        show(empty_state("🧐", "Không tìm thấy dụng cụ phù hợp", "Thử từ khóa khác, ví dụ “cắt”, “kẹp”, “đo”."))
        return

    for start in range(0, len(tools), COLUMNS):
        cols = st.columns(COLUMNS, gap="medium")
        for col, tool in zip(cols, tools[start:start + COLUMNS]):
            with col:
                show(tool_card(tool, kb.group(tool.group), tool.id in ai_ids, tool.id in profile["read_tools"]))
                label = "Ôn lại bài" if tool.id in profile["read_tools"] else "Học bài này"
                if st.button(f"📖 {label}", key=f"lib_open_{tool.id}", width="stretch"):
                    tool_dialog(tool.id)
        st.write("")
