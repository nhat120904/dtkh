"""Hộp thoại bài học chi tiết của một dụng cụ (dùng ở Kho kiến thức và trang Nhận dạng)."""

from __future__ import annotations

import streamlit as st

from core import progress
from core.gamification import XP_READ_TOOL
from ui import state
from ui.components import show, tool_detail_head, tool_detail_sections


def _finish_reading(tool_id: str, tool_name: str) -> None:
    with state.update_profile() as profile:
        progress.mark_tool_read(profile, tool_id, tool_name)


@st.dialog("📖 Bài học dụng cụ", width="large")
def tool_dialog(tool_id: str) -> None:
    kb = state.get_kb()
    tool = kb.get(tool_id)
    if tool is None:
        st.error("Không tìm thấy dụng cụ này trong kho kiến thức.")
        return
    group = kb.group(tool.group)
    show(tool_detail_head(tool, group, tool.id in state.ai_tool_ids()))
    show(tool_detail_sections(tool))

    st.write("")
    if tool.id in state.current_profile()["read_tools"]:
        st.success("✅ Em đã hoàn thành bài học này. Ôn lại bất cứ lúc nào nhé!")
        if st.button("Đóng", width="stretch"):
            st.rerun()
    elif st.button(f"✅ Em đã đọc và hiểu bài (+{XP_READ_TOOL} XP)", type="primary", width="stretch"):
        _finish_reading(tool.id, tool.name)
        st.rerun()
