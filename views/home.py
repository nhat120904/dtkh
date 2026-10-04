"""Trang chủ: lời chào, thống kê nhanh, dụng cụ của ngày, giới thiệu ba nhóm dụng cụ."""

from __future__ import annotations

from datetime import date

import streamlit as st

from core import progress
from core.gamification import level_for
from ui import state
from ui.components import (
    banner, feature, groups_overview, hero, show, stat_tiles, tool_of_the_day,
)
from ui.dialogs import tool_dialog


def render() -> None:
    kb = state.get_kb()
    profile = state.current_profile()
    summary = progress.summarize(profile)
    lp = level_for(profile["xp"])
    ai_ids = state.ai_tool_ids()

    show(hero(profile["name"], kb))
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.page_link(state.page("recognize"), label="Nhận dạng dụng cụ ngay", icon="📷", width="stretch")
    c2.page_link(state.page("library"), label="Mở kho kiến thức", icon="📚", width="stretch")
    c3.page_link(state.page("quiz"), label="Thử thách quiz", icon="🧠", width="stretch")
    st.write("")

    accuracy = f"{summary['accuracy'] * 100:.0f}<small>%</small>" if summary["accuracy"] is not None else "–"
    show(stat_tiles([
        ("⚡", f"{summary['xp']}", "Điểm kinh nghiệm (XP)"),
        (lp.level.icon, f"{lp.level.number}<small> · {lp.level.title}</small>", "Cấp độ"),
        ("🔍", f"{len(set(profile['discovered']) & set(ai_ids))}<small>/{len(ai_ids)}</small>", "Dụng cụ đã khám phá bằng AI"),
        ("📖", f"{summary['read']}<small>/{len(kb)}</small>", "Bài đã học"),
        ("🎯", accuracy, "Tỉ lệ trả lời quiz đúng"),
    ]))
    st.write("")

    left, right = st.columns([6, 5], gap="large")
    with left:
        tools = kb.all()
        tool = tools[date.today().toordinal() % len(tools)]
        show(tool_of_the_day(tool, kb.group(tool.group)))
        st.write("")
        if st.button(f"📖 Học bài “{tool.name}”", width="stretch"):
            tool_dialog(tool.id)
    with right:
        show(feature("📷", "var(--yellow)", "Nhìn – Nhận dạng",
                     "Tải ảnh, chụp webcam hoặc bật camera trực tiếp. AI cho biết tên dụng cụ và độ tin cậy."))
        st.write("")
        show(feature("🦺", "var(--orange-2)", "An toàn là số 1",
                     "Mỗi kết quả đều kèm <b>lưu ý an toàn</b> khi sử dụng – điều quan trọng nhất ở xưởng."))
        st.write("")
        show(feature("🏆", "var(--teal-2)", "Học mà chơi",
                     "Gom XP, lên cấp, mở khóa huy hiệu. Mỗi dụng cụ mới khám phá được thưởng thêm XP!"))

    st.write("")
    st.subheader("🧰 Ba nhóm dụng cụ cơ khí")
    show(groups_overview(kb))

    _, status = state.get_classifier()
    st.write("")
    if status.is_demo:
        show(banner("demo", "🧪", "AI đang dùng ảnh minh họa",
                    "Các chức năng học tập và trò chơi vẫn hoạt động bình thường."))
    else:
        show(banner("ok", "🤖", "AI đã sẵn sàng nhận dạng", status.message))
