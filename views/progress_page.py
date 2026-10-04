"""Trang Tiến độ: cấp độ, huy hiệu, lịch sử nhận dạng, kết quả quiz, bộ sưu tập, quản lý dữ liệu."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from core import progress
from core.config import SAMPLE_PROFILE_PATH
from core.gamification import BADGES, level_for
from core.quiz import TOPICS
from ui import state
from ui.components import album, badge_grid, empty_state, history_rows, level_card, page_header, show, stat_tiles

HISTORY_PAGE = 20


def _clear_history() -> None:
    with state.update_profile() as profile:
        progress.clear_history(profile)
    st.session_state.pop("scan_cache", None)
    st.session_state.pop("current_scan", None)
    state.notify("Đã xóa lịch sử nhận dạng", "🧹")


def _load_sample_profile() -> None:
    data = json.loads(SAMPLE_PROFILE_PATH.read_text(encoding="utf-8"))
    profile = state.get_store().import_profile(data)
    state.switch_profile(profile["id"])
    state.notify(f"Đã nạp hồ sơ mẫu “{profile['name']}”", "🧪")


def _delete_profile() -> None:
    profile = state.current_profile()
    state.get_store().delete(profile["id"])
    state.switch_profile("")
    state.notify(f"Đã xóa hồ sơ “{profile['name']}”", "🗑️")


# ----------------------------------------------------------------------------- tabs
def _badges_tab(profile: dict) -> None:
    owned = profile["badges"]
    st.caption(f"Đã mở khóa {len(owned)}/{len(BADGES)} huy hiệu.")
    show(badge_grid(BADGES, owned, st.session_state.get("new_badges", set())))


def _history_tab(profile: dict) -> None:
    kb = state.get_kb()
    history = profile["history"]
    if not history:
        show(empty_state("🕘", "Chưa có lượt nhận dạng nào", "Vào trang <b>Nhận dạng</b> để thử AI nhé!"))
        return
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    flt = c1.segmented_control("Lọc", ["all", "ok", "unsure"], key="hist_filter", default="all",
                               format_func={"all": "Tất cả", "ok": "✅ Chắc chắn", "unsure": "🤔 Chưa chắc"}.get) or "all"
    with c2.popover("🧹 Xóa lịch sử", width="stretch"):
        st.write("Xóa toàn bộ lịch sử nhận dạng? XP và huy hiệu vẫn được giữ.")
        st.button("Xóa", type="primary", on_click=_clear_history, width="stretch")
    rows = [h for h in history if flt == "all" or (h.get("confident") == (flt == "ok"))]
    shown = st.session_state.get("hist_limit", HISTORY_PAGE)
    show(history_rows(rows[:shown], kb))
    if len(rows) > shown:
        st.write("")
        if st.button(f"Xem thêm ({len(rows) - shown} lượt)", width="stretch"):
            st.session_state.hist_limit = shown + HISTORY_PAGE
            st.rerun()


def _learning_tab(profile: dict) -> None:
    attempts = profile["quiz_attempts"]
    if not attempts:
        show(empty_state("🧠", "Chưa làm bài quiz nào", "Làm thử một bài quiz để xem biểu đồ tiến bộ của em."))
        return
    df = pd.DataFrame(attempts)
    df["lan"] = range(1, len(df) + 1)
    df["chu_de"] = df["topic"].map(TOPICS)
    df["ngay"] = df["time"].str.replace("T", " ").str[:16]

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**📈 Điểm qua các lần làm bài**")
        base = alt.Chart(df).encode(
            x=alt.X("lan:O", title="Lần làm bài", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("score:Q", title="Điểm (thang 10)", scale=alt.Scale(domain=[0, 10])),
        )
        trend = base.mark_line(color="#1B1F2A", strokeWidth=2.5, strokeDash=[6, 4])
        points = base.mark_circle(size=170, stroke="#1B1F2A", strokeWidth=2, opacity=1).encode(
            color=alt.Color("chu_de:N", title="Chủ đề"),
            tooltip=[alt.Tooltip("ngay", title="Thời gian"), alt.Tooltip("chu_de", title="Chủ đề"),
                     alt.Tooltip("score", title="Điểm"), alt.Tooltip("xp", title="XP")],
        )
        line = (trend + points).properties(height=280)
        st.altair_chart(line, width="stretch")
    with right:
        st.markdown("**🎯 Tỉ lệ đúng theo chủ đề**")
        by_topic = df.groupby("chu_de", as_index=False)[["correct", "total"]].sum()
        by_topic["ti_le"] = by_topic["correct"] / by_topic["total"]
        bars = alt.Chart(by_topic).mark_bar(cornerRadiusEnd=6, stroke="#1B1F2A", strokeWidth=2).encode(
            y=alt.Y("chu_de:N", title=None, sort="-x"),
            x=alt.X("ti_le:Q", title="Tỉ lệ đúng", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("chu_de:N", legend=None),
            tooltip=[alt.Tooltip("chu_de", title="Chủ đề"), alt.Tooltip("ti_le", title="Tỉ lệ đúng", format=".0%")],
        ).properties(height=280)
        st.altair_chart(bars, width="stretch")

    st.markdown("**🗒️ Bảng kết quả**")
    table = df.iloc[::-1][["ngay", "chu_de", "correct", "total", "score", "xp", "best_streak", "duration_s"]]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "ngay": "Thời gian", "chu_de": "Chủ đề", "correct": "Câu đúng", "total": "Số câu",
        "score": st.column_config.ProgressColumn("Điểm", min_value=0, max_value=10, format="%.1f"),
        "xp": "XP", "best_streak": "Chuỗi đúng", "duration_s": st.column_config.NumberColumn("Thời gian (giây)", format="%.0f"),
    })

    if profile["xp_log"]:
        with st.expander("⚡ Nhật kí XP gần đây"):
            log = pd.DataFrame(profile["xp_log"][:50])
            log["time"] = log["time"].str.replace("T", " ").str[:16]
            st.dataframe(log.rename(columns={"time": "Thời gian", "amount": "XP", "reason": "Lí do"}),
                         hide_index=True, width="stretch")


def _collection_tab(profile: dict) -> None:
    kb = state.get_kb()
    ai_ids = state.ai_tool_ids()
    ai_tools = [kb.get(t) for t in ai_ids]
    st.markdown(f"**🔍 Bộ sưu tập AI** – khám phá {len(set(profile['discovered']) & set(ai_ids))}/{len(ai_tools)} "
                "dụng cụ bằng cách nhận dạng chúng")
    show(album(ai_tools, profile["discovered"], "ĐÃ KHÁM PHÁ", "CHƯA KHÁM PHÁ"))
    st.write("")
    st.markdown(f"**📖 Sổ tay kiến thức** – đã học {len(profile['read_tools'])}/{len(kb)} bài")
    show(album(kb.all(), profile["read_tools"], "ĐÃ HỌC", "CHƯA HỌC"))


def _data_tab(profile: dict) -> None:
    st.markdown("Dữ liệu của em được lưu **ngay trên máy tính này** (thư mục `user_data/profiles/`), không gửi đi đâu cả.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("💾 Tải hồ sơ (JSON)", json.dumps(profile, ensure_ascii=False, indent=1),
                       file_name=f"{profile['id']}.json", mime="application/json", width="stretch")
    if SAMPLE_PROFILE_PATH.exists():
        c2.button("🧪 Nạp hồ sơ mẫu để xem thử", on_click=_load_sample_profile, width="stretch",
                  help="Tạo một hồ sơ mới có sẵn lịch sử, kết quả quiz để xem trước bảng tiến độ.")
    with c3.popover("🗑️ Xóa hồ sơ này", width="stretch"):
        st.warning(f"Xóa vĩnh viễn hồ sơ “{profile['name']}”? Không thể hoàn tác.")
        st.button("Xóa hồ sơ", type="primary", on_click=_delete_profile, width="stretch")


def render() -> None:
    profile = state.current_profile()
    summary = progress.summarize(profile)
    show(page_header("04", "Tiến độ của em", "Cấp độ, huy hiệu, lịch sử nhận dạng và kết quả học tập."))
    show(level_card(level_for(profile["xp"])))
    st.write("")
    accuracy = f"{summary['accuracy'] * 100:.0f}<small>%</small>" if summary["accuracy"] is not None else "–"
    show(stat_tiles([
        ("📸", f"{summary['scans_total']}", "Lượt nhận dạng"),
        ("✅", f"{summary['scans_confident']}", "Nhận dạng chắc chắn"),
        ("🧠", f"{summary['quizzes']}", "Bài quiz đã làm"),
        ("🎯", accuracy, "Tỉ lệ trả lời đúng"),
        ("🔥", f"{summary['best_streak']}", "Chuỗi đúng dài nhất"),
    ]))
    st.write("")
    tabs = st.tabs(["🏅 Huy hiệu", "🕘 Lịch sử nhận dạng", "📈 Kết quả học tập", "🗂️ Bộ sưu tập", "💾 Dữ liệu"])
    with tabs[0]:
        _badges_tab(profile)
    with tabs[1]:
        _history_tab(profile)
    with tabs[2]:
        _learning_tab(profile)
    with tabs[3]:
        _collection_tab(profile)
    with tabs[4]:
        _data_tab(profile)
