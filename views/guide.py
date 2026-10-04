"""Trang hiển thị trạng thái nhận dạng AI."""

from __future__ import annotations

import streamlit as st

from ui import state
from ui.components import banner, page_header, show


def render() -> None:
    show(page_header("05", "Trạng thái AI", "Kiểm tra tính năng nhận dạng dụng cụ."))

    _, status = state.get_classifier()
    if status.is_demo:
        show(banner(
            "demo",
            "🧪",
            "AI đang ở chế độ trải nghiệm",
            "Em vẫn có thể khám phá các bài học và thử nhận dạng bằng ảnh minh họa.",
        ))
    else:
        show(banner(
            "ok",
            "🤖",
            "AI đã sẵn sàng",
            "Chọn “Nhận dạng” để bắt đầu khám phá dụng cụ.",
        ))
