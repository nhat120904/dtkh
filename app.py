"""AI Xưởng Cơ Khí – Công nghệ 8.

Chạy ứng dụng:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI Xưởng Cơ Khí – Công nghệ 8",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="auto",
)

from ui import state  # noqa: E402  (phải import sau set_page_config)
from ui.components import inject_css, show  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from views import home, library, progress_page, quiz, recognize  # noqa: E402

inject_css()
state.flush_notifications()

pages = {
    "home": st.Page(home.render, title="Trang chủ", icon="🏠", url_path="trang-chu", default=True),
    "recognize": st.Page(recognize.render, title="Nhận dạng", icon="📷", url_path="nhan-dang"),
    "library": st.Page(library.render, title="Kho kiến thức", icon="📚", url_path="kho-kien-thuc"),
    "quiz": st.Page(quiz.render, title="Quiz", icon="🧠", url_path="quiz"),
    "progress": st.Page(progress_page.render, title="Tiến độ", icon="🏆", url_path="tien-do"),
}
state.PAGES = pages  # để các trang tạo liên kết qua lại

navigation = st.navigation(list(pages.values()), position="top")
render_sidebar()
navigation.run()

show('<div class="xw-footer">AI Xưởng Cơ Khí dành cho môn Công nghệ 8</div>')
state.flush_notifications()
