"""Trang Nhận dạng: tải ảnh, chụp webcam, camera trực tiếp (OpenCV) hoặc thử ảnh mẫu."""

from __future__ import annotations

import time

import cv2
import streamlit as st
from PIL import Image, ImageOps

import json

from core.config import SAMPLES_DIR, SAMPLES_REAL_DIR
from core.imaging import (
    ImageError, assess_quality, bgr_to_pil, image_digest, load_image, make_thumbnail_b64, pil_to_bytes,
)
from core.progress import find_scan, record_scan, set_scan_feedback
from ui import state
from ui.components import (
    banner, chip, confidence_label, empty_state, esc, page_header, prediction_name, result_card, show, topk_bars,
)
from ui.dialogs import tool_dialog

SOURCES = {
    "upload": "📤 Tải ảnh",
    "camera": "📸 Webcam",
    "live": "🎥 Trực tiếp",
    "sample": "🧪 Ảnh mẫu",
}
LIVE_PREDICT_EVERY_S = 0.35


# ----------------------------------------------------------------------------- processing
def process_image(data: bytes, source: str) -> None:
    """Nhận dạng một ảnh, lưu lịch sử + XP (mỗi ảnh chỉ tính một lần trong phiên)."""
    digest = image_digest(data)
    cache: dict = st.session_state.setdefault("scan_cache", {})
    if digest in cache:
        st.session_state.current_scan = cache[digest]
        return
    try:
        image = load_image(data)
    except ImageError as exc:
        st.error(str(exc))
        return

    kb = state.get_kb()
    classifier, _ = state.get_classifier()
    with st.spinner("🤖 AI đang quan sát…"):
        result = classifier.predict(image)
    tool = kb.get(result.best.tool_id)
    with state.update_profile() as profile:
        outcome = record_scan(profile, result, source=source, thumb_b64=make_thumbnail_b64(image),
                              threshold=state.threshold(), top_k=5, tool_name=tool.name if tool else "")

    preview = image.copy()
    preview.thumbnail((720, 720))
    scan = {
        "entry_id": outcome.entry["id"],
        "result": result,
        "preview": pil_to_bytes(preview),
        "tips": assess_quality(image).tips,
        "xp": outcome.xp,
        "new_tool": outcome.new_tool,
        "note": outcome.note,
    }
    cache[digest] = scan
    st.session_state.current_scan = scan
    st.rerun()  # vẽ lại toàn trang để thanh bên cập nhật XP/cấp độ ngay


def _send_feedback(entry_id: str, correct: bool, key: str | None = None) -> None:
    corrected = st.session_state.get(key) if key else None
    with state.update_profile() as profile:
        set_scan_feedback(profile, entry_id, correct, corrected)
    state.notify("Cảm ơn em! Phản hồi giúp thầy cô cải thiện AI.", "🙏")


# ----------------------------------------------------------------------------- input widgets
def _upload_input() -> None:
    file = st.file_uploader("Chọn ảnh dụng cụ (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp", "bmp"],
                            key="upload_file")
    if file is not None:
        process_image(file.getvalue(), "upload")
        scan = st.session_state.get("current_scan")
        if scan:
            st.image(scan["preview"], caption="Ảnh em vừa tải lên", width="stretch")
    else:
        st.caption("💡 Mẹo: chụp dụng cụ trên nền trơn, đủ sáng, dụng cụ chiếm phần lớn khung hình.")


def _camera_input() -> None:
    if not st.toggle("Bật webcam", key="cam_on", help="Trình duyệt sẽ hỏi quyền dùng camera."):
        st.caption("Bật webcam để chụp ảnh dụng cụ ngay trên trình duyệt.")
        return
    shot = st.camera_input("Đưa dụng cụ vào giữa khung hình rồi bấm chụp", key="cam_shot")
    if shot is not None:
        process_image(shot.getvalue(), "camera")


@st.cache_data(show_spinner=False)
def _square_thumb(path: str, mtime: float) -> bytes:  # noqa: ARG001 – mtime làm mới cache khi ảnh đổi
    with Image.open(path) as im:
        return pil_to_bytes(ImageOps.fit(im.convert("RGB"), (320, 320), Image.Resampling.LANCZOS))


def _sample_grid(paths, key: str, captions: dict | None = None) -> None:
    with st.container(key=key):
        cols = st.columns(3)
        for i, path in enumerate(paths):
            with cols[i % 3]:
                st.image(_square_thumb(str(path), path.stat().st_mtime), width="stretch",
                         caption=(captions or {}).get(path.name))
                if st.button(f"Thử #{i + 1}", key=f"sample_{path.parent.name}_{path.stem}", width="stretch"):
                    process_image(path.read_bytes(), "sample")


def _sample_input() -> None:
    real = sorted(SAMPLES_REAL_DIR.glob("*.jpg"))
    drawn = sorted(SAMPLES_DIR.glob("*.png"))
    if not real and not drawn:
        st.info("Chưa có ảnh mẫu. Chạy: python scripts/generate_assets.py")
        return
    _, status = state.get_classifier()
    if real and not status.is_demo:
        st.markdown("**📷 Ảnh chụp thật** – AI chưa từng thấy các ảnh này khi học")
        credits = {}
        credit_file = SAMPLES_REAL_DIR / "CREDITS.json"
        if credit_file.exists():
            for m in json.loads(credit_file.read_text(encoding="utf-8")):
                credits[m["file"]] = f"© {m['creator']} · {m['license']}"
        _sample_grid(real, "sample_grid_real", credits)
        st.write("")
    st.markdown("**🎨 Ảnh minh họa** – dùng được cả khi chưa có model")
    _sample_grid(drawn, "sample_grid")


def _snap_live() -> None:
    st.session_state["live_on"] = False  # tắt camera để xem kết quả đầy đủ
    st.session_state["live_snap_pending"] = True


def _live_controls() -> tuple[bool, int]:
    st.caption("Dùng webcam **của máy đang chạy ứng dụng** (OpenCV) – AI dự đoán liên tục. "
               "Bấm “Chụp & lưu” để ghi kết quả vào lịch sử.")
    with st.expander("Tùy chọn camera"):
        cam_index = st.number_input("Số thứ tự camera", min_value=0, max_value=5, value=0, step=1, key="live_cam_index",
                                    help="Máy có nhiều camera thì thử 1, 2…")
    if st.session_state.pop("live_snap_pending", False) and st.session_state.get("live_last"):
        process_image(st.session_state["live_last"], "live")
    running = st.toggle("▶️ Bật camera trực tiếp", key="live_on")
    st.button("📸 Chụp & lưu kết quả", type="primary", width="stretch", on_click=_snap_live,
              disabled=not (running and st.session_state.get("live_last")))
    return running, int(cam_index)


def _run_live(frame_slot, info_slot, cam_index: int) -> None:
    """Vòng lặp camera OpenCV. Tự dừng khi người dùng bấm nút khác (Streamlit chạy lại script)."""
    kb = state.get_kb()
    classifier, _ = state.get_classifier()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        frame_slot.error("Không mở được camera. Kiểm tra camera có đang được ứng dụng khác dùng không, "
                         "và đã cấp quyền camera cho Terminal/Python (macOS: Cài đặt → Quyền riêng tư → Camera).")
        return
    state.flush_notifications()
    last_predict, fails = 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    frame_slot.warning("Camera không gửi hình ảnh. Hãy tắt và bật lại.")
                    break
                time.sleep(0.05)
                continue
            fails = 0
            image = ImageOps.mirror(bgr_to_pil(frame))
            image.thumbnail((640, 640))
            now = time.time()
            if now - last_predict >= LIVE_PREDICT_EVERY_S:
                result = classifier.predict(image)
                last_predict = now
                best = result.best
                name = prediction_name(best, kb)
                tone = "teal" if best.confidence >= state.threshold() and best.tool_id else "red"
                info_slot.html(
                    f'<div class="xw-card"><div class="xw-kicker">🎥 Đang nhận dạng trực tiếp</div>'
                    f'<div class="xw-title" style="font-size:2rem">{esc(name)}</div>'
                    f'<div class="xw-chips" style="margin:.2rem 0 .8rem">{chip(f"{best.confidence * 100:.0f}% · {confidence_label(best.confidence)}", tone)}</div>'
                    f"{topk_bars(result.top(state.top_k()), kb)}</div>"
                )
                st.session_state.live_last = pil_to_bytes(image)
            frame_slot.image(image, width="stretch")
            time.sleep(0.03)
    finally:
        cap.release()


# ----------------------------------------------------------------------------- result panel
def _result_panel() -> None:
    kb = state.get_kb()
    scan = st.session_state.get("current_scan")
    if not scan:
        show(empty_state("🔎", "Chưa có ảnh nào",
                         "Chọn một cách đưa ảnh ở bên trái. Kết quả nhận dạng sẽ hiện ở đây."))
        return
    result = scan["result"]
    threshold = state.threshold()
    show(result_card(result, kb, threshold, state.top_k(), scan["tips"]))

    if scan["xp"]:
        msg = f"+{scan['xp']} XP" + (" · 🎁 Em vừa khám phá một dụng cụ mới!" if scan["new_tool"] else "")
        show(banner("ok", "⚡", msg, "Tiếp tục khám phá các dụng cụ khác để hoàn thành bộ sưu tập."))
    elif scan["note"]:
        show(banner("warn", "ℹ️", scan["note"]))

    best = result.best
    tool = kb.get(best.tool_id)
    if tool and best.confidence >= threshold and not best.is_background:
        if st.button(f"📖 Học bài đầy đủ về “{tool.name}”", width="stretch"):
            tool_dialog(tool.id)

    entry = find_scan(state.current_profile(), scan["entry_id"])
    if entry is None:
        return
    st.write("")
    if entry.get("feedback"):
        st.caption("✅ Em đã phản hồi kết quả này. Cảm ơn em!")
        return
    st.markdown("**🧑‍🏫 Kết quả này có đúng không?** Phản hồi giúp thầy cô biết AI cần học thêm gì.")
    c1, c2 = st.columns(2)
    c1.button("✅ Đúng rồi", key=f"fb_ok_{entry['id']}", width="stretch",
              on_click=_send_feedback, args=(entry["id"], True))
    with c2.popover("❌ Sai rồi", width="stretch"):
        key = f"fb_tool_{entry['id']}"
        st.selectbox("Dụng cụ đúng là:", kb.ids(), format_func=lambda i: kb.get(i).name, key=key)
        st.button("Gửi phản hồi", key=f"fb_send_{entry['id']}", type="primary", width="stretch",
                  on_click=_send_feedback, args=(entry["id"], False, key))


# ----------------------------------------------------------------------------- page
def render() -> None:
    show(page_header("01", "Nhận dạng dụng cụ",
                     "Đưa ảnh dụng cụ cho AI xem. AI sẽ cho biết tên, độ tin cậy, công dụng và lưu ý an toàn."))
    _, status = state.get_classifier()
    if status.is_demo:
        show(banner("demo", "🧪", "AI đang ở chế độ trải nghiệm",
                    "Kết quả dùng ảnh minh họa và chỉ để em làm quen với tính năng nhận dạng."))
        st.write("")

    left, right = st.columns([5, 7], gap="large")
    with left:
        source = st.pills("Cách đưa ảnh", list(SOURCES), format_func=SOURCES.get,
                          default="upload", key="scan_source") or "upload"
        live = None
        if source == "upload":
            _upload_input()
        elif source == "camera":
            _camera_input()
        elif source == "sample":
            _sample_input()
        else:
            live = _live_controls()
            frame_slot = st.empty()

    with right:
        if source == "live" and live and live[0]:
            info_slot = st.empty()
        else:
            _result_panel()

    if live and live[0]:
        _run_live(frame_slot, info_slot, live[1])
