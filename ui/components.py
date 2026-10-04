"""Các khối giao diện dựng bằng HTML + CSS (assets/style.css).

Mọi hàm trả về chuỗi HTML; gọi show() để hiển thị. Nội dung do người dùng nhập luôn được escape.
"""

from __future__ import annotations

import base64
import html as _html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from core.config import CSS_PATH, ICONS_DIR
from core.gamification import LEVELS, Badge, LevelProgress
from core.imaging import b64_to_data_uri
from core.knowledge import KnowledgeBase, Tool, ToolGroup
from core.model import Prediction, PredictionResult

FALLBACK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle cx="100" cy="100" r="58" '
    'fill="#FFC93C" stroke="#1B1F2A" stroke-width="5"/><text x="100" y="122" font-size="64" '
    'text-anchor="middle" font-family="sans-serif" fill="#1B1F2A">?</text></svg>'
)


def esc(value) -> str:
    return _html.escape(str(value), quote=True)


def show(markup: str) -> None:
    st.html(markup)


@lru_cache(maxsize=4)
def _read_css(path: str, mtime: float) -> str:  # mtime chỉ để làm mới cache khi sửa CSS
    return Path(path).read_text(encoding="utf-8")


def inject_css() -> None:
    if CSS_PATH.exists():
        st.html(f"<style>{_read_css(str(CSS_PATH), CSS_PATH.stat().st_mtime)}</style>")


@lru_cache(maxsize=64)
def icon_svg(tool_id: str | None) -> str:
    """Hình minh họa dụng cụ dạng <img> (st.html lọc bỏ thẻ <svg> trực tiếp nên nhúng qua data URI)."""
    path = ICONS_DIR / f"{tool_id}.svg"
    svg = path.read_text(encoding="utf-8") if tool_id and path.exists() else FALLBACK_SVG
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="xw-ico" src="data:image/svg+xml;base64,{data}" alt="" draggable="false"/>'


def group_tint(color: str) -> str:
    """Màu nền nhạt cho từng nhóm dụng cụ."""
    return {"#2F5BEA": "#DCE5FF", "#0E9F8A": "#D2F4EC", "#E8590C": "#FFE2D1"}.get(color, "#FFF0C9")


def pct(value: float) -> int:
    return int(round(value * 100))


# ----------------------------------------------------------------------------- generic
def chip(text: str, tone: str = "") -> str:
    return f'<span class="xw-chip {tone}">{esc(text)}</span>'


def page_header(num: str, title: str, subtitle: str = "") -> str:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return (f'<div class="xw-pagehead"><div class="text"><span class="num">{esc(num)}</span>'
            f"<h1>{esc(title)}</h1>{sub}</div></div>")


def banner(kind: str, icon: str, title: str, text: str = "") -> str:
    body = f"<p>{text}</p>" if text else ""
    return f'<div class="xw-banner {kind}"><div class="ic">{icon}</div><div><b>{esc(title)}</b>{body}</div></div>'


def empty_state(icon: str, title: str, text: str = "") -> str:
    return f'<div class="xw-empty"><div class="ic">{icon}</div><b>{esc(title)}</b><p>{text}</p></div>'


def stat_tiles(items: list[tuple[str, str, str]]) -> str:
    """items: (icon, value_html, label)"""
    tiles = "".join(
        f'<div class="xw-stat"><span class="ic">{icon}</span><div class="val">{value}</div>'
        f'<div class="lbl">{esc(label)}</div></div>' for icon, value, label in items
    )
    return f'<div class="xw-stats">{tiles}</div>'


def xp_bar(ratio: float) -> str:
    return f'<div class="xw-xp"><span style="width:{max(3, min(100, ratio * 100)):.1f}%"></span></div>'


def section_title(icon: str, text: str) -> str:
    return f'<div class="xw-sec-title">{icon} {esc(text)}</div>'


def bullet_list(items, cls: str = "xw-list") -> str:
    return f'<ul class="{cls}">' + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


# ----------------------------------------------------------------------------- sidebar
def brand() -> str:
    return (f'<div class="xw-brand"><div class="logo">{icon_svg("co_le")}</div>'
            '<div><b>AI Xưởng Cơ Khí</b><span>Công nghệ 8</span></div></div>')


def player_card(name: str, lp: LevelProgress) -> str:
    initials = "".join(w[0] for w in name.split()[-2:]).upper() or "?"
    nxt = f"còn {lp.xp_to_next} XP lên cấp {lp.next_level.number}" if lp.next_level else "Cấp tối đa!"
    return (
        f'<div class="xw-player"><div class="top"><div class="xw-avatar">{esc(initials)}</div><div>'
        f'<div class="name">{esc(name)}</div><div class="lvl">{lp.level.icon} Cấp {lp.level.number} · {esc(lp.level.title)}</div>'
        f"</div></div>{xp_bar(lp.ratio)}"
        f'<div class="xpline"><span>⚡ {lp.xp} XP</span><span>{esc(nxt)}</span></div></div>'
    )


# ----------------------------------------------------------------------------- home
def hero(name: str, kb: KnowledgeBase) -> str:
    tools = "".join(f'<div class="t">{icon_svg(t)}</div>' for t in ("bua", "co_le", "tua_vit", "kim") if kb.get(t))
    return f"""
<div class="xw-hero"><div class="xw-hero-grid">
  <div>
    <div class="hello">👋 Xin chào, {esc(name)}!</div>
    <h1><span class="ai">AI</span>Xưởng<br/>Cơ Khí</h1>
    <p class="lead">Đưa dụng cụ trước camera – AI sẽ cho em biết <b>tên</b>, <b>công dụng</b> và
    <b>cách dùng an toàn</b>. Học xong thì làm quiz để lên cấp từ <i>Học việc</i> tới <i>Bậc thầy cơ khí</i>!</p>
  </div>
  <div class="xw-hero-tools">{tools}</div>
</div></div>"""


def feature(icon: str, bg: str, title: str, text: str) -> str:
    return (f'<div class="xw-card xw-feature"><div class="ic" style="background:{bg}">{icon}</div>'
            f"<div><h4>{esc(title)}</h4><p>{text}</p></div></div>")


def groups_overview(kb: KnowledgeBase) -> str:
    cards = []
    for g in kb.groups():
        icons = "".join(f"<span>{icon_svg(t.id)}</span>" for t in kb.by_group(g.id))
        cards.append(f'<div class="xw-group"><div class="head" style="background:{g.color}">{g.icon} {esc(g.short)}</div>'
                     f'<div class="icons">{icons}</div><p>{esc(g.description)}</p></div>')
    return f'<div class="xw-groups">{"".join(cards)}</div>'


def tool_of_the_day(tool: Tool, group: ToolGroup) -> str:
    return (f'<div class="xw-card xw-totd"><div class="art" style="background:{group_tint(group.color)}">{icon_svg(tool.id)}</div><div>'
            f'<div class="xw-kicker">🌟 Dụng cụ của ngày</div><div class="xw-title" style="font-size:1.9rem">{esc(tool.name)}</div>'
            f'<div class="xw-muted" style="font-size:.95rem">{esc(tool.summary_sentence)}.</div>'
            f'<div class="xw-fact" style="margin-top:.7rem"><span class="ic">💡</span><span>{esc(tool.fun_fact)}</span></div>'
            "</div></div>")


# ----------------------------------------------------------------------------- recognition
def confidence_label(conf: float) -> str:
    if conf >= 0.85:
        return "Rất chắc chắn"
    if conf >= 0.6:
        return "Khá chắc chắn"
    return "Chưa chắc chắn"


def _ring(conf: float, threshold: float) -> str:
    color = "var(--teal)" if conf >= threshold else ("var(--orange)" if conf >= 0.35 else "var(--red)")
    return (f'<div class="xw-ring" style="--p:{pct(conf)};--c:{color}"><span>{pct(conf)}%'
            "<small>TIN CẬY</small></span></div>")


def prediction_name(p: Prediction, kb: KnowledgeBase) -> str:
    if p.is_background:
        return "Không có dụng cụ"
    tool = kb.get(p.tool_id)
    return tool.name if tool else p.label


def topk_bars(predictions: list[Prediction], kb: KnowledgeBase) -> str:
    rows = []
    for i, p in enumerate(predictions):
        rows.append(
            f'<div class="xw-bar {"first" if i == 0 else ""}"><span class="rank">{i + 1}</span>'
            f'<div class="stack"><div class="label">{esc(prediction_name(p, kb))}</div>'
            f'<div class="track"><span style="width:{max(1.5, p.confidence * 100):.1f}%"></span></div></div>'
            f'<span class="pct">{p.confidence * 100:.1f}%</span></div>'
        )
    return f'<div class="xw-bars">{"".join(rows)}</div>'


def result_card(result: PredictionResult, kb: KnowledgeBase, threshold: float, top_k: int,
                tips: list[str] | None = None) -> str:
    best = result.best
    tool = kb.get(best.tool_id)
    bars_title = section_title("📊", f"Top {top_k} dự đoán · {result.elapsed_ms:.0f} ms")
    bars = f'<div style="border-top:var(--border);padding:1rem 1.2rem">{bars_title}{topk_bars(result.top(top_k), kb)}</div>'
    mode_chip = chip("🧪 Ảnh minh họa", "orange") if result.mode == "demo" else chip("🤖 AI", "teal")

    if tool and not best.is_background and best.confidence >= threshold:
        group = kb.group(tool.group)
        return f"""
<div class="xw-card xw-result">
  <div class="xw-result-head">
    <div class="xw-result-icon" style="background:{group_tint(group.color)}">{icon_svg(tool.id)}</div>
    <div>
      <div class="xw-chips">{chip(group.short, "yellow")}{mode_chip}</div>
      <div class="xw-result-name">{esc(tool.name)}</div>
      <div class="xw-result-en">{esc(tool.english)} · {esc(confidence_label(best.confidence))}</div>
    </div>
    {_ring(best.confidence, threshold)}
  </div>
  <div style="border-top:var(--border);padding:.9rem 1.2rem;font-size:.97rem">{esc(tool.description)}</div>
  <div class="xw-result-body">
    <div>{section_title("🎯", "Công dụng")}{bullet_list(tool.uses)}</div>
    <div class="xw-safety">{section_title("⚠️", "Lưu ý an toàn")}{bullet_list(tool.safety)}</div>
  </div>
  {bars}
</div>"""

    if best.is_background:
        headline, emoji = "AI không thấy dụng cụ nào trong ảnh", "🫥"
    else:
        headline, emoji = f"Có thể là “{prediction_name(best, kb)}”… nhưng AI chưa chắc", "🤔"
    advice = list(tips or []) + [
        "Đặt dụng cụ ở giữa khung hình, chụp gần hơn.",
        "Dùng nền trơn (tờ giấy trắng, mặt bàn) để AI dễ nhìn.",
        "Đủ ánh sáng, không bị bóng che.",
    ]
    demo_note = ""
    if result.mode == "demo":
        demo_note = ('<p style="margin:.6rem 0 0;font-size:.9rem"><b>AI đang ở chế độ trải nghiệm</b> nên chỉ nhận ra '
                     'các <b>ảnh mẫu</b>. Chọn thẻ “Ảnh mẫu” để thử.</p>')
    return f"""
<div class="xw-card xw-result">
  <div class="xw-unsure">
    <div class="big">{emoji}</div>
    <div>
      <div class="xw-chips" style="margin-bottom:.3rem">{chip("Chưa chắc chắn", "red")}{mode_chip}</div>
      <div class="xw-title" style="font-size:1.6rem">{esc(headline)}</div>
      <div class="xw-muted" style="font-size:.93rem">Độ tin cậy cao nhất là <b>{pct(best.confidence)}%</b>,
        thấp hơn ngưỡng <b>{pct(threshold)}%</b>. Mẹo chụp lại:</div>
      {bullet_list(advice)}
      {demo_note}
    </div>
  </div>
  {bars}
</div>"""


# ----------------------------------------------------------------------------- library
def tool_card(tool: Tool, group: ToolGroup, ai: bool, read: bool) -> str:
    corner = chip("🤖 AI nhận dạng", "dark") if ai else ""
    done = chip("✓ Đã đọc", "teal") if read else ""
    return f"""
<div class="xw-tool">
  <div class="art" style="--g:{group_tint(group.color)}"><span class="corner">{corner}</span><span class="corner2">{done}</span>{icon_svg(tool.id)}</div>
  <div class="body">
    <div class="xw-chips" style="margin-bottom:.4rem">{chip(group.short)}</div>
    <div class="name">{esc(tool.name)}</div>
    <div class="en">{esc(tool.english)}</div>
    <p class="sum">{esc(tool.summary_sentence)}.</p>
  </div>
</div>"""


def tool_detail_head(tool: Tool, group: ToolGroup, ai: bool) -> str:
    chips = chip(group.icon + " " + group.name, "yellow") + (chip("🤖 AI nhận dạng được", "dark") if ai else "")
    return f"""
<div class="xw-detail-head">
  <div class="art" style="--g:{group_tint(group.color)}">{icon_svg(tool.id)}</div>
  <div>
    <div class="xw-chips">{chips}</div>
    <div class="xw-title" style="font-size:2.4rem;margin-top:.4rem">{esc(tool.name)}</div>
    <div class="xw-result-en">{esc(tool.english)}</div>
    <p style="margin:.5rem 0 0">{esc(tool.description)}</p>
  </div>
</div>"""


def tool_detail_sections(tool: Tool) -> str:
    return f"""
<div class="xw-detail-grid">
  <div class="xw-card">{section_title("🧩", "Cấu tạo")}{bullet_list(tool.structure)}</div>
  <div class="xw-card">{section_title("🎯", "Công dụng")}{bullet_list(tool.uses)}</div>
  <div class="xw-card">{section_title("🛠️", "Cách sử dụng")}{bullet_list(tool.how_to_use, "xw-steps")}</div>
  <div class="xw-card xw-safety">{section_title("⚠️", "Lưu ý an toàn")}{bullet_list(tool.safety)}</div>
  <div class="xw-fact wide"><span class="ic">💡</span><span><b>Em có biết?</b> {esc(tool.fun_fact)}</span></div>
</div>"""


# ----------------------------------------------------------------------------- progress
def level_card(lp: LevelProgress) -> str:
    ladder = "".join(f'<span class="{"on" if lvl.number <= lp.level.number else ""}">{lvl.icon} {esc(lvl.title)}</span>'
                     for lvl in LEVELS)
    sub = (f"{lp.xp_into_level}/{lp.xp_needed} XP · còn {lp.xp_to_next} XP để lên “{lp.next_level.title}”"
           if lp.next_level else "Em đã đạt cấp cao nhất – tuyệt vời!")
    return f"""
<div class="xw-level">
  <div class="shield">{lp.level.icon}<b>LV{lp.level.number}</b></div>
  <div>
    <div class="xw-kicker">Cấp độ hiện tại · {lp.xp} XP</div>
    <div class="title">{esc(lp.level.title)}</div>
    {xp_bar(lp.ratio)}
    <div class="sub">{esc(sub)}</div>
    <div class="xw-ladder">{ladder}</div>
  </div>
</div>"""


def badge_grid(badges: list[Badge], owned: dict, new_ids: set[str] | None = None) -> str:
    new_ids = new_ids or set()
    cells = []
    for b in badges:
        got = b.id in owned
        cls = "xw-badge" + ("" if got else " locked") + (" new" if b.id in new_ids else "")
        date = f'<div class="date">✓ {esc(owned[b.id][:10])}</div>' if got else '<div class="date" style="color:var(--ink-3)">🔒 Chưa mở</div>'
        cells.append(f'<div class="{cls}"><div class="medal">{b.icon}</div><b>{esc(b.name)}</b>'
                     f"<small>{esc(b.description)}</small>{date}</div>")
    return f'<div class="xw-badges">{"".join(cells)}</div>'


SOURCE_LABELS = {"upload": "📤 Tải ảnh", "camera": "📸 Webcam", "live": "🎥 Camera trực tiếp", "sample": "🧪 Ảnh mẫu"}


def history_rows(entries: list[dict], kb: KnowledgeBase) -> str:
    rows = []
    for h in entries:
        tool = kb.get(h.get("tool_id"))
        if tool:
            title = tool.name
        else:
            guess_tool = kb.get(h["top"][0]["tool_id"]) if h.get("top") else None
            title = f"Chưa chắc chắn ({guess_tool.name if guess_tool else h['label']})"
        img = (f'<img src="{b64_to_data_uri(h["thumb"])}" alt=""/>' if h.get("thumb")
               else f'<div class="noimg">{icon_svg(h.get("tool_id"))}</div>')
        fb = h.get("feedback")
        fb_chip = ""
        if fb:
            if fb["correct"]:
                fb_chip = chip("✓ Đúng", "teal")
            else:
                fixed = kb.get(fb.get("tool_id"))
                fb_chip = chip("✗ Sai" + (f" → {fixed.name}" if fixed else ""), "red")
        when = h["time"].replace("T", " ")[:16]
        xp = chip(f"+{h['xp']} XP", "yellow") if h.get("xp") else ""
        mode = " · ảnh minh họa" if h.get("mode") == "demo" else ""
        rows.append(
            f'<div class="xw-hist-row">{img}<div><div class="t">{esc(title)}</div>'
            f'<div class="m">{esc(when)} · {esc(SOURCE_LABELS.get(h.get("source"), h.get("source", "")))}{mode}</div>'
            f'<div class="xw-chips" style="margin-top:.3rem">{xp}{fb_chip}</div></div>'
            f'<div class="r"><div class="pct">{h["confidence"] * 100:.0f}%</div><div class="m">tin cậy</div></div></div>'
        )
    return f'<div class="xw-hist">{"".join(rows)}</div>'


def album(tools: list[Tool], unlocked: dict | set | list, label_on: str, label_off: str) -> str:
    cells = []
    for t in tools:
        on = t.id in unlocked
        cells.append(f'<div class="xw-sticker {"" if on else "off"}">{icon_svg(t.id)}'
                     f'<b>{esc(t.name if on else "???")}</b><small>{esc(label_on if on else label_off)}</small></div>')
    return f'<div class="xw-album">{"".join(cells)}</div>'


# ----------------------------------------------------------------------------- quiz
def quiz_progress(total: int, index: int, answers: list[bool]) -> str:
    dots = []
    for i in range(total):
        cls = "ok" if i < len(answers) and answers[i] else "no" if i < len(answers) else "cur" if i == index else ""
        dots.append(f'<i class="{cls}"></i>')
    return f'<div class="xw-dots">{"".join(dots)}</div>'


def question_card(prompt: str, index: int, total: int, topic_label: str, tool_id: str | None, show_icon: bool) -> str:
    art = f'<div class="qart">{icon_svg(tool_id)}</div>' if show_icon and tool_id else ""
    return (f'<div class="xw-card xw-question"><div class="xw-kicker">Câu {index + 1}/{total} · {esc(topic_label)}</div>'
            f'<div class="q">{esc(prompt)}</div>{art}</div>')


def answered_options(options: tuple[str, ...], answer: int, choice: int) -> str:
    rows = []
    for i, opt in enumerate(options):
        cls = "correct" if i == answer else "wrong" if i == choice else "dim"
        mark = "✓" if i == answer else "✗" if i == choice else "ABCD"[i]
        rows.append(f'<div class="xw-opt {cls}"><span class="k">{mark}</span><span>{esc(opt)}</span></div>')
    return "".join(rows)


def explain_box(correct: bool, xp: int, explanation: str, correct_option: str) -> str:
    if correct:
        head = f"🎉 Chính xác! +{xp} XP" if xp else "🎉 Chính xác!"
        return f'<div class="xw-explain good"><b>{esc(head)}</b><div>{esc(explanation)}</div></div>'
    return (f'<div class="xw-explain bad"><b>😅 Chưa đúng rồi!</b><div>Đáp án đúng: <b>{esc(correct_option)}</b>.'
            f" {esc(explanation)}</div></div>")


def thumbnail_img(b64: str) -> str:
    return f'<img src="{b64_to_data_uri(b64)}" style="width:100%;border-radius:14px;border:2.5px solid var(--ink)"/>'
