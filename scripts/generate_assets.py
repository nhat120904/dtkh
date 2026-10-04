"""Sinh hình minh họa dụng cụ cơ khí (SVG cho giao diện + PNG ảnh mẫu).

Mỗi dụng cụ được mô tả bằng vài hình học đơn giản trong hệ tọa độ 200x200.
Cùng một mô tả được vẽ ra:
  * assets/icons/<id>.svg      – icon sắc nét cho giao diện
  * assets/samples/<id>_<n>.png – ảnh mẫu để thử nhận dạng (chế độ demo / kiểm tra model)

Chạy lại bất cứ lúc nào:  python scripts/generate_assets.py
"""

from __future__ import annotations

import math
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = ROOT / "assets" / "icons"
SAMPLES_DIR = ROOT / "assets" / "samples"

INK = "#1B1F2A"
STROKE = 4.5

STEEL = "#C3CDD8"
STEEL_D = "#8C9AAE"
STEEL_L = "#E6ECF2"
STEEL_X = "#5D6B80"
WOOD = "#E9A55E"
WOOD_D = "#C9803C"
YEL = "#FFC93C"
YEL_D = "#E0A800"
ORG = "#FF7A30"
ORG_D = "#D9541A"
RED = "#F25C54"
BLUE = "#3D6BF0"
DARK = "#3A4255"


# --------------------------------------------------------------------------- shapes
@dataclass
class Shape:
    fill: str
    stroke: bool = True
    sw: float = STROKE


@dataclass
class Rect(Shape):
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    rx: float = 0

    def points(self, steps: int = 6) -> list[tuple[float, float]]:
        r = min(self.rx, self.w / 2, self.h / 2)
        if r <= 0:
            return [(self.x, self.y), (self.x + self.w, self.y),
                    (self.x + self.w, self.y + self.h), (self.x, self.y + self.h)]
        corners = [
            (self.x + self.w - r, self.y + r, -90),
            (self.x + self.w - r, self.y + self.h - r, 0),
            (self.x + r, self.y + self.h - r, 90),
            (self.x + r, self.y + r, 180),
        ]
        pts = []
        for cx, cy, start in corners:
            for i in range(steps + 1):
                a = math.radians(start + 90 * i / steps)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts


@dataclass
class Poly(Shape):
    pts: list = field(default_factory=list)
    hole: list | None = None


@dataclass
class Circle(Shape):
    cx: float = 0
    cy: float = 0
    r: float = 0


@dataclass
class Ring(Shape):
    cx: float = 0
    cy: float = 0
    r: float = 0
    r_in: float = 0


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 3
    color: str = INK


def R(x, y, w, h, fill, rx=0, **kw):
    return Rect(fill=fill, x=x, y=y, w=w, h=h, rx=rx, **kw)


def P(pts, fill, hole=None, **kw):
    return Poly(fill=fill, pts=pts, hole=hole, **kw)


def C(cx, cy, r, fill, **kw):
    return Circle(fill=fill, cx=cx, cy=cy, r=r, **kw)


def arc(cx, cy, r, a0, a1, steps=14):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / steps)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / steps))) for i in range(steps + 1)]


# --------------------------------------------------------------------------- tool drawings
def icon_bua():
    return -32, [
        R(91, 70, 18, 112, WOOD, rx=8),
        Line(95, 150, 105, 150, 3, WOOD_D), Line(95, 160, 105, 160, 3, WOOD_D), Line(95, 170, 105, 170, 3, WOOD_D),
        P([(152, 40), (178, 47), (178, 63), (152, 70)], STEEL_D),
        R(44, 34, 112, 40, STEEL, rx=7),
        R(34, 38, 16, 32, STEEL_D, rx=4),
        Line(62, 44, 138, 44, 3, STEEL_L),
    ]


def icon_kim():
    return -20, [
        P([(88, 98), (102, 106), (78, 186), (62, 180)], RED),
        P([(98, 106), (112, 98), (138, 180), (122, 186)], RED),
        P([(84, 102), (92, 32), (100, 16), (108, 32), (116, 102)], STEEL),
        Line(100, 24, 100, 84, 3, INK),
        Line(93, 48, 98, 48, 2, STEEL_X), Line(102, 48, 107, 48, 2, STEEL_X),
        Line(92, 58, 98, 58, 2, STEEL_X), Line(102, 58, 108, 58, 2, STEEL_X),
        Line(91, 68, 98, 68, 2, STEEL_X), Line(102, 68, 109, 68, 2, STEEL_X),
        C(100, 96, 11, STEEL_D),
        C(100, 96, 4, INK, stroke=False),
    ]


def icon_tua_vit():
    return 38, [
        R(95, 24, 10, 88, STEEL, rx=2),
        P([(95, 26), (105, 26), (103, 10), (97, 10)], STEEL_D),
        R(88, 106, 24, 12, STEEL_D, rx=3),
        R(82, 116, 36, 70, YEL, rx=14),
        Line(92, 130, 92, 172, 3, YEL_D), Line(100, 130, 100, 172, 3, YEL_D), Line(108, 130, 108, 172, 3, YEL_D),
    ]


def icon_co_le():
    jaw = [(76, 12), (76, 40)] + arc(100, 40, 24, 180, 0)[1:-1] + [(124, 40), (124, 12), (111, 12), (111, 34), (89, 34), (89, 12)]
    return -40, [
        R(91, 46, 18, 100, STEEL, rx=6),
        Line(100, 64, 100, 128, 3, STEEL_D),
        P(jaw, STEEL),
        Ring(fill=STEEL, cx=100, cy=166, r=24, r_in=12),
    ]


def icon_mo_let():
    return -38, [
        R(90, 84, 22, 100, STEEL, rx=11),
        C(101, 168, 5, DARK, stroke=False),
        R(68, 60, 64, 32, STEEL, rx=10),
        P([(68, 72), (68, 22), (82, 14), (90, 22), (90, 66)], STEEL),
        R(110, 30, 18, 38, STEEL_D, rx=3),
        R(95, 70, 20, 15, YEL, rx=3),
        Line(100, 71, 100, 84, 2, INK), Line(105, 71, 105, 84, 2, INK), Line(110, 71, 110, 84, 2, INK),
    ]


def icon_cua_tay():
    teeth = [(48, 114)]
    for x in range(48, 166, 6):
        teeth += [(x + 3, 121), (x + 6, 114)]
    teeth += [(166, 114)]
    return -12, [
        P(teeth, STEEL, sw=2.2),
        R(48, 104, 118, 11, STEEL_L, rx=1, sw=3),
        R(42, 52, 130, 12, BLUE, rx=5),
        R(158, 52, 12, 66, BLUE, rx=5),
        R(44, 52, 12, 66, BLUE, rx=5),
        C(52, 109, 4, DARK, stroke=False),
        C(163, 109, 4, DARK, stroke=False),
        P([(170, 100), (184, 92), (184, 120), (170, 113)], YEL),
        P([(50, 98), (32, 102), (18, 160), (44, 168), (58, 116)], ORG),
    ]


def icon_giua():
    hatch = [Line(92, y + 5, 108, y, 2, STEEL_X) for y in range(24, 120, 7)]
    return 42, [
        P([(91, 18), (95, 12), (105, 12), (109, 18), (111, 130), (89, 130)], STEEL_D),
        *hatch,
        R(86, 128, 28, 9, STEEL, rx=2),
        R(84, 136, 32, 54, WOOD, rx=14),
        Line(92, 150, 92, 178, 3, WOOD_D), Line(100, 150, 100, 178, 3, WOOD_D), Line(108, 150, 108, 178, 3, WOOD_D),
    ]


def icon_duc():
    return 30, [
        R(90, 30, 20, 122, STEEL, rx=3),
        Line(96, 40, 96, 146, 2, STEEL_L),
        R(87, 18, 26, 16, STEEL_D, rx=6),
        P([(90, 150), (110, 150), (117, 180), (83, 180)], STEEL),
        Line(87, 172, 113, 172, 2, STEEL_L),
    ]


def icon_e_to():
    return 0, [
        R(46, 150, 108, 16, DARK, rx=4),
        R(62, 110, 92, 42, BLUE, rx=6),
        R(156, 84, 22, 12, STEEL, rx=2),
        R(174, 50, 8, 80, STEEL, rx=4),
        C(178, 48, 7, DARK), C(178, 132, 7, DARK),
        R(96, 46, 14, 50, ORG, rx=1),
        R(46, 62, 46, 54, BLUE, rx=6),
        R(114, 62, 46, 54, BLUE, rx=6),
        R(88, 56, 8, 38, STEEL_D, rx=1),
        R(110, 56, 8, 38, STEEL_D, rx=1),
        Line(56, 74, 80, 74, 3, "#6F91FF"), Line(124, 74, 148, 74, 3, "#6F91FF"),
    ]


def icon_thuoc_la():
    ticks = []
    for i, x in enumerate(range(26, 170, 6)):
        ticks.append(Line(x, 84, x, 84 + (15 if i % 5 == 0 else 8), 2, INK))
    return -24, [
        R(18, 84, 164, 32, STEEL_L, rx=3),
        *ticks,
        C(174, 106, 4, DARK, stroke=False),
    ]


def icon_thuoc_cap():
    ticks = []
    for i, x in enumerate(range(118, 182, 5)):
        ticks.append(Line(x, 82, x, 82 - (9 if i % 2 == 0 else 5), 2, INK))
    return -14, [
        Line(184, 73, 196, 73, 3, STEEL_X),
        R(24, 64, 162, 18, STEEL_L, rx=3),
        *ticks,
        P([(24, 82), (46, 82), (46, 146), (30, 146)], STEEL),
        P([(30, 64), (42, 64), (40, 40), (33, 43)], STEEL),
        R(70, 58, 42, 30, STEEL, rx=4),
        P([(72, 88), (92, 88), (88, 146), (72, 146)], STEEL),
        P([(74, 58), (84, 58), (80, 36), (76, 38)], STEEL),
        C(103, 52, 6, YEL),
        Line(76, 70, 106, 70, 2, STEEL_X),
    ]


def icon_e_ke():
    ticks = []
    for i, x in enumerate(range(76, 178, 6)):
        ticks.append(Line(x, 140, x, 140 + (9 if i % 5 == 0 else 5), 2, INK))
    return -10, [
        P([(80, 30), (80, 126), (176, 126)], YEL, hole=[(93, 66), (93, 114), (141, 114)]),
        R(40, 38, 28, 126, WOOD, rx=4),
        C(54, 70, 4, STEEL_D), C(54, 134, 4, STEEL_D),
        R(68, 140, 112, 18, STEEL_L, rx=2),
        *ticks,
    ]


def icon_mui_vach():
    hatch = []
    for y in range(72, 116, 8):
        hatch += [Line(91, y, 109, y + 8, 1.6, STEEL_X), Line(109, y, 91, y + 8, 1.6, STEEL_X)]
    return 36, [
        P([(96, 150), (104, 150), (100, 184)], STEEL_L),
        P([(96, 26), (104, 26), (100, 8)], STEEL_L),
        R(96, 22, 8, 130, STEEL, rx=3),
        R(90, 70, 20, 50, STEEL_D, rx=4),
        *hatch,
    ]


def icon_cham_dau():
    hatch = []
    for y in range(40, 116, 9):
        hatch += [Line(88, y, 112, y + 9, 1.6, STEEL_X), Line(112, y, 88, y + 9, 1.6, STEEL_X)]
    return 24, [
        P([(94, 154), (106, 154), (100, 182)], STEEL_L),
        P([(86, 122), (114, 122), (106, 156), (94, 156)], STEEL),
        R(86, 32, 28, 92, STEEL_D, rx=4),
        *hatch,
        R(84, 20, 32, 14, STEEL, rx=5),
    ]


def icon_khoan():
    return -6, [
        R(178, 66, 20, 7, STEEL_L, rx=1, sw=3),
        Line(183, 66, 187, 73, 1.6, INK), Line(189, 66, 193, 73, 1.6, INK),
        P([(158, 56), (180, 62), (180, 77), (158, 83)], STEEL),
        R(146, 54, 14, 31, DARK, rx=3),
        P([(70, 88), (104, 88), (98, 158), (64, 158)], DARK),
        R(102, 94, 10, 20, YEL, rx=3),
        R(36, 46, 114, 46, ORG, rx=20),
        Line(58, 58, 58, 80, 3, ORG_D), Line(66, 58, 66, 80, 3, ORG_D), Line(74, 58, 74, 80, 3, ORG_D),
        R(50, 154, 66, 24, DARK, rx=5),
        R(50, 154, 66, 7, YEL, rx=3),
    ]


ICONS = {
    "bua": icon_bua, "kim": icon_kim, "tua_vit": icon_tua_vit, "co_le": icon_co_le,
    "mo_let": icon_mo_let, "cua_tay": icon_cua_tay, "giua": icon_giua, "duc": icon_duc,
    "e_to": icon_e_to, "thuoc_la": icon_thuoc_la, "thuoc_cap": icon_thuoc_cap, "e_ke": icon_e_ke,
    "mui_vach": icon_mui_vach, "cham_dau": icon_cham_dau, "khoan": icon_khoan,
}
# Các dụng cụ có ảnh mẫu để thử nhận dạng (trùng với 6 lớp model mặc định)
SAMPLE_TOOLS = ["bua", "kim", "tua_vit", "co_le", "cua_tay", "giua"]


# --------------------------------------------------------------------------- SVG output
def _pts(points) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _stroke_attr(shape: Shape) -> str:
    if not shape.stroke:
        return 'stroke="none"'
    return f'stroke="{INK}" stroke-width="{shape.sw}" stroke-linejoin="round" stroke-linecap="round"'


def shape_svg(shape, shadow: bool = False) -> str:
    if isinstance(shape, Line):
        if shadow:
            return ""
        return (f'<line x1="{shape.x1}" y1="{shape.y1}" x2="{shape.x2}" y2="{shape.y2}" '
                f'stroke="{shape.color}" stroke-width="{shape.width}" stroke-linecap="round"/>')
    fill = INK if shadow else shape.fill
    stroke = 'stroke="none"' if shadow else _stroke_attr(shape)
    if isinstance(shape, Rect):
        return (f'<rect x="{shape.x}" y="{shape.y}" width="{shape.w}" height="{shape.h}" rx="{shape.rx}" '
                f'fill="{fill}" {stroke}/>')
    if isinstance(shape, Circle):
        return f'<circle cx="{shape.cx}" cy="{shape.cy}" r="{shape.r}" fill="{fill}" {stroke}/>'
    if isinstance(shape, Ring):
        cx, cy, r, ri = shape.cx, shape.cy, shape.r, shape.r_in
        d = (f"M{cx - r},{cy} a{r},{r} 0 1,0 {2 * r},0 a{r},{r} 0 1,0 {-2 * r},0 Z "
             f"M{cx - ri},{cy} a{ri},{ri} 0 1,0 {2 * ri},0 a{ri},{ri} 0 1,0 {-2 * ri},0 Z")
        return f'<path d="{d}" fill-rule="evenodd" fill="{fill}" {stroke}/>'
    if isinstance(shape, Poly):
        if shape.hole:
            d = "M" + " L".join(f"{x},{y}" for x, y in shape.pts) + " Z M" + " L".join(f"{x},{y}" for x, y in shape.hole) + " Z"
            return f'<path d="{d}" fill-rule="evenodd" fill="{fill}" {stroke}/>'
        return f'<polygon points="{_pts(shape.pts)}" fill="{fill}" {stroke}/>'
    raise TypeError(shape)


def to_svg(tool_id: str) -> str:
    angle, shapes = ICONS[tool_id]()
    body = "".join(shape_svg(s) for s in shapes)
    shadow = "".join(shape_svg(s, shadow=True) for s in shapes)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" role="img">'
        f'<g transform="rotate({angle} 100 100)">'
        f'<g transform="translate(5 7)" opacity="0.16">{shadow}</g>'
        f"{body}</g></svg>"
    )


# --------------------------------------------------------------------------- PNG output
def _rot(points, angle, cx=100.0, cy=100.0):
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in points]


def _hex(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha


def _outline(draw: ImageDraw.ImageDraw, pts, width: float):
    closed = list(pts) + [pts[0], pts[1]]
    draw.line(closed, fill=_hex(INK), width=max(1, round(width)), joint="curve")


def render_png(tool_id: str, px: int, extra_angle: float = 0.0) -> Image.Image:
    """Vẽ icon ra ảnh RGBA nền trong suốt (siêu lấy mẫu 4x cho mượt)."""
    k = 4
    big = px * k
    s = big / 200.0
    angle, shapes = ICONS[tool_id]()
    angle += extra_angle

    def tf(points):
        return [(x * s, y * s) for x, y in _rot(points, angle)]

    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ds = ImageDraw.Draw(shadow)
    sh_off = (5 * s, 7 * s)

    for shape in shapes:
        if isinstance(shape, Line):
            (x1, y1), (x2, y2) = tf([(shape.x1, shape.y1), (shape.x2, shape.y2)])
            d.line([(x1, y1), (x2, y2)], fill=_hex(shape.color), width=max(1, round(shape.width * s)))
            r = shape.width * s / 2
            for x, y in ((x1, y1), (x2, y2)):
                d.ellipse([x - r, y - r, x + r, y + r], fill=_hex(shape.color))
            continue
        sw = shape.sw * s
        if isinstance(shape, (Rect, Poly)):
            pts = tf(shape.points() if isinstance(shape, Rect) else shape.pts)
            ds.polygon([(x + sh_off[0], y + sh_off[1]) for x, y in pts], fill=(27, 31, 42, 46))
            d.polygon(pts, fill=_hex(shape.fill))
            if shape.stroke:
                _outline(d, pts, sw)
            if isinstance(shape, Poly) and shape.hole:
                hole = tf(shape.hole)
                d.polygon(hole, fill=(0, 0, 0, 0))
                ds.polygon([(x + sh_off[0], y + sh_off[1]) for x, y in hole], fill=(0, 0, 0, 0))
                if shape.stroke:
                    _outline(d, hole, sw)
        elif isinstance(shape, (Circle, Ring)):
            (cx, cy), = tf([(shape.cx, shape.cy)])
            r = shape.r * s
            ds.ellipse([cx - r + sh_off[0], cy - r + sh_off[1], cx + r + sh_off[0], cy + r + sh_off[1]], fill=(27, 31, 42, 46))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_hex(shape.fill),
                      outline=_hex(INK) if shape.stroke else None, width=round(sw) if shape.stroke else 0)
            if isinstance(shape, Ring):
                ri = shape.r_in * s
                d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=(0, 0, 0, 0), outline=_hex(INK), width=round(sw))
                ds.ellipse([cx - ri + sh_off[0], cy - ri + sh_off[1], cx + ri + sh_off[0], cy + ri + sh_off[1]], fill=(0, 0, 0, 0))

    shadow = shadow.filter(ImageFilter.GaussianBlur(3 * k))
    out = Image.alpha_composite(shadow, layer)
    return out.resize((px, px), Image.Resampling.LANCZOS)


SAMPLE_BACKGROUNDS = [
    ("#F4EEE2", 0.0, (0, 0)),     # giấy kraft sáng
    ("#DDE5EE", 22.0, (10, -8)),  # bàn thép xám xanh, xoay thêm
]


def make_sample(tool_id: str, variant: int, size: int = 448) -> Image.Image:
    color, extra_angle, (dx, dy) = SAMPLE_BACKGROUNDS[variant]
    rng = np.random.default_rng(zlib.crc32(f"{tool_id}:{variant}".encode()))
    base = np.array(Image.new("RGB", (size, size), color), dtype=np.float32)
    # Nền có chút nhiễu + gradient ánh sáng để giống ảnh chụp
    yy, xx = np.mgrid[0:size, 0:size]
    light = 1.0 + 0.06 * (1 - (xx + yy) / (2 * size))
    base = base * light[..., None] + rng.normal(0, 3.0, base.shape)
    bg = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    art = render_png(tool_id, int(size * 0.86), extra_angle)
    pos = ((size - art.width) // 2 + dx, (size - art.height) // 2 + dy)
    bg.alpha_composite(art, pos)
    return bg.convert("RGB")


def main() -> int:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for tool_id in ICONS:
        (ICONS_DIR / f"{tool_id}.svg").write_text(to_svg(tool_id), encoding="utf-8")
    for tool_id in SAMPLE_TOOLS:
        for variant in range(len(SAMPLE_BACKGROUNDS)):
            make_sample(tool_id, variant).save(SAMPLES_DIR / f"{tool_id}_{variant + 1}.png", optimize=True)
    print(f"✓ {len(ICONS)} icon SVG  →  {ICONS_DIR.relative_to(ROOT)}")
    print(f"✓ {len(SAMPLE_TOOLS) * len(SAMPLE_BACKGROUNDS)} ảnh mẫu PNG  →  {SAMPLES_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
