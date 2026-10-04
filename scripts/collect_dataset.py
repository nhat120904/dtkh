"""Thu thập ảnh huấn luyện từ các nguồn ảnh có giấy phép tự do.

Nguồn:
  * Openverse (openverse.org) – ảnh Creative Commons từ Flickr, Wikimedia…
  * Wikimedia Commons – theo danh mục (category)
  * Open Images V7 (Google) – nhãn do con người xác nhận; dùng cho lớp "Nền"
    (ảnh tay, mặt người, bàn học, phòng… không có dụng cụ) và bổ sung ảnh dụng cụ

Kết quả: <out>/<class_id>/*.jpg  +  <out>/manifest.jsonl (nguồn, giấy phép, tác giả từng ảnh).
Ảnh được thu nhỏ (cạnh dài 512 px) và loại trùng lặp.

Chạy:  python scripts/collect_dataset.py --out dataset_raw
(Bước tiếp theo: lọc nhiễu bằng scripts/clean_dataset.py rồi huấn luyện bằng scripts/train_model.py)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

USER_AGENT = "XuongCoKhiEdu/1.0 (offline educational tool classifier; non-commercial)"
MAX_SIDE = 512
OPENVERSE = "https://api.openverse.org/v1/images/"
COMMONS = "https://commons.wikimedia.org/w/api.php"
OI_LABELS = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv"
OI_SIZE = 2_735_816_020
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"

# Lớp -> (truy vấn Openverse: số trang), danh mục Commons, nhãn Open Images
CLASSES: dict[str, dict] = {
    "bua": {
        "openverse": {"claw hammer": 12, "hammer tool": 12, "hammer nails": 12},
        "commons": ["Claw hammers", "Ball-peen hammers", "Lump hammers", "Hammers", "Rubber hammers",
                    "Wooden hammers", "Sledgehammers", "Cross-peen hammers"],
        "openimages": ["/m/03l9g", "/m/059bky", "/m/04c04r", "/m/07qym5", "/m/059bk7"],
    },
    "kim": {
        "openverse": {"pliers": 12, "needle nose pliers": 6, "combination pliers": 6, "pliers tool": 12,
                      "linesman pliers": 6, "wire cutters pliers": 6},
        "commons": ["Pliers", "Linesman's pliers", "Needle-nose pliers", "Long-nose pliers", "Diagonal cutters",
                    "Tongue-and-groove pliers", "Locking pliers", "Slip-joint pliers", "Bent nose pliers"],
        "openimages": ["/m/02qtr5", "/m/06y28c", "/m/0cqs9q", "/m/03npntq", "/m/0cqt3q", "/m/0dgq8nv"],
    },
    "tua_vit": {
        "openverse": {"screwdriver tool": 12, "phillips screwdriver": 6, "flathead screwdriver": 6,
                      "screwdriver": 12},
        "commons": ["Screwdrivers", "Electrician screwdrivers", "Jeweler's screwdrivers", "Stanley screwdrivers",
                    "Screwdrivers from the GDR", "Yankee screwdrivers"],
        "openimages": ["/m/01bms0"],
    },
    "co_le": {
        "openverse": {"combination wrench": 8, "spanner tool": 8, "open end wrench": 8, "wrench tool": 12,
                      "spanner": 12, "ring spanner": 6},
        "commons": ["Combination wrenches", "Open-end wrenches", "Box-end wrenches", "Ring spanners",
                    "Wrenches", "Spanners"],
        "openimages": ["/m/01j5ks"],
    },
    "mo_let": {
        "openverse": {"adjustable wrench": 12, "adjustable spanner": 8, "crescent wrench": 4,
                      "adjustable wrench tool": 12, "monkey wrench tool": 6, "adjustable spanner wrench": 6},
        "commons": ["Adjustable spanners", "Monkey wrenches"],
        "openimages": ["/m/0bsfj", "/m/02dv6f"],
    },
    "cua_tay": {
        "openverse": {"hacksaw": 12, "hand saw wood": 12, "hacksaw blade": 8, "handsaw": 12, "wood saw tool": 8,
                      "carpenter saw": 8, "hack saw": 8,
                      "saw cutting wood": 12, "hacksaw cutting metal": 6, "japanese saw": 8, "coping saw": 6,
                      "bow saw": 6, "pruning saw": 6},
        "commons": ["Hacksaws", "Hand saws", "Rip saws", "Crosscut saws", "Back saws", "Coping saws"],
        "openimages": ["/m/043384", "/m/0swtt", "/m/01b82r"],
    },
    "giua": {
        "openverse": {"metal file tool": 6, "rasp tool": 6, "needle files": 6, "file rasp": 6, "metal file": 12,
                      "hand file tool": 6, "rasp": 6, "file tool workshop": 6, "needle file set": 4,
                      "mill file": 12, "half round file": 12, "filing metal": 12, "woodworking rasp": 1,
                      "triangular file": 12, "flat file tool": 8, "jewelers files": 8},
        "commons": ["Files (tools)", "Rasps", "Needle files"],
        "openimages": ["/m/02p01q", "/m/050s45"],
    },
}
TOOL_LABELS = {lab for spec in CLASSES.values() for lab in spec["openimages"]} | {
    "/m/0_dqb", "/m/07ln6l", "/m/05jh2z", "/m/0hdln", "/m/02k8bg", "/m/01d380", "/m/07k1x", "/m/03lvx3",
    "/m/04sfs7", "/m/01_gqv", "/m/02x2_48", "/m/026zs_",
}
# Cảnh "không có dụng cụ" mà webcam/ảnh chụp trong lớp học hay gặp
BACKGROUND_LABELS = {
    "/m/0k65p": 260,  # Human hand
    "/m/0dzct": 200,  # Human face
    "/m/01y9k5": 120,  # Desk
    "/m/04hyxm": 80,  # Classroom
    "/m/06ht1": 120,  # Room
    "/m/04bcr3": 120,  # Table
    "/m/01c648": 60,  # Laptop
    "/m/0bt_c3": 60,  # Book
    "/m/050k8": 60,  # Mobile phone
    "/m/0k1tl": 40,  # Pen
    "/m/01m2v": 40,  # Computer keyboard
}

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT
lock = threading.Lock()


def log(msg: str) -> None:
    print(msg, flush=True)


def get_json(url: str, params: dict, tries: int = 4):
    for attempt in range(tries):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 20 * (attempt + 1)))
            log(f"   … bị giới hạn tần suất, chờ {wait}s")
            time.sleep(wait)
            continue
        if r.ok:
            try:
                return r.json()
            except ValueError:
                pass
        time.sleep(2 * (attempt + 1))
    return None


# ----------------------------------------------------------------------------- candidate listing
def openverse_candidates(query: str, pages: int) -> list[dict]:
    out = []
    for page in range(1, pages + 1):
        data = get_json(OPENVERSE, {"q": query, "page_size": 20, "page": page})
        if not data or not data.get("results"):
            break
        for r in data["results"]:
            out.append({"url": r.get("url"), "thumb": r.get("thumbnail"), "source": f"openverse:{r.get('source')}",
                        "license": f"{r.get('license')} {r.get('license_version') or ''}".strip(),
                        "creator": r.get("creator"), "page": r.get("foreign_landing_url"), "title": r.get("title")})
        if page >= (data.get("page_count") or 0):
            break
        time.sleep(3.2)  # tối đa 20 lượt/phút
    return out


def commons_candidates(category: str) -> list[dict]:
    out, cont = [], {}
    while True:
        params = {"action": "query", "format": "json", "generator": "categorymembers",
                  "gcmtitle": f"Category:{category}", "gcmtype": "file", "gcmlimit": 50,
                  "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": MAX_SIDE, **cont}
        data = get_json(COMMONS, params)
        if not data:
            break
        for page in (data.get("query", {}).get("pages") or {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            meta = info.get("extmetadata", {})
            out.append({"url": info.get("thumburl") or info.get("url"), "thumb": None, "source": "wikimedia-commons",
                        "license": (meta.get("LicenseShortName") or {}).get("value", ""),
                        "creator": (meta.get("Artist") or {}).get("value", "")[:200],
                        "page": info.get("descriptionurl"), "title": page.get("title")})
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(0.5)
    return out


def openimages_index(cache: Path) -> dict[str, set[str]]:
    """ImageID -> tập nhãn dương tính (đọc 16 đoạn song song của file nhãn 2,7 GB, chỉ giữ dòng cần)."""
    if not cache.exists():
        wanted = {f",{lab}," for lab in TOOL_LABELS | set(BACKGROUND_LABELS)}
        chunk = OI_SIZE // 16 + 1
        rows: list[str] = []

        def fetch(i: int) -> None:
            r = session.get(OI_LABELS, headers={"Range": f"bytes={i * chunk}-{(i + 1) * chunk - 1}"}, stream=True, timeout=120)
            local = [line for line in r.iter_lines(decode_unicode=True)
                     if line and any(w in line for w in wanted) and line.rsplit(",", 1)[-1].startswith("1")]
            with lock:
                rows.extend(local)

        log("   tải nhãn Open Images (16 luồng)…")
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(fetch, range(16)))
        cache.write_text("\n".join(rows), encoding="utf-8")
    index: dict[str, set[str]] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) >= 4 and parts[3].startswith("1"):
            index.setdefault(parts[0], set()).add(parts[2])
    return index


# ----------------------------------------------------------------------------- downloading
def download(item: dict, dest_dir: Path, seen: set[str]) -> dict | None:
    url = item.get("thumb") or item.get("url")
    if not url:
        return None
    try:
        r = session.get(url, timeout=30)
        if not r.ok and item.get("thumb") and item.get("url"):
            r = session.get(item["url"], timeout=30)
        r.raise_for_status()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content)))
        if min(img.size) < 120:
            return None
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        digest = hashlib.md5(buf.getvalue()).hexdigest()
        with lock:
            if digest[:16] in seen:
                return None
            seen.add(digest[:16])
        path = dest_dir / f"{digest[:16]}.jpg"
        path.write_bytes(buf.getvalue())
        return {**item, "file": str(path.relative_to(dest_dir.parent)), "class": dest_dir.name}
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_raw")
    ap.add_argument("--classes", nargs="*", default=list(CLASSES) + ["nen"])
    ap.add_argument("--sources", default="openverse,commons,openimages",
                    help="nguồn cần lấy, cách nhau bởi dấu phẩy")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    sources = set(args.sources.split(","))
    seen: set[str] = set()
    done_urls: set[str] = set()
    if manifest.exists():  # chạy tiếp: bỏ qua ảnh đã tải
        for line in manifest.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done_urls.add(rec.get("thumb") or rec.get("url"))
            seen.add(Path(rec["file"]).stem)
    index = openimages_index(out / "openimages_labels.csv")
    rng = random.Random(2026)

    for cls in args.classes:
        dest = out / cls
        dest.mkdir(exist_ok=True)
        items: list[dict] = []
        if cls == "nen" and "openimages" in sources:
            no_tool = [iid for iid, labs in index.items() if not labs & TOOL_LABELS]
            by_label: dict[str, list[str]] = {}
            for iid in no_tool:
                for lab in index[iid] & set(BACKGROUND_LABELS):
                    by_label.setdefault(lab, []).append(iid)
            chosen: set[str] = set()
            for lab, n in BACKGROUND_LABELS.items():
                pool = sorted(set(by_label.get(lab, [])) - chosen)
                chosen.update(rng.sample(pool, min(n, len(pool))))
            items = [{"url": OI_IMAGE.format(i), "source": "openimages", "license": "CC BY 2.0",
                      "creator": None, "page": f"https://storage.googleapis.com/openimages/web/visualizer/index.html?image={i}",
                      "title": i} for i in sorted(chosen)]
        elif cls in CLASSES:
            spec = CLASSES[cls]
            if "openverse" in sources:
                done_file = out / "openverse_done.txt"
                done_q = set(done_file.read_text(encoding="utf-8").splitlines()) if done_file.exists() else set()
                for query, pages in spec["openverse"].items():
                    if query in done_q:
                        continue
                    found = openverse_candidates(query, pages)
                    log(f"[{cls}] Openverse “{query}”: {len(found)}")
                    items += found
                    with done_file.open("a", encoding="utf-8") as fh:
                        fh.write(query + "\n")
            for cat in spec["commons"] if "commons" in sources else []:
                found = commons_candidates(cat)
                log(f"[{cls}] Commons “{cat}”: {len(found)}")
                items += found
            if "openimages" in sources:
                oi = [iid for iid, labs in index.items() if labs & set(spec["openimages"])]
                log(f"[{cls}] Open Images: {len(oi)}")
                items += [{"url": OI_IMAGE.format(i), "source": "openimages", "license": "CC BY 2.0", "creator": None,
                           "page": None, "title": i} for i in oi]
        # bỏ URL trùng
        uniq = [it for key, it in {(it.get("thumb") or it.get("url")): it for it in items if it.get("url")}.items()
                if key not in done_urls]
        with ThreadPoolExecutor(args.workers) as pool:
            results = [r for r in pool.map(lambda it: download(it, dest, seen), uniq) if r]
        with manifest.open("a", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"✓ [{cls}] tải được {len(results)}/{len(uniq)} ảnh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
