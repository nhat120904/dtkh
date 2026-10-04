"""Lọc nhiễu bộ ảnh thu thập được bằng mô hình CLIP (open_clip) chạy trên máy.

CLIP chỉ dùng ở bước chuẩn bị dữ liệu này – ứng dụng KHÔNG cần CLIP.
Cài thêm:  pip install torch open_clip_torch

Với mỗi ảnh, CLIP so sánh với mô tả của từng lớp dụng cụ và với các mô tả "loại bỏ"
(tranh vẽ, logo, đồ chơi, dụng cụ khác…). Ảnh được giữ khi CLIP đồng ý với lớp
(xác suất >= ngưỡng); ảnh tải nhầm lớp nhưng CLIP rất chắc chắn thì được chuyển sang lớp đúng.

Chạy:  python scripts/clean_dataset.py --raw dataset_raw --out dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import open_clip
import torch
from PIL import Image

TOOL_PROMPTS = {
    "bua": ["a hammer", "a claw hammer", "a ball-peen hammer", "a steel hammer with a wooden handle"],
    "kim": ["pliers", "a pair of pliers", "needle-nose pliers", "combination pliers with red handles"],
    "tua_vit": ["a screwdriver", "a phillips screwdriver", "a flathead screwdriver"],
    "co_le": ["a combination wrench", "an open-end spanner", "a ring spanner", "a double open-end wrench"],
    "mo_let": ["an adjustable wrench", "an adjustable spanner", "a crescent wrench with a worm screw"],
    "cua_tay": ["a hacksaw", "a hand saw", "a metal hacksaw with a blade", "a wood saw"],
    "giua": ["a metal file tool", "a hand file for metal", "a rasp", "a set of needle files"],
}
BACKGROUND_PROMPTS = [
    "a person", "a person's face", "a human hand", "an empty hand", "a room", "a classroom", "a desk",
    "a table", "a laptop on a desk", "a book", "a mobile phone", "a wall", "students in a classroom",
]
REJECT_PROMPTS = [
    "a drawing", "a cartoon", "a logo", "an icon", "clip art", "a diagram", "a painting", "text",
    "a toy", "a keychain", "a multitool", "a pocket knife", "a toolbox full of tools", "many different tools",
    "a chisel", "an electric drill", "a chainsaw", "a jackhammer", "a vise", "a tape measure", "a ruler",
    "a socket wrench set", "a nail file for manicure", "a gavel", "a sculpture", "a building",
    "a pipe wrench", "a torque wrench", "a knife", "an axe",
    "a circular saw", "a power saw", "a miter saw", "a table saw", "a band saw", "a sawmill",
    "screwdriver bits", "nails", "screws", "a set of wrenches", "a cheese grater", "a plant",
    "a movie poster", "a book cover", "a machine", "a crowd of people",
]
# Các mô tả "loại bỏ" là dụng cụ / đồ vật giống dụng cụ – ảnh nền không được chứa chúng
TOOLISH_REJECTS = {
    "a toy", "a keychain", "a multitool", "a pocket knife", "a toolbox full of tools", "many different tools",
    "a chisel", "an electric drill", "a chainsaw", "a jackhammer", "a vise", "a tape measure", "a ruler",
    "a socket wrench set", "a nail file for manicure", "a gavel", "a pipe wrench", "a torque wrench", "a knife", "an axe",
    "a circular saw", "a power saw", "a miter saw", "a table saw", "a band saw", "screwdriver bits", "nails", "screws",
    "a set of wrenches", "a machine",
}
TEMPLATES = ["a photo of {}.", "a close-up photo of {}.", "a photo of {} on a table.", "a photo of a person holding {}."]

KEEP_PROB = 0.45      # xác suất tối thiểu (trên tổng tất cả mô tả) để giữ ảnh ở lớp của nó
KEEP_PROB_CLASS = {"giua": 0.6, "cua_tay": 0.6}  # lớp có nguồn ảnh nhiễu hơn -> ngưỡng chặt hơn
MOVE_PROB = 0.75      # ảnh của lớp khác được chuyển nếu CLIP rất chắc chắn
NEN_MAX_TOOL = 0.10   # ảnh nền bị loại nếu tổng xác suất "dụng cụ" vượt mức này
NEN_MAX_IMAGES = 500  # giới hạn số ảnh nền để cân bằng với các lớp dụng cụ
OUT_SIDE = 384


def build_text_matrix(model, tokenizer, device):
    names, groups = [], []
    for cls, prompts in TOOL_PROMPTS.items():
        names.append(cls)
        groups.append(prompts)
    names.append("nen")
    groups.append(BACKGROUND_PROMPTS)
    for p in REJECT_PROMPTS:
        names.append(f"reject:{p}")
        groups.append([p])
    embs = []
    with torch.no_grad():
        for prompts in groups:
            texts = [t.format(p) for p in prompts for t in TEMPLATES]
            e = model.encode_text(tokenizer(texts).to(device))
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            embs.append(e / e.norm())
    return names, torch.stack(embs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset_raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--overrides", default=str(Path(__file__).with_name("dataset_overrides.json")),
                    help="file JSON kết quả duyệt tay: {force: {file: lớp}, drop: [file]}")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    names, text = build_text_matrix(model, tokenizer, device)
    scale = model.logit_scale.exp().item()

    manifest = {}
    for line in (raw / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        manifest[rec["file"]] = rec

    files = sorted(manifest)
    decisions = []
    batch = 64
    for start in range(0, len(files), batch):
        chunk = files[start:start + batch]
        imgs = []
        for f in chunk:
            try:
                imgs.append(preprocess(Image.open(raw / f).convert("RGB")))
            except Exception:
                imgs.append(None)
        valid = [i for i, t in enumerate(imgs) if t is not None]
        if not valid:
            continue
        with torch.no_grad():
            feats = model.encode_image(torch.stack([imgs[i] for i in valid]).to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (scale * feats @ text.T).softmax(dim=-1).cpu().numpy()
        for row, i in enumerate(valid):
            f = chunk[i]
            src = manifest[f]["class"]
            p = probs[row]
            top = int(p.argmax())
            tool_sum = float(sum(p[names.index(c)] for c in TOOL_PROMPTS))
            if src == "nen":
                toolish = sum(float(p[names.index(f"reject:{r}")]) for r in TOOLISH_REJECTS)
                target = "nen" if tool_sum + toolish <= NEN_MAX_TOOL else None
            elif names[top] == src and p[top] >= KEEP_PROB_CLASS.get(src, KEEP_PROB):
                target = src
            elif names[top] in TOOL_PROMPTS and p[top] >= max(MOVE_PROB, KEEP_PROB_CLASS.get(names[top], 0)):
                target = names[top]  # tải nhầm lớp -> chuyển
            else:
                target = None
            decisions.append({"file": f, "src": src, "target": target, "top": names[top], "p_top": round(float(p[top]), 3)})
        print(f"   {min(start + batch, len(files))}/{len(files)}", flush=True)

    overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8")) if Path(args.overrides).exists() else {}
    for d in decisions:
        if d["file"] in overrides.get("drop", []):
            d["target"] = None
        if d["file"] in overrides.get("force", {}):
            d["target"] = overrides["force"][d["file"]]

    nen = [d for d in decisions if d["target"] == "nen"]
    if len(nen) > NEN_MAX_IMAGES:
        import random
        random.Random(0).shuffle(nen)
        for d in nen[NEN_MAX_IMAGES:]:
            d["target"] = None

    if out.exists():
        for d in out.iterdir():
            if d.is_dir() and d.name in list(TOOL_PROMPTS) + ["nen"]:
                shutil.rmtree(d)
    out.mkdir(parents=True, exist_ok=True)
    kept = Counter()
    with (out / "CREDITS.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "class", "source", "license", "creator", "page"])
        for d in decisions:
            if not d["target"]:
                continue
            rec = manifest[d["file"]]
            dest = out / d["target"] / Path(d["file"]).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(raw / d["file"]).convert("RGB")
            img.thumbnail((OUT_SIDE, OUT_SIDE), Image.Resampling.LANCZOS)
            img.save(dest, "JPEG", quality=88)
            kept[d["target"]] += 1
            writer.writerow([str(dest.relative_to(out)), d["target"], rec.get("source"), rec.get("license"),
                             (rec.get("creator") or "")[:120], rec.get("page")])
    (out / "clean_report.json").write_text(json.dumps({
        "kept": kept, "total_raw": len(files),
        "moved": Counter(f"{d['src']}->{d['target']}" for d in decisions if d["target"] and d["target"] != d["src"]),
        "dropped_by_class": Counter(d["src"] for d in decisions if not d["target"]),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "clean_decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions), encoding="utf-8")
    print("✓ Giữ lại:", dict(kept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
