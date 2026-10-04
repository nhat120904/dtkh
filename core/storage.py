"""Lưu hồ sơ học sinh ra file JSON trên máy (không cần cơ sở dữ liệu, không cần mạng)."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from core.config import PROFILES_DIR
from core.knowledge import normalize_text
from core.progress import ensure_schema, new_profile, now_iso

MAX_NAME_LENGTH = 40


def slugify(name: str) -> str:
    return normalize_text(name).replace(" ", "-")[:MAX_NAME_LENGTH] or "hoc-sinh"


def clean_name(name: str) -> str:
    return " ".join(str(name).split())[:MAX_NAME_LENGTH]


class ProfileStore:
    def __init__(self, base_dir: Path = PROFILES_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, profile_id: str) -> Path:
        return self.base_dir / f"{slugify(profile_id)}.json"

    def exists(self, profile_id: str) -> bool:
        return self._path(profile_id).is_file()

    def list(self) -> list[dict]:
        """Danh sách hồ sơ {id, name, xp, updated_at}, mới dùng gần nhất trước."""
        items = []
        for path in self.base_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                items.append({"id": path.stem, "name": data.get("name", path.stem),
                              "xp": data.get("xp", 0), "updated_at": data.get("updated_at", "")})
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda d: d["updated_at"], reverse=True)

    def load(self, profile_id: str) -> dict | None:
        path = self._path(profile_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Giữ lại file hỏng để giáo viên có thể kiểm tra, không ghi đè
            path.rename(path.with_suffix(f".corrupt-{datetime.now():%Y%m%d%H%M%S}.json.bak"))
            return None
        data["id"] = path.stem
        return ensure_schema(data)

    def save(self, profile: dict) -> None:
        profile["updated_at"] = now_iso()
        path = self._path(profile["id"])
        fd, tmp = tempfile.mkstemp(dir=self.base_dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(profile, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, path)  # ghi nguyên tử: không bao giờ để file ghi dở
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _unique_id(self, name: str) -> str:
        base = slugify(name)
        candidate, i = base, 2
        while self.exists(candidate):
            candidate, i = f"{base}-{i}", i + 1
        return candidate

    def create(self, name: str) -> dict:
        name = clean_name(name)
        if not name:
            raise ValueError("Tên không được để trống")
        profile = new_profile(name, self._unique_id(name))
        self.save(profile)
        return profile

    def import_profile(self, data: dict, name: str | None = None) -> dict:
        """Nhập hồ sơ (vd hồ sơ mẫu) thành một hồ sơ mới, không ghi đè hồ sơ có sẵn."""
        profile = json.loads(json.dumps(data))  # bản sao sâu
        profile["name"] = clean_name(name or profile.get("name", "Học sinh"))
        profile["id"] = self._unique_id(profile["name"])
        profile = ensure_schema(profile)
        self.save(profile)
        return profile

    def delete(self, profile_id: str) -> None:
        self._path(profile_id).unlink(missing_ok=True)
