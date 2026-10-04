"""Kho kiến thức dụng cụ cơ khí: nạp dữ liệu, tra cứu, khớp nhãn model với dụng cụ."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.config import TOOLS_PATH

# Nhãn mà người dùng hay đặt cho lớp "không có dụng cụ" trong Teachable Machine
BACKGROUND_ALIASES = {
    "background", "nen", "phong nen", "khac", "other", "others", "none", "nothing",
    "khong co", "khong co dung cu", "trong", "empty",
}


def normalize_text(text: str) -> str:
    """Bỏ dấu tiếng Việt, chữ thường, chỉ giữ chữ-số và khoảng trắng.

    >>> normalize_text("  Cờ_lê  ")
    'co le'
    """
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


@dataclass(frozen=True)
class ToolGroup:
    id: str
    name: str
    short: str
    icon: str
    color: str
    description: str


@dataclass(frozen=True)
class Tool:
    id: str
    name: str
    english: str
    emoji: str
    group: str
    aliases: tuple[str, ...]
    summary: str
    description: str
    structure: tuple[str, ...]
    uses: tuple[str, ...]
    how_to_use: tuple[str, ...]
    safety: tuple[str, ...]
    key_safety: str | None
    fun_fact: str

    @property
    def summary_sentence(self) -> str:
        """'dùng để ...' -> 'Dùng để ...'"""
        return self.summary[:1].upper() + self.summary[1:]


class KnowledgeBase:
    def __init__(self, tools: list[Tool], groups: list[ToolGroup]):
        self._tools = {t.id: t for t in tools}
        self._groups = {g.id: g for g in groups}
        unknown = {t.group for t in tools} - set(self._groups)
        if unknown:
            raise ValueError(f"Dụng cụ tham chiếu nhóm không tồn tại: {unknown}")
        # alias đã chuẩn hóa -> tool_id, sắp alias dài trước để khớp chính xác hơn
        alias_map: dict[str, str] = {}
        for t in tools:
            for alias in (t.id.replace("_", " "), t.name, t.english, *t.aliases):
                key = normalize_text(alias)
                if key:
                    alias_map.setdefault(key, t.id)
        self._alias_map = alias_map
        self._aliases_by_len = sorted(alias_map, key=len, reverse=True)

    # ----------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Path = TOOLS_PATH) -> "KnowledgeBase":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        groups = [ToolGroup(**g) for g in raw["groups"]]
        tools = []
        for t in raw["tools"]:
            tools.append(Tool(
                id=t["id"], name=t["name"], english=t.get("english", ""), emoji=t.get("emoji", "🛠️"),
                group=t["group"], aliases=tuple(t.get("aliases", [])), summary=t["summary"],
                description=t.get("description", ""), structure=tuple(t.get("structure", [])),
                uses=tuple(t.get("uses", [])), how_to_use=tuple(t.get("how_to_use", [])),
                safety=tuple(t.get("safety", [])), key_safety=t.get("key_safety"),
                fun_fact=t.get("fun_fact", ""),
            ))
        return cls(tools, groups)

    # ----------------------------------------------------------------- lookups
    def __len__(self) -> int:
        return len(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def ids(self) -> list[str]:
        return list(self._tools)

    def get(self, tool_id: str | None) -> Tool | None:
        return self._tools.get(tool_id) if tool_id else None

    def groups(self) -> list[ToolGroup]:
        return list(self._groups.values())

    def group(self, group_id: str) -> ToolGroup:
        return self._groups[group_id]

    def by_group(self, group_id: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.group == group_id]

    def search(self, query: str) -> list[Tool]:
        """Tìm không phân biệt dấu trong tên, tên tiếng Anh, bí danh, công dụng, mô tả."""
        q = normalize_text(query)
        if not q:
            return self.all()
        words = q.split()
        results = []
        for t in self._tools.values():
            haystack = normalize_text(" ".join([t.name, t.english, *t.aliases, t.summary, t.description, *t.uses]))
            if all(w in haystack for w in words):
                results.append(t)
        return results

    # ----------------------------------------------------------------- label matching
    @staticmethod
    def is_background_label(label: str) -> bool:
        return normalize_text(label) in BACKGROUND_ALIASES

    def match_label(self, label: str) -> Tool | None:
        """Khớp nhãn của model (vd 'Cờ lê', 'co_le', 'Wrench', 'Bua dinh') với một dụng cụ."""
        key = normalize_text(label)
        if not key or key in BACKGROUND_ALIASES:
            return None
        if key in self._alias_map:
            return self._tools[self._alias_map[key]]
        padded = f" {key} "
        for alias in self._aliases_by_len:
            if f" {alias} " in padded:
                return self._tools[self._alias_map[alias]]
        return None


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.load()
