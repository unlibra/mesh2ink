# SPDX-License-Identifier: AGPL-3.0-or-later
"""設定（preset.json）の読み書きと検証。外から入る設定は、ここで1回だけ検証する。"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import Mesh2inkError

PRESET_VERSION = 1
SIZE_RANGE = (256, 4096)


@dataclass(frozen=True)
class Preset:
    blender: str | None
    size: int
    subdivision: int
    line_width: float
    line_smooth: int
    face: str
    eye: str
    line_color: tuple[float, float, float]
    seed: int

    def bridge_config(self) -> dict[str, Any]:
        return {"size": self.size, "subdivision": self.subdivision, "line_width": self.line_width, "line_smooth": self.line_smooth}

    def to_json(self) -> dict[str, Any]:
        """書き出す設定。Blenderの場所は機械ごとに違うので入れない。"""
        return {
            "version": PRESET_VERSION,
            "bridge": self.bridge_config(),
            "core": {"roles": {"face": self.face, "eye": self.eye}, "line_color": list(self.line_color), "seed": self.seed},
        }


DEFAULT = Preset(
    blender=None, size=1600, subdivision=2, line_width=5, line_smooth=3, face="auto", eye="auto", line_color=(0.30, 0.18, 0.20), seed=0
)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _section(data: Any, name: str, required: set[str], optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    """区画が辞書で、項目が過不足なくそろっていることを確かめる。"""
    if not isinstance(data, dict):
        raise Mesh2inkError(f"設定の {name} の型が違います。項目の組にしてください")
    prefix = f"{name}." if name != "preset" else ""
    for key in data:
        if key not in required and key not in optional:
            raise Mesh2inkError(f"設定に知らない項目があります：{prefix}{key}")
    for key in sorted(required):
        if key not in data:
            raise Mesh2inkError(f"設定に項目が足りません：{prefix}{key}")
    return data


def _check(ok: bool, name: str, expect: str) -> None:
    if not ok:
        raise Mesh2inkError(f"設定の値の型か範囲が違います：{name}（{expect}）")


def parse_preset(data: Any) -> Preset:
    top = _section(data, "preset", {"version", "bridge", "core"}, frozenset({"blender"}))
    if top["version"] != PRESET_VERSION:
        raise Mesh2inkError(f"設定の形式の版が {top['version']} です。対応している版は {PRESET_VERSION} です")
    blender = top.get("blender")
    _check(blender is None or isinstance(blender, str), "blender", "文字列か null")
    bridge = _section(top["bridge"], "bridge", {"size", "subdivision", "line_width", "line_smooth"})
    core = _section(top["core"], "core", {"roles", "line_color", "seed"})
    roles = _section(core["roles"], "core.roles", {"face", "eye"})

    size = bridge["size"]
    if not _is_int(size) or not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise Mesh2inkError(f"設定の bridge.size が範囲の外です。{SIZE_RANGE[0]}〜{SIZE_RANGE[1]} の整数にしてください")
    _check(_is_int(bridge["subdivision"]) and 0 <= bridge["subdivision"] <= 4, "bridge.subdivision", "0〜4の整数")
    _check(_is_number(bridge["line_width"]) and 0 < bridge["line_width"] <= 64, "bridge.line_width", "0より大きく64以下の数")
    _check(_is_int(bridge["line_smooth"]) and 0 <= bridge["line_smooth"] <= 32, "bridge.line_smooth", "0〜32の整数")
    for key in ("face", "eye"):
        _check(isinstance(roles[key], str) and roles[key] != "", f"core.roles.{key}", '"auto"、"none"、材質の名前のどれか')
    color = core["line_color"]
    _check(
        isinstance(color, list) and len(color) == 3 and all(_is_number(c) and 0 <= c <= 1 for c in color),
        "core.line_color",
        "0〜1の数が3つ",
    )
    _check(_is_int(core["seed"]) and core["seed"] >= 0, "core.seed", "0以上の整数")
    return Preset(
        blender=blender,
        size=size,
        subdivision=bridge["subdivision"],
        line_width=bridge["line_width"],
        line_smooth=bridge["line_smooth"],
        face=roles["face"],
        eye=roles["eye"],
        line_color=(float(color[0]), float(color[1]), float(color[2])),
        seed=core["seed"],
    )


def load_preset(path: Path) -> Preset:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Mesh2inkError(f"設定のJSONを読めません：{path}") from None
    return parse_preset(data)
