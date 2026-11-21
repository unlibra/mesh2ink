# SPDX-License-Identifier: AGPL-3.0-or-later
"""画像の組（passes）を読む。橋渡しと本体の間の約束事は、このフォルダの形だけである。"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import Mesh2inkError
from .imageio import read_png

PASSES_VERSION = 1
FILES = ("color", "parts", "depth", "normal", "light", "line_contour", "line_material")
PART_COLOR_TOLERANCE = 8


@dataclass(frozen=True)
class Part:
    id: int
    material: str
    pixels: int


@dataclass(frozen=True)
class Passes:
    width: int
    height: int
    parts: tuple[Part, ...]  # 写っている部位だけ。番号の順
    labels: np.ndarray  # 画素ごとの部位の番号。背景は0
    rgb: np.ndarray  # 陰影なしの色。0〜1
    line: np.ndarray  # 形の線の濃さ。0〜1。輪郭と部位の境目を合わせたもの

    @property
    def scale(self) -> float:
        """画素で決めた値を、1600pxを基準に画像の大きさへ合わせる倍率。"""
        return max(self.width, self.height) / 1600


def load_passes(folder: Path) -> Passes:
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Mesh2inkError("画像の組の meta.json を読めません") from None
    if not isinstance(meta, dict) or meta.get("version") != PASSES_VERSION:
        version = meta.get("version") if isinstance(meta, dict) else None
        raise Mesh2inkError(f"画像の組の形式の版が {version} です。対応している版は {PASSES_VERSION} です")
    for name in FILES:
        if not (folder / f"{name}.png").is_file():
            raise Mesh2inkError(f"画像の組に {name}.png がありません")
    width, height = meta["width"], meta["height"]
    parts_im = read_png(folder / "parts.png")
    color_im = read_png(folder / "color.png")
    contour = read_png(folder / "line_contour.png")
    material = read_png(folder / "line_material.png")
    for name, im in (("parts", parts_im), ("color", color_im), ("line_contour", contour), ("line_material", material)):
        if im.shape[:2] != (height, width):
            raise Mesh2inkError(f"画像の組の {name}.png の大きさが meta.json と合いません")

    foreground = parts_im[..., 3] > 127
    labels = np.zeros((height, width), np.uint8)
    parts = []
    for entry in meta["parts"]:
        near = np.abs(parts_im[..., :3].astype(np.int16) - np.array(entry["color"], np.int16)).max(2) <= PART_COLOR_TOLERANCE
        mask = foreground & near
        labels[mask] = entry["id"]
        if mask.any():
            parts.append(Part(id=entry["id"], material=entry["material"], pixels=int(mask.sum())))
    if not foreground.any():
        raise Mesh2inkError("カメラに写る物がありません。カメラの向きと、物の表示を確かめてください")
    if (foreground & (labels == 0)).any():
        raise Mesh2inkError("画像の組の parts.png に、部位の表に無い色があります")
    rgb = color_im[..., :3].astype(np.float32) / 255
    line = 1 - np.minimum(contour, material).astype(np.float32) / 255
    return Passes(width=width, height=height, parts=tuple(parts), labels=labels, rgb=rgb, line=line)
