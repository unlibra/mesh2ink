# SPDX-License-Identifier: AGPL-3.0-or-later
"""下塗りの段。部位ごとに、色の画像から色を取って塗る。道具は色を決めない。

顔と瞳以外：その部位の色の中央値で平らに塗る。部位の中で色が大きく違うまとまりは分け、まとまりごとの中央値で塗る。
顔：線にした描き込みを、まわりの地の色で埋める。頬の赤みはテクスチャのものが残る。
瞳：テクスチャの色をそのまま使う。
"""
import re
from dataclasses import dataclass

import cv2
import numpy as np

from .lineart import LineArt
from .passes import Passes
from .roles import Roles, ellipse, px

SPLIT_DIFF = 0.20  # 部位の色からこれ以上離れた所を、別の色のまとまりの候補にする
SPLIT_AREA = 150  # 別の色のまとまりにする面積の下限（1600px基準）
SPLIT_DEPTH = 3  # まとまりの中をさらに分ける回数の上限


@dataclass(frozen=True)
class Layer:
    name: str
    rgba: np.ndarray  # 8ビットのRGBA


def fill_name(part_id: int, material: str) -> str:
    """レイヤーの名前。材質の名前からは、英数字と _ だけを残す。"""
    safe = re.sub(r"[^A-Za-z0-9_]", "", material)
    return f"fill_{part_id:02d}_{safe}" if safe else f"fill_{part_id:02d}"


def face_base(passes: Passes, lineart: LineArt, face: np.ndarray) -> np.ndarray:
    """顔の色から、線にした描き込みを消したもの。消した所は、まわりの顔の色をならして埋める。"""
    hole = cv2.dilate(lineart.drawn.astype(np.uint8), ellipse(px(passes, 5))) > 0
    valid = (face & ~hole).astype(np.float32)
    sigma = 12 * passes.scale
    weight = cv2.GaussianBlur(valid, (0, 0), sigma)
    spread = cv2.GaussianBlur(passes.rgb * valid[..., None], (0, 0), sigma) / np.maximum(weight, 1e-6)[..., None]
    spread[weight < 1e-3] = np.median(passes.rgb[face], 0)
    return np.where(hole[..., None], spread, passes.rgb)


def flat_colors(passes: Passes, mask: np.ndarray, out: np.ndarray, depth: int = 0) -> None:
    """mask の中を色の中央値で塗る。色が大きく違うまとまりは分け、同じやり方で塗り直す。"""
    base = np.median(passes.rgb[mask], 0)
    out[mask] = base
    if depth == SPLIT_DEPTH:
        return
    far = mask & (np.linalg.norm(passes.rgb - base, axis=2) > SPLIT_DIFF)
    far = cv2.morphologyEx(far.astype(np.uint8), cv2.MORPH_OPEN, ellipse(px(passes, 2)))
    count, components, stats, _ = cv2.connectedComponentsWithStats(far, connectivity=8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= SPLIT_AREA * passes.scale**2:
            flat_colors(passes, components == i, out, depth + 1)


def build_fills(passes: Passes, roles: Roles, lineart: LineArt) -> list[Layer]:
    layers = []
    flat = np.zeros_like(passes.rgb)
    for part in passes.parts:
        mask = passes.labels == part.id
        if part.id == roles.face:
            color = face_base(passes, lineart, mask)[mask]
        elif part.id == roles.eye:
            color = passes.rgb[mask]
        else:
            flat_colors(passes, mask, flat)
            color = flat[mask]
        rgba = np.zeros((passes.height, passes.width, 4), np.uint8)
        rgba[mask, :3] = np.round(np.clip(color, 0, 1) * 255).astype(np.uint8)
        rgba[mask, 3] = 255
        layers.append(Layer(name=fill_name(part.id, part.material), rgba=rgba))
    return layers


def line_layer(lineart: LineArt, line_color: tuple[float, float, float]) -> Layer:
    alpha = np.round(np.clip(lineart.alpha, 0, 1) * 255).astype(np.uint8)
    rgba = np.empty((*alpha.shape, 4), np.uint8)
    rgba[..., :3] = np.round(np.array(line_color) * 255).astype(np.uint8)
    rgba[..., 3] = alpha
    return Layer(name="line", rgba=rgba)
