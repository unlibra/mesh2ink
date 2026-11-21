# SPDX-License-Identifier: AGPL-3.0-or-later
"""部位の役割（顔、瞳）を決める。設定で指定が無ければ、部位の画像と色の画像から決める。"""
from dataclasses import dataclass

import cv2
import numpy as np

from .errors import Mesh2inkError
from .passes import Passes

DARK_RATIO = 0.66  # 部位の地の明るさに対して、これより暗い所を描き込みとみなす
EYE_MAX_AREA = 0.05  # 瞳の部位は、写っている物の面積のこの割合より小さい
EYE_ENCLOSED = 0.5  # 瞳の部位は、まわりのこの割合以上を1つの部位（顔）に囲まれている。残りは前髪などが重なる
EYE_OPEN = 0.1  # 瞳の部位のまわりで、背景に接してよい割合の上限
PAIR_RATIO = 0.2  # 瞳の2つのかたまりとみなす、大きいほうに対する面積の割合の下限
FACE_MIN_BASE = 0.5  # 顔の部位の地の明るさの下限。暗い地に明るい模様のある部位を、顔に選ばないため
FACE_MIN_DARK = 0.0002  # 顔の部位は、画像の画素の数のこの割合以上の暗い描き込みを持つ
FACE_MARGIN = 2.0  # 顔の部位は、次点のこの倍以上の暗い描き込みを持つ


@dataclass(frozen=True)
class Roles:
    face: int | None  # 部位の番号。無ければ None
    eye: int | None

    def name(self, part_id: int) -> str:
        return "face" if part_id == self.face else "eye" if part_id == self.eye else "other"


def ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def px(passes: Passes, value: float) -> int:
    """1600pxを基準に決めた画素の数を、画像の大きさに合わせる。1より小さくはしない。"""
    return max(1, round(value * passes.scale))


def dark_pixels(passes: Passes, part_id: int) -> int:
    """部位の中で、地の明るさより十分に暗い画素の数。地が暗い部位は0とする。"""
    value = passes.rgb.max(2)[passes.labels == part_id]
    base = float(np.median(value))
    if base < FACE_MIN_BASE:
        return 0
    return int((value < base * DARK_RATIO).sum())


def enclosing_part(passes: Passes, part_id: int) -> int | None:
    """部位が小さく、背景にほぼ接さず、まわりの半分以上が1つの部位なら、その部位の番号を返す。"""
    mask = passes.labels == part_id
    foreground = int((passes.labels > 0).sum())
    if mask.sum() > foreground * EYE_MAX_AREA:
        return None
    ring = (cv2.dilate(mask.astype(np.uint8), ellipse(px(passes, 3))) > 0) & ~mask
    around = np.bincount(passes.labels[ring], minlength=256)
    around[part_id] = 0
    best = int(around[1:].argmax()) + 1
    total = int(ring.sum())
    return best if around[best] >= total * EYE_ENCLOSED and around[0] <= total * EYE_OPEN else None


def is_pair(passes: Passes, part_id: int) -> bool:
    """部位が、同じくらいの大きさの2つのかたまり（左右の瞳）でできているか。"""
    count, _, stats, _ = cv2.connectedComponentsWithStats((passes.labels == part_id).astype(np.uint8), connectivity=8)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), reverse=True)
    return len([a for a in areas if a >= areas[0] * PAIR_RATIO]) == 2


def _lookup(passes: Passes, material: str, key: str) -> int:
    for part in passes.parts:
        if part.material == material:
            return part.id
    names = "、".join(p.material for p in passes.parts)
    raise Mesh2inkError(f"設定の core.roles.{key} の材質「{material}」が、写っている部位にありません。写っている材質：{names}")


def resolve_roles(passes: Passes, face: str, eye: str) -> Roles:
    """設定の値（"auto"、"none"、材質の名前）から、顔と瞳の部位を決める。"""
    names = "、".join(p.material for p in passes.parts)
    face_id = None if face in ("auto", "none") else _lookup(passes, face, "face")
    if face == "none":
        if eye not in ("auto", "none"):
            raise Mesh2inkError('設定の core.roles.face が "none" の時は、core.roles.eye も "none" か "auto" にしてください')
        return Roles(face=None, eye=None)

    eye_id = None
    if eye == "auto":
        enclosed = {p.id: enclosing_part(passes, p.id) for p in passes.parts}
        candidates = [i for i, outer in enclosed.items() if outer is not None and face_id in (None, outer) and i != face_id]
        if len(candidates) > 1:
            # 首や飾りのような、ほかの小さい部位と区別する
            candidates = [i for i in candidates if is_pair(passes, i)] or candidates
        if len(candidates) > 1:
            found = "、".join(p.material for p in passes.parts if p.id in candidates)
            raise Mesh2inkError(
                f'瞳の部位を自動で決められません（候補：{found}）。設定の core.roles.eye に、材質の名前か "none" を書いてください'
            )
        if candidates:
            eye_id = candidates[0]
            face_id = enclosed[eye_id]
    elif eye != "none":
        eye_id = _lookup(passes, eye, "eye")
        if face == "auto":
            face_id = enclosing_part(passes, eye_id)

    if face_id is None:
        scores = sorted(((dark_pixels(passes, p.id), p.id) for p in passes.parts if p.id != eye_id), reverse=True)
        best, best_id = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0
        if best < passes.width * passes.height * FACE_MIN_DARK or best < second * FACE_MARGIN:
            raise Mesh2inkError(
                "顔の部位を自動で決められません。設定の core.roles.face に、顔の材質の名前を書いてください。"
                f'顔の無いものを描く時は "none" と書いてください。写っている材質：{names}'
            )
        face_id = best_id
    if eye_id == face_id:
        raise Mesh2inkError("設定の core.roles の face と eye に、同じ材質が指定されています")
    return Roles(face=face_id, eye=eye_id)
