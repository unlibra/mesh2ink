# SPDX-License-Identifier: AGPL-3.0-or-later
"""線の段と描き込みの段。形の線に、色の画像から取った顔の描き込みを足して、線画を1枚にする。

形の線：橋渡しがLine Artで作った線。目のまわりでは消し、描き込みを優先する。
顔の描き込み：テクスチャに描かれたまつ毛、目の縁、瞳、眉、口。色の画像から取る。
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .passes import Passes
from .roles import DARK_RATIO, Roles, ellipse, px

COLOR_DIFF = 0.12  # 顔の地の色からこれ以上離れた所を、描き込みの候補にする
THIN_WIDTH = 14  # これより幅の狭い所を、細い描き込みとみなす（1600px基準の画素）
THIN_AREA = (40, 6000)  # 細い描き込みとみなす面積の下限と上限（1600px基準）


@dataclass(frozen=True)
class LineArt:
    alpha: np.ndarray  # 線の濃さ。0〜1
    drawn: np.ndarray  # 顔の部位の中で、線にした描き込みの画素。下塗りで地の色に埋める
    dark: np.ndarray  # 顔と瞳の中の暗い描き込み。アイコンの切り出しの基準にも使う


def build_lineart(passes: Passes, roles: Roles) -> LineArt:
    shape = passes.line
    none = np.zeros(passes.labels.shape, bool)
    if roles.face is None:
        return LineArt(alpha=shape, drawn=none, dark=none)

    rgb = passes.rgb
    face = passes.labels == roles.face
    eye = passes.labels == roles.eye if roles.eye is not None else none
    value = rgb.max(2)
    base_value = float(np.median(value[face]))
    base_color = np.median(rgb[face], 0)
    diff = np.linalg.norm(rgb - base_color, axis=2)

    # 暗い部分（まつ毛、目の縁、瞳孔）はそのまま塗りつぶす
    dark = (face | eye) & (value < base_value * DARK_RATIO)
    # 地の色と違う細い部分（眉、口）。幅の広い部分（頬の赤み、瞳）は、明るさによらず入れない
    mid = (face & ~dark & (diff > COLOR_DIFF)).astype(np.uint8)
    mid = cv2.morphologyEx(mid, cv2.MORPH_OPEN, ellipse(px(passes, 1)))
    narrow = mid & ~cv2.morphologyEx(mid, cv2.MORPH_OPEN, ellipse(px(passes, THIN_WIDTH / 2)))
    count, components, stats, _ = cv2.connectedComponentsWithStats(narrow, connectivity=8)
    low, high = (a * passes.scale**2 for a in THIN_AREA)
    thin_ids = [i for i in range(1, count) if low <= stats[i, cv2.CC_STAT_AREA] <= high]
    thin_blobs = np.isin(components, thin_ids)
    # 眉や口は、塗りつぶさず芯の線にする。モデルでは薄い色の細い形で、塗ると太く見えすぎる
    thin = cv2.dilate(cv2.ximgproc.thinning(thin_blobs.astype(np.uint8) * 255), ellipse(px(passes, 1))) > 0

    drawing = dark | thin
    if roles.eye is not None:
        eye8 = eye.astype(np.uint8)
        # 瞳の中の色の境目（虹彩の濃淡、ハイライトの形）と、瞳の外周
        gray = cv2.GaussianBlur((rgb * 255).astype(np.uint8), (0, 0), 1.2 * max(passes.scale, 0.5))
        edges = np.zeros(passes.labels.shape, np.uint8)
        for ch in range(3):
            edges |= cv2.Canny(gray[..., ch], 30, 70)
        drawing |= (cv2.dilate(edges, ellipse(1)) > 0) & (cv2.erode(eye8, ellipse(px(passes, 2))) > 0)
        drawing |= cv2.morphologyEx(eye8, cv2.MORPH_GRADIENT, ellipse(1)) > 0

    # 目の中と目のまわりは、形の線（眼球の丸み、まつ毛やまぶたのメッシュの縁）を使わない
    zone8 = (face | eye).astype(np.uint8)
    suppress = cv2.dilate(eye.astype(np.uint8), ellipse(px(passes, 4))) > 0
    suppress |= (cv2.dilate((dark | eye).astype(np.uint8), ellipse(px(passes, 16))) > 0) & (cv2.erode(zone8, ellipse(px(passes, 6))) > 0)
    drawn_alpha = cv2.GaussianBlur(drawing.astype(np.float32), (0, 0), 0.7)
    alpha = np.maximum(np.where(suppress, 0.0, shape), drawn_alpha).astype(np.float32)
    return LineArt(alpha=alpha, drawn=face & (dark | thin_blobs), dark=dark)
