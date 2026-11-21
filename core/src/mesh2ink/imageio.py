# SPDX-License-Identifier: AGPL-3.0-or-later
"""画像の読み書きを1か所にまとめる。日本語を含むパスで動くよう、ファイルはバイト列で扱う。"""
from pathlib import Path

import cv2
import numpy as np

from .errors import Mesh2inkError


def read_png(path: Path) -> np.ndarray:
    """PNGを読む。色つきの画像は RGB(A) の順で返す。ビットの深さは元のまま。"""
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        raise Mesh2inkError(f"画像を読めません：{path.name}") from None
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise Mesh2inkError(f"画像を読めません：{path.name}")
    if im.ndim == 3:
        im = im[..., [2, 1, 0, 3]] if im.shape[2] == 4 else im[..., ::-1]
    return im


def write_png(path: Path, im: np.ndarray) -> None:
    """8ビットの RGB(A) かグレーの画像をPNGで書く。メタデータは入らない。"""
    if im.ndim == 3:
        im = im[..., [2, 1, 0, 3]] if im.shape[2] == 4 else im[..., ::-1]
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(im))
    if not ok:
        raise RuntimeError(f"PNGにできません：{path.name}")
    buf.tofile(path)
