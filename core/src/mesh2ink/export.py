# SPDX-License-Identifier: AGPL-3.0-or-later
"""書き出しの段。レイヤーごとのPNG、PSD、合成、アイコンの一覧。"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

from .fills import Layer
from .imageio import write_png

ICON_SCALE = 2.2  # アイコンの正方形の辺を、顔の暗い描き込み（目のまわり）の広がりの何倍にするか
SHEET_GRAY = 235


def write_psd(path: Path, layers: list[Layer]) -> None:
    height, width = layers[0].rgba.shape[:2]
    psd = PSDImage.new("RGBA", (width, height))
    for layer in layers:
        psd.append(psd.create_pixel_layer(Image.fromarray(layer.rgba, "RGBA"), name=layer.name, top=0, left=0))
    with path.open("wb") as f:
        psd.save(f)


def read_psd_composite(path: Path) -> np.ndarray:
    """PSDを読み戻し、レイヤーを通常の合成で重ねた絵を返す。背景は透明で、透明な画素の色は0にそろえる。"""
    with path.open("rb") as f:
        rgba = np.array(PSDImage.open(f).composite(force=True).convert("RGBA"))
    rgba[rgba[..., 3] == 0] = 0
    return rgba


def icon_square(foreground: np.ndarray, dark: np.ndarray) -> tuple[int, int, int]:
    """アイコンに切り出す正方形（左、上、辺）。顔があれば顔の描き込みを中心に寄せ、無ければ写った物の全体を入れる。"""
    if dark.any():
        ys, xs = np.nonzero(dark)
        side = round(max(xs.max() - xs.min(), ys.max() - ys.min()) * ICON_SCALE)
    else:
        ys, xs = np.nonzero(foreground)
        side = int(max(xs.max() - xs.min(), ys.max() - ys.min())) + 1
    cx, cy = (int(xs.min()) + int(xs.max())) / 2, (int(ys.min()) + int(ys.max())) / 2
    return round(cx - side / 2), round(cy - side / 2), side


def icon_sheet(final: np.ndarray, square: tuple[int, int, int]) -> np.ndarray:
    """アイコンの見え方の一覧：400px、円、96px、48px。絵は白の上に置く。"""
    a = final[..., 3:].astype(np.float32) / 255
    on_white = np.round(final[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    left, top, side = square
    pad = side
    padded = cv2.copyMakeBorder(on_white, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    crop = padded[top + pad : top + pad + side, left + pad : left + pad + side]
    icon = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA)
    sheet = np.full((420, 956, 3), SHEET_GRAY, np.uint8)
    sheet[10:410, 10:410] = icon
    circle = np.zeros((400, 400), np.uint8)
    cv2.circle(circle, (200, 200), 200, 255, -1, cv2.LINE_AA)
    c = circle[..., None].astype(np.float32) / 255
    sheet[10:410, 420:820] = np.round(icon * c + SHEET_GRAY * (1 - c)).astype(np.uint8)
    sheet[10:58, 840:888] = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    sheet[70:166, 840:936] = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    return sheet


def write_outputs(folder: Path, layers: list[Layer], foreground: np.ndarray, dark: np.ndarray) -> list[str]:
    """レイヤーの組を folder に書き出す。layers は下から上の順。書いたレイヤーのファイル名を返す。"""
    (folder / "layers").mkdir()
    names = []
    for order, layer in enumerate(layers, 1):
        names.append(f"{order:02d}_{layer.name}.png")
        write_png(folder / "layers" / names[-1], layer.rgba)
    write_psd(folder / "illustration.psd", layers)
    # 合成した絵は、書いたPSDを読み戻して作る。PSDを開いた時の見え方と、final.png が食い違わない
    final = read_psd_composite(folder / "illustration.psd")
    write_png(folder / "final.png", final)
    write_png(folder / "icon_sheet.png", icon_sheet(final, icon_square(foreground, dark)))
    return names
