# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from mesh2ink.imageio import write_png

FIXTURE = Path(__file__).parent / "fixtures" / "testscene"
REPO = Path(__file__).resolve().parents[2]

SKIN = (255, 222, 204)
PART_COLORS = {1: (64, 0, 0), 2: (128, 0, 0), 3: (191, 0, 0), 4: (255, 0, 0)}


def write_passes(folder: Path, labels: np.ndarray, rgb: np.ndarray, materials: dict[int, str], line: np.ndarray | None = None) -> Path:
    """部位の番号の画像と色の画像から、画像の組を作る。line は線のある画素（真偽）。"""
    folder.mkdir(parents=True)
    h, w = labels.shape
    parts = np.zeros((h, w, 4), np.uint8)
    for part_id in materials:
        parts[labels == part_id, :3] = PART_COLORS[part_id]
    parts[labels > 0, 3] = 255
    color = np.dstack([rgb, np.where(labels > 0, 255, 0).astype(np.uint8)])
    line_im = np.where(line, 0, 255).astype(np.uint8) if line is not None else np.full((h, w), 255, np.uint8)
    write_png(folder / "parts.png", parts)
    write_png(folder / "color.png", color)
    write_png(folder / "line_contour.png", line_im)
    write_png(folder / "line_material.png", np.full((h, w), 255, np.uint8))
    for name in ("depth", "normal", "light"):
        write_png(folder / f"{name}.png", np.zeros((h, w, 4), np.uint8))
    meta = {
        "version": 1,
        "width": w,
        "height": h,
        "projection": "PERSP",
        "pixel_size": 0.001,
        "depth_near": 1.0,
        "depth_far": 2.0,
        "parts": [{"id": i, "color": list(PART_COLORS[i]), "material": m} for i, m in materials.items()],
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return folder


def face_scene(size: int = 400) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """顔（肌の色の円）、左右の瞳、髪、服でできた絵。顔には、まつ毛（暗い太い線）と口（細い線）がある。"""
    labels = np.zeros((size, size), np.uint8)
    rgb = np.zeros((size, size, 3), np.uint8)
    cv2.rectangle(labels, (100, 300), (300, 399), 4, -1)  # 服
    cv2.circle(labels, (200, 180), 130, 3, -1)  # 髪
    cv2.circle(labels, (200, 200), 110, 1, -1)  # 顔
    for x in (150, 250):
        cv2.circle(labels, (x, 200), 20, 2, -1)  # 瞳
    rgb[labels == 1] = SKIN
    rgb[labels == 2] = (40, 90, 200)
    rgb[labels == 3] = (120, 70, 50)
    rgb[labels == 4] = (70, 110, 190)
    cv2.rectangle(rgb, (180, 330), (220, 370), (250, 220, 40), -1)  # 服の上の、色の違う飾り
    for x in (150, 250):
        cv2.line(rgb, (x - 26, 172), (x + 26, 172), (30, 20, 25), 7)  # まつ毛
    cv2.line(rgb, (180, 262), (220, 262), (190, 90, 90), 2)  # 口
    return labels, rgb, {1: "face", 2: "eye", 3: "hair", 4: "服"}


@pytest.fixture
def face_passes(tmp_path):
    labels, rgb, materials = face_scene()
    return write_passes(tmp_path / "work" / "passes", labels, rgb, materials)


@pytest.fixture
def fixture_work(tmp_path):
    """リポジトリに置いた、テスト用のシーンの画像の組を写した作業用のフォルダ。"""
    work = tmp_path / "work"
    shutil.copytree(FIXTURE / "passes", work / "passes")
    return work
