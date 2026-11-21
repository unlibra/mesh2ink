# SPDX-License-Identifier: AGPL-3.0-or-later
"""書き出しの段と、本体の通しの流れ。リポジトリに置いた画像の組を使い、Blender無しで回る。"""
import json

import numpy as np
from conftest import FIXTURE
from psd_tools import PSDImage

from mesh2ink.export import icon_sheet, icon_square, read_psd_composite
from mesh2ink.imageio import read_png
from mesh2ink.pipeline import process_passes
from mesh2ink.preset import load_preset

PRESET = load_preset(FIXTURE / "preset.json")


def test_icon_square_without_a_face_holds_everything_in_view():
    foreground = np.zeros((100, 200), bool)
    foreground[20:60, 30:150] = True
    left, top, side = icon_square(foreground, np.zeros_like(foreground))
    assert side == 120 and left == 30 and top + side >= 60 and top <= 20


def test_icon_sheet_puts_the_picture_on_white():
    final = np.zeros((100, 100, 4), np.uint8)
    sheet = icon_sheet(final, (0, 0, 100))
    assert sheet.shape == (420, 956, 3)
    assert tuple(sheet[200, 200]) == (255, 255, 255)  # 400pxのアイコンの中。絵が透明なら白
    assert tuple(sheet[12, 422]) != (255, 255, 255)  # 円の外


def test_outputs_of_the_test_scene(fixture_work):
    result = process_passes(fixture_work, PRESET)
    layers = sorted(p.name for p in (fixture_work / "layers").iterdir())
    names = ["fill_01_body", "fill_02_eye", "fill_03_face", "fill_04_hair", "fill_05", "line"]
    assert layers == [f"{i:02d}_{name}.png" for i, name in enumerate(names, 1)]
    assert result["layers"] == layers
    roles = {p["material"]: p["role"] for p in result["parts"]}
    assert roles == {"body": "other", "eye": "eye", "face": "face", "hair": "other", "服": "other"}
    for name in ("illustration.psd", "final.png", "icon_sheet.png"):
        assert (fixture_work / name).is_file()
    final = read_png(fixture_work / "final.png")
    assert final.shape == (512, 512, 4)
    assert final[0, 0, 3] == 0  # 背景は透明


def test_psd_has_the_named_layers_and_reads_back_to_the_final_picture(fixture_work):
    process_passes(fixture_work, PRESET)
    psd = PSDImage.open(fixture_work / "illustration.psd")
    assert [layer.name for layer in psd] == ["fill_01_body", "fill_02_eye", "fill_03_face", "fill_04_hair", "fill_05", "line"]
    assert all(layer.size == (512, 512) for layer in psd)
    assert (read_psd_composite(fixture_work / "illustration.psd") == read_png(fixture_work / "final.png")).all()


def test_final_picture_shows_the_line_over_the_fills(fixture_work):
    process_passes(fixture_work, PRESET)
    final = read_png(fixture_work / "final.png")
    line = read_png(fixture_work / "layers" / "06_line.png")
    hair = read_png(fixture_work / "layers" / "04_fill_04_hair.png")
    solid_line = line[..., 3] == 255
    only_hair = (hair[..., 3] == 255) & (line[..., 3] == 0)
    assert solid_line.any() and only_hair.any()
    assert (final[solid_line] == line[solid_line]).all()
    assert (final[only_hair] == hair[only_hair]).all()

def test_same_passes_and_preset_give_the_same_pixels(fixture_work, tmp_path):
    import shutil

    second = tmp_path / "second"
    shutil.copytree(fixture_work, second)
    process_passes(fixture_work, PRESET)
    process_passes(second, PRESET)
    for path in sorted(p for p in fixture_work.rglob("*.png")):
        assert (read_png(path) == read_png(second / path.relative_to(fixture_work))).all(), path.name


def test_regression_against_the_stored_picture(fixture_work):
    process_passes(fixture_work, PRESET)
    final = read_png(fixture_work / "final.png").astype(int)
    expected = read_png(FIXTURE / "expected_final.png").astype(int)
    assert np.abs(final - expected).max() <= 2


def test_line_color_comes_from_the_preset(fixture_work):
    import dataclasses

    process_passes(fixture_work, dataclasses.replace(PRESET, line_color=(0.0, 1.0, 0.0)))
    line = read_png(fixture_work / "layers" / "06_line.png")
    assert (line[line[..., 3] > 0][:, :3] == (0, 255, 0)).all()


def test_face_none_draws_only_shape_lines(fixture_work):
    import dataclasses

    result = process_passes(fixture_work, dataclasses.replace(PRESET, face="none"))
    assert {p["role"] for p in result["parts"]} == {"other"}
    assert json.dumps(result["layers"]).count("fill_") == 5
