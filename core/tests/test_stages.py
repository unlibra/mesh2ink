# SPDX-License-Identifier: AGPL-3.0-or-later
"""画像の組の読み込み、役割、線、下塗りの段。小さい絵を作って確かめる。"""
import json

import cv2
import numpy as np
import pytest
from conftest import SKIN, face_scene, write_passes

from mesh2ink.errors import Mesh2inkError
from mesh2ink.fills import build_fills, fill_name
from mesh2ink.lineart import build_lineart
from mesh2ink.passes import load_passes
from mesh2ink.roles import resolve_roles


def stages(folder, face="auto", eye="auto"):
    passes = load_passes(folder)
    roles = resolve_roles(passes, face, eye)
    lineart = build_lineart(passes, roles)
    return passes, roles, lineart, build_fills(passes, roles, lineart)


# ---- 画像の組


def test_passes_list_only_visible_parts(tmp_path):
    labels, rgb, materials = face_scene()
    labels[labels == 3] = 1  # 髪を無くす
    passes = load_passes(write_passes(tmp_path / "passes", labels, rgb, materials))
    assert [p.material for p in passes.parts] == ["face", "eye", "服"]
    assert passes.parts[0].pixels == int((labels == 1).sum())


def test_passes_of_another_version_are_refused(face_passes):
    meta = json.loads((face_passes / "meta.json").read_text(encoding="utf-8"))
    meta["version"] = 2
    (face_passes / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(Mesh2inkError, match="版が 2 .*対応している版は 1"):
        load_passes(face_passes)


def test_missing_pass_is_named(face_passes):
    (face_passes / "depth.png").unlink()
    with pytest.raises(Mesh2inkError, match="depth.png"):
        load_passes(face_passes)


def test_nothing_in_view_is_refused(tmp_path):
    labels, rgb, materials = face_scene()
    folder = write_passes(tmp_path / "passes", np.zeros_like(labels), rgb, materials)
    with pytest.raises(Mesh2inkError, match="写る物がありません"):
        load_passes(folder)


# ---- 役割


def test_roles_are_found_from_the_picture(face_passes):
    passes, roles, _, _ = stages(face_passes)
    assert (roles.face, roles.eye) == (1, 2)
    assert [roles.name(p.id) for p in passes.parts] == ["face", "eye", "other", "other"]


def test_face_without_an_eye_part_is_found_by_its_dark_drawing(tmp_path):
    labels, rgb, materials = face_scene()
    labels[labels == 2] = 1  # 瞳を顔と同じ部位にする
    del materials[2]
    _, roles, _, _ = stages(write_passes(tmp_path / "passes", labels, rgb, materials))
    assert (roles.face, roles.eye) == (1, None)


def test_roles_can_be_named_in_the_preset(face_passes):
    _, roles, _, _ = stages(face_passes, face="face", eye="none")
    assert (roles.face, roles.eye) == (1, None)
    _, roles, _, _ = stages(face_passes, face="auto", eye="eye")
    assert (roles.face, roles.eye) == (1, 2)


def test_face_none_gives_no_roles(face_passes):
    _, roles, _, _ = stages(face_passes, face="none", eye="auto")
    assert (roles.face, roles.eye) == (None, None)


def test_unknown_material_lists_the_visible_ones(face_passes):
    with pytest.raises(Mesh2inkError, match="core.roles.face.*「nose」.*face、eye、hair、服"):
        stages(face_passes, face="nose")


def test_undecidable_face_tells_how_to_set_it(tmp_path):
    labels = np.zeros((200, 200), np.uint8)
    labels[50:150, 50:150] = 1
    rgb = np.full((200, 200, 3), 128, np.uint8)
    with pytest.raises(Mesh2inkError, match='core.roles.face.*"none".*plain'):
        stages(write_passes(tmp_path / "passes", labels, rgb, {1: "plain"}))


def test_two_eye_candidates_are_told_apart_by_being_a_pair(tmp_path):
    labels, rgb, materials = face_scene()
    cv2.circle(labels, (200, 230), 8, 4, -1)  # 顔の中の、瞳でない小さい部位を1つ
    labels[300:, :] = np.where(labels[300:, :] == 4, 3, labels[300:, :])  # 服を無くし、4番を小さい部位だけにする
    _, roles, _, _ = stages(write_passes(tmp_path / "passes", labels, rgb, materials))
    assert roles.eye == 2


# ---- 線


def test_dark_drawing_and_thin_drawing_become_ink(face_passes):
    _, _, lineart, _ = stages(face_passes)
    assert lineart.alpha[172, 150] > 0.9  # まつ毛
    assert lineart.alpha[262, 200] > 0.5  # 口
    assert lineart.alpha[120, 200] == 0  # 何も無い額
    assert lineart.alpha[330, 200] == 0  # 顔でない部位の模様は線にしない


def test_shape_lines_are_kept_except_around_the_eyes(tmp_path):
    labels, rgb, materials = face_scene()
    line = np.zeros(labels.shape, bool)
    line[120, 150:250] = True  # 額を横切る形の線
    line[200, 140:160] = True  # 瞳を横切る形の線
    _, _, lineart, _ = stages(write_passes(tmp_path / "passes", labels, rgb, materials, line))
    assert lineart.alpha[120, 200] == 1
    assert lineart.alpha[200, 143] == 0


def test_without_a_face_only_shape_lines_remain(tmp_path):
    labels, rgb, materials = face_scene()
    line = np.zeros(labels.shape, bool)
    line[200, 140:160] = True
    _, _, lineart, _ = stages(write_passes(tmp_path / "passes", labels, rgb, materials, line), face="none")
    assert lineart.alpha[200, 143] == 1
    assert lineart.alpha[172, 150] == 0


# ---- 下塗り


def test_fill_names_keep_only_ascii():
    assert fill_name(3, "Hair.001") == "fill_03_Hair001"
    assert fill_name(12, "服") == "fill_12"
    assert fill_name(1, "body_服2") == "fill_01_body_2"


def test_fills_take_their_colors_from_the_model(face_passes):
    passes, _, _, fills = stages(face_passes)
    by_name = {f.name: f.rgba for f in fills}
    assert list(by_name) == ["fill_01_face", "fill_02_eye", "fill_03_hair", "fill_04"]
    assert tuple(by_name["fill_03_hair"][60, 200]) == (120, 70, 50, 255)
    assert tuple(by_name["fill_02_eye"][200, 150]) == (40, 90, 200, 255)
    assert by_name["fill_03_hair"][200, 200, 3] == 0  # ほかの部位の所は透明
    for f in fills:
        part_id = int(f.name.split("_")[1])
        assert ((f.rgba[..., 3] == 255) == (passes.labels == part_id)).all()


def test_a_group_of_another_color_inside_a_part_keeps_its_color(face_passes):
    _, _, _, fills = stages(face_passes)
    cloth = fills[3].rgba
    assert tuple(cloth[350, 200, :3]) == (250, 220, 40)  # 飾り
    assert tuple(cloth[350, 120, :3]) == (70, 110, 190)  # 服の地


def test_small_color_differences_inside_a_part_are_flattened(tmp_path):
    labels, rgb, materials = face_scene()
    hair = labels == 3
    rows = np.broadcast_to(np.arange(400)[:, None], (400, 400))
    rgb[..., 0][hair] = (120 + (rows[hair] - 50) // 8).clip(0, 255)  # 髪にゆるい濃淡を付ける
    _, _, _, fills = stages(write_passes(tmp_path / "passes", labels, rgb, materials))
    assert len(np.unique(fills[2].rgba[labels == 3][:, :3], axis=0)) == 1


def test_face_fill_keeps_skin_and_blush_but_not_the_inked_drawing(tmp_path):
    labels, rgb, materials = face_scene()
    blush = np.zeros(labels.shape, np.float32)
    cv2.circle(blush, (140, 240), 22, 1.0, -1)
    blush = cv2.GaussianBlur(blush, (0, 0), 8)[..., None]
    face = (labels == 1)[..., None]
    rgb = np.where(face, rgb * (1 - 0.3 * blush) + np.array([255, 150, 150]) * 0.3 * blush, rgb).astype(np.uint8)
    _, _, _, fills = stages(write_passes(tmp_path / "passes", labels, rgb, materials))
    face_fill = fills[0].rgba
    assert np.abs(face_fill[172, 150, :3].astype(int) - SKIN).max() <= 12  # まつ毛の下は肌の色
    assert np.abs(face_fill[262, 200, :3].astype(int) - SKIN).max() <= 12  # 口の下も肌の色
    assert tuple(face_fill[240, 140, :3]) == tuple(rgb[240, 140])  # 頬の赤みはそのまま
    assert face_fill[240, 140, 1] < SKIN[1] - 10
