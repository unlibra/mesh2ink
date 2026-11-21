# SPDX-License-Identifier: AGPL-3.0-or-later
"""悪い入力のうち、Blenderを起動する前に分かるもの。出力先に何も書かず、理由を1行返し、0以外で終わる。"""
import copy
import json
import os
import stat
import sys

import pytest

from mesh2ink import blender
from mesh2ink.cli import main
from mesh2ink.preset import DEFAULT


def fake_blender(folder, version_line):
    """版の問い合わせに version_line を返すだけの、にせのBlender。"""
    if sys.platform == "win32":
        path = folder / "blender.cmd"
        path.write_text(f"@echo off\r\necho {version_line}\r\n", encoding="ascii")
    else:
        path = folder / "blender"
        path.write_text(f"#!/bin/sh\necho '{version_line}'\n", encoding="ascii")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "scene.blend"
    path.write_bytes(b"BLENDER")
    return path


def refused(capsys, out, args, expect):
    """実行が0以外で終わり、理由が1行で、出力先が作られていないことを確かめる。"""
    code = main(["run", *map(str, args), "--out", str(out)])
    err = capsys.readouterr().err
    assert code != 0
    assert err.startswith("mesh2ink: ") and err.count("\n") == 1, err
    for text in expect:
        assert text in err, err
    assert not out.exists()


def write_preset(tmp_path, change):
    data = copy.deepcopy(DEFAULT.to_json())
    change(data)
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_missing_scene(tmp_path, capsys):
    refused(capsys, tmp_path / "out", [tmp_path / "none.blend"], ["none.blend", "読めません"])


def test_scene_that_is_not_a_blend_file(tmp_path, capsys):
    other = tmp_path / "scene.fbx"
    other.write_bytes(b"x")
    refused(capsys, tmp_path / "out", [other], ["scene.fbx", ".blend"])


def test_unreadable_preset(tmp_path, capsys, scene):
    broken = tmp_path / "preset.json"
    broken.write_text("not json", encoding="utf-8")
    refused(capsys, tmp_path / "out", [scene, "--preset", broken], ["設定のJSONを読めません"])


def test_preset_of_another_version(tmp_path, capsys, scene):
    preset = write_preset(tmp_path, lambda d: d.update(version=3))
    refused(capsys, tmp_path / "out", [scene, "--preset", preset], ["版が 3", "対応している版は 1"])


def test_preset_with_an_unknown_item(tmp_path, capsys, scene):
    preset = write_preset(tmp_path, lambda d: d["core"].update(shade=True))
    refused(capsys, tmp_path / "out", [scene, "--preset", preset], ["core.shade"])


def test_preset_with_a_wrong_type(tmp_path, capsys, scene):
    preset = write_preset(tmp_path, lambda d: d["bridge"].update(line_smooth="3"))
    refused(capsys, tmp_path / "out", [scene, "--preset", preset], ["bridge.line_smooth"])


def test_preset_with_size_out_of_range(tmp_path, capsys, scene):
    preset = write_preset(tmp_path, lambda d: d["bridge"].update(size=100))
    refused(capsys, tmp_path / "out", [scene, "--preset", preset], ["256〜4096"])


def test_blender_not_found_at_the_preset_location(tmp_path, capsys, scene):
    preset = write_preset(tmp_path, lambda d: d.update(blender=str(tmp_path / "nowhere" / "blender.exe")))
    refused(capsys, tmp_path / "out", [scene, "--preset", preset], ["Blenderが見つかりません", "blender 欄"])


def test_blender_not_found_at_the_environment_location(tmp_path, capsys, scene, monkeypatch):
    monkeypatch.setenv(blender.ENV, str(tmp_path / "nowhere" / "blender.exe"))
    refused(capsys, tmp_path / "out", [scene], ["Blenderが見つかりません", blender.ENV])


def test_blender_not_found_anywhere_lists_where_it_looked(tmp_path, capsys, scene, monkeypatch):
    monkeypatch.delenv(blender.ENV, raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    monkeypatch.setenv("ProgramW6432", str(tmp_path / "pf"))
    refused(capsys, tmp_path / "out", [scene], ["探した場所", str(tmp_path / "pf"), "blender 欄", blender.ENV])


def test_unsupported_blender_version(tmp_path, capsys, scene, monkeypatch):
    monkeypatch.setenv(blender.ENV, str(fake_blender(tmp_path, "Blender 4.2.3 LTS")))
    refused(capsys, tmp_path / "out", [scene], ["4.2.3", "対応している版は 5.2"])


def test_out_dir_with_other_files_is_left_alone(tmp_path, capsys, scene):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("keep", encoding="utf-8")
    code = main(["run", str(scene), "--out", str(out)])
    err = capsys.readouterr().err
    assert code != 0 and err.count("\n") == 1 and "出力先を空にするか" in err
    assert [p.name for p in out.iterdir()] == ["notes.txt"]


def test_out_dir_with_a_previous_output_plus_other_files_is_left_alone(tmp_path, capsys, scene):
    out = tmp_path / "out"
    out.mkdir()
    (out / "report.json").write_text("{}", encoding="utf-8")
    (out / "mine.png").write_bytes(b"x")
    code = main(["run", str(scene), "--out", str(out)])
    assert code != 0 and "出力先を空にするか" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["mine.png", "report.json"]


def test_out_dir_that_is_a_file(tmp_path, capsys, scene):
    out = tmp_path / "out"
    out.write_text("x", encoding="utf-8")
    code = main(["run", str(scene), "--out", str(out)])
    assert code != 0 and "フォルダではありません" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "x"


def test_program_files_is_where_blender_is_looked_for(monkeypatch, tmp_path):
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramW6432", raising=False)
    assert blender.default_locations() == [tmp_path / "Blender Foundation" / "Blender 5.2" / "blender.exe"]
    assert os.environ["ProgramFiles"] == str(tmp_path)
