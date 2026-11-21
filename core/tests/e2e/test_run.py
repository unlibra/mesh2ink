# SPDX-License-Identifier: AGPL-3.0-or-later
"""Blenderを起動して、シーンから出力までを通すテスト。手元で `pytest -m blender` で回す。

Blenderは、環境変数 MESH2INK_BLENDER か、既定の入れ場所から探す。
環境変数 MESH2INK_LOCAL_SCENES に .blend を置いたフォルダを指定すると、手元のモデルにも同じ確かめをする。
"""
import copy
import getpass
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO

from mesh2ink.blender import find_blender
from mesh2ink.export import read_psd_composite
from mesh2ink.imageio import read_png
from mesh2ink.preset import DEFAULT

pytestmark = pytest.mark.blender

OUTPUTS = {"passes", "layers", "illustration.psd", "final.png", "icon_sheet.png", "report.json", "preset.json"}
PNG_TEXT_CHUNKS = (b"tEXt", b"iTXt", b"zTXt", b"tIME", b"eXIf")
LOCAL_SCENES = sorted(Path(os.environ["MESH2INK_LOCAL_SCENES"]).glob("*.blend")) if os.environ.get("MESH2INK_LOCAL_SCENES") else []


def make_scene(path, variant=None):
    cmd = [str(find_blender(None)), "--background", "--factory-startup", "--python", str(REPO / "bridge" / "make_test_scene.py")]
    subprocess.run([*cmd, "--", str(path), *([variant] if variant else [])], check=True, capture_output=True)
    assert path.is_file()
    return path


def mesh2ink(*args):
    return subprocess.run([sys.executable, "-m", "mesh2ink.cli", *map(str, args)], capture_output=True, encoding="utf-8")


def small_preset(folder, **roles):
    data = copy.deepcopy(DEFAULT.to_json())
    data["bridge"].update(size=512, line_width=2)
    data["core"]["roles"].update(roles)
    path = folder / "small.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def scene(tmp_path_factory):
    return make_scene(tmp_path_factory.mktemp("scene") / "testscene.blend")


@pytest.fixture(scope="session")
def first_run(scene, tmp_path_factory):
    """既定の設定（1600px）での1回の実行。"""
    out = tmp_path_factory.mktemp("run") / "out"
    before = hashlib.sha256(scene.read_bytes()).hexdigest()
    result = mesh2ink("run", scene, "--out", out)
    assert result.returncode == 0, result.stderr
    return out, before


def check_outputs(out):
    """出力のフォルダの形と、記録の中身。"""
    assert {p.name for p in out.iterdir()} == OUTPUTS
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    layers = sorted(p.name for p in (out / "layers").iterdir())
    assert layers == report["layers"]
    kinds = {name.split("_", 1)[1].removesuffix(".png") for name in layers}
    assert "line" in kinds and len(kinds) >= 2
    assert sorted(p.name for p in (out / "passes").iterdir()) == sorted(
        ["meta.json", *(f"{n}.png" for n in ("color", "parts", "depth", "normal", "light", "line_contour", "line_material"))]
    )
    final = read_png(out / "final.png")
    assert final.shape[2] == 4 and max(final.shape[:2]) == report["preset"]["bridge"]["size"]
    assert (read_psd_composite(out / "illustration.psd") == final).all()
    assert read_png(out / "icon_sheet.png").shape == (420, 956, 3)
    assert json.loads((out / "preset.json").read_text(encoding="utf-8")) == report["preset"]
    assert report["blender"].startswith("5.2.") and set(report["times"]) >= {"bridge", "core", "total"}
    return report


def check_no_private_data(out):
    """出力の全部のファイルに、絶対パス、ユーザー名、Blenderが書くメタデータが無い。"""
    needles = {str(Path.home()), str(Path.home()).replace("\\", "/"), str(Path.home()).replace("\\", "\\\\"), getpass.getuser()}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        data = path.read_bytes()
        if path.suffix == ".json":
            for needle in needles:
                assert needle not in data.decode("utf-8"), f"{path.name}: {needle}"
            assert "Blender Foundation" not in data.decode("utf-8")
        else:
            for needle in (n.encode("utf-8") for n in needles if len(n) > 8):
                assert needle not in data, path.name
            assert b"Blender" not in data, path.name
        if path.suffix == ".png":
            for chunk in PNG_TEXT_CHUNKS:
                assert chunk not in data, f"{path.name}: {chunk!r}"


def test_outputs(first_run):
    report = check_outputs(first_run[0])
    assert len(report["layers"]) >= 4
    assert {p["material"]: p["role"] for p in report["parts"]}["face"] == "face"
    assert report["preset"] == DEFAULT.to_json()


def test_blend_unchanged(scene, first_run):
    assert hashlib.sha256(scene.read_bytes()).hexdigest() == first_run[1]
    assert [p.name for p in scene.parent.iterdir()] == [scene.name]


def test_no_private_data(first_run):
    check_no_private_data(first_run[0])


def test_deterministic(scene, first_run, tmp_path):
    """同じシーンと設定の2回目が、1回目と画素で一致する。1回目の出力の preset.json を渡す。"""
    first = first_run[0]
    second = tmp_path / "second"
    result = mesh2ink("run", scene, "--out", second, "--preset", first / "preset.json")
    assert result.returncode == 0, result.stderr
    images = sorted(p.relative_to(first) for p in first.rglob("*.png"))
    assert len(images) >= 13
    for rel in images:
        assert (read_png(first / rel) == read_png(second / rel)).all(), rel
    assert (read_psd_composite(second / "illustration.psd") == read_png(first / "final.png")).all()
    assert (first / "passes" / "meta.json").read_bytes() == (second / "passes" / "meta.json").read_bytes()

    def stable(folder):
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        del report["times"]
        return report

    assert stable(first) == stable(second)


def test_japanese_path(scene, tmp_path):
    folder = tmp_path / "日本語の フォルダ"
    folder.mkdir()
    local = folder / "シーン 1.blend"
    local.write_bytes(scene.read_bytes())
    out = folder / "出力 先"
    result = mesh2ink("run", local, "--out", out, "--preset", small_preset(tmp_path))
    assert result.returncode == 0, result.stderr
    assert len(check_outputs(out)["layers"]) >= 4


def test_previous_output_is_replaced(scene, tmp_path):
    out = tmp_path / "out"
    preset = small_preset(tmp_path)
    assert mesh2ink("run", scene, "--out", out, "--preset", preset).returncode == 0
    (out / "layers" / "99_old.png").write_bytes(b"old")
    assert mesh2ink("run", scene, "--out", out, "--preset", preset).returncode == 0
    check_outputs(out)


def test_scene_with_one_part_and_no_face(tmp_path):
    scene = make_scene(tmp_path / "plain.blend", "plain")
    out = tmp_path / "out"
    result = mesh2ink("run", scene, "--out", out, "--preset", small_preset(tmp_path, face="none"))
    assert result.returncode == 0, result.stderr
    assert check_outputs(out)["layers"] == ["01_fill_01_plain.png", "02_line.png"]


@pytest.mark.parametrize(
    ("variant", "expect"),
    [
        ("no_camera", ["カメラ"]),
        ("no_mesh", ["写る物がありません"]),
        ("away", ["写る物がありません"]),
        ("no_material", ["材質の無い面", "bare"]),
        ("too_many_parts", ["65", "64"]),
        ("plain", ["顔の部位を自動で決められません", "core.roles.face", "plain"]),
    ],
)
def test_bad_scene_is_refused_without_output(tmp_path, variant, expect):
    scene = make_scene(tmp_path / f"{variant}.blend", variant)
    out = tmp_path / "out"
    result = mesh2ink("run", scene, "--out", out, "--preset", small_preset(tmp_path))
    assert result.returncode != 0
    assert result.stderr.startswith("mesh2ink: ") and result.stderr.count("\n") == 1, result.stderr
    for text in expect:
        assert text in result.stderr, result.stderr
    assert not out.exists()


@pytest.mark.skipif(not LOCAL_SCENES, reason="MESH2INK_LOCAL_SCENES が指定されていない")
@pytest.mark.parametrize("local_scene", LOCAL_SCENES, ids=lambda p: p.stem)
def test_local_scene(local_scene, tmp_path):
    """手元のモデルのシーン。出力先は MESH2INK_LOCAL_OUT があればその下、無ければ一時のフォルダ。"""
    base = Path(os.environ["MESH2INK_LOCAL_OUT"]) if os.environ.get("MESH2INK_LOCAL_OUT") else tmp_path
    out = base / local_scene.stem
    before = hashlib.sha256(local_scene.read_bytes()).hexdigest()
    preset = local_scene.with_suffix(".preset.json")
    result = mesh2ink("run", local_scene, "--out", out, *(["--preset", preset] if preset.is_file() else []))
    assert result.returncode == 0, result.stderr
    report = check_outputs(out)
    assert "face" in {p["role"] for p in report["parts"]}
    check_no_private_data(out)
    assert hashlib.sha256(local_scene.read_bytes()).hexdigest() == before
