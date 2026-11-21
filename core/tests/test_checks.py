# SPDX-License-Identifier: AGPL-3.0-or-later
"""読み込みの境界の検査と、公開前の検査。このリポジトリが通ることと、違反を見つけられることを確かめる。"""
import importlib.util
import shutil

from conftest import FIXTURE, REPO


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "checks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boundary = load("check_boundary")
publish = load("check_publish")
HEADER = "# SPDX-License-Identifier: {}\n"


def test_this_repository_keeps_the_import_boundary():
    assert boundary.check(REPO) == []


def test_this_repository_passes_the_publish_check():
    assert publish.check(REPO, publish.tracked_files(REPO)) == []


def test_boundary_check_finds_imports_across_the_boundary(tmp_path):
    (tmp_path / "core" / "src").mkdir(parents=True)
    (tmp_path / "bridge").mkdir()
    (tmp_path / "core" / "src" / "a.py").write_text("import numpy\nfrom mathutils import Vector\n", encoding="utf-8")
    (tmp_path / "core" / "src" / "b.py").write_text("def f():\n    import bpy.types\n", encoding="utf-8")
    (tmp_path / "bridge" / "c.py").write_text("import bpy\nfrom mesh2ink.passes import load_passes\n", encoding="utf-8")
    problems = boundary.check(tmp_path)
    assert len(problems) == 3
    assert any("core/src/a.py" in p and "mathutils" in p for p in problems)
    assert any("core/src/b.py" in p and "bpy" in p for p in problems)
    assert any("bridge/c.py" in p and "mesh2ink" in p for p in problems)


def make_repo(root):
    for folder, title in (("core", "GNU AFFERO GENERAL PUBLIC LICENSE"), ("bridge", "GNU GENERAL PUBLIC LICENSE")):
        (root / folder).mkdir()
        (root / folder / "LICENSE").write_text(f"    {title}\n", encoding="utf-8")
    (root / "checks").mkdir()
    (root / "checks" / "allowed-images.txt").write_text("core/ok.png\n", encoding="utf-8")
    (root / "core" / "a.py").write_text(HEADER.format("AGPL-3.0-or-later") + "x = 1\n", encoding="utf-8")
    (root / "bridge" / "b.py").write_text(HEADER.format("GPL-3.0-or-later") + "x = 1\n", encoding="utf-8")
    shutil.copy(FIXTURE / "passes" / "parts.png", root / "core" / "ok.png")
    return ["core/LICENSE", "bridge/LICENSE", "checks/allowed-images.txt", "core/a.py", "bridge/b.py", "core/ok.png"]


def test_publish_check_passes_a_clean_tree(tmp_path):
    assert publish.check(tmp_path, make_repo(tmp_path)) == []


def test_publish_check_finds_an_image_that_is_not_listed(tmp_path):
    files = make_repo(tmp_path)
    for name in ("core/model.PNG", "core/scene.blend", "docs/sample.psd"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    problems = publish.check(tmp_path, [*files, "core/model.PNG", "core/scene.blend", "docs/sample.psd"])
    assert [p.split(":")[0] for p in problems] == ["core/model.PNG", "core/scene.blend", "docs/sample.psd"]


def test_publish_check_finds_metadata_in_a_listed_png(tmp_path):
    files = make_repo(tmp_path)
    png = tmp_path / "core" / "ok.png"
    png.write_bytes(png.read_bytes() + b"tEXt")
    assert publish.check(tmp_path, files) == ["core/ok.png: PNGにメタデータが入っている"]


def test_publish_check_finds_absolute_paths(tmp_path):
    files = make_repo(tmp_path)
    samples = {"win.md": "C:" + "\\Users\\" + "someone\\x", "mac.md": "/Users" + "/someone/x", "linux.md": "/home" + "/someone/x"}
    for name, text in samples.items():
        (tmp_path / name).write_text(f"see {text}\n", encoding="utf-8")
    (tmp_path / "fine.md").write_text("see docs/Users and ~/x\n", encoding="utf-8")
    problems = publish.check(tmp_path, [*files, *samples, "fine.md"])
    assert sorted(p.split(":")[0] for p in problems) == sorted(samples)


def test_publish_check_finds_a_missing_or_wrong_license_header(tmp_path):
    files = make_repo(tmp_path)
    (tmp_path / "core" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "bridge" / "b.py").write_text(HEADER.format("AGPL-3.0-or-later"), encoding="utf-8")
    problems = publish.check(tmp_path, files)
    assert len(problems) == 2 and "core/a.py" in problems[0] and "GPL-3.0-or-later" in problems[1]


def test_publish_check_finds_a_missing_license_file(tmp_path):
    files = make_repo(tmp_path)
    (tmp_path / "bridge" / "LICENSE").write_text("MIT\n", encoding="utf-8")
    assert publish.check(tmp_path, files) == ["bridge/LICENSE: GNU GENERAL PUBLIC LICENSE の全文が無い"]
