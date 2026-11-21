# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import json

import pytest

from mesh2ink.errors import Mesh2inkError
from mesh2ink.preset import DEFAULT, load_preset, parse_preset


def valid():
    return copy.deepcopy(DEFAULT.to_json())


def test_default_matches_the_chosen_style():
    assert (DEFAULT.size, DEFAULT.subdivision, DEFAULT.line_width, DEFAULT.line_smooth) == (1600, 2, 5, 3)
    assert (DEFAULT.face, DEFAULT.eye) == ("auto", "auto")


def test_written_preset_reads_back_to_the_same_preset():
    assert parse_preset(valid()) == DEFAULT


def test_written_preset_does_not_carry_the_blender_location():
    data = valid()
    data["blender"] = "somewhere/blender"
    assert "blender" not in parse_preset(data).to_json()
    assert parse_preset(data).blender == "somewhere/blender"


@pytest.mark.parametrize(
    ("path", "name"),
    [((), "extra"), (("bridge",), "bridge.extra"), (("core",), "core.extra"), (("core", "roles"), "core.roles.extra")],
)
def test_unknown_item_is_named(path, name):
    data = valid()
    target = data
    for key in path:
        target = target[key]
    target["extra"] = 1
    with pytest.raises(Mesh2inkError, match=f"知らない項目.*{name}"):
        parse_preset(data)


def test_missing_item_is_named():
    data = valid()
    del data["bridge"]["line_width"]
    with pytest.raises(Mesh2inkError, match="bridge.line_width"):
        parse_preset(data)


def test_other_version_reports_both_versions():
    data = valid()
    data["version"] = 2
    with pytest.raises(Mesh2inkError, match="版が 2 .*対応している版は 1"):
        parse_preset(data)


@pytest.mark.parametrize("size", [255, 4097, 800.5, "1600", True])
def test_size_out_of_range_reports_the_range(size):
    data = valid()
    data["bridge"]["size"] = size
    with pytest.raises(Mesh2inkError, match="256〜4096"):
        parse_preset(data)


@pytest.mark.parametrize("size", [256, 4096])
def test_size_at_the_edges_is_accepted(size):
    data = valid()
    data["bridge"]["size"] = size
    assert parse_preset(data).size == size


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("bridge", "subdivision", -1),
        ("bridge", "subdivision", 1.5),
        ("bridge", "line_width", 0),
        ("bridge", "line_width", "5"),
        ("bridge", "line_smooth", -1),
        ("core", "line_color", [0.1, 0.2]),
        ("core", "line_color", [0.1, 0.2, 1.5]),
        ("core", "seed", -1),
        ("core", "seed", "0"),
    ],
)
def test_wrong_type_or_range_is_named(section, key, value):
    data = valid()
    data[section][key] = value
    with pytest.raises(Mesh2inkError, match=f"{section}.{key}"):
        parse_preset(data)


def test_role_must_be_text():
    data = valid()
    data["core"]["roles"]["face"] = 3
    with pytest.raises(Mesh2inkError, match="core.roles.face"):
        parse_preset(data)


def test_unreadable_file(tmp_path):
    broken = tmp_path / "preset.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(Mesh2inkError, match="読めません"):
        load_preset(broken)
    with pytest.raises(Mesh2inkError, match="読めません"):
        load_preset(tmp_path / "missing.json")


def test_load_reads_utf8(tmp_path):
    data = valid()
    data["core"]["roles"]["face"] = "顔"
    path = tmp_path / "設定.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert load_preset(path).face == "顔"
