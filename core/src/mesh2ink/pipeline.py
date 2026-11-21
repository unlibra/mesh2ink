# SPDX-License-Identifier: AGPL-3.0-or-later
"""シーンから全部の書き出しまでを通す。出力は一時のフォルダに作り、全部そろってから出力先へ移す。"""
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from . import __version__
from .blender import blender_version, export_passes, find_blender
from .errors import Mesh2inkError
from .export import write_outputs
from .fills import build_fills, line_layer
from .lineart import build_lineart
from .passes import load_passes
from .preset import Preset
from .roles import resolve_roles

REPORT_VERSION = 1
OUTPUT_NAMES = {"passes", "layers", "illustration.psd", "final.png", "icon_sheet.png", "report.json", "preset.json"}


def check_out_dir(out: Path) -> None:
    """出力先が、無い、空、前回のmesh2inkの出力のどれかであることを確かめる。"""
    if not out.exists():
        return
    if not out.is_dir():
        raise Mesh2inkError(f"出力先がフォルダではありません：{out}")
    names = {p.name for p in out.iterdir()}
    if names and not ("report.json" in names and names <= OUTPUT_NAMES):
        raise Mesh2inkError(f"出力先に、mesh2inkの出力でないファイルがあります。出力先を空にするか、別の場所を指定してください：{out}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_passes(work: Path, preset: Preset) -> dict[str, Any]:
    """本体の処理。work/passes を読み、work にレイヤーの組を書き出す。記録に入れる中身を返す。"""
    times = {}
    start = time.perf_counter()
    passes = load_passes(work / "passes")
    roles = resolve_roles(passes, preset.face, preset.eye)
    times["load"] = time.perf_counter() - start

    start = time.perf_counter()
    lineart = build_lineart(passes, roles)
    times["line"] = time.perf_counter() - start

    start = time.perf_counter()
    layers = [*build_fills(passes, roles, lineart), line_layer(lineart, preset.line_color)]
    times["fill"] = time.perf_counter() - start

    start = time.perf_counter()
    layer_files = write_outputs(work, layers, passes.labels > 0, lineart.dark)
    times["export"] = time.perf_counter() - start
    return {
        "parts": [{"id": p.id, "material": p.material, "role": roles.name(p.id), "pixels": p.pixels} for p in passes.parts],
        "layers": layer_files,
        "times": times,
    }


def run(scene: Path, out: Path, preset: Preset) -> None:
    if scene.suffix.lower() != ".blend" or not scene.is_file():
        raise Mesh2inkError(f"シーンのファイルを読めません。.blend のファイルを指定してください：{scene}")
    check_out_dir(out)
    blender = find_blender(preset.blender)
    version = blender_version(blender)

    total_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mesh2ink-") as tmp:
        work = Path(tmp) / "out"
        work.mkdir()
        export_passes(blender, scene.resolve(), preset.bridge_config(), work / "passes")
        bridge_time = time.perf_counter() - total_start
        core_start = time.perf_counter()
        result = process_passes(work, preset)
        write_json(work / "preset.json", preset.to_json())
        now = time.perf_counter()
        times = {"bridge": bridge_time, **result["times"], "core": now - core_start, "total": now - total_start}
        report = {
            "version": REPORT_VERSION,
            "mesh2ink": __version__,
            "blender": version,
            "preset": preset.to_json(),
            "parts": result["parts"],
            "layers": result["layers"],
            "times": {k: round(v, 3) for k, v in times.items()},
        }
        write_json(work / "report.json", report)
        if out.exists():
            shutil.rmtree(out)
        shutil.move(work, out)
