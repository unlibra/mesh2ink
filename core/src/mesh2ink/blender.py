# SPDX-License-Identifier: AGPL-3.0-or-later
"""Blenderを探し、版を確かめ、橋渡しのスクリプトを背景で起動する。Blenderのモジュールは読み込まない。"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .errors import Mesh2inkError

SUPPORTED = (5, 2)
ENV = "MESH2INK_BLENDER"
BRIDGE = Path(__file__).resolve().parents[3] / "bridge" / "export_passes.py"


def default_locations() -> list[Path]:
    name = f"Blender {SUPPORTED[0]}.{SUPPORTED[1]}"
    roots = [os.environ[k] for k in ("ProgramFiles", "ProgramW6432") if k in os.environ]
    return [Path(root) / "Blender Foundation" / name / "blender.exe" for root in dict.fromkeys(roots)]


def find_blender(preset_path: str | None) -> Path:
    """設定の blender 欄、環境変数、既定の入れ場所の順に探す。指定された場所に無ければ、先へは進まない。"""
    if preset_path is not None:
        if not Path(preset_path).is_file():
            raise Mesh2inkError(f"Blenderが見つかりません。設定の blender 欄の場所にありません：{preset_path}")
        return Path(preset_path)
    if os.environ.get(ENV):
        if not Path(os.environ[ENV]).is_file():
            raise Mesh2inkError(f"Blenderが見つかりません。環境変数 {ENV} の場所にありません：{os.environ[ENV]}")
        return Path(os.environ[ENV])
    locations = default_locations()
    for path in locations:
        if path.is_file():
            return path
    searched = "、".join(str(p) for p in locations)
    raise Mesh2inkError(f"Blenderが見つかりません。探した場所：{searched}。設定の blender 欄か、環境変数 {ENV} で場所を指定してください")


def blender_version(blender: Path) -> str:
    """版を確かめ、「5.2.1」の形で返す。対応していない版なら止まる。"""
    try:
        out = subprocess.run([str(blender), "--version"], capture_output=True, timeout=60).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired):
        raise Mesh2inkError(f"Blenderを起動できません：{blender}") from None
    m = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", out)
    if m is None:
        raise Mesh2inkError(f"Blenderの版を読めません：{blender}")
    if (int(m[1]), int(m[2])) != SUPPORTED:
        raise Mesh2inkError(f"Blenderの版が {m[1]}.{m[2]}.{m[3]} です。対応している版は {SUPPORTED[0]}.{SUPPORTED[1]} です")
    return f"{m[1]}.{m[2]}.{m[3]}"


def export_passes(blender: Path, scene: Path, config: dict[str, Any], out_dir: Path) -> None:
    """橋渡しを起動して、画像の組を out_dir に書き出させる。シーンが条件に合わなければ、理由を返して止まる。"""
    config_path = out_dir.parent / "bridge.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    cmd = [str(blender), "--background", "--factory-startup", str(scene), "--python-exit-code", "1", "--python", str(BRIDGE)]
    result = subprocess.run([*cmd, "--", str(config_path), str(out_dir)], capture_output=True)
    config_path.unlink()
    error = out_dir / "error.txt"
    if error.is_file():
        raise Mesh2inkError(error.read_text(encoding="utf-8"))
    if result.returncode != 0 or not (out_dir / "meta.json").is_file():
        lines = (result.stderr + result.stdout).decode("utf-8", "replace").strip().splitlines()
        raise Mesh2inkError(f"Blenderでの書き出しに失敗しました：{lines[-1] if lines else '出力がありません'}")
