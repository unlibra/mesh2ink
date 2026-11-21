# SPDX-License-Identifier: AGPL-3.0-or-later
"""読み込みの境界の検査。本体がBlenderのモジュールを読み込まず、橋渡しが本体を読み込まないことを確かめる。

使い方：python checks/check_boundary.py [<リポジトリのルート>]
"""
import ast
import sys
from pathlib import Path

BLENDER_MODULES = {"bpy", "bmesh", "mathutils", "bpy_extras", "gpu", "bl_math", "idprop", "aud"}
CORE_MODULES = {"mesh2ink"}


def imported_modules(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def check(root: Path) -> list[str]:
    problems = []
    for folder, banned, what in (("core", BLENDER_MODULES, "Blenderのモジュール"), ("bridge", CORE_MODULES, "本体のモジュール")):
        for path in sorted((root / folder).rglob("*.py")):
            if ".venv" in path.parts:
                continue
            for name in sorted(imported_modules(path) & banned):
                problems.append(f"{path.relative_to(root).as_posix()}: {what} {name} を読み込んでいる")
    return problems


if __name__ == "__main__":
    found = check(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    print("\n".join(found) if found else "読み込みの境界：問題なし")
    sys.exit(1 if found else 0)
