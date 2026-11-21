# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import sys
from pathlib import Path

from .errors import Mesh2inkError
from .pipeline import run
from .preset import DEFAULT, load_preset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mesh2ink", description="Blenderのシーンから、レイヤーつきの2Dイラストを作る")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="シーンから全部の出力を作る")
    run_parser.add_argument("scene", type=Path, help="Blenderのシーン（.blend）")
    run_parser.add_argument("--out", type=Path, required=True, help="出力先のフォルダ")
    run_parser.add_argument("--preset", type=Path, help="設定のJSON。省くと既定の設定を使う")
    args = parser.parse_args(argv)
    try:
        preset = load_preset(args.preset) if args.preset else DEFAULT
        run(args.scene, args.out, preset)
    except Mesh2inkError as e:
        print(f"mesh2ink: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
