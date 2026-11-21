# SPDX-License-Identifier: AGPL-3.0-or-later
"""公開前の検査。gitで追跡しているファイルを調べる。

- 置いてよい画像の一覧（checks/allowed-images.txt）に無い画像や、モデルのファイルが入っていない
- 置いてよいPNGに、テキストのメタデータ（パス、日時）が入っていない
- 個人の環境の絶対パスが入っていない
- 本体と橋渡しのフォルダに、それぞれのライセンスがあり、各ソースの先頭に表示がある

使い方：python checks/check_publish.py [<リポジトリのルート>]
"""
import re
import subprocess
import sys
from pathlib import Path

BINARY_SUFFIXES = {
    *(".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tga", ".tif", ".tiff", ".exr", ".psd", ".clip", ".kra", ".ora"),
    *(".blend", ".blend1", ".fbx", ".vrm", ".glb", ".gltf", ".obj", ".unitypackage", ".zip", ".7z"),
}
PNG_TEXT_CHUNKS = (b"tEXt", b"iTXt", b"zTXt", b"tIME", b"eXIf")
# ドライブ名つきの利用者のフォルダと、Unix系の利用者のフォルダ
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+\w|/Users/\w[^/\s]*/|/home/\w[^/\s]*/")
LICENSES = {
    "core": ("GNU AFFERO GENERAL PUBLIC LICENSE", "AGPL-3.0-or-later"),
    "bridge": ("GNU GENERAL PUBLIC LICENSE", "GPL-3.0-or-later"),
}
OTHER_SOURCE_LICENSE = "AGPL-3.0-or-later"


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [name for name in out.decode("utf-8").split("\0") if name]


def check(root: Path, files: list[str]) -> list[str]:
    problems = []
    allowed = set((root / "checks" / "allowed-images.txt").read_text(encoding="utf-8").split())
    for name in files:
        path = root / name
        suffix = path.suffix.lower()
        if suffix in BINARY_SUFFIXES:
            if name not in allowed:
                problems.append(f"{name}: 置いてよい画像の一覧に無い")
            elif suffix == ".png" and any(chunk in path.read_bytes() for chunk in PNG_TEXT_CHUNKS):
                problems.append(f"{name}: PNGにメタデータが入っている")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{name}: 文字として読めない。置いてよい画像の一覧にも無い")
            continue
        if ABSOLUTE_PATH.search(text):
            problems.append(f"{name}: 個人の環境の絶対パスが入っている")
        if suffix == ".py":
            top = name.split("/")[0]
            expect = LICENSES[top][1] if top in LICENSES else OTHER_SOURCE_LICENSE
            if not text.startswith(f"# SPDX-License-Identifier: {expect}\n"):
                problems.append(f"{name}: 先頭にライセンスの表示（{expect}）が無い")
    for name in sorted(allowed - set(files)):
        problems.append(f"{name}: 置いてよい画像の一覧にあるが、リポジトリに無い")
    for folder, (title, _) in LICENSES.items():
        license_file = root / folder / "LICENSE"
        if f"{folder}/LICENSE" not in files or title not in license_file.read_text(encoding="utf-8")[:400]:
            problems.append(f"{folder}/LICENSE: {title} の全文が無い")
    return problems


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    found = check(base, tracked_files(base))
    print("\n".join(found) if found else "公開前の検査：問題なし")
    sys.exit(1 if found else 0)
