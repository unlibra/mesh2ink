# mesh2ink

Blenderで用意した3Dモデルのシーンから、レイヤーつきの2Dイラストを作る道具。線画と、部位ごとの下塗りを出す。開発中で、設定と出力の形はまだ変わる。

対応しているのは、Windows 11とBlender 5.2 LTS。

## 使い方

[uv](https://docs.astral.sh/uv/)が要る。リポジトリを取得したフォルダで実行する。

```
uv run --project core mesh2ink run scene.blend --out out/
```

`out/` に次が出る。

- `layers/`：レイヤーごとの透過PNG。下から、材質ごとの下塗り（`fill_NN_<材質名>`）、線画（`line`）
- `illustration.psd`：レイヤーつきのPSD
- `final.png`：合成した絵。背景は透明
- `icon_sheet.png`：アイコンにした時の見え方（400px、円、96px、48px）
- `passes/`：Blenderが書き出した画像の組
- `report.json`、`preset.json`：使った設定と記録。同じ絵をもう一度出す時は、`--preset out/preset.json` を付ける

Blenderは、設定の `blender` 欄、環境変数 `MESH2INK_BLENDER`、既定の入れ場所の順に探す。元の `.blend` は変更しない。

## シーンに求める条件

- 有効なカメラが1つある
- カメラに写るメッシュがあり、その全部の面に材質がある。材質1つを1つの部位として扱う
- 材質は64個まで

顔と瞳の部位は、絵から自動で決める。決められない時は、設定の `core.roles` に材質の名前を書く。顔の無いものを描く時は `"none"` と書く。

## テスト

```
uv run --project core pytest core/tests              # Blender無しで回る
uv run --project core pytest core/tests -m blender   # Blenderを起動する
uv run --project core python checks/check_boundary.py
uv run --project core python checks/check_publish.py
```

## ライセンス

- `core/`：AGPL-3.0-or-later（[core/LICENSE](core/LICENSE)）。Blenderの外で動き、Blenderのモジュールを読み込まない
- `bridge/`：GPL-3.0-or-later（[bridge/LICENSE](bridge/LICENSE)）。Blenderの中で動き、画像の組を書き出す
