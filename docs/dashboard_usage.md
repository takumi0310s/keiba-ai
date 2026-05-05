# Dashboard 使い方

`tools/dashboard.py` で **累計収支 + Phase 2.5 進捗** を 1 枚の HTML に集約。

## 起動

```bash
# 基本: HTML + JSON 出力 + Discord 通知
python tools/dashboard.py

# Discord に投げない
python tools/dashboard.py --no-discord

# 標準出力も抑制 (cron 用)
python tools/dashboard.py --silent --no-discord
```

## 出力

| path | 内容 |
|------|------|
| `data/dashboard/dashboard.html` | plotly インタラクティブ HTML (ブラウザで開く) |
| `data/dashboard/data.json` | 生データ (CI/別 script から再利用可) |

ブラウザで開く:
```bash
start data/dashboard/dashboard.html         # Windows
```

## 表示要素

1. **累計収支推移** (折れ線) — USER 実投資ベース。`HANDOFF_5_5_TO_5_9.md` の数値由来。
   - 4/12 +23,480円 → 5/3 +14,140円 (現在)
   - 撤退ライン -50,000円 を赤点線で表示
2. **Phase 2.5 進捗** (ドーナツ) — 完了 / pending / 手動 の比率
   - Session #1-#18 + 残タスク (H1-H3 / M1-M7 / L1-L4) を集計
3. **撤退ライン余裕** (gauge) — 現在累計 ¥14,140 / 撤退まで余裕 ¥64,140
4. **BATCH 仮想日別** (棒+折れ線) — `data/daily_results/*.csv` から全レース 700円投資想定の参考シミュレーション
5. **Phase 2.5 残タスク表** — 14 件、優先度別
6. **schtasks 一覧** — JRA / NAR 分類、次回起動時刻、状態

## データ更新ルール (HANDOFF v2 厳守)

- **USER 実投資** と **BATCH 仮想** は厳密に分離する。
- `USER_CUMULATIVE` (`tools/dashboard.py` 上部の定数) は HANDOFF v2 由来。新しい投資日が増えたらここに追記する。
- `PHASE_25_TASKS` も同様。新規タスク完了 / 残タスク追加は手動で edit。
- **追記時は必ず生データを再検証**。引き継ぎ書 v1 の数字は使わない (handoff_v1_v2_diff.md 参照)。

## 自動化 (任意)

毎晩 `Keiba-NightlySanity` (23:00) に同梱して走らせるなら:

```bat
python C:\Users\takum\keiba-ai\tools\dashboard.py --silent
```

を `nightly_sanity_check.bat` の末尾に追記。

## 依存

- `plotly>=5` (インタラクティブ HTML、CDN 経由で plotly.js を読む)
- 標準 `csv`, `json`, `subprocess`, `argparse`

## 衝突回避

このツールは `tools/dashboard.py`, `data/dashboard/*`, `docs/dashboard_usage.md` の 3 か所のみ書き込む。
他並行セッション (V15.1 / NAR / README 整備) と完全に独立。
