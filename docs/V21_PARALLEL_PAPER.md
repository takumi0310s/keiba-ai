# V21 並行 paper 予測 — 設計ドキュメント

> 作成: 2026-05-23  
> 対象ファイル: `tools/v21_paper_predict.py`

---

## 概要

V15 production の完全不変を保証したまま、V21 候補モデルを並行 paper 評価するスクリプト。

- **V15**: 実 cash 投票に使用する production 予測（本スクリプトは V15 予測も再実行するが、Discord は V21 paper channel にのみ送信）
- **V21**: paper 予測のみ。Discord メッセージは必ず「【V21 paper】」と「投票しないでください」を含む
- V21 モデルファイルが存在しない場合はグレースフルに無視し、V15 予測ログのみ記録して終了

---

## Usage

```bash
# 日付単位で当日全レースを処理
python tools/v21_paper_predict.py --date 20260524

# 単一レース指定
python tools/v21_paper_predict.py --race 202605240511

# Discord 送信なし (テスト)
python tools/v21_paper_predict.py --date 20260524 --dry-run
```

---

## モデルパス

| モデル | パス |
|--------|------|
| V15 production | `keiba_model_v15_central.pkl.gz` (ルート直下) |
| V21 candidate | `models/v21_candidate.pkl.gz` |

V21 モデルは parallel agent が作成予定。ファイルが存在しない場合スクリプトはエラーなく続行。

---

## 戦略フィルタ

両モデルに同一の 戦略⑦案C + C4 フィルタを適用:

| フィルタ | 内容 |
|---------|------|
| 06_特別除外 | G/L/OPEN特別でない平場特別を除外 |
| 京都除外 (P0-2) | 重賞・L以外の京都レースを除外 |
| 条件E除外 | 頭数 ≤7 |
| 条件B除外 | 重〜不馬場 |
| 条件X除外 | 15頭+重〜不馬場 (重賞除く) |
| C4: 条件A 1600-1800m | Cond-A かつ距離 1600-1800m を除外 |

---

## Discord チャンネル

| 用途 | 環境変数 | フォールバック |
|------|---------|--------------|
| V21 paper | `DISCORD_WEBHOOK_V21_PAPER` | `DISCORD_WEBHOOK_UPDATES` |
| V15 production | `DISCORD_WEBHOOK_BETS` | (本スクリプトは送信しない) |

V15 production の Discord 送信は `race_auto_notify.py` が担当。本スクリプトは V21 paper channel のみ送信。

---

## V21 Discord メッセージ形式

```
【V21 paper — 投票しないでください】
📝 東京11R ヴィクトリアマイル
   芝1600m 良 条件A (12頭)
V21候補モデル (TYB全部入り) の予測 / 実 cash 投票は V15 のみ

三連複フォーメーション 7点 (V21 paper)
1列目: 3
2列目: 4, 5
3列目: 4, 5, 6, 7, 8

V15との差異:
  軸馬: V15=1(馬A) V21=3(馬C) → 相違 ← 注目

⚠ これはpaper予測です。実際の投票は V15 買い目のみ使用してください。
```

---

## paper ログ

```
data/v21_paper_log/{YYYYMMDD}/{race_id}.json
```

各 JSON に含まれる主要フィールド:

| フィールド | 内容 |
|-----------|------|
| `race_id` | 12桁レース ID |
| `timestamp` | 予測実行時刻 (ISO 8601) |
| `v15_top5` / `v21_top5` | 馬番 top-5 (降順) |
| `v15_formation` / `v21_formation` | 買い目リスト |
| `v15_strategy_pass` / `v21_strategy_pass` | フィルタ通過フラグ |
| `top1_match` | 軸馬一致フラグ |
| `cond_key` | 条件分類 (A-E, X) |

---

## 週末比較レポート (compare_v21_paper.py 計画)

週末終了後に以下を実行して V15 vs V21 の ROI を比較する予定:

```bash
# 計画 (未実装)
python tools/compare_v21_paper.py --date 20260524
```

比較観点:
1. top-1 agreement rate (軸馬一致率)
2. V21 の的中率 / ROI (paper 値)
3. 相違レースの個別分析 (← 注目フラグ)
4. V21 strategy pass 率 vs V15

GO/no-go 基準 (Phase 3 後半、6/29-6/30):
- WF AUC ≥ 0.880
- LIVE retro winner_top1 ≥ 30%
- shift ≤ 12x
- paper ROI ≥ 110%

---

## 重要制約

- `tools/race_auto_notify.py` — 変更禁止
- `tools/predict_core.py` — 変更禁止
- 既存モデルファイル — 変更禁止
- V21 Discord メッセージは必ず「【V21 paper】」「投票しないでください」を含む
- V21 モデルロード失敗はグレースフル (V15 production 継続)
