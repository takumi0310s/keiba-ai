# Session #65 D: 朝予測 vs 1h 前 比較 framework

## 1. tools/stage_compare_5_9.py

read-only 集計 tool。 V15 完全独立。

### 入力
| source | path | 役割 |
|--------|------|------|
| 朝 (Stage 1) | `data/daily_predictions/20260509.csv` | top1/2/3 + score |
| 1h 前 (Stage 2) | `data/v18/pre_race_predict_5_9_R*.json` (Glob) | stage2 ensemble result |
| 実結果 (verdict) | `data/v18/verdicts_5_9_realtime.json` (Session #61 産物) | actual_trio |

### 出力
- `data/v18/stage_compare_5_9_summary.{md,json}`
- `data/v18/stage_compare_5_9_by_race.json` (--by-race --json 時)

## 2. CLI

```bash
python tools/stage_compare_5_9.py --summary    # 累積 metric markdown
python tools/stage_compare_5_9.py --by-race    # R 別 table
python tools/stage_compare_5_9.py --by-race --json   # JSON 形式
```

## 3. metrics

| metric | 説明 | 解釈 |
|--------|------|------|
| top1 変更率 | Stage 2 で top1 が変わった R 割合 | 高い = Stage 2 効果大 (or noise) |
| top3 重複 mean | 朝 top3 と Stage 2 top3 の重複数 (0-3) | 低い = re-shuffle 大 |
| score diff mean | 朝 top1 score - Stage 2 top1 score | + = Stage 2 で confidence 上昇 |
| 朝 top1 in trio rate | 5/10 verdict 後 算出可 | baseline |
| Stage 2 top1 in trio rate | 同上 | baseline + diff = Stage 2 効果 |

## 4. 5/10 朝 backfill 想定

5/10 朝 (verdict_5_9_realtime.json が満) に再実行すると `morning_top1_in_trio_rate` / `stage2_top1_in_trio_rate` が埋まる。 差分 (pt) で Stage 2 の hit rate 効果を定量評価。

```bash
# 5/10 朝
python tools/stage_compare_5_9.py --summary
# → data/v18/stage_compare_5_9_summary.md に書き出し
```

## 5. 18:00 / 20:30 連携

Session #61 の cumulative (17:00) / summary (20:30) と独立稼働。 stage_compare はその時点で取得済の Stage 2 結果のみ集計。 cache+verdict 進行に応じて metric 更新。

## 6. 5/9 13:10 時点 (実 Stage 2 fire 前)

```
比較対象 R: 0
Stage 2 成功 R: 0
top1 変更 R: 0 (0.0% of OK)
top3 重複 mean: 0.00 / 3
verdict 取得 R: 0
実結果未取得 (5/10 朝 backfill 後 再実行)
```

placeholder 状態。 13:30+ watchdog が Stage 2 fire し始めると徐々に充足。

## 7. 干渉禁止確認

- predict_core.py / V15 model file 触らない (read-only 集計のみ)
- daily_predict.py / race_auto_notify.py 呼ばない
- Discord 通知なし (CLI のみ、 print + file out)
- 5/9 投票方針 不変
