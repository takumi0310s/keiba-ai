# Strategy Layer v2 - Paper Shadow Eval 運用 guide

**作成日**: 2026-05-16
**対象**: 5/18 (土) 朝以降 の paper shadow eval
**target**: 30+ race 蓄積で 真の ROI delta 検証

## 1. 前提

- V15 production / `tools/race_auto_notify.py` / `tools/daily_predict.py` / `tools/predict_core.py` は **完全 不変**
- 本 paper eval は **production と並行** で running、 Discord 通知なし、 実 bet なし
- 出力 file は `data/v21/strategy_v2_shadow_YYYYMMDD.csv` のみ

## 2. 5/16 (G1 前日) / 5/17 (ヴィクトリアM 当日) の 取扱

- ★ **5/16-5/17 は 何も しない** ★ (本 v2 を 起動しない)
- 5/16 23:00+ G1 day 通知 は 既存 `race_auto_notify.py` の strategy 7 で 動く
- 5/17 当日 朝 通知 も 既存 logic、 v2 介入なし
- 5/18 (日、 5/18 開催) 朝 以降 から paper shadow start

## 3. 5/18 paper shadow 起動 手順

### 3-1. 前提 条件

5/18 朝 8:00 daily_predict 完了後、 以下が 揃っていること:

```
data/daily_predictions/20260518.csv         # 既存 daily_predict 生成 (top1_score, top1_num 含む)
data/daily_predictions_full/20260518.csv    # tools/save_all_horse_scores.py で生成 (全頭 V15_score + odds)
data/odds_base_20260518.csv                 # race_auto_notify.py で 蓄積 (5 分前 odds)
```

`daily_predictions_full/20260518.csv` の生成は **手動 trigger** が必要:

```powershell
# 5/18 朝 daily_predict 後 (08:30+) に実行
python tools/save_all_horse_scores.py --date 20260518
```

または schtasks 登録 (★ user 確認後 ★)。

### 3-2. shadow eval 実行

```powershell
python tools/strategy_layer_v2.py --shadow 20260518
```

出力:
```
[INFO] N races for 20260518
[INFO] daily_predictions_full/20260518.csv loaded: M horses
[INFO] odds_base_20260518.csv loaded: N races
=== shadow date: 20260518 ===
total races: N
score_sum 使用 (full): N / N
odds resolved: N / N
v2 recommended: K
  - 700  base: K0
  - 1400 (2x): K1
  - 2100 (3x): K2
output: data\v21\strategy_v2_shadow_20260518.csv
```

### 3-3. 夜 (20:00) 結果 集計

`tools/daily_results.py` 後 (cumulative_results.csv 更新後) に:

```powershell
python tools/strategy_v2_aggregate.py  # ★ 別途 作る (5/18+ 蓄積 後)
```

→ shadow csv と cumulative_results.csv を join、 ROI delta 算出。

## 4. 出力 CSV 構造 (`data/v21/strategy_v2_shadow_YYYYMMDD.csv`)

| column | 内容 |
|--------|------|
| race_id | レース ID |
| date | YYYYMMDD |
| course / race_num / race_name / condition / num_horses / distance | レース info |
| top1_num | top1 horse 馬番 |
| top1_score | V15 raw score (full csv 優先、 fallback daily_predictions) |
| top1_odds | top1 horse 単勝オッズ (08:00 baseline) |
| score_sum | 全頭 V15_score 合計 (full csv より) |
| odds_resolved | True = 真の odds 取得済 |
| p_calibrated | top1 calibrated 確率 (raw 0.7 + iso 0.3 blend) |
| ev_top1 | 動的 EV (clip 10.0) |
| s7_pass | 戦略⑦ 通過 |
| v2_recommended | 最終的に bet するかどうか |
| v2_bet_size | 0 / 700 / 1400 / 2100 |
| v2_reason | 採否理由 |

## 5. 評価 指標

### 5-1. 1 日 単位

```
shadow csv 集計:
  - 全 race N
  - v2_recommended True race K
  - 700 円: K0、 1400 円: K1、 2100 円: K2
  - 合計 v2_inv = K0*700 + K1*1400 + K2*2100
```

### 5-2. 累計 (30 race+ 蓄積後)

cumulative_results.csv の actual_payout を merge して:

| metric | baseline (戦略⑦ 実 bet) | strategy_v2 shadow (paper) | delta |
|--------|---:|---:|---:|
| race 数 | M | M (内 recommended K) | - |
| total inv | M × 700 | v2_inv | delta_inv |
| total pay | sum(actual) | sum(actual × scale) | delta_pay |
| ROI | base_roi | v2_roi | delta_roi |
| PnL | base_pnl | v2_pnl | delta_pnl |

### 5-3. 判定基準 (★ honest ★)

- 30 race 蓄積で **|delta_roi| < ±5pt**: 統計的に 無差別、 継続観察
- 50 race 蓄積で **delta_roi >= +5pt**: 採用検討、 ただし 1 日 サンプル 偏り 注意
- 100 race 蓄積で **delta_roi >= +10pt**: 採用候補、 user 判断
- 100 race 蓄積で **delta_roi <= -5pt**: 撤退、 EV 動的閾値 不適 確定

「期待 +15-30% ROI 改善」 は **想定** であって 検証必須。

## 6. 安全装置

- production `race_auto_notify.py` への影響 0% (v2 は別 process)
- Discord 通知なし (shadow のみ)
- cumulative_results.csv 書き込みなし (read-only)
- 累計収支 +5,240 円 への影響 0% ※ 旧 +13,530 円 は drift、 5/16 P0-1 真値 (docs/ROI_DISCREPANCY_2026_05_16.md)
- 撤退ライン -50,000 円 への影響 0%

## 7. 失敗 時 (★ honest ★)

| 症状 | 対応 |
|------|------|
| 5/18 paper shadow ROI が baseline 同等 (-5pt〜+5pt) | 30 race まで 継続観察 |
| paper shadow ROI が baseline より顕著悪化 (-10pt+) | calibrator が悪さ、 blend factor を 0.0 (raw のみ) で再 simulate |
| 全 race が 3x (saturation) | top1_score の閾値再設計、 score_sum ベース EV へ移行検討 |
| EV 計算 unstable (odds 不在 etc) | shadow csv に odds_resolved=False で記録、 評価対象外 |

## 8. 次 step (5/18+)

1. 5/18 朝 8:00 daily_predict 後、 8:30 で `save_all_horse_scores.py --date 20260518`
2. 9:00 で `strategy_layer_v2.py --shadow 20260518`
3. 夜 20:30 で paper shadow csv の bet 数 + 累計 inv を 手動 record
4. 翌日 朝 cumulative_results.csv 更新後、 join 集計
5. 1 週間 (5/18-5/25 想定 20 race) 蓄積 後 中間 review
6. 30 race 以上 で 第 1 回 honest 判定

## 9. 関連 file

- `tools/strategy_layer_v2.py`: 本 module (新規)
- `data/calibrator_v15_pilot.pkl`: calibrator (21 sample、 audit 別途)
- `data/v21/strategy_layer_v2_design.md`: design doc
- `data/v21/strategy_layer_calibrator_audit.md`: calibrator audit
- `data/v21/strategy_v2_simulation_report.md`: 5/16 時点 backtest 結果 (未実証)
- `data/daily_predictions_full/`: 全頭 V15 score (5/10+)
- `data/odds_base_YYYYMMDD.csv`: 当日 odds (08:00 baseline)
