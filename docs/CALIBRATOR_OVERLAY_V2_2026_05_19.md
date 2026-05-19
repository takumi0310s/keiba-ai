# CALIBRATOR OVERLAY V2 — 設計・評価レポート
**作成日**: 2026-05-19  
**タスク**: 強-5 calibration improvement  
**ステータス**: paper only — production 投入は 6/17 採用判定後  

---

## 1. 既存 calibrator_overlay.py の現状

`tools/calibrator_overlay.py` (P0-5 順4、commit 存在) は以下の overlay 方式を実装済み:

| 項目 | 内容 |
|------|------|
| 方式 | heuristic overlay (parametric delta) |
| 補正源 | -15 min odds shift + 馬体重変化 |
| delta cap | ±0.10 |
| V15 prob | **不変** (別 layer で加算) |
| 既存 pkl | `data/calibrator_v15_pilot_v2.pkl` (315 sample retrain、Isotonic framework) |
| 特徴 | post-hoc delta 加算方式。数式ベースで解釈性が高い |

**課題**: 既存は rule-based heuristic。odds/体重以外の系統的 miscalibration を補正できない。

---

## 2. データ状況 audit (2026-05-19)

### cumulative_results.csv (2026-03-14 〜 2026-05-17)

| 項目 | 値 |
|------|-----|
| 総行数 | 663 |
| settled 行数 | 662 |
| score + label が揃う有効行 | **153** |
| score 範囲 | 0.139 〜 0.832 |
| score 平均 | 0.522 |
| top-3 hit rate (finish<=3) | 0.627 |
| 観測会場 | 阪神(48) / 東京(47) / 新潟(38) / 中山(12) / 中京(8) |
| odds 列 | **なし** (cumulative_results.csv に未保存) |

**注記**: top1_score は V15 の P(finish<=3) 出力と推定される (平均 0.52、実 hit rate 0.63 → 平均的に under-prediction)。

### 既存 calibration curve (8 bins)

| 予測値 | 実 hit rate | 過/不足 |
|--------|------------|--------|
| 0.169 | 0.684 | 過小 |
| 0.201 | 1.000 | 過小 |
| 0.345 | 0.583 | 過大 |
| 0.461 | 0.538 | 過大 |
| 0.551 | 0.676 | 過小 |
| 0.645 | 0.697 | 過大 |
| 0.733 | 0.500 | 過大 |
| 0.818 | 0.667 | 過大 |

→ 低スコア域 (<0.3) で under-prediction、高スコア域 (>0.6) で over-prediction。  
→ isotonic regression が補正に有効な形状。

---

## 3. 3 calibration method 設計

### a. IsotonicCalibrator (global)

- **実装**: `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip', increasing=True)`
- **特徴**: non-parametric、monotonic 制約あり、柔軟な曲線フィット
- **vs Platt scaling**: Platt は 2-param sigmoid (少量データで安定)。153 サンプルでは Platt の方が汎化しやすいケースもあるが、isotonic の方が形状自由度が高く calibration curve の山谷に対応可能
- **インターフェース**: `fit(y_pred, y_true)` / `predict(y_pred)` / `save()` / `load()`

### b. VenueCalibrator (per-場)

- **設計**: 会場ごとに別の IsotonicCalibrator を保持、未知会場は global fallback
- **最低サンプル数**: 30 (現状 30 以上: 阪神48 / 東京47 / 新潟38)
- **期待効果**: 会場ごとの track bias (クッション値傾向、内外有利等) を吸収
- **本格 fit 推奨時期**: 2026-06 中旬 (300+ サンプル蓄積後)

### c. OddsBinCalibrator (per-odds bin)

- **設計**: 単勝オッズ帯別 ({<1.5 / 1.5-3.0 / 3.0-5.0 / 5.0-10.0 / 10.0-20.0 / 20.0+}) で別 calibrator
- **最低サンプル数**: 20 per bin
- **現状**: cumulative_results.csv に odds 列なし → **設計のみ実装済み**
- **データ取得方法**: `odds_base_*.csv` または `race_notify_log_v2` からのマージが必要
- **期待効果**: 高オッズ馬 (10倍+) の過小評価 / 低オッズ馬 (<3倍) の過大評価を補正

---

## 4. 評価結果

### Brier score / log loss (N=153)

| 指標 | Raw score | 5-fold CV Isotonic | delta |
|------|-----------|---------------------|-------|
| Brier score | 0.2836 | **0.2496** | **-0.034** (改善) |
| Log loss | 0.7788 | 0.8439 | +0.066 (悪化) |

**解釈**:
- Brier score は -0.034 改善 (CV 評価)。確率の二乗誤差が小さくなる
- Log loss は悪化。Isotonic が極端値 (0 or 1 近辺) に fit → log(near-0) penalty が増大
- この divergence は N=153 での過学習リスクを示す。サンプル増加で収束見込み

### hold-out 検証 (train: ~4/30, test: 5/1~5/17)

| 指標 | Raw score | Isotonic (train=20 samples) | delta |
|------|-----------|------------------------------|-------|
| Brier | 0.2524 | 0.3835 | +0.131 (大幅悪化) |
| Log loss | 0.7007 | 5.298 | +4.597 (大幅悪化) |

**解釈**: 20 サンプルでの isotonic fit は test set に過適合。  
→ hold-out の時系列分割は現時点では信頼できない。**CV 評価 (5-fold) が現時点での唯一の信頼指標**。

### top-3 / top-5 hit rate への影響

- 現状 cumulative_results.csv は top-1 predicted horse の finish のみ記録
- **top-3 hit rate (top1 finish<=3)**: 0.627 (raw)
- calibration は確率の精度改善であり、ranking 順序は変わりにくい → top-N hit rate への直接改善は限定的
- ランキング変動が起こるのは 2 馬の score 差が小さいケースのみ
- **正直な推定**: Brier -0.034 改善は calibration の精度向上だが、top-5 hit rate の変化は ±0-1% 程度と推定 (OOS paper eval で確認必要)

---

## 5. honest verdict

| 項目 | 評価 |
|------|------|
| **真の signal** | CV Brier -0.034 は存在するが **N=153 では統計的不確かさ大** |
| **hold-out** | 時系列分割では train=20 で信頼不能 → **3-4 か月データ蓄積が前提** |
| **per-venue** | 阪神/東京/新潟 は fit 可能 (各 38-48)、中山/中京 は未到達 |
| **per-odds-bin** | odds データ未取得 → **設計のみ、data 蓄積後** |
| **top-5 hit rate 改善** | calibration alone では ranking 変動小 → **±0-1% 推定、paper eval 必須** |
| **production 投入推奨** | **6/17 採用判定後**、それまでは paper eval のみ |

**結論**: IsotonicCalibrator は CV Brier で改善信号あり。  
ただし N=153 は過学習リスクが高く、hold-out の時系列分割では悪化した。  
5/24+ paper eval (実レースへの shadow 適用) で OOS 確認後に採用判定。

---

## 6. 6/17 採用判定 path

| ステップ | 時期 | 内容 |
|---------|------|------|
| paper eval 開始 | 5/24+ | `calibrate_v2(scores, method='isotonic')` を shadow でレース毎に記録 |
| odds データ取得 | 6/1+ | `odds_base_*.csv` または `race_notify_log_v2` から odds マージ |
| OddsBinCalibrator fit | 6/7+ | odds 蓄積後 (各 bin 20+ サンプル) |
| per-venue 追加評価 | 6/7+ | 全会場 30+ サンプル達成後 |
| 採用判定 | 6/17 | paper eval ROI + hit rate + Brier で V15 raw を上回るか確認 |
| 条件 | — | OOS Brier < raw / hit rate +1pt 以上 / paper eval n≥200 |

**採用条件 (6/17 判定基準)**:
- OOS (paper eval) Brier ≤ 0.270 (raw 0.2836 から -0.013 以上)
- top-3 hit rate ≥ 0.640 (raw 0.627 + 1.3pt)
- paper eval n ≥ 200 races
- V15 .pkl.gz / predict_core / app.py 一切改変なし

---

## 7. ファイル一覧

| ファイル | 役割 |
|---------|------|
| `tools/calibrator_overlay_v2.py` | IsotonicCalibrator / VenueCalibrator / OddsBinCalibrator 実装 |
| `tests/test_calibrator_overlay_v2.py` | 10 tests (全 PASS 確認済) |
| `data/v21/calibrator_v2_isotonic.pkl` | fit 後の保存先 (--fit-isotonic で生成) |
| `data/v21/calibrator_v2_venue.pkl` | fit 後の保存先 (--fit-venue で生成) |
| `data/v21/calibrator_v2_oddsbin.pkl` | fit 後の保存先 (odds データ取得後) |

### CLI 使用例

```bash
# CV 評価のみ (cumulative_results.csv を使用)
python tools/calibrator_overlay_v2.py --eval-cumulative

# IsotonicCalibrator を全データで fit + save
python tools/calibrator_overlay_v2.py --fit-isotonic

# VenueCalibrator を全データで fit + save
python tools/calibrator_overlay_v2.py --fit-venue

# 結果を JSON 出力
python tools/calibrator_overlay_v2.py --eval-cumulative --output-json data/v21/calibrator_eval.json
```
