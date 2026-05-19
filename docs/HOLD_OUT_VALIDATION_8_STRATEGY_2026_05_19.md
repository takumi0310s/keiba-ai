# 8 Strategy Hold-out Validation (2026-05-19)

## 概要

`data/cumulative_results.csv` n=662 settled レースを期間分割 3 種類 × 4 strategy (+ 4 paper-only) で hold-out validation。
目的: 全期間 in-sample での C3+C4 +23.8pt (D-0 audit) が真の signal か overfitting か最終 verify。

---

## 1. データ確認

| 項目 | 値 |
|------|-----|
| Total rows | 663 (status='settled' 662, status='20260505' 1 row excluded) |
| Settled N | **662** |
| strat_c filter (cond A/C/D + 非京都) | **515** |
| 期間 | 2026-03-14 ~ 2026-05-17 |
| top4_num 有効行 (strat_c 内) | **465 / 515** |
| top4_num 不明行 (C3 不適用) | **50 / 515** (投資 700 円維持) |
| bet2_only_hit 件数 (全期間) | **2** (行 idx=7: 1780円, idx=642: 4900円) |
| investment 列 | 全行 700 円 (strat_c 内, umaren 行なし) |
| strat_c 列 | なし (venue_code + condition でフィルタ再構築) |

### 月別 baseline ROI (strat_c)

| 月 | N | ROI | PnL |
|----|---|-----|-----|
| 3月 (3/14-3/31) | 162 | 81.0% | -21,570 |
| 4月 | 214 | 139.6% | +59,330 |
| 5月 (5/1-5/17) | 139 | 67.9% | -31,220 |

---

## 2. 各 strategy 定義

| strategy | 定義 | データ要件 |
|----------|------|-----------|
| **baseline** | strat_c = cond A/C/D + 非京都 | — |
| **C4** | baseline + Cond-A 1600-1800m を skip | distance 既知のみ適用 |
| **C3** | baseline + bet2 (T1-T2-T4) を除外: invest 700→600, bet2_only_hit → payout=0 | top4_num 既知のみ invest 減額 |
| **C3+C4** | C3 AND C4 同時適用 | 両方 |
| no_1pop / divergence / ev_filter / odds_filter | paper eval 専用 (race_notify_log_v2 データ未蓄積) | **N/A** |

### C3 メカニズム詳細

- top4_num 有効 465 行 → 投資 700 → 600 円 (1 行あたり -100 円)
- 投資削減総額: 465 × 100 = **46,500 円** (全期間)
- bet2_only_hit 2 件 → payout 喪失: 1,780 + 4,900 = **6,680 円**
- 純コスト削減: 46,500 - 6,680 = **+39,820 円** のコスト節約効果
- ROI 算出: payout は -6,680 円、 invest は -46,500 円 → 低 payout レースでは ROI 向上

---

## 3. 全期間結果 (n=515)

| strategy | N | ROI | PnL | hits | hit_rate | delta vs baseline |
|----------|---|-----|-----|------|----------|-------------------|
| **baseline** | **515** | **101.81%** | **+6,540** | **124** | **24.1%** | — |
| C4 | 454 | 109.74% | +30,950 | 107 | 23.6% | **+7.93pt** |
| C3 | 515 | 114.76% | +46,360 | 122 | 23.7% | **+12.95pt** |
| **C3+C4** | **454** | **123.31%** | **+64,670** | **105** | **23.1%** | **+21.50pt** |

注: D-0 doc の C3+C4 +23.76pt と微差があるのは C3 investment 調整の精密化による (D-0 は C3 payout 削減のみ、本稿は investment も正確に 600 円適用)。

---

## 4. 期間分割 3 種類 (Hold-out Validation)

### Split 1: train ≤2026-05-03 vs test ≥2026-05-04

| 期間 | strategy | N | ROI | PnL | delta vs baseline |
|------|----------|---|-----|-----|-------------------|
| **TRAIN** (n=407) | baseline | 407 | 108.66% | +24,660 | — |
| TRAIN | C4 | 368 | 115.58% | +40,140 | +6.92pt |
| TRAIN | C3 | 407 | 123.51% | +58,580 | +14.85pt |
| TRAIN | **C3+C4** | **368** | **131.07%** | **+70,160** | **+22.41pt** |
| **TEST** (n=108) | baseline | 108 | 76.03% | -18,120 | — |
| TEST | C4 | 86 | 84.73% | -9,190 | **+8.70pt** |
| TEST | C3 | 108 | 81.14% | -12,220 | **+5.11pt** |
| TEST | **C3+C4** | **86** | **89.36%** | **-5,490** | **+13.33pt** |

### Split 2: train ≤2026-04-30 vs test ≥2026-05-01

| 期間 | strategy | N | ROI | PnL | delta vs baseline |
|------|----------|---|-----|-----|-------------------|
| **TRAIN** (n=376) | baseline | 376 | 114.35% | +37,760 | — |
| TRAIN | C4 | 345 | 120.24% | +48,870 | +5.89pt |
| TRAIN | C3 | 376 | 129.74% | +68,580 | +15.39pt |
| TRAIN | **C3+C4** | **345** | **136.13%** | **+76,590** | **+21.78pt** |
| **TEST** (n=139) | baseline | 139 | 67.91% | -31,220 | — |
| TEST | C4 | 109 | 76.51% | -17,920 | **+8.60pt** |
| TEST | C3 | 139 | 73.36% | -22,220 | **+5.45pt** |
| TEST | **C3+C4** | **109** | **81.77%** | **-11,920** | **+13.86pt** |

### Split 3: train ≤2026-05-10 vs test ≥2026-05-11

| 期間 | strategy | N | ROI | PnL | delta vs baseline |
|------|----------|---|-----|-----|-------------------|
| **TRAIN** (n=471) | baseline | 471 | 104.57% | +15,070 | — |
| TRAIN | C4 | 421 | 111.72% | +34,550 | +7.15pt |
| TRAIN | C3 | 471 | 119.26% | +55,390 | +14.69pt |
| TRAIN | **C3+C4** | **421** | **127.12%** | **+69,870** | **+22.55pt** |
| **TEST** (n=44) | baseline | 44 | 72.31% | -8,530 | — |
| TEST | C4 | 33 | 84.42% | -3,600 | **+12.11pt** |
| TEST | C3 | 44 | 65.80% | -9,030 | **-6.51pt** |
| TEST | **C3+C4** | **33** | **73.74%** | **-5,200** | **+1.43pt** |

#### Split 3 C3 test delta 負の理由

Split 3 test (n=44, 5/11-5/17) では bet2_only_hit が 1 件 (payout 4,900 円) 発生。
- 全 44 行が top4_num 有効 → investment 700→600 (合計 -4,400 円)
- bet2_only_hit 1 件: payout 4,900 → 0 円
- 純効果: payout -4,900 円 vs invest -4,400 円 → net negative
- 小 N (n=44) × 高配当 bet2 ヒット 1 件のランダムノイズ (期待値ではなく 1 標本の偶然)

---

## 5. C3+C4 真の verdict

| 計算方法 | C3+C4 delta | 備考 |
|---------|-------------|------|
| in-sample 全期間 n=515 | **+21.50pt** | 本稿 (D-0: +23.76pt は旧計算) |
| Split 1 OOS test (n=108) | **+13.33pt** | 5/4-5/19 |
| Split 2 OOS test (n=139) | **+13.86pt** | 5/1-5/19 |
| Split 3 OOS test (n=44) | **+1.43pt** | 5/11-5/19 (小 N) |
| **OOS 平均 delta (3 split)** | **+9.5pt** | |

**C4 単体 OOS delta**:
| Split | OOS delta |
|-------|-----------|
| Split 1 | +8.70pt |
| Split 2 | +8.60pt |
| Split 3 | +12.11pt |
| **平均** | **+9.8pt** |

**C3 単体 OOS delta**:
| Split | OOS delta |
|-------|-----------|
| Split 1 | +5.11pt |
| Split 2 | +5.45pt |
| Split 3 | -6.51pt (小 N ノイズ) |
| **平均** | **+1.4pt** |

---

## 6. 過学習チェック (train delta vs test delta)

| Split | strategy | train delta | test delta | 倍率 | 判定 |
|-------|----------|------------|-----------|------|------|
| Split 1 | C4 | +6.92pt | +8.70pt | 0.80x | ✅ 過学習なし (test > train) |
| Split 2 | C4 | +5.89pt | +8.60pt | 0.68x | ✅ 過学習なし (test > train) |
| Split 3 | C4 | +7.15pt | +12.11pt | 0.59x | ✅ 過学習なし (test > train) |
| Split 1 | C3+C4 | +22.41pt | +13.33pt | 1.68x | ⚠️ 要注意 (train > test) |
| Split 2 | C3+C4 | +21.78pt | +13.86pt | 1.57x | ⚠️ 要注意 (train > test) |
| Split 3 | C3+C4 | +22.55pt | +1.43pt | 15.8x | ⚠️ Split 3 は N=44 小 N 注意 |

**解釈**:
- **C4**: 全 split でテストがトレーニングを上回る → 過学習なし、経済合理性あり (Cond-A 1600-1800m の低 ROI は構造的)
- **C3+C4**: train delta > test delta だが OOS での +9.5pt は正の方向を維持 → 軽度の in-sample 過大評価あり (23pt → 9.5pt OOS)、しかし符号は一貫

---

## 7. 統計的有意性 (全期間 n=515)

### Bootstrap 95% CI (n_boot=10,000)

| strategy | ROI | 95% CI | 100% 含む | 判定 |
|----------|-----|--------|-----------|------|
| baseline | 101.81% | [68.2%, 144.9%] | ✅ | N.S. |
| C4 | 109.74% | [71.6%, 157.1%] | ✅ | N.S. |
| C3 | 114.76% | [76.2%, 163.9%] | ✅ | N.S. |
| C3+C4 | 123.31% | [79.8%, 177.6%] | ✅ | N.S. |

全 CI が 100% を含む → **いずれも統計的に有意な優位性なし** (n=515 は検定力不足)

### Welch t-test (C4 kept vs excluded races)

- t = 1.099, **p = 0.2725** → N.S. (有意差なし)
- ただし effect direction は一貫 (除外対象レースの ROI が著しく低い: 42.8%)

---

## 8. Power Analysis (paper eval 5/24-6/16 用)

per-race return の標準偏差 σ = 4.47 (baseline, n=515)

| N レース | MDE (β=0.8, α=0.05 片側) |
|---------|--------------------------|
| 24 | ±226.7pt ROI |
| 50 | ±157.1pt ROI |
| 100 | ±111.1pt ROI |
| 150 | ±90.7pt ROI |
| 200 | ±78.5pt ROI |

| 検出したい delta | 必要 N (β=0.8) |
|----------------|----------------|
| +5pt (最小意義) | **49,347** |
| +21.5pt (in-sample C3+C4) | **2,669** |
| +9.5pt (OOS 平均 C3+C4) | **12,080** |

**結論**: paper eval 4 週末 N≈120-150 では統計的有意性検出は不可能。
paper eval の目的は「大きな逆転 (< -10pt) がないこと確認」と「方向確認」に限定。

---

## 9. paper strategy (N/A 説明)

以下 4 strategy は `race_notify_log_v2` にデータが未蓄積のため hold-out 計算不可:

| strategy | 説明 | 状態 |
|----------|------|------|
| no_1pop | V15 top1 が市場 1 番人気のとき skip | **N/A** (pop_rank 列なし in cumulative_results.csv) |
| divergence | top1_pop_rank ≥ 3 のみ bet | **N/A** (同上) |
| ev_filter | EV 閾値フィルタ | **N/A** (EV 列なし) |
| odds_filter | odds band フィルタ (1.5-20.0) | **N/A** (odds 列なし) |

これら 4 strategy は 5/24 paper eval 開始後のデータで評価予定 (race_notify_log_v2_summary 蓄積後)。

---

## 10. honest verdict per strategy

| strategy | full in-sample ROI | OOS 平均 delta | 方向一貫性 | 過学習 | 5/24 paper 推奨 |
|----------|-------------------|---------------|-----------|-------|----------------|
| baseline (strat_c) | 101.81% | — | — | — | ✅ 継続 |
| **C4** | 109.74% (+7.93pt) | **+9.8pt** | ✅ 全 3 split 正 | ✅ なし | ✅ **採用 GO** (既に production 適用済) |
| C3 | 114.76% (+12.95pt) | **+1.4pt** | ⚠️ Split 3 で逆転 | ⚠️ 軽度 | ⚠️ 要 paper eval (Split 3 に高配当 bet2 hit ノイズ混入) |
| **C3+C4** | 123.31% (+21.50pt) | **+9.5pt** | ✅ 全 3 split 正 | ⚠️ 軽度 (in-sample の半分程度が OOS) | ✅ **paper eval 継続** (方向は維持) |
| no_1pop | N/A | N/A | N/A | N/A | 📋 paper eval で初評価 |
| divergence | N/A | N/A | N/A | N/A | 📋 paper eval で初評価 |
| ev_filter | N/A | N/A | N/A | N/A | 📋 paper eval で初評価 |
| odds_filter | N/A | N/A | N/A | N/A | 📋 paper eval で初評価 |

---

## 11. 6/17 採用判定への推奨値

### C4 (既に production 適用)
- 確認: OOS +9.8pt 平均、全 split 正方向 → **6/17 での継続確定推奨**
- paper eval での逆転リスク: Cond-A 1600-1800m の低 ROI (36.4%) は構造的 → 大きな逆転なし見込み

### C3 採用基準
- paper eval 目標: C3 delta ≥ 0pt (マイナスが 3 週以上連続で NOT-GO)
- 注意: 1 件の高配当 bet2 ヒットで大きく振れる → 最低 N = 60R 必要
- 推奨: paper eval 6/15 時点で delta ≥ 0pt → 採用、< -10pt → 廃止

### C3+C4 採用基準
- paper eval 目標 delta: **+5pt 以上** (OOS 平均 +9.5pt の半分)
- 最低観測 N: 24R (検定力は低いが大崩れ検知に十分)
- 6/17 GO 条件: paper ROI ≥ baseline + 5pt AND 連続 miss ≤ 3 週

### 一般推奨
- **paper eval は検定力目的でなく「大崩れの早期検知」目的**
- 4 週 × 30R = 120R では +9.5pt delta の統計的有意性検証は不可能
- 逆に -20pt 以上の悪化 (C3+C4 ROI < baseline - 10pt) を 2 週以上確認した場合のみ NOT-GO

---

## 12. 出典・計算環境

- data: `data/cumulative_results.csv` (663 rows, status='settled' 662 rows 使用)
- strat_c filter: condition in ['A','C','D'] AND venue_code != '08' (京都) → n=515
- C4: distance 既知行のみ適用 (null distance 行は skip せず include)
- C3: top4_num 有効行 465/515 で investment 600 円、 bet2_only_hit 2 行で payout=0
- Bootstrap: n_boot=10,000, seed=42
- 計算日: 2026-05-19
- 参照 docs: D-0 (docs/D0_C3C4_N662_REBACKTEST_2026_05_19.md)
