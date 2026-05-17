# 比較-3: 5/16 全 R 直前予測 mock simulation 完全版

作成日: 2026-05-17 (土曜深夜〜日曜未明)
作成者: P0-5 mock simulation
基盤: `tools/mock_sim_5_16.py` + `tools/calibrator_overlay.py` (P0-5 順4)

★ V15 production / predict_core 完全不変、 mock data のみ、 read-only、 git op なし ★

---

## 0. 結論

| 項目 | 朝 base | mock -15min | delta |
|---|---|---|---|
| 投票候補 R (全) | 34 | 34 | 0 |
| top1 swap (critical) | — | 4 R | — |
| ranking change (any) | — | 16 R / 34 (47%) | — |
| 朝 base 仮想 trio 7点 ROI | **227.35%** | 227.35% | **±0.00 pt** (記録 payout 同一) |
| 朝 base hits | 12 / 34 | 12 / 34 | ±0 |
| 投票候補 (戦略⑦案 C) | 19 R | 19 R | 0 |
| 案 C 朝 ROI | **104.14%** | 104.14% | **±0.00 pt** |

ただし **mock formation hit 差分は 2 R 発生** (★ 注 ★):
- **morning hit / mock miss**: 1 R (新潟 5R 3歳未勝利、 actual 1-9-11)
- **morning miss / mock hit**: 1 R (東京 5R 3歳未勝利、 actual 2-10-12)

両方とも `daily_results.csv` に actual trio payout が 0 で記録 (daily 投票が miss だった、 もしくは 配当 csv 未更新)。
推定 trio 配当 ヒューリスティック (o1×o2×o3×係数):
- 新潟 5R: 約 3,000〜8,000円 (morning hit 失った場合 -3,000〜-8,000円)
- 東京 5R: 約 350〜1,000円 (mock 取れた場合 +350〜+1,000円)

→ ★ net delta ≈ **-2,650 〜 -7,000円** (★ noise 5% で hit set が変わる 振れ幅、 1日のみ偶然範囲 ★)

★ P0-5 真の期待効果 estimation: **5/24+ live odds で 再評価必須**。 mock noise ± 5% は 実 -15min odds 振れ幅 より小さい可能性 (G1 / 重賞 で ±15-30% も)。

---

## 1. 5/16 全 34 R table

| race_id | 場 | R | レース名 | 条件 | 朝 top3 | mock -15min top3 | severity | 実 1-3着 | 朝 hit | mock hit | 配当 (記録) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 202604010501 | 新潟 | 1 | 3歳未勝利 | C | 4-8-7 | 8-7-4 | critical | 4-7-8 | ★1 | ★1 | 1,040 |
| 202604010502 | 新潟 | 2 | 3歳未勝利 | D | 14-12-3 | 14-12-3 | none | 3-12-14 | ★1 | ★1 | 710 |
| 202604010503 | 新潟 | 3 | 3歳未勝利 | C | 9-8-14 | 9-8-14 | none | 9-11-14 | ★1 | ★1 | 2,680 |
| 202604010504 | 新潟 | 4 | 3歳未勝利 | A | 8-10-11 | 8-10-11 | none | 8-10-11 | ★1 | ★1 | 1,370 |
| 202604010505 | 新潟 | 5 | 3歳未勝利 | C | 1-8-11 | 1-2-8 | **major** | 1-9-11 | ★1 | 0 (★ inv) | 0 (記録 miss) |
| 202604010507 | 新潟 | 7 | 4歳以上1勝クラス | D | 1-3-2 | 1-3-2 | none | 9-12-13 | 0 | 0 | 0 |
| 202604010508 | 新潟 | 8 | 4歳以上1勝クラス | A | 5-9-6 | 9-5-6 | **critical** | 5-7-9 | ★1 | ★1 | 540 |
| 202604010509 | 新潟 | 9 | 石打特別 | C | 12-14-8 | 12-6-14 | major | 6-12-14 | ★1 | ★1 | 1,050 |
| 202604010510 | 新潟 | 10 | 中ノ岳特別 | A | 5-8-4 | 4-5-8 | **critical** | 1-2-8 | 0 | 0 | 0 |
| 202604010511 | 新潟 | 11 | 新潟大賞典 G3 | C | 14-4-6 | 14-4-6 | none | **3-9-11** | 0 | 0 | 0 |
| 202604010512 | 新潟 | 12 | 4歳以上1勝クラス | D | 11-1-5 | 11-16-1 | major | 10-11-13 | 0 | 0 | 0 |
| 202605020701 | 東京 | 1 | 3歳未勝利 | D | 1-13-7 | 1-7-13 | minor | 1-3-10 | 0 | 0 | 0 |
| 202605020702 | 東京 | 2 | 3歳未勝利 | C | 5-3-15 | 5-3-15 | none | 5-13-14 | 0 | 0 | 0 |
| 202605020703 | 東京 | 3 | 3歳未勝利 | D | 12-16-11 | 12-16-10 | major | 10-12-16 | ★1 | ★1 | 7,080 |
| 202605020704 | 東京 | 4 | 3歳未勝利 | A | 13-12-7 | 13-12-8 | major | 4-8-12 | 0 | 0 | 0 |
| 202605020705 | 東京 | 5 | 3歳未勝利 | D | 2-11-12 | 2-12-11 | minor | 2-10-12 | 0 | ★1 (★ inv) | 0 (記録 miss) |
| 202605020706 | 東京 | 6 | 3歳1勝クラス | A | 4-3-8 | 4-3-8 | none | 3-4-8 | ★1 | ★1 | 430 |
| 202605020707 | 東京 | 7 | 3歳1勝クラス | D | 8-3-6 | 8-3-6 | none | 4-9-13 | 0 | 0 | 0 |
| 202605020708 | 東京 | 8 | 4歳以上1勝クラス | C | 15-2-11 | 15-2-11 | none | 1-14-15 | 0 | 0 | 0 |
| 202605020709 | 東京 | 9 | 調布特別 | A | 3-6-1 | 3-6-1 | none | 3-5-7 | 0 | 0 | 0 |
| 202605020710 | 東京 | 10 | 立川特別 | D | 9-12-6 | 9-12-8 | major | 6-8-12 | 0 | 0 | 0 |
| 202605020711 | 東京 | 11 | 六社S | C | 6-7-2 | 6-7-2 | none | **4-7-8** | 0 | 0 | 0 |
| 202605020712 | 東京 | 12 | 4歳以上1勝クラス | A | 10-9-1 | 10-9-1 | none | 4-9-11 | 0 | 0 | 0 |
| 202608030701 | 京都 | 1 | 3歳未勝利 | D | 14-12-8 | 14-8-12 | minor | 4-5-8 | 0 | 0 | 0 |
| 202608030702 | 京都 | 2 | 3歳未勝利 | A | 9-6-11 | 9-6-11 | none | 1-6-9 | ★1 | ★1 | 1,560 |
| 202608030703 | 京都 | 3 | 3歳未勝利 | D | 4-18-16 | 4-18-16 | none | 2-13-18 | 0 | 0 | 0 |
| 202608030704 | 京都 | 4 | 3歳未勝利 | A | 6-8-7 | 6-8-7 | none | 2-6-9 | 0 | 0 | 0 |
| 202608030705 | 京都 | 5 | 3歳未勝利 | A | 2-12-4 | 2-4-12 | minor | 2-5-14 | 0 | 0 | 0 |
| 202608030706 | 京都 | 6 | 4歳以上1勝クラス | D | 5-13-7 | 5-13-7 | none | 5-7-14 | 0 | 0 | 0 |
| 202608030707 | 京都 | 7 | 4歳以上1勝クラス | A | 1-5-3 | 1-5-6 | major | 3-5-6 | 0 | 0 | 0 |
| 202608030709 | 京都 | 9 | あずさ賞 | A | 3-6-4 | 3-4-6 | minor | 1-2-8 | 0 | 0 | 0 |
| 202608030710 | 京都 | 10 | **上賀茂S** | C | 13-10-1 | 13-10-1 | none | **1-13-14** | ★1 | ★1 | **33,200 (★ windfall)** |
| 202608030711 | 京都 | 11 | 鞍馬S | D | 13-16-15 | 13-12-16 | major | 3-6-12 | 0 | 0 | 0 |
| 202608030712 | 京都 | 12 | 4歳以上2勝クラス | D | 10-2-13 | 10-2-13 | none | 10-13-15 | ★1 | ★1 | 4,450 |

★ inv = formation hit 結果 違うが 記録 daily_results.csv に payout なし。

---

## 2. 統計

| metric | 朝 base | mock -15min | delta |
|---|---:|---:|---:|
| 投票候補 R | 34 | 34 | 0 |
| ranking change R | — | 16 / 34 (47.1%) | — |
| critical (top1 swap) | — | 4 R | — |
| major (top1 same, 2/3 set diff) | — | 8 R | — |
| minor (top1 same, swap only) | — | 5 R | — |
| none | — | 18 R | — |
| formation hits | 12 | 12 | ±0 (記録 base) |
| total invest | 23,800 円 | 23,800 円 | 0 |
| total recorded payout | 54,110 円 | 54,110 円 | ±0 |
| ROI | **227.35%** | **227.35%** | **±0.00 pt** |
| 純益 | +30,310 円 | +30,310 円 | ±0 |

★ 朝 ROI 227% は 5/16 偶然好調 (上賀茂S 33,200円 windfall が大半) ★

---

## 3. 戦略⑦案 C interaction (京都 + 06_特別 除外)

| metric | 朝 base | mock -15min |
|---|---:|---:|
| candidate R | 19 | 19 |
| hits | 8 | 8 |
| invest | 13,300 円 | 13,300 円 |
| payout | 13,850 円 | 13,850 円 |
| ROI | **104.14%** | 104.14% |
| 純益 | +550 円 | +550 円 |

★ 案 C で 上賀茂S (京都) windfall を 失うため ROI 急減 ★
★ 京都除外は 5/11 以降 再評価対象だが、 5/16 は 案 C 適用すると ほぼ break-even ★

---

## 4. 重賞個別 verdict

### 4-1. 新潟大賞典 G3 (新潟11R)
- 朝 V15 top3: **14-4-6**
- mock -15min top3: 14-4-6 (順位変動なし)
- 実 1-3着: **3-9-11**
- verdict: **完全外し** (V15 14 番、 4 番、 6 番 いずれも 着外)。 mock noise ± 5% でも market divergence 出ず。
- ★ post-mortem: V15 が 3 番 / 9 番 / 11 番 を 見抜けず、 P0-5 でも mock noise 範囲では 救えない。 真の paddock / 馬体重 情報 必要 (5/24+ 実 fetch + V21 video で 検証) ★

### 4-2. 六社S (東京11R オープン特別)
- 朝 V15 top3: **6-7-2**
- mock -15min top3: 6-7-2 (順位変動なし)
- 実 1-3着: **4-7-8**
- verdict: top2 (7番) 命中、 top1/top3 外し。 mock simulation でも改善なし。

### 4-3. 上賀茂S (京都10R) ★ windfall ★
- 朝 V15 top3: **13-10-1**
- mock -15min top3: 13-10-1 (順位変動なし)
- 実 1-3着: **1-13-14**
- 配当: **33,200 円** (trio hit、 ★ 1日 jackpot ★)
- verdict: 朝 V15 top1=13 (実 2着) + top3=1 (実 1着)、 top6 に 14 番 含まれていた為 formation hit。 ★ mock simulation でも 朝 base でも 同一 ★。
- ★ jackpot 取り込み: 朝 base / mock どちらも捕獲、 ranking change なし ★

---

## 5. P0-5 真の期待効果 estimation (mock base)

### 5-1. mock simulation 結果
- noise ±5% で **16 R / 34 (47%)** が ranking 変動 (severity none 以外)
- うち **critical (top1 swap) 4 R** (12%)
- formation hit 差分: 2 R (1 改善 + 1 改悪 = 偶然差)
- 記録 payout 基準で **delta ROI 0.00 pt**

### 5-2. 推定 (heuristic trio payout)
- mock 改善 1 R (東京 5R、 2-10-12): est +350〜+1,000円
- mock 改悪 1 R (新潟 5R、 1-9-11): est -3,000〜-8,000円
- **net est: -2,650 〜 -7,000円 (1日のみ偶然範囲)**

### 5-3. ★ 1日のみ noise 範囲 / honest 注記 ★
- noise std 5% は assumption、 実 -15min odds 振れ幅 は data 取得後検証必要
- 5/16 のみ 1 day = 統計的有意性なし
- **真の検証: 5/24+ live odds fetch + 1 ヶ月 paper trading で P0-5 真効果 計測**
- target value: P0-5-B 設計時 想定 ROI 改善 **+2-5pt**、 ただし 5/16 mock では達成できず

---

## 6. mock simulation 限界

| 項目 | 内容 |
|---|---|
| noise scale | ± 5% Gaussian、 実 -15min odds 振れ幅 (G1/重賞は ±15-30% も) より小さい |
| 直前情報なし | 馬体重 / 馬場急変 / パドック気配 / 出走取消 等の真の情報なし |
| weight_diff = 0 | overlay の weight 補正 効かず、 odds_shift のみで delta 計算 |
| seed 固定 (42) | 1 seed のみ、 noise 多様性 限定的 (Monte Carlo 1000 試行で改善余地あり) |
| formation 構築 | top1 軸 + top2-3 + top4-6 の simplified、 実 predict_core の trio_bets_str 完全再現は別途 |
| 配当 | 記録 (daily_results.csv) base、 mock-only hit の actual trio payout は heuristic 推定 |

→ **5/24+ 実 -15min direct odds fetch で 真効果 検証**。 mock は 「P0-5 mechanism は動作する」 検証のみ。

---

## 7. V15 production 不変保証 ✅

- 本 simulation は **read-only** (daily_predictions_full + daily_results 参照のみ)
- `predict_core.py` / `keiba_model_v135b_*.pkl.gz` 一切 touch せず
- mock data のみ生成 (numpy seed=42、 確定 reproducible)
- 出力先: `data/v21/mock_sim_5_16_detail.csv` + `data/v21/mock_sim_5_16_summary.json` (新規 + V21 領域)
- ★ V15 production 投票判断 影響なし、 v15.2 training (PID 23528) 完全不変 ★

---

## 8. 完了通知

```
比較-3 完了、 5/16 改善 R = 1 (東京 5R) / 改悪 R = 1 (新潟 5R)、
記録 payout base ROI delta = 0.00 pt、 P0-5 期待効果 = net -2.6k 〜 -7k 円
推定 (mock base、 1 day noise 範囲 / 5/24+ live fetch で 検証必要)
```

★ 1日のみ偶然範囲、 noise ±5% は assumption、 真の P0-5 効果は 5/24+ paper eval で計測 ★
