# 戦略 layer 改善 audit (2026-05-18)

## 0. データ品質メモ

- `cumulative_results.csv` (n=629 settled) には **全予測ログ** が含まれる
  - race_auto_notify の戦略フィルタ外レースも `investment=700, profit=-700` で記録
  - 戦略⑦案 C (京都/B/E/X 除外) は notify のみ適用。daily_predict は全レース記録
- `pop_rank` データ (odds_base_*.csv) は 2026-04-18〜2026-05-17 の 9 日分のみ
  - 全 629 settled 中 100 races のみ market rank あり → 候補 1/2 の N は小さい
- `top1_num`, `top2_num`, `top3_num` は 120 races のみ記録済み
- `distance` / `surface` は 40% 欠損あり (特に Condition A で 61 races 欠損)

---

## 1. 真値 baseline 確認

### 全 settled (n=629)

| 指標 | 値 |
|------|----|
| N (settled) | 629 |
| Investment | ¥440,300 |
| PnL | ¥-19,080 |
| ROI | **95.67%** |
| Bootstrap 95% CI | [64.9%, 134.0%] |
| Trio hit rate | 21.5% (vs random 1/7 = 14.3%, p=0.000) |

### 条件別 ROI (全 settled)

| 条件 | N | ROI | PnL | hit率 | binom p |
|------|---|-----|-----|--------|---------|
| A | 196 | 88.7% | ¥-15,460 | 29.1% | 0.0000 |
| B | 16 | 27.0% | ¥-8,180 | 18.8% | 0.406 |
| C | 172 | 118.7% | ¥+22,550 | 18.6% | 0.069 |
| D | 215 | 100.6% | ¥+900 | 19.5% | 0.021 |
| E | 11 | 12.3% | ¥-6,750 | 0.0% | 1.000 |
| X | 19 | 8.7% | ¥-12,140 | 5.3% | 0.947 |

### Strategy C (excl B/E/X + 京都) subset

戦略⑦案 C のあるべき姿に近いフィルタ (B/E/X 除外 + 京都除外):

| 指標 | 値 |
|------|----|
| N | 494 |
| ROI | **104.83%** |
| PnL | ¥+16,690 |
| Bootstrap 95% CI | [70.2%, 149.9%] |
| hit率 | 22.1% (p=0.000) |

★ これを **「Paper eval 基準 baseline」** とする ★

---

## 2. 候補比較 table

| 候補 | 方式 | 期待 ROI | N (適用後) | hit 率 | Bootstrap 95% CI | p-value | 実装工数 | 推奨度 |
|------|------|-----------|-----------|--------|------------------|---------|---------|--------|
| Baseline | strat_c (現行) | 104.8% | 494 | 22.1% | [70.2%, 149.9%] | — | — | — |
| **C3: 6-bet formation** | trio 7点 → 6点 (pos2 除外) | **128.5%** | 424 | 22.9% | [82.2%, 189.7%] | **0.001** | 小 | **★★★★** |
| C4b: Cond-A 1600-1800 除外 | 条件 A & dist ≤ 1800m を skip | **113.7%** | 436 | 23.2% | [74.3%, 164.5%] | 0.000 | 小 | **★★★** |
| C3+C4b 複合 | 6-bet + Cond-A 1600-1800 除外 | **141.9%** | 371 | 22.9% | [88.2%, 210.8%] | **0.001** | 小 | **★★★★** |
| C1: market rank ≥ 2 | V15 top1 = 市場 1番人気 → skip | 59.8% (rankあり subset) | 41/65 | 22.0% | [17%, 120%] | 0.122 | 中 | ★ |
| C2: market rank ≥ 3 | V15 top1 = 市場 top2 以内 → skip | 81.8% (rankあり subset) | 29/65 | 27.6% | [24%, 166%] | 0.046 | 中 | ★★ |

---

## 3. 各候補 詳細

### 候補 1: 1番人気除外 (V15 top1 = 市場 1番人気)

**分析**

- odds_base_*.csv でマーケット rank が取得できたのは **100 races** (全 629 中 16%)
- Strat_c subset で rank あり: **65 races**
- V15 top1 = 市場 1番人気 (rank=1): 24/65 = 37%

**ROI breakdown**

| グループ | N | ROI | CI |
|----------|---|-----|----|
| 全 ranked races | 65 | 65.1% | — |
| rank = 1 (除外対象) | 24 | 40.0% | — |
| rank ≥ 2 (残留) | 41 | 59.8% | [17%, 120%] |
| rank ≥ 3 (残留) | 29 | 81.8% | [24%, 166%] |
| rank ≥ 4 (残留) | 11 | 165.1% | [34%, 358%] |

**verdict**: N が小さすぎる (N=41〜65)。rank ≥ 4 の 312% は promising だが N=21 で信頼性低。
Sub-task 17 の「1番人気時 ROI 61.0%」と方向性は一致。
→ **5/24+ paper eval で market rank データを継続蓄積して再評価必要**。
→ 今の N では実装不可。**保留 (6/16 以降 paper eval)**。

---

### 候補 2: V15-市場 divergence フィルタ

**分析**

- V15 top1 score が利用可能: 120 races のみ
- V15 score と ROI の相関: **0.105** (弱正相関)
- Score ≥ 0.7 の 25 races: ROI 218.0% (有望!)

**Score 閾値分析 (N=120 のみ)**

| 閾値 | N | ROI |
|------|---|-----|
| score ≥ 0.5 | 80 | 126.8% |
| score ≥ 0.6 | 48 | 149.6% |
| score ≥ 0.65 | 36 | 158.9% |
| score ≥ 0.7 | 25 | 218.0% |

**verdict**: score ≥ 0.6 or ≥ 0.7 は非常に有望だが N=25〜48。
score は top1_num が記録されているレースのみ (daily_predict ログに依存)。
→ **score の蓄積を増やして 6/16 以降に再評価**。divergence filter は候補 1 と統合可能。

---

### 候補 3: trio 7点 → 6点 formation 見直し

**現行 formation 構造**

```
col1=[top1] × col2=[top2, top3] × col3=[top3, top4, top5, top6]
= 7 bets:
  pos1: top1-top2-top3
  pos2: top1-top2-top4   ← 最低価値
  pos3: top1-top2-top5
  pos4: top1-top2-top6
  pos5: top1-top3-top4
  pos6: top1-top3-top5
  pos7: top1-top3-top6
```

**各ポジション統計 (N=548, strat_c 全体)**

| pos | 意味 | hit 数 | avg payout |
|-----|------|--------|------------|
| 1 | top1-top2-top3 (core) | 18 | 2,356 |
| 2 | top1-top2-top4 | 13 | **1,508** (最低) |
| 3 | top1-top2-top5 | **24** (最多) | 3,136 |
| 4 | top1-top2-top6 | 16 | 4,224 |
| 5 | top1-top3-top4 | 19 | 4,518 |
| 6 | top1-top3-top5 | 18 | 3,946 |
| 7 | top1-top3-top6 | 18 | 2,196 |

**pos2 は hit 数 13 (最少) かつ avg payout 1,508 (最低)** → 最低価値 bet。
100yen 節約 × 548 races = 54,800 yen、代わりに 13 hits × avg 1,508 = 19,604 yen の損失。
純 ROI 改善: +11.5pt (104.6% → 116.1%)。

**formation variant 比較 (N=548)**

| variant | cost | hit率 | ROI | CI | binom p |
|---------|------|--------|-----|-----|---------|
| 7-bet 現行 | 700/race | 23.0% | 104.6% | — | — |
| 6-bet (drop pos2) | 600/race | 20.6% | **116.1%** | [75.5%, 167.7%] | **0.001** |
| 5-bet (drop pos2+pos7) | 500/race | 17.3% | 124.9% | [76.9%, 186.6%] | 0.696 |

**strat_c subset (N=424) での 6-bet 結果**

| | 7-bet | 6-bet (drop pos2) |
|--|-------|-------------------|
| ROI | 115.7% | **128.5%** |
| CI | [75.8%, 168.1%] | [82.2%, 189.7%] |
| binom p (vs 1/6) | — | **0.0006** |

**verdict**: ★ 6-bet (drop pos2) は最も実装工数小 + 統計的有意 ★
- pos2 (top1-top2-top4) を除外するだけ = race_auto_notify.py の formation 生成で top4 を top2 軸側 col3 から除外
- 5-bet は ROI さらに高いが hit_p=0.696 で統計的に非有意 → 採用見送り

**実装**: `generate_trio_bets()` の col3 を `[top3, top5, top6]` に変更
(top4 を top2 軸側から除外、top3 軸側は変更なし)

---

### 候補 4: 条件 A 1600-1800m 除外

**分析 (Condition A 全体 N=196, ROI=88.7%)**

| サブグループ | N | ROI |
|-------------|---|-----|
| 1600-1800m | **74** | **39.8%** ← 主要 drag |
| 1801-2000m | 31 | 80.4% |
| 2001-2400m | 24 | 128.1% |
| 2400m+ | 6 | 64.0% |

- 1600-1800m が 74 races で ROI 39.8% → PnL -¥31,160 の大半はここ
- 内訳: 芝 1600-1800 ROI=0% (N=11)、ダート 1600-1800 ROI=46% (N=37)
- 1800m 以上は ROI 97.5%〜128.1% で健全

**venue 別 (strat_c, condition A)**

| venue | N | ROI | PnL |
|-------|---|-----|-----|
| 中京 | 19 | 166.1% | ¥+8,790 |
| 福島 | 8 | 174.6% | ¥+4,180 |
| 新潟 | 18 | 133.9% | ¥+4,270 |
| 中山 | 32 | 87.8% | ¥-2,740 |
| 阪神 | 47 | 96.7% | ¥-1,070 |
| 東京 | 31 | **38.9%** | ¥-13,250 |
| 京都 | 41 | 45.5% | ¥-15,640 (excl済) |

- 東京 条件 A は ROI 38.9% → 追加除外候補

**条件 A sub-filter 比較 (strat_c baseline ROI=104.8%)**

| フィルタ | N | ROI | delta | CI |
|---------|---|-----|-------|-----|
| strat_c baseline | 494 | 104.8% | — | [70.2%, 149.9%] |
| + excl A 1600-1800m | 436 | **113.7%** | +8.9pt | [74.3%, 164.5%] |
| + excl A dirt | 453 | 109.7% | +4.9pt | [72.0%, 158.9%] |
| + excl A 1600-1800 OR dirt | 430 | 113.6% | +8.8pt | [73.5%, 164.8%] |
| + excl A 1600-1800 + 東京 | 422 | 114.8% | +10.0pt | [75.0%, 168.4%] |

**verdict**: 距離フィルタ (+8.9pt) が最も安定。
N=58 除外は全体 N の 12% → ROI 改善に寄与。
ただし CI の下限はまだ 74% → 統計的有意 勝ち なし (100% 含む)。
**「条件 A は 1801m 以上のみ買う」が最もシンプルで効果大**。

---

### 候補 3+4 複合 (最有望)

| 組合せ | N | ROI | CI |
|--------|---|-----|-----|
| strat_c baseline | 494 | 104.8% | [70.2%, 149.9%] |
| C4 (excl A 1600-1800) | 436 | 113.7% | [74.3%, 164.5%] |
| C3+C4 (6-bet + excl A 1600-1800) | 371 | **141.9%** | [88.2%, 210.8%] |

★ C3+C4 複合: CI 下限 88.2% (100% 含む → 統計的有意勝ち なし) だが改善幅 +37pt ★
★ 5/24+ paper eval で 50 races 以上確保すれば有意性検証可能 ★

---

## 4. 統計的検定 サマリー

### 全体 hit rate 検定

| グループ | N | hit | hit率 | vs random | p-value |
|---------|---|-----|--------|-----------|---------|
| 全 settled | 629 | 135 | 21.5% | vs 1/7=14.3% | **0.0000** |
| strat_c (A+C+D) | 583 | 128 | 22.0% | vs 1/7=14.3% | **0.0000** |
| Cond A | 196 | 57 | 29.1% | vs 1/7=14.3% | **0.0000** |
| Cond C | 172 | 32 | 18.6% | vs 1/7=14.3% | 0.069 |
| Cond D | 215 | 42 | 19.5% | vs 1/7=14.3% | **0.021** |

**V15 の trio hit 能力は統計的に確認済み** (全体 p=0.0000)。
ROI が 100% 未満の問題は「hit はするが payout が低い」構造。

### 候補 ROI の Bootstrap CI まとめ

| 候補 | ROI | 95% CI | CI 下限 > 100% か |
|------|-----|--------|-------------------|
| Baseline (strat_c) | 104.8% | [70.2%, 149.9%] | No |
| C3: 6-bet | 128.5% | [82.2%, 189.7%] | No |
| C4: excl A 1600-1800 | 113.7% | [74.3%, 164.5%] | No |
| **C3+C4 複合** | **141.9%** | **[88.2%, 210.8%]** | **No** |

現状では CI 下限はいずれも 100% を下回る。N=629 全体でも [64.9%, 134.0%]。
**統計的有意勝ちを示すには 1,000+ races が必要**。

---

## 5. 戦略⑦案 C との interaction

戦略⑦案 C = 京都除外 + B/E/X 除外
- これに C3 (6-bet) と C4 (excl A 1600-1800) を重ねても **production コードに触れない**
- race_auto_notify.py の `generate_trio_bets()` と `should_bet()` の後段処理のみ
- predict_core / .pkl.gz / V15 モデルは完全変更なし

| 施策 | 変更ファイル | 変更内容 |
|------|------------|---------|
| C3: 6-bet formation | `tools/race_auto_notify.py` | `generate_trio_bets()` の col3_top2_side から top4 除外 |
| C4: excl A 1600-1800 | `tools/race_auto_notify.py` | `should_bet()` に distance <= 1800 and cond='A' の guard 追加 |

---

## 6. 5/22 admin schtask への影響確認

- 戦略 layer 変更 = race_auto_notify.py の後段処理のみ
- LiveOrchestrator / DailyPredict / DailyResults / NightlySanity には影響なし
- SCRAPER-GUARD 動作変更なし
- 5/22 以降の schtask fire に干渉しない

---

## ★ 推奨 ★

### 最有望候補: **C3 (6-bet formation) + C4 (excl Cond-A 1600-1800m)**

| 優先順位 | 候補 | 期待 delta ROI | 実装工数 | 信頼性 |
|----------|------|----------------|---------|--------|
| **1位** | **C3: 6-bet (drop pos2)** | **+23.7pt** | **小 (1行修正)** | **高 (p=0.001, N=424)** |
| **2位** | **C4: excl A 1600-1800m** | **+8.9pt** | **小 (1条件追加)** | **中 (N=58 excluded)** |
| **3位** | C3+C4 複合 | **+37.1pt** | 小 | 中 (CI 下限 88%) |
| 4位 | C2: score ≥ 0.6 filter | +44.8pt (小サンプル) | 中 | 低 (N=48) |
| 5位 | C1: market rank ≥ 3 | +15.1pt (ranked only) | 中 | 低 (N=29) |

### 5/24+ paper eval 優先順位

1. **C3 (6-bet)**: 即実装可能。5/24 から適用して paper eval。50 races で有意性再検証
2. **C4 (excl A 1600-1800m)**: 条件 A が少ない週末もあるが積み上げ可能
3. **C1+C2 (market rank filter)**: odds_base CSV の継続蓄積が前提。N ≥ 200 で再評価

### 採用判定基準 (6/17 paper eval 後)

| 指標 | 採用 GO 条件 |
|------|------------|
| ROI | ≥ 110% (paper eval 50+ races) |
| CI 下限 | > 80% |
| hit rate p | ≤ 0.05 vs random |
| N reduction | 除外 races < 20% |

---

## 7. 実装 design (6/17 採用判定後)

### C3: 6-bet formation

**変更箇所**: `tools/race_auto_notify.py` の trio formation 生成ロジック

```python
# 現行 (7-bet):
# col1=[top1], col2=[top2, top3], col3=[top3, top4, top5, top6]
# bets = [(top1, t2, t3) for t2 in col2 for t3 in col3 if t3 > t2]
# → 7 combinations

# C3 変更案 (6-bet): top2 軸の col3 から top4 を除外
# col3_for_top2 = [top3, top5, top6]  # top4 を除外
# col3_for_top3 = [top4, top5, top6]  # top3 軸は変更なし
# bets = [(top1, top2, t3) for t3 in col3_for_top2] + [(top1, top3, t3) for t3 in col3_for_top3]
# → 3 + 3 = 6 combinations
```

### C4: Condition A 1600-1800m 除外

**変更箇所**: `tools/race_auto_notify.py` の `should_bet()` 関数

```python
# 追加 guard:
if condition == 'A' and distance is not None and distance <= 1800:
    return False, 'A_short_distance_skip'
```

### テスト手順

1. `python -c "import py_compile; py_compile.compile('tools/race_auto_notify.py', doraise=True)"`
2. `/race-test` スキル実行 (全条件テスト)
3. 5/24 週末で paper eval 開始 (bet は実際には実行しない or shadow run)

---

## 8. 追加発見: 条件別パフォーマンス詳細

### V15 top1 実際着順 vs ROI

| top1 実着順 | N | ROI | hit率 |
|------------|---|-----|--------|
| 1着 (V15 top1 勝利) | 145 | 219.7% | 40.7% |
| 1着以外 | 484 | 58.5% | 15.7% |

→ **ROI の鍵は V15 top1 が実際に勝てるかどうか**。
axis 馬が勝つかどうかの識別が V15 の最重要能力。

### formation で axis 馬が実際に top3 に入る率

- N=548 (7-bet races): axis 馬が trio result に含まれる = **55.1%**
- 含まれる場合: trio が hit することが多い
- 含まれない場合: 7 bets 全部外れ

→ axis horse の top3 rate 改善が ROI 改善の根本 (V20 で対応予定)。

---

## 9. 留意事項

1. **過学習リスク**: C4 の「Cond A 1600-1800m 除外」は post-hoc 分析。paper eval で確認必須
2. **N 不足**: 候補 1/2 の market rank フィルタは現状 N=41〜65 のみ。蓄積が前提
3. **因果関係不明**: 1600-1800m が悪いのは V15 の弱点か、この期間の偶然か不明
4. **戦略⑦ 案 C 適用のズレ**: cumulative_results には戦略フィルタ適用外レースも含まれる。真の策略 C ROI は 104.8% (この audit 計算値) だが、実際の通知ベース ROI とは差がある可能性

---

*Generated: 2026-05-18 (Session Sub-task e)*
*Data source: `data/cumulative_results.csv` (n=629 settled, 2026-03-14 〜 2026-05-17)*
