# sib_expanding 4 variant 探索結果 (Session #42 F)

**作成**: 2026-05-08 (Session #42 F、 ユーザー仕事中)
**前提**: Session #41 D で full expanding 実装、 LIVE retro +6.89pt 改善確認
**目的**: window/decay/segment/cross-source 4 variant で更なる corr 改善を狙う

---

## 1. CRITICAL RESULT: variant A (window) が最良

### 1.1 corr_target 比較

| variant | settings | corr(target) | vs Session #41 D |
|---------|---------|--------------|----------------|
| LEAK 旧 sib_top3_rate (Session #38) | static all years | 0.2939 | (リーク含む) |
| Session #41 D full expanding | window=∞ | **0.1689** | (baseline) |
| **variant A w=3** | rolling 3 走 | **0.1993** | **+0.030** |
| **variant A w=5** ★ | rolling 5 走 | **0.2010** | **+0.032 ★ best** |
| variant A w=10 | rolling 10 走 | 0.1938 | +0.025 |

### 1.2 解釈

- 全期間 expanding (Session #41 D) は 古いレース の影響で sample 平均化 → 直近 trend 鈍化
- **直近 5 走 window** にすると 各馬の現在の状態 (mother の最近の産駒成績) を反映
- → corr +0.032 は実質 **+19% 改善** (0.1689 → 0.2010)
- リーク版 0.2939 と比較 すると 真の信号 比率: 0.2010 / 0.2939 = 68.4% (Session #38 hybrid 仮説の 67% と整合)

→ **window=5 が sib_exp の最適化点**

---

## 2. 試作 4 variant

### 2.1 variant A: window size (本 Session で実装 + 検証 ✅)

```python
# 過去 N 走 rolling (current 含まず)
df["mother_recent_top3"] = grp["is_top3"].transform(
    lambda s: s.shift(1).rolling(window, min_periods=1).sum()
)
df["mother_recent_races"] = grp["is_top3"].transform(
    lambda s: s.shift(1).rolling(window, min_periods=1).count()
)
sib_top3_rate_exp_w{N} = (top3 + α*0.30) / (races + α)
```

**結果 (corr)**:
- w=3: 0.1993
- **w=5: 0.2010** ★ best
- w=10: 0.1938

**出力**: `data/netkeiba_siblings_expanding_w{3,5,10}.csv`

### 2.2 variant B: weight decay (実装済、 試行は Phase 3 後半)

```python
# exponential decay weighted sum
cum_top3 = cum_top3 * decay + row["is_top3"]
cum_runs = cum_runs * decay + 1
sib_top3_rate_exp_d{decay} = (cum_top3 + α*0.30) / (cum_runs + α)
```

**期待**:
- decay=0.95: 古いレース 5 年前 の重み = 0.95^60 ≈ 0.046 (大幅減)
- decay=0.98: 5 年前 の重み = 0.98^60 ≈ 0.298 (中)
- decay=0.99: 5 年前 の重み ≈ 0.547 (緩)

**Phase 3 後半 (6/9-13)** で実 retro 比較予定。

### 2.3 variant C: 性別・距離別 segmented (実装済、 計算重い)

```python
# mother × sex_code × dist_bucket で個別 expanding
grp = df.groupby(["mother", "sex_code", "dist_bucket"])
df["sib_top3_seg"] = grp["is_top3"].cumsum() - df["is_top3"]
```

**dist_bucket**: <1500 (短) / 1500-2000 (中) / >2000 (長)
**sex_code**: 牡/牝/セン

**期待**: 同じ母から異なる性別/距離適性を分離 → 距離別予測精度 向上
**risk**: sample 細分化で α smoothing 必要 (高 alpha 推奨)

**Phase 3 後半** で 実 retro 比較予定。

### 2.4 variant D: JRA-NAR 横断 (Phase 3 後半 6/9-13、 本 Session では plan のみ)

NAR レースの兄弟成績も含めた expanding:
- mother 単位で JRA + NAR 全 race 集計
- horse_id 単位で JRA / NAR 移籍 馬の連続性確保
- → V20 (JRA + NAR 統合) で本格採用

**期待**: NAR で実績ある母系の JRA 仔馬 評価精度 向上
**risk**: NAR data 鮮度 (1 年 stale)、 horse_id mapping 精度

---

## 3. window=5 (variant A) を Phase 3 で本格採用 plan

### 3.1 5/24+ Phase 3 前半

```python
# train/v18v19_sib_exp/run_v18v19_sib_exp_v2_window5.py (新規予定)
# Session #41 D の sib_exp v1 (full expanding) → v2 (window=5) に切替
# expected:
#   BT 2025 winner_top1 (race_part): 45.88% → 46.0-46.5% (+0.1-0.6pt)
#   LIVE 5/2-5/3 winner_top1: 31.03% → 32-34% (+1-3pt)
#   shift_factor: 1.48x → 1.45x (微改善)
```

### 3.2 期待効果

V18/V19 v2 (window=5):
- LIVE winner_top1: 31% → **33-35%** (推定)
- vs OLD (sib含 リーク) 34.48%: **ほぼ完全回復**
- vs no_sib 24.14%: **+9-11pt**

→ **5/16 V18/V19 投入 GO 確率: 70-80% に上昇** (Session #41 D 60-70% から +10pt)

### 3.3 実装 priority

| variant | 実装 priority | 採用見込み |
|---------|-------------|----------|
| **A w=5** | ★★★★★ | **5/24+ Phase 3 即採用候補** |
| A w=3 / w=10 | ★ | 比較用、 採用なし |
| B decay 0.95 | ★★★ | 6/9-13 で比較 |
| C segmented | ★★ | 6/9-13 で比較 |
| D JRA-NAR | ★★★ | V20 で採用 |

---

## 4. 学習時の注意点

### 4.1 race_id format

- `data/netkeiba_siblings_expanding_w5.csv` の race_id 10 chars (jrdb-internal)
- v17 cache の race_id と直接 merge 可能 (Session #41 D 確認済 98.4% match)

### 4.2 sib_total_races_exp / offspring_exp も追加可

variant A では top3_rate / shinba_wr のみ。 Phase 3 で v2 学習時に:
- `sib_total_recent_races_w5` (直近 5 走の母産駒数)
- `sib_recent_offspring_count_w5` (直近 5 走の unique 仔馬数)

を追加して features 6 個に拡張 (現 4 個 + 2 個)。

### 4.3 horse_id 形式変換

LIVE retro では:
- netkeiba 出馬表 horse_id: 10 chars (例 `2023101394`)
- blood_full / sib_expanding: 8 chars (例 `23101394`)

→ 既存 `tools/v18_v19_retro_sib_exp.py` の `_hid_to_blood_id` 関数 流用

---

## 5. 5/9 V15 投資保護 (F 領域)

✅ V15 production 完全不変 (新規 csv 出力のみ)
✅ predict_core / daily_predict / app.py / V15 model 不変
✅ schtasks 既存 task 不変
✅ csv は data/netkeiba_siblings_expanding_w*.csv の新規 path

→ **5/9 朝 V15 完全保証**

---

## 6. 結論

✅ F1: variant A (window) 実装 + 検証 (3/5/10 走比較)
✅ F2: **window=5 が最良 corr 0.2010** (full expanding 0.1689 から **+0.032 改善**)
✅ F3: variant B (decay) / C (segmented) 実装 (実 retro は Phase 3)
✅ F4: variant D (JRA-NAR) plan
✅ F5: 5/24+ Phase 3 前半 で window=5 即採用候補
✅ F6: 5/16 GO 確率 60-70% → **70-80% に上昇** (window=5 効果見込)

→ **sib_exp v2 (window=5) で V18/V19 復活路線 強化**

---

**Session #42 F 完了**
