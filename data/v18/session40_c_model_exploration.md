# Session #40 C: モデル探索 (新features + アンサンブル + NAR深掘り + 共通馬)

**作成**: 2026-05-07 (Session #40 C)
**目的**: Phase 3 (5/24+) に向けた モデル候補 / 探索の前倒し設計
**前提**: V15 (現行 150 features) ベースに 5/24+ で 増分テスト

---

## 1. C1 — 新規 features 探索 (4 候補)

### 1.1 候補 a: 馬場差 features (各場 季節別)

**仮説**: 同じ "良" でも、 開催場 + 季節で馬場特性が異なる:
- 春 中山 = 重め (湿度高 + 雨多)
- 夏 札幌 = 軽め (パンパンの良)
- 秋 京都 = 中庸
- 冬 小倉 = 軽め (乾燥)

**実装案**:
```python
# 開催場 × 季節 × 馬場状態 × 距離 のクロス feature
df['track_seasonal_bias'] = df.groupby(['course', 'season', 'condition_enc'])['agari_3f'].transform(
    lambda s: s.expanding().mean()  # date 順、 cumsum-current
)
df['track_seasonal_bias_diff'] = df['track_seasonal_bias'] - df['avg_last3f_3r']
```

**期待効果**: AUC +0.001-0.003 (各場 specialty を捕捉)

**risk**: 低サンプル sub-cluster で過学習 → α=20+ smoothing 必須

### 1.2 候補 b: レース流れ features (前半 / 後半 ペース予測)

**仮説**: race_lap データから、 前半上がり / 後半上がり の傾向を予測:
- 同距離 過去 race の前半/後半 ペース 平均
- 当 race の出走馬構成 (脚質バランス) で補正

**実装案**:
```python
# 既存 prev_race_first3f / prev_race_last3f を活用
df['expected_first3f'] = df.groupby(['course', 'distance', 'surface_enc'])['prev_race_first3f'].transform(
    lambda s: s.expanding().mean()
)
df['expected_last3f'] = df.groupby(['course', 'distance', 'surface_enc'])['prev_race_last3f'].transform(
    lambda s: s.expanding().mean()
)
df['expected_pace_diff'] = df['expected_last3f'] - df['expected_first3f']
```

**期待効果**: AUC +0.002-0.005 (V15 で既に prev_race_pace_diff 死んでいる、 race-level に集約で復活)

**risk**: 出走馬構成 (脚質) が unknown → expected pace の精度限界

### 1.3 候補 c: TimeSeries features (過去 5 走 trend)

**仮説**: 過去 N 走の着順 / 上がり 3F の推移 (トレンド) は **回復 / 低下** indicator:
- 着順が 8→6→4→2→1 のトレンド: 回復強気
- 上がり 3F が 36.0→35.5→35.0→34.5: 末脚向上

**実装案**:
```python
# 過去 5 走 finish の単回帰 slope
def trend_slope(s):
    if len(s) < 3: return 0
    x = np.arange(len(s))
    return np.polyfit(x, s.values, 1)[0]

df['finish_trend_5r'] = df.groupby('horse_id')['prev_finish'].transform(
    lambda s: s.shift(0).rolling(5, min_periods=3).apply(trend_slope, raw=False)
)
df['agari_trend_5r'] = df.groupby('horse_id')['prev_last3f'].transform(...)
```

**期待効果**: AUC +0.002-0.004 (好調馬を早期キャッチ)

**risk**: 短期 noise が trend に混ざる → smoothing 必要

### 1.4 候補 d: 相対 features (出走馬間)

**仮説**: 当該 race の他出走馬と比較した相対値 (race-level normalize):
- horse の jockey_wr_calc - race 内平均 jockey_wr_calc
- horse の prev_finish - race 内平均 prev_finish
- horse の training_4f - race 内平均 training_4f

**実装案**:
```python
for col in ['jockey_wr_calc', 'prev_finish', 'training_time_filled']:
    df[f'{col}_rel'] = df[col] - df.groupby('race_id')[col].transform('mean')
    df[f'{col}_rank'] = df.groupby('race_id')[col].rank(method='dense', ascending=False)
```

**期待効果**: AUC +0.003-0.008 (V15 IntraRace Attention の代替 features 化、 LGB/XGB でも捕捉可能)

**risk**: race 内の競走馬数 (頭数) でスケール変動 → race 内 zscore も追加

### 1.5 統合判定

Phase 3 (5/24+) で 4 候補 を 1 つずつ V18/V19 v2 に追加して BT、 採用基準:
- AUC + 0.002 以上
- 全年 gap < 0.05 (過学習 NG)
- LIVE retro でも winner_top1 ≥ 30%

---

## 2. C2 — アンサンブル比率 最適化

### 2.1 V15 現状 (Grid Search 既定)

| 年 | LGB | XGB | FT-Trans | IntraRaceAttn | total |
|----|-----|-----|----------|--------------|-------|
| 2020 | 0.25 | 0.30 | 0.10 | 0.35 | 1.00 |
| 2021 | 0.25 | 0.25 | 0.15 | 0.35 | 1.00 |
| 2022 | 0.25 | 0.25 | 0.15 | 0.35 | 1.00 |
| 2023 | 0.30 | 0.25 | 0.10 | 0.35 | 1.00 |
| 2024 | 0.25 | 0.30 | 0.10 | 0.35 | 1.00 |
| 2025 | 0.25 | 0.30 | 0.10 | 0.35 | 1.00 |

→ IntraRaceAttn 0.35 が最大寄与 (race 内相対関係を捕捉)、 FT-Trans は 0.10-0.15 で補助。

### 2.2 grid search 範囲提案 (Phase 3)

```
LGB / XGB / FT / IR ∈ [0.05, 0.10, 0.15, ..., 0.50]
constraint: LGB + XGB + FT + IR = 1.00
total grid: ~300 combinations × 6 fold = 1800 evaluations
所要時間: 約 2-3 h (V15 model 既存)
```

### 2.3 V18/V19 v2 / V20 への横展開

- V18/V19 v2 (sib_*_exp 込み): LGB+XGB のみ → 2 重み grid
- V20: LGB+XGB+FT+IR (V15 と同構成) → 4 重み grid

### 2.4 期待効果

V15 重みは既に Grid Search 済 → 大改善 +0 想定。
V18/V19 v2 / V20 は **新 features 追加 + 重み再最適化** で +0.002-0.005 期待。

---

## 3. C3 — NAR モデル深掘り

### 3.1 NAR v4 現状

| 指標 | 値 |
|------|----|
| WF AUC | 0.8145 |
| OOS AUC | 0.8519 |
| 学習 features | 22 |
| 学習 data | 1 年 stale (2024-03 〜 2025-05) |

### 3.2 V15 features (150) から NAR 適用候補

NAR で利用可能な V15 features:

| カテゴリ | NAR 利用可 | NAR 利用不可 (理由) |
|---------|-----------|------------------|
| 基本 (距離/性別/年齢/枠/頭数) | ✅ 全 14 | — |
| 騎手・調教師 (expanding wr) | ✅ 全 3 | — |
| 前走 (prev_finish, prev_last3f, etc.) | ✅ 全 10 | — |
| 集計 (avg_finish_3r, top3_count_3r) | ✅ 全 5 | — |
| 派生 (dist_change, age_sex etc.) | ✅ 全 11 | — |
| V9.2 (career, sire wr) | ✅ ほぼ全 | — |
| V9.3 (race_first3f, sakaro) | ⚠ 部分 | sakaro_best 系は 中央のみ data |
| V12 (speed_index) | ❌ | netkeiba speed_index は中央のみ |
| JRDB 系 (jrdb_kyi etc.) | ❌ | JRDB は中央専用 |
| **計 NAR 適用可** | **約 80-90 features** | (V15 150 のうち 60%) |

### 3.3 NAR v5 (Phase 3 後半 6 月想定) features

```python
NAR_V5_FEATURES = [
    # NAR v4 (22) 全て
    *NAR_V4_FEATURES,
    # 追加候補 (V15 から NAR 適用可な 60+)
    'jockey_wr_calc', 'jockey_course_wr_calc', 'jockey_surface_wr',
    'horse_career_races', 'horse_career_wr', 'horse_career_top3r',
    'sire_surface_wr', 'sire_dist_wr', 'bms_surface_wr',
    'avg_finish_3r', 'best_finish_3r', 'top3_count_3r', 'finish_trend',
    'avg_last3f_3r',
    # sib_*_exp (Session #39 A) NAR 版
    'sib_top3_rate_exp_nar',  # 新規、 NAR レース データで expanding 計算
    'sib_shinba_wr_exp_nar',
    # その他 V12 / V9.3 で expanding 可能なもの
    ...
]
# 計 80-90 features
```

### 3.4 期待効果

NAR v4 → v5:
- 学習 features 22 → 80-90 (約 4 倍)
- 期待 AUC +0.01 〜 +0.03 (= 0.825-0.845)
- 学習 data 拡張: 5/24+ で 2025-06 〜 2026-05 の 1 年分 update → +50% data

### 3.5 Phase 3 schedule (NAR 関連)

| 期間 | 内容 |
|------|------|
| 5/24-6/8 | NAR scraping update (2025-06 〜 2026-05 の 1 年分) |
| 6/9-13 | sib_*_exp_nar 構築 (NAR レース データで expanding) |
| 6/14-20 | NAR v5 学習 + WF 検証 |
| 6/21-25 | V20 (JRA + NAR 統合) 学習に NAR v5 features 統合 |

---

## 4. C4 — JRA-NAR 共通馬の使い回し

### 4.1 仮説

地方→中央移籍 / 中央→地方転厩する馬の features を統合可能なら:
- 馬 単位の career features が 連続化 (NAR period + JRA period)
- 学習 data 拡張 (1 馬で複数 period のレース record)

### 4.2 マッピング手法

#### 案 A: 馬名 fuzzy match
```python
from rapidfuzz import fuzz, process
jra_names = jra_blood['horse_name'].unique()
nar_names = nar_horses['horse_name'].unique()

mapping = {}
for nm in nar_names:
    match, score, _ = process.extractOne(nm, jra_names, scorer=fuzz.ratio)
    if score >= 95:
        mapping[nm] = match
```

#### 案 B: 血統 (父+母) で確定マッピング
```python
# 父 + 母 + 生年月日 が一致する馬は同一個体
key = ['father', 'mother', 'birthday']
common = pd.merge(jra_blood, nar_blood, on=key, suffixes=('_jra', '_nar'))
# horse_id_jra ↔ horse_id_nar の mapping table
```

→ **案 B が確実** (馬名は表記揺れあり、 血統+生年月日は一意)

### 4.3 共通馬 推定数

CLAUDE.md より:
- JRA blood_full.csv: 81,986 行
- NAR horses: 約 50,000 行 (推定)
- 共通馬 (重複): 200-500 頭 想定

### 4.4 V20 学習 data 拡張効果

共通馬 200-500 頭 × 平均 30 races/horse = 6,000-15,000 race records 追加
→ V20 学習 data 50 万 race rows (JRA) + 5 万 (NAR) + 1 万 (共通馬重複) = 56 万

期待効果: AUC +0.001-0.002 (data 量 +2% は微増)

### 4.5 Phase 3 後半 (6 月) 着手

- 6/9-13 で `tools/jra_nar_horse_mapping.py` 試作
- 6/14-20 で V20 学習 data に common 馬 期間連続化を統合
- 効果不発 (+0 AUC) なら 採用 NO-GO、 個別 model 維持

---

## 5. C5 — 統合 (Phase 3 採用候補 list)

### 5.1 5/24+ で実装する 候補 features

| 候補 | C# | 期待 AUC | 工数 | 優先度 |
|------|---|---------|------|--------|
| 馬場差 (track_seasonal_bias) | C1a | +0.001-0.003 | 低 | ★★ |
| レース流れ (expected_pace_diff) | C1b | +0.002-0.005 | 中 | ★★★ |
| TimeSeries trend | C1c | +0.002-0.004 | 中 | ★★ |
| 相対 features | C1d | +0.003-0.008 | 中 | ★★★★ |
| アンサンブル grid | C2 | +0.000-0.002 | 中 | ★★ (V15 既最適) |
| NAR v5 features 拡張 | C3 | +0.01-0.03 (NAR) | 高 | ★★★★★ |
| JRA-NAR 共通馬 mapping | C4 | +0.001-0.002 | 中 | ★★ |

### 5.2 Phase 3 schedule との整合

| 期間 | 採用候補 |
|------|---------|
| 5/24-6/8 | C4 mapping 試作 + C3 NAR data update |
| 6/9-13 | C1d 相対 features (V18/V19 v2 統合) |
| 6/14-20 | C1b レース流れ + C2 アンサンブル grid (V20 学習) |
| 6/21-25 | C1a 馬場差 + C1c TimeSeries (V20 BT 検証) |
| 6/26-30 | V20 GO/no-go 判定、 全 候補 評価 |

---

## 6. 5/9 V15 投資保護 (C 領域)

✅ V15 model file / production 経路 完全不変 (本 doc は設計のみ)
✅ 5/24+ Phase 3 で実装、 5/9 朝には影響なし

---

## 7. 結論

✅ 4 features 候補 (C1a-d) 設計
✅ アンサンブル grid search plan (V18/V19 v2 / V20)
✅ NAR v5 拡張 plan (V15 features 60+ 適用)
✅ JRA-NAR 共通馬 mapping (血統+生年月日 ベース)
✅ Phase 3 採用 schedule 整合 (5/24-6/30)

→ **Phase 3 モデル探索 候補 完備**

---

**Session #40 C 完了**
