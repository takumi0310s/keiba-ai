# Session #57 A: V20 interaction features 候補 選定

**作成**: 2026-05-09 (Session #57 A)
**目的**: V15 145 features 飽和の打開、 interaction (組み合わせ) で深掘り
**branch**: dev/v20-interaction (main 6c0680ad から分岐)

---

## 1. 背景

- V15 ベースライン: 145 features、 LGB+XGB+FT+IR ensemble、 BT 2025 AUC 0.8856
- Session #51 単一 feature 追加で飽和 確認
- → interaction (2-way / 3-way) で組み合わせ効果を抽出

---

## 2. V15 主要 features (145 個 抜粋、 interaction の素材)

### 2.1 ID 系 (groupby key)

| col | 説明 | unique 数 (推定) |
|-----|------|------|
| `horse_id` | 馬 | ~80,000 |
| `jockey_id` | 騎手 | ~250 |
| `trainer_id` | 調教師 | ~250 |
| `sire_enc` | 父 (TOP100 encoded) | 101 |
| `bms_enc` | 母父 (TOP100 encoded) | 101 |

### 2.2 環境 features (groupby key)

| col | 説明 | unique 数 |
|-----|------|------|
| `course_enc` | 競馬場 (中央 10 場) | 10 |
| `surface_enc` | 芝/ダート/障害 | 3 |
| `distance` | 距離 (m) | ~20 (categorical bin で 5) |
| `dist_cat` | 距離 bin 0-4 | 5 |
| `condition_enc` | 馬場 (良〜不良) | 4 |
| `class_code` | クラス (新馬〜G1) | ~10 |

### 2.3 target (集計対象)

| col | 説明 |
|-----|------|
| `is_win` | 1着 |
| `is_top3` | 複勝圏 (≤3着) |
| `finish` | 着順 (raw) |

---

## 3. interaction 候補 10 件 (本 PoC で実装)

### 3.1 単一馬 × 騎手 (rider compatibility)

| # | 名前 | groupby key | 集計 | 期待 |
|---|------|-------------|------|------|
| 1 | `int_horse_jockey_top3r` | (horse_id, jockey_id) | expanding top3 rate | +0.001 |

(注: V15 既存の `jockey_horse_wr` / `jockey_horse_top3r` と被るため、 alpha smoothing / 直近 N 走 重みなどで差別化)

### 3.2 騎手 × 環境

| # | 名前 | groupby key | 集計 | 期待 |
|---|------|-------------|------|------|
| 2 | `int_jockey_course_top3r` | (jockey_id, course_enc) | expanding top3 rate | +0.0015 |
| 3 | `int_jockey_distcat_top3r` | (jockey_id, dist_cat) | expanding top3 rate | +0.001 |
| 4 | `int_jockey_baba_top3r` | (jockey_id, condition_enc) | expanding top3 rate | +0.0008 |
| 5 | `int_jockey_class_top3r` | (jockey_id, class_code) | expanding top3 rate | +0.0008 |

### 3.3 調教師 × 環境

| # | 名前 | groupby key | 集計 | 期待 |
|---|------|-------------|------|------|
| 6 | `int_trainer_course_top3r` | (trainer_id, course_enc) | expanding top3 rate | +0.0008 |

### 3.4 父系 × 環境

| # | 名前 | groupby key | 集計 | 期待 |
|---|------|-------------|------|------|
| 7 | `int_sire_course_top3r` | (sire_enc, course_enc) | expanding top3 rate | +0.0007 |
| 8 | `int_sire_distcat_top3r` | (sire_enc, dist_cat) | expanding top3 rate | +0.0008 |
| 9 | `int_sire_baba_top3r` | (sire_enc, condition_enc) | expanding top3 rate | +0.0006 |

### 3.5 騎手 × 厩舎 (連携)

| # | 名前 | groupby key | 集計 | 期待 |
|---|------|-------------|------|------|
| 10 | `int_jockey_trainer_top3r` | (jockey_id, trainer_id) | expanding top3 rate | +0.001 |

---

## 4. 期待 AUC contribution 合計

```
合計 +0.0085 (10 件) を上限と想定
実際は redundant / fold 別ばらつきで +0.002-0.005 が現実的
```

V20 base AUC 0.8752 → V20 + interaction 期待 AUC 0.877-0.880

---

## 5. 実装方針 (Area B で実装)

### 5.1 expanding 計算 (リーク防止)

```python
# 全 features 共通 pattern
# date 順 sort → groupby cumsum - current で当該レース除外
df = df.sort_values('date_num').reset_index(drop=True)
df['_cum_t3'] = df.groupby(key)['is_top3'].cumsum() - df['is_top3']
df['_cum_n']  = df.groupby(key).cumcount()
# Bayesian smoothing (alpha smoothing)
prior = df['is_top3'].mean()
df[fname] = (df['_cum_t3'] + alpha * prior) / (df['_cum_n'] + alpha)
```

### 5.2 alpha (shrinkage prior)

| feature | alpha | 理由 |
|---------|-------|------|
| jockey_course | 10 | 騎手 × 場、 件数中程度 |
| jockey_distcat | 10 | 同上 |
| jockey_baba | 5 | 不良馬場は件数少 |
| jockey_class | 5 | クラス分散 |
| trainer_course | 10 | 調教師 × 場 |
| sire_course | 30 | sire は件数多、 強い shrinkage |
| sire_distcat | 30 | 同上 |
| sire_baba | 20 | 同上 |
| jockey_trainer | 5 | 連携、 件数少のものは prior 寄せ |
| horse_jockey | 3 | 既存 v15 と差別化、 弱 shrinkage |

### 5.3 リーク厳禁

- すべて expanding window (cumsum - current)
- date_num 昇順 sort 必須
- fillna は global prior (population mean)

---

## 6. NEXT (Area B)

→ tools/v20_interaction_features.py 実装 + 過去 6 年分 (2020-2025) 計算

---

**Session #57 A 完了**
