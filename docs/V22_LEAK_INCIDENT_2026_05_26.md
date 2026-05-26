# V22 リーク事例 (2026-05-26)

## 概要

V22 (WF 0.8813) は `track_lap__` features による **POST-RACE leak** が原因で INVALID と判定。
paper shadow 停止、V22 WF 0.8813 は参照禁止。

---

## リーク内容

### 汚染 features (3件 POST-RACE、確定)

| feature | V22 LGB gain rank | 正体 |
|---------|------------------|----|
| `track_lap__pace_diff_race` | #37 (4,699) | 現レース前後半ペース差 |
| `track_lap__pace_second_half` | #44 (3,990) | 現レース後半ペース |
| `track_lap__lap_last_3f_race` | #97 (1,125) | 現レース上がり3F |

### 根本原因

`train/features_track_lap.py` の実装:

```python
# NG: 現レースの race_id で lap を JOIN している
df = df.merge(lap.rename(columns={'race_id': '_nk_id'}), on='_nk_id', how='left')
```

`lap` は `netkeiba_race_lap.csv` = 現レースの `pace_first_half / pace_second_half / lap_times`。
これらは発走後にしか計算できない。

### 追加汚染

`paci_ninki_idx` (オッズ派生、Pattern A 不可) が V22 top features rank #2 (gain 242,915)。

---

## 影響

### WF 数値の内訳

```
V22 WF 0.8813 = 4-model Grid (LGB+XGB+FT+IR) × leaky features
                (IR が fold 2021-2025 で weight 0.50-0.60 → POST-RACE leak 直撃)

V22 LGB+XGB only (leaky込み) ≈ 0.8699
vs V15 genuine LGB+XGB      = 0.8678
→ delta +0.0021 (leaky込み)
→ paci_ninki_idx/track_lap 除去後 ≈ 0 改善
```

### 実予測時の証拠

`data/v20/v22_vs_v15_2025_retro.json`:
- `v22_missing_feats`: `track_lap__pace_diff_race`, `track_lap__pace_second_half`, `track_lap__lap_last_3f_race`
- `v22_available`: 95 (100のはず)
- **`delta_auc`: -0.0062** (V22は V15 stored pkl より劣る)

→ leak features が取得できない実際の予測では V22 は V15 に負ける

---

## 再発防止

### T4 leak audit gate

`train/t4_leak_audit.py` を作成。全学習スクリプトの冒頭で実行:

```python
from train.t4_leak_audit import run_leak_audit
run_leak_audit(df, features, mode='pattern_a')  # FAIL時 sys.exit(1)
```

### 新規 feature 設計ルール

1. **JOIN が race_id 単独 → POST-RACE 疑い** (現レース情報)
   - 前走ラップを使う場合は horse timeline で shift(1) して前走 race_id に変換
2. **corr_target > 0.4 は WARNING / > 0.6 は ERROR**
3. **static CSV をそのまま使うな** → expanding window (cumsum.shift(1)) 必須
4. **4-model Grid と 2-model LGB+XGB の WF を混在させない**

### track_lap の正しい使い方 (V23 で実装)

```python
# NG: 現レース race_id で JOIN
df.merge(netkeiba_race_lap, on='race_id')  # → POST-RACE

# OK: 馬のタイムライン上で前走 race_id を計算してから JOIN
df_sorted = df.sort_values(['horse_id', 'date'])
df_sorted['prev_race_id'] = df_sorted.groupby('horse_id')['race_id'].shift(1)
df_sorted.merge(netkeiba_race_lap, left_on='prev_race_id', right_on='race_id')
```

---

## 教訓 (過去の失敗と同根)

| 事例 | 共通パターン |
|------|------------|
| V22 track_lap | race_id JOIN = 現レース情報 混入 |
| SKB POST-RACE leak (Session #38) | JRDB SKB = レース後拡張データ |
| sib_top3_rate static (Session #38) | static CSV = 未来データ混入 |
| dam_top3r leak (旧) | expanding 化せずに使用 |

**共通原則**: 「なぜ WF AUC が大幅改善したのか」 = まず疑え。
飽和 (0.866-0.868) を突き抜ける +0.01 以上の飛躍は leak の可能性 > 90%。

---

## タイムライン

- 2026-05-26 (この Session): V22 リーク監査実施 → INVALID 確定
- V22 paper shadow: 停止 (`tools/paper_shadow_v15_full.py` から除外)
- V23: 前走ラップ (shift) で clean rebuild 予定
