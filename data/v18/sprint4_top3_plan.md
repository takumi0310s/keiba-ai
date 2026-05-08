# Sprint 4 ★★★ 3 件 plan (5/8)

**作成**: 2026-05-08 (Session #50 Sprint 4 A 領域)
**前提**: AUDIT-1 E [audit_unused_features_top30_5_8.md](audit_unused_features_top30_5_8.md) Top 3
**branch**: dev/sprint4 (main 6c0680ad から分岐)
**期待 V15.5**: AUC 0.894-0.899 (V15 baseline 0.8788/0.8939 比較)

---

## 1. 実装対象 ★★★ 3 件

| # | feature 群 | source | csv | 期待 AUC | 工数 |
|---|-----------|--------|-----|---------|------|
| 1 | bias 6 fields (1corner/2corner/backstr/3corner/4corner/straight) | JRDB SRB | `data/jrdb_srb.csv` | +0.003-0.005 | 4h |
| 2 | master_index 5 indices (time/master/start/chase/agari) | netkeiba マスター | `data/netkeiba_master_index.csv` | +0.003-0.005 | 6h |
| 3 | jrdb_jo cid_idx / ls_idx | JRDB JO | `data/jrdb_jo.csv` | +0.002-0.003 | 3h |
| 合計 | 13 features | 3 source | — | **+0.008-0.013** | 13h |

---

## 2. データ join key

V15 cache (`data/_v15_optuna_df_cache.pkl.gz`) の race_id (10 chars JRDB-style)
→ netkeiba 12 chars 形式に変換 → SRB / MI / JO と join。

```python
def v15_rid_to_nk(rid: str) -> str:
    # VV+YY+K+N+RR (10 chars) → 20YY+VV+0K+0N+RR (12 chars)
    s = str(rid)
    return f'20{s[2:4]}{s[0:2]}0{s[4]}0{s[5]}{s[6:8]}'
```

カバー率: 95.6% (19822 / 20733 race_id matched)。 非 match 911 件は B/C/A 含む特殊コード (障害 / 訓練)。

---

## 3. 各 feature 詳細

### #1 SRB bias 6 fields (★★★)
- **source**: jrdb_srb.csv (race-level、 1 race = 1 row)
- **fields**: bias_1corner, bias_2corner, bias_backstr, bias_3corner, bias_4corner, bias_straight
- **値**: コーナー/ストレート別のトラックバイアス (内/中/外、 数値 or カテゴリ)
- **リーク risk**: pre-race (レース後のレース comment 系を除外)
- **実装**:
  - 6 列を race_id 単位で merge
  - encoding: 数値ならそのまま、 カテゴリなら one-hot or target-mean (expanding)
- **出力 feature**: `srb_bias_1c`, `srb_bias_2c`, `srb_bias_bs`, `srb_bias_3c`, `srb_bias_4c`, `srb_bias_st`

### #2 master_index 5 indices (★★★)
- **source**: netkeiba_master_index.csv (horse-level、 1 race × 1 horse)
- **fields**: time_index, master_index, start_index, chase_index, agari_index
- **値**: netkeiba マスター限定 数値指数
- **リーク risk**: ここは要 audit。 race 後の集計値の可能性あり (finish_order が同 csv にある = post-race 含む可能性)。
  → backtest で gap > 0.05 なら post-race 判定で除外
  → expanding window 化 が必要なら 馬の前走までの mean を使用
- **実装**:
  - race_id × umaban で merge
  - 当該レース値 = post-race の場合 → 「前走までの 5 indices mean」 を使う (expanding)
  - dam_top3r 教訓に従う
- **出力 feature**: `mi_time_idx_prev`, `mi_master_idx_prev`, `mi_start_idx_prev`, `mi_chase_idx_prev`, `mi_agari_idx_prev`

### #3 jrdb_jo cid_idx / ls_idx (★★★)
- **source**: jrdb_jo.csv (horse-level、 1 race × 1 horse)
- **fields**: cid_idx, ls_idx
- **値**: JRDB JO の数値指数 (cid = 西田指数系、 ls = ライディング指数系)
- **リーク risk**: pre-race (JO は朝段階で 確定)
- **実装**:
  - race_id × umaban で merge
  - そのまま numeric として使用
- **出力 feature**: `jo_cid_idx`, `jo_ls_idx`

---

## 4. 実装 file 構成

| file | 内容 |
|------|------|
| tools/sprint4_feature1.py | SRB bias 6 features build + backtest |
| tools/sprint4_feature2.py | master_index 5 features build + backtest |
| tools/sprint4_feature3.py | jrdb_jo 2 features build + backtest |
| tools/v15_5_features.py | V15 + ★★★ 13 features 統合 wrapper |
| data/v18/sprint4_feature1_5_8.md | feature 1 結果 |
| data/v18/sprint4_feature2_5_8.md | feature 2 結果 |
| data/v18/sprint4_feature3_5_8.md | feature 3 結果 |
| data/v18/sprint4_v15_5_poc_5_8.md | V15.5 PoC 統合結果 |

---

## 5. backtest 方針

- **対象**: 2023-2025 (過去 3 年分、 V15 cache 内)
- **基線**: V15 (145 features) → AUC 0.8788 (CLAUDE.md 既知値)
- **比較**: V15 + ★★★ 13 features = V15.5
- **手法**: walk-forward 1 年単位 5-fold (2020 train → 2021 eval、 etc.)
- **評価**: AUC (overall) + AUC (条件 A-X)
- **リーク監査**: train AUC vs eval AUC gap > 0.05 で警告

---

## 6. 絶対遵守 (Sprint 4 全体)

🔴 NEVER:
- main branch / 既存 dev branches 変更
- predict_core.py / daily_predict.py / app.py / 既存 tools・train 変更
- V15 model file 変更
- schtasks 既存 41 件 変更

🟢 OK:
- dev/sprint4 新規 branch
- tools/sprint4_*.py + tools/v15_5_features.py 新規
- data/v18/sprint4_*.md 新規 doc

→ V15 production 完全保護
→ 5/15 22:00 merge 予定 (sprint1+2+training-poc+two-stage+sprint4 一括)
