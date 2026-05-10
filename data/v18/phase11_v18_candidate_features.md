# Phase 11 完了: JRDB 未統合 features 実装 (V18 candidate predict_core_v18.py)

date: 2026-05-10 18:30
session: Phase 11 (Opus 4.7、 caveman mode)

---

## 実装方針

ユーザー要望「完全 predict_core_v18.py 複製」を採用:
- `tools/predict_core_v18.py` = `tools/predict_core.py` の完全 cp
- `build_features()` の末尾、 `use_features fallback` 直前に **V18 candidate 15 features** を挿入
- V15 model_data['features'] には 15 features が含まれない → V15 推論は完全不変
- V18 学習時にこれらを feature list に取り込む form

## 追加 15 features (scaffold + default values)

Phase 11 段階 = scaffold + 妥当な default 設定。
5/12+ で JRDB data lookup + expanding window 集計 を本実装予定。

### A. 外厩 (training farm) - 4 features
source: JRDB UKC (馬基本) / CHA (前走詳細)

| feature | default | 計画 logic |
|---|---|---|
| `gaika_id_enc` | 0 | 外厩 ID encoded (UKC.外厩 column lookup) |
| `gaika_top3r_3r` | 0.33 | 過去 3 R 外厩 top3 率 (expanding window) |
| `gaika_winrate` | 0.20 | 外厩別 通算 勝率 (Bayesian alpha=20) |
| `gaika_dist_winrate` | 0.20 | 外厩 X 距離別 勝率 |

### B. 時系列オッズ - 4 features
source: JRDB OT/OV/OW (時系列オッズ) または save_odds_base 蓄積データ

| feature | default | 計画 logic |
|---|---|---|
| `odds_change_3h_v18` | 0.0 | 3h 前 → 直前 odds 変化率 |
| `odds_change_30m_v18` | 0.0 | 30m 前 → 直前 odds 変化率 |
| `popularity_shift_v18` | 0 | 朝人気 - 直前人気 (整数差) |
| `odds_volatility_v18` | 0.0 | 期間内 odds std / mean |

### C. 騎手マスタ拡張 - 4 features
source: JRDB KKA (既統合 90.4%) を distance/track/class/trainer で再集計

| feature | default | 計画 logic |
|---|---|---|
| `jockey_dist_winrate` | 0.10 | 騎手 X 距離 勝率 (alpha=10 prior) |
| `jockey_track_winrate` | 0.10 | 騎手 X 芝/ダート 勝率 |
| `jockey_class_winrate` | 0.10 | 騎手 X クラス 勝率 |
| `jockey_x_trainer_wr` | 0.15 | 騎手 X 調教師 連携 勝率 (combo lookup) |

### D. 返し馬 / パドック - 3 features
source: JRDB CYB (調教コメント) / TYB (直前) を 既存 `jrdb_paddock_idx` 超えで評価

| feature | default | 計画 logic |
|---|---|---|
| `return_horse_score` | 0.0 | 返し馬 評価 (-3 〜 +3) |
| `paddock_eval_v18` | 0.0 | パドック評価 (jrdb_paddock_idx と独立スコア) |
| `saddle_room_score` | 0.0 | 装鞍所 状態 (-3 〜 +3) |

---

## 動作 test 結果

```python
race_name, horses, horse_ids, race_info = pcv18.parse_shutuba('202608030612')
df = pcv18.build_features(horses, race_info, model)
```

| metric | value |
|---|---|
| df shape | 14 行 × **277** columns |
| V15 features in df | **150/150** (完全不変) |
| V18 features in df | **15/15** (全追加成功) |
| V15 model 推論 | OK (use_features fallback で V18 features 削除) |
| sample 値 | gaika_top3r_3r=0.33, gaika_winrate=0.20, jockey_x_trainer_wr=0.15 |

→ ★ V15 推論不変 + V18 features 追加成功 ★

---

## V18 candidate 構成

| 項目 | V15 (現行) | V18 candidate (Phase 11) |
|---|---|---|
| feature 数 | 150 | **165** (150 + 15) |
| 学習 model | LGB+XGB ensemble | (未学習) |
| build_features | predict_core.py | predict_core_v18.py (新規) |
| 推論経路 | predict_one_race.py | (V18 用 wrapper 5/12+ 作成予定) |

---

## V15 投資保護

✅ V15 model 不変
✅ tools/predict_core.py 不変
✅ tools/daily_predict.py 不変
✅ tools/race_auto_notify.py 不変
✅ app.py 不変
✅ schtask 不変
✅ 累計 +¥13,420 維持 (5/10 終了時)

---

## 5/11-5/15 task

### 5/12 (火) 平日 (開催無し)
- [ ] 外厩 features 本実装 (UKC.外厩 lookup + expanding window)
- [ ] 時系列オッズ features 本実装 (save_odds_base から 3h/30m 差分計算)

### 5/13 (水)
- [ ] 騎手マスタ拡張 features 本実装 (KKA を distance/track/class で再集計)
- [ ] 返し馬/パドック features 本実装 (CYB/TYB 詳細解析)

### 5/14 (木)
- [ ] V18 学習 data 構築 (V15 base 150 + 新 15)
- [ ] V18 model 学習 (4-fold WF、 同 LGB+XGB ensemble 構成)

### 5/15 (金)
- [ ] V18 WF 評価 (vs V15 baseline AUC 0.8939)
- [ ] LIVE retro (5/14 dry-run)

### 5/16 (土) 来週末
- [ ] V18 paper trading 開始
- [ ] V15 主軸維持、 V18 retro 比較

---

## 期待効果 (V18 全 features 本実装後)

- AUC: V15 0.8939 → V18 candidate 0.90+ (期待 +0.01-0.02)
- ROI: V15 113.8% (5/10 day) → V18 候補 120%+ (時系列オッズ + 騎手拡張効果)
- 5/10 弱点 (1 番人気/低オッズ過信) → 騎手 X 距離/クラス 補正で改善期待
