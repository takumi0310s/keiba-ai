# Session #54 A: NAR V4 現状 audit

**作成**: 2026-05-09 (Session #54 A)
**対象**: data/nar/models/keiba_model_nar_v4.pkl
**位置付け**: V5 改善 検討の base line 確定

---

## 1. V4 model spec

| 項目 | 値 |
|------|----|
| version | nar_v4 |
| AUC | **0.8145** (LGB 0.8142 / XGB 0.8144 / 重み LGB 50% / XGB 50%) |
| 学習 rows | 49,213 |
| 学習 races | 4,821 |
| trained_at | 2026-03-09 17:28:13 |
| ensemble | LGB + XGB (50/50) |
| n_features | **22** (Pattern B、 live data 込み) |

### V4 features (22)
```
odds_log, num_horses, distance, surface_enc, condition_enc,
course_enc, horse_weight, weight_carry, age, sex_enc,
horse_num, bracket, horse_num_ratio, bracket_pos,
carry_diff, dist_cat, weight_cat, age_group,
jockey_wr, jockey_place_rate, pop_rank, is_nar
```

注: V4 は **Pattern B (live)**。 paper trade 専用なので odds_log / horse_weight / pop_rank / condition_enc の 含み 妥当。

---

## 2. 学習 data source

| ファイル | 行数 | 期間 | 内容 |
|---------|-----|------|------|
| `data/nar_all_races.csv` | **54,159 rows** | 2024-01-01 ~ 2025-05-14 | V4 training base (33 cols) |
| `data/nar_merged.csv` | 14,093 rows | 旧 (V4 leak-free 版) | 49 cols 拡張済 (一部) |
| `data/chihou_full_enriched.csv` | 17,071 rows | 2009-2020 (古い) | 旧 NAR data |
| `data/chihou_races_full.csv` | -- | 2009-2020 | 旧 |

V4 は `nar_all_races.csv` (54K rows) を base に学習。 49,213 / 54,159 = 91% 利用 (finish 取消除外、 NaN drop 等)。

---

## 3. nar_all_races.csv 取得済 features (V4 未使用 含む)

| col | 非 NaN | 用途 |
|-----|------|------|
| race_id, race_name, race_date, course, course_code | 100% | meta |
| distance, surface, condition, weather, class_info | 100% | race-level |
| num_horses, finish, bracket, horse_num | 100% | base |
| horse_name, sex_age, weight_carry | 100% | horse |
| jockey_name, trainer_name | 100% | jockey/trainer |
| **time, last3f, margin** | **98%+** | **post-race (LEAK)** / **last3f は 過去レース集計可** |
| **odds, pop_rank** | **99%** | live (V4 採用済) |
| **horse_weight** | **99%** | live (V4 採用済) |
| **horse_weight_change** | **99%** | **★ V4 未活用、 V5 候補 ★** |
| pass_order | -- | post-race |
| tansho/fukusho/umaren/wide/trio/tierce_payout | -- | post-race (検証用) |

→ V5 候補は **horse_weight_change** + **last3f (前走集計)** + **trainer 系 expanding** + **horse 個別 expanding**

---

## 4. NAR data source 状況

| source | 内容 | 取得状況 |
|--------|------|---------|
| 楽天競馬 (rakuten) | NAR odds / 出馬表 / 結果 | tools/scrape_nar_*.py で取得済 |
| 南関 (nankan.jp) | 大井/船橋/川崎/浦和 | scrape_nar_today.py |
| netkeiba 地方 | 一部 | 部分カバー |
| **JV-Link** | **不対応** | NAR 取得不可 |
| **JRDB** | **不対応** | NAR 取得不可 |

→ NAR は 中央 V15 と独立 system、 JV-Link / JRDB の 恩恵なし。

---

## 5. NAR 既存 train script 状況

| ファイル | 内容 |
|---------|------|
| `archive/nar/train_nar_v4.py` | Pattern B (live)、 22 features → 現行 V4 model |
| `archive/nar/train_nar_v4_leakfree.py` | Pattern A (leak-free)、 37 features (base 30 + V4_NEW 7) |
| `archive/nar/train_nar_v3*.py` | V3 旧版 |
| `archive/nar/backtest_nar_*.py` | backtest |

注: archive/nar/train_nar_v4_leakfree.py は 37 features 設計 だが 学習 model 未保存。
→ V5 では **leak-free 拡張 7 features を Pattern B に統合** + 新規 features 追加

---

## 6. V4 vs V5 改善 path

| 改善 path | 期待 | 工数 |
|----------|------|------|
| 1. expanding features 7 件 統合 (V4 leak-free 版から移植) | +0.005-0.010 | 1h |
| 2. horse_weight_change 追加 | +0.001-0.003 | 0.5h |
| 3. last3f 集計 (前走平均) | +0.001-0.003 | 1h |
| 4. trainer_wr (騎手と同様 expanding) | +0.001-0.002 | 0.5h |
| 5. course × distance 適性 (大井/船橋/川崎/浦和 別) | +0.002-0.005 | 1h |

→ V5 候補: V4 22 + expanding 7 + horse_weight_change + last3f + trainer + course_dist = **約 35 features**
→ 期待 AUC: 0.8145 + 0.005-0.015 = **0.820-0.830** (audit 予想 0.82-0.83 と一致)

---

## 7. 5/12 paper trade 投入候補性

✅ **NAR は paper trade のみ** (実投票なし) → V5 投入は安全
✅ 中央 V15 と完全分離
✅ V4 は production NAR system で運用中だが、 V5 は paper のみで A/B 比較可能

---

## 8. 結論

✅ V4 audit 完成: 22 features, AUC 0.8145, 49K rows, 4,821 races
✅ data/nar_all_races.csv (54K rows, 2024-2025) が学習 base
✅ V4 Pattern B (live) → V5 でも Pattern B 維持
✅ 改善 path: expanding features + horse_weight_change + last3f + course_dist 適性
✅ V5 期待 AUC 0.820-0.830 (+0.005-0.015)
✅ 5/12 paper 投入候補性 高
