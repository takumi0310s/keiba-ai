# Phase 11 真値化 — data audit (2026-05-10 22:30)

## 結論 (honest scope)

| group | features | 実装可能 | data source | 状態 |
|-------|----------|----------|-------------|------|
| A. 外厩 (gaika) | 4 | **1 / 4** | jrdb_kyi.csv 放牧先 | gaika_id_enc のみ |
| B. 時系列オッズ | 4 | **0 / 4** | — | 全 NG (data なし) |
| C. 騎手拡張 | 4 | **4 / 4** | jrdb_kyi.csv 騎手 columns + history | 全 OK |
| D. 返し馬/パドック | 3 | **1-2 / 3** | TYB (直前) | 部分 |
| **計** | **15** | **6-7 / 15** | — | partial |

→ 1-3h scope で 6-7 features 真値化、 残 8-9 は data 工事が別途必要。

---

## 1. JRDB data 状態

| dir | size | 内容 | 期間 |
|-----|------|------|------|
| `data/jrdb/extracted/` | 6.3GB | 全 JRDB record types | 2015-01〜2026-05 |
| `data/jrdb_kyi.csv` | (大容量) | KYI parsed CSV | 全期間 |
| `data/jrdb_kka_v2.csv` | (大容量) | KKA parsed CSV | 全期間 |
| `data/odds_base_*.csv` | 35-491 row/日 | 朝 8:00 odds snapshot | 5/8-5/10 |

KYI 既存 columns (一部):
```
場コード, 年, 回, 日, R, 馬番, 血統登録番号, 馬名,
IDM, 騎手指数, 情報指数, 総合指数, 脚質, 距離適性, 上昇度,
基準オッズ, 基準人気順位, 基準複勝オッズ, 基準複勝人気順位,
人気指数, 調教指数, 厩舎指数, 騎手期待連対率, 騎手期待単勝率, 騎手期待3着内率,
騎手コード, 調教師コード, 入厩何走目, 入厩何日前,
放牧先, 放牧先ランク, 厩舎ランク,
jra_race_id, nk_race_id, …
```

→ **放牧先 (farm name) + 放牧先ランク (rank A-E) + 騎手期待 3 種** は実 data あり。

---

## 2. group 別 audit

### A. 外厩 (4 features)

| feature | 実装可能? | source | note |
|---------|----------|--------|------|
| `gaika_id_enc` | ✅ | KYI 放牧先 (text) | hash 化 → 整数 ID |
| `gaika_top3r_3r` | ❌ | history aggregation 必要 | 馬個別 過去 3 R 外厩成績、 当 race 横断的集計 |
| `gaika_winrate` | ❌ | 外厩 × 全馬 history | 外厩 ID 別 通算 winrate (expanding window) |
| `gaika_dist_winrate` | ❌ | 外厩 × 距離帯 history | 同上 + 距離 cut |

**注**: V15 は既に `gaisha_rank` (放牧先ランク A-E、 0-5 整数) を実装済 (`tools/predict_core.py` `compute_gaisha_features`)。 Phase 11 が追加するのは ID 化 + 集計値。 集計には:
- 全 KYI history (~10 年分) を 馬 ID + 放牧先 で結合
- finish 結果と join (RCA SED data)
- expanding window (date 順 cumsum-current) で計算
が必要、 1-3h scope を超える。 → **5/12 平日に別 task**

### B. 時系列オッズ (4 features)

| feature | 実装可能? | 理由 |
|---------|----------|------|
| `odds_change_3h_v18` | ❌ | 多 snapshot odds なし |
| `odds_change_30m_v18` | ❌ | 同上 |
| `popularity_shift_v18` | ❌ | 同上 |
| `odds_volatility_v18` | ❌ | 同上 |

odds_base CSV: 1 (race, horse) → 1 snapshot のみ (朝 8:00)。
JRDB OW: 1 file/日のみ (1 snapshot)。
→ **time-series odds infrastructure 構築が前提** (5/24+ JV-Link backfill 後)

### C. 騎手拡張 (4 features)

| feature | 実装可能? | source | note |
|---------|----------|--------|------|
| `jockey_dist_winrate` | ✅ | V15 cache の jockey + distance history | expanding |
| `jockey_track_winrate` | ✅ | 既存 V15 jockey_surface_wr 拡張 | 既に近い |
| `jockey_class_winrate` | ✅ | KYI クラスコード + history | expanding |
| `jockey_x_trainer_wr` | ✅ | KYI 騎手コード × 調教師コード + history | expanding |

→ 4/4 実装可能。 V15 cache (527K row、 2015-2025) + KYI から build。

### D. 返し馬/パドック (3 features)

| feature | 実装可能? | source | note |
|---------|----------|--------|------|
| `return_horse_score` | △ | TYB 返し馬 column 確認必要 | TYB (直前) parsing |
| `paddock_eval_v18` | ✅ | V15 既存 jrdb_paddock_idx を re-encode | 数値変換 |
| `saddle_room_score` | △ | TYB 装鞍所 column 確認必要 | TYB parsing |

→ 1-2/3 実装可能。 TYB parser 確認後判断。

---

## 3. 実装 priority (scope 1-3h、 honest)

| step | 時間 | 内容 |
|------|------|------|
| 1 | 30 min | gaika_id_enc 真値化 (KYI 放牧先 → encode) |
| 2 | 60 min | jockey 4 features 真値化 (KYI + V15 cache history) |
| 3 | 30 min | paddock_eval_v18 真値化 |
| 4 | 30 min | 動作 test (5/10 sample) + commit |

合計 ~2.5h。 残 8 features (gaika 集計 3、 odds 4、 return/saddle 1-2) は **5/12 平日 task** へ持ち越し。

---

## 4. Phase 15 教訓尊重

| ルール | 適用 |
|--------|------|
| ★ fabrication 禁止 ★ | data なき features は constant default のまま、 "data-pending" 明示 |
| ★ scope 現実調整 ★ | 6-7 features 真値化、 残 honest report |
| ★ V15 完全不変 ★ | tools/predict_core_v18.py のみ touch |
| ★ AI session 完結 ★ | 1-3h 内完結、 不可なら halt |

---

## 5. V15 投資保護

| 不変 | 状態 |
|------|------|
| `predict_core.py` | ★完全不変★ |
| `daily_predict.py` | ★完全不変★ |
| `app.py` | ★完全不変★ |
| `keiba_model_v15_central*.pkl.gz` | ★完全不変★ |
| 累計 +¥14,140 | ★維持★ |

予定 changes: `tools/predict_core_v18.py` のみ (V18 candidate predict core、 V15 推論 path 影響ゼロ)。
