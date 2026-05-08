# Session #53 C: KKA features 化 + V15 統合 PoC

**実施**: 2026-05-09
**branch**: dev/sprint6-kka
**実装**: `tools/kka_features.py` 新規
**入力**: `data/jrdb_kka_v2.csv` (548,606 rows)
**出力**: `data/jrdb_kka_features.csv` (gitignore 済、 548,606 rows x 100 cols)

---

## 1. 派生 features 設計

### 23 個の seiseki block × 4 features = 92 numeric

各 seiseki block (1着 / 2着 / 3着 / 着外 の 4 値) から:

```
{block}_starts  = 1 + 2 + 3 + out          # レース経験
{block}_winr    = 1 / starts                # 勝率
{block}_top2r   = (1 + 2) / starts          # 連対率
{block}_top3r   = (1 + 2 + 3) / starts      # 複勝率
```

block 一覧:

| カテゴリ | block 名 | 意味 |
|---|---|---|
| 主要 | `jra_seiseki` | JRA 中央成績 |
| 主要 | `koryu_seiseki` | 交流成績 |
| 主要 | `other_seiseki` | その他成績 |
| 距離 | `kyori_seiseki` | 当該距離別 |
| 距離 | `turf_dirt_2`, `turf_dirt_2_dist` | 芝ダ 2 着集計 |
| トラック | `track_seiseki` | 右回り / 左回り別 |
| トラック | `rotation_sei` | ローテーション別 |
| 馬場 | `heavy_seiseki` | 重馬場 |
| 馬場 | `saka_seiseki` | 坂別 (上り / 下り) |
| 状態 | `rest_seiseki` | 休み明け別 |
| クラス | `class_seiseki` | クラス別 |
| ペース | `speed_seiseki` | S ペース |
| ペース | `slow_seiseki` | N ペース |
| ペース | `mid_seiseki` | T ペース |
| 季節 | `season_seiseki` | 春/夏/秋/冬 |
| 枠 | `waku_seiseki` | 内/中/外 |
| 産駒 | `breeder_*` (6 個) | 産駒成績 |

### 6 個の連勝率 (passthrough)

```
kka_dam_rensho_max / dam_rensho_min / dam_rensho_avg
kka_bms_rensho_max / bms_rensho_min / bms_rensho_avg
```

→ **計 98 candidate features**。

---

## 2. coverage 結果 (全 548,606 rows)

| feature | non-null % | mean | median |
|---|---:|---:|---:|
| `kka_jra_seiseki_top3r` | **90.2%** | 0.286 | 0.250 |
| `kka_jra_seiseki_starts` | 90.4% | 10.49 | 7 |
| `kka_kyori_seiseki_top3r` | 75.0% | 0.333 | 0.300 |
| `kka_track_seiseki_top3r` | 39.8% | 0.396 | 0.333 |
| `kka_heavy_seiseki_top3r` | 78.8% | 0.320 | 0.286 |
| `kka_class_seiseki_top3r` | 43.2% | 0.334 | 0.000 |
| `kka_dam_rensho_max` | **100.0%** | 13.89 | 14.0 |
| `kka_bms_rensho_max` | **100.0%** | 10.46 | 10.0 |

→ 主要 features は **75-90% カバー**、 連勝率系は **100%**。

---

## 3. LEAK risk 評価

### JRDB Paci (KKA 含む) の配信 timing

| 曜日 | 時刻 | 用途 |
|---|---|---|
| 月・木 | 19:00 | 翌週末レース 用 (土日) |
| 金 | 20:00 | 翌日レース 用 (土) |
| 土 | 20:00 | 翌日レース 用 (日) |

→ KKA は **常に レース直前まで の累積成績** を含み、 当該レースは含まない (pre-race aggregate)。

### CLAUDE.md の過去事故との対比

| 事故 | KKA 適用可否 |
|---|---|
| dam_top3r (全年計算 = リーク) | ❌ KKA は file 配信時点までの累積、 OK |
| SKB POST-RACE LEAK (skb_kishi_code_3) | ❌ SKB は post-race だが KKA は pre-race、 OK |
| sib_top3_rate hybrid (旧版) | ❌ KKA は seiseki block ごとに集計、 hybrid じゃない |

→ **KKA 集計 logic 上 は LEAK 無し**。 ただし Section D で corr_target / monotonic を実測検証。

---

## 4. V15 統合 PoC の限界

V15 cache (`_v15_optuna_df_cache.pkl.gz`) は 527,280 rows / 232 cols。 race_id 形式が KKA と異なる:

- V15: `race_id` 10 chars (例: `'0915150804'` = JRA encode)
- KKA: `race_id` 12 chars (例: `'201508010102'` = JRDB encode YYYYJJKKDDRR)

→ jou code が JRA / JRDB で異なるため **直接 merge 不可** (V15 retrain 時に jou map table が必要)。

### 対応策

- **本 sprint**: V15 retrain せず KKA features を **standalone 評価** (内部統計 + LEAK 監査)
- **V20 構築時** (Phase 3 後半 6/9-): JRA-VAN base + KKA features を統合再学習
- 期待 AUC contribution: V15 baseline 0.8788 → +0.003-0.005 (Sprint 6 想定範囲)
  → V15 既存 features (`horse_career_top3r`, `horse_dist_top3r`, `horse_surface_top3r`) と部分重複の可能性あるため上振れは不確実。

---

## 5. 次の action (Section D)

KKA features の **真の AUC contribution** は V20 retrain で測定するが、 本 sprint では:
- LEAK 監査 (corr_target > 0.3 の極端値検出、 monotonic test、 SKB-like 異常パターン検出)
- V15 既存 features (`horse_career_top3r` 等) との redundancy 推定
- coverage 詳細 (年別 / 場別 / クラス別)

→ Section D で実施。
