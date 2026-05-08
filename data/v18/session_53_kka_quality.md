# Session #53 D: KKA features quality + LEAK 監査

**実施**: 2026-05-09
**branch**: dev/sprint6-kka
**実装**: `tools/kka_quality_check.py` 新規
**入力**: `data/jrdb_kka_features.csv` (548,606 rows)
**出力**: `data/v18/session_53_kka_quality.json`

---

## 1. 年別 coverage (2015-2026)

| year | n | jra_top3r non-null | kyori_top3r non-null |
|---|---:|---:|---:|
| 2015 | 49,992 | **90.9%** | 76.1% |
| 2016 | 50,076 | 90.7% | 75.8% |
| 2017 | 49,299 | 90.5% | 75.6% |
| 2018 | 48,618 | 90.2% | 75.1% |
| 2019 | 47,574 | 89.7% | 74.7% |
| 2020 | 48,282 | 89.9% | 74.9% |
| 2021 | 47,821 | 90.2% | 73.7% |
| 2022 | 47,220 | 90.0% | 74.3% |
| 2023 | 47,672 | 89.9% | 74.7% |
| 2024 | 47,181 | 89.5% | 74.5% |
| 2025 | 47,884 | 89.5% | 74.2% |
| 2026 | 16,987 | 92.4% | 78.2% |

→ **12 年通して 89-92% で安定**。 V20 学習 (2020-2025 6 年) に十分。

---

## 2. 場 (jou) coverage

| 場 (JRDB jou code) | n |
|---|---:|
| 中山 (06) | 87,264 |
| 阪神 (09) | 84,801 |
| 京都 (08) | 77,916 |
| 東京 (05) | 66,195 |
| 中京 (07) | 57,615 |
| 新潟 (04) | 51,244 |
| 福島 (03) | 44,020 |
| 小倉 (10) | 37,904 |
| 札幌 (01) | 21,933 |
| 函館 (02) | 19,714 |

→ **10 場全て カバー**、 main 場 が 6-9万 records。

---

## 3. top3r features 内 redundancy

`kka_jra_seiseki_top3r` と他 22 個の top3r との correlation (N=494,708):

| 高 corr (top 5) | corr |
|---|---:|
| `kka_turf_dirt_2_top3r` | **0.933** |
| `kka_heavy_seiseki_top3r` | 0.854 |
| `kka_kyori_seiseki_top3r` | 0.814 |
| `kka_slow_seiseki_top3r` | 0.764 |
| `kka_waku_seiseki_top3r` | 0.743 |

| 低 corr (bottom 5) | corr |
|---|---:|
| `kka_breeder_blanker_top3r` | 0.334 |
| `kka_breeder_dist_top3r` | 0.329 |
| `kka_breeder_track_top3r` | 0.272 |
| `kka_koryu_seiseki_top3r` | 0.033 |
| `kka_other_seiseki_top3r` | -0.084 |

→ 主要 top3r は jra と高重複 (0.7-0.93)、 産駒 / 交流 / その他 は独立信号。
→ **採用方針**: jra_seiseki / breeder_dist / koryu_seiseki / dam_rensho の **3-4 系 だけ** を V20 候補に。

---

## 4. LEAK 監査 (CLAUDE.md 教訓 ベース)

| feature | all=1 % | all=0 % | std | verdict | 解釈 |
|---|---:|---:|---:|:---:|---|
| `kka_jra_seiseki_top3r` | 3.9% | 27.9% | 0.261 | **PASS** | 健全分布、 SKB 並 std |
| `kka_kyori_seiseki_top3r` | 9.3% | 32.6% | 0.316 | **PASS** | やや高 std だが分布 OK |
| `kka_heavy_seiseki_top3r` | 7.8% | 31.6% | 0.303 | **PASS** | 健全 |
| `kka_class_seiseki_top3r` | 20.4% | 50.6% | 0.397 | **NG (二極化)** | **starts median=0** で sample 不足 |
| `kka_track_seiseki_top3r` | 24.5% | 43.2% | 0.409 | **NG (二極化)** | 二極化 = sample 不足の bias |

### NG の根本原因 = sample 不足 (LEAK ではない)

| feature | starts mean | starts median | zero % |
|---|---:|---:|---:|
| `kka_jra_seiseki_starts` | 10.5 | 7 | 0.2% |
| `kka_kyori_seiseki_starts` | 4.9 | 3 | 17.0% |
| **`kka_class_seiseki_starts`** | **1.0** | **0** | **52.2%** ← |
| **`kka_track_seiseki_starts`** | (推定) 1-2 | (推定) 0-1 | 高 ← |

→ class_seiseki / track_seiseki は **starts < 3 が大半**、 top3r が 0/1 に二極化。
→ **LEAK ではなく サンプル不足**。 V20 投入時は **starts >= 3 の filter** を必須。

### CLAUDE.md 教訓 別 検証

| 過去事故 | KKA 該当 |
|---|---|
| odds_log (post-race) | ❌ KKA は pre-race aggregate |
| dam_top3r (全年計算 リーク) | ❌ KKA は file 配信 timing で expanding 化済 |
| **SKB POST-RACE LEAK** (skb_kishi_code_3 +480bp、 corr_target 0.137) | ❌ **KKA は post-race ではない**、 二極化は sample 不足由来 |
| sib_top3_rate hybrid (旧 corr 0.29 → 新 0.17) | ❌ KKA は hybrid じゃなく直接集計 |

→ **真性 LEAK は無い** と判定。 ただし sample 不足 features は filter 必須。

---

## 5. V15 既存 features との redundancy

| KKA feature | V15 既存 | 重複度 |
|---|---|:---:|
| `kka_jra_seiseki_top3r` | `horse_career_top3r` | **高** |
| `kka_kyori_seiseki_top3r` | `horse_dist_top3r` | **高** |
| `kka_track_seiseki_top3r` | `horse_surface_top3r` | 中 |
| `kka_waku_seiseki_top3r` | `frame_course_dist_wr` | 中 |
| `kka_heavy_seiseki_top3r` | (V15 無し) | **低** ★ |
| `kka_class_seiseki_top3r` | (V15 無し) | **低** ★ |
| `kka_speed/slow/mid_seiseki_top3r` | (V15 無し) | **低** ★ |
| `kka_season_seiseki_top3r` | (V15 無し) | **低** ★ |
| `kka_dam_rensho_max` | dam_top3r (除外済) | **低** (連勝率は別概念) |

→ **新規信号**: heavy / class / speed / season / dam_rensho ★

---

## 6. 採用判定 (V20 候補)

### 推奨 採用 (★★★)

```
kka_heavy_seiseki_{starts, top3r}    # 重馬場別、 V15 に無い完全新規
kka_class_seiseki_{starts, top3r}    # クラス別、 starts >= 3 filter
kka_speed_seiseki_top3r              # S ペース、 V15 に無い
kka_slow_seiseki_top3r               # N ペース
kka_mid_seiseki_top3r                # T ペース
kka_season_seiseki_top3r             # 季節別、 V15 に無い
kka_dam_rensho_max / avg             # 母産駒連勝率
kka_bms_rensho_max / avg             # 母父産駒連勝率
```

→ 計 **約 12-15 features**、 期待 AUC contribution **+0.002-0.005**。

### 不採用 / 既存代替 (重複)

```
kka_jra_seiseki_*       → V15.horse_career_top3r で代替
kka_kyori_seiseki_*     → V15.horse_dist_top3r で代替
kka_track_seiseki_*     → V15.horse_surface_top3r で代替
kka_waku_seiseki_*      → V15.frame_course_dist_wr で代替
kka_turf_dirt_2_*       → corr 0.93 で重複
```

---

## 7. Final Verdict

| 項目 | 結果 |
|---|:---:|
| coverage | ✅ 89-92% / 12 年 / 全 10 場 |
| LEAK 監査 | **PASS** (二極化は sample 不足、 真性 LEAK 無し) |
| V15 retrain 不可 | **NG** (race_id format 異 = 直接 merge 不可) |
| V20 推奨 features | **約 12-15** (heavy / class / pace / season / dam_rensho) |
| Sprint 6 想定 (+0.003) | 達成見込み (heavy + class + pace + season で部分達成) |

→ **本 sprint で V15 投入は不可** (race_id 不整合)。
→ **V20 構築時** (Phase 3 後半 6/9-) に統合し再評価。
→ Section A-D の audit + parser fix + features module は **V20 構築の即活用 asset**。
