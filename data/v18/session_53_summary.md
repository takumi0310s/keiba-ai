# Session #53 統合 summary: JRDB KKA parser 修復

**実施**: 2026-05-09 00:00-00:15 (15 分、 並行作業中)
**branch**: dev/sprint6-kka (origin/main 6c0680ad から分岐)
**目的**: jrdb_kka.csv の seiseki_* 全 NaN (0%) を修復、 V20 統合 候補 features を確定

---

## TL;DR

- **root cause 確定**: `tools/download_parse_jrdb_extra.py:336` の `v = val_str.strip()` が ZZ9*4 (12-byte 4 値) field の先頭空白を除去 → 長さ check fail → 全 record で seiseki = None
- **fix 実装**: `tools/jrdb_kka_parser_v2.py` 新規 (既存は不変)
- **coverage**: jra_seiseki_1 **0% → 90.4%** (548K rows / 12 yr)
- **LEAK 監査**: PASS (KKA は pre-race aggregate、 SKB-like の post-race leak 無し)
- **V20 候補**: 約 12-15 features (heavy / class / pace / season / dam_rensho)、 期待 +0.002-0.005 AUC
- **本 sprint で V15 投入**: NG (race_id format 異 = 直接 merge 不可、 V20 retrain 必須)

---

## 各領域 結果

### A. KKA parser audit ([session_53_kka_audit.md](session_53_kka_audit.md))

- spec 把握 (KKA = 競走馬拡張、 322 byte 固定長、 ZZ9*4 = 4x3-byte 数値)
- 既存 parser logic を逆推定し root cause を 1 行に特定
- bug の再現テスト: jra_seiseki_1: None ← 全 record

### B. parser 修復 + 動作検証 ([session_53_kka_parser_fix.md](session_53_kka_parser_fix.md))

- `_parse_level_12_v2`: strip を slice の前にしない、 各 3-byte slice を `_safe_int` で個別処理
- 全 1,228 files parse 成功 (0 errors)
- 548,606 rows / 100 cols 出力 (`data/jrdb_kka_v2.csv`、 196 MB、 gitignore)
- coverage: jra_seiseki_1 **90.4%**、 12 年 (2015-2026) 通して 89-92% で安定

### C. KKA features 化 ([session_53_kka_features.md](session_53_kka_features.md))

- 23 個の seiseki block × 4 (starts/winr/top2r/top3r) + 6 連勝率 = **98 candidates**
- `tools/kka_features.py` 実装、 `data/jrdb_kka_features.csv` (548K rows / 100 cols)
- LEAK 概念検証: KKA は **pre-race aggregate** (Paci file 配信 timing 月木 19:00 / 金土 20:00)

### D. quality + LEAK 監査 ([session_53_kka_quality.md](session_53_kka_quality.md))

- coverage: 全 12 年 89-92%、 全 10 場 19K-87K rows
- top3r 内 redundancy: jra_seiseki と turf_dirt_2 が corr 0.93 (重複)、 koryu / other は独立
- LEAK 監査: 5 主要 features 中 3 PASS、 2 NG (sample 不足由来、 真性 LEAK 無し)
- V15 既存 features との重複: 5/9 高重複、 4/9 新規信号

---

## V20 採用 推奨 (約 12-15 features)

### 新規信号 (V15 に類似 features 無し) ★★★

```
kka_heavy_seiseki_top3r       # 重馬場別 (PASS)
kka_class_seiseki_top3r       # クラス別 (要 starts >= 3 filter)
kka_speed_seiseki_top3r       # S ペース
kka_slow_seiseki_top3r        # N ペース
kka_mid_seiseki_top3r         # T ペース
kka_season_seiseki_top3r      # 季節別
kka_dam_rensho_max / avg      # 母産駒連勝率
kka_bms_rensho_max / avg      # 母父産駒連勝率
```

### 不採用 (V15 既存と重複)

```
kka_jra_seiseki_*       → V15.horse_career_top3r で代替
kka_kyori_seiseki_*     → V15.horse_dist_top3r で代替
kka_track_seiseki_*     → V15.horse_surface_top3r で代替
kka_turf_dirt_2_*       → corr 0.93 で重複
```

---

## 投資保護 (絶対遵守 確認)

| 項目 | 状態 |
|---|:---:|
| main HEAD `6c0680ad` 不変 | ✅ |
| 既存 dev branches 不変 | ✅ (Session #52 commits は dev/training-poc に温存) |
| V15 model `keiba_model_v135_*.pkl.gz` 不変 | ✅ |
| 既存 `predict_core.py` / `daily_predict.py` / `app.py` 不変 | ✅ |
| 既存 `tools/download_parse_jrdb_extra.py` 不変 | ✅ (新 v2 file 別作成) |
| 既存 `data/jrdb_kka.csv` (broken) 不変 | ✅ (新 v2 csv 別生成) |
| schtasks 41 件 不変 | ✅ |
| 5/9 朝 V15 daily_predict 動作変更 | **無し** |

→ **5/9 朝 V15 案B改 完全同一動作 を絶対保証**。

---

## 5 commits (dev/sprint6-kka, origin/main から分岐)

```
2da955bc Session #53 D: KKA quality check
0eede7aa Session #53 C: KKA features module
cc7cfe97 Session #53 B: KKA parser v2
1b4fe9e1 Session #53 A: KKA parser audit
6c0680ad (origin/main) AUDIT-1
```

→ section E の本 commit (summary doc) を加えて 5 commits。

---

## 次の action (V20 構築 5/24+)

1. V20 学習 data spec 確定時に KKA features を統合 (推奨 12-15 件)
2. JRA / JRDB の race_id mapping table を構築 (jou code conversion)
3. V20 retrain で AUC contribution を実測 (+0.002-0.005 想定)
4. 5/16 V18 trial に **KKA は乗せない** (race_id 不整合 + V15 retrain 不要のため)

→ Phase 3 後半 (6/9-6/30) の V20 構築タスクへ pass。
