# Phase 18 D: V18 再学習で features 寄与確認 plan

**作成**: 2026-05-10 (Session #91 Phase 18 D、 ★ Opus 4.7 ★)
**前提**: Phase 13 stub features 真値化後 (Phase 18 A/B/C) に V18 再学習で寄与確認
**目的**: Phase 11/12/13 の合計 57 features (15+17+25) の真の AUC 寄与を測定

---

## 1. V18 再学習 必要条件 (本 session で実施 NG)

| # | 条件 | 状態 | 担当 phase |
|---|------|------|----------|
| 1 | Phase 11 JRDB 15 features 真値化 | ⚠ 未確認 (Phase 11 commit a2a2279b で skeleton) | Phase 18 拡張 |
| 2 | Phase 12 DataLab 17 features 真値化 | ⚠ 未確認 (Phase 12 commit b1751da5 で skeleton) | Phase 18 拡張 |
| 3 | Phase 13 master 25 features 真値化 | ⚠ 未着手 (Phase 18 A 待ち) | Phase 18 A |
| 4 | 直近 6 ヶ月 master backfill | ⚠ 未着手 (Phase 18 B 待ち) | Phase 18 B |
| 5 | V18 学習 train data に各 phase features 統合 | ⚠ 未着手 | Phase 18 D |
| 6 | GPU 環境 (RTX 4070 Ti SUPER 16GB) で LGB/XGB/FT/IR 学習 | ✅ 利用可 | Phase 18 D |

→ 本 session で V18 再学習は **不能** (依存条件 1-5 全て未達)

---

## 2. V18 再学習 設計 (5/24+ Phase 3 後半)

### 2.1 学習構成

| 段階 | features | 目的 | 期待 AUC |
|------|---------|------|---------|
| baseline (V18 sib_w5) | 190 (5/8 完了) | リーク 0% baseline | 0.8847 (BT 2025) |
| V18.1 (+ Phase 11 真値) | 205 | JRDB 拡張 寄与 | 0.886-0.890 |
| V18.2 (+ Phase 12 真値) | 222 | DataLab 拡張 寄与 | 0.888-0.895 |
| V18.3 (+ Phase 13 真値、 cache hit のみ) | 247 | master 真の寄与 | 0.890-0.910 |
| V20 (V18.3 + 4-model ensemble) | 247 | LGB+XGB+FT+IR | 0.910-0.925 |

### 2.2 fold 構成

- WF 6-fold (2020-2025、 各年 test、 残り train)
- 各 fold で feature importance ranking 出力
- Phase 13 25 features 個別 P/L (importance < 0.001 で削除候補)

### 2.3 学習時間 (GPU、 RTX 4070 Ti 16GB)

| model | 単 fold 時間 | 6 fold | 備考 |
|-------|-------------|--------|------|
| LightGBM (GPU) | 2-3 min | 15-20 min | gpu_use_dp=True |
| XGBoost (GPU) | 5-8 min | 30-50 min | tree_method=hist + device=cuda |
| FT-Transformer (GPU) | 30-45 min | 3-5 h | rt_dl パッケージ |
| IntraRace Attention (GPU) | 20-30 min | 2-3 h | custom PyTorch |
| Grid weight 最適化 | 10-15 min | 1 fold で十分 | scipy.optimize |

**合計**: 5-9 h (GPU)、 12-20 h (CPU only)

→ 5/24-5/29 (1 週間) で 6-fold + 4-model 学習完了 想定

---

## 3. feature importance 評価 (V18 再学習後)

### 3.1 LGB importance ranking 出力例

```
1.  master_horse_aitenkai_score    gain=0.025  ★高
2.  master_horse_lap_avg_last3f    gain=0.020  ★高
3.  jrdb_kyi_idm                   gain=0.018  ★高
4.  master_pace_pred                gain=0.012  中
5.  jv_o5_trio_odds_open            gain=0.010  中
...
207. master_horse_lap_dec_phase     gain=0.0001 ★低 → 削除候補
```

### 3.2 削除判断 閾値

- gain < 0.001: 削除候補
- gain < 0.0005: 確実削除 (over-fitting risk)
- gain >= 0.005: V20 採用確実

### 3.3 期待 削除数

Phase 11/12/13 計 57 features のうち、 真値化後でも 10-15 features は default
fill のままに留まる (selector 不明 / DOM 構造 NG / 計算 features は元 data
不足 等)。 これらは LGB importance < 0.0005 で削除される見込。

→ V20 採用 features 数: **207 → 192-197 程度** (10-15 削除)

---

## 4. V20 投入 候補 (6/8 GO 判定)

### 4.1 GO 条件 (再掲、 Session #44 F より)

| 条件 | 閾値 | 備考 |
|------|------|------|
| WF AUC | ≥ 0.880 | V18.3 で達成見込 |
| LIVE retro winner_top1 | ≥ 30% | V18 sib_w5 5/2-5/3 で 34.48% 達成 |
| shift_factor | ≤ 12x | V18 sib_w5 で 1.32x (大幅余裕) |
| paper trade ROI | ≥ 110% | 6/2-6/7 で確認 |
| LEAK 監査 | PASS | 全 features corr_target チェック |

### 4.2 V18.3 / V20 投入 schedule

```
5/24   Phase 11 JRDB 真値化 (預け script 起動)
5/25   Phase 12 DataLab 真値化 (oid hg vendor api)
5/26   Phase 13 master 真値化 + 6 ヶ月 backfill 起動
5/27-30  V18.1/V18.2/V18.3 単 fold 試行 (LGB only)
6/1-3   V18.3 6-fold WF + LIVE retro
6/4-7   V20 4-model ensemble + paper trade
6/8    V20 GO 判定
```

---

## 5. V15 投資保護 (絶対遵守、 V20 投入後も継続)

✅ V18 再学習中も V15 production 完全不変
✅ V20 投入後も 1 ヶ月並行運用 (累計収支 +¥14,140 維持)
✅ 撤退ライン -¥50,000 厳守
✅ V18 候補 model 失敗時は paper trade のみ、 1 円も実弾投入しない

---

## 6. 結論

⚠ 本 session で V18 再学習は不能 (Phase 18 A/B/C 全て未完了)
✅ Phase 18 D 設計確立 (5/24+ Phase 3 後半 で着手、 GPU 5-9h)
✅ feature importance ranking で Phase 11/12/13 真の寄与を判定
✅ V20 GO 判定 6/8 (1 ヶ月前倒し) 維持

---

**Phase 18 D 完了** (Opus 4.7)
