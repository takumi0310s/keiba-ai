# Session #51 サマリ: AUDIT-1 Top 27 backtest

**日付**: 2026-05-08 (Session #51)
**branch**: dev/audit-backtest (main 不変、 V15 投資保護)
**主目的**: AUDIT-1 Top 30 features (★★★ 3 件 = Session #50 並行) を除く 27 件 (★★ 14 + ★ 13) の 一括 backtest

---

## 0. 5 領域 完了

| 領域 | 内容 | 出力 |
|------|------|------|
| A | Top 27 plan + 分類 (18 即実装/7 中期/2 長期) | data/v18/sprint5_top27_classification.md |
| B | 一括 backtest 18 件 (V15 base + 単一 feature) | data/v18/sprint5_top27_backtest_results.md + sprint5_backtest_metrics.json |
| C | combo search 6C2 = 15 pair 全検証 | data/v18/sprint5_combo_top5.md + sprint5_combo_metrics.json |
| D | Sprint 5/6/V20/Phase 4 roadmap 確定 | data/v18/sprint5_6_v20_roadmap.md |
| E | 統合 + push + Discord | (本 doc) |

---

## 1. 重大発見 (★★★)

### 1-1. LEAK 2 件 確定 (V20 学習前 必須)

| feature | AUC | 詳細 |
|---------|-----|------|
| #18 jrdb_sed | **1.0000** | SED は finish/time_sec/abnormal を含む完全 POST-RACE |
| #22 race_review_score | **0.9981** | review_score 自体が POST-RACE 評価 (v12.1 不採用 確認) |

→ V20_LEAK_FEATURES に **+12 件 追加** (SED 10 + review 2)

### 1-2. V15 145 features は 高度 飽和

- combo 全 15 pair で delta ≤ 0
- 単一 feature 追加 / 2-feature combo では V15 superset 構築 不可
- **領域違い** (TFJV) + **異 modal** (動画) が 必須

### 1-3. JRDB KKA parser 不全 (Sprint 6 修復必要)

- `jra_seiseki_*`, `kyori_seiseki_*`, `track_seiseki_*`, `heavy_seiseki_*`, `class_seiseki_*` 全 0% NaN
- AUDIT-1 期待 +0.002-0.005 を 検証不能
- → Sprint 5 では 不採用、 Sprint 6 で parser 修復後 再評価

---

## 2. 27 件 振り分け

| 振り分け先 | 件数 | 工数 | 期待 AUC |
|-----------|----|------|---------|
| Sprint 5 (5/16-5/22): 軽量補正 | 5 | 8h | +0.0006 (V15 0.8939 → V15.5 0.8945) |
| Sprint 6 (5/23-5/30): KKA + UKC | 3 | 16h | +0.003 (→ V15.6 0.8975) |
| V20 (5/22-6/8): TFJV 大規模 (本命) | 4 | 30h | +0.001-0.005 (→ V20 0.8935-0.8985) |
| Phase 4 (7-9月): 動画 + JG | 2 | 86h+ | +0.005-0.010 (→ V21 0.8985-0.9035) |
| **合計** | **14** | **140h+** | **+0.010-0.025** |

(LEAK 2 件 + parser 不全 4 件 + 重複/不採用 7 件 を 除外、 採用 14 件)

---

## 3. AUC 軌跡 予想

```
2026-05-09  V15        0.8939 (本番、 案B改 単独継続)
2026-05-22  V15.5     0.8945 (Sprint 5 軽量補正、 +0.0006)
2026-05-30  V15.6     0.8975 (Sprint 6 KKA + UKC、 +0.0036)
2026-06-08  V20        0.8935-0.8985 (TFJV 大規模、 主軸)
2026-09-01  V21        0.8985-0.9035 (動画 + JG、 中位想定)
```

---

## 4. branch 状態

- `dev/audit-backtest`: 5 commits 独立 (A, B, C, D, E)
- `main` 不変
- `dev/sprint4` (Session #50 並行): 干渉なし
- `dev/training-poc` (Session #48 ベース): 干渉なし
- `predict_core / daily_predict / app.py`: 不変
- V15 model file: 不変
- 5/9 朝 V15 動作: 不変

→ V15 production 完全保護: ✅

---

## 5. 5 commits 履歴

1. `Session #51 A: Top 27 plan + 分類 (18 即/7 中期/2 長期)` (d6a73ca4)
2. `Session #51 B: Top 18 一括 backtest + 重大 LEAK 2 件発見` (cherry-pick 46f35fad)
3. `Session #51 C: combo search 全 15 pair 検証 (V15 飽和 確認)` (cherry-pick a74a355e)
4. `Session #51 D: Sprint 5/6/V20/Phase 4 roadmap 確定` (cherry-pick c3ecf8e7)
5. `Session #51 E: doc 統合` (本 commit)

---

## 6. 結論

✅ AUDIT-1 全 27 件 (★★★ 除く) backtest 完了
✅ LEAK 2 件 確定、 V20_LEAK_FEATURES に 反映必須
✅ V15 飽和 検出 → 大規模 path (TFJV + 動画) 必須
✅ Sprint 5/6/V20/Phase 4 roadmap 確定
✅ V15 投資保護: 不変
✅ Session #50 (Sprint 4 ★★★ 3 件) と独立、 干渉なし

5/9 朝 V15 案B改 維持。 5/16 V18 trial、 5/22 Sprint 5 着手 へ。
