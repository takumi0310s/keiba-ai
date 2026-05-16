# 30 年 backtest 環境 構築 timeline (Session #84)

> Sprint 6 (V20) → Sprint 9 (30 年 backtest) までの schedule。
> 作成: 2026-05-09 (Session #84)

---

## 1. 全体 schedule

| Sprint | 期間 | event | output |
|--------|------|-------|--------|
| Sprint 5.5 | 5/9-5/15 | V15 単独運用 + data 蓄積 | cumulative_results 推進 |
| Sprint 6 (V18) | 5/16 | V18 trial 1 day | V18 LIVE retro |
| Sprint 6 (V20) | 5/22-6/8 | V20 構築 (Session #44 F roadmap v3) | V20 model |
| Sprint 7 | 6/8-6/30 | V20 paper trade | V20 ROI baseline |
| Sprint 7 投入 | 7/1 | V20 段階投入 (5,000円/日) | V20 production |
| Sprint 8 (V21) | 7/15-9/2 | V21 構築 + JRA-VAN RV 動画統合 | V21 model |
| Sprint 8 投入 | 9/2 | V21 投入候補 | V21 production |
| **Sprint 9 (本件)** | **10/1-11/1** | **V22 RL PoC + 30 年 backtest 開始** | **5 戦略 backtest** |
| Sprint 9 完了 | 11/1 | V22 paper + backtest 完了 | report + 採用判定 |
| Sprint 10 | 12/1 | V22 投入候補 | V22 production |

---

## 2. Sprint 6 (V18 + V20 構築) 詳細

### 2.1 V18 trial (5/16)
- 1 day のみ trial、 V18 上限 5,000円
- LIVE retro winner_top1 / shift / max DD 計測
- → Session #74 plan v5 (作成済)

### 2.2 V20 構築 (5/22-6/8、 Sprint 6 残り)
- Session #44 F の Phase 3-5 v3 roadmap に従う
- JV-Link + KKA + sib_*_exp 統合
- 4-model ensemble (LGB+XGB+FT+IR)
- WF AUC 検証 + LIVE retro

---

## 3. Sprint 7 (V20 投入) 詳細

### 3.1 V20 paper trade (6/8-6/30)
- 4 weekend × 4 R = ~16 day paper
- ROI / shift / max DD 計測
- 採用判定: 6/30 GO で 7/1 投入

### 3.2 V20 production 投入 (7/1+)
- 上限 5,000円/日 strict
- 7/1-7/14: 段階投入 (週末のみ)
- 7/15+: 平日含む 1万円/日 候補

---

## 4. Sprint 8 (V21 動画統合) 詳細

### 4.1 V21 構築 (7/15-9/2)
- Session #44 G の Phase 4 plan
- JRA-VAN RV 動画 + YOLOv8 + DLC SuperAnimal
- 5 features: stride / gait / head_bobbing / ear_pos / posture
- V20 + 動画 features → V21 学習

### 4.2 V21 投入 (9/2)
- 採用条件: WF AUC >= V20 + 0.005、 LIVE retro winner_top1 >= V20 + 1pt

---

## 5. ★ Sprint 9 (10/1-11/1) 詳細 — 30 年 backtest 着手 ★

### 5.1 Week 1 (10/1-10/7): pipeline 実装
- `tools/build_30y_raw_parquet.py` 実装
- `tools/build_30y_features.py` 実装
- TFJV 30 年抽出 → raw parquet
- features parquet 生成 (50 GB)

### 5.2 Week 2 (10/8-10/14): model 学習 (5 戦略 × 5 fold)
- multiprocessing 並列学習
- 30-55 時間 想定、 1 週間以内 完走
- LGB+XGB+FT+IR × 5 戦略 = 25 学習 jobs

### 5.3 Week 3 (10/15-10/21): backtest + monte carlo
- walk-forward backtest 5 fold
- monte carlo simulation (1000 回)
- bootstrap CI95 算出
- 5 戦略 × 5 fold = 25 result JSON

### 5.4 Week 4 (10/22-10/28): report 生成
- `docs/BACKTEST_30Y_REPORT.md` 生成
- plot 作成 (累計 P/L / AUC trend / DD curve)
- 6 件閾値 PASS/FAIL 判定
- robustness 結論

### 5.5 Week 5 (10/29-11/1): V22 RL PoC + 採用判定
- V22 RL agent PoC (Session #83 連動)
- 30 年 backtest 結果 で V20 / hybrid / V22 RL の優劣 確定
- 採用判定 sheet 完成

---

## 6. Sprint 10 (V22 投入) 詳細

### 6.1 V22 paper trade (11/1-11/30)
- 4 weekend × 4 R = ~16 day paper
- RL agent の動的最適化 効果確認

### 6.2 V22 投入 (12/1+)
- 採用条件: 30 年 backtest 6 件 ALL PASS + paper ROI >= 130%

---

## 7. 制約 + risk

### 7.1 V20 + V21 + V22 の並行 risk
- 同時に 3 model 構築は 工数過多
- → V20 (5/22-6/8) → V21 (7/15-9/2) → V22 (10/1-11/1) の sequential 厳守

### 7.2 30 年 backtest 学習時間
- 30-55 h (single CPU)、 multiprocessing で 8-15 h 候補
- Sprint 9 Week 2 の 1 週間で 完走見込

### 7.3 storage 100 GB
- 現在の D drive 余裕確認必要
- TFJV 6 GB は維持、 backtest 専用 dir 新規

### 7.4 V15 production 完全不変
- Sprint 9 中も V15 案B改 strict 運用継続
- backtest 環境構築は dev branch / 別 dir で実施

---

## 8. 投資保護 (絶対遵守)

- **5/9 V15 案B改 strict 単独継続** (Session #38 NO-GO 確定後の唯一 path)
- **撤退ライン**: 累計 -¥50,000 (現在 **+¥5,240**) ※ 旧 +¥13,530 は drift、 5/16 P0-1 真値 (docs/ROI_DISCREPANCY_2026_05_16.md)
- **30 年 backtest 着手中も V15 production 完全不変保証**

---

## 9. 関連 doc

- `docs/BACKTEST_30_YEAR_DESIGN.md` — 範囲設計
- `docs/BACKTEST_DATA_PIPELINE.md` — pipeline 設計
- `docs/BACKTEST_STRATEGY_COMPARISON.md` — 戦略比較
- `docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md` (Session #44 F)
- `docs/STRATEGY_HYBRID_DESIGN.md` (Session #82)

---

**結論**: 30 年 backtest 環境構築は Sprint 9 (10/1-11/1) で着手。 4 week pipeline 実装 → model 学習 → backtest → report の 順序。 V20 / V21 / V22 の並行構築は避けて sequential 厳守。 V15 production は 完全不変保証。
