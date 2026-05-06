# 5/9 本番直前 精度改善余地 検出 (research / 2026-05-06 PM)

**前提**: 5/9 案B改 V15 単独 維持 (累計 +14,140 → +13,530円死守)
**最新 commit**: c722d403
**重要**: 改善は 5/9 戦略の安定性を損なってはいけない

---

## 1. 5/9 で 即適用可能な改善

### 1.1 multi_stage_predict 3段運用 (緊急度 🟡 / 既に実装済)

| 項目 | 内容 |
|------|------|
| 改善内容 | `tools/multi_stage_predict.py` の 3 stage (10:00 test / 14:50 / 15:45) を 5/9 本番投入 |
| 工数 | 0h (実装+5/3 dry-run 完了、commit f408d93d 系で着地済) |
| 期待効果 | 馬体重補正で TOP1 score 朝→現で **+0.21〜+0.31** 上昇 ※朝予測 default 480kg vs 実値 |
| リスク | 🟢 LOW — 朝予測 fallback あり、predict_one_race 例外時も morning_only で進行 |
| 5/9 適用判定 | **GO** (既に admin guide / test 完了、Discord 通知 format 確認済) |

→ V15 model は **馬体重を相当重視**。3 stage 化で買い目決定 stage (15:45) の予測信頼度が最大化される。
→ stage 統合ではなく **並行運用** (10:00 = 観察、14:50 = 11R 一括、15:45 = 12R 主戦場) が最適、stage ごとに R を区切ることで衝突なし。

### 1.2 当日確定オッズ (odds_log) の Pattern B 活用 (緊急度 🟡 / 既存)

| 項目 | 内容 |
|------|------|
| 改善内容 | 15:45 stage で `fetch_realtime_odds_full()` (predict_core L1135) によりリアルタイム単勝オッズを取得、`odds_log` / `prev_odds_log` を更新 (L1827-1831) |
| 工数 | 0h (既存実装) |
| 期待効果 | Pattern B 8 特徴量 (odds_log, pop_rank, weight_*, condition_enc, cond_surface) を完全活用 |
| リスク | 🟢 LOW — 取得失敗時は前走オッズで fallback |
| 5/9 適用判定 | **GO** (既に `predict_one_race` で取得経路確立) |

### 1.3 馬場・天候情報の Pattern B 活用 (緊急度 🟡 / 既存)

| 項目 | 内容 |
|------|------|
| 改善内容 | `fetch_jra_and_weather()` (L1385) でクッション値・含水率・気温・湿度・風速・降水量・天候・condition_enc を取得 |
| 工数 | 0h (既存) |
| 期待効果 | Pattern B 132 features full 活用、馬場急変日 (重・不良) で予測精度向上 |
| リスク | 🟢 LOW — 取得失敗時は欠損 (0) 扱い、V15 は欠損耐性あり |
| 5/9 適用判定 | **GO** (multi_stage_predict 内で既に呼ばれる) |

### 1.4 騎手変更検出 (緊急度 🟡 / 既存)

| 項目 | 内容 |
|------|------|
| 改善内容 | `jockey_change` / `jockey_change_to_top` (L2074-2091) で騎手変更を pre-race に検出 |
| 工数 | 0h (既存) |
| 期待効果 | リーディング上位騎手への乗替で score 補正 |
| リスク | 🟢 LOW |
| 5/9 適用判定 | **GO** |

### 1.5 V15.1 SKB の 5/9 投入 — **🔴 不採用** (緊急度 🟠 検討したが)

| 項目 | 内容 |
|------|------|
| 検討内容 | V15 予測 + V15.1 SKB score の重み付け平均 (`alpha * v15 + (1-alpha) * v15_1`) |
| 期待効果 | retro AUC +0.0694 → 軸 top3 率 改善余地 (ただし AUC ≠ 軸 top3 率) |
| リスク | 🔴 **HIGH** — 以下の理由で 5/9 戦略の安定性を損なう |
| 5/9 適用判定 | **NO-GO**、5/16 paper trading に回す |

**不採用理由** (`data/v18/v15_1_evaluation.md` §5.1 と整合):

1. V15.1 は LGB single quick mode (LR=0.1) の retro、4-model ensemble (LGB+XGB+FT+IR) との互換性 未確認
2. `predict_core.py` への統合工数 (>=2h) — 5/9 当日まで安全テスト時間不足
3. JRDB SKB の 5/9 朝 06:00 DailyJrdbKyi で SKB 取得確実性 未検証 (5/2-5/3 retro で 50-90% は OK だが 5/9 当日確認必要)
4. 軸 top3 率 (TOP1 精度) 改善は AUC 改善と必ずしも一致しない、retro WF 軸 top3 検証なし
5. **取り返し禁止 / +13,530円死守** ルールに反する追加 risk

→ 5/9 投入なら緊急度 🟠 の評価だが、現時点では **fall-back 確保困難** で **NO-GO**。

---

## 2. 5/16 までに準備すべき改善

### 2.1 race-level normalize の本番統合 (V18/V19 試行 前提条件 #1)

- **状態**: ⏳ 未対応 (`data/v18/race_normalize_5_4_result.md` で softmax T=1.0 推奨済、実装ツール `tools/race_normalize.py` あり)
- **工数**: 30分 (predict_core.py / race_auto_notify.py への softmax T=1.0 組込)
- **期限**: 5/15 まで
- **目的**: V18/V19 の bet=0 問題解決 (BT/retro race_max_p 27.7x shift を緩和)
- **5/9 影響**: **なし** (V15 単独維持、V18/V19 paper のみ用)

### 2.2 V15.1 paper trading 並列運用 (Phase 3 前倒し検討)

- **5/16-5/24 で並列実行**:
  - V15 案B改 (本番、実投資)
  - V15.1 SKB-only (paper、Discord 通知のみ)
  - sample 30+ bets 蓄積、軸 top3 率比較
- **6 月前半 GO/no-go 判定**

### 2.3 V18/V19 paper retro の sample 蓄積 (前提条件 #2/#3)

- **状態**: 5/2-5/3 で 9 bets, 67 races
- **目標**: 5/15 までに 30+ bets, ROI > 120%
- **判定**: 5/15 夜判断ミーティング、達成時のみ 5/16 v18 単勝 1,000円/日 試行

### 2.4 winner_top1 rate gap 調査 (前提条件 #4)

- **現状**: BT 47.8% vs retro 34.5% (-13.3pt)
- **判定**: 5/16 までに ≥40% (5/2-5/3 比 +5pt) 必要
- **着手**: feature distribution shift 調査と並列

### 2.5 feature distribution shift 調査 (前提条件 #5)

- **状態**: ❌ 未着手
- **工数**: 60分
- **目的**: BT vs production の feature 値差分検出 → winner_top1 13pt 劣化の根本原因特定

---

## 3. Phase 3 (5/24+) 改善ロードマップ

| Step | 工数 | 内容 |
|------|------|------|
| 1 | 4h | SKB のみ追加で V15.1a 4-model ensemble 学習 (LGB+XGB+FT+IR) |
| 2 | 2h | 軸top3 率 retro WF 検証 (2020-2025) |
| 3 | 1h | predict_core.py に SKB merge 統合 (race_id 変換器 `netkeiba_to_jrdb_internal()` 既確立、99.8% match) |
| 4 | 1h | DailyJrdbKyi で SKB 取得確実化 |
| 5 | 2h | Pattern A + Pattern B 両方 学習 |
| **合計** | **10h** | Phase 3 中盤 (6 月前半)、5/末 V15→V15.1 切替判定 |

### 並列タスク (V20 統合)

- v18/v19 + V15 アンサンブル統合 (累計 sample 100+ bets で統計判定)
- EV 精緻化: 三連複 7点 各組合せの実勝率 → 控除率 25% 超 edge 検出 (現状 race_auto_notify.py に EV 明示計算なし、V20 で導入)
- 京都 ROI 80%+ 目標 (`course_renovated` 永久化 4/27 適用済の効果検証)

---

## 4. 結論

### 5/9 即適用可能な改善 (4 件、全て既存実装)

1. **multi_stage_predict 3段運用** — 馬体重補正で TOP1 score +0.2〜0.3 ↑、5/3 dry-run ALL PASS
2. **当日確定オッズ Pattern B 活用** — `fetch_realtime_odds_full` 既動作
3. **馬場・天候 Pattern B 活用** — `fetch_jra_and_weather` 既動作
4. **騎手変更検出** — `jockey_change` 既動作

→ **追加実装 0 件、全て既存 commit (f408d93d 系) で 5/9 自動運用に組込済**。

### 5/9 不採用 (リスク高)

- V15.1 SKB の本番組込 → 5/16 paper trading に回す (AUC +0.0694 ですが軸 top3 率 unknown、4-model 互換性未検証、取り返し禁止ルールに反する)
- formation 拡張 (V15+T1-T4-T5 等) → 既 retro で 7 点 baseline が ROI 最良と判明 (`improvements_prototyped_5_3.md`)

### 5/9 戦略

- **V15 案B改 単独 維持** (12R 1勝クラスのみ、上限 2,100円/日 700×3R)
- 戦略⑦ filter (06_特別 / 京都 / 条件E / 条件B 除外) 適用済 (commit 4/27)
- 撤退ライン: 単日 ROI <50% → 5/10 停止 / 累計 -50,000円 → 完全撤退

### V18/V19 試行 (5/16) 暫定評価

| # | 条件 | 5/6 時点状態 | 達成度 |
|---|------|-------------|--------|
| 1 | normalize 本番統合 | ⏳ 未着手 | ✗ |
| 2 | 5/2-5/15 paper retro ROI >120% | ⏳ 5/2-5/3 のみ (9 bets) | △ |
| 3 | sample 30+ bets | 9 bets | ✗ |
| 4 | winner_top1 ≥ 40% | 34.5% | ✗ |
| 5 | feature shift 調査 | ❌ 未着手 | ✗ |

→ **5/16 GO 暫定評価: NO-GO** (5 条件中 0 達成、5/15 までに #1 #5 完了させても #3 #4 は実 race 蓄積必須で間に合わない可能性高)。

### 改善は 5/16 以降に段階投入推奨

5/9-5/10 は V15 案B改 単独維持で **+13,530円 死守** 最優先。
改善 (V15.1 SKB / V18/V19 / EV 精緻化) は 5/16 以降に paper trading で慎重段階導入、Phase 3 (5/24+) で本格採用。

---

## 関連ファイル

| path | 内容 |
|------|------|
| `data/v18/v15_1_evaluation.md` | V15.1 SKB +0.0699 の retro 評価 |
| `data/v18/v18_v19_integration_plan_5_4_pm.md` | 5/16 V18/V19 試行 5 条件 |
| `data/v18/multi_stage_predict_test_5_6.md` | 3 stage 5/3 dry-run 結果 |
| `data/v18/morning_horse_weight_check_design.md` | 馬体重補正 設計 |
| `data/v18/race_normalize_5_4_result.md` | softmax T=1.0 推奨 |
| `data/v18/distribution_shift_analysis.md` | BT vs retro の race_max_p 27.7x shift |
| `data/v18/improvements_prototyped_5_3.md` | formation 拡張 retro (7点 baseline 最良) |
| `data/v18/risk_management_5_9.md` | 撤退ライン |
| `tools/predict_core.py` | fetch_realtime_odds_full / fetch_jra_and_weather / jockey_change |
| `tools/multi_stage_predict.py` | 3 stage 並行運用 |
| `train/v15_1_features.py` | V15.1 SKB 統合 module + race_id 変換器 |
