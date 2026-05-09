# V22 RL 投票最適化 設計

**作成**: Session #83 (2026-05-09)
**Phase**: 5 (10-12 月、 最終 phase)
**目標**: 強化学習で 投票金額・買い目を **完全 AI 最適化**、 ROI 150-180% 候補

---

## 1. 概要

V20 / V21 の **予測 (probability)** と V22 RL の **投票 strategy** を分離。
RL agent が state → action を学習し、 累計収支を最大化する policy を獲得する。

| 役割 | model |
|------|-------|
| **予測** | V20 4-model ensemble (AUC 0.90025) + V21 動画統合 (期待 0.92) |
| **投票最適化** | V22 RL (PPO base) — **本 doc の対象** |

★ 既存の案B改 strict / hybrid (Session #82) を AI 学習 data として活用 ★

---

## 2. MDP (Markov Decision Process) 定義

### 2-1. State (状態空間)

| 種別 | 内容 | dim |
|------|------|-----|
| **score 系** | V20/V21 score (各 R 全頭、 max 18 頭) | 18 |
| | 上位 3 頭 score 差分 (top1-top2、 top1-top3) | 2 |
| | レース内 score variance | 1 |
| **オッズ系** | 単勝 / 複勝 / 馬連 / 三連複 オッズ (現在値) | 4×N |
| | オッズ flow (前 5 分 / 10 分 diff) | 2×N |
| | 上位 人気 vs score 乖離 | 1 |
| **bankroll 系** | 累計収支 (current bankroll) | 1 |
| | 当日 累計 投資 / 払戻 | 2 |
| | 直近 7 日 ROI | 1 |
| | drawdown 現在値 | 1 |
| **race meta** | クラス (A-X) | 6 (one-hot) |
| | 距離 / 馬場 / 頭数 | 3 |
| | 京都 flag / 06_特別 flag | 2 |
| **horse meta** | 当日体重 (Session #48 B) | N |
| | パドック features (Phase 4 V21) | N×M |
| | KKA features (Session #53) | 12-15 |

合計 state dim: **約 100-150** (R によって可変)

### 2-2. Action (行動空間)

#### 離散 action

| action | 内容 |
|--------|------|
| 0 | skip (投票しない) |
| 1 | 単勝 (1 着固定) |
| 2 | 複勝 (3 着以内) |
| 3 | 馬連 (2 頭) |
| 4 | 馬単 (順序付き 2 頭) |
| 5 | 三連複 (3 頭、 順不同) |
| 6 | 三連単 (3 頭、 順序付き) — 高難度・高配当 |
| 7 | wide (3 着以内 2 頭) |

#### 連続 action

| action | range |
|--------|-------|
| 投票金額 | [¥0, ¥10,000] (Eighth Kelly base、 cap ¥10,000) |
| 買い目組み合わせ | top-K 軸選択 (K=1-5)、 紐 horse 選択 (max 5) |

★ PPO が連続 + 離散 混在に対応するため hybrid action space を採用 ★

### 2-3. Reward (報酬)

| 種別 | 計算 | weight |
|------|------|--------|
| **短期 reward** | 払戻金 - 投資金額 (R ごと) | 1.0 |
| **長期 reward** | 累計利益 (1 開催終了時) | 0.3 |
| **Sharpe ratio** | mean(daily_return) / std(daily_return) | 0.2 |
| **drawdown penalty** | -1.0 × (max_drawdown - 30%)+ | 強い負 reward |
| **投票しない penalty** | 0 (skip も valid action) | — |

総合 reward: `r = 1.0 * profit + 0.3 * cum_profit + 0.2 * sharpe - drawdown_penalty`

---

## 3. 候補 algorithm

| algorithm | 特性 | V22 適合 |
|-----------|------|---------|
| PPO (Proximal Policy Optimization) | on-policy、 stable、 連続+離散 混在 OK | ★ **推奨** ★ |
| SAC (Soft Actor-Critic) | off-policy、 sample 効率 高、 連続 action 専用 | 次点 |
| DQN (Deep Q-Network) | off-policy、 離散 action 専用 | 投票金額 連続化で不適 |
| A2C (Advantage Actor-Critic) | on-policy、 PPO の前身 | PPO で代替可 |

### PPO 推奨理由

1. 投票金額 (連続) + 馬券種 (離散) 混在 action space に対応
2. 学習 stability が高く、 over-fitting 抑制機構 (clip ratio) あり
3. stable-baselines3 で 実装容易
4. 30 年 data 規模に sample efficiency 十分

---

## 4. 学習 data

| 項目 | 値 |
|------|----|
| 期間 | 1996-2025 (30 年、 Session #84 backtest 環境連携) |
| source | TFJV 90 年分 のうち 30 年抽出 |
| race 数 | 約 20 万 R |
| horse-runs | 約 200 万件 |
| split | train: 1996-2020 / val: 2021-2023 / test: 2024-2025 |
| environment | gym-style env (1 R = 1 episode、 OR 1 開催 = 1 episode) |

### env 設計 2 案

| 案 | 1 episode | pros | cons |
|---|----------|------|------|
| **A** R 単位 | 1 R = 1 episode | sample 効率 高 | 長期収支 学習困難 |
| **B** 開催単位 | 1 day = 1 episode (12-36 R) | 累計収支 学習 OK | sample 効率 低 |

★ 推奨: **B 案 (開催単位)**、 累計収支 が真の目的 ★

---

## 5. 期待効果

| metric | V15 (現状) | V20 + 案B改 | V22 RL 目標 |
|--------|-----------|-------------|------------|
| ROI | 119.2% (戦略⑦込み 140%) | 130-145% | **150-180%** |
| 投票 R 数 / 日 | 3 R 固定 | 3 R 固定 | **動的 (1-10 R)** |
| max drawdown | 11.1% (MC 平均) | 同程度 | **8% 以下 制御** |
| Sharpe ratio | (測定要) | (測定要) | **1.5+ 目標** |
| 撤退判断 | 手動 | 手動 | **自動** |
| メンタル要素 | あり | あり | **完全排除** |

---

## 6. 投入 timing

| 日付 | task |
|------|------|
| 2026-10-01 | V22 PoC 開始 (gym env 構築 + PPO baseline) |
| 2026-10-15 | 30 年 backtest 環境 統合 (Session #84) |
| 2026-11-01 | paper trade 開始 (V20/V21 score + RL action) |
| 2026-11-30 | paper 30 日 評価 |
| **2026-12-01** | **V22 投入候補 判定** |

---

## 7. 投入条件 (GO 5 項目)

| # | 条件 | 閾値 |
|---|------|------|
| 1 | walk-forward backtest ROI | ≥ 150% (test 期間) |
| 2 | Sharpe ratio | ≥ 1.5 |
| 3 | max drawdown | ≤ 15% |
| 4 | paper 30 日 ROI | ≥ V20 + 案B改 + 5pt |
| 5 | risk audit (Section 後述 doc) | PASS |

---

## 8. V22 と既存 strategy の関係

V22 RL は **既存 strategy を完全置換しない**。

| layer | 役割 |
|-------|------|
| layer 1: V20/V21 prediction | 馬の能力 score 算出 (変更なし) |
| layer 2: 案B改 strict / hybrid | safety net (V22 異常時の fallback) |
| layer 3: V22 RL | 完全最適化 layer (12/1+) |

★ V22 NO-GO でも V20 + 案B改 で運用継続 ★

---

## 9. 投資保護 (絶対遵守)

- 5/9-12/1 期間中も V15 production **完全不変**
- V22 PoC は **10 月以降**、 それまで V15 / V20 / V21 paper のみ
- 撤退ライン: 累計 -¥50,000 (現状 +¥12,830、 余裕 +¥62,830)

---

## 10. 関連

- [V22_RL_INFRA.md](V22_RL_INFRA.md) — GPU + library + 学習時間
- [RL_VS_STRATEGY_COMPARISON.md](RL_VS_STRATEGY_COMPARISON.md) — paper 比較
- [V22_RISK_ANALYSIS.md](V22_RISK_ANALYSIS.md) — リスク 4 種 + 撤退 logic
- [V20_BUILD_DETAILED_PLAN.md](V20_BUILD_DETAILED_PLAN.md) — V20 base
- [PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md](PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md) — Phase 5 全体
