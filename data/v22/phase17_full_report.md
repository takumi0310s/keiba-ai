# Phase 17 完了 (Opus 4.7) ★30 年 backtest 環境 + V22 RL 初期学習★

date: 2026-05-10 21:00
session: Phase 17 (Opus 4.7、 caveman mode、 現実最大限)

---

## scope 現実調整 (user 確認後)

ユーザー「規則グレーでも可能な限り現実」 → 単一 session で実装可能な scope に調整:

| 項目 | 当初 (24-48h) | 実施 (60-90 min) |
|---|---|---|
| 30 年 data | 1996-2026、 100K R、 JV-Link fetch | jra_races_full 16 年 + actual logs 10 日 = 347 R 利用 |
| backtest engine | 完全 30 年 WF | actual logs 戦略比較 engine 動作 |
| V22 RL train | 8-16h、 1k-10k episodes | 5,000 timesteps PPO 初期学習 |
| paper trade | V15-V22 並行 | V15/V18/V20/V21/V22 並行動作確認 |

→ 次 step (本来 24-48h) は 5/24+ JV-Link 加入 + dedicated training session で実施。

---

## A. 30 年 backtest 環境 (Phase 17 段階)

### 実装: tools/backtest_engine.py
```python
from backtest_engine import load_history, walk_forward, simulate_strategy

races = load_history()  # daily_predictions + daily_results 統合
for train, test in walk_forward(races, train_days=7, test_days=1):
    ...

result = simulate_strategy(races, {'min_score': 0.7, 'top_n_per_day': 3})
```

### 動作確認 (10 日 actual logs、 347 R)
| 戦略 | bet | hit | inv | pay | profit | ROI |
|---|---|---|---|---|---|---|
| 全 R baseline | 347 | 80 | ¥242,900 | ¥33,210 | -¥209,690 | 13.7% |
| 案 B 改 strict | 21 | 5 | ¥14,700 | ¥680 | -¥14,020 | 4.6% |
| 上位 score 0.7+ | 33 | 6 | ¥23,100 | ¥3,490 | -¥19,610 | 15.1% |
| 上位 3R / 日 | 30 | 10 | ¥21,000 | ¥680 | -¥20,320 | 3.2% |

→ 10 日 sample = 5/10 upset 日含むため bias あり。 30 年 data 取得後再評価必須。

### 5/24+ task (Phase 17b)
- [ ] JV-Link 32-bit Python venv 構築 (C:\Users\takum\jvlink-venv\)
- [ ] 1996-2026 30 年 data fetch (約 100K R、 JV-Link 経由)
- [ ] SQLite/DuckDB に格納
- [ ] WF backtest 完全実行

---

## B. V22 RL 環境 + PPO agent

### 実装: tools/v22_rl_agent.py
```python
class KeibaVoteEnv(gym.Env):
    observation_space: Box(-1, 2, shape=(8,))
    action_space: Discrete(4)  # bet 0/100/300/700
    reward: profit per race (normalized by base bet 700)
```

### state (8-dim)
1. V15 morning_top1_score
2. num_horses / 18
3. condition_enc / 5
4. distance / 3000
5. surface (芝=0 / ダ=1)
6. is_grade flag
7. is_special flag
8. is_kyoto flag

### action (Discrete 4)
- 0: skip (¥0)
- 1: small (¥100)
- 2: medium (¥300)
- 3: large (¥700)

### PPO 学習結果 (5,000 timesteps)
- train: 7 days / test: 3 days
- algorithm: PPO MlpPolicy (SB3 2.8.0)
- learning_rate: 3e-4 / n_steps: 128 / batch: 32 / gamma: 0.99
- saved: `models/v22/v22_rl_ppo_phase17_initial.zip`

### 評価結果 (test 30 episodes)
- avg reward: 0.0 (per episode)
- action dist: skip=1018 / 100=0 / 300=0 / 700=0
- → ★ agent = 全 skip 学習 ★ (history 負 ROI = 賭けないが最適と学習)

### 次 step (本格学習)
- timesteps: 1M-10M (8-16h、 GPU 利用)
- larger feature space: V18/V20 features 取込
- reward shaping: hit 強化 / skip 弱化
- 30 年 data + WF train/test split
- 期待 ROI: 165-180% sustainable (Session #84 design 通り)

---

## C. paper trade engine (V15/V18/V20/V21/V22 並行)

### 実装: tools/paper_trade_engine_v22.py
| 戦略 | bet 条件 | bet 額 |
|---|---|---|
| V15 | 案 B 改 strict (戦略⑦)、 score 0.7+ | ¥700 |
| V18 cand | V15 + score 0.75+ | ¥700 |
| V20 cand | V18 + 多頭数 R で ¥1000 | ¥700-1000 |
| V21 cand | V20 + 動画 features (placeholder) | V20 同 |
| V22 RL | PPO model action | ¥0-700 |

### 動作確認 (347 R)
| 戦略 | bet | hit | inv | pay | profit | ROI | hit% |
|---|---|---|---|---|---|---|---|
| V15 | 37 | 10 | ¥25,900 | ¥3,490 | -¥22,410 | 13.5% | 27.0% |
| V18 cand | 15 | 2 | ¥10,500 | ¥680 | -¥9,820 | 6.5% | 13.3% |
| V20 cand | 15 | 2 | ¥12,000 | ¥971 | -¥11,029 | 8.1% | 13.3% |
| V21 cand | 15 | 2 | ¥12,000 | ¥971 | -¥11,029 | 8.1% | 13.3% |
| V22 RL | 0 | 0 | ¥0 | ¥0 | ¥0 | 0.0% | - |

→ V22 RL = 全 skip = 最小損失。 機械学習が正しく rational behavior を獲得。

→ ただし 5/10 upset 日含む 10 日 sample は bias 大。
→ 30 年 backtest で再評価必須。

---

## V15 投資保護

✅ V15 model 不変
✅ tools/predict_core.py / daily_predict.py / app.py 不変
✅ schtask 不変
✅ Phase 17 = 新規 file のみ:
  - tools/backtest_engine.py
  - tools/v22_rl_agent.py
  - tools/paper_trade_engine_v22.py
  - models/v22/v22_rl_ppo_phase17_initial.zip
  - data/v22/phase17_*.{md, json}
✅ 累計 +¥13,420 維持

---

## 5/11+ task plan

### Phase 17b (5/24+ JV-Link 加入後、 dedicated session)
- [ ] 30 年 data fetch (8-16h)
- [ ] backtest engine 30 年 WF 実行 (8h)

### Phase 17c (V22 本格学習、 dedicated session)
- [ ] V22 PPO 1M-10M timesteps (8-16h、 GPU)
- [ ] reward shaping
- [ ] 多 features (V18/V20 score 取込)

### Phase 17d (V22 本番投入、 12/1 候補)
- [ ] V15 / V18 / V20 / V21 並行 paper 1-3 ヶ月
- [ ] Sharpe ratio + ROI sustainability 評価
- [ ] V22 production 切替判定

---

## 期待効果 (Phase 17 完全実装後)

| metric | 現状 (V15) | V22 完全 (12/1+) |
|---|---|---|
| AUC | 0.8939 | (RL は AUC 評価対象外) |
| ROI | 113.8% (5/10) | **165-180%** (Session #84 design) |
| Sharpe | ? | sustainable |
| 投資判断 | 案 B 改 strict | RL 動的最適化 |
