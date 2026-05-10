# Phase 21F: 5/15 (木) 夜 実装 checklist (V22 RL reward v1)

date: 2026-05-11 (Phase 21F 設計時)
target: 2026-05-15 21:00-23:00 実装 + 100K test
依存: data/v22/phase21f_v22_rl_reward_design.md

---

## 1. 事前準備 (5/15 21:00 開始前)

- [ ] git pull (最新 main 取得)
- [ ] models/v22/v22_rl_ppo_phase17_initial.zip 存在確認 (v0 backup)
- [ ] python -c "import stable_baselines3, gymnasium, torch; print(torch.cuda.is_available())" 確認
- [ ] data/daily_predictions / daily_results 最新分 取得済 (5/10 + 5/17 投票なし含)

---

## 2. 実装手順 (21:00-22:00、 60 分)

### 2-1. tools/v22_rl_agent.py 改修 (30 分)

```python
# import 追加
import numpy as np

# === reward function v1 (新規追加) ===
def compute_reward_v1(state_dict, action, race, daily_cumulative_bet) -> float:
    BET_AMOUNTS = [0, 100, 300, 700]
    bet = BET_AMOUNTS[int(action)]
    score = state_dict['top1_score']
    
    # 1. base profit + payout 倍率 weighted
    if bet == 0:
        base = 0.0; payout_bonus = 0.0
    elif race.trio_hit:
        ratio = bet / 700.0
        scaled_pay = race.payout * ratio
        base = (scaled_pay - bet) / 700.0
        multiplier = race.payout / 700.0
        payout_bonus = 0.5 * (np.sqrt(multiplier) - 1.0) if multiplier > 1.0 else 0.0
    else:
        base = -bet / 700.0; payout_bonus = 0.0
    
    # 2. 戦略⑦ alignment
    s7p = 0.0
    if bet > 0:
        if state_dict['is_grade']: s7p -= 0.5
        if state_dict['is_special']: s7p -= 0.3
        if state_dict['is_kyoto']: s7p -= 0.2
    
    # 3. budget 上限
    bp = -1.0 if (bet > 0 and daily_cumulative_bet + bet > 2100) else 0.0
    
    # 4. skip rational
    sr = 0.0
    if bet == 0:
        if score < 0.5: sr = 0.05
        elif state_dict['is_grade'] or state_dict['is_special']: sr = 0.03
    
    # 5. 1番人気過信
    op = -0.2 if (bet > 0 and score >= 0.8 and state_dict['is_grade'] and not race.trio_hit) else 0.0
    
    # 6. 中位 score 微小 bonus
    mb = 0.0
    if (0.6 <= score < 0.7 and bet > 0 and race.trio_hit
        and not state_dict['is_grade']
        and not state_dict['is_special']
        and not state_dict['is_kyoto']):
        mb = 0.1
    
    return float(base + payout_bonus + s7p + bp + sr + op + mb)


# === KeibaVoteEnv 改修 ===
class KeibaVoteEnv(gym.Env):
    def __init__(self, races_by_date, dates):
        super().__init__()
        ...
        # state 8-dim → 9-dim (daily_cumulative_bet/2100 追加)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.bet_amounts = [0, 100, 300, 700]
        self.daily_cumulative_bet = 0
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_date = self.np_random.choice(self.dates)
        self.current_races = self.races_by_date[self.current_date]
        self.current_idx = 0
        self.daily_cumulative_bet = 0
        return self._make_obs(0), {}
    
    def _make_obs(self, idx):
        if idx >= len(self.current_races):
            return np.zeros(9, dtype=np.float32)
        base = _race_to_state(self.current_races[idx])
        return np.concatenate([base, [self.daily_cumulative_bet / 2100.0]]).astype(np.float32)
    
    def step(self, action):
        race = self.current_races[self.current_idx]
        bet = self.bet_amounts[int(action)]
        state_dict = {
            'top1_score': race.morning_top1_score,
            'is_grade': any(g in race.race_name for g in ['G1','G2','G3','GⅠ','GⅡ','GⅢ']),
            'is_special': '特別' in race.race_name,
            'is_kyoto': race.course == '京都',
        }
        reward = compute_reward_v1(state_dict, action, race, self.daily_cumulative_bet)
        self.daily_cumulative_bet += bet
        self.current_idx += 1
        terminated = self.current_idx >= len(self.current_races)
        obs = self._make_obs(self.current_idx) if not terminated else np.zeros(9, dtype=np.float32)
        return obs, float(reward), terminated, False, {}
```

### 2-2. CLI argument 追加 (5 分)

```python
p.add_argument('--reward-version', type=str, default='v1')
p.add_argument('--lr', type=float, default=3e-4)
```

### 2-3. unit test (15 分)

```python
# 代表 case 8 件 (data/v22/phase21f_v22_rl_reward_design.md table 参照)
def test_reward_v1():
    from tools.v22_rl_agent import compute_reward_v1
    # case 1: 案 B 改 hit (上位)
    state = {'top1_score': 0.72, 'is_grade': False, 'is_special': False, 'is_kyoto': False}
    race = type('R', (), {'trio_hit': True, 'payout': 2810, 'race_name': '東京R8'})()
    r = compute_reward_v1(state, 3, race, 0)
    assert 3.0 <= r <= 4.0, f"expected ~3.5, got {r}"
    
    # case 2: graded で bet (NG)
    state = {'top1_score': 0.81, 'is_grade': True, 'is_special': False, 'is_kyoto': False}
    race = type('R', (), {'trio_hit': False, 'payout': 0, 'race_name': '京都R9 烏丸S'})()
    r = compute_reward_v1(state, 3, race, 0)
    assert -2.0 <= r <= -1.5
    
    # case 3: 低 score skip
    state = {'top1_score': 0.45, 'is_grade': False, 'is_special': False, 'is_kyoto': False}
    race = type('R', (), {'trio_hit': False, 'payout': 0, 'race_name': '東京R3'})()
    r = compute_reward_v1(state, 0, race, 0)
    assert 0.04 <= r <= 0.06
    
    # case 4: budget 超過
    state = {'top1_score': 0.72, 'is_grade': False, 'is_special': False, 'is_kyoto': False}
    race = type('R', (), {'trio_hit': False, 'payout': 0, 'race_name': '東京R12'})()
    r = compute_reward_v1(state, 3, race, 2100)  # 既に上限
    assert -2.5 <= r <= -1.5
```

---

## 3. 学習 + 評価 (22:00-23:00、 60 分)

### 3-1. 100K 学習 (30-40 分、 GPU 使用)
```bash
cd C:\Users\takum\keiba-ai
python tools/v22_rl_agent.py --steps 100000 --reward-version v1 --seed 42
```

### 3-2. action distribution 確認 (10 分)

期待 (success criteria):
- skip: 50-70% (v0 100% から減)
- bet 100/300/700: 30-50% (戦略⑦ pass + 高 score race で 集中)
- graded で bet: < 10%
- 京都で bet: < 20%
- 中位 score (0.6-0.7) で bet: 10-30%
- ¥2,100/日 上限超過: < 10%

### 3-3. 結果 doc + commit (10 分)
- data/v22/phase21f_100k_results.md (ROI / hit / action dist 集計)
- git add + commit + push

---

## 4. 失敗 case 対応

| 症状 | 原因候補 | 対応 |
|------|---------|------|
| 全 skip 継続 (skip 100%) | reward shaping 弱 / skip_reward 過大 | skip_reward 0.05 → 0.02 / payout_bonus 強化 |
| graded で bet 多発 | strategy_7_penalty 弱 | -0.5 → -1.0 |
| budget 超過 多発 | budget_penalty 弱 | -1.0 → -3.0 |
| 中位 score over-bet | midband_bonus 過大 | 0.1 → 0.05 / 削除 |
| 学習 unstable (loss explode) | lr / clip_range 不適 | lr 3e-4 → 1e-4 |

---

## 5. ★ V15 production 完全保護 ★ (絶対)

- [ ] 5/15 実装中 V15 model file 不変
- [ ] tools/predict_core.py / daily_predict.py 不変
- [ ] schtask 不変
- [ ] 5/16 (金) 通常運用 案 B 改 strict 維持
- [ ] 5/17 (土) 投票 案 B 改 strict (V22 は paper のみ)
- [ ] 累計 +¥13,420 維持

---

## 6. 完了通知

実装 + 100K 学習 完了時:
```bash
python tools/notify_done.py "Phase 21F V22 RL v1 実装完了" "100K 学習 + action dist 検証 OK、 5/16 500K へ"
```
