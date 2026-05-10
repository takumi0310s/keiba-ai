# Phase 21F: V22 RL reward function 設計改善 + 5/15-5/16 学習 plan

date: 2026-05-11
session: Phase 21F (Opus 4.7、 caveman mode、 design only / 完全 read-only)
依存: Phase 17 (V22 RL agent v0、 5K timesteps、 全 skip 学習) + Phase 21C (5/10 score 帯別 重大訂正)

---

## 0. 背景

### Phase 17 V22 RL v0 (5K timesteps、 5/10 学習)
- agent: PPO MlpPolicy、 SB3 2.8.0
- state: 8-dim (top1 score / num_horses / cond / dist / surface / grade / special / kyoto)
- action: Discrete(4) = bet ¥0/100/300/700
- reward: profit normalized by ¥700 base
- 結果: ★ test 30 ep 全 skip ★ (history 負 ROI = bet しない rational behavior)
- evaluation: 動作確認は OK、 学習 量 不足 + reward shaping 不足

### Phase 21C 5/10 score 帯別 重大訂正
- ❌ 「0.6-0.7 中位 ROI 242% 最強」 = 新潟R9 荒川峡特別 ¥12,570 outlier 由来
- 戦略⑦ filter 適用後 中位 ROI 52% (negative profit)
- 案 B 改 strict (≥0.7、 graded/special/Kyoto 除外) ROI 134% (5/10) ← N=3 統計弱
- 中位 hit 率 50% (5/10) は事実だが **配当倍率 低**:
  - 中位 hit 倍率: 京都R7 0.99x、 京都R10 0.64x (払戻 ≤700)
  - 上位 hit 倍率: 東京R8 4.0x (¥2,810)

---

## A. 既存 V22 RL env 仕様 review

### state (8-dim)
```python
[top1_score, num_horses/18, cond_enc/5, dist/3000,
 surface, is_grade, is_special, is_kyoto]
```

### action (Discrete 4)
- 0: skip (¥0)
- 1: small (¥100)
- 2: medium (¥300)
- 3: large (¥700)

### reward (現状)
```python
if bet == 0:
    reward = 0.0
else:
    ratio = bet / 700.0
    scaled_payout = race.payout * ratio if race.trio_hit else 0
    reward = (scaled_payout - bet) / 700.0
```

### 問題点 (Phase 21F 指摘)
1. **戦略⑦ alignment なし**: graded/special/京都 で bet しても penalty なし → 案 B 改 strict 違反
2. **¥2,100/日 上限 なし**: 累計 bet 制約 reward に反映されず
3. **skip incentive 弱**: skip の reward = 0 だが、 低 score race で skip した場合の **明示的 reward** なし
4. **payout 倍率 weighted なし**: ¥700 hit 1 件と ¥2,810 hit 1 件が同価値扱い (絶対値 profit のみ)
5. **1番人気過信 penalty なし**: 高 confidence (高 score) で graded race bet → 過信失敗ケース未捕捉

---

## B. ★ V22 RL reward function v1 設計 (Phase 21F core) ★

### 設計原則 (絶対遵守)

1. **case B 改 strict 戦略 reflect** (累計 +¥13,420 守る投資保護整合)
2. **Phase 21C 重大訂正 反映**: 中位 over-weight しない、 outlier に踊らされない
3. **payout 倍率 weighted**: 高配当 hit に大 reward
4. **skip rational behavior 保護**: 低 confidence race の skip を明示的に正報酬
5. **budget hard constraint**: ¥2,100/日 上限 hard penalty

### 数式 (v1)

```python
def compute_reward_v1(state, action, race, daily_cumulative_bet) -> float:
    """V22 RL reward function v1 (Phase 21F)
    
    state: dict (top1_score, is_grade, is_special, is_kyoto, ...)
    action: int (0=skip, 1=¥100, 2=¥300, 3=¥700)
    race: RaceRecord (trio_hit, payout, race_name, ...)
    daily_cumulative_bet: int (current day total bet so far)
    """
    BET_AMOUNTS = [0, 100, 300, 700]
    bet = BET_AMOUNTS[action]
    score = state['top1_score']
    
    # === component 1: base profit (倍率 weighted) ===
    if bet == 0:
        base = 0.0
    else:
        ratio = bet / 700.0
        scaled_pay = race.payout * ratio if race.trio_hit else 0
        # log-scaled payout で 高配当 を強調
        if race.trio_hit:
            base = (scaled_pay - bet) / 700.0  # raw profit ratio
            # bonus: payout 倍率 sqrt scaling (¥2,810 hit > ¥700 hit を明示)
            multiplier = race.payout / 700.0  # hit 時の倍率
            payout_bonus = 0.5 * (np.sqrt(multiplier) - 1.0)  # ¥2,810→+0.5、 ¥700→0、 ¥10,000→+1.4
        else:
            base = -bet / 700.0  # full loss
            payout_bonus = 0.0
    
    # === component 2: 戦略⑦ alignment (案 B 改 strict 反映) ===
    strategy_7_penalty = 0.0
    if bet > 0:
        if state['is_grade']:
            strategy_7_penalty -= 0.5   # graded 重賞 = 戦略⑦除外
        if state['is_special']:
            strategy_7_penalty -= 0.3   # 06_平場特別 = 戦略⑦除外
        if state['is_kyoto']:
            strategy_7_penalty -= 0.2   # 京都 = 戦略⑦除外 (5/11 暫定)
    
    # === component 3: budget 上限 hard constraint ===
    budget_penalty = 0.0
    if bet > 0 and daily_cumulative_bet + bet > 2100:
        budget_penalty -= 1.0  # ¥2,100 上限超 → hard penalty
    
    # === component 4: skip rational reward ===
    skip_reward = 0.0
    if bet == 0:
        if score < 0.5:
            skip_reward += 0.05   # 低 score skip = rational
        elif state['is_grade'] or state['is_special']:
            skip_reward += 0.03   # 戦略⑦ 除外 race の skip = rational
        # (高 score race の skip は機会損失なので reward なし)
    
    # === component 5: 1 番人気過信 penalty (高 score race の bet hit/miss 区別) ===
    # 高 score (≥0.8) + graded race + bet → 過信判定
    # (現状 V15 が graded で AUC 低めの傾向、 5/10 NHK マイル C 0.703 + 8 着 失敗 ref)
    overconfidence_penalty = 0.0
    if bet > 0 and score >= 0.8 and state['is_grade'] and not race.trio_hit:
        overconfidence_penalty -= 0.2
    
    # === component 6: 中位 score (0.6-0.7) ===
    # ★ Phase 21C: 「中位最強」 は誤り、 戦略⑦適用で 52% ★
    # → 中位 over-weight しない。 base + payout_bonus に任せる。
    # → 中位 で 戦略⑦ pass + hit した場合のみ 微小 bonus (high hit% を活用)
    midband_bonus = 0.0
    if (0.6 <= score < 0.7
        and bet > 0 and race.trio_hit
        and not state['is_grade']
        and not state['is_special']
        and not state['is_kyoto']):
        midband_bonus += 0.1   # 微小 bonus (over-weight しない)
    
    return float(
        base + payout_bonus
        + strategy_7_penalty
        + budget_penalty
        + skip_reward
        + overconfidence_penalty
        + midband_bonus
    )
```

### reward decomposition (代表 case 試算)

| case | bet | score | hit | grade | special | kyoto | base | pay_bonus | 戦略⑦ | budget | skip | overconf | midband | **合計** |
|------|-----|-------|-----|-------|---------|-------|------|-----------|--------|--------|------|----------|---------|----------|
| 案 B 改 hit (上位) | 700 | 0.72 | ✓ ¥2,810 | F | F | F | +3.0 | +0.5 | 0 | 0 | 0 | 0 | 0 | **+3.5** |
| 案 B 改 miss (上位) | 700 | 0.72 | ✗ | F | F | F | -1.0 | 0 | 0 | 0 | 0 | 0 | 0 | **-1.0** |
| graded で bet (NG) | 700 | 0.81 | ✗ | T | F | F | -1.0 | 0 | -0.5 | 0 | 0 | -0.2 | 0 | **-1.7** |
| graded skip (OK) | 0 | 0.81 | - | T | F | F | 0 | 0 | 0 | 0 | +0.03 | 0 | 0 | **+0.03** |
| 中位 hit (戦略⑦ pass) | 700 | 0.65 | ✓ ¥1,810 | F | F | F | +1.59 | +0.30 | 0 | 0 | 0 | 0 | +0.1 | **+1.99** |
| 中位 miss | 700 | 0.65 | ✗ | F | F | F | -1.0 | 0 | 0 | 0 | 0 | 0 | 0 | **-1.0** |
| 低 score skip | 0 | 0.45 | - | F | F | F | 0 | 0 | 0 | 0 | +0.05 | 0 | 0 | **+0.05** |
| budget 超過 | 700 | 0.72 | ✗ | F | F | F | -1.0 | 0 | 0 | -1.0 | 0 | 0 | 0 | **-2.0** |

### 想定挙動 (1M timesteps 学習後)

| 状況 | agent action 期待 |
|------|--------------------|
| ≥0.7 + 戦略⑦ pass + budget OK | bet 700 |
| ≥0.7 + graded/special | skip |
| 0.6-0.7 + 戦略⑦ pass + budget OK | bet 100-300 (中位 微小 bonus 反映、 over-weight 回避) |
| <0.5 | skip (skip reward + miss 確率高) |
| budget 超過時 | skip 強制 (hard penalty) |

### Phase 17 v0 (現行) との差分

| 項目 | v0 | v1 (Phase 21F) |
|------|----|----|
| 戦略⑦ filter | × | ✓ (graded -0.5 / special -0.3 / kyoto -0.2) |
| budget 上限 | × | ✓ (¥2,100/日 hard -1.0) |
| skip 報酬 | × | ✓ (低 score / 除外 race +0.03-0.05) |
| payout 倍率 weighted | × | ✓ (sqrt 倍率 bonus) |
| 1 番人気過信 penalty | × | ✓ (graded + 高 score + miss -0.2) |
| 中位 score 微小 bonus | × | ✓ (戦略⑦ pass + hit +0.1) |
| state 拡張 | (8-dim) | (8-dim 維持、 state には daily_cumulative_bet を追加 = 9-dim) |

→ env (state space) 拡張: 8-dim → **9-dim** (daily_cumulative_bet / 2100 normalized)
→ action space は v1 では 拡張せず Discrete(4) 維持

---

## C. action space 拡張検討

### phase 1 (5/15-5/16): Discrete(4) 維持
- 学習安定性 優先
- v0 → v1 reward function 改修 のみ で 効果 検証
- 100K-1M timesteps で reward shaping 効き目 confirm

### phase 2 (5/24+ JV-Link 加入後): MultiDiscrete([4, 3])
- bet 額: Discrete(4) = ¥0/100/300/700
- 馬券種: Discrete(3) = trio (3連複7点) / umaren (馬連 2 点) / fukusho (複勝 1 点)
- 状況依存 馬券種 (条件E では umaren、 ≤7 頭で 三連複は的中率 73% だが配当低)
- 30 年 backfill data + 1M timesteps 学習で 状況別 ticket type 最適化

### phase 3 (12/1+、 V22 production 候補): MultiDiscrete([4, 3, K])
- bet 額 + 馬券種 + 軸馬選択 (top1 / top2 / top3 のどれを軸にするか)
- K=3 (top1/top2/top3 軸選択) または K=18 (全馬から軸選択)
- 連続値 action は学習不安定 + 解釈性 低 → **採用見送り**
- 検証コスト 大、 30 年 data + 10M timesteps + GPU 16h+ 必要

### 採用判定

| phase | timing | 期待効果 | 検証コスト | 採用 |
|-------|--------|----------|-----------|------|
| phase 1 | 5/15-5/16 | reward 改修 効果 検証 | 100K = 30 min | ★ 即実施 ★ |
| phase 2 | 5/24-7/15 | ticket type 最適化 | 1M = 4-8h | 5/24+ 検討 |
| phase 3 | 12/1+ | 軸馬最適化 | 10M = 16h+ | 慎重判定 |

---

## D. 5/15-5/16 学習計画

### PC スペック
- CPU: AMD Ryzen 7 (Zen 4 世代想定、 8 core / 16 thread)
- RAM: 32 GB
- GPU: RTX 4070 Ti SUPER (16 GB VRAM)
- SB3 PPO は MlpPolicy で GPU 利用は marginal、 CPU 主体 (env step が bottleneck)

### 工数試算

| timesteps | 想定 時間 | 用途 |
|-----------|-----------|------|
| 5K | 30 sec | smoke test (Phase 17 完了済) |
| 100K | 30 min | reward v1 効果 即時 検証 |
| 500K | 2.5h | hyperparameter sweep (lr / clip_range / gamma) |
| 1M | 5-10h | full convergence 確認 |
| 10M | 50-100h | 30 年 data 必須 (5/24+ Phase 17c) |

### 5/15 (木) 夜 plan: 100K test
- 21:00-21:30 reward v1 実装 (tools/v22_rl_agent.py 改修、 reward function refactor)
- 21:30-22:00 100K timesteps 学習 + test 30 ep 評価
- 22:00-22:30 action distribution 確認、 案 B 改 strict alignment check
  - 期待: bet 700 を ≥0.7 + 戦略⑦ pass で選択、 graded で skip
- 22:30-23:00 結果 doc 作成、 commit + push

### 5/16 (金) 朝 plan: 500K validation
- 06:00-08:30 500K timesteps (background)
- 08:30-09:00 結果 review、 hyperparameter 最適化判定
  - lr: 3e-4 (default) vs 1e-4 vs 5e-5
  - clip_range: 0.2 vs 0.1 vs 0.3
  - gamma: 0.99 vs 0.95 vs 0.999

### 5/16 (金) 夜 plan: 1M full
- 19:00-04:00 (overnight) 1M timesteps × 3 seed (variance 評価)
- 朝 5/17 (土) 結果 review

### 5/17 (土) plan: paper trade 並行
- 朝 daily_predict 後 V22 RL agent test inference
- 5/17 中央競馬 actual results に対し V22 RL agent action sequence + simulated profit
- ★ V15 production は 案 B 改 strict 維持 (絶対) ★

### 5/24+ Phase 17c (JV-Link 加入後 dedicated session)
- 30 年 data fetch (8-16h)
- 1M-10M timesteps re-training (GPU 16-24h)
- WF backtest (30 年 train / 1 年 test rolling)
- V22 production 投入判定: WF AUC sustainability + Sharpe ratio + ROI sustainability
- 期待 ROI: 165-180% sustainable (Session #84 design)

---

## E. 検証 metric (5/15 100K test 用)

| metric | 期待値 | 測定方法 |
|--------|--------|----------|
| 全 skip 率 | < 80% (v0 100%) | test 30 ep action dist |
| 戦略⑦ pass で bet 率 | > 30% | grade=F + special=F + kyoto=F の race で bet>0 比率 |
| graded で skip 率 | > 80% | grade=T で action=0 比率 |
| 案 B 改 strict 整合 | bet 全 race で score ≥ 0.65 | bet>0 race の score 分布 |
| ¥2,100/日 上限遵守 率 | > 90% | 日次 cumulative bet 集計 |
| 中位 score (0.6-0.7) bet 率 | 10-30% | 中位 race の bet 比率 (over-weight 回避) |
| avg reward | > -0.5 | 全 ep 平均 (v0 = 0.0 全 skip) |

→ 100K で「全 skip 解消 + 戦略⑦ alignment」 = ★ Phase 21F success criteria ★
→ 1M で「ROI sustainability + Sharpe」 = Phase 17c 評価 (5/24+)

---

## F. 実装方針 (5/15 21:00-22:00 用 メモ)

### tools/v22_rl_agent.py 改修箇所

```python
# 1. KeibaVoteEnv.__init__: state 9-dim + daily_cumulative_bet tracking
self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(9,), dtype=np.float32)
self.daily_cumulative_bet = 0

# 2. reset: daily_cumulative_bet = 0
def reset(self, *, seed=None, options=None):
    ...
    self.daily_cumulative_bet = 0
    return self._make_obs(0), {}

# 3. _make_obs: 9-dim (state 8 + cumulative_bet/2100)
def _make_obs(self, idx):
    base = _race_to_state(self.current_races[idx])
    return np.concatenate([base, [self.daily_cumulative_bet / 2100.0]]).astype(np.float32)

# 4. step: reward v1 計算 + cumulative_bet 累積
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
    ...
```

### 学習 script

```bash
# 5/15 夜
python tools/v22_rl_agent.py --steps 100000 --reward-version v1 --seed 42

# 5/16 朝 (lr sweep)
for lr in 3e-4 1e-4 5e-5; do
    python tools/v22_rl_agent.py --steps 500000 --reward-version v1 --lr $lr --seed 42
done

# 5/16 夜 (3 seed × 1M)
for seed in 42 123 7; do
    python tools/v22_rl_agent.py --steps 1000000 --reward-version v1 --seed $seed --bg
done
```

---

## G. ★ V15 投資保護 ★ (絶対遵守)

| 項目 | 状態 |
|------|------|
| V15 model (keiba_model_v135*) | ✅ 不変 |
| tools/predict_core.py | ✅ 不変 |
| tools/daily_predict.py | ✅ 不変 |
| tools/race_auto_notify.py | ✅ 不変 |
| 案 B 改 strict 戦略 | ✅ 維持 (5/9-5/16) |
| 累計 +¥13,420 | ✅ 維持 |
| 撤退余裕 +¥63,420 | ✅ 維持 |
| schtask | ✅ 不変 |
| Phase 21F 成果物 | data/v22/phase21f_*.md (新規 doc のみ、 code 不変) |

V22 RL は 5/24+ JV-Link 加入後の 30 年 data 学習を経て、 12/1+ production 投入候補。
それまで V15 案 B 改 strict が production 唯一。

---

## H. 結論 + 次 step

### Phase 21F 成果
1. ✅ V22 RL reward function v1 設計完了 (Phase 21C 重大訂正反映済)
2. ✅ action space 拡張 phase 分け (phase 1 維持 / phase 2 5/24+ / phase 3 12/1+)
3. ✅ 5/15-5/16 学習 plan 確定 (100K → 500K → 1M、 RTX 4070 Ti SUPER 工数試算)
4. ✅ V15 投資保護 完全維持

### 5/15 着手項目 (実装 phase)
- [ ] tools/v22_rl_agent.py 改修 (state 9-dim + reward v1)
- [ ] compute_reward_v1 単体 test (代表 case 8 件)
- [ ] 100K timesteps 学習 + 全 skip 解消 確認
- [ ] action distribution 検証 (戦略⑦ alignment)

### 5/24+ Phase 17c (JV-Link 加入後)
- [ ] 30 年 data fetch
- [ ] 1M-10M timesteps full training
- [ ] WF backtest 30 年
- [ ] V22 production 投入判定

---

## I. 参考: Phase 21C 5/10 score 帯別 重大訂正 要点 (再掲)

| 戦略 | N | hit | hit% | inv | pay | profit | ROI | 評価 |
|------|---|-----|------|-----|-----|--------|-----|------|
| 中位 0.6-0.7 全 R | 10 | 5 | 50% | 7,000 | 16,950 | +9,950 | 242% | ❌ outlier 由来 |
| 中位 + 戦略⑦ | 7 | 3 | 43% | 4,900 | 2,570 | -2,330 | **52%** | ❌ negative |
| 案 B 改 strict (≥0.7、 京都除外) | 3 | 1 | 33% | 2,100 | 2,810 | +710 | **134%** | ✅ 推奨 |
| 全 R baseline | 34 | 11 | 32% | 23,800 | 27,090 | +3,290 | 114% | (参考) |

→ V22 RL reward function は 「中位 over-weight 罠」 を回避し、 案 B 改 strict 戦略 を学習 base に。
→ 中位 score の 高 hit% (50%) は微小 bonus +0.1 で 反映 (over-weight 防止)。
