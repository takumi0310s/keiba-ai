# V22 RL リスク 分析

**作成**: Session #83
**用途**: V22 RL 投入前の リスク 4 種類 + 撤退 logic 設計

---

## 1. 想定リスク 4 種

| # | リスク | 重大度 | 対策 |
|---|--------|--------|------|
| 1 | **over-fitting risk** | HIGH | walk-forward + dropout + early stop |
| 2 | **投資金額膨張 risk** | CRITICAL | Eighth Kelly cap + max ¥10,000 / R |
| 3 | **メンタル risk (人間)** | HIGH | AI 自動 disable + cooldown |
| 4 | **catastrophic action risk** | CRITICAL | action mask + sanity check |

---

## 2. risk 1: over-fitting

### 2-1. 危険性

- 30 年 backtest data でも RL は **環境記憶** リスク
- 学習 data の 配当パターン に過適合 → real run で崩壊
- 2020-2025 の特定 race で agent が記憶し、 同じレースで高賭金

### 2-2. 対策

| 対策 | 内容 |
|------|------|
| **walk-forward validation** | train: 1996-2020 / val: 2021-2023 / test: 2024-2025 |
| **dropout (network 正則化)** | PPO policy network に dropout 0.1-0.3 |
| **early stop** | val reward 5 epoch 連続 改善なし → 停止 |
| **train-test gap monitoring** | train ROI - test ROI > 30pt → over-fit 判定 |
| **augmentation** | オッズ noise 注入 (±5%)、 race order shuffle |
| **shift 監査** | BT → LIVE shift ≤ 10x (V20 SKB 教訓) |

### 2-3. acceptance criteria

- train ROI - val ROI ≤ 20pt
- val ROI - test ROI ≤ 15pt
- LIVE retro shift ≤ 10x

---

## 3. risk 2: 投資金額膨張

### 3-1. 危険性

- RL agent が **aggressive policy** を学習する可能性
  - 連勝後 「もっと賭ければ もっと稼げる」 → ¥10,000 / R 連発
  - 配当 高 race で 一発逆転狙い → 累計 -¥30,000 で壊滅
- **報酬 hack** リスク (Sharpe penalty を回避するため意図的に skip 多発、 重要 race で 高賭金)

### 3-2. 対策 (4 段階)

#### 3-2-1. action space 制限 (hard limit)

```python
# action 設計
max_bet_per_R = 10_000  # ¥10,000 / R 上限
min_bet_per_R = 0       # skip OK
bet_increment = 100     # ¥100 単位
```

#### 3-2-2. Eighth Kelly cap

```python
# Kelly formula × 1/8
kelly_full = (b * p - q) / b  # b=odds-1, p=win_prob, q=1-p
bet = bankroll * kelly_full / 8  # eighth Kelly
bet = min(bet, max_bet_per_R, bankroll * 0.05)  # 5% bankroll 上限
```

#### 3-2-3. daily / weekly cap

```python
max_bet_per_day = 30_000   # ¥30,000 / 日
max_bet_per_week = 100_000  # ¥100,000 / 週
```

#### 3-2-4. drawdown auto stop

```python
if current_drawdown >= 30%:
    bet = 0  # 自動停止
    cooldown = 7  # 7 日間 paper のみ
```

### 3-3. monitoring

- 毎日 22:00: 投資額 / ROI / drawdown を Discord 通知
- ¥5,000 / R 超え → 即時 alert

---

## 4. risk 3: メンタル risk (人間)

### 4-1. 危険性

| 状況 | 人間の感情 | 危険行動 |
|------|----------|---------|
| 黒字 連続 (累計 +¥50,000+) | 「もっと」 | 投資金額 手動 増額 |
| 連敗 連続 (10 連敗+) | 「もう」 | 撤退 OR 倍プッシュ |
| 大勝直後 (+¥10,000 / 日) | 「もう一発」 | 高配当狙いの aggressive bet |
| 大敗直後 (-¥10,000 / 日) | 「取り返す」 | doubling-down |

### 4-2. 対策 (AI 自動制御)

#### 4-2-1. 黒字 急増時 lock

```python
if cum_profit >= cum_profit_30d_max * 1.3:
    # 過去 30 日 max の 1.3 倍 達成 → 1 週間 自動 paper 化
    auto_paper_mode = True
    cooldown = 7
```

#### 4-2-2. 連敗時 cooldown

```python
if losing_streak >= 7:
    auto_paper_mode = True
    cooldown = 3  # 3 日 paper のみ
```

#### 4-2-3. 累計 損失時 強制停止

```python
if cum_profit <= -50_000:
    full_stop = True
    cooldown = 30  # 30 日 完全停止
```

#### 4-2-4. user 手動 override 防止

- 投資金額 manual edit を `.env` flag で **禁止**
- override したい場合は 24 時間 wait period 必須

### 4-3. 心理的 safety net

- **Discord 通知** で 状況可視化 (黒字 / 赤字 / drawdown)
- **週次 自動 report** で 客観評価
- **paper trade 並行** で V20 + 案B改 fallback 確保

---

## 5. risk 4: catastrophic action

### 5-1. 危険性

- RL agent が **明らかに無意味な action** を選択するリスク
  - 全 horse score < 0.1 で 高賭金
  - skip すべき低期待値 R で 三連単 高額購入
  - 同じ horse を 全 action で重複購入

### 5-2. 対策

#### 5-2-1. action mask

```python
def action_mask(state, action):
    # top1 score < 0.3 → skip 強制
    if state.top1_score < 0.3:
        return SKIP
    # 京都 + 06_特別 → skip 強制 (戦略⑦)
    if state.is_kyoto and state.is_06_special:
        return SKIP
    # ev < 1.0 → skip 推奨 (penalty)
    if state.ev_estimate < 1.0:
        return SKIP_OR_LOW_BET
    return action
```

#### 5-2-2. sanity check (post-action)

```python
def sanity_check(action):
    # 1 R 投資 > bankroll の 5% → reject
    if action.bet > bankroll * 0.05:
        action.bet = bankroll * 0.05
    # 同一 horse 重複 → reject
    if has_duplicate_horse(action.combination):
        return SKIP
    return action
```

#### 5-2-3. expert override

- 条件 X (15 頭+ / 重〜不良) で AI が aggressive 過ぎる場合、 V20 + 案B改 fallback 強制

---

## 6. 撤退 logic (3 段階)

| level | trigger | 対応 |
|-------|---------|------|
| **L1 (warning)** | drawdown ≥ 15% | Discord alert、 paper mode 1 日 |
| **L2 (cooldown)** | drawdown ≥ 30% OR 累計 -¥30,000 | 自動 paper mode 7 日 |
| **L3 (full stop)** | 累計 -¥50,000 (撤退ライン) | 30 日 完全停止、 V22 再評価 |

### 6-1. 撤退後の復活 logic

```
L1 → 1 日 paper → 復活
L2 → 7 日 paper、 drawdown 5% 以下回復で 復活 / そうでなければ L3
L3 → 30 日 完全停止 → V22 全面 audit (over-fit / shift / action audit)
       → audit PASS で 再 paper 30 日 → GO で 復活
```

---

## 7. risk audit checklist (V22 投入前)

| # | 項目 | PASS 条件 |
|---|------|---------|
| 1 | over-fit | train-val gap ≤ 20pt、 shift ≤ 10x |
| 2 | 投資膨張 | hard limit + Kelly cap + daily cap 全実装 |
| 3 | メンタル | 黒字/連敗/累計損失 自動制御 全実装 |
| 4 | catastrophic | action mask + sanity check + expert override 全実装 |
| 5 | 撤退 logic | L1-L3 全実装、 復活 logic test PASS |

★ 5/5 PASS で V22 投入候補 ★
★ 4/5 以下 → NO-GO、 12 月 paper 延長 ★

---

## 8. 投資保護 大原則 (V22 投入後も遵守)

1. 撤退ライン: 累計 -¥50,000 (現状 +¥12,830、 余裕 +¥62,830)
2. 取り返し禁止 (損切り後 翌日へ持ち越さない)
3. RL action は **AI 学習 data に依存**、 人間判断より優先
4. 異常時は V20 + 案B改 fallback (損失最小化)
5. 月次 audit (over-fit / shift / risk metric) **必須**

---

## 9. 関連

- [V22_RL_DESIGN.md](V22_RL_DESIGN.md) — MDP + algorithm
- [V22_RL_INFRA.md](V22_RL_INFRA.md) — GPU + library
- [RL_VS_STRATEGY_COMPARISON.md](RL_VS_STRATEGY_COMPARISON.md) — paper 比較
- [V20_DEPLOYMENT_CHECKLIST.md](V20_DEPLOYMENT_CHECKLIST.md) — V20 投入 (前段階)
