# Strategy Layer v2 - Design Document

**実施日**: 2026-05-16
**ファイル**: `tools/strategy_layer_v2.py`
**目的**: 戦略⑦ (V15 production) 完全不変前提で、 EV 動的閾値 + calibration を **追加 layer** で適用する shadow runner を構築

## 1. 既存 EV 計算 logic 確認 (read-only)

### `tools/predict_core.py` `calc_horse_ev()` (line 2384-2414)

```python
# 各馬の複勝圏確率を スコア比率 × 3 で推定 (clip 0.85)
top3_probs = clip(scores / score_sum * 3.0, 0.0, 0.85)

# trio 三連複の想定配当倍率 ≈ 単勝オッズ × 2.0
trio_multiplier = odds * 2.0
ev = p × trio_multiplier
```

### `app.py` `calc_expected_values()` (line 4094+)

- 単勝 EV = win_prob × odds  (win_prob = score_ratio × 1.5、 clip 0.6)
- ワイド EV = p1 × p2 × 0.8 × wide_odds
- **UI 表示用** で、 戦略 layer での bet 判定には未連携

### 現行 EV 閾値 = **無し (固定 0)**

`tools/race_auto_notify.py` には EV filter は存在しない。 戦略⑦ は race-level 条件のみ:
- 06_平場特別 除外
- 条件 E (頭数<=7) 除外
- 条件 B (重〜不馬場) 除外
- 京都 filter は 5/10 で 削除済 (※race_name + course)

bet_size は **常に 700 円 / race** で固定 (uma_ren 2 点 350+350 含む)。

## 2. v2 で追加する layer

### 2-1. 入力

```python
{
    'race_meta': {
        'race_id': str,
        'race_name': str,
        'course': str,        # 中山/京都/...
        'condition': str,     # A/B/C/D/E/X
        'num_horses': int,
        'distance': int,
        'track_condition': str,
        'bet_type': str,      # trio / umaren
    },
    'horses_df': pd.DataFrame  # columns: 馬番, 馬名, スコア, 単勝オッズ
}
```

### 2-2. 処理 chain

```
Step 1: 戦略⑦ filter (既存 logic を 純粋再実装、 変更なし)
   ↓ (PASS した race のみ)
Step 2: V15 raw scores → calibration blend (isotonic 30% weight)
   ↓
Step 3: race-level EV 計算
   - 三連複 fmt EV = max(top3 horses 期待値) または mean(top3 horses EV)
   - 主指標: top1 EV (top1 horse の prob × odds × 2.0)
   ↓
Step 4: 動的 bet_size 決定
   - EV < 1.0  → recommended=False (skip)
   - 1.0 <= EV < 2.0 → bet_size = 700  (現状維持)
   - 2.0 <= EV < 3.0 → bet_size = 1400 (2x)
   - EV >= 3.0       → bet_size = 2100 (3x)
   ↓
Step 5: output
```

### 2-3. 出力

```python
{
    'recommended': bool,         # 最終的に bet するかどうか
    'bet_size': int,             # 700 / 1400 / 2100 / 0
    'ev_top1': float,            # 主指標
    'ev_mean_top3': float,       # 参考
    'p_calibrated_top1': float,  # calibrated top1 prob
    'reason': str,               # 採否理由 ("strategy_7_pass + ev_2.3")
    'strategy_7_pass': bool,
    'meta': {...},               # debug
}
```

## 3. 安全装置

### 3-1. **V15 production 不変保証**

- `tools/strategy_layer_v2.py` は **standalone**
- `tools/race_auto_notify.py` は import せず、 戦略⑦ logic を **独自 reimplementation** (上記 race_auto_notify.py:171-184 の copy)
- 5/16-5/17 G1 day 本番 通知 logic に影響 0%

### 3-2. shadow mode (default)

- `notify=False` で動作、 Discord 送信なし
- 結果は `data/v21/strategy_v2_shadow.csv` に append のみ
- 既存 `cumulative_results.csv` は **read-only**、 書き込まない

### 3-3. calibrator 安全 clip

- isotonic 単体は p=0.3 以上で 1.0 飽和 → bet 判定崩壊
- blend factor 0.3 で raw 重視 (70% raw + 30% calibrated)
- ★ 平均 V15 prob は 0.17、 blend 後の prob は 約 0.20-0.25 程度に納まる

### 3-4. 数値範囲 sanity

- EV > 10.0 は abnormal → log warn + bet_size 700 へ clip (誤った odds=999 等の防護)
- bet_size 上限 2100 (3x) で固定、 増額暴走 防止

## 4. backtest 制約

### data 制約 (★ 重要 ★)

- `data/cumulative_results.csv` 529 settled rows のうち **top1_score available は 20 行のみ**
- 残り 509 行は top1_score が NaN (5/10 以前 cumulative_results.csv に score 書き込まれていない 既知 bug)
- → EV 計算には **top1_score + 単勝オッズ + 全馬 scores** が必要だが、 cumulative_results は **race-level outcome のみ**

### backtest 方針

1. **baseline ROI**: cumulative_results.csv の actual_payout / investment を真値として 戦略⑦ (E/B/06_特別 除外) 後の ROI を計算
2. **v2 ROI**: top1_score available な 20 行のみ EV 計算可能、 これは 統計的 sample 不足のため **simulation 上 shadow result** に留める
3. **v2 expected impact**: 想定 (paper shadow eval が真の検証)

→ 本 backtest は **baseline 確定値 + v2 limited sample** の 2 段構え 報告
→ ★ 真の検証は 5/18+ paper shadow data 蓄積で実施

## 5. paper shadow eval (5/18+)

### data flow

```
tools/daily_predict.py (既存、 不変)
    ↓ writes daily_predictions/YYYYMMDD.csv  (top1_score 含む、 5/11+ 修正済)
tools/strategy_layer_v2.py --shadow (新規、 5/18+ 起動)
    ↓ reads daily_predictions/YYYYMMDD.csv + 当日 odds
    ↓ writes data/v21/strategy_v2_shadow_YYYYMMDD.csv
    ↓ (notify せず、 production 戦略⑦ 通知は別 path で 並行)
夜 (daily_results 後)
    ↓ 実 outcome を join、 ROI 比較 csv 更新
```

### 比較指標

- baseline 戦略⑦ ROI (race_auto_notify.py 実 bet)
- v2 ROI (shadow only、 actual bet なし)
- 差分 (ROI delta、 bet 数 delta、 hit_rate delta)

評価 window: 累計 30 races (約 1 週間) 以上で次 step 判定。

## 6. リスク mitigation

| リスク | 対応 |
|--------|------|
| calibrator 過剰飽和 | blend 0.3、 raw 重視 |
| 21 sample 学習 | 5/18+ shadow 蓄積 → 200 件超で再 train |
| odds 取得失敗 | 固定 odds=10.0 fallback (predict_core 既存と同じ) |
| 戦略⑦ logic drift | reimplementation 内に既存 logic copy、 PR 時 race_auto_notify.py と一致 check |
| 5/16-5/17 本番影響 | strategy_layer_v2.py は 起動されない、 race_auto_notify 不変 |
| top1_score=NaN race | EV 計算 skip、 strategy_7_pass のみで 700円 bet 維持 (≒ baseline 動作) |
