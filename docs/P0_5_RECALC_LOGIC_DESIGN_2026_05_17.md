# P0-5-B: -15 min 再計算 logic 設計

**作成**: 2026-05-17 Sun JST
**作成 source**: 親 agent 指示 Sub-task P0-5-B (read-only design)
**作業 mode**: 設計のみ。 V15 production / predict_core.py / daily_predict.py / race_auto_notify.py / app.py / 戦略⑦案 C logic / schtasks 完全不変
**前提**:
- P0-4 (`docs/P0_4_FINAL_VERDICT_2026_05_16.md`) で TYB content value = 0 確定 → TYB 経路は P0-5 でも使わない
- 戦略⑦案 C (`docs/P0_2_EXTENSION_DESIGN_2026_05_16.md` / `tools/race_auto_notify.py:171-284`) で 06_特別 / 京都 平場 / 条件E / 条件B を除外済
- V15 ensemble = LGB + XGB + FT-Transformer + IntraRace Attention、 145 truncate (predict_core.py:2160-2163)

---

## 0. 結論 (★ verdict ★)

| 項目 | 値 |
|------|-----|
| **★ 推奨案 ★** | **案 B (post-hoc calibrator) + 案 C (Discord 直前情報通知) combined** |
| **V15 production 改変** | ✅ **0** (新規 script / 別 output / 別 channel のみ) |
| **想定 +AUC** | **est. +0.003 〜 +0.008** (assumption、 30R paper shadow 蓄積後 検証) |
| **想定 +ROI** | **est. +2 〜 +5pt** (assumption、 calibrator 補正 + 通知強化の混合) |
| **着手** | **5/17 (Sun) 21:00 以降** (本日 G1 day = 19R 終了後) |
| **paper shadow eval 期間** | **5/19 (Tue) 〜 6/16 (Tue)、 4 週末 = est. 24〜32 R** |
| **採用判定** | **6/17 (Wed)、 paper ROI delta + 統計検定 PASS で P1 投入候補** |

★ 推奨理由 ★:
- **案 A (zero-shot inference)** は V15 150-feature 固定で新規 features の重み未学習 → 効果ゼロ前提 → 不採用
- **案 B (post-hoc calibrator)** は V15 prob 不変、 別 layer で odds_band + weight_diff (-15 min snapshot) で 微補正 → 既存 `tools/v15_calibration_layer.py` framework 流用可、 production prob は影響なし
- **案 C (直前情報 Discord 通知)** は production 投票 logic 不変、 user 判断材料のみ追加 → 即時 deploy 安全
- ★ 案 B 単独は paper 期間中 30R 不足で 統計的に有意性不確定 → 案 C を併用で 即時 user value 提供 ★

★ honest 注記 ★: +AUC/+ROI 数値は assumption、 実 effect は 5/19+ paper shadow data で検証 確定。

---

## 1. 既存予測 flow 再読 (read-only)

### 1-1. 朝 8:00 daily_predict.py 入出力

`tools/daily_predict.py:233 run_daily_predict()`:

```
1. Cookie 検証 (ensure_cookie_valid)
2. model_data = load_models()  # V15 .pkl.gz
3. races = fetch_race_list(date_str)
4. for race in races:
   a. parse_shutuba(race_id) → horses, race_info
   b. fetch_realtime_odds_full(race_id) → odds_dict, pop_dict
      → save_odds_base(race_id, odds_full, date_str)  # 08:00 morning snapshot
   c. fetch_jra_and_weather(course_name) → jra_info, weather_info
   d. for each horse: get_horse_stats(horse_id) → expanding stats
   e. build_features(horses, race_info, model_data, ...)  # 150 features
   f. merge_jrdb_predict_features(df, race_id)  # KYI 前日 + TYB 直前 (但し TYB は P0-4 で value=0)
   g. predict_race(df, model_data, odds_available, race_info)  # V15 ensemble forward
4. CSV 出力: data/daily_predictions/{date_str}.csv
   columns: race_id, course, race_num, race_name, condition, num_horses, distance,
            surface, track_condition, top1_num, top1_name, top1_score,
            top2_num, top2_name, top3_num, top3_name, trio_bets, bet_type, investment
```

### 1-2. V15 ensemble 構造 (predict_core.py:2149-2347)

```
X = df[use_features].values                                # shape (n_horses, 150)
n_lgb_features = model.num_feature()  # 145 ← TYB 5 features truncate
X_lgb = X[:, :n_lgb_features]                              # shape (n, 145)

lgb_pred = lgb_model.predict_proba(X_lgb)[:, 1]            # LGB
xgb_pred = xgb_model.predict(xgb.DMatrix(X_lgb))           # XGB
ft_pred  = ft_model(scaled X_lgb)                          # FT-Transformer
ir_pred  = ir_model(padded X_ir, mask)                     # IntraRace Attention

combined = w_lgb*lgb_pred + w_xgb*xgb_pred + w_ft*ft_pred + w_ir*ir_pred
# 典型: w_lgb=0.25, w_xgb=0.25-0.30, w_ft=0.10-0.15, w_ir=0.45 (V22 設計参考)

df['スコア'] = combined
sorted_df = df.sort_values('スコア', ascending=False)
top1_score = sorted_df.iloc[0]['スコア']
```

### 1-3. 戦略⑦案 C 適用箇所 (race_auto_notify.py:171-284)

```python
# fetch race_id → race_name / course / num_horses / surface / condition
if course == '06':  # 06_特別 平場
    skip
if course == '京都' and not (G1 or G2 or G3 or L or OPEN):
    skip  # P0-2 案 C (5/17 適用)
if num_horses <= 7:
    skip  # 条件E
if condition in ('重', '不良'):
    skip  # 条件B
# 残った race のみ Discord #買い目 通知
```

★ production 通知 (#買い目) は race_auto_notify.py のみが発火、 daily_predict.py は CSV 出力のみ ★

---

## 2. -15 min 再計算 全体 flow

### 2-1. 想定 flow (新規 script、 V15 完全不変)

```
朝 08:00 daily_predict (★ 既存、 完全不変 ★):
  → V15 ensemble forward
  → data/daily_predictions/{date}.csv (base prediction)
  → data/odds_base/{date}/{race_id}.json (08:00 morning odds snapshot)

朝 08:45 race_auto_notify (★ 既存、 完全不変 ★):
  → 戦略⑦案 C filter 適用
  → 残った race を Discord #買い目 通知

【新規 layer P0-5】 race -20 min: Keiba-DataPreFetch-15min
  trigger: nowracedata_json.json polling (now_data.now_hassotime 検出)
  action:
    - JV-Link O1 (連続 odds) fetch  ← 既存 tools/jvlink_fetcher.py 流用
    - JV-Link WF (馬体重)      fetch  ← 既存 tools/jvlink_fetcher.py 流用
    - JV-Link TCOV (馬場)      fetch  ← 既存 tools/jvlink_fetcher.py 流用
  output:
    data/live_pre_features/{date}/{race_id}_pre.json
    {
      "race_id": "...",
      "fetched_at": "YYYY-MM-DD HH:MM:SS",
      "odds_t15": {umaban: tansho_odds, ...},
      "odds_morning": {...},                    # 08:00 snapshot コピー
      "odds_shift": {umaban: t15/morning, ...},
      "horse_weights": {umaban: weight_kg, ...},
      "weight_diff": {umaban: diff_kg, ...},    # 前走比
      "track": {"cushion": .., "moisture": ..},
      "weather": {...}
    }

【新規 layer P0-5】 race -15 min: Keiba-Recalc-15min
  trigger: race -15 min schtask (毎 5 min poll、 hassotime - 15 で発火)
  action:
    - 朝 base prediction load (daily_predictions/{date}.csv の該当 race row)
    - live_pre_features load
    - calibrator_overlay.apply(top3_prob, pre_features) → calibrated_prob
    - re-rank if calibrated_prob 順位変動あり
  output:
    data/recalc_predictions/{date}/{race_id}_t15.csv
    columns: race_id, top1_morning, top1_t15, top2_morning, top2_t15,
             top3_morning, top3_t15, rank_diff_flag, primary_factor

【新規 layer P0-5】 race -10 min: Keiba-DiscordNotify-10min
  trigger: race -10 min schtask
  filter:
    - 戦略⑦案 C 適用後の race list (race_auto_notify.py の通知済 race のみ)
    - 順位変動 flag = True の race のみ
  action: Discord #updates (★ NOT #買い目 ★) 通知
  output: Discord notification (production 投票判断には反映しない、 user 判断材料)
```

### 2-2. V15 score 不変保証

★ recalc 段で V15 forward pass は 1 回も 走らせない ★。
- 朝 8:00 V15 forward (既存) の output (top3_prob、 top1_score) を **そのまま load**
- calibrator overlay は **scalar 補正** のみ (top1_score → top1_score_calibrated)
- production 通知 (Discord #買い目) は 朝 8:45 の race_auto_notify が既に発火済 → -10 min 通知は 別 channel (#updates)

---

## 3. 戦略 3 案 比較

### 3-1. 案 A: zero-shot inference (★ 不採用 ★)

**手法**: V15 150-feature 入力に 新規 features (odds_shift / weight_diff / cushion_t15) を 追加 column として inject、 model.predict_proba を再 forward

**問題**:
- V15 .pkl.gz は 150 features 固定 (predict_core.py:2151 use_features = model_data['features'])
- 新規 features を追加すると `len(X[0]) != 150` → predict_proba が ValueError
- 新規 features を 既存 150 column に上書きすると、 既存 features 破壊 → 別 feature 推定
- ★ V15 retrain なしで zero-shot に効果出すのは不可能 ★

**判定**: 不採用

### 3-2. 案 B: post-hoc calibrator overlay (★ 推奨 ★)

**手法**:
- V15 朝 8:00 forward の output prob を そのまま load
- 別 layer (calibrator_overlay.py 新規) で `pre_features` を 入力 とした **scalar 補正** を かける
- calibrator は **小規模 sklearn Isotonic + linear correction** (sub-task f2a60a50 framework 流用)
- ★ V15 .pkl.gz / production prob は 完全不変、 補正は別 dict で 出力 ★

**実装 sketch**:
```python
# tools/calibrator_overlay.py (新規、 5/17 21:00+ 着手)
def apply_pre_feature_overlay(top3_prob_morning, pre_features):
    """V15 朝 8:00 prob + -15 min pre features → 補正後 prob.

    V15 prob は そのまま load、 本 func は 別 scalar 補正.

    Args:
        top3_prob_morning: (n_horses,) V15 morning prob
        pre_features: dict from live_pre_features/{date}/{race_id}_pre.json

    Returns:
        (n_horses,) 補正後 prob (same shape)
    """
    # odds_shift = t15 / morning (人気急変)
    # weight_diff = 前走比 馬体重変化 (±10kg は警戒)
    # 各馬 ごと delta 算出 (calibrator は paper shadow で fit)
    delta = calibrator.predict(np.column_stack([odds_shift, weight_diff, ...]))
    return np.clip(top3_prob_morning + delta, 0, 1)
```

**長所**:
- V15 production prob 完全不変
- calibrator は 小規模、 30R 蓄積で fit 可能
- paper shadow で reproducibility 検証可

**短所**:
- 30R 蓄積必要 (5/19-6/16、 約 4 週)
- 統計的有意性検定 必要

**判定**: ★ 推奨 ★ (5/19-6/16 paper shadow、 6/17 GO 判定)

### 3-3. 案 C: Discord 直前情報通知のみ (★ 即時採用 ★)

**手法**:
- V15 score / production 投票判断 完全不変
- race -10 min に Discord #updates (NOT #買い目) で 「直前情報 update」 を 通知
- 内容例:
  ```
  📊 直前情報 (race -10 min): 東京 11R アネモネS
  ・5番 オッズ急変 12.4 → 4.8 (-61%、 人気急上昇)
  ・3番 馬体重 -12kg (前走比、 警戒)
  ・馬場 cushion 8.2 → 7.8 (-0.4、 やや渋り)
  V15 朝予測: 1, 2, 5 (★ 投票確定済、 本通知は user 判断材料 ★)
  ```

**長所**:
- production 完全独立、 即時 deploy 安全
- user に 直前情報 が 流通 (現状 14:00 投票確定後は 情報 ゼロ)
- paper shadow eval の 入力 source としても 流用可

**短所**:
- 投票判断 自体は 変わらない (user が直前情報で再投票するかは別)
- effect 数値化 困難

**判定**: ★ 即時採用 ★ (5/18+ Discord 通知 script 整備、 5/24 (土) から運用)

### 3-4. ★ 推奨 = 案 B + 案 C combined ★

| 案 | 効果 | risk | 着手 |
|----|------|------|------|
| 案 A | 0 | 高 (実装失敗) | × |
| **案 B** | est. +AUC 0.003-0.008 | 中 (paper 4 週) | 5/17 21:00+ 設計、 5/19+ paper |
| **案 C** | user value (情報共有) | 低 | 5/18 着手、 5/24 運用 |

→ ★ 案 B + 案 C 並走 ★

---

## 4. Discord 通知 design

### 4-1. 通知 channel 振り分け

| 通知 | timing | channel | 既存/新規 |
|------|--------|---------|----------|
| 朝 09:30 V15 base + 戦略⑦案 C 適用 後 通知 | 09:30 | DISCORD_WEBHOOK_BETS (#買い目) | 既存、 完全不変 |
| 14:00 投票確定通知 | 14:00 | DISCORD_WEBHOOK_BETS (#買い目) | 既存、 完全不変 |
| ★ race -10 min 速報 (P0-5、 順位変動 + 直前情報) ★ | race -10 min | **DISCORD_WEBHOOK_UPDATES (#updates)** | **新規** |
| 結果照合 (daily_results) | 20:00 | DISCORD_WEBHOOK_UPDATES (#updates) | 既存、 完全不変 |

★ 重要 ★:
- race -10 min 通知は **#updates のみ** (production 投票判断には絶対 inject しない)
- 朝 09:30 V15 base 通知に 「★ 直前情報 update あり race は 14:00 に追加通知あり」 案内のみ (本文 不変)
- 14:00 投票確定通知後の race -10 min 速報は 別個に 発火 (重複ではなく追加 layer)

### 4-2. 通知 format

```
🏇 直前情報 (-10 min) | 東京 11R アネモネS 15:45 発走

【V15 朝 8:00 予測 (★ 投票確定済) ★】
TOP3: 1 ホワイトオーキッド (0.612) / 2 サンライズ (0.534) / 5 ライラック (0.498)
買い目: 三連複 7点 / 700円

【直前情報 update】
🔸 5 ライラック: オッズ 急上昇 12.4 → 4.8 (-61%)
🔸 3 ニューモア: 馬体重 -12kg (前走比、 警戒)
🔸 馬場: cushion 8.2 → 7.8 (-0.4)

【calibrator overlay (paper shadow)】
  → 5 ライラック の prob 補正 0.498 → 0.531 (+0.033)
  → 順位変動 なし (TOP3 同じ)、 stake 維持推奨

★ 本通知は 投票確定後の 直前情報 共有、 V15 production 不変 ★
```

### 4-3. notify.py 拡張 (新規 func 追加)

`tools/notify.py` に `send_recalc_alert(race_id, base_pred, recalc, pre_features)` 関数 追加。
内部で `DISCORD_WEBHOOK_UPDATES` のみに POST。
★ DISCORD_WEBHOOK_BETS には絶対送らない ★

---

## 5. 戦略⑦案 C 整合性

### 5-1. race_auto_notify.py の race list 参照

★ 重要 ★: P0-5 の Discord notify (-10 min) は **戦略⑦案 C 適用後の race のみ** で発火。

実装:
```python
# tools/discord_recalc_notify.py (新規、 5/17 21:00+ 着手)
def get_notified_races(date_str):
    """race_auto_notify.py が朝 09:30 で 通知した race list を取得.

    Source: data/notified_races/{date_str}.csv (★ race_auto_notify が朝 09:30 で save 必要 ★)

    Returns: [race_id, ...]
    """
    path = f"data/notified_races/{date_str}.csv"
    if not os.path.exists(path):
        return []  # safe fallback: 通知しない
    return pd.read_csv(path)['race_id'].tolist()
```

★ 注意 ★: 現状 race_auto_notify.py は 通知済 race を log 保存していない (4/27 確認)。
P0-5 着手 (5/17 21:00+) 前に **race_auto_notify.py に 通知済 race log 出力 を追加 必要** → ★ これは V15 production 通知 logic 不変 (log 出力 追加のみ、 通知判定 logic 不変)、 別 sub-task で実装 ★

### 5-2. 京都 / 条件E / 条件B 除外の維持

- 京都 R が -15 min 段で 大幅 odds shift 観測されても、 戦略⑦案 C で skip 維持
- P0-5 の -10 min 通知も skip (notified_races/{date_str}.csv に含まれない)
- ★ recalc 自体は 全 race 実行 (paper shadow eval data 蓄積目的)、 通知 のみ filter ★

---

## 6. fallback chain

### 6-1. -15 min fetch 失敗時

| failure | fallback |
|---------|----------|
| JV-Link O1 fetch fail | 朝 8:00 odds 使用、 odds_shift = 0 で calibrator 入力 |
| JV-Link WF fetch fail | weight_diff = 0 で calibrator 入力 |
| JV-Link TCOV fetch fail | 朝 08:00 fetch_jra_and_weather の値で計算 |
| 全 source fail | 朝 8:00 V15 prediction そのまま、 通知 skip |

### 6-2. recalc 失敗時

| failure | fallback |
|---------|----------|
| daily_predictions/{date}.csv 読込 fail | recalc skip、 ログ + Discord #updates エラー通知 |
| calibrator overlay fail | 朝 V15 prob そのまま使用、 通知 skip |
| -10 min Discord notify fail | log only、 production 影響なし |

### 6-3. ★ V15 production fallback chain は完全不変 ★

- race_auto_notify.py の既存 fallback (5/8 修正 commit、 -5 min snapshot 等) は不変
- P0-5 失敗は 既存 V15 通知 channel に絶対影響しない (別 process、 別 script)

---

## 7. 実装 step (5/17 21:00+ 着手用)

★ 全 step は V15 production / predict_core.py / daily_predict.py / race_auto_notify.py / app.py 完全不変 ★

### Step 1: data source fetcher (5/17 21:00 着手、 est. 4-8h)

- 新規: `tools/live_data_fetcher.py`
- 既存 `tools/jvlink_fetcher.py` (32-bit Python venv `C:\Users\takum\jvlink-venv\`) を call
- 各 race の 発走 -20 min trigger を nowracedata_json.json polling で検出
- O1 / WF / TCOV を fetch
- output: `data/live_pre_features/{date}/{race_id}_pre.json`

★ dependency ★:
- JV-Link DLL (5/7 動作確認済、 32-bit Python venv)
- nowracedata_json.json (auth 不要、 P0-4 §1-5 確認済)

### Step 2: calibrator overlay (5/18 着手、 est. 2-4h)

- 新規: `tools/calibrator_overlay.py`
- 既存 `tools/v15_calibration_layer.py` の framework 流用 (Isotonic + Platt)
- 入力: V15 morning prob (n_horses,) + pre_features dict
- 出力: 補正後 prob (n_horses,)
- ★ paper shadow eval 30R 蓄積前は **identity (補正なし)** で動作、 30R 後 fit ★

### Step 3: recalc script (5/18 着手、 est. 2-4h)

- 新規: `tools/recalc_15min.py`
- race -15 min schtask 発火、 1 race の prob 補正
- output: `data/recalc_predictions/{date}/{race_id}_t15.csv`

### Step 4: Discord 通知 (5/18 着手、 est. 1-2h)

- 新規: `tools/discord_recalc_notify.py`
- race -10 min schtask 発火
- 戦略⑦案 C 通知済 race のみ filter
- 順位変動 / 直前情報 を #updates に通知

### Step 5: race_auto_notify.py 通知済 race log 出力追加 (5/18 着手、 est. 1h)

- ★ V15 production 通知 logic 不変 ★
- 通知発火後に `data/notified_races/{date_str}.csv` に race_id を append のみ追加
- 出力なし path は P0-5 通知が動かない (safe fallback)

### Step 6: schtask 設計書 (5/18 着手、 est. 1h、 別 sub-task P0-5-C)

- 新規 schtask 3 件 設計 (★ 実登録は user 判断後 admin 実行 ★):
  - `Keiba-LiveDataPreFetch-15min` (race -20 min trigger、 Sat/Sun 08:00-17:30)
  - `Keiba-Recalc-15min`            (race -15 min trigger、 同上)
  - `Keiba-DiscordNotify-10min`     (race -10 min trigger、 同上)
- ★ 既存 schtasks 完全不変 (DailyPredict / RaceAutoNotify / DailyResults 等) ★

### Step 7: paper shadow eval (5/19 Tue 〜 6/16 Tue、 4 週末 = 約 24-32 R)

- production 通知 (-10 min) は **disable** (paper のみ)
- `data/recalc_predictions/` vs `data/cumulative_results.csv` 比較
- 統計検定: paired t-test / Wilcoxon signed-rank、 ROI delta 95% CI

### Step 8: GO/NO-GO 判定 (6/17 Wed)

判定基準 (★ all PASS で P1 投入 候補 ★):
- ✅ GO 条件 1: paper shadow 24+ R で calibrator overlay 後 AUC delta ≥ +0.003 (95% CI 下限 > 0)
- ✅ GO 条件 2: paper shadow 24+ R で ROI delta ≥ +2pt (95% CI 下限 > 0)
- ✅ GO 条件 3: -15 min fetch 成功率 ≥ 95% (JV-Link O1 / WF / TCOV)
- ✅ GO 条件 4: 順位変動 ≥ 5% 観測 (calibrator が動いている確認)
- ✅ GO 条件 5: 既存 V15 production 通知への影響 = 0 (回帰 test)

GO 判定なら P1 で `predict_core.py` への live features 統合 or V15→V20 (RL retrain) plan に組み込み。

---

## 8. V15 production 不変保証 ✅

| 項目 | 不変 |
|------|------|
| `keiba_model_v15_central*.pkl.gz` | ✅ retrain なし |
| `tools/predict_core.py` | ✅ 改変なし |
| `tools/daily_predict.py` | ✅ 改変なし |
| `tools/race_auto_notify.py` (通知 logic) | ✅ 改変なし、 ★ Step 5 で log 出力 追加のみ ★ |
| `app.py` | ✅ 改変なし |
| 既存 schtasks (DailyPredict 08:00 / RaceAutoNotify 08:45 / DailyResults 20:00 等) | ✅ 完全不変 |
| 戦略⑦案 C logic (race_auto_notify.py:171-284) | ✅ 完全不変 |
| Discord #買い目 channel 通知 | ✅ 完全不変 |
| V15 ensemble 重み (LGB/XGB/FT/IR) | ✅ 不変 |
| calibrator overlay 出力 | ★ 別 file (data/recalc_predictions/)、 production prob には絶対 inject しない ★ |

---

## 9. 5/19-6/16 paper shadow eval plan

### 9-1. 蓄積 data

- 各 race の morning prob (V15 朝 8:00) + recalc prob (calibrator overlay 後)
- 結果照合: cumulative_results.csv で 真の top1/top3 / 配当を join

### 9-2. 評価 metric

| metric | 計算 |
|--------|------|
| Brier delta | brier(recalc) - brier(morning) (- が改善) |
| Top1 winner rate | top1_recalc == 真 1 着 の確率 |
| Top3 hit rate | top3_recalc 内に 真 3 着以内 の確率 |
| ROI delta | recalc 順位で 買い目組替えた場合の ROI - morning 順位 ROI |
| Rank shift mean | mean(\|rank_recalc - rank_morning\|) 順位変動量 |

### 9-3. 統計検定

- paired t-test (race 単位 ROI delta)
- Wilcoxon signed-rank test (non-parametric backup)
- bootstrap 95% CI (10,000 trial)

### 9-4. paper shadow 出力先

- `data/recalc_predictions/{date}/{race_id}_t15.csv` (each race)
- `data/recalc_predictions/paper_summary_5_19_6_16.csv` (集計、 6/17 generate)
- `docs/P0_5_PAPER_SHADOW_REPORT_2026_06_17.md` (採用判定書、 6/17 作成)

---

## 10. 採用判定基準 (6/17+)

### 10-1. ALL PASS で P1 投入

P1 候補:
- 案 A: V15 production への calibrator overlay 統合 (race -10 min 通知が production prob に反映)
- 案 B: V20 (RL retrain) の plan に -15 min features を 学習 data として 含める

### 10-2. 部分 PASS (1-2 条件 fail) で 再 paper

- paper shadow 期間 延長 (6/17 → 7/15、 8 週 = 64R+)
- 失敗 metric の根本原因 分析

### 10-3. NO-GO で 永久放棄 + 案 C のみ存続

- 案 C (Discord 直前情報通知) は 即時 user value あり、 calibrator overlay 不要でも 単独運用可
- 案 B 関連 file (tools/calibrator_overlay.py、 tools/recalc_15min.py) は archive

---

## 11. honest 限界

1. ★ +AUC / +ROI 数値は assumption ★、 paper shadow 30R 蓄積後の検証で 確定
2. ★ JV-Link O1 / WF / TCOV の -15 min 実 fetch 成功率 は 5/24 (Sat) 実観測 待ち ★ (5/15 JV-Link unlock 後の production fetch 未実施)
3. ★ nowracedata JSON polling の長期 安定性 ★: P0-4 §1-5 で 1 日 観測のみ、 5/24-6/16 で 連続観測必要
4. ★ race_auto_notify.py の 通知済 race log 出力 ★ は Step 5 で追加実装、 通知 logic 完全不変 だが file IO 追加は あり (production 通知 channel に影響なし)
5. ★ calibrator overlay の fit data 不足 risk ★: 30R は計量経済的に 統計検定 ぎりぎり (95% CI 幅 大きい)、 必要なら paper 期間 4→8 週 延長
6. ★ JRA-VAN 規約 ★: JV-Link 公式経路は 規約上 OK (5/7 加入確認)、 30-60 秒 polling 想定
7. ★ V15 production prob 不変保証 ★ は code path 確認 (predict_core.py:2149) で 担保、 ただし 将来 V15 retrain 時に この設計の 前提が崩れる可能性 (V20 投入時 別 sub-task で再設計)

---

## 12. 重要 警告

- ★ P0-5 実装は 5/17 21:00 (G1 day 19R 終了後) 以降 ★、 本日中 (5/17 朝-夕) は 一切着手しない
- ★ V15 production / predict_core.py / daily_predict.py / race_auto_notify.py / app.py / V15 .pkl.gz は **全 step で 完全不変** ★
- ★ schtask 実登録は user 判断後、 admin 権限 で 別途実行 ★ (本設計では PowerShell script の dry-run 確認のみ)
- ★ git commit / push は親集中 ★ (本 sub-task では 出力 docs 配置のみ、 commit なし)
- ★ G1 day 投票判断 / Discord 通知に 影響する変更 は P0-5 範囲外 ★

---

## 13. 参考 / 出典

- **既存 V15 ensemble**: `tools/predict_core.py:2149-2347` (predict_race func)
- **既存 daily predict flow**: `tools/daily_predict.py:233-470` (run_daily_predict func)
- **戦略⑦案 C 実装**: `tools/race_auto_notify.py:171-284`
- **既存 calibrator framework**: `tools/v15_calibration_layer.py` (Isotonic + Platt、 5/14 完成)
- **JV-Link 経路 (32-bit Python venv)**: `tools/jvlink_fetcher.py` (5/7 動作確認、 29 file 取得 OK)
- **nowracedata JSON 経路**: P0-4 §1-5 (auth 不要、 30-60 秒 polling)
- **P0-4 TYB 永久放棄 verdict**: `docs/P0_4_FINAL_VERDICT_2026_05_16.md` (TYB content value = 0)
- **P0-2 京都除外 戦略⑦案 C**: `docs/P0_2_EXTENSION_DESIGN_2026_05_16.md`
- **既存 live features 5/17 tool (read-only history lookup)**: `tools/live_features_5_17.py`

---

★ 本設計書は 設計のみ。 実装 着手は 5/17 21:00 以降 + user 判断後 admin 権限 schtask 登録のみで 進行 ★
★ V15 production 完全不変、 V15 .pkl.gz retrain なし、 commit/push 親集中、 fabrication なし ★
