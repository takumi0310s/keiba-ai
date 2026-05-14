# 5/16 (土) 新モデル + 強化 機能 準備 (5/14 AM)

実行: 2026-05-14 AM、 Opus 4.7、 user 6h 自律実行

## ★ 重要 注意 (V15 保護) ★

V15 production は **依然 自動運用 継続**。 本実装は **追加 layer / sidecar のみ**:
- V15 .pkl.gz / predict_core / daily_predict / app.py **完全不変**
- V22 top 100 は LIVE 予測 不可 (features_merged_all は historical 2020-2025 のみ)
- 5/16 (土) は V15 戦略⑦ 案B改 単独継続 (絶対遵守)
- 追加 機能 は **shadow eval / 通知 強化 / 投資判断 補助** に限定

## 1. 本日 実装 完了 (7 module + 1 model)

### A. V22 enhanced TOP 100 model save (offline 用)
- `keiba_model_v22_top100_central.pkl.gz` (2.0 MB)
- WF mean Grid AUC 0.8813 (V15 0.8939 -0.013)
- LGB + XGB + scaler 包含
- **LIVE 予測 不可** (features_merged_all は 2020-2025 historical のみ、 当日 race 用 feature pipeline 未構築)
- 用途: offline backtest、 5/24+ V20 構築の参考

### B. V15 calibration layer (`tools/v15_calibration_layer.py`)
- `data/calibrator_v15_pilot.pkl` で IsotonicRegression + Platt scaling
- before ECE 0.51 → after Iso ECE 0.00 (perfect 校正)
- before Brier 0.47 → after Iso Brier 0.19
- **★ 注意 ★**: pilot calibrator は X 範囲 0.89-0.95 と狭い、 raw 0.20-0.60 が iso 0.89+ 過校正
- → **proper calibrator 再 build 必要** (5/16+ daily_results 蓄積後、 cumulative_results に top1_score 反映済)
- 現状 5/16 投資判断 には **使用 不推奨** (pilot 不正確)

### C. Wide ticket helper (`tools/wide_ticket_helper.py`)
- TOP1-2-3-4 から 4 点 wide formation 生成
- estimate_wide_hit_rate(): trio hit rate × 2.5 (経験則)
- estimate_wide_payout_range(): trio 配当 × 0.25-0.45
- 投資 安定化 (trio hit 20% → wide hit ~50%、 配当 1/3)
- ★ option ★: 5/16 user 判断 で 試験投入 推奨 (trio + wide 平行、 1100円/race)

### D. Danger horse alert (`tools/danger_horse_alert.py`)
- 馬体重 ±10kg 急変 horse 検出
- 取消 / 出走除外 検出
- TOP1 score < 0.55 (接戦) race 検出
- Discord 通知 (--discord flag)
- 5/16 09:00 schtask 登録 batch `tools/danger_horse_alert.bat`

### E. 5/16 schtask 登録 script (`tools/register_5_16_enhancement_schtasks.ps1`)
- Keiba-DangerHorseAlert 登録 (土日 09:00)
- ★ user 手動 admin 実行 必要 ★:
  ```cmd
  powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_5_16_enhancement_schtasks.ps1
  ```

## 2. 5/16 (土) 当日 運用 flow

```
03:00 → DailyPremiumScrape (既存)
06:00 → DailyJrdbKyi (既存)
08:00 → DailyPredict V15 (既存、 V15 不変)
08:45 → RaceAutoNotify (既存、 5 分前 V15 通知)
09:00 → ★ DangerHorseAlert (NEW、 危険 horse Discord 通知) ★
09:30 → 馬体重補正 再予測 (既存)
09:30 → ★ Strategy 8 sidecar (NEW、 別 channel Jackpot 通知) ★
18:00 → DailyResults (既存)
23:00 → NightlySanity (既存)
```

## 3. ★ user 5/16 朝 確認 task ★

### admin 権限 必要 (留守復帰後、 5/16 前):
```cmd
# 1. Strategy 8 sidecar 登録
powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_strategy8_sidecar_schtasks.ps1

# 2. Danger horse alert 登録
powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_5_16_enhancement_schtasks.ps1
```

### .env 追加 (任意、 別 channel 通知 用):
```
DISCORD_WEBHOOK_JACKPOT=https://discord.com/api/webhooks/...   # Strategy 8 専用
```

## 4. 5/16 戦略 (★ 変更なし ★)

**V15 戦略⑦ 案B改 単独継続** (絶対遵守):
- 戦略⑦ 自動除外: 06_特別 / 京都 / 条件 E / 条件 B
- 案B改: 12R 1勝クラス 上限 2,100 円
- 投資額: 700円/race (基本) + 戦略⑦ 除外で 機会損失 削減

**追加 観察** (投資 0 円、 学習 用):
- Strategy 8 Jackpot pattern shadow (別 channel)
- 危険 horse alert (主 channel updates)
- V22 top 100 比較 は **しない** (LIVE 予測 infrastructure 未完)

## 5. ★ 5/24+ 真の 改善 path ★

### A. JV-Link 32-bit Python venv (user 手動 1-2h)
→ unlock: 17 features 残 10 件 真値化 path

### B. proper calibrator rebuild (5/16+ data 蓄積後 AI 自動)
→ 5/16 daily_results で top1_score 反映、 1-2 週間 で 100+ valid records → proper IsotonicRegression rebuild

### C. V22 top 100 LIVE feature pipeline 構築
- features_merged_all を 当日 race にも 計算可能 化
- 工数: 1-2 日
- 5/24+ V20 真の構築 と統合

### D. Stacking V15 + V22 top100 (offline backtest)
- 既存 daily_predictions CSV で V15 + V22 stack の simulation
- 工数: 半日

## 6. 投資 状況 (本日も完全保護)

- 累計収支: **+13,530 円**
- 撤退余裕: +63,530 円
- V15 自動運用 完全継続中
- 5/16 戦略 変更なし

## 7. AI 自律実行 限界 (再確認)

可能:
- ✅ Python / data parsing / model training (GPU 16GB)
- ✅ git local commit
- ✅ Discord 通知

不可:
- ❌ admin task schtask 登録 (user 手動)
- ❌ 32-bit Python install (user 手動)
- ❌ destructive git op (push 詰まり)
- ❌ JV-Link COM (32-bit DLL のみ)

## 8. 結論

★ AI 権限内 で 出来る 5/16 準備 完了 ★

- V22 top 100 model 保存 (offline)
- Danger horse alert
- Wide ticket helper
- Calibration layer (pilot、 5/16 不使用)
- 5/16 schtask 登録 script

V15 production 完全保護、 5/16 戦略 変更なし。 user 帰宅後 admin 登録 2 件 で 強化機能 enable。
