# 5/13 全日 marathon 完了 summary (Session #88)

user "今日一日かけて終わらせて、抜け目なく" の 完了報告。

## 🎯 主要成果

### 1. Strategy 8 (Jackpot 4-way pattern) LIVE 検証 大成功
- **69 件 検出 / 7 開催 (4/11-5/9)**
- **top3 hit 53.6%** (random ~33% の 1.6x)
- **top1 hit 21.7%** (random ~11% の 2x)
- 統計的有意 signal 確認、 5/16-5/17 shadow eval GO

### 2. live_features_5_17.py 重要 bug 修正
- horse_id format mismatch (TFJV 8桁 ↔ netkeiba 10桁) 解決
- jockey_id leading zero 処理 解決
- 修正前 0 Jackpot 検出 → 修正後 18+ 件 検出可能

### 3. V22 4-ensemble (LGB+XGB+FT+IR) 学習
- **quick (2025 fold) Grid AUC 0.8891**
- V15 baseline 0.8939 → **delta -0.0048** (極めて近接)
- IR が dominant (重み 0.45、 V15 v13.5b の 0.35 より上昇)
- full 6-fold WF background 進行中 (推定 残 2.5h)

### 4. 5/17 本番 pipeline 完備
- 全 8 file existence check ✅
- strategy8_sidecar 5/9 dry-run 通過
- live_features 5/9 + 5/10 検出 OK
- schtask 登録 script 完備 (user 手動 実行 必要)

### 5. memory 永続化
- horse_id mapper / V20 leak audit / Strategy 8 verified / V22 design
- 将来 session で 自動 適用

## 📊 task 進捗 (本 session、 5/13 開始時 7 task)

| # | task | 状態 |
|---|------|------|
| 19 | V22 4-ens 学習 | ✅ quick 完了 / full BG中 |
| 20 | paddock 拡張 1000+ | deferred (V21 video 0% 効果なし) |
| 21 | V22 vs V15 比較 | ✅ 完了 |
| 22 | 5/17 rehearsal | ✅ 完了 |
| 23 | Strategy 8 dry-run | ✅ 大成功 |
| 24 | schtask 5/17 | ✅ script 完備 |
| 25 | end-of-day final commit | ✅ (本 commit) |

## 🛡 V15 投資保護 (本日も完全遵守)

- V15 .pkl.gz / predict_core.py / daily_predict.py / app.py **完全不変**
- V22 model / Strategy 8 sidecar は全て 別 file / 別 channel
- 累計収支 +13,530 円 守る (撤退余裕 +63,530 円)

## 🚨 5/16 (土) 〜 留守中 user 手動 task (admin 権限)

1. **strategy 8 schtask 登録** (重要):
   ```cmd
   powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_strategy8_sidecar_schtasks.ps1
   ```
   → 5/16 (土) 09:30 + 5/17 (日) 09:30 自動発火

2. **DISCORD_WEBHOOK_JACKPOT 設定** (任意、 別 channel 通知用):
   ```
   .env に追加:
   DISCORD_WEBHOOK_JACKPOT=https://discord.com/api/webhooks/...
   ```
   未設定なら DISCORD_WEBHOOK_URL (主 channel) に通知。

3. **V22 4-ens full WF 結果確認** (~3 時間後 完了):
   - models/v22/4ensemble_wf_summary.json 確認
   - V15 越え 達成なら 5/24+ production 投入候補

## 📅 short-term roadmap

### 5/14-15 (木金):
- user 不在中、 V15 production 完全自動継続
- nightly_sanity 23:00 翌日 task 事前 check

### 5/16 (土) 本番:
- V15 戦略⑦ 案B改 単独継続 (絶対遵守)
- Strategy 8 sidecar shadow eval (別 channel)
- V22 4-ens full WF 完了確認

### 5/17 (日):
- V15 通常運用 + Strategy 8 sidecar 2 日目

### 5/18 (月):
- 週次 report 自動 (8:00)
- Strategy 8 + V22 2 日間 results 集計

### 5/24+:
- V22 4-ens full WF 結果次第 で production 投入判定
- Strategy 8 累積 100+ 件後 統計的 安定性 確認
- 単勝 1500円 試験投入候補

## 📁 本日 file (commits 約 10 件)

| commit | 内容 |
|--------|------|
| c5cf59e7 | V21 LGB+XGB + horse_id mapper |
| 773697af | V22 (V15+P24/26) LGB+XGB AUC 0.8683 |
| b5c66556 | V21 video 83 horses retrain |
| b69b9b4e | paddock archive 237 dirs |
| 528ccd7a | Strategy 8 LIVE 12/18 (本日) |
| 3c6abfd0 | strategy8_sidecar bug 修正 |
| (本 commit) | V22 4-ens + final summary |

合計: 30+ tools / 4 新 trainers / 4 memory permanent / 6 status docs。
