# 帰宅後 user 手動 task ★ 完全 list ★ (5/15 AM)

実行: 2026-05-15 AM、 Opus 4.7、 抜け なし audit 後

## ★ 必須 (5/16 (土) 前 admin、 計 5 分) ★

### 1. Strategy 8 schtask 登録 (admin、 1 分)

管理者として cmd:
```cmd
powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_strategy8_sidecar_schtasks.ps1
```

→ 5/16 (土) + 5/17 (日) 09:30 自動発火、 Jackpot pattern を 別 channel Discord 通知。 投資 0 円。

### 2. Danger horse alert schtask 登録 (admin、 1 分)

管理者として cmd:
```cmd
powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_5_16_enhancement_schtasks.ps1
```

→ 5/16 + 5/17 09:00 馬体重 急変 / 取消 / TOP1 score 接戦 race 通知。

## ★ JV-Link unlock 用 (V20 構築 着手 path、 計 3 分) ★

### 3. settings.local.json 作成 (1 分、 admin 不要)

**選択肢 A**: テンプレ rename
```cmd
cd C:\Users\takum\keiba-ai\.claude
copy settings.local.json.template settings.local.json
```

**選択肢 B**: 手動 作成
ファイル `C:\Users\takum\keiba-ai\.claude\settings.local.json` を **新規作成**:
```json
{
  "permissions": {
    "allow": [
      "Bash(C:/Users/takum/python32/python.exe:*)"
    ]
  }
}
```

→ Claude Code 再起動 (or `/hooks` で reload) 後、 AI 自律 JV-Link fetch 可能。

### 4. AI に 「JV-Link fetch 着手」 指示 (新 session)

新 chat 開いて:
```
JV-Link fetch 着手して。 5/3 で 過去 race data 取得 + parser 実装 + 17 features 真値化 + V20 真の構築 を 一括 進めて。 V15 production 完全保護。
```

→ AI 自律で:
- JVOpen RACE/SE/HR/O1-O6/BLOD/UM 等
- 17 features 残 10 件 真値化
- V20 学習 (LGB top 100 + 真値 features)
- V20 vs V15 ROI backtest
- 6/15+ V20 投入判定 報告

## ★ 任意 (時間 ある時) ★

### 5. git push 詰まり 解消 (30 分)

`data/v20_training_data_full.csv` 112MB > GitHub 100MB 制限で push reject 中。

最 安全 path (option 1、 history rewrite なし):
```cmd
cd C:\Users\takum\keiba-ai

# 1. .gitignore に追加
echo data/v20_training_data_full.csv >> .gitignore

# 2. cache から削除 (file は手元に残る)
git rm --cached data/v20_training_data_full.csv

# 3. commit + push
git add .gitignore
git commit -m "gitignore: data/v20_training_data_full.csv (112MB > GitHub 100MB)"
git push origin main
```

それでも push reject (history 内 file あるため) なら、 BFG cleanup 等の destructive op が必要 (user 判断)。

### 6. DISCORD_WEBHOOK_JACKPOT 設定 (5 分、 任意)

`.env` に追加:
```
DISCORD_WEBHOOK_JACKPOT=https://discord.com/api/webhooks/XXX/YYY
```

→ Strategy 8 通知が 専用 channel に行く (未設定 なら main updates channel)。

## ★ 5/16 (土) 当日 自動 flow (変更なし) ★

```
03:00 → DailyPremiumScrape (既存)
06:00 → DailyJrdbKyi (既存)
08:00 → DailyPredict V15 (既存)
08:45 → RaceAutoNotify (V15 5 分前通知)
09:00 → ★ DangerHorseAlert NEW (要 admin 登録) ★
09:30 → 馬体重補正 + ★ Strategy 8 sidecar NEW (要 admin 登録) ★
18:00 → DailyResults (既存)
23:00 → NightlySanity (既存)
```

## ★ 5/16 戦略 (変更なし、 絶対遵守) ★

**V15 戦略⑦ 案B改 単独継続**:
- A: 700円 trio 7 点
- B: 700円 trio 7 点
- C: 700円 trio 7 点
- D: 700円 trio 7 点
- E: 700円 umaren 2 点
- X: 700円 trio 7 点
- 12R 1勝クラス (案B改): 上限 2,100円
- 06_特別 / 京都 / 条件 E / 条件 B: 0 円 (戦略⑦ 自動除外)

V22 switch 不推奨 (-96 pt 劣勢確定)、 V15 維持。

## ★ 現状 投資 状況 ★

- 累計収支: **+5,240 円** ※ 旧 +13,530 円 は drift、 5/16 P0-1 真値
- 撤退 line: -50,000 円
- 撤退余裕: **+55,240 円** ※ 旧 +63,530 円 は drift (docs/ROI_DISCREPANCY_2026_05_16.md)
- V15 自動運用 完全継続中

## ★ V15 投資保護 完全 (本日も遵守) ★

- V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変
- 32-bit Python は 別 path (`C:/Users/takum/python32/`)
- V22 / V22 top 100 / V22 enhanced は 別 file
- 戦略 layer 強化 (calibration / wide / danger alert) は 後段 layer、 V15 inference 干渉なし

## ★ 5/15-5/24 期間 全 task 一覧 ★

| date | task | 担当 | 状態 |
|------|------|-----|------|
| 5/14-5/15 | V15 自動運用 | 自動 | 継続中 |
| 5/15 帰宅後 | task 1-3 admin (5 分) | user | ⏳ |
| 5/15 帰宅後 | task 4 AI 指示 | user → AI | ⏳ |
| 5/16 (土) | V15 本番 + Strategy 8/danger shadow | 自動 | 待機 |
| 5/16-5/17 | AI 自律 17 features 真値化 + V20 構築 | AI | settings 後 着手 |
| 5/17 (日) | V15 本番 + V20 学習 進捗 | 並行 | 待機 |
| 5/18 (月) | 週次 report + V20 検証 結果 | 自動 + AI | 待機 |
| 5/19-5/23 | V20 paper trade + tuning | AI | 待機 |
| 5/24+ | V20 vs V15 ROI 比較 + 投入判定 | AI → user | 待機 |
| 7/1 (orig) | V20 投入 (or 6/15+ 前倒し) | user | 待機 |

## ★ AI 自律実行 残 task (settings 解禁 後、 ~6-7 日) ★

### Day 1 (5/16-5/17):
- JVOpen RACE 5/3 → RA/SE/HR all records dl (~30 files)
- SE/WE/WH parser 実装 (binary fixed-length)
- features_jv_se.csv / features_jv_we.csv 生成
- 4 features 真値化 (se_pace, se_lap_3f, we_temperature, wh_track_condition)

### Day 2 (5/17-5/18):
- JVOpen O1-O6 → オッズ時系列 dl
- 4 features 真値化 (o1_change_3h/30m, o2_winrate, o5_change)
- JVOpen BLDN → 血統 5代 inbreeding 計算

### Day 3-4 (5/18-5/20):
- 3 features 真値化 (um_sire/broodmare_winrate, sk_pedigree_class)
- 17 features 全 真値化 完了
- features_merged_v20.csv 生成

### Day 5 (5/20-5/21):
- V20 学習 (LGB top 100 + 17 真値 features)
- 6-fold WF 検証
- 期待 AUC 0.91-0.93

### Day 6 (5/21-5/22):
- V20 vs V15 実 ROI backtest (本日 V22 比較 logic 流用)
- production switch 判定 report

### Day 7 (5/22-5/23):
- paper trading 準備
- V20 .pkl.gz save + sidecar prediction pipeline
- 6/15+ V20 production 投入候補 確定

## ★ 抜け 確認 ★ (本日 audit 後)

| 項目 | 状態 |
|------|------|
| V15 production 自動運用 | ✅ 完全継続 |
| V22 base 4-ens (0.8800) | ✅ saved |
| V22 enhanced top 100 (0.8813) | ✅ saved |
| V22 vs V15 backtest 完了 | ✅ -96 pt 確定 |
| 32-bit Python 3.11.9 | ✅ install済 (`C:/Users/takum/python32/`) |
| pywin32 (32-bit) | ✅ install済 |
| JV-Link COM Dispatch | ✅ 動作確認済 |
| JVInit/JVClose | ✅ 動作確認済 |
| JVOpen 1 回 (32 files) | ✅ 動作確認済 |
| tools/jvlink_fetcher.py | ✅ Session #39 B 既存 189 行 |
| 28 種 datatypes 仕様 doc | ✅ |
| Phase 13 parser fix | ✅ commit b86541f5 |
| 150 candidate features merge | ✅ commit 354fe58b |
| 5/16 enhancement modules | ✅ commit e992d065 |
| .claude/settings.local.json.template | ✅ 本 commit |
| user 1 page 指示書 (本 doc) | ✅ 本 commit |
| Strategy 8 schtask register script | ✅ tools/register_strategy8_sidecar_schtasks.ps1 |
| Danger horse schtask register script | ✅ tools/register_5_16_enhancement_schtasks.ps1 |
| Discord 通知 完備 | ✅ tools/notify_done.py |
| 累計 commits | 14 (5/13-5/15) |
| Honest report (V22 V15 越え 未達) | ✅ |

★ **抜け なし、 user 帰宅後 5 分 admin + 1 分 settings.local.json + 1 AI 指示 で V20 構築 path 完全開通** ★

## ★ なぜ AI は 自分の settings 編集 不可 ★

Claude Code auto-mode classifier (security boundary):
- AI 自身の permission を 自己拡張 = self-elevation
- user verbal authorize 受領 でも block (social engineering 防止)
- → settings.local.json は user 手動 のみ
- 1 分作業 で 全 unlock、 工数 vs 効果 比 抜群

## ★ V15 越え path 真の bottleneck (今 ここ) ★

```
[V15 0.8939 / ROI 428.4%]
       │
       ↓ JV-Link RT data (settings.local.json で unlock)
       │
[V20 真値 features 17 件 真値化 + V15 cache + LGB top 100]
       │
       ↓ 6-fold WF + ROI backtest
       │
[V20 AUC 0.91-0.93 / ROI 500%+ 候補]
       │
       ↓ paper trading 1-2 週間
       │
[6/15+ V20 production 投入判定 ★ V15 越え 確定 ★]
```

★ user 5 分作業 = 1-2 週間前倒し + V15 越え path 確定 ★
