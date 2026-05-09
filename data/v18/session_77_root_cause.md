# Session #77 B: root cause 特定

## 結論: 2 件の bat 物理欠如

| schtask | 期待 path | 実在 |
|---------|----------|------|
| `Keiba-PreRacePredict_Watchdog_5_9` | `C:\Users\takum\keiba-ai\pre_race_predict_runner.bat` | NO ★ |
| `Keiba-RaceDayReport_Sat` | `C:\Users\takum\keiba-ai\race_day_report.bat` | NO ★ |
| `Keiba-RaceDayReport_Sun` | `C:\Users\takum\keiba-ai\race_day_report.bat` | NO ★ |

(他 35 件は bat 在中、 健全)

## 経緯

### case 1: pre_race_predict_runner.bat
- Session #65 commit `3e5a0ea6` で schtask `Keiba-PreRacePredict_Watchdog_5_9` 登録
- bat + python 本体 (`tools/stage2_predict.py`) は **dev/two-stage** branch のみ存在
- main branch には merge されず、 git history 不在
- main checkout 中は file 物理欠如 → silent_runner Line 24 popup
- Session #73 audit で既知問題と認識済 (silent fail 容認)

### case 2: race_day_report.bat
- schtask 自体は admin 経由で登録済 (recent)
- bat 本体 commit 不在 (`git log -- race_day_report.bat` 空)
- python 本体 `tools/race_day_report.py` は main 在中
- bat wrapper のみ欠如

## 5/9 18:30 popup の最有力候補

| time | task | bat 在中 | popup 可能性 |
|------|------|----------|------|
| 18:00 | RaceDayReport_Sat | NO | ★★★ |
| 19:30 | PreRacePredict_Watchdog_5_9 (30 分毎) | NO | ★★★ |

両者とも Last Result=0 の表示だが、 Run() raise 時 vbs は abort し result 反映が不安定。
ユーザー目視 popup 18:30 頃 → 18:00 RaceDayReport_Sat 起因確度高。

## 既存 task 不在の影響

Session #73 doc 引用 (data/v18/session_73_schtasks_check.md L86-93):

> **PreRacePredict_Watchdog_5_9**: ★ silent fail (5/10 動作不能) ★
> PreRacePredict_Watchdog の silent fail は補助 通知の停止のみで、
> V15 production / 投票 logic / 累計収支 +13,530 円 に影響なし。

→ 機能停止は補助通知のみ、 V15 投資保護に影響なし。
→ ただし popup 出現で ユーザー UX 損なう → 修復必須。
