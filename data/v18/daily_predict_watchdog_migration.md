# DailyPredict → watchdog 移行手順

生成: 2026-05-03

## 状況

既存 `DailyPredict` タスクは `\keiba-ai\` フォルダ配下の HIGHEST 権限タスクで、通常権限では変更不可。
今 (5/3 14:55) の自動変更は admin elevation が無いため不可。

| 項目 | 値 |
|------|---|
| TaskName | DailyPredict |
| TaskPath | \keiba-ai\ |
| 現状 Execute | `C:\Users\takum\keiba-ai\daily_predict.bat` |
| **目標 Execute** | `C:\Users\takum\keiba-ai\daily_predict_watchdog.bat` |
| LastResult | 3221225786 (STATUS_CONTROL_C_EXIT = 5/2 Ctrl+C 中断と一致) |
| NextRun | 2026/05/04 8:00 |

## 手動変更手順 (5/3 中に管理者権限で実施推奨)

### 方法1: PowerShell (管理者として実行)

```powershell
# 管理者として PowerShell を起動 → 以下実行
$task = Get-ScheduledTask -TaskName "DailyPredict" -TaskPath "\keiba-ai\"
$action = New-ScheduledTaskAction -Execute "C:\Users\takum\keiba-ai\daily_predict_watchdog.bat"
Set-ScheduledTask -TaskName "DailyPredict" -TaskPath "\keiba-ai\" -Action $action
# 確認
(Get-ScheduledTask -TaskName "DailyPredict").Actions | Format-List Execute, Arguments
```

### 方法2: cmd.exe (管理者として実行)

```cmd
schtasks /Change /TN "\keiba-ai\DailyPredict" /TR "C:\Users\takum\keiba-ai\daily_predict_watchdog.bat" /F
schtasks /Query /TN "\keiba-ai\DailyPredict" /V /FO LIST | findstr "実行"
```

### 方法3: タスクスケジューラ GUI

1. `taskschd.msc` 起動
2. 左ペインで `タスク スケジューラ ライブラリ\keiba-ai` を選択
3. 中央ペインで `DailyPredict` を右クリック → プロパティ
4. 「操作」タブ → 編集 → プログラム/スクリプトを `C:\Users\takum\keiba-ai\daily_predict_watchdog.bat` に変更
5. OK → OK

## 変更前/変更後の動作

### 変更前 (現状)

```
06:00 DailyJrdbKyi (JRDB取得)
08:00 DailyPredict → daily_predict.bat → daily_predict.py
       ↳ 中断検知なし、Cookie 切れ自動修復なし
       ↳ 5/2 朝の事故再発リスク
```

### 変更後 (watchdog 化)

```
06:00 DailyJrdbKyi
08:00 DailyPredict → daily_predict_watchdog.bat → daily_predict_watchdog.py
       ↳ 5分ごと進捗監視
       ↳ Cookie 切れ → refresh_cookie.py 自動実行
       ↳ 中断検知 → --resume max 3回再起動
       ↳ Discord (#updates) green/red 通知
       ↳ 5/2 事故と同じ問題は構造的に防止
```

## ロールバック手順 (もし問題発生)

```powershell
# 管理者として PowerShell
$action = New-ScheduledTaskAction -Execute "C:\Users\takum\keiba-ai\daily_predict.bat"
Set-ScheduledTask -TaskName "DailyPredict" -TaskPath "\keiba-ai\" -Action $action
```

## テスト方法 (本番前確認)

```bash
# watchdog を手動実行してログ確認
cd /c/Users/takum/keiba-ai
python tools/daily_predict_watchdog.py --date 20260503

# ログ末尾で完了 + Discord 通知確認
tail -30 logs/daily_predict_watchdog_20260503.log
```

## 推奨対応

**5/3 中 (本日中)**:
- 管理者として上記 方法1 を実行
- 5/4 朝 08:00 の DailyPredict 実行で watchdog が動作

**5/4 朝 0809:00 起床時 確認**:
- Discord (#updates) に "[Watchdog] daily_predict 完了" green 通知が届く
- なら成功
- 届かない or red アラートなら手動対応
