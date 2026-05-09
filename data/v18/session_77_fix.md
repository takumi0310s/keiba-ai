# Session #77 C: 修復実装

## 修復内容: 2 件 bat 新規作成 (main 在)

### file 1: pre_race_predict_runner.bat (新規)

stage2_predict.py が main 不在のため graceful no-op stub:

```bat
@echo off
REM Session #77: silent_runner.vbs Line 24 ERROR_FILE_NOT_FOUND fix
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist logs mkdir logs
echo [%DATE% %TIME%] pre_race_predict_runner stub args=%* >> logs\pre_race_predict.log
if exist tools\stage2_predict.py (
  python tools\stage2_predict.py %* >> logs\pre_race_predict.log 2>&1
) else (
  echo [%DATE% %TIME%]   stage2_predict.py not on main, no-op exit 0 >> logs\pre_race_predict.log
)
exit /b 0
```

挙動:
- main: stage2_predict.py 不在 → 無害 no-op、 exit 0、 popup 出ず
- dev/two-stage merge 後: 自動切替で stage2_predict.py 実行

### file 2: race_day_report.bat (新規)

```bat
@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python tools/race_day_report.py %* >> logs\race_day_report.log 2>&1
```

`tools/race_day_report.py` は main 在中 (commit `30b6c1bb`)。 bat wrapper を補完。

## 動作確認

```
$ cmd /c "wscript.exe ...silent_runner.vbs ...pre_race_predict_runner.bat --check-next-1h"
ExitCode: 0   ★popup 出ず★

$ cmd /c "wscript.exe ...silent_runner.vbs ...race_day_report.bat --help"
ExitCode: 0   ★popup 出ず★

$ schtasks /Run /TN "Keiba-PreRacePredict_Watchdog_5_9"
SUCCESS: Last Result=0、 log 正常記録

logs/pre_race_predict.log:
  [2026/05/09 19:43:42.67] pre_race_predict_runner stub args="--check-next-1h"
  [2026/05/09 19:43:42.67]   stage2_predict.py not on main, no-op exit 0
```

## 全 38 schtasks bat 在 確認

修復後 audit:
```
$ verify all silent_runner-using tasks
ALL_BATS_EXIST  ✓ (38/38 bat 物理在)
```

## 不採用 案

| 案 | 理由 |
|----|------|
| stage2_predict.py を dev/two-stage から cherry-pick | Session #65/72 依存多数、 並行 Session #73/74/75/76 干渉 |
| schtask `Keiba-PreRacePredict_Watchdog_5_9` 削除 | destructive op、 5/10+ 復元忘れ risk |
| silent_runner.vbs に file 存在 check 追加 | 全 38 task 影響、 副作用 risk 大 |

→ no-op stub bat 案 採用 (最小侵襲、 dev/two-stage merge 時 自動有効化)。
