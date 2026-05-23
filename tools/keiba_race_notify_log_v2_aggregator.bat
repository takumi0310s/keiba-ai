@echo off
:: ============================================================
:: Keiba-RaceNotifyLogV2-Aggregator (DAILY 20:30)
:: ★ 完成-1 sub-task (5/18) で作成 ★
::
:: 実 module: tools/race_notify_log_v2_aggregator.py
:: DailyResultsEvening (20:00) 完了後 3 phase log 集計
:: --date arg を schtask %date% から渡す
:: 注記: register_race_notify_log_v2_aggregator_schtask.bat は
::       --date を渡さない (default = 当日)。 完成-1 では --date 明示
:: ============================================================
cd /d C:\Users\takum\keiba-ai
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

:: yyyymmdd 形式 (cmd %date% は環境依存、 ja-JP は 'yyyy/mm/dd')
set TODAY=%date:~0,4%%date:~5,2%%date:~8,2%

%PYTHON_EXE% -u tools\race_notify_log_v2_aggregator.py --date %TODAY% >> logs\keiba_race_notify_log_v2_aggregator_%TODAY%.log 2>&1
exit /b %ERRORLEVEL%
