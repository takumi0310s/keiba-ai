@echo off
REM daily cumulative audit -- read-only drift detect (5/16 sub-task 10)
REM schedule: daily 21:00 (DailyResultsEvening 20:00 の 1 時間後想定)
cd /d "C:\Users\takum\keiba-ai"
python tools\daily_cumulative_audit.py >> logs\daily_cumulative_audit.log 2>&1
