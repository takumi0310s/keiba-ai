#!/usr/bin/env python
"""全年一括スクレイプ：training 2020-2022 + comments 2023-2025"""
import subprocess, sys, os, time
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
py = sys.executable

tasks = [
    # Training (+ comments) for 2020-2022
    ([py, os.path.join(BASE, 'tools', 'scrape_premium_data.py'), '--year', '2020'], 'Training 2020'),
    ([py, os.path.join(BASE, 'tools', 'scrape_premium_data.py'), '--year', '2021'], 'Training 2021'),
    ([py, os.path.join(BASE, 'tools', 'scrape_premium_data.py'), '--year', '2022'], 'Training 2022'),
    # Comments only for 2023-2025 (training already complete)
    ([py, os.path.join(BASE, 'tools', 'scrape_premium_data.py'), '--year', '2023'], 'Comments 2023'),
    ([py, os.path.join(BASE, 'tools', 'scrape_premium_data.py'), '--year', '2024'], 'Comments 2024'),
    ([py, os.path.join(BASE, 'tools', 'scrape_premium_data.py'), '--year', '2025'], 'Comments 2025'),
]

start = time.time()
for cmd, label in tasks:
    print(f'\n{"="*60}')
    print(f'  {label} -{datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*60}')
    r = subprocess.run(cmd, cwd=BASE, timeout=14400)
    if r.returncode != 0:
        print(f'  WARN: {label} returned {r.returncode}')

elapsed = (time.time() - start) / 60
print(f'\n{"="*60}')
print(f'  ALL DONE in {elapsed:.0f} minutes')
print(f'{"="*60}')

# Show final stats
import pandas as pd
for name, csv in [('Training', 'data/netkeiba_training_times.csv'),
                   ('Comments', 'data/netkeiba_stable_comments.csv')]:
    df = pd.read_csv(os.path.join(BASE, csv), encoding='utf-8-sig', usecols=['race_id'], dtype=str)
    print(f'\n{name}: {len(df)} rows, {df["race_id"].nunique()} unique races')
    for year in range(2020, 2026):
        n = df[df['race_id'].str.startswith(str(year))]['race_id'].nunique()
        print(f'  {year}: {n}')

# Notify
try:
    sys.path.insert(0, os.path.join(BASE, 'tools'))
    from notify_done import send_notification
    send_notification("Full Scrape完了", f"全年training+comments取得完了\n所要: {elapsed:.0f}分")
except Exception:
    pass
