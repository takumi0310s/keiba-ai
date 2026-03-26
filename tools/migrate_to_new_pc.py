#!/usr/bin/env python
"""別PC移行用パッケージ作成

gitignore対象のローカル専用ファイルをzipにまとめる。

Usage:
    python tools/migrate_to_new_pc.py
"""
import os
import sys
import io
import zipfile
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    ('.env', '.env'),
    ('keiba_race_results.db', 'keiba_race_results.db'),
    ('keiba_predictions.db', 'keiba_predictions.db'),
    ('data/feature_lookups.pkl', 'data/feature_lookups.pkl'),
    ('data/feature_lookups.pkl.gz', 'data/feature_lookups.pkl.gz'),
    ('data/feature_lookups_lite.pkl', 'data/feature_lookups_lite.pkl'),
    ('data/jra_races_full.csv', 'data/jra_races_full.csv'),
    ('data/training_times.csv', 'data/training_times.csv'),
    ('data/blood_full.csv', 'data/blood_full.csv'),
    ('data/jra_payouts.csv', 'data/jra_payouts.csv'),
    ('data/netkeiba_speed_index.csv', 'data/netkeiba_speed_index.csv'),
    ('data/netkeiba_training_times.csv', 'data/netkeiba_training_times.csv'),
    ('data/netkeiba_stable_comments.csv', 'data/netkeiba_stable_comments.csv'),
    ('data/scrape_progress.json', 'data/scrape_progress.json'),
    ('jockey_wr.json', 'jockey_wr.json'),
]

def main():
    date_str = datetime.now().strftime('%Y%m%d')
    out_path = os.path.join(BASE_DIR, f'migrate_package_{date_str}.zip')

    print("=" * 50)
    print("  Migration Package Builder")
    print("=" * 50)

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        added = 0
        for src, arc in FILES:
            full = os.path.join(BASE_DIR, src)
            if os.path.exists(full):
                size = os.path.getsize(full) / 1024 / 1024
                zf.write(full, arc)
                print(f"  + {arc} ({size:.1f} MB)")
                added += 1
            else:
                print(f"  - {arc} (not found, skip)")

    pkg_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  Package: {out_path}")
    print(f"  Files: {added}, Size: {pkg_size:.1f} MB")
    print(f"\n  移行先で展開してkeiba-aiディレクトリに上書きしてください。")


if __name__ == '__main__':
    main()
