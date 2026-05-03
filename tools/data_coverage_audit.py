"""5/4 朝 データカバレッジ完全監査.

主要データソースを date range / row count / 欠損で分析。
"""
import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)

now = datetime.now()


def file_info(fp):
    if not os.path.exists(fp): return None
    sz = os.path.getsize(fp)
    mt = datetime.fromtimestamp(os.path.getmtime(fp))
    return {'size': sz, 'mtime': mt.strftime('%Y-%m-%d %H:%M'),
             'age_d': (now - mt).days, 'size_mb': sz/(1024*1024)}


def date_range_from_race_id(df, col='race_id'):
    if col not in df.columns: return None
    rids = df[col].astype(str)
    if len(rids) == 0: return None
    yrs = rids.str[:4]
    yr_counts = yrs.value_counts().sort_index()
    return {
        'rows': len(df),
        'min_year': yrs.min(),
        'max_year': yrs.max(),
        'unique_yrs': len(yr_counts),
        'recent_5yrs': {y: int(c) for y, c in yr_counts.tail(5).items()},
    }


def analyze_jrdb_csv(csv_path):
    if not os.path.exists(csv_path):
        return {'status': 'MISSING'}
    info = file_info(csv_path)
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False, usecols=lambda c: c in ('race_id','日','年','回','race_id_str','date'))
        if 'race_id' in df.columns:
            dr = date_range_from_race_id(df)
            return {**info, **dr}
        elif '年' in df.columns:
            return {**info, 'rows': len(df), 'note': 'no race_id, has 年/回/日/R'}
        else:
            return {**info, 'rows': len(df), 'note': 'no race_id'}
    except Exception as e:
        return {**info, 'error': str(e)[:80]}


print("=" * 70)
print("5/4 データカバレッジ完全監査")
print(f"Time: {now}")
print("=" * 70)

# === Section 1: JRDB CSV ===
print("\n=== 1. JRDB CSV ===")
jrdb_csvs = ['kyi','kka','skb','sed','srb','hjc','ukc','cz','kz','ot','ou','ov','ow','oz',
             'paci','tyb','jo','joa','kab','kta','cha','cyb','bac','kaa','csa','ksa']
jrdb_results = {}
for n in jrdb_csvs:
    fp = f'data/jrdb_{n}.csv'
    res = analyze_jrdb_csv(fp)
    jrdb_results[n] = res
    if res.get('status') == 'MISSING':
        print(f"  jrdb_{n}.csv: ✗ MISSING")
        continue
    rows = res.get('rows', 0)
    yr2026 = res.get('recent_5yrs', {}).get('2026', 0)
    age = res.get('age_d', 99)
    flag = '🟢' if age <= 1 else ('🟡' if age <= 7 else '🔴')
    note = res.get('note', '')
    print(f"  {flag} jrdb_{n}.csv: rows={rows:>9,} 2026={yr2026:>6,} age={age:2d}d size={res.get('size_mb',0):6.1f}MB {note}")

# === Section 2: netkeiba CSV ===
print("\n=== 2. netkeiba CSV ===")
nk_csvs = sorted(glob.glob('data/netkeiba_*.csv'))
for fp in nk_csvs:
    info = file_info(fp)
    if not info: continue
    age = info['age_d']
    flag = '🟢' if age <= 2 else ('🟡' if age <= 7 else '🔴')
    name = os.path.basename(fp)
    try:
        df = pd.read_csv(fp, dtype=str, low_memory=False, nrows=10)
        cols = list(df.columns)[:6]
    except: cols = []
    rows = '?'
    try:
        with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
            rows = sum(1 for _ in f) - 1
    except: pass
    print(f"  {flag} {name:42s} rows≈{rows:>7} age={age:2d}d size={info['size_mb']:6.1f}MB cols={cols[:3]}")

# === Section 3: 訓練データ ===
print("\n=== 3. 訓練データ (training_times, jra_races etc) ===")
train_data = ['data/training_times.csv', 'data/jra_races_full.csv', 'data/blood_full.csv',
              'data/odds_history.csv', 'data/horse_history_full.csv',
              'data/jockey_history_full.csv', 'data/trainer_history_full.csv']
for fp in train_data:
    info = file_info(fp)
    if not info:
        print(f"  ✗ MISSING: {fp}")
        continue
    rows = '?'
    try:
        with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
            rows = sum(1 for _ in f) - 1
    except: pass
    age = info['age_d']
    flag = '🟢' if age <= 7 else ('🟡' if age <= 30 else '🔴')
    print(f"  {flag} {os.path.basename(fp):30s} rows={rows:>10} age={age:2d}d size={info['size_mb']:7.1f}MB")

# === Section 4: odds_base ===
print("\n=== 4. odds_base (前日オッズ) ===")
ob_files = sorted(glob.glob('data/odds_base_*.csv'))
print(f"  total: {len(ob_files)} files")
for fp in ob_files[-10:]:
    info = file_info(fp)
    rows = '?'
    try:
        with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
            rows = sum(1 for _ in f) - 1
    except: pass
    print(f"    {os.path.basename(fp):30s} rows={rows:>5} mtime={info['mtime']}")

# === Section 5: jra_payouts ===
print("\n=== 5. jra_payouts (JRA公式配当) ===")
fp = 'data/jra_payouts.csv'
if os.path.exists(fp):
    df = pd.read_csv(fp, dtype=str, low_memory=False)
    rdates = df.get('race_date', df.get('date'))
    if rdates is not None:
        rdates = rdates.dropna().astype(str)
        latest = rdates.max() if len(rdates) else 'N/A'
        recent = rdates.value_counts().sort_index().tail(5).to_dict()
        info = file_info(fp)
        print(f"  rows={len(df):,} latest_date={latest} mtime={info['mtime']} age={info['age_d']}d")
        print(f"  recent dates: {recent}")
    else:
        print(f"  rows={len(df):,} (no race_date column)")

# === Section 6: weekly_premium_cache ===
print("\n=== 6. weekly_premium_cache ===")
cache_dirs = sorted(glob.glob('data/weekly_premium_cache/*'))
print(f"  total dirs: {len(cache_dirs)}")
for d in cache_dirs[-7:]:
    if not os.path.isdir(d): continue
    files = glob.glob(os.path.join(d, '*'))
    sizes = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    print(f"    {os.path.basename(d):20s} {len(files)} files, {sizes/1024:.0f} KB")

# === Section 7: jrdb extracted ===
print("\n=== 7. jrdb/extracted ===")
ext_dirs = sorted([d for d in glob.glob('data/jrdb/extracted/*') if os.path.isdir(d)])
for d in ext_dirs:
    type_name = os.path.basename(d)
    files_2026 = glob.glob(os.path.join(d, f'*26*.txt'))
    if not files_2026: continue
    # Latest 2026 file
    files_2026.sort()
    latest = files_2026[-1] if files_2026 else 'none'
    n = len(files_2026)
    print(f"  {type_name:8s}: 2026 files={n:3d} latest={os.path.basename(latest)}")

# === Section 8: predictions/results coverage ===
print("\n=== 8. daily_predictions/results coverage ===")
preds = sorted(glob.glob('data/daily_predictions/2026*.csv'))
results = sorted(glob.glob('data/daily_results/2026*.csv'))
print(f"  predictions: {len(preds)} files, latest={os.path.basename(preds[-1]) if preds else 'none'}")
print(f"  results:     {len(results)} files, latest={os.path.basename(results[-1]) if results else 'none'}")
# missing pairs
pred_dates = {os.path.basename(p)[:8] for p in preds}
res_dates = {os.path.basename(r)[:8] for r in results if not r.endswith('_payouts.json')}
both = sorted(pred_dates & res_dates)
print(f"  paired (pred AND result): {len(both)} dates: {both}")
print(f"  pred only: {sorted(pred_dates - res_dates)}")
print(f"  result only: {sorted(res_dates - pred_dates)}")

# === Section 9: training data cache ===
print("\n=== 9. v17/v18 train cache ===")
caches = ['data/v17/_v17_train_df_cache.pkl',
          'data/_v15_optuna_df_cache.pkl.gz']
for c in caches:
    info = file_info(c)
    if info:
        print(f"  {c}: {info['size_mb']:.0f}MB age={info['age_d']}d")
    else:
        print(f"  ✗ MISSING: {c}")

# Save summary JSON
with open('data/results/data_coverage_audit_5_4_raw.json', 'w', encoding='utf-8') as f:
    json.dump({'jrdb': jrdb_results, 'time': str(now)}, f, indent=2, default=str, ensure_ascii=False)
print("\nSaved data/results/data_coverage_audit_5_4_raw.json")
