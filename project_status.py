#!/usr/bin/env python
"""KEIBA AI Project Status CLI
Usage:
  python project_status.py              # 全セクション表示
  python project_status.py --section model  # モデル精度のみ
  python project_status.py --section data   # データ資産のみ
  python project_status.py --json           # JSON出力
  python project_status.py --export         # data/project_status_report.json に保存
"""
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TASKS_PATH = os.path.join(DATA_DIR, 'project_tasks.json')

# ========== Helpers ==========

def fmt_size(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def fmt_num(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def safe_load_pkl(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def count_csv(path, encoding='utf-8-sig', max_probe=5):
    """Count rows, cols, and probe date range from a CSV."""
    info = {'rows': 0, 'cols': 0, 'date_min': '', 'date_max': '', 'columns': []}
    if not os.path.exists(path):
        return info
    try:
        for enc in [encoding, 'utf-8', 'cp932', 'latin-1']:
            try:
                with open(path, 'r', encoding=enc, errors='replace') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        info['cols'] = len(header)
                        info['columns'] = header
                    count = 0
                    for _ in reader:
                        count += 1
                    info['rows'] = count
                break
            except Exception:
                continue
    except Exception:
        pass

    # Probe date range from columns named year/date/race_date
    try:
        import pandas as pd
        for enc in [encoding, 'utf-8', 'cp932']:
            try:
                df = pd.read_csv(path, encoding=enc, nrows=5, dtype=str, low_memory=False)
                break
            except Exception:
                df = None
        if df is not None:
            info['columns'] = list(df.columns)
            info['cols'] = len(df.columns)
            # Try to find date range efficiently
            date_cols = [c for c in df.columns if c in ('year', 'year_full', 'date', 'race_date')]
            if date_cols:
                # Read just first and last rows
                df_head = pd.read_csv(path, encoding=enc, nrows=3, dtype=str, low_memory=False)
                df_tail = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False,
                                      skiprows=max(1, info['rows'] - 2),
                                      header=None, names=df_head.columns)
                dc = date_cols[0]
                vals = list(df_head[dc].dropna()) + list(df_tail[dc].dropna())
                if vals:
                    info['date_min'] = str(min(vals))
                    info['date_max'] = str(max(vals))
    except Exception:
        pass

    return info


def get_missing_pct(path, sample_rows=1000):
    """Sample CSV and compute per-column missing rate."""
    try:
        import pandas as pd
        for enc in ['utf-8-sig', 'utf-8', 'cp932']:
            try:
                df = pd.read_csv(path, encoding=enc, nrows=sample_rows,
                                 dtype=str, low_memory=False)
                break
            except Exception:
                df = None
        if df is None:
            return {}
        result = {}
        for col in df.columns:
            missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == '').sum()
            pct = missing / len(df) * 100
            if pct > 0:
                result[col] = round(pct, 1)
        return result
    except Exception:
        return {}


# ========== Section 1: Model Accuracy ==========

def section_model():
    models = [
        ('Central V9.2', os.path.join(BASE_DIR, 'keiba_model_v9_central.pkl')),
        ('NAR V2', os.path.join(BASE_DIR, 'keiba_model_v9_nar.pkl')),
        ('V8 Baseline', os.path.join(BASE_DIR, 'keiba_model_v8.pkl')),
    ]

    # Leak-free results
    lf_path = os.path.join(BASE_DIR, 'backtest_leakfree_results.json')
    lf_data = {}
    if os.path.exists(lf_path):
        with open(lf_path, 'r', encoding='utf-8') as f:
            lf_data = json.load(f)

    # WF results
    wf_path = os.path.join(DATA_DIR, 'optimal_betting_jra.json')
    wf_data = {}
    if os.path.exists(wf_path):
        with open(wf_path, 'r', encoding='utf-8') as f:
            wf_data = json.load(f)

    results = []
    print("\n  === 1. MODEL ACCURACY ===\n")
    print(f"  {'Model':<18} {'Version':<10} {'AUC':<8} {'Ens.AUC':<8} "
          f"{'Train Rows':<12} {'Features':<8} {'Trained':<20}")
    print("  " + "-" * 90)

    for label, path in models:
        data = safe_load_pkl(path)
        if data is None:
            print(f"  {label:<18} {'N/A':<10} (file not found)")
            continue
        if not isinstance(data, dict):
            print(f"  {label:<18} {'legacy':<10} (non-dict format)")
            continue

        ver = data.get('version', '?')
        auc = data.get('auc', 0)
        ens_auc = data.get('ensemble_auc', 0)
        n_train = data.get('n_train', '?')
        n_feat = len(data.get('features', []))
        trained = data.get('trained_at', '?')

        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        ens_str = f"{ens_auc:.4f}" if isinstance(ens_auc, float) else str(ens_auc)
        n_train_str = fmt_num(n_train) if isinstance(n_train, (int, float)) else str(n_train)

        print(f"  {label:<18} {ver:<10} {auc_str:<8} {ens_str:<8} "
              f"{n_train_str:<12} {n_feat:<8} {trained:<20}")

        results.append({
            'label': label, 'version': ver, 'auc': auc,
            'ensemble_auc': ens_auc, 'n_train': n_train,
            'n_features': n_feat, 'trained_at': trained,
        })

    # LF info
    if lf_data:
        print(f"\n  Leak-Free BT: train={lf_data.get('train_period','?')}, "
              f"test={lf_data.get('test_period','?')}, "
              f"AUC={lf_data.get('test_auc',0):.4f}, "
              f"N={lf_data.get('races_with_payouts',0)} (scraped payouts)")

    # WF info
    if wf_data:
        fa = wf_data.get('fold_aucs', {})
        if fa:
            avg = sum(float(v) for v in fa.values()) / len(fa)
            years = ', '.join(f"{y}={float(v):.3f}" for y, v in sorted(fa.items()))
            print(f"  Walk-Forward BT: avg AUC={avg:.4f} [{years}]")

    return results


# ========== Section 2: Data Assets ==========

def section_data():
    print("\n  === 2. DATA ASSETS ===\n")
    files = []
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ('.csv', '.json'):
            continue
        size = os.path.getsize(fpath)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M')

        if ext == '.csv':
            info = count_csv(fpath)
            files.append({
                'file': fname, 'rows': info['rows'], 'cols': info['cols'],
                'date_range': f"{info['date_min']}~{info['date_max']}" if info['date_min'] else '',
                'size': size, 'modified': mtime,
            })
        else:
            # JSON: just show size
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
                n_keys = len(jdata) if isinstance(jdata, dict) else len(jdata) if isinstance(jdata, list) else 0
            except Exception:
                n_keys = 0
            files.append({
                'file': fname, 'rows': n_keys, 'cols': 0,
                'date_range': '', 'size': size, 'modified': mtime,
            })

    print(f"  {'File':<38} {'Rows':>8} {'Cols':>5} {'Size':>8} {'Modified':<16} {'DateRange'}")
    print("  " + "-" * 105)
    for f in files:
        ext = os.path.splitext(f['file'])[1]
        rows_str = fmt_num(f['rows']) if f['rows'] else '-'
        cols_str = str(f['cols']) if f['cols'] else '-'
        print(f"  {f['file']:<38} {rows_str:>8} {cols_str:>5} "
              f"{fmt_size(f['size']):>8} {f['modified']:<16} {f['date_range']}")

    print(f"\n  Total: {len(files)} files, {fmt_size(sum(f['size'] for f in files))}")
    return files


# ========== Section 3: Condition Performance ==========

def section_conditions():
    print("\n  === 3. CONDITION PERFORMANCE ===\n")

    # Central conditions from app.py imports would be circular; read from JSON instead
    wf_path = os.path.join(DATA_DIR, 'optimal_betting_jra.json')
    lf_path = os.path.join(BASE_DIR, 'backtest_leakfree_results.json')

    results = {'central': [], 'nar': []}

    # Central
    print("  [Central / JRA]")
    cond_defs = {
        'A': '8-14H / 1600m+ / Good',
        'B': '8-14H / 1600m+ / Heavy',
        'C': '15H+ / 1600m+ / Good',
        'D': '<=1400m (Sprint)',
        'E': '<=7H (Small)',
        'X': '15H+ / Heavy',
    }

    # Load WF data
    wf_conds = {}
    if os.path.exists(wf_path):
        with open(wf_path, 'r', encoding='utf-8') as f:
            wf = json.load(f)
        wf_conds = wf.get('condition_results', {})
        rec = wf.get('recommended', {})

    # Load LF data
    lf_conds = {}
    if os.path.exists(lf_path):
        with open(lf_path, 'r', encoding='utf-8') as f:
            lf = json.load(f)
        lf_conds = lf.get('condition_stats', {})

    print(f"  {'Cond':<6} {'Definition':<24} {'N(WF)':>7} {'Bet':>7} "
          f"{'LF-ROI':>8} {'WF-ROI':>8} {'HitR%':>7} {'Grade':>6} {'Confidence'}")
    print("  " + "-" * 100)

    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        desc = cond_defs.get(ckey, '?')

        # WF data
        wf_c = wf_conds.get(ckey, {})
        # Find best bet type
        best_bt = None
        best_roi = 0
        best_n = 0
        best_hit = 0
        for bt in ['trio', 'umaren', 'wide', 'combo']:
            bt_data = wf_c.get(bt, {})
            roi = bt_data.get('roi', 0)
            if roi > best_roi:
                best_roi = roi
                best_bt = bt
                best_n = bt_data.get('n', 0)
                best_hit = bt_data.get('hit_rate', 0)

        # LF data
        lf_c = lf_conds.get(ckey, {})
        lf_trio = lf_c.get('trio', {})
        lf_roi = lf_trio.get('roi', 0)
        lf_n = lf_c.get('n', 0)

        # Grade
        ref_roi = lf_roi if lf_roi > 0 else best_roi
        if ref_roi >= 120:
            grade = 'SSS'
        elif ref_roi >= 100:
            grade = 'SS'
        elif ref_roi >= 80:
            grade = 'S'
        else:
            grade = '-'

        # Confidence
        if best_n >= 1000 and lf_n >= 50:
            conf = 'HIGH'
        elif best_n >= 500 or lf_n >= 30:
            conf = 'MID'
        elif best_n >= 100:
            conf = 'LOW'
        else:
            conf = 'INSUF'

        bt_str = best_bt or '-'
        lf_str = f"{lf_roi:.1f}%" if lf_roi else '-'
        wf_str = f"{best_roi:.1f}%" if best_roi else '-'
        hit_str = f"{best_hit:.1f}%" if best_hit else '-'

        print(f"  {ckey:<6} {desc:<24} {best_n:>7} {bt_str:>7} "
              f"{lf_str:>8} {wf_str:>8} {hit_str:>7} {grade:>6} {conf}")

        results['central'].append({
            'condition': ckey, 'definition': desc,
            'n_wf': best_n, 'best_bet': bt_str,
            'lf_roi': lf_roi, 'wf_roi': best_roi,
            'hit_rate': best_hit, 'grade': grade, 'confidence': conf,
        })

    # NAR conditions
    print(f"\n  [NAR / Local]")
    nar_path = os.path.join(DATA_DIR, 'optimal_betting_nar.json')
    nar_conds = {}
    if os.path.exists(nar_path):
        with open(nar_path, 'r', encoding='utf-8') as f:
            nar = json.load(f)
        nar_conds = nar.get('condition_results', {})

    nar_bt_path = os.path.join(BASE_DIR, 'backtest_nar_condition.json')
    nar_bt = {}
    if os.path.exists(nar_bt_path):
        with open(nar_bt_path, 'r', encoding='utf-8') as f:
            nar_bt = json.load(f)

    print(f"  {'Cond':<6} {'Definition':<24} {'N':>7} {'Bet':>7} "
          f"{'ROI':>8} {'HitR%':>7} {'Grade':>6} {'Recommended'}")
    print("  " + "-" * 90)

    nar_cond_results = nar_bt.get('condition_results', {})
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        desc = cond_defs.get(ckey, '?')
        nc = nar_cond_results.get(ckey, {})
        bt = nc.get('bet_type', '-')
        roi = nc.get('roi', 0)
        hit = nc.get('hit_rate', 0)
        n = nc.get('n', 0)
        rec = nc.get('recommended', False)

        if roi >= 120: grade = 'SSS'
        elif roi >= 100: grade = 'SS'
        elif roi >= 80: grade = 'S'
        else: grade = '-'

        roi_str = f"{roi:.1f}%" if roi else '-'
        hit_str = f"{hit:.1f}%" if hit else '-'
        rec_str = 'YES' if rec else 'NO'

        print(f"  {ckey:<6} {desc:<24} {n:>7} {bt:>7} "
              f"{roi_str:>8} {hit_str:>7} {grade:>6} {rec_str}")

        results['nar'].append({
            'condition': ckey, 'definition': desc,
            'n': n, 'bet_type': bt, 'roi': roi,
            'hit_rate': hit, 'grade': grade, 'recommended': rec,
        })

    return results


# ========== Section 4: Missing Information ==========

def section_missing():
    print("\n  === 4. MISSING INFORMATION CHECK ===\n")

    checks = [
        ('Jockey stats', 'jockey_history_full.csv', 'jockey_wr.json or JV-Link BR_DATA'),
        ('Trainer stats', 'trainer_history_full.csv', 'JV-Link BR_DATA'),
        ('Blood/Pedigree', 'blood_full.csv', 'JV-Link KT_DATA'),
        ('Horse history', 'horse_history_full.csv', 'JV-Link SE_DATA'),
        ('Race results', 'jra_races_full.csv', 'JV-Link SE_DATA / target_odds.csv'),
        ('Odds history', 'odds_history.csv', 'JV-Link HY_DATA'),
        ('Training times', 'training_times.csv', 'JV-Link CK_DATA / target_sakaro/wood'),
        ('Lap times', 'lap_times.csv', 'TARGET Frontier export'),
        ('NAR races', 'chihou_races_full.csv', 'KDSCOPE or netkeiba scraping'),
        ('Track bias index', None, 'Compute from lap_times.csv per course/day'),
        ('Pace index', None, 'Compute from race pass data in jra_races_full'),
        ('Weather data', None, 'JMA API (api.open-meteo.com) per course/date'),
        ('Jockey-Trainer combo', None, 'Compute from jra_races_full (jockey_id x trainer_id)'),
        ('Wide/Umaren/Trio payouts', None, 'JV-Link HR records or netkeiba scraping'),
    ]

    results = []
    print(f"  {'Data':<28} {'Status':<5} {'File':<30} {'Acquisition Method'}")
    print("  " + "-" * 100)

    for label, fname, method in checks:
        if fname:
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
                status = 'OK'
                icon = 'o'
            else:
                # Check base dir too
                fpath2 = os.path.join(BASE_DIR, fname)
                if os.path.exists(fpath2) and os.path.getsize(fpath2) > 100:
                    status = 'OK'
                    icon = 'o'
                else:
                    status = 'MISS'
                    icon = 'x'
        else:
            status = 'MISS'
            icon = 'x'

        fname_str = fname or '(not created)'
        print(f"  {icon} {label:<26} {status:<5} {fname_str:<30} {method}")
        results.append({'data': label, 'status': status, 'file': fname_str, 'method': method})

    # Data quality for existing CSVs
    print(f"\n  --- Data Quality (top missing columns, sampled 1000 rows) ---")
    key_csvs = [
        'jra_races_full.csv', 'training_times.csv', 'blood_full.csv',
        'horse_history_full.csv', 'odds_history.csv',
    ]
    quality = {}
    for fname in key_csvs:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        missing = get_missing_pct(fpath)
        if missing:
            top5 = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:5]
            cols_str = ', '.join(f"{c}({p}%)" for c, p in top5)
            print(f"  {fname:<30} {cols_str}")
            quality[fname] = missing
        else:
            print(f"  {fname:<30} (no missing data)")

    return results, quality


# ========== Section 5: Tasks ==========

DEFAULT_TASKS = [
    {"id": 1, "priority": "HIGH", "status": "TODO",
     "title": "JV-Link HR payout data extraction",
     "desc": "Extract trio/wide/umaren payouts from TFJV SE_DATA SH records for accurate ROI"},
    {"id": 2, "priority": "HIGH", "status": "TODO",
     "title": "Walk-forward with actual payouts",
     "desc": "Re-run WF backtest using real payouts instead of tansho odds estimates"},
    {"id": 3, "priority": "MID", "status": "TODO",
     "title": "Weather data integration",
     "desc": "Fetch JMA weather per course/date for track condition prediction"},
    {"id": 4, "priority": "MID", "status": "TODO",
     "title": "Track bias feature",
     "desc": "Compute inner/outer track bias per course/day from passing data"},
    {"id": 5, "priority": "MID", "status": "TODO",
     "title": "Jockey-Trainer combination stats",
     "desc": "Expanding window win rate for (jockey_id, trainer_id) pairs"},
    {"id": 6, "priority": "LOW", "status": "TODO",
     "title": "LINE notification integration",
     "desc": "Send bet recommendations via LINE Notify API"},
    {"id": 7, "priority": "LOW", "status": "TODO",
     "title": "Wood training horse_id format fix",
     "desc": "Match rate 0% due to horse_id format mismatch between CSVs"},
    {"id": 8, "priority": "LOW", "status": "TODO",
     "title": "NAR model v3 with more data sources",
     "desc": "Integrate additional NAR data beyond KDSCOPE/netkeiba"},
]


def section_tasks():
    print("\n  === 5. PROJECT TASKS ===\n")

    if os.path.exists(TASKS_PATH):
        with open(TASKS_PATH, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    else:
        tasks = DEFAULT_TASKS
        os.makedirs(os.path.dirname(TASKS_PATH), exist_ok=True)
        with open(TASKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        print(f"  (Created {TASKS_PATH} with default tasks)\n")

    # Sort by priority
    pri_order = {'HIGH': 0, 'MID': 1, 'LOW': 2}
    tasks.sort(key=lambda t: (pri_order.get(t.get('priority', 'LOW'), 9),
                               0 if t.get('status') == 'TODO' else 1))

    print(f"  {'#':<4} {'Pri':<5} {'Status':<8} {'Title':<45} Description")
    print("  " + "-" * 100)
    for t in tasks:
        tid = t.get('id', '?')
        pri = t.get('priority', '?')
        status = t.get('status', '?')
        title = t.get('title', '')[:44]
        desc = t.get('desc', '')[:50]
        print(f"  {tid:<4} {pri:<5} {status:<8} {title:<45} {desc}")

    return tasks


# ========== Section 6: Git Info ==========

def section_git():
    print("\n  === 6. GIT INFO ===\n")

    info = {}
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H|%ai|%s'],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split('|', 2)
            if len(parts) == 3:
                info['last_commit_hash'] = parts[0][:8]
                info['last_commit_date'] = parts[1]
                info['last_commit_msg'] = parts[2]
                print(f"  Last commit: {parts[0][:8]} ({parts[1]})")
                print(f"  Message:     {parts[2]}")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            info['remote_url'] = url
            print(f"  Remote:      {url}")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            info['branch'] = branch
            print(f"  Branch:      {branch}")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['git', 'status', '--short'],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if result.returncode == 0:
            changes = result.stdout.strip()
            n_changes = len([l for l in changes.split('\n') if l.strip()]) if changes else 0
            info['uncommitted_changes'] = n_changes
            if n_changes:
                print(f"  Uncommitted: {n_changes} file(s)")
            else:
                print(f"  Working tree clean")
    except Exception:
        pass

    return info


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description='KEIBA AI Project Status')
    parser.add_argument('--section', choices=['model', 'data', 'conditions', 'missing', 'tasks', 'git'],
                        help='Show only this section')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--export', action='store_true', help='Export to data/project_status_report.json')
    args = parser.parse_args()

    sections = args.section
    all_sections = not sections

    report = {'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    if not args.json:
        print("\n" + "=" * 70)
        print("  KEIBA AI - PROJECT STATUS")
        print(f"  Generated: {report['generated_at']}")
        print("=" * 70)

    if all_sections or sections == 'model':
        report['models'] = section_model()

    if all_sections or sections == 'data':
        report['data_assets'] = section_data()

    if all_sections or sections == 'conditions':
        report['conditions'] = section_conditions()

    if all_sections or sections == 'missing':
        missing_info, quality = section_missing()
        report['missing'] = missing_info
        report['data_quality'] = {k: v for k, v in quality.items()}

    if all_sections or sections == 'tasks':
        report['tasks'] = section_tasks()

    if all_sections or sections == 'git':
        report['git'] = section_git()

    if not args.json:
        print("\n" + "=" * 70)
        print("  END OF STATUS REPORT")
        print("=" * 70 + "\n")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    if args.export:
        export_path = os.path.join(DATA_DIR, 'project_status_report.json')
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        if not args.json:
            print(f"  Exported to: {export_path}")


if __name__ == '__main__':
    main()
