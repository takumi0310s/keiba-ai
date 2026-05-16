"""features integrity monitor — V15 cache 145 features 自動 audit

TYB merge bug 級事故 (1 年以上 検出されず TYB が実質ゼロ寄与) の永久防止 monitor。
data/_v15_optuna_df_cache.pkl.gz の 145 features を read-only audit、 red flag を
Discord に警告通知する。

★ V15 production 完全不変、 read-only ★
★ V15 .pkl.gz / predict_core.py / cumulative_results.csv は一切 modify しない ★

usage:
    python tools/features_integrity_monitor.py            # full run + Discord notify
    python tools/features_integrity_monitor.py --check-only  # local check only
    python tools/features_integrity_monitor.py --no-discord  # skip Discord

判定 criteria:
    RED_CONSTANT       : unique <= 1 (全行同値、 TYB 級事故候補)
    RED_LOW_UNIQUE     : 2 <= unique <= 10 (low cardinality、 default filler 疑い)
    RED_IMP_BUT_CONST  : lgb_gain > 0 かつ unique <= 1 (model に入るが分散なし、 最大警告)
    WARN_HIGH_NULL     : null_rate > 50% (取得失敗濃厚)
    WARN_QUASI_CONST   : most_common_rate > 95% (実質 constant)
    INFO_UNUSED        : unique > 100 かつ lgb_gain == 0 かつ xgb_gain == 0

output:
    data/T1_features_audit_<YYYY_MM_DD>.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

CACHE_PATH = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz')

# 既知 red flag (5/17 audit 時点): documented separately to suppress repeat noise
KNOWN_RED_CONSTANT_FEATURES = {
    'is_nar',              # JRA 専用 cache のため 0 で固定 (intentional)
    'prev_odds_log',       # cache 内で全行 fill default 2.7726 (log(16))、 LEAK 除去後の残骸
    'prev_race_first3f',   # cache 内で全行 default 35.8s
    'prev_race_last3f',    # cache 内で全行 default 36.5s
    'prev_race_pace_diff', # 0.0 default
    'sire_shinba_top3r',   # 0.22 default、 新馬戦評価
    'pci',                 # 1.0195 default、 ペースチェンジ指数
    'gaisha_rank',         # 0 default、 既に 4/26 audit で死特徴量確認済
}


def load_cache():
    with gzip.open(CACHE_PATH, 'rb') as f:
        return pickle.load(f)


def load_v15_importance():
    with gzip.open(MODEL_PATH, 'rb') as f:
        m = pickle.load(f)
    feats = m['features']
    try:
        lgb_imp = m['model'].feature_importance(importance_type='gain')
    except Exception:
        lgb_imp = [0.0] * len(feats)
    try:
        xgb_imp_raw = m['xgb_model'].get_score(importance_type='gain')
        # XGB は f0, f1, ... の positional 形式で返すことがあるため両対応
        xgb_imp_d = {}
        for k, v in xgb_imp_raw.items():
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feats):
                    xgb_imp_d[feats[idx]] = float(v)
            else:
                xgb_imp_d[k] = float(v)
    except Exception:
        xgb_imp_d = {}
    imp = {}
    for i, f in enumerate(feats):
        imp[f] = {
            'lgb_gain': float(lgb_imp[i]) if i < len(lgb_imp) else 0.0,
            'xgb_gain': float(xgb_imp_d.get(f, 0.0)),
        }
    return imp, feats


def audit_features(df: pd.DataFrame, v15_features, importance_map):
    audit = []
    for col in v15_features:
        if col not in df.columns:
            audit.append({'feature': col, 'status': 'MISSING', 'flags': ['MISSING']})
            continue
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        try:
            vc = s.value_counts(normalize=True, dropna=False)
            mcv = str(vc.index[0]) if len(vc) else None
            mcr = float(vc.iloc[0]) if len(vc) else None
        except Exception:
            mcv = None
            mcr = None
        rec = {
            'feature': col,
            'dtype': str(s.dtype),
            'unique': int(s.nunique(dropna=True)),
            'null_rate': float(s.isna().mean()),
            'most_common_value': mcv,
            'most_common_rate': mcr,
            'lgb_gain': importance_map.get(col, {}).get('lgb_gain', 0.0),
            'xgb_gain': importance_map.get(col, {}).get('xgb_gain', 0.0),
        }
        if is_num:
            try:
                rec['min'] = float(s.min())
                rec['max'] = float(s.max())
                rec['mean'] = float(s.mean())
                rec['std'] = float(s.std())
            except Exception:
                pass
        flags = []
        if rec['unique'] <= 1:
            flags.append('RED_CONSTANT')
        elif rec['unique'] <= 10:
            flags.append('RED_LOW_UNIQUE')
        if rec['null_rate'] > 0.5:
            flags.append('WARN_HIGH_NULL')
        if rec['most_common_rate'] is not None and rec['most_common_rate'] > 0.95:
            flags.append('WARN_QUASI_CONSTANT')
        if rec['lgb_gain'] > 0 and rec['unique'] <= 1:
            flags.append('RED_IMP_BUT_CONST')
        if rec['unique'] > 100 and rec['lgb_gain'] == 0 and rec['xgb_gain'] == 0:
            flags.append('INFO_UNUSED')
        rec['flags'] = flags
        rec['known'] = col in KNOWN_RED_CONSTANT_FEATURES
        audit.append(rec)
    return audit


def summarize(audit):
    summary = {
        'total': len(audit),
        'missing': [r for r in audit if 'MISSING' in r.get('flags', [])],
        'red_constant_all': [r for r in audit if 'RED_CONSTANT' in r['flags']],
        'red_constant_new': [r for r in audit
                             if 'RED_CONSTANT' in r['flags'] and not r.get('known', False)],
        'red_low_unique': [r for r in audit if 'RED_LOW_UNIQUE' in r['flags']],
        'red_imp_but_const': [r for r in audit if 'RED_IMP_BUT_CONST' in r['flags']],
        'warn_high_null': [r for r in audit if 'WARN_HIGH_NULL' in r['flags']],
        'warn_quasi_constant': [r for r in audit if 'WARN_QUASI_CONSTANT' in r['flags']],
    }
    return summary


def format_discord_message(summary):
    lines = []
    lines.append('**T1 Features Integrity Audit**')
    lines.append(f"  total features: {summary['total']}")
    lines.append(f"  RED_CONSTANT (unique<=1): {len(summary['red_constant_all'])} "
                 f"(new={len(summary['red_constant_new'])})")
    lines.append(f"  RED_IMP_BUT_CONST (model入力but分散なし): {len(summary['red_imp_but_const'])}")
    lines.append(f"  RED_LOW_UNIQUE (2-10): {len(summary['red_low_unique'])}")
    lines.append(f"  WARN_HIGH_NULL (>50%): {len(summary['warn_high_null'])}")
    lines.append(f"  WARN_QUASI_CONSTANT (mcr>95%): {len(summary['warn_quasi_constant'])}")
    if summary['red_imp_but_const']:
        lines.append('\n**CRITICAL (importance>0 but constant):**')
        for r in summary['red_imp_but_const'][:10]:
            lines.append(f"  - {r['feature']} (lgb_gain={r['lgb_gain']:.1f})")
    if summary['red_constant_new']:
        lines.append('\n**NEW RED_CONSTANT (not in known list):**')
        for r in summary['red_constant_new'][:10]:
            lines.append(f"  - {r['feature']} (value={r.get('most_common_value')})")
    return '\n'.join(lines)


def notify_discord(message: str):
    try:
        import requests
    except Exception:
        print('requests not available, skipping Discord')
        return False
    url = (os.environ.get('DISCORD_WEBHOOK_UPDATES')
           or os.environ.get('DISCORD_WEBHOOK_URL'))
    if not url:
        print('No DISCORD_WEBHOOK_UPDATES / DISCORD_WEBHOOK_URL env set, skipping')
        return False
    try:
        r = requests.post(url, json={'content': message}, timeout=10)
        return r.status_code in (200, 204)
    except Exception as e:
        print(f'Discord notify failed: {e}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true',
                        help='Skip JSON save, print summary only')
    parser.add_argument('--no-discord', action='store_true',
                        help='Skip Discord notification')
    args = parser.parse_args()

    print('[features_integrity_monitor] Loading V15 cache...')
    cache = load_cache()
    df = cache['df']
    v15_features = cache['features']
    print(f'  df shape: {df.shape}, V15 features: {len(v15_features)}')

    print('[features_integrity_monitor] Loading V15 importance...')
    imp_map, model_feats = load_v15_importance()
    print(f'  V15 model features: {len(model_feats)}')

    print('[features_integrity_monitor] Auditing...')
    audit = audit_features(df, v15_features, imp_map)
    summary = summarize(audit)

    print('\n=== SUMMARY ===')
    print(f"total: {summary['total']}")
    print(f"RED_CONSTANT total: {len(summary['red_constant_all'])} "
          f"(known={len(summary['red_constant_all']) - len(summary['red_constant_new'])}, "
          f"new={len(summary['red_constant_new'])})")
    print(f"RED_IMP_BUT_CONST (critical): {len(summary['red_imp_but_const'])}")
    print(f"RED_LOW_UNIQUE: {len(summary['red_low_unique'])}")
    print(f"WARN_HIGH_NULL: {len(summary['warn_high_null'])}")
    print(f"WARN_QUASI_CONSTANT: {len(summary['warn_quasi_constant'])}")

    if not args.check_only:
        today = datetime.now().strftime('%Y_%m_%d')
        out_path = os.path.join(DATA_DIR, f'T1_features_audit_{today}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
        print(f'\nSaved audit JSON: {out_path}')

    # Discord notify if RED_IMP_BUT_CONST or NEW RED_CONSTANT
    has_critical = (len(summary['red_imp_but_const']) > 0
                    or len(summary['red_constant_new']) > 0
                    or len(summary['missing']) > 0)
    if has_critical and not args.no_discord:
        msg = format_discord_message(summary)
        ok = notify_discord(msg)
        print(f'Discord notified: {ok}')
    elif has_critical:
        print('CRITICAL detected but --no-discord set')
    else:
        print('No critical issues (only known red flags)')

    # exit code: 0 always (read-only audit, do not block pipeline)
    return 0


if __name__ == '__main__':
    sys.exit(main())
