#!/usr/bin/env python
"""データヘルスチェック

1. 全CSVのヘッダ整合性（期待列数/列名）
2. 年別行数カバレッジ（想定範囲を満たしているか）
3. v16再評価トリガー判定（upset > 50%, training_eval > 40%, master_index > 30%）
4. 異常時に Discord 赤警告 + data/data_health_report.json に結果保存

Usage:
    python tools/data_health_check.py              # フルチェック
    python tools/data_health_check.py --notify     # Discord にも送信
    python tools/data_health_check.py --strict     # 警告も exit 1
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORT_PATH = os.path.join(DATA_DIR, 'data_health_report.json')

# 期待スキーマ定義 (file: {cols: expected_columns, min_total: N, years_required: list})
EXPECTED = {
    'netkeiba_upset_level.csv': {
        'cols': ['race_id', 'upset_level', 'top_popularity_reliability'],
        'years_required': ['2020', '2021', '2022', '2023', '2024', '2025'],
        'min_per_year': 100,
    },
    'netkeiba_track_bias.csv': {
        'cols_contains': ['race_id', 'track_index'],
        'years_required': ['2020', '2021', '2022', '2025'],
        'min_per_year': 500,
    },
    'netkeiba_race_lap.csv': {
        'cols_contains': ['race_id', 'lap_times'],
        'years_required': ['2020', '2021', '2022', '2025'],
        'min_per_year': 500,
    },
    'netkeiba_training_eval.csv': {
        'cols': ['race_id', 'umaban', 'horse_name', 'prev_review', 'training_date',
                 'training_course', 'training_condition', 'training_rider',
                 'training_time_raw', 'training_position', 'training_intensity',
                 'training_move', 'training_rank'],
        'years_required': ['2024', '2025'],
        'min_per_year': 10000,
    },
    'netkeiba_master_index.csv': {
        'cols': ['race_id', 'umaban', 'horse_name', 'finish_order', 'time_index',
                 'master_index', 'start_index', 'chase_index', 'agari_index'],
        'years_required': ['2025'],
        'min_per_year': 5000,
    },
    'netkeiba_race_review.csv': {
        'cols_contains': ['race_id', 'umaban', 'review_score'],
        'years_required': ['2020', '2021', '2022', '2023', '2024', '2025'],
        'min_per_year': 30000,
    },
    'netkeiba_speed_index.csv': {
        'cols_contains': ['race_id'],
        'years_required': ['2020', '2021', '2022', '2023', '2024', '2025'],
        'min_per_year': 30000,
    },
}

# v16再評価トリガー閾値 (race数ベース, v15学習期間2010-2025で1年あたり~3500レース想定)
JRA_YEARLY_RACES_APPROX = 3500
V16_THRESHOLDS = {
    'netkeiba_upset_level.csv': 0.50,       # 50% -> 各年1750レース/全期間10500
    'netkeiba_training_eval.csv': 0.40,     # 40%
    'netkeiba_master_index.csv': 0.30,      # 30%
}


def _load_head_cols(path):
    import pandas as pd
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', nrows=0)
        return list(df.columns)
    except Exception as e:
        return None


def _year_counts(path):
    import pandas as pd
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', usecols=['race_id'], dtype=str, low_memory=False)
        yr = df['race_id'].astype(str).str[:4].value_counts().to_dict()
        return len(df), {y: int(yr.get(y, 0)) for y in ['2020','2021','2022','2023','2024','2025','2026']}
    except Exception:
        return 0, {}


def check_header(path, expected):
    cols = _load_head_cols(path)
    issues = []
    if cols is None:
        return ['ヘッダ読み込み失敗'], cols
    if 'cols' in expected:
        exp = expected['cols']
        if len(cols) != len(exp):
            issues.append(f"列数不一致: {len(cols)} != {len(exp)}")
        missing = [c for c in exp if c not in cols]
        extra = [c for c in cols if c not in exp]
        if missing:
            issues.append(f"欠落列: {missing}")
        if extra:
            issues.append(f"余分列: {extra}")
    elif 'cols_contains' in expected:
        missing = [c for c in expected['cols_contains'] if c not in cols]
        if missing:
            issues.append(f"必須列欠落: {missing}")
    return issues, cols


def check_coverage(path, expected):
    total, years = _year_counts(path)
    issues = []
    warnings = []
    min_y = expected.get('min_per_year', 0)
    for y in expected.get('years_required', []):
        cnt = years.get(y, 0)
        if cnt == 0:
            issues.append(f"year {y} 全欠落")
        elif cnt < min_y:
            warnings.append(f"year {y}: {cnt} < min {min_y}")
    return issues, warnings, total, years


def check_v16_triggers(cov_map):
    """v16再評価トリガー条件を評価"""
    triggers = {}
    # 5年間(2020-2024) x ~3500レース = 17500レースを満点として%算出
    baseline = JRA_YEARLY_RACES_APPROX * 5
    for fn, threshold in V16_THRESHOLDS.items():
        total = cov_map.get(fn, {}).get('total', 0)
        # upset_level / training_eval は race-level、master_index は horse-level
        # 簡易にtotal / (baseline * multiplier) でカバレッジ推定
        if 'upset' in fn:
            coverage_pct = total / baseline
        elif 'training_eval' in fn or 'master_index' in fn:
            coverage_pct = total / (baseline * 14)  # horse数 ~14頭/レース
        else:
            coverage_pct = total / baseline
        triggers[fn] = {
            'total': total,
            'coverage_pct': round(coverage_pct * 100, 1),
            'threshold_pct': round(threshold * 100, 1),
            'met': coverage_pct >= threshold,
        }
    all_met = all(t['met'] for t in triggers.values())
    return all_met, triggers


def run():
    report = {
        'timestamp': datetime.now().isoformat(),
        'files': {},
        'errors': [],
        'warnings': [],
    }
    cov_map = {}
    for fn, exp in EXPECTED.items():
        p = os.path.join(DATA_DIR, fn)
        entry = {'exists': os.path.exists(p)}
        if not entry['exists']:
            entry['status'] = 'MISSING'
            report['files'][fn] = entry
            report['errors'].append(f"{fn}: ファイル不在")
            continue
        h_issues, cols = check_header(p, exp)
        c_issues, c_warns, total, years = check_coverage(p, exp)
        entry.update({
            'columns': cols,
            'total': total,
            'years': years,
            'header_issues': h_issues,
            'coverage_issues': c_issues,
            'coverage_warnings': c_warns,
            'status': 'OK' if (not h_issues and not c_issues) else 'ERR',
        })
        if h_issues:
            report['errors'].append(f"{fn}: ヘッダ {h_issues}")
        if c_issues:
            report['errors'].append(f"{fn}: カバレッジ {c_issues}")
        if c_warns:
            report['warnings'].append(f"{fn}: {c_warns}")
        cov_map[fn] = {'total': total, 'years': years}
        report['files'][fn] = entry

    # v16再評価トリガー
    v16_ok, v16_triggers = check_v16_triggers(cov_map)
    report['v16_reeval_triggers'] = v16_triggers
    report['v16_reeval_ready'] = v16_ok

    return report


def render_report(report):
    lines = []
    lines.append(f"=== data health check {report['timestamp']} ===")
    for fn, e in report['files'].items():
        status = e.get('status', '?')
        total = e.get('total', 0)
        lines.append(f"  [{status}] {fn:<35} total={total}")
        for w in e.get('header_issues', []):
            lines.append(f"    ! header: {w}")
        for w in e.get('coverage_issues', []):
            lines.append(f"    ! cov   : {w}")
        for w in e.get('coverage_warnings', []):
            lines.append(f"    ~ warn  : {w}")
    lines.append('')
    lines.append(f"  errors: {len(report['errors'])}, warnings: {len(report['warnings'])}")
    lines.append(f"  v16 reeval ready: {'YES' if report['v16_reeval_ready'] else 'NO'}")
    for fn, t in report['v16_reeval_triggers'].items():
        mark = 'OK' if t['met'] else 'NG'
        lines.append(f"    [{mark}] {fn}: {t['coverage_pct']}% (threshold {t['threshold_pct']}%)")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notify', action='store_true')
    ap.add_argument('--strict', action='store_true', help='警告でもexit 1')
    args = ap.parse_args()

    report = run()
    text = render_report(report)
    print(text)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nreport saved: {REPORT_PATH}")

    if args.notify:
        try:
            from notify import send_discord
            if report['errors']:
                send_discord(
                    f"🔴 data health ERR ({len(report['errors'])})",
                    "\n".join(report['errors'][:10]),
                    color='red', channel='updates',
                )
            elif report['warnings']:
                send_discord(
                    f"🟡 data health WARN ({len(report['warnings'])})",
                    "\n".join(str(w) for w in report['warnings'][:10]),
                    color='yellow', channel='updates',
                )
            else:
                send_discord(
                    "✅ data health OK",
                    f"v16 reeval ready: {'YES' if report['v16_reeval_ready'] else 'NO'}",
                    color='green', channel='updates',
                )
            if report['v16_reeval_ready']:
                send_discord(
                    "🎯 v16 再評価可能",
                    "カバレッジ閾値を全て満たしました。\n"
                    "`python train/retrain_v16.py` で再学習を開始できます。",
                    color='green', channel='updates',
                )
        except Exception as e:
            print(f"[WARN] notify failed: {e}")

    has_err = bool(report['errors'])
    has_warn = bool(report['warnings'])
    if has_err or (args.strict and has_warn):
        sys.exit(1)


if __name__ == '__main__':
    main()
