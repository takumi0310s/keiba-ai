"""
P0-5 採用判定 evaluation script (template、 6/17 確定値で update)

usage:
  python tools/p0_5_evaluation.py --start 20260518 --end 20260616

出力:
  docs/P0_5_ADOPTION_VERDICT_2026_06_17.md (5 項目評価 + GO/NO-GO 判定)

★ template、 6/17 当日 実 data shape 確定後に fine-tuning 想定 ★
★ V15 production を 1 byte も変更しない (read-only) ★
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
CUM_CSV = REPO / 'data' / 'cumulative_results.csv'
RECALC_DIR = REPO / 'data' / 'recalc_15min'
DAILY_PRED_DIR = REPO / 'data' / 'daily_predictions'
DAILY_RES_DIR = REPO / 'data' / 'daily_results'
OUT_MD = REPO / 'docs' / 'P0_5_ADOPTION_VERDICT_2026_06_17.md'

BASELINE_AUC = 0.8939
AUC_TOLERANCE = 0.001


def load_period_data(start_date: str, end_date: str):
    """期間内 V15 production + P0-5 paper data load.

    Returns
    -------
    cum : pd.DataFrame
        V15 production settled rows (status=='settled')
    paper_data : list[dict]
        P0-5 paper data (recalc_15min/{date}/{race_id}.json)
    """
    # ---- V15 production ----
    cum = pd.DataFrame()
    if CUM_CSV.exists():
        cum = pd.read_csv(CUM_CSV, dtype={'race_id': str})
        if 'status' in cum.columns:
            cum = cum[cum['status'] == 'settled']
        # date filter (★ format 確認、 cum 内 date column 名前は session 依存 ★)
        date_col = None
        for cand in ('date', 'race_date', 'predict_date'):
            if cand in cum.columns:
                date_col = cand
                break
        if date_col is not None:
            cum[date_col] = cum[date_col].astype(str).str.replace('-', '', regex=False)
            cum = cum[(cum[date_col] >= start_date) & (cum[date_col] <= end_date)]

    # ---- P0-5 paper ----
    paper_data: list[dict] = []
    if RECALC_DIR.exists():
        # 期間 filter: {start_date} <= subdir name <= {end_date}
        for subdir in sorted(RECALC_DIR.iterdir()):
            if not subdir.is_dir():
                continue
            name = subdir.name.replace('-', '')
            if not (start_date <= name <= end_date):
                continue
            for f in sorted(subdir.glob('*.json')):
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        d = json.load(fp)
                    d.setdefault('_source_file', str(f.relative_to(REPO)))
                    paper_data.append(d)
                except Exception as exc:  # noqa: BLE001
                    paper_data.append({
                        '_source_file': str(f.relative_to(REPO)),
                        'status': 'parse_error',
                        'error': str(exc),
                    })
    return cum, paper_data


def evaluate_auc(cum: pd.DataFrame, paper_data: list[dict]) -> dict:
    """AUC 維持確認 (V15 0.8939 ± 0.001).

    ★ paper eval は overlay 適用後の prob、 AUC は overall AUC ではなく
       overlay 適用後 ranking と finish 真値の AUC を 計算 ★
    """
    # ---- placeholder: 6/17 当日 paper_data の schema 確定後に実装 ----
    # 想定: 各 d in paper_data に recalc_ranking + finish ground truth がある
    p0_5_auc = None
    note = 'TODO: paper_data schema 確定後に実装 (recalc_ranking + finish 真値 で sklearn roc_auc_score)'

    delta = (p0_5_auc - BASELINE_AUC) if p0_5_auc is not None else None
    return {
        'production_auc': BASELINE_AUC,
        'p0_5_auc': p0_5_auc,
        'delta': delta,
        'pass': (delta is not None) and (abs(delta) < AUC_TOLERANCE),
        'note': note,
    }


def evaluate_roi(cum: pd.DataFrame, paper_data: list[dict]) -> dict:
    """paper ROI 改善 (V15 production 上回り)."""
    # ---- V15 production ROI ----
    v15_roi = None
    if not cum.empty and {'actual_payout', 'investment'}.issubset(cum.columns):
        total_inv = cum['investment'].fillna(0).sum()
        total_payout = cum['actual_payout'].fillna(0).sum()
        if total_inv > 0:
            v15_roi = (total_payout / total_inv) * 100.0

    # ---- P0-5 paper ROI (★ recalc 結果 から 仮想 bet simulation ★) ----
    p0_5_roi = None
    note = 'TODO: paper_data の recalc_ranking で simulate された投資・配当 sum で 計算'

    delta = (p0_5_roi - v15_roi) if (p0_5_roi is not None and v15_roi is not None) else None
    return {
        'v15_roi': v15_roi,
        'p0_5_roi': p0_5_roi,
        'delta': delta,
        'pass': (delta is not None) and (delta > 0),
        'note': note,
    }


def evaluate_leak(cum: pd.DataFrame, paper_data: list[dict]) -> dict:
    """LEAK 監査 PASS (v15_2_train_gate exit 0)."""
    gate = REPO / 'tools' / 'v15_2_train_gate.py'
    features = REPO / 'data' / 'v15_2' / 'features_v152_candidates.txt'
    if not gate.exists():
        return {'pass': False, 'reason': f'gate not found: {gate}', 'exit_code': None}
    if not features.exists():
        return {'pass': False, 'reason': f'features not found: {features}', 'exit_code': None}

    try:
        result = subprocess.run(
            [sys.executable, str(gate), '--features', str(features)],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as exc:  # noqa: BLE001
        return {'pass': False, 'reason': f'subprocess error: {exc}', 'exit_code': None}

    return {
        'exit_code': result.returncode,
        'pass': result.returncode == 0,
        'stdout': result.stdout[-500:],
        'stderr': result.stderr[-500:],
    }


def evaluate_live_stability(paper_data: list[dict]) -> dict:
    """LIVE 安定 (90%+ 成功率、 4 週末 24-32 R 想定)."""
    total = len(paper_data)
    if total == 0:
        return {'success_rate': 0.0, 'pass': False, 'total': 0, 'success': 0}
    success = sum(1 for d in paper_data if d.get('status') == 'ok')
    rate = success / total
    return {
        'total': total,
        'success': success,
        'success_rate': rate,
        'pass': rate >= 0.90,
    }


def _race_level_roi(cum: pd.DataFrame) -> np.ndarray:
    """V15 production R-level ROI samples を抽出."""
    if cum.empty or not {'race_id', 'actual_payout', 'investment'}.issubset(cum.columns):
        return np.array([])
    grp = cum.groupby('race_id').agg(
        inv=('investment', 'sum'),
        pay=('actual_payout', 'sum'),
    )
    grp = grp[grp['inv'] > 0]
    return ((grp['pay'] / grp['inv']) * 100.0).to_numpy()


def _paper_race_level_roi(paper_data: list[dict]) -> np.ndarray:
    """P0-5 paper R-level ROI samples を抽出 (★ schema 確定後に実装 ★)."""
    # placeholder: paper_data 内 race-level investment / payout が必要
    return np.array([])


def evaluate_statistical(cum: pd.DataFrame, paper_data: list[dict]) -> dict:
    """Welch's t-test + paired bootstrap 1000 iter、 p<0.05."""
    v15_samples = _race_level_roi(cum)
    p0_5_samples = _paper_race_level_roi(paper_data)

    if len(v15_samples) < 5 or len(p0_5_samples) < 5:
        return {
            'p_value': None,
            'pass': False,
            'reason': f'sample too small (v15={len(v15_samples)}, p0_5={len(p0_5_samples)})',
            'note': 'TODO: paper R-level ROI 計算実装後に再評価',
        }

    t_stat, p_value = stats.ttest_ind(v15_samples, p0_5_samples, equal_var=False)

    # bootstrap 95% CI on mean delta
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(1000):
        v_s = rng.choice(v15_samples, size=len(v15_samples), replace=True)
        p_s = rng.choice(p0_5_samples, size=len(p0_5_samples), replace=True)
        deltas.append(float(np.mean(p_s) - np.mean(v_s)))
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'ci_95_low': float(ci_low),
        'ci_95_high': float(ci_high),
        'mean_v15': float(np.mean(v15_samples)),
        'mean_p0_5': float(np.mean(p0_5_samples)),
        'n_v15': int(len(v15_samples)),
        'n_p0_5': int(len(p0_5_samples)),
        'pass': bool(p_value < 0.05),
    }


def write_verdict_md(results: dict, judgment: dict) -> None:
    """採用判定 verdict md 出力."""
    lines = []
    lines.append(f"# P0-5 採用判定 verdict ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    lines.append("")
    lines.append(f"## 判定: {judgment['decision']}")
    lines.append("")
    lines.append(f"- PASS 数: {judgment['pass_count']} / 5")
    lines.append(f"- 推奨 next action: {judgment['next_action']}")
    lines.append("")
    lines.append("## 5 項目評価")
    for k, v in results.items():
        icon = "[PASS]" if v.get('pass') else "[FAIL]"
        lines.append("")
        lines.append(f"### {icon} {k}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(v, indent=2, ensure_ascii=False, default=str))
        lines.append("```")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='20260518', help='start date YYYYMMDD')
    ap.add_argument('--end', default='20260616', help='end date YYYYMMDD')
    args = ap.parse_args()

    cum, paper_data = load_period_data(args.start, args.end)

    results = {
        '1_auc_maintained': evaluate_auc(cum, paper_data),
        '2_roi_improved': evaluate_roi(cum, paper_data),
        '3_leak_passed': evaluate_leak(cum, paper_data),
        '4_live_stability': evaluate_live_stability(paper_data),
        '5_statistical_significance': evaluate_statistical(cum, paper_data),
    }

    pass_count = sum(1 for v in results.values() if v.get('pass'))
    if pass_count == 5:
        decision = 'GO'
        next_action = '6/18+ production 投入候補 (別 sub-task で race_auto_notify 統合設計)'
    elif pass_count >= 3:
        decision = 'ACCUMULATE'
        next_action = '5/18-7/15 蓄積継続、 7/16 再判定'
    else:
        decision = 'NO-GO'
        next_action = '実装見直し or 永久放棄判定'

    judgment = {
        'decision': decision,
        'pass_count': pass_count,
        'next_action': next_action,
        'period': f'{args.start}-{args.end}',
        'n_paper': len(paper_data),
        'n_cum_settled': int(len(cum)) if cum is not None else 0,
    }

    write_verdict_md(results, judgment)
    print(json.dumps(
        {'results': results, 'judgment': judgment},
        indent=2,
        ensure_ascii=False,
        default=str,
    ))

    return 0 if pass_count == 5 else 1


if __name__ == "__main__":
    sys.exit(main())
