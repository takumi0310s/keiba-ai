"""V16 paper shadow tracker: V15 vs V16 ROI comparison with popularity-divergence detection.

V16 = オッズ系8件除外 ability model (WF 0.8677 ≈ V15 0.8678).
Tracks whether V16 picks up high-ability / low-popularity horses that V15 misses.

採用基準 (N≥30後判定):
  - 人気乖離レースでの V16 ROI > V15 ROI
  - TOP1 連対率 ≥ 40%
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[1]
V16_LOG_DIR = BASE_DIR / 'data' / 'v16_paper_log'

# A race is "popularity-divergent" when V15's top pick has lower pop_rank than V16's
# (i.e., V15 favors a popular horse but V16 prefers a different horse)
POPULARITY_DIVERGE_THRESHOLD = 2  # |pop_rank difference| for top pick to count as divergent


def _log_path(date_str: Optional[str] = None) -> Path:
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    return V16_LOG_DIR / f'v16_paper_{date_str}.jsonl'


def log_race(
    race_id: str,
    v15_top: list[dict],   # [{'horse_num': int, 'score': float, 'pop_rank': int}, ...]
    v16_top: list[dict],   # same structure
    result_top3: Optional[list[int]] = None,   # actual finishing horse_nums 1-3
    payout_trio: Optional[int] = None,         # actual trio payout (yen, 0 if miss)
    date_str: Optional[str] = None,
) -> dict:
    """Log one race comparison entry to JSONL."""
    os.makedirs(V16_LOG_DIR, exist_ok=True)

    v15_top1_num = v15_top[0]['horse_num'] if v15_top else None
    v16_top1_num = v16_top[0]['horse_num'] if v16_top else None

    v15_pop1 = v15_top[0].get('pop_rank') if v15_top else None
    v16_pop1 = v16_top[0].get('pop_rank') if v16_top else None

    popularity_divergent = False
    if v15_pop1 is not None and v16_pop1 is not None and v15_top1_num != v16_top1_num:
        popularity_divergent = abs(v15_pop1 - v16_pop1) >= POPULARITY_DIVERGE_THRESHOLD

    v15_top3_nums = [h['horse_num'] for h in v15_top[:3]]
    v16_top3_nums = [h['horse_num'] for h in v16_top[:3]]

    v15_hit = v16_hit = None
    if result_top3 is not None:
        v15_hit = set(v15_top3_nums) == set(result_top3[:3]) if len(result_top3) >= 3 else False
        v16_hit = set(v16_top3_nums) == set(result_top3[:3]) if len(result_top3) >= 3 else False

    rec = {
        'race_id': str(race_id),
        'ts': datetime.now().isoformat(),
        'v15': {
            'top1': v15_top1_num,
            'top3': v15_top3_nums,
            'pop_rank_top1': v15_pop1,
        },
        'v16': {
            'top1': v16_top1_num,
            'top3': v16_top3_nums,
            'pop_rank_top1': v16_pop1,
        },
        'popularity_divergent': popularity_divergent,
        'result_top3': result_top3,
        'payout_trio': payout_trio,
        'v15_hit': v15_hit,
        'v16_hit': v16_hit,
    }

    with open(_log_path(date_str), 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return rec


def compute_stats(days: int = 90) -> dict:
    """Compute V15 vs V16 comparison stats from logs.

    Returns dict with:
      n_total, n_divergent, v15_roi, v16_roi,
      v16_roi_divergent, v15_top1_hit, v16_top1_hit
    """
    cutoff = datetime.now() - timedelta(days=days)
    records = []

    if not V16_LOG_DIR.exists():
        return {}

    for p in sorted(V16_LOG_DIR.glob('v16_paper_*.jsonl')):
        try:
            ds = p.stem.replace('v16_paper_', '')
            fd = datetime.strptime(ds, '%Y%m%d')
        except Exception:
            continue
        if fd < cutoff:
            continue
        try:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

    if not records:
        return {'n_total': 0}

    n = len(records)
    settled = [r for r in records if r.get('payout_trio') is not None]
    divergent = [r for r in records if r.get('popularity_divergent')]
    div_settled = [r for r in divergent if r.get('payout_trio') is not None]

    def roi(recs):
        if not recs:
            return None
        total_bet = len(recs) * 700
        v15_return = sum(r['payout_trio'] if r.get('v15_hit') else 0 for r in recs)
        v16_return = sum(r['payout_trio'] if r.get('v16_hit') else 0 for r in recs)
        return {
            'v15': round(v15_return / total_bet * 100, 1) if total_bet else None,
            'v16': round(v16_return / total_bet * 100, 1) if total_bet else None,
            'n': len(recs),
        }

    overall = roi(settled)
    divergent_roi = roi(div_settled)

    v15_top1_hits = sum(1 for r in settled if r.get('result_top3') and r['v15']['top1'] in (r['result_top3'] or []))
    v16_top1_hits = sum(1 for r in settled if r.get('result_top3') and r['v16']['top1'] in (r['result_top3'] or []))
    n_settled = len(settled)

    return {
        'n_total': n,
        'n_settled': n_settled,
        'n_divergent': len(divergent),
        'n_divergent_settled': len(div_settled),
        'overall': overall,
        'divergent': divergent_roi,
        'v15_top1_hit_rate': round(v15_top1_hits / n_settled * 100, 1) if n_settled else None,
        'v16_top1_hit_rate': round(v16_top1_hits / n_settled * 100, 1) if n_settled else None,
        'adoption_ready': n_settled >= 30,
    }


def print_report(days: int = 90) -> None:
    stats = compute_stats(days)
    if not stats or stats.get('n_total', 0) == 0:
        print('[V16 paper] No data yet.')
        return

    print(f"\n=== V16 Paper Shadow Report (last {days}d) ===")
    print(f"  N total: {stats['n_total']}  |  settled: {stats['n_settled']}")
    print(f"  Divergent races: {stats['n_divergent']} (settled: {stats['n_divergent_settled']})")

    ov = stats.get('overall') or {}
    print(f"\n  Overall ROI — V15: {ov.get('v15')}%  V16: {ov.get('v16')}%  (N={ov.get('n')})")

    dv = stats.get('divergent') or {}
    print(f"  Divergent ROI — V15: {dv.get('v15')}%  V16: {dv.get('v16')}%  (N={dv.get('n')})")

    print(f"\n  V15 top1 hit rate: {stats.get('v15_top1_hit_rate')}%")
    print(f"  V16 top1 hit rate: {stats.get('v16_top1_hit_rate')}%")

    if stats.get('adoption_ready'):
        dv_v15 = dv.get('v15') or 0
        dv_v16 = dv.get('v16') or 0
        v16_top1 = stats.get('v16_top1_hit_rate') or 0
        go = dv_v16 > dv_v15 and v16_top1 >= 40.0
        print(f"\n  ★ Adoption check (N≥30): {'GO' if go else 'NO-GO'}")
        print(f"    divergent ROI V16>V15: {dv_v16 > dv_v15}  top1≥40%: {v16_top1 >= 40.0}")
    else:
        need = 30 - stats.get('n_settled', 0)
        print(f"\n  Adoption check: not ready (need {need} more settled races)")


if __name__ == '__main__':
    print_report()
