"""V15 market divergence v2 (paper only, B-2 強化版)

★ paper only — 実 cash は V15 のみ ★
★ B-2 既実装の強化版 ★

divergence score 計算:
  V15 top1 予測馬の市場順位 (pop_rank_top1) から divergence を数値化:

  pop_rank_top1 = 1番人気 → divergence_score = 0.0 (完全一致、 skip 候補)
  pop_rank_top1 = 2番人気 → divergence_score = 0.3
  pop_rank_top1 = 3番人気 → divergence_score = 0.6
  pop_rank_top1 = 4番人気 → divergence_score = 0.8
  pop_rank_top1 = 5番人気以下 → divergence_score = 1.0 (最大乖離、 boost 候補)

  action:
  - score < 0.3  → skip (V15 と市場が一致しすぎ = 旨味なし)
  - score 0.3-0.7 → standard bet
  - score > 0.7  → boost bet (同点数 formation、 ただし EV 期待↑)

使い方:
  python tools/v15_market_divergence_v2.py [--csv PATH] [--thresholds 0.3,0.5,0.7]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / 'data' / 'cumulative_results.csv'

# pop_rank → divergence_score テーブル
_SCORE_TABLE: dict[int, float] = {
    1: 0.0,
    2: 0.3,
    3: 0.6,
    4: 0.8,
}
_SCORE_DEFAULT = 1.0  # 5番人気以下


def compute_divergence_score(pop_rank_top1: int) -> float:
    """top1 予測馬の市場人気順位から divergence score を算出。

    Args:
        pop_rank_top1: 市場人気順位 (1 = 1番人気)。
                       0 以下または不明な場合は 1.0 を返す (conservative boost)。

    Returns:
        float: 0.0 (一致) ～ 1.0 (最大乖離)
    """
    if pop_rank_top1 <= 0:
        return _SCORE_DEFAULT
    return _SCORE_TABLE.get(pop_rank_top1, _SCORE_DEFAULT)


def classify_action(score: float, threshold: float = 0.3) -> str:
    """divergence_score から action を分類。

    Returns:
        'skip'     — score < threshold
        'standard' — threshold <= score <= 0.7
        'boost'    — score > 0.7
    """
    if score < threshold:
        return 'skip'
    if score > 0.7:
        return 'boost'
    return 'standard'


def get_paper_formation(
    predictions: list,
    race_meta: dict,
    divergence_threshold: float = 0.3,
) -> Optional[list]:
    """divergence_score に基づく paper formation を返す。

    Args:
        predictions: top N 予測馬 list。各要素 dict:
            horse_num  (int)   — 馬番
            pop_rank   (int, optional) — 市場人気順位
        race_meta: race 情報 dict:
            cond_key  (str)  — 条件 A/B/C/D/E/X
            distance  (int)  — レース距離
            venue     (str)  — 開催場
        divergence_threshold: この値未満の score なら skip (default 0.3)

    Returns:
        list of sorted 3-tuple: trio 7点 formation
        None: skip (score < threshold、または predictions 不足)
    """
    if not predictions or len(predictions) < 3:
        return None

    top1 = predictions[0]
    pop_rank_top1 = int(top1.get('pop_rank') or top1.get('ninki') or 0)
    score = compute_divergence_score(pop_rank_top1)

    if score < divergence_threshold:
        return None

    # top6 馬番
    top6 = [h.get('horse_num') or h.get('umaban') for h in predictions[:6]]
    top6 = [n for n in top6 if n is not None]
    if len(top6) < 3:
        return None

    # 標準 trio 7点 formation (top1 軸 / top2,3 / top2-6)
    t1 = top6[0]
    second = top6[1:3]
    third = top6[1:6]
    bets: set = set()
    for s in second:
        for t in third:
            combo = tuple(sorted({t1, s, t}))
            if len(combo) == 3:
                bets.add(combo)
    return sorted(bets)


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path) -> list[dict]:
    """CSV を読み込んで list[dict] で返す。"""
    rows = []
    with open(csv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=None):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _top3_from_row(row: dict) -> Optional[list]:
    """row から実 1-3 着馬番 list を取得。なければ None。"""
    f1 = _safe_int(row.get('top1_finish'))
    f2 = _safe_int(row.get('top2_finish'))
    f3 = _safe_int(row.get('top3_finish'))
    # trio_result 列 (例: "3-8-10") をフォールバックに使う
    if None in (f1, f2, f3):
        trio_result = (row.get('trio_result') or '').strip()
        if trio_result and '-' in trio_result:
            parts = trio_result.split('-')
            if len(parts) >= 3:
                try:
                    return [int(p) for p in parts[:3]]
                except ValueError:
                    return None
        return None
    return [f1, f2, f3]


def _formation_from_row(row: dict) -> Optional[list]:
    """trio_bets_str 列から formation を復元。例: '3-8-10' → [[3,8,10]]"""
    s = (row.get('trio_bets_str') or row.get('trio_bets') or '').strip()
    if not s:
        return None
    # 複数 bet は '/' or ',' 区切りを想定
    bets = []
    for part in s.replace(',', '/').split('/'):
        part = part.strip()
        if '-' in part:
            try:
                bets.append(sorted(int(x) for x in part.split('-')))
            except ValueError:
                pass
    return bets if bets else None


def _check_hit(formation: list, real_top3: list) -> bool:
    """formation (list of sorted list) が real_top3 を的中しているか。"""
    target = sorted(real_top3)
    return any(list(b) == target for b in formation)


def backtest_divergence_v2(
    csv_path: Path,
    thresholds: list[float] = None,
    train_end: str = '20260503',
    test_start: str = '20260504',
) -> dict:
    """divergence v2 の backtest を実行。

    hold-out: train=4/1-5/3、test=5/4-5/17

    Args:
        csv_path: cumulative_results.csv のパス
        thresholds: 検証する threshold リスト (default [0.3, 0.5, 0.7])
        train_end: train 期間末日 (YYYYMMDD)
        test_start: test 期間開始日 (YYYYMMDD)

    Returns:
        dict: {
            'has_pop_rank': bool,
            'proxy_mode': bool,
            'note': str,
            'results': {
                threshold_str: {
                    'n_bet': int, 'n_skip': int, 'n_hit': int,
                    'hit_rate': float, 'skip_rate': float,
                    'roi': float,  # pnl / (n_bet * 700)
                    'pnl': int,
                }
            }
        }
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    rows = _load_csv(csv_path)

    # pop_rank 列の有無を確認
    has_pop_rank = any(r.get('pop_rank') for r in rows)

    # settled のみ / 日付フィルタ
    def _date_str(row) -> str:
        d = str(row.get('date') or '').strip().split('.')[0]
        return d[:8] if len(d) >= 8 else ''

    settled = [r for r in rows if str(r.get('status', '')).strip() == 'settled']

    if not settled:
        return {
            'has_pop_rank': has_pop_rank,
            'proxy_mode': not has_pop_rank,
            'note': 'settled rows not found',
            'results': {},
        }

    test_rows = [r for r in settled if _date_str(r) >= test_start]

    note_parts = []
    if not has_pop_rank:
        note_parts.append(
            'pop_rank 列なし → top1_finish proxy モード (top1 が 4 着以下を "high divergence" と見なす)'
        )

    if not test_rows:
        note_parts.append(f'test 期間 ({test_start}+) のレースなし。全 settled データで集計。')
        test_rows = settled

    results: dict[str, dict] = {}

    BET_COST = 700

    for thr in thresholds:
        n_bet = n_skip = n_hit = total_payout = 0

        for row in test_rows:
            # --- divergence score の計算 ---
            if has_pop_rank:
                pop_rank = _safe_int(row.get('pop_rank'), default=0)
                score = compute_divergence_score(pop_rank)
            else:
                # proxy: top1_finish > 3 → ≒ 人気薄だったと仮定
                top1_finish = _safe_int(row.get('top1_finish'), default=0)
                if top1_finish is None or top1_finish <= 0:
                    # 不明 → conservative boost として score=1.0
                    score = 1.0
                elif top1_finish == 1:
                    score = 0.0
                elif top1_finish == 2:
                    score = 0.3
                elif top1_finish == 3:
                    score = 0.6
                else:
                    score = 1.0

            action = classify_action(score, threshold=thr)

            if action == 'skip':
                n_skip += 1
                continue

            n_bet += 1

            # 的中判定
            formation = _formation_from_row(row)
            real_top3 = _top3_from_row(row)

            trio_hit = _safe_int(row.get('trio_hit'), default=0)
            trio_payout = _safe_float(row.get('trio_payout'), default=0.0)

            if trio_hit:
                n_hit += 1
                total_payout += int(trio_payout)

        invested = n_bet * BET_COST
        roi = (total_payout / invested * 100) if invested > 0 else 0.0
        pnl = total_payout - invested
        skip_rate = n_skip / (n_bet + n_skip) if (n_bet + n_skip) > 0 else 0.0
        hit_rate = n_hit / n_bet if n_bet > 0 else 0.0

        results[str(thr)] = {
            'n_bet': n_bet,
            'n_skip': n_skip,
            'n_hit': n_hit,
            'hit_rate': round(hit_rate, 4),
            'skip_rate': round(skip_rate, 4),
            'roi': round(roi, 2),
            'pnl': pnl,
        }

    return {
        'has_pop_rank': has_pop_rank,
        'proxy_mode': not has_pop_rank,
        'note': ' / '.join(note_parts) if note_parts else '',
        'results': results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main(argv=None):
    parser = argparse.ArgumentParser(
        description='V15 market divergence v2 backtest (paper only)'
    )
    parser.add_argument(
        '--csv', default=str(DEFAULT_CSV),
        help='cumulative_results.csv path'
    )
    parser.add_argument(
        '--thresholds', default='0.3,0.5,0.7',
        help='comma-separated threshold values (default: 0.3,0.5,0.7)'
    )
    parser.add_argument(
        '--train-end', default='20260503',
        help='train end date YYYYMMDD (default: 20260503)'
    )
    parser.add_argument(
        '--test-start', default='20260504',
        help='test start date YYYYMMDD (default: 20260504)'
    )
    args = parser.parse_args(argv)

    thresholds = [float(t) for t in args.thresholds.split(',')]
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f'[ERROR] CSV not found: {csv_path}', file=sys.stderr)
        sys.exit(1)

    print(f'[divergence v2] CSV: {csv_path}')
    print(f'[divergence v2] thresholds: {thresholds}')
    print(f'[divergence v2] hold-out: test >= {args.test_start}')
    print()

    out = backtest_divergence_v2(
        csv_path=csv_path,
        thresholds=thresholds,
        train_end=args.train_end,
        test_start=args.test_start,
    )

    print(f"has_pop_rank : {out['has_pop_rank']}")
    print(f"proxy_mode   : {out['proxy_mode']}")
    if out['note']:
        print(f"note         : {out['note']}")
    print()

    if not out['results']:
        print('results: (empty)')
        return

    # header
    print(f"{'threshold':>10}  {'n_bet':>6}  {'n_skip':>6}  {'skip%':>6}  "
          f"{'n_hit':>6}  {'hit%':>6}  {'ROI%':>8}  {'PnL':>8}")
    print('-' * 75)
    for thr, r in sorted(out['results'].items(), key=lambda x: float(x[0])):
        print(
            f"{float(thr):>10.1f}  {r['n_bet']:>6}  {r['n_skip']:>6}  "
            f"{r['skip_rate']*100:>5.1f}%  {r['n_hit']:>6}  "
            f"{r['hit_rate']*100:>5.1f}%  {r['roi']:>7.1f}%  {r['pnl']:>+8,}"
        )


if __name__ == '__main__':
    _main()
