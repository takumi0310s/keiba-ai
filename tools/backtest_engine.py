"""Phase 17: 30 年 backtest 環境 (実装は available data 16 年 + actual logs 15 日)

Phase 17 段階:
  - 30 年 fetch は 5/24+ JV-Link 加入後 (Phase 3)
  - 現状 available: jra_races_full.csv 2010-2025 (16 年) + daily_predictions/results 15 日
  - 本 engine = WF backtest framework + 戦略 simulation 基盤

V15 投資保護: read-only、 既存 model 不変。
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

BASE = Path(r"C:/Users/takum/keiba-ai")
DAILY_PRED_DIR = BASE / "data" / "daily_predictions"
DAILY_RES_DIR = BASE / "data" / "daily_results"


@dataclass
class RaceRecord:
    """1 R の予測 + 結果"""
    race_id: str
    date: str
    course: str
    race_num: int
    race_name: str
    condition: str
    num_horses: int
    distance: int
    surface: str
    morning_top1: str
    morning_top1_score: float
    morning_top2: str
    morning_top3: str
    trio_bets: str
    bet_type: str
    investment: int
    actual_top3: list[str]
    trio_hit: bool
    payout: int
    profit: int


def load_history(start_date: str | None = None, end_date: str | None = None) -> list[RaceRecord]:
    """daily_predictions + daily_results から履歴 R 集約"""
    races: list[RaceRecord] = []
    pred_files = sorted(DAILY_PRED_DIR.glob("2026*.csv"))
    for pf in pred_files:
        date = pf.stem
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        rf = DAILY_RES_DIR / f"{date}.csv"
        if not rf.exists():
            continue
        with open(pf, encoding='utf-8-sig') as f:
            preds = {r['race_id']: r for r in csv.DictReader(f)}
        with open(rf, encoding='utf-8-sig') as f:
            results = {r['race_id']: r for r in csv.DictReader(f)}
        for rid, p in preds.items():
            res = results.get(rid)
            if not res:
                continue
            actual_top3 = [res.get(c, '').strip() for c in ('top1_finish', 'top2_finish', 'top3_finish')]
            actual_top3 = [t for t in actual_top3 if t]
            try:
                trio_hit = str(res.get('trio_hit', '0')).strip() == '1'
                payout = int(float(res.get('actual_payout', 0) or 0))
                investment = int(float(res.get('investment', 0) or 0))
                profit = int(float(res.get('profit', 0) or 0))
                num_horses = int(p.get('num_horses', 0) or 0)
                distance = int(p.get('distance', 0) or 0)
                race_num = int(p.get('race_num', 0) or 0)
                top1_score = float(p.get('top1_score', 0) or 0)
            except (ValueError, TypeError):
                continue
            races.append(RaceRecord(
                race_id=rid, date=date,
                course=p.get('course', ''), race_num=race_num,
                race_name=p.get('race_name', ''),
                condition=p.get('condition', ''),
                num_horses=num_horses, distance=distance,
                surface=p.get('surface', ''),
                morning_top1=p.get('top1_num', ''),
                morning_top1_score=top1_score,
                morning_top2=p.get('top2_num', ''),
                morning_top3=p.get('top3_num', ''),
                trio_bets=p.get('trio_bets', ''),
                bet_type=p.get('bet_type', ''),
                investment=investment,
                actual_top3=actual_top3,
                trio_hit=trio_hit, payout=payout, profit=profit,
            ))
    return races


def walk_forward(races: list[RaceRecord], train_days: int = 7, test_days: int = 1
                 ) -> Iterator[tuple[list[RaceRecord], list[RaceRecord]]]:
    """WF: 連続 train_days train + 後続 test_days test"""
    by_date = defaultdict(list)
    for r in races:
        by_date[r.date].append(r)
    dates = sorted(by_date.keys())
    for i in range(train_days, len(dates) - test_days + 1):
        train_dates = dates[i - train_days: i]
        test_dates = dates[i: i + test_days]
        train = [r for d in train_dates for r in by_date[d]]
        test = [r for d in test_dates for r in by_date[d]]
        yield train, test


def simulate_strategy(races: list[RaceRecord], strategy: dict) -> dict:
    """戦略 simulation. strategy = {'min_score': 0.7, 'class_filter': [...], ...}"""
    min_score = strategy.get('min_score', 0.0)
    class_excluded = set(strategy.get('exclude_class', []))  # E, B
    name_excluded = strategy.get('exclude_name_substr', [])  # 06_平場特別
    venue_excluded = set(strategy.get('exclude_venue', []))
    bet_amount = strategy.get('bet_amount', 700)
    top_n = strategy.get('top_n_per_day', 999)

    races_by_date = defaultdict(list)
    for r in races:
        races_by_date[r.date].append(r)

    n_total, n_bet, n_hit = 0, 0, 0
    total_inv, total_pay = 0, 0
    for date, day_races in races_by_date.items():
        day_races.sort(key=lambda r: -r.morning_top1_score)
        day_picked = 0
        for r in day_races:
            n_total += 1
            if r.morning_top1_score < min_score:
                continue
            if r.condition in class_excluded:
                continue
            if r.course in venue_excluded:
                continue
            if any(s in r.race_name for s in name_excluded):
                continue
            if day_picked >= top_n:
                break
            n_bet += 1
            day_picked += 1
            total_inv += bet_amount
            if r.trio_hit:
                n_hit += 1
                total_pay += r.payout
    return {
        'strategy': strategy,
        'n_total': n_total, 'n_bet': n_bet, 'n_hit': n_hit,
        'investment': total_inv, 'payout': total_pay,
        'profit': total_pay - total_inv,
        'roi_pct': 100.0 * total_pay / max(total_inv, 1),
        'hit_rate': 100.0 * n_hit / max(n_bet, 1),
    }


def main():
    races = load_history()
    print(f"Loaded {len(races)} race records from {len({r.date for r in races})} days")
    if not races:
        return

    # 戦略比較
    strategies = [
        {'name': '全 R (baseline)', 'min_score': 0.0},
        {'name': '案 B 改 strict', 'min_score': 0.7,
         'exclude_class': ['E', 'B'],
         'exclude_name_substr': ['特別'],
         'top_n_per_day': 3},
        {'name': '上位 score 0.7+', 'min_score': 0.7, 'top_n_per_day': 5},
        {'name': '上位 3 R / 日', 'min_score': 0.0, 'top_n_per_day': 3},
    ]

    print("\n=== 戦略比較 (16-day actual logs) ===\n")
    print(f"{'戦略':<25} {'bet':>4} {'hit':>4} {'inv':>8} {'pay':>8} {'profit':>8} {'ROI%':>7}")
    print("-" * 80)
    for s in strategies:
        name = s.pop('name')
        result = simulate_strategy(races, s)
        print(f"{name:<25} {result['n_bet']:>4} {result['n_hit']:>4} "
              f"{result['investment']:>8} {result['payout']:>8} "
              f"{result['profit']:>+8} {result['roi_pct']:>6.1f}%")
        s['name'] = name


if __name__ == '__main__':
    main()
