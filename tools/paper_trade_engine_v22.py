"""Phase 17: V15/V18/V20/V21/V22 並行 paper trade engine

V15 model: production (read-only)
V18 candidate: predict_core_v18.py (Phase 11、 165 features、 未学習)
V20 candidate: AUC 0.91-0.93 期待 (Phase 10 plan、 未学習)
V21 candidate: 動画統合 (Phase 4 plan、 9/2 投入予定、 未学習)
V22 RL: PPO 投票最適化 (Phase 17 initial、 5000 steps trained)

paper_trade = 仮想 ¥700 投票 + 結果照合 (V15 不変、 read-only)。
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
import json

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))

from backtest_engine import load_history, RaceRecord


def paper_v15(race: RaceRecord) -> dict:
    """V15 production strategy: 案 B 改 strict (戦略⑦)"""
    name = race.race_name
    is_grade = any(g in name for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'])
    is_special = '特別' in name and not is_grade
    if race.condition in ('E', 'B'):
        return {'bet': 0, 'reason': 'cond_skip'}
    if is_special:
        return {'bet': 0, 'reason': 'tokubetsu_skip'}
    if race.morning_top1_score < 0.7:
        return {'bet': 0, 'reason': 'low_score'}
    return {'bet': 700, 'reason': 'V15_strict'}


def paper_v18(race: RaceRecord) -> dict:
    """V18 candidate: V15 + より厳しい score 閾値 (期待 +0.05 AUC、 0.75+ 限定)"""
    base = paper_v15(race)
    if base['bet'] == 0:
        return base
    if race.morning_top1_score < 0.75:
        return {'bet': 0, 'reason': 'V18_strict_score'}
    return {'bet': 700, 'reason': 'V18_strict'}


def paper_v20(race: RaceRecord) -> dict:
    """V20 candidate: V18 + 多頭数 R 重み増 (16+ 頭で bet 増)"""
    base = paper_v18(race)
    if base['bet'] == 0:
        return base
    bet = 700
    if race.num_horses >= 16:
        bet = 1000
    return {'bet': bet, 'reason': 'V20_strict_dynamic'}


def paper_v21(race: RaceRecord) -> dict:
    """V21 candidate: V20 + 動画 features (placeholder、 同 V20)"""
    return paper_v20(race)


def paper_v22(race: RaceRecord, model=None) -> dict:
    """V22 RL: PPO 学習済 model から action 取得"""
    if model is None:
        return {'bet': 0, 'reason': 'no_model'}
    import numpy as np
    from v22_rl_agent import _race_to_state
    state = _race_to_state(race)
    action, _ = model.predict(state, deterministic=True)
    bet_amounts = [0, 100, 300, 700]
    bet = bet_amounts[int(action)]
    return {'bet': bet, 'reason': f'V22_RL_action_{int(action)}'}


def evaluate_strategy(races: list[RaceRecord], strategy_fn, **kwargs) -> dict:
    """1 戦略を全 R で評価"""
    n_total = len(races)
    n_bet = 0
    n_hit = 0
    total_inv = 0
    total_pay = 0
    for r in races:
        decision = strategy_fn(r, **kwargs) if kwargs else strategy_fn(r)
        bet = decision['bet']
        if bet == 0:
            continue
        n_bet += 1
        total_inv += bet
        if r.trio_hit:
            n_hit += 1
            ratio = bet / 700.0
            total_pay += int(r.payout * ratio)
    return {
        'n_total': n_total, 'n_bet': n_bet, 'n_hit': n_hit,
        'investment': total_inv, 'payout': total_pay,
        'profit': total_pay - total_inv,
        'roi_pct': 100.0 * total_pay / max(total_inv, 1),
        'hit_rate': 100.0 * n_hit / max(n_bet, 1),
    }


def main():
    races = load_history()
    print(f"Loaded {len(races)} races")

    # Load V22 model if exists
    v22_model = None
    v22_path = BASE / "models" / "v22" / "v22_rl_ppo_phase17_initial.zip"
    if v22_path.exists():
        try:
            from stable_baselines3 import PPO
            v22_model = PPO.load(str(v22_path))
            print(f"V22 RL model loaded: {v22_path}")
        except Exception as e:
            print(f"V22 load failed: {e}")

    print("\n=== Paper trade 比較 (履歴 logs base) ===\n")
    print(f"{'戦略':<12} {'bet':>4} {'hit':>4} {'inv':>8} {'pay':>8} {'profit':>8} {'ROI%':>7} {'hit%':>6}")
    print("-" * 80)

    strategies = [
        ('V15', paper_v15),
        ('V18 cand', paper_v18),
        ('V20 cand', paper_v20),
        ('V21 cand', paper_v21),
    ]
    results = {}
    for name, fn in strategies:
        r = evaluate_strategy(races, fn)
        results[name] = r
        print(f"{name:<12} {r['n_bet']:>4} {r['n_hit']:>4} {r['investment']:>8} {r['payout']:>8} "
              f"{r['profit']:>+8} {r['roi_pct']:>6.1f}% {r['hit_rate']:>5.1f}%")

    if v22_model:
        r = evaluate_strategy(races, lambda race: paper_v22(race, model=v22_model))
        results['V22 RL'] = r
        print(f"{'V22 RL':<12} {r['n_bet']:>4} {r['n_hit']:>4} {r['investment']:>8} {r['payout']:>8} "
              f"{r['profit']:>+8} {r['roi_pct']:>6.1f}% {r['hit_rate']:>5.1f}%")

    # save
    out = BASE / "data" / "v22" / "phase17_paper_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nsaved: {out}")
    return results


if __name__ == '__main__':
    main()
