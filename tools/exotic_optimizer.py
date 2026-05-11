#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B3: Pari-mutuel exotic 最適化 (三連複 / 三連単 / 馬連 EV 最大点数).

V15 現行: trio 7 点 固定 (TOP1 軸 - TOP2,3 - TOP2-6 フォーメーション、 700 円)。
本 script: 各 race ごとに pred 確率 + 想定オッズから **EV 最大化 点数組合せ** を選び直す。

【V15 投資保護】 predict_core / daily_predict 一切触らず、 既存 V15 予測結果を post-process。

Usage:
    # 単一 race 試算 (8 頭 想定 確率配列 + オッズ pool)
    python tools/exotic_optimizer.py demo

    # CSV 一括 (race_id + 各馬 pred + 想定オッズ table) は今後拡張
"""
import argparse
import itertools
import json
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def normalize_probs(probs):
    """合計 1 に正規化."""
    s = sum(probs)
    if s <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / s for p in probs]


def trio_prob(p, idx_triple):
    """3 頭 i,j,k が 1-3 着に入る確率 (順序不問). Plackett-Luce 近似."""
    i, j, k = idx_triple
    # P(top3) = sum over permutations of (p_a * p_b/(1-p_a) * p_c/(1-p_a-p_b))
    horses = [i, j, k]
    total = 0.0
    for perm in itertools.permutations(horses):
        a, b, c = perm
        denom1 = 1.0
        denom2 = max(1e-9, 1.0 - p[a])
        denom3 = max(1e-9, 1.0 - p[a] - p[b])
        total += (p[a] / denom1) * (p[b] / denom2) * (p[c] / denom3)
    return total


def trifecta_prob(p, idx_triple_ordered):
    """順序付き 3 頭 (1着,2着,3着 確定) 確率."""
    a, b, c = idx_triple_ordered
    denom2 = max(1e-9, 1.0 - p[a])
    denom3 = max(1e-9, 1.0 - p[a] - p[b])
    return (p[a]) * (p[b] / denom2) * (p[c] / denom3)


def umaren_prob(p, pair):
    """馬連 i,j が 1-2 着 (順序不問) 確率."""
    i, j = pair
    return (p[i] * (p[j] / max(1e-9, 1 - p[i]))) + (p[j] * (p[i] / max(1e-9, 1 - p[j])))


def estimate_trio_odds(p, idx_triple, market_take=0.20):
    """Pari-mutuel 想定 trio オッズ. 全 trio の market_prob 算出 + 控除率反映."""
    # 真の確率 ≒ market 想定で odds = (1 - take) / prob
    return (1 - market_take) / max(1e-9, trio_prob(p, idx_triple))


def estimate_umaren_odds(p, pair, market_take=0.20):
    return (1 - market_take) / max(1e-9, umaren_prob(p, pair))


def estimate_trifecta_odds(p, idx_triple_ordered, market_take=0.25):
    return (1 - market_take) / max(1e-9, trifecta_prob(p, idx_triple_ordered))


def select_optimal_trio(probs, n_horses, top_k_axis=1, max_points=10,
                       min_ev=0.0, market_take=0.20):
    """EV 最大化 trio 点数選択.

    Returns: list of (triple, prob, odds, ev) sorted by EV desc.
    """
    p = normalize_probs(probs)
    # 軸 = top_k_axis、 残り 2 頭は他から
    sorted_idx = sorted(range(n_horses), key=lambda i: -p[i])
    axes = sorted_idx[:top_k_axis]
    others = sorted_idx  # 軸 含む全 (フォーメーションでも軸単独でもよい)

    candidates = []
    seen = set()
    for ax in axes:
        for j, k in itertools.combinations(others, 2):
            if j == ax or k == ax:
                continue
            triple = tuple(sorted([ax, j, k]))
            if triple in seen:
                continue
            seen.add(triple)
            prob = trio_prob(p, triple)
            odds = estimate_trio_odds(p, triple, market_take=market_take)
            # EV per bet (100 円 賭けて prob で odds*100、 残りは 0)
            ev = prob * odds - 1
            candidates.append({
                'triple': triple,
                'prob': prob,
                'odds_est': odds,
                'ev': ev,
            })

    candidates.sort(key=lambda x: -x['ev'])
    selected = [c for c in candidates if c['ev'] >= min_ev][:max_points]
    return selected


def cmd_demo(args):
    print('=== Pari-mutuel exotic 最適化 demo ===')
    print('15 頭立て、 V15 出力 確率分布 想定 (top 1 = 18%, top 6 = 6%, tail ~3%)\n')
    probs = [0.18, 0.13, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.02, 0.02]
    n = len(probs)
    probs = normalize_probs(probs)

    print('現行 V15 戦略 (TOP1 - TOP2,3 - TOP2-6 7 点 フォーメーション):')
    current_triples = []
    for j in [1, 2]:
        for k in [1, 2, 3, 4, 5]:
            if k <= j:
                continue
            triple = tuple(sorted([0, j, k]))
            if triple in current_triples:
                continue
            current_triples.append(triple)
    current_triples = current_triples[:7]
    total_prob_cur = 0
    total_ev_cur = 0
    for t in current_triples:
        prob = trio_prob(probs, t)
        odds = estimate_trio_odds(probs, t)
        ev = prob * odds - 1
        total_prob_cur += prob
        total_ev_cur += ev
        print(f'  {t}  prob={prob:.4f}  odds={odds:>6.2f}  EV={ev:+.3f}')
    print(f'  → 合計 prob={total_prob_cur:.3f}, 平均 EV={total_ev_cur/len(current_triples):+.3f}\n')

    print('Optimal Top 7 (EV 最大) 選択:')
    optimal = select_optimal_trio(probs, n, top_k_axis=2, max_points=7, min_ev=0.0)
    total_prob_opt = 0
    total_ev_opt = 0
    for c in optimal:
        total_prob_opt += c['prob']
        total_ev_opt += c['ev']
        print(f'  {c["triple"]}  prob={c["prob"]:.4f}  odds={c["odds_est"]:>6.2f}  EV={c["ev"]:+.3f}')
    print(f'  → 合計 prob={total_prob_opt:.3f}, 平均 EV={total_ev_opt/max(1,len(optimal)):+.3f}\n')

    diff_ev = total_ev_opt/max(1,len(optimal)) - total_ev_cur/len(current_triples)
    diff_prob = total_prob_opt - total_prob_cur
    print(f'差分: 平均 EV {diff_ev:+.3f}, 合計 prob {diff_prob:+.3f}')
    print('→ optimal は 控除率込 想定オッズ ベース、 実 odds でもう一度判定 推奨')

    print('\n馬連 (2 点 想定、 条件 E):')
    pairs = [(0, j) for j in [1, 2, 3]]
    for pair in pairs:
        prob = umaren_prob(probs, pair)
        odds = estimate_umaren_odds(probs, pair)
        ev = prob * odds - 1
        print(f'  {pair}  prob={prob:.4f}  odds={odds:>6.2f}  EV={ev:+.3f}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Pari-mutuel exotic 最適化 (B3)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('demo')

    args = ap.parse_args()
    if args.cmd == 'demo':
        return cmd_demo(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
