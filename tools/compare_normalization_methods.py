"""B. race-level normalization 3案比較.

retro 5/2-5/3 の prob を 3案で normalize し、
- prob distribution の BT 接近度 (race_max_p mean 0.347 が target)
- winner_top1 rate の改善
- bet 候補数 (p>=0.5 ev>=1.2)
- 想定 ROI

を比較する。
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
import json

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)

# BT 2025 OOS target
BT_RACE_MAX_MEAN = 0.347
BT_RACE_SUM_MEAN = 0.753
BT_WINNER_TOP1 = 0.478

# Phase 2 filter
T_PROB_MIN = 0.5
T_EV_MIN = 1.2
F_PROB_MIN = 0.7
F_EV_MIN = 1.1


def softmax_per_race(df, prob_col, race_col='race_id', temperature=1.0, eps=1e-9):
    """logit(p)/T → softmax. (T<1 sharpen, T>1 soften)"""
    out = df.copy()
    new = np.zeros(len(df))
    for rid, sub in df.groupby(race_col):
        idx = sub.index
        p = np.clip(sub[prob_col].astype(float).values, eps, 1 - eps)
        logits = np.log(p / (1 - p)) / temperature
        ex = np.exp(logits - logits.max())
        sm = ex / ex.sum()
        new[idx] = sm
    out[prob_col + '_norm'] = new
    return out


def power_normalize_per_race(df, prob_col, race_col='race_id', temperature=1.0, eps=1e-9):
    """p^(1/T) / sum (sum-to-1)"""
    out = df.copy()
    new = np.zeros(len(df))
    for rid, sub in df.groupby(race_col):
        idx = sub.index
        p = np.clip(sub[prob_col].astype(float).values, eps, 1.0)
        pT = p ** (1.0 / temperature)
        nm = pT / pT.sum()
        new[idx] = nm
    out[prob_col + '_norm'] = new
    return out


def rank_scale_per_race(df, prob_col, race_col='race_id', target_max=BT_RACE_MAX_MEAN):
    """各レースで rank に基づき max を target_max に rescale (top1=target_max, top-N=低)"""
    out = df.copy()
    new = np.zeros(len(df))
    for rid, sub in df.groupby(race_col):
        idx = sub.index
        p = sub[prob_col].astype(float).values
        if p.max() <= 0:
            new[idx] = p
            continue
        # 簡易: linear rescale 上位を target_max に、下位は同比率
        scaled = p * (target_max / p.max())
        new[idx] = scaled
    out[prob_col + '_norm'] = new
    return out


def evaluate_method(df, method_name, prob_col, race_col='race_id'):
    """各 race の max prob, sum, winner top1 rate, bet候補数を measure."""
    g = df.groupby(race_col)
    maxes, sums = [], []
    winner_in_top1 = 0
    n_winner_known = 0
    for rid, sub in g:
        if len(sub) < 2: continue
        ps = sub[prob_col].astype(float).values
        maxes.append(float(ps.max()))
        sums.append(float(ps.sum()))
        win = sub[sub['is_win'] == 1]
        if len(win) > 0:
            n_winner_known += 1
            wp = float(win[prob_col].iloc[0])
            if wp == ps.max():
                winner_in_top1 += 1
    return {
        'method': method_name,
        'race_max_p_mean': float(np.mean(maxes)) if maxes else 0,
        'race_max_p_p95': float(np.percentile(maxes, 95)) if maxes else 0,
        'race_max_p_max': float(np.max(maxes)) if maxes else 0,
        'race_sum_p_mean': float(np.mean(sums)) if sums else 0,
        'winner_top1_rate': winner_in_top1 / max(n_winner_known, 1),
        'n_winner_known': n_winner_known,
    }


def simulate_bets(df, prob_col, ev_col, win_col, prob_min, ev_min, label, BET=100):
    """Phase 2 filter + ROI simulation."""
    m = (df[prob_col] >= prob_min) & (df[ev_col] >= ev_min) & (df['odds'] > 0)
    n_bet = int(m.sum())
    if n_bet == 0:
        return {'label': label, 'bet': 0, 'win': 0, 'inv': 0, 'pay': 0, 'roi': None}
    wins = m & (df[win_col] == 1)
    n_win = int(wins.sum())
    inv = n_bet * BET
    pay = float((wins.astype(int) * df['odds'] * BET).sum())
    return {
        'label': label,
        'bet': n_bet, 'win': n_win,
        'inv': inv, 'pay': float(pay),
        'roi': float(pay / inv * 100) if inv > 0 else None,
        'hit_rate': float(n_win / n_bet) if n_bet else 0,
    }


def main():
    print('=== B. race-level normalization 3 case comparison ===\n')

    df = pd.read_csv('data/v18/v18_v19_retro_full_predictions.csv', dtype={'race_id':str})
    df_w = df[df['winner_known'] == 1].copy()
    df_w.reset_index(drop=True, inplace=True)
    print(f'winner_known horses: {len(df_w)} ({df_w["race_id"].nunique()} races)')

    # baseline
    base_eval = evaluate_method(df_w, 'BASELINE_raw', 'p_tansho')
    print(f"BASELINE  race_max_p mean={base_eval['race_max_p_mean']:.3f} winner_top1={base_eval['winner_top1_rate']*100:.1f}%")

    results = {'baseline': base_eval}
    bet_results = {'baseline_tansho': simulate_bets(df_w, 'p_tansho', 'ev_tansho', 'is_win', T_PROB_MIN, T_EV_MIN, 'baseline_tansho')}

    # === softmax variants ===
    print('\n--- 案1 softmax (logit/T → softmax) ---')
    for T in [0.3, 0.5, 0.7, 1.0, 1.5]:
        d = softmax_per_race(df_w, 'p_tansho', temperature=T)
        ev = evaluate_method(d, f'softmax_T={T}', 'p_tansho_norm')
        # update ev_tansho_norm
        d['ev_tansho_norm'] = d['p_tansho_norm'] * d['odds']
        bt = simulate_bets(d, 'p_tansho_norm', 'ev_tansho_norm', 'is_win', T_PROB_MIN, T_EV_MIN, f'softmax_T={T}')
        print(f"  T={T}: max_mean={ev['race_max_p_mean']:.3f} sum_mean={ev['race_sum_p_mean']:.3f} top1={ev['winner_top1_rate']*100:.1f}% bet={bt['bet']} ROI={bt['roi']}")
        results[f'softmax_T={T}'] = ev
        bet_results[f'softmax_T={T}'] = bt

    # === power-normalize ===
    print('\n--- 案2 power-normalize (p^(1/T) / sum) ---')
    for T in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        d = power_normalize_per_race(df_w, 'p_tansho', temperature=T)
        ev = evaluate_method(d, f'power_T={T}', 'p_tansho_norm')
        d['ev_tansho_norm'] = d['p_tansho_norm'] * d['odds']
        bt = simulate_bets(d, 'p_tansho_norm', 'ev_tansho_norm', 'is_win', T_PROB_MIN, T_EV_MIN, f'power_T={T}')
        print(f"  T={T}: max_mean={ev['race_max_p_mean']:.3f} sum_mean={ev['race_sum_p_mean']:.3f} top1={ev['winner_top1_rate']*100:.1f}% bet={bt['bet']} ROI={bt['roi']}")
        results[f'power_T={T}'] = ev
        bet_results[f'power_T={T}'] = bt

    # === rank-scale (target_max=BT_RACE_MAX_MEAN) ===
    print('\n--- 案3 rank-scale (linear rescale to target max) ---')
    for tgt in [BT_RACE_MAX_MEAN, 0.5, 0.7, 0.9]:
        d = rank_scale_per_race(df_w, 'p_tansho', target_max=tgt)
        ev = evaluate_method(d, f'rank_tgt={tgt}', 'p_tansho_norm')
        d['ev_tansho_norm'] = d['p_tansho_norm'] * d['odds']
        bt = simulate_bets(d, 'p_tansho_norm', 'ev_tansho_norm', 'is_win', T_PROB_MIN, T_EV_MIN, f'rank_tgt={tgt}')
        print(f"  tgt={tgt}: max_mean={ev['race_max_p_mean']:.3f} sum_mean={ev['race_sum_p_mean']:.3f} top1={ev['winner_top1_rate']*100:.1f}% bet={bt['bet']} ROI={bt['roi']}")
        results[f'rank_tgt={tgt}'] = ev
        bet_results[f'rank_tgt={tgt}'] = bt

    # === fukusho 同様 ===
    print('\n=== 複勝 (v19) ===')
    print('--- 案2 power-normalize for fukusho ---')
    for T in [0.5, 1.0, 1.5]:
        d = power_normalize_per_race(df_w, 'p_fukusho', temperature=T)
        ev = evaluate_method(d, f'fuk_power_T={T}', 'p_fukusho_norm')
        # ev_fukusho_norm using is_top3 instead
        d['ev_fukusho_norm'] = d['p_fukusho_norm'] * d['odds'] * 0.3
        # tweak: simulate hits via is_top3
        m = (d['p_fukusho_norm'] >= F_PROB_MIN) & (d['ev_fukusho_norm'] >= F_EV_MIN) & (d['odds'] > 0)
        n_bet = int(m.sum())
        if n_bet == 0:
            print(f"  T={T}: max_mean={ev['race_max_p_mean']:.3f} bet=0")
            continue
        hits = m & (d['is_top3'] == 1)
        inv = n_bet * 100
        pay = float((hits.astype(int) * d['odds'] * 0.3 * 100).sum())
        roi = pay / inv * 100
        print(f"  T={T}: max_mean={ev['race_max_p_mean']:.3f} bet={n_bet} hit={int(hits.sum())} ROI~={roi:.1f}%")
        bet_results[f'fuk_power_T={T}'] = {'bet': n_bet, 'hit': int(hits.sum()), 'roi': roi}

    # save
    out = {
        'baseline_evals': results,
        'bet_simulations': bet_results,
        'BT_target': {'race_max_p_mean': BT_RACE_MAX_MEAN, 'race_sum_p_mean': BT_RACE_SUM_MEAN, 'winner_top1': BT_WINNER_TOP1},
    }
    with open('data/v18/normalization_compare_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print('\n[OK] data/v18/normalization_compare_results.json')

    # markdown summary - find best method by ROI
    print('\n=== summary (best by ROI per shop, n>=3) ===')
    best = None
    for k, v in bet_results.items():
        if 'fuk' in k: continue
        if v.get('bet', 0) >= 3 and v.get('roi') is not None:
            if best is None or v['roi'] > best[1]['roi']:
                best = (k, v)
    if best:
        print(f"BEST tansho method: {best[0]} → bet={best[1]['bet']} win={best[1].get('win',0)} ROI={best[1]['roi']:.1f}%")
    else:
        print("WARN: 全 method で bet<3、より緩い filter で評価が必要")

    # also save: best method config to use in retro
    with open('data/v18/normalization_compare_results.json', 'r', encoding='utf-8') as f:
        existing = json.load(f)
    existing['best_tansho'] = {'method': best[0] if best else None, 'roi': best[1]['roi'] if best else None, 'bet': best[1]['bet'] if best else 0}
    with open('data/v18/normalization_compare_results.json', 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    main()
