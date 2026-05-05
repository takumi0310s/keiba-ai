"""v18/v19 prob distribution analysis (BT 2025 OOS vs 5/2-5/3 retro).

distribution shift の正体を特定する。
- 全体 scaling の問題か (max/mean)
- race-level の構造問題か (top1/top2 ratio, 1着馬の prob ranking)
- racing 制約問題か (sum-to-1)
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
import json

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)


def percentile_summary(s, name):
    s = pd.Series(s).dropna()
    return {
        'name': name,
        'n': int(len(s)),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'p95': float(s.quantile(0.95)),
        'p99': float(s.quantile(0.99)),
        'max': float(s.max()),
        'min': float(s.min()),
    }


def race_level_stats(df, prob_col, race_col='race_id', win_col='is_win', label=''):
    """For each race, compute max prob, sum, top1 rank correctness, top1/top2 ratio."""
    g = df.groupby(race_col)
    maxes, sums, ratios_t1_t2, winner_p = [], [], [], []
    winner_in_top1, winner_in_top3 = 0, 0
    n_race = 0
    n_race_with_winner_known = 0
    for rid, sub in g:
        if len(sub) < 2: continue
        n_race += 1
        ps = np.sort(sub[prob_col].astype(float).values)[::-1]
        maxes.append(float(ps[0]))
        sums.append(float(ps.sum()))
        if ps[1] > 0:
            ratios_t1_t2.append(float(ps[0]/ps[1]))
        if win_col in sub.columns:
            winner = sub[sub[win_col] == 1]
            if len(winner) > 0:
                n_race_with_winner_known += 1
                wp = float(winner[prob_col].iloc[0])
                winner_p.append(wp)
                # rank of winner in this race (1=top, N=bottom)
                rank_of_winner = (sub[prob_col].astype(float) > wp).sum() + 1
                if rank_of_winner == 1: winner_in_top1 += 1
                if rank_of_winner <= 3: winner_in_top3 += 1
    out = {
        'label': label,
        'n_race': n_race,
        'n_race_winner_known': n_race_with_winner_known,
        'winner_top1_rate': winner_in_top1 / max(n_race_with_winner_known, 1),
        'winner_top3_rate': winner_in_top3 / max(n_race_with_winner_known, 1),
        'race_max_p': percentile_summary(maxes, 'race_max_p'),
        'race_sum_p': percentile_summary(sums, 'race_sum_p'),
        'top1_top2_ratio': percentile_summary(ratios_t1_t2, 'top1_top2_ratio'),
    }
    if winner_p:
        out['winner_p'] = percentile_summary(winner_p, 'winner_p')
    return out


def main():
    out_path = 'data/v18/distribution_shift_analysis.json'
    text_out_path = 'data/v18/distribution_shift_analysis.md'
    print('=== A. v18/v19 distribution shift analysis ===\n')

    bt = pd.read_csv('data/v18/v18_tansho_oos_2025.csv', dtype={'race_id':str})
    # BT race_id は composite (race+umaban suffix) のため race-level は [:-2]
    bt['race_id'] = bt['race_id'].str[:-2]
    print(f'BT 2025 OOS: {len(bt)} horses, {bt["race_id"].nunique()} races (race-level)')

    retro = pd.read_csv('data/v18/v18_v19_retro_full_predictions.csv')
    retro['race_id'] = retro['race_id'].astype(str)
    print(f'Retro 5/2-5/3: {len(retro)} horses, {retro["race_id"].nunique()} races')

    cal = pd.read_csv('data/v18/v18_v19_retro_calibrated.csv')
    cal['race_id'] = cal['race_id'].astype(str)
    print(f'Calibrated 5/2-5/3: {len(cal)} horses\n')

    # --- 1. horse-level distribution ---
    bt_p = bt['p_ens']
    retro_p_t = retro['p_tansho']
    retro_p_f = retro['p_fukusho']
    cal_p_t = cal['p_tansho_cal']
    cal_p_f = cal['p_fukusho_cal']

    summaries = {
        'horse_level_distribution': {
            'BT_v18_p_ens (tansho)': percentile_summary(bt_p, 'BT v18 p_ens'),
            'Retro_v18_p_tansho_raw': percentile_summary(retro_p_t, 'Retro v18 raw'),
            'Retro_v18_p_tansho_cal': percentile_summary(cal_p_t, 'Retro v18 cal'),
            'Retro_v19_p_fukusho_raw': percentile_summary(retro_p_f, 'Retro v19 raw'),
            'Retro_v19_p_fukusho_cal': percentile_summary(cal_p_f, 'Retro v19 cal'),
        }
    }

    # --- 2. race-level statistics ---
    print('--- BT 2025 OOS race-level ---')
    bt_race = race_level_stats(bt, 'p_ens', 'race_id', 'is_win', 'BT_2025_OOS')
    print(f"  n_race={bt_race['n_race']} winner_top1={bt_race['winner_top1_rate']*100:.1f}% top3={bt_race['winner_top3_rate']*100:.1f}%")
    print(f"  race_max_p: max={bt_race['race_max_p']['max']:.3f} mean={bt_race['race_max_p']['mean']:.3f} p95={bt_race['race_max_p']['p95']:.3f}")
    print(f"  race_sum_p: max={bt_race['race_sum_p']['max']:.3f} mean={bt_race['race_sum_p']['mean']:.3f}")
    print(f"  top1/top2 ratio: mean={bt_race['top1_top2_ratio']['mean']:.2f} p95={bt_race['top1_top2_ratio']['p95']:.2f}")

    print('\n--- Retro 5/2-5/3 raw race-level ---')
    retro_race = race_level_stats(retro, 'p_tansho', 'race_id', 'is_win', 'Retro_raw')
    print(f"  n_race={retro_race['n_race']} winner_top1={retro_race['winner_top1_rate']*100:.1f}% top3={retro_race['winner_top3_rate']*100:.1f}%")
    print(f"  race_max_p: max={retro_race['race_max_p']['max']:.3f} mean={retro_race['race_max_p']['mean']:.3f} p95={retro_race['race_max_p']['p95']:.3f}")
    print(f"  race_sum_p: max={retro_race['race_sum_p']['max']:.3f} mean={retro_race['race_sum_p']['mean']:.3f}")
    print(f"  top1/top2 ratio: mean={retro_race['top1_top2_ratio']['mean']:.2f} p95={retro_race['top1_top2_ratio']['p95']:.2f}")

    print('\n--- Retro 5/2-5/3 calibrated race-level ---')
    cal_race = race_level_stats(cal, 'p_tansho_cal', 'race_id', 'is_win', 'Retro_cal')
    print(f"  n_race={cal_race['n_race']} winner_top1={cal_race['winner_top1_rate']*100:.1f}% top3={cal_race['winner_top3_rate']*100:.1f}%")
    print(f"  race_max_p: max={cal_race['race_max_p']['max']:.3f} mean={cal_race['race_max_p']['mean']:.3f}")

    summaries['race_level_distribution'] = {
        'BT_2025_OOS': bt_race,
        'Retro_raw': retro_race,
        'Retro_calibrated': cal_race,
    }

    # --- 3. shift attribution ---
    bt_max = bt_race['race_max_p']['max']
    retro_max = retro_race['race_max_p']['max']
    bt_max_mean = bt_race['race_max_p']['mean']
    retro_max_mean = retro_race['race_max_p']['mean']

    bt_top1_top2 = bt_race['top1_top2_ratio']['mean']
    retro_top1_top2 = retro_race['top1_top2_ratio']['mean']

    print('\n--- shift attribution ---')
    print(f'BT race_max_p mean: {bt_max_mean:.3f}, retro: {retro_max_mean:.3f} → factor {bt_max_mean/max(retro_max_mean,1e-9):.2f}x')
    print(f'BT top1/top2 ratio mean: {bt_top1_top2:.2f}, retro: {retro_top1_top2:.2f}')
    print(f'BT winner top1 rate: {bt_race["winner_top1_rate"]*100:.1f}%, retro: {retro_race["winner_top1_rate"]*100:.1f}%')

    summaries['shift_attribution'] = {
        'race_max_p_factor': float(bt_max_mean / max(retro_max_mean, 1e-9)),
        'top1_top2_ratio_diff': float(bt_top1_top2 - retro_top1_top2),
        'winner_top1_rate_diff': float(bt_race['winner_top1_rate'] - retro_race['winner_top1_rate']),
        'verdict': '',
    }

    # 判断ロジック
    factor = summaries['shift_attribution']['race_max_p_factor']
    rank_diff = abs(summaries['shift_attribution']['winner_top1_rate_diff'])
    if factor > 2 and rank_diff < 0.1:
        verdict = 'GLOBAL_SCALING_SHIFT — 全体 scaling 問題、馬選定は正しい。race-level normalization で解決可能性高'
    elif rank_diff > 0.1:
        verdict = 'RANK_SHIFT — 1着馬の選定自体がBTより劣化。feature distribution shift 疑い'
    else:
        verdict = 'PARTIAL_SCALING — 部分的 scaling シフト'
    print(f'\n判定: {verdict}')
    summaries['shift_attribution']['verdict'] = verdict

    # --- 4. winner_known races の prob ranking ---
    df_w = retro[retro['winner_known'] == 1]
    print(f'\n--- winner_known retro: {df_w["race_id"].nunique()} races ---')
    if len(df_w) > 0:
        # winner の prob と rank
        win_info = []
        for rid, sub in df_w.groupby('race_id'):
            winner_row = sub[sub['is_win'] == 1]
            if len(winner_row) == 0: continue
            wp = float(winner_row['p_tansho'].iloc[0])
            sub_sorted = sub.sort_values('p_tansho', ascending=False).reset_index(drop=True)
            rank = int(sub_sorted[sub_sorted['umaban'] == int(winner_row['umaban'].iloc[0])].index[0]) + 1
            win_info.append({'rid': rid, 'wp': wp, 'rank': rank, 'race_size': len(sub)})
        wi = pd.DataFrame(win_info)
        print(f'  winner_p: mean={wi["wp"].mean():.4f} max={wi["wp"].max():.4f}')
        print(f'  winner rank distribution: {wi["rank"].value_counts().sort_index().to_dict()}')
        summaries['winner_rank_in_retro'] = wi['rank'].value_counts().sort_index().to_dict()
        summaries['winner_p_in_retro'] = percentile_summary(wi['wp'], 'winner_p')

    # save
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n[OK] {out_path}')

    # markdown summary
    md = ['# v18/v19 distribution shift analysis (Phase 2.5)', '',
          f'生成: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', '',
          '## 1. horse-level distribution', '',
          '| dataset | n | mean | median | p95 | p99 | max |', '|---|---:|---:|---:|---:|---:|---:|']
    for k, v in summaries['horse_level_distribution'].items():
        md.append(f'| {k} | {v["n"]} | {v["mean"]:.4f} | {v["median"]:.4f} | {v["p95"]:.4f} | {v["p99"]:.4f} | {v["max"]:.4f} |')
    md += ['', '## 2. race-level distribution', '',
           '| dataset | n_race | race_max_p mean | race_max_p p95 | race_max_p max | race_sum_p mean | top1/top2 ratio mean | winner_top1 | winner_top3 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k, v in summaries['race_level_distribution'].items():
        md.append(f'| {k} | {v["n_race"]} | {v["race_max_p"]["mean"]:.3f} | {v["race_max_p"]["p95"]:.3f} | {v["race_max_p"]["max"]:.3f} | {v["race_sum_p"]["mean"]:.3f} | {v["top1_top2_ratio"]["mean"]:.2f} | {v["winner_top1_rate"]*100:.1f}% | {v["winner_top3_rate"]*100:.1f}% |')
    md += ['', '## 3. shift attribution', '',
           f'- race_max_p factor (BT/retro): **{summaries["shift_attribution"]["race_max_p_factor"]:.2f}x**',
           f'- top1/top2 ratio diff: {summaries["shift_attribution"]["top1_top2_ratio_diff"]:+.2f}',
           f'- winner_top1 rate diff (BT - retro): {summaries["shift_attribution"]["winner_top1_rate_diff"]*100:+.1f}pt',
           f'- **判定: {verdict}**', '']
    if 'winner_rank_in_retro' in summaries:
        md += ['## 4. retro winner rank (in pred top-N)', '']
        md.append('| rank | count |')
        md.append('|---:|---:|')
        for k, v in summaries['winner_rank_in_retro'].items():
            md.append(f'| {k} | {v} |')
    with open(text_out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'[OK] {text_out_path}')


if __name__ == '__main__':
    main()
