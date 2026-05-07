"""V18/V19 sib抜き LIVE retro 結果 分析 (Session #38 B).

入力:
- data/v18/v18v19_retraining/no_sib_retro_5_2_5_3_predictions.csv
- data/v18/v18_v19_retro_full_predictions.csv (旧 含 sib)
- data/v18/v18_tansho_oos_2025.csv / v18v19_retraining/v18_no_sib_oos_2025.csv (BT)

出力:
- data/v18/v18v19_retraining/no_sib_live_retro_analysis.md
- data/v18/v18v19_retraining/no_sib_live_retro_metrics.json
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score


def main():
    paths = {
        'live_old': 'data/v18/v18_v19_retro_full_predictions.csv',
        'live_new': 'data/v18/v18v19_retraining/no_sib_retro_5_2_5_3_predictions.csv',
        'bt_old': 'data/v18/v18_tansho_oos_2025.csv',
        'bt_new': 'data/v18/v18v19_retraining/v18_no_sib_oos_2025.csv',
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            print(f"[ERR] missing {k}: {p}")
            return

    live_old = pd.read_csv(paths['live_old'])
    live_new = pd.read_csv(paths['live_new'])
    bt_old = pd.read_csv(paths['bt_old'])
    bt_new = pd.read_csv(paths['bt_new'])

    # ===== LIVE 比較 (5/2-5/3) =====
    print("=" * 60)
    print("LIVE 5/2-5/3 retro: OLD (含 sib) vs NEW (sib抜き)")
    print("=" * 60)

    metrics = {'live': {}, 'bt': {}}

    for label, df in [('OLD', live_old), ('NEW', live_new)]:
        wk = df[df['winner_known'] == 1].copy()
        n_races = wk['race_id'].nunique()
        # winner_top1
        top1 = wk.loc[wk.groupby('race_id')['p_tansho'].idxmax()]
        winner_top1 = top1['is_win'].mean() if len(top1) > 0 else 0
        # is_top3 hit rate among top3 prediction (proxy for v19)
        top3_in = []
        for rid, grp in wk.groupby('race_id'):
            top3 = grp.nlargest(3, 'p_fukusho')
            top3_in.append((top3['is_top3'].sum() >= 1))
        top3_hit_rate = np.mean(top3_in) if top3_in else 0
        # AUC (winner_known races のみ)
        try:
            auc_v18 = roc_auc_score(wk['is_win'], wk['p_tansho'])
        except: auc_v18 = None
        try:
            auc_v19 = roc_auc_score(df['is_top3'], df['p_fukusho'])
        except: auc_v19 = None

        mean_p18 = df['p_tansho'].mean()
        max_p18 = df['p_tansho'].max()
        mean_p19 = df['p_fukusho'].mean()

        print(f"\n--- {label} ---")
        print(f"  n_races (winner_known): {n_races}")
        print(f"  winner_top1: {winner_top1:.4f} ({int(winner_top1*n_races)}/{n_races})")
        print(f"  top3_hit_rate: {top3_hit_rate:.4f}")
        print(f"  AUC v18: {auc_v18:.4f}" if auc_v18 else "  AUC v18: N/A")
        print(f"  AUC v19: {auc_v19:.4f}" if auc_v19 else "  AUC v19: N/A")
        print(f"  mean p18: {mean_p18:.4f}, max p18: {max_p18:.4f}")
        print(f"  mean p19: {mean_p19:.4f}")

        metrics['live'][label] = {
            'n_races': int(n_races),
            'winner_top1': float(winner_top1),
            'top3_hit_rate': float(top3_hit_rate),
            'auc_v18': float(auc_v18) if auc_v18 else None,
            'auc_v19': float(auc_v19) if auc_v19 else None,
            'mean_p18': float(mean_p18),
            'max_p18': float(max_p18),
            'mean_p19': float(mean_p19),
        }

    # ===== BT vs LIVE shift factor =====
    print("\n" + "=" * 60)
    print("BT vs LIVE shift factor")
    print("=" * 60)

    # OLD BT (含 sib): use p_ens
    bt_old['race_part'] = bt_old['race_id'].astype(str).str[:-2]
    bt_old_top1 = bt_old.loc[bt_old.groupby('race_part')['p_ens'].idxmax()]
    bt_old_winner_top1 = bt_old_top1['is_win'].mean()
    bt_old_mean_p18 = bt_old['p_ens'].mean()

    # NEW BT (sib抜き): use p_v18_no_sib
    bt_new['race_part'] = bt_new['race_id'].astype(str).str[:-2]
    bt_new_top1 = bt_new.loc[bt_new.groupby('race_part')['p_v18_no_sib'].idxmax()]
    bt_new_winner_top1 = bt_new_top1['is_win'].mean()
    bt_new_mean_p18 = bt_new['p_v18_no_sib'].mean()

    print(f"\n--- OLD (含 sib) ---")
    print(f"  BT 2025 OOS:    winner_top1={bt_old_winner_top1:.4f} mean_p18={bt_old_mean_p18:.4f}")
    print(f"  LIVE 5/2-5/3:   winner_top1={metrics['live']['OLD']['winner_top1']:.4f} mean_p18={metrics['live']['OLD']['mean_p18']:.4f}")
    shift_old = bt_old_mean_p18 / max(metrics['live']['OLD']['mean_p18'], 1e-6)
    print(f"  shift_factor (BT/LIVE mean_p18): {shift_old:.1f}x")

    print(f"\n--- NEW (sib抜き) ---")
    print(f"  BT 2025 OOS:    winner_top1={bt_new_winner_top1:.4f} mean_p18={bt_new_mean_p18:.4f}")
    print(f"  LIVE 5/2-5/3:   winner_top1={metrics['live']['NEW']['winner_top1']:.4f} mean_p18={metrics['live']['NEW']['mean_p18']:.4f}")
    shift_new = bt_new_mean_p18 / max(metrics['live']['NEW']['mean_p18'], 1e-6)
    print(f"  shift_factor (BT/LIVE mean_p18): {shift_new:.1f}x")

    metrics['bt'] = {
        'OLD': {'winner_top1': float(bt_old_winner_top1), 'mean_p18': float(bt_old_mean_p18)},
        'NEW': {'winner_top1': float(bt_new_winner_top1), 'mean_p18': float(bt_new_mean_p18)},
    }
    metrics['shift_factor'] = {
        'OLD': float(shift_old),
        'NEW': float(shift_new),
        'improvement': float(shift_old - shift_new),
    }

    # ===== 仮説判定 =====
    print("\n" + "=" * 60)
    print("仮説判定 (a / b / c)")
    print("=" * 60)

    delta_winner_top1 = metrics['live']['NEW']['winner_top1'] - metrics['live']['OLD']['winner_top1']
    print(f"\nLIVE winner_top1 Δ: {delta_winner_top1:+.4f} ({delta_winner_top1*100:+.1f}pt)")
    print(f"shift_factor 改善: OLD {shift_old:.1f}x → NEW {shift_new:.1f}x ({shift_old-shift_new:+.1f}x)")

    if delta_winner_top1 >= 0.035:
        hypothesis = 'a'
        verdict = 'sib リーク仮説 正しい (LIVE で sib抜きが有意に向上)'
        recommend = '5/16 paper trading + 段階投入候補'
    elif delta_winner_top1 <= -0.05:
        hypothesis = 'b'
        verdict = 'sib は本番でも有効 (sib抜き で LIVE 悪化)'
        recommend = '5/16 NO-GO、 V18/V19 全面再検討'
    else:
        hypothesis = 'c'
        verdict = 'sib は ノイズ範囲 (LIVE で大差なし)'
        recommend = '5/16 paper trading 維持、 Phase 3 6-fold WF 集中'

    print(f"\n仮説: ({hypothesis}) {verdict}")
    print(f"5/16 推奨: {recommend}")

    metrics['hypothesis'] = hypothesis
    metrics['verdict'] = verdict
    metrics['recommend'] = recommend
    metrics['delta_winner_top1'] = float(delta_winner_top1)

    # ===== 出力 =====
    out_json = 'data/v18/v18v19_retraining/no_sib_live_retro_metrics.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[OK] {out_json}")

    out_md = 'data/v18/v18v19_retraining/no_sib_live_retro_analysis.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# V18/V19 sib抜き LIVE retro 分析 (Session #38 B)\n\n")
        f.write(f"**仮説**: ({hypothesis}) {verdict}\n\n")
        f.write(f"**5/16 推奨**: {recommend}\n\n")
        f.write("---\n\n## LIVE 5/2-5/3 比較\n\n")
        f.write("| metric | OLD (含 sib) | NEW (sib抜き) | Δ |\n")
        f.write("|--------|------------|--------------|---|\n")
        f.write(f"| winner_top1 | {metrics['live']['OLD']['winner_top1']:.4f} | {metrics['live']['NEW']['winner_top1']:.4f} | {delta_winner_top1:+.4f} |\n")
        f.write(f"| top3_hit_rate | {metrics['live']['OLD']['top3_hit_rate']:.4f} | {metrics['live']['NEW']['top3_hit_rate']:.4f} | {metrics['live']['NEW']['top3_hit_rate']-metrics['live']['OLD']['top3_hit_rate']:+.4f} |\n")
        if metrics['live']['OLD']['auc_v18']:
            f.write(f"| AUC v18 | {metrics['live']['OLD']['auc_v18']:.4f} | {metrics['live']['NEW']['auc_v18']:.4f} | {metrics['live']['NEW']['auc_v18']-metrics['live']['OLD']['auc_v18']:+.4f} |\n")
        f.write(f"| mean p18 | {metrics['live']['OLD']['mean_p18']:.4f} | {metrics['live']['NEW']['mean_p18']:.4f} | - |\n")
        f.write(f"| max p18 | {metrics['live']['OLD']['max_p18']:.4f} | {metrics['live']['NEW']['max_p18']:.4f} | - |\n")
        f.write("\n## BT 2025 OOS 比較\n\n")
        f.write("| metric | OLD (含 sib, Ens) | NEW (sib抜き, LGB) | Δ |\n")
        f.write("|--------|------------------|------------------|---|\n")
        f.write(f"| winner_top1 | {bt_old_winner_top1:.4f} | {bt_new_winner_top1:.4f} | {bt_new_winner_top1-bt_old_winner_top1:+.4f} |\n")
        f.write(f"| mean p18 | {bt_old_mean_p18:.4f} | {bt_new_mean_p18:.4f} | - |\n")
        f.write("\n## shift factor (BT/LIVE)\n\n")
        f.write(f"- OLD: {shift_old:.1f}x\n- NEW: {shift_new:.1f}x\n- 改善: {shift_old-shift_new:+.1f}x\n")
    print(f"[OK] {out_md}")
    return metrics


if __name__ == '__main__':
    main()
