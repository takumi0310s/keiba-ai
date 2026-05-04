"""5/2-5/3 retro predictions に calibration 適用 + ROI再計算.

calibrator (Platt) を v18/v19 retro 確率に適用、
filter (p>=X, EV>=Y) で bet 数 / hit / ROI を before/after 比較.
"""
import sys, os, io, json, pickle
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

CALI_V18 = 'data/v18/models/v18_tansho_calibrator.pkl'
CALI_V19 = 'data/v18/models/v19_fukusho_calibrator.pkl'
RETRO_CSV = 'data/v18/v18_v19_retro_full_predictions.csv'


def apply_lr(lr, p_raw):
    eps = 1e-8
    p = np.clip(np.asarray(p_raw, dtype=float), eps, 1-eps)
    X = np.log(p / (1 - p)).reshape(-1, 1)
    return lr.predict_proba(X)[:, 1]


def main():
    print("=" * 60)
    print("Calibration 適用 5/2-5/3 retro 再評価")
    print("=" * 60)

    df = pd.read_csv(RETRO_CSV)
    print(f"Loaded retro: {len(df)} horses")

    with open(CALI_V18, 'rb') as f: lr_v18 = pickle.load(f)
    with open(CALI_V19, 'rb') as f: lr_v19 = pickle.load(f)
    print("Loaded calibrators v18/v19")

    df['p_tansho_cal'] = apply_lr(lr_v18, df['p_tansho'].values)
    df['p_fukusho_cal'] = apply_lr(lr_v19, df['p_fukusho'].values)
    df['ev_tansho_cal'] = df['p_tansho_cal'] * df['odds']
    df['ev_fukusho_cal'] = df['p_fukusho_cal'] * df['odds']

    # Distribution stats
    print("\n=== probability distribution ===")
    print(f"  v18 raw  : max={df['p_tansho'].max():.4f} mean={df['p_tansho'].mean():.4f}")
    print(f"  v18 cal  : max={df['p_tansho_cal'].max():.4f} mean={df['p_tansho_cal'].mean():.4f}")
    print(f"  v19 raw  : max={df['p_fukusho'].max():.4f} mean={df['p_fukusho'].mean():.4f}")
    print(f"  v19 cal  : max={df['p_fukusho_cal'].max():.4f} mean={df['p_fukusho_cal'].mean():.4f}")

    # === Filter sweeps (calibrated 版) ===
    BET = 100
    df_w = df[df['winner_known'] == 1].copy()

    print("\n=== v18 単勝 (calibrated) ===")
    print(f"{'p_min':>6} {'ev_min':>6} {'n_bet':>6} {'n_win':>6} {'inv':>8} {'pay':>10} {'ROI':>8}")
    for prob_t in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        for ev_t in [1.0, 1.1, 1.2, 1.5]:
            m = (df_w['p_tansho_cal'] >= prob_t) & (df_w['ev_tansho_cal'] >= ev_t) & (df_w['odds'] > 0)
            n_bet = int(m.sum())
            if n_bet == 0: continue
            wins = m & (df_w['is_win'] == 1)
            n_win = int(wins.sum())
            inv = n_bet * BET
            pay = float((wins.astype(int) * df_w['odds'] * BET).sum())
            roi = pay/inv*100 if inv>0 else 0
            print(f"  {prob_t:.2f}   {ev_t:.2f}   {n_bet:>5d}  {n_win:>5d}  {int(inv):>7,d}  {int(pay):>9,d}  {roi:>6.1f}%")

    print("\n=== v19 複勝 (calibrated, 簡易: 複勝odds = tansho × 0.3) ===")
    print(f"{'p_min':>6} {'ev_min':>6} {'n_bet':>6} {'n_hit':>6} {'inv':>8} {'pay~':>10} {'ROI~':>8}")
    for prob_f in [0.30, 0.40, 0.50, 0.60, 0.70]:
        for ev_f in [1.0, 1.1, 1.2]:
            m = (df['p_fukusho_cal'] >= prob_f) & (df['ev_fukusho_cal'] >= ev_f) & (df['odds'] > 0)
            n_bet = int(m.sum())
            if n_bet == 0: continue
            hits = m & (df['is_top3'] == 1)
            n_hit = int(hits.sum())
            inv = n_bet * BET
            pay = float((hits.astype(int) * df['odds'] * 0.3 * BET).sum())
            roi = pay/inv*100 if inv>0 else 0
            print(f"  {prob_f:.2f}   {ev_f:.2f}   {n_bet:>5d}  {n_hit:>5d}  {int(inv):>7,d}  {int(pay):>9,d}  {roi:>6.1f}%")

    df.to_csv('data/v18/v18_v19_retro_calibrated.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved data/v18/v18_v19_retro_calibrated.csv")


if __name__ == '__main__':
    main()
