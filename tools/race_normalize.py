"""race-level probability normalization.

usage:
  from tools.race_normalize import normalize_per_race
  df['p_norm'] = normalize_per_race(df, prob_col='p_tansho', race_col='race_id', method='softmax', T=1.0)

CLI:
  python tools/race_normalize.py <input_csv> --prob-col p_tansho --race-col race_id \
    --method softmax --T 1.0 --output <output_csv>

methods:
  softmax: logit(p)/T → softmax (default, sum=1, race-aware)
  power:   p^(1/T) / sum (sum=1, よりシンプル、log0 リスクなし)
  rank:    各レース max を target_max に linear rescale (sum 制約なし)

theory:
  retro 5/2-5/3 で BT 2025 OOS 比 race_max_p mean 27.7x 縮小確認 (data/v18/distribution_shift_analysis.json)。
  softmax T=1.0 で sum=1 強制 + 構造保持 → bet>0 化、retro 上 ROI 725%-1156% (sample 小)。
  詳細: data/v18/race_normalize_5_4_result.md

参考:
  - phase 2 BT で v18 calibration: Platt scaling 微改善 (max 0.94 → 0.95) のみ。retro 大幅 shift には対応不能。
  - normalize は raw model 出力を race内 で再分配するため calibration とは別レイヤ。併用可能。
"""
from __future__ import annotations

import sys, os, argparse
import numpy as np
import pandas as pd


def _softmax_per_race(df: pd.DataFrame, prob_col: str, race_col: str, T: float, eps: float) -> np.ndarray:
    new = np.zeros(len(df))
    for _, sub in df.groupby(race_col):
        idx = sub.index
        p = np.clip(sub[prob_col].astype(float).values, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p)) / max(T, 1e-6)
        ex = np.exp(logits - logits.max())  # numerical stability
        new[idx] = ex / ex.sum()
    return new


def _power_per_race(df: pd.DataFrame, prob_col: str, race_col: str, T: float, eps: float) -> np.ndarray:
    new = np.zeros(len(df))
    inv_T = 1.0 / max(T, 1e-6)
    for _, sub in df.groupby(race_col):
        idx = sub.index
        p = np.clip(sub[prob_col].astype(float).values, eps, 1.0)
        pT = p ** inv_T
        s = pT.sum()
        new[idx] = pT / s if s > 0 else pT
    return new


def _rank_scale_per_race(df: pd.DataFrame, prob_col: str, race_col: str, target_max: float) -> np.ndarray:
    new = np.zeros(len(df))
    for _, sub in df.groupby(race_col):
        idx = sub.index
        p = sub[prob_col].astype(float).values
        m = p.max()
        new[idx] = p * (target_max / m) if m > 0 else p
    return new


def normalize_per_race(
    df: pd.DataFrame,
    prob_col: str = 'p_tansho',
    race_col: str = 'race_id',
    method: str = 'softmax',
    T: float = 1.0,
    target_max: float = 0.347,
    eps: float = 1e-9,
) -> np.ndarray:
    """Race-level normalization. Returns numpy array aligned with df.index.

    Args:
        df: dataframe with prob and race_id columns. Must have unique row index per group.
        prob_col: probability column to normalize
        race_col: race grouping column
        method: 'softmax' | 'power' | 'rank'
        T: temperature (softmax/power のみ). T<1 sharpen, T>1 soften, T=1 = identity logit space
        target_max: rank method の各レース max prob 目標値 (BT 2025 OOS race_max_p mean = 0.347)
        eps: 数値安定化用

    Returns:
        np.ndarray of normalized probs (len = len(df))
    """
    if df[race_col].isna().any():
        raise ValueError(f"{race_col} に NaN が含まれています")

    # ensure unique index for assignment
    df_local = df.reset_index(drop=False).rename(columns={'index':'_orig_index'})
    if method == 'softmax':
        out = _softmax_per_race(df_local, prob_col, race_col, T, eps)
    elif method == 'power':
        out = _power_per_race(df_local, prob_col, race_col, T, eps)
    elif method == 'rank':
        out = _rank_scale_per_race(df_local, prob_col, race_col, target_max)
    else:
        raise ValueError(f"Unknown method: {method}. choose 'softmax' / 'power' / 'rank'")

    # restore order: df_local._orig_index → 元の order
    arr = np.zeros(len(df))
    for new_i, orig_i in enumerate(df_local['_orig_index'].values):
        # df の reset_index で 0..N-1 になっていない場合に備えて pos lookup
        # 簡易: df.index.get_loc(orig_i) — index が unique 前提
        arr[df.index.get_loc(orig_i)] = out[new_i]
    return arr


def main():
    p = argparse.ArgumentParser(description='Race-level probability normalization')
    p.add_argument('input', help='Input CSV path')
    p.add_argument('--prob-col', default='p_tansho')
    p.add_argument('--race-col', default='race_id')
    p.add_argument('--method', choices=['softmax','power','rank'], default='softmax')
    p.add_argument('--T', type=float, default=1.0, help='Temperature (softmax/power)')
    p.add_argument('--target-max', type=float, default=0.347, help='rank method 用 target max prob')
    p.add_argument('--output', default=None, help='Output CSV (default: <input>_norm.csv)')
    p.add_argument('--out-col', default=None, help='Output column name (default: <prob-col>_norm)')
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype={args.race_col: str})
    out_col = args.out_col or (args.prob_col + '_norm')
    df[out_col] = normalize_per_race(df, args.prob_col, args.race_col, args.method, args.T, args.target_max)

    out_path = args.output or (os.path.splitext(args.input)[0] + '_norm.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    # summary
    g = df.groupby(args.race_col)[out_col].agg(['max','sum','mean'])
    print(f"[OK] saved {out_path}")
    print(f"  method={args.method} T={args.T}")
    print(f"  race-level: max_mean={g['max'].mean():.3f}, sum_mean={g['sum'].mean():.3f}, p95_max={g['max'].quantile(0.95):.3f}")


if __name__ == '__main__':
    main()
