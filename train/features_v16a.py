#!/usr/bin/env python3
"""V16a 新特徴量モジュール

4グループの能力系特徴量を追加:
  1. Jockey × Trainer combo  (jockey_trainer_top3_rate_exp, jockey_trainer_n_exp)
  2. 馬体重 slope / std       (horse_weight_slope_5, horse_weight_std_5)
  3. 正規化パフォーマンス     (horse_perf_score_exp) — ELO proxy
  4. 枠番バイアス             (gate_bias_top3_exp, gate_bias_n_exp)

全て expanding window (cumsum - current) でリークフリー。
Bayesian smoothing でサンプル不足時の過学習を抑制。

Usage (training):
    from train.features_v16a import compute_all_v16a_features, V16A_FEATURE_NAMES

Usage (lookup table generation):
    from train.features_v16a import build_lookup_tables, save_lookup_tables
"""
from __future__ import annotations

import json
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ── Feature name registry ──────────────────────────────────────────────────

V16A_FEATURE_NAMES = [
    'jockey_trainer_top3_rate_exp',
    'jockey_trainer_n_exp',
    'horse_weight_slope_5',
    'horse_weight_std_5',
    'horse_perf_score_exp',
    'gate_bias_top3_exp',
    'gate_bias_n_exp',
]

V16A_DEFAULTS = {
    'jockey_trainer_top3_rate_exp': 0.33,
    'jockey_trainer_n_exp': 0,
    'horse_weight_slope_5': 0.0,
    'horse_weight_std_5': 5.0,
    'horse_perf_score_exp': 0.33,
    'gate_bias_top3_exp': 0.33,
    'gate_bias_n_exp': 0,
}

# Bayesian smoothing priors
_JT_PRIOR_TOP3 = 0.33
_JT_ALPHA = 30
_PERF_PRIOR = 0.33
_PERF_ALPHA = 10
_GATE_PRIOR_TOP3 = 0.33
_GATE_ALPHA = 50


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if 'is_top3' not in df.columns:
        df['is_top3'] = (df['finish'].between(1, 3)).astype(int)
    if 'year_full' not in df.columns:
        df['year_full'] = df['year'].astype(int) + 2000
    if 'date_num' not in df.columns:
        df['date_num'] = df['year_full'] * 10000 + df['month'].astype(int) * 100 + df['day'].astype(int)
    return df


# ── 1. Jockey × Trainer combo ─────────────────────────────────────────────

def compute_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding window jockey×trainer top3 rate with Bayesian smoothing."""
    print("  [v16a] jockey × trainer combo features...")
    df = _ensure_cols(df)
    df = df.sort_values('date_num').reset_index(drop=True)

    jt_key = ['jockey_id', 'trainer_id']
    df['_jt_cumcnt'] = df.groupby(jt_key).cumcount()           # races BEFORE this
    df['_jt_cumtop3'] = (
        df.groupby(jt_key)['is_top3'].cumsum() - df['is_top3']  # top3 BEFORE this
    )

    df['jockey_trainer_n_exp'] = df['_jt_cumcnt'].astype(float)
    df['jockey_trainer_top3_rate_exp'] = (
        (df['_jt_cumtop3'] + _JT_PRIOR_TOP3 * _JT_ALPHA) /
        (df['_jt_cumcnt'] + _JT_ALPHA)
    )

    drop = [c for c in ['_jt_cumcnt', '_jt_cumtop3'] if c in df.columns]
    df.drop(columns=drop, inplace=True)

    fill = (df['jockey_trainer_top3_rate_exp'] > 0).mean()
    print(f"    jt top3 rate fill: {fill*100:.1f}%  n_exp mean: {df['jockey_trainer_n_exp'].mean():.1f}")
    return df


# ── 2. 馬体重 slope & std (直近 5走) ──────────────────────────────────────

def compute_weight_slope_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horse weight trend over last 5 previous races (Pattern A — no current weight)."""
    print("  [v16a] horse weight slope/std features...")
    df = _ensure_cols(df)
    df = df.sort_values(['horse_id', 'date_num']).reset_index(drop=True)

    g = df.groupby('horse_id')['horse_weight']
    lags = {}
    for k in range(1, 6):
        lags[k] = g.shift(k)

    hw_mat = pd.concat(lags, axis=1).values.astype(float)  # (N, 5): lag1..lag5

    n_rows = len(hw_mat)
    slopes = np.full(n_rows, np.nan)
    stds   = np.full(n_rows, np.nan)
    x = np.array([5, 4, 3, 2, 1], dtype=float)  # lag5 oldest, lag1 newest
    x_mean = x.mean()
    x_dev = x - x_mean
    ss_x = (x_dev ** 2).sum()

    for i in range(n_rows):
        row = hw_mat[i]
        valid_mask = ~np.isnan(row)
        n_valid = valid_mask.sum()
        if n_valid < 2:
            continue
        y = row[valid_mask]
        xv = x[valid_mask]
        xv_dev = xv - xv.mean()
        ss_xv = (xv_dev ** 2).sum()
        if ss_xv == 0:
            slopes[i] = 0.0
        else:
            slopes[i] = (xv_dev * (y - y.mean())).sum() / ss_xv
        stds[i] = y.std()

    df['horse_weight_slope_5'] = slopes
    df['horse_weight_std_5']   = stds

    # fillna: slope=0 (neutral), std=5 (typical variability)
    df['horse_weight_slope_5'].fillna(0.0, inplace=True)
    df['horse_weight_std_5'].fillna(5.0, inplace=True)

    fill = (~pd.isna(pd.Series(slopes))).mean()
    print(f"    weight slope fill: {fill*100:.1f}%  slope mean: {np.nanmean(slopes):.3f}")
    return df


# ── 3. 正規化パフォーマンス (ELO proxy) ──────────────────────────────────

def compute_perf_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding normalized performance score: (num_horses - finish) / (num_horses - 1).

    Score = 1.0 for winner, 0.0 for last. Bayesian prior = 0.33.
    Captures quality of performance across all finishes (not just binary win/top3).
    """
    print("  [v16a] horse perf score (ELO proxy) features...")
    df = _ensure_cols(df)
    df = df.sort_values(['horse_id', 'date_num']).reset_index(drop=True)

    # Normalised finish score (avoid div0 for single-horse races)
    denom = (df['num_horses'] - 1).clip(lower=1)
    df['_norm_finish'] = ((df['num_horses'] - df['finish']) / denom).clip(0, 1)

    g = df.groupby('horse_id')
    df['_pf_cumcnt']  = g.cumcount()
    df['_pf_cumsum']  = g['_norm_finish'].cumsum() - df['_norm_finish']

    df['horse_perf_score_exp'] = (
        (df['_pf_cumsum'] + _PERF_PRIOR * _PERF_ALPHA) /
        (df['_pf_cumcnt'] + _PERF_ALPHA)
    )

    drop = [c for c in ['_norm_finish', '_pf_cumcnt', '_pf_cumsum'] if c in df.columns]
    df.drop(columns=drop, inplace=True)

    print(f"    perf score mean: {df['horse_perf_score_exp'].mean():.3f}  "
          f"std: {df['horse_perf_score_exp'].std():.3f}")
    return df


# ── 4. 枠番バイアス ───────────────────────────────────────────────────────

def _dist_bin(distance):
    """400m刻みの distance bucket。"""
    return (distance // 400) * 400


def compute_gate_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding window gate/post position top3 rate by (course, dist_bin, surface, umaban)."""
    print("  [v16a] gate bias features...")
    df = _ensure_cols(df)
    df = df.sort_values('date_num').reset_index(drop=True)

    df['_dist_bin'] = df['distance'].apply(_dist_bin)
    gate_key = ['course', '_dist_bin', 'surface', 'umaban']

    df['_gate_cumcnt']  = df.groupby(gate_key).cumcount()
    df['_gate_cumtop3'] = (
        df.groupby(gate_key)['is_top3'].cumsum() - df['is_top3']
    )

    df['gate_bias_n_exp'] = df['_gate_cumcnt'].astype(float)
    df['gate_bias_top3_exp'] = (
        (df['_gate_cumtop3'] + _GATE_PRIOR_TOP3 * _GATE_ALPHA) /
        (df['_gate_cumcnt'] + _GATE_ALPHA)
    )

    drop = [c for c in ['_dist_bin', '_gate_cumcnt', '_gate_cumtop3'] if c in df.columns]
    df.drop(columns=drop, inplace=True)

    print(f"    gate bias top3 mean: {df['gate_bias_top3_exp'].mean():.3f}  "
          f"std: {df['gate_bias_top3_exp'].std():.3f}")
    return df


# ── Master compute function ────────────────────────────────────────────────

def compute_all_v16a_features(df: pd.DataFrame) -> pd.DataFrame:
    """全 v16a 特徴量を一括計算。訓練 DataFrame を返す。"""
    print("[v16a] Computing all V16a features...")
    df = _ensure_cols(df)
    df = compute_jockey_trainer_features(df)
    df = compute_weight_slope_features(df)
    df = compute_perf_score_features(df)
    df = compute_gate_bias_features(df)
    print(f"[v16a] Done. Added {len(V16A_FEATURE_NAMES)} features.")
    return df


# ── Lookup table generation (for inference) ───────────────────────────────

def build_lookup_tables(df: pd.DataFrame) -> dict:
    """学習後の df から推論用 lookup tables を生成。

    Returns:
        {
          'jt': {(jockey_id, trainer_id): {top3_rate, n}},
          'perf': {horse_id: score},
          'weight': {horse_id: {slope, std, last_weight}},
          'gate': {(course, dist_bin, surface, umaban): {top3_rate, n}},
          'built_at': ISO timestamp,
        }
    """
    print("[v16a] Building lookup tables from training data...")

    df = _ensure_cols(df)
    df = df.sort_values('date_num')

    # ── jt lookup: latest expanding values per (jockey_id, trainer_id) ──
    jt_key = ['jockey_id', 'trainer_id']
    jt_last = (
        df.sort_values('date_num')
          .groupby(jt_key)
          .agg(
              top3_rate=('jockey_trainer_top3_rate_exp', 'last'),
              n=('jockey_trainer_n_exp', 'last'),
          )
          .reset_index()
    )
    jt_lookup = {
        (str(r.jockey_id), str(r.trainer_id)): {
            'top3_rate': float(r.top3_rate),
            'n': int(r.n),
        }
        for r in jt_last.itertuples()
    }

    # ── perf lookup: latest score per horse ──
    perf_last = (
        df.groupby('horse_id')['horse_perf_score_exp']
          .last()
          .reset_index()
    )
    perf_lookup = {
        str(r.horse_id): float(r.horse_perf_score_exp)
        for r in perf_last.itertuples()
    }

    # ── weight lookup: last 5 weights per horse → slope, std, last_weight ──
    df_sorted = df.sort_values(['horse_id', 'date_num'])
    x = np.array([5, 4, 3, 2, 1], dtype=float)
    weight_lookup = {}
    for hid, grp in df_sorted.groupby('horse_id'):
        weights = grp['horse_weight'].dropna().values[-5:]
        if len(weights) < 2:
            weight_lookup[str(hid)] = {'slope': 0.0, 'std': 5.0, 'last': float(weights[-1]) if len(weights) else 460.0}
            continue
        xv = x[-len(weights):]
        xv_dev = xv - xv.mean()
        ss = (xv_dev ** 2).sum()
        slope = float((xv_dev * (weights - weights.mean())).sum() / ss) if ss > 0 else 0.0
        weight_lookup[str(hid)] = {
            'slope': slope,
            'std': float(weights.std()),
            'last': float(weights[-1]),
        }

    # ── gate lookup: latest expanding values per (course, dist_bin, surface, umaban) ──
    df['dist_bin_lkp'] = df['distance'].apply(_dist_bin)
    gate_key = ['course', 'dist_bin_lkp', 'surface', 'umaban']
    gate_last = (
        df.sort_values('date_num')
          .groupby(gate_key)
          .agg(
              top3_rate=('gate_bias_top3_exp', 'last'),
              n=('gate_bias_n_exp', 'last'),
          )
          .reset_index()
    )
    gate_lookup = {
        (str(r.course), int(r.dist_bin_lkp), str(r.surface), int(r.umaban)): {
            'top3_rate': float(r.top3_rate),
            'n': int(r.n),
        }
        for r in gate_last.itertuples()
    }

    import datetime
    return {
        'jt': jt_lookup,
        'perf': perf_lookup,
        'weight': weight_lookup,
        'gate': gate_lookup,
        'built_at': datetime.datetime.now().isoformat(),
    }


def save_lookup_tables(tables: dict, path: str | None = None) -> str:
    """lookup tables を pickle で保存。パスを返す。"""
    if path is None:
        path = os.path.join(DATA_DIR, 'v16a_lookup_tables.pkl')
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(tables, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[v16a] Lookup tables saved → {path} ({size_mb:.1f} MB)")
    return path


def load_lookup_tables(path: str | None = None) -> dict:
    """lookup tables を読み込む。"""
    if path is None:
        path = os.path.join(DATA_DIR, 'v16a_lookup_tables.pkl')
    if not os.path.exists(path):
        return {}
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
