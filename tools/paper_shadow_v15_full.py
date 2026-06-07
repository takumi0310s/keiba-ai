"""
Paper shadow prediction for v15_full and v15.2 candidate models.
Runs in parallel with V15 production (no real cash impact).
D-3 2026-05-19

V15 .pkl.gz / predict_core / app.py 完全不変。
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_MODELS = {
    'v15_full_optuna': BASE_DIR / 'models' / 'v15_full_optuna_candidate.pkl.gz',
    'v15_2': BASE_DIR / 'models' / 'v15_2_candidate.pkl.gz',
    # v22_top100 DISABLED 2026-05-26: track_lap POST-RACE leak confirmed (pace_diff_race/
    # pace_second_half/lap_last_3f_race are current-race lap joined by race_id → post-race).
    # Retro 2025: V22 (95 feats, missing leaks) AUC delta = -0.0062 vs V15 → worse.
    # DO NOT re-enable without clean rebuild (prev_race lap shift required).
    'v20_base': BASE_DIR / 'keiba_model_v20_base_central.pkl.gz',
    # V16: オッズ系8件除外 ability model (WF 0.8677 ≈ V15 0.8678), T4 PASS
    # paper shadow のみ、本番投票なし。採用基準: N≥30後 人気乖離ROI > V15 かつ TOP1連対率≥40%
    'v16_ability': BASE_DIR / 'models' / 'v16_ability_candidate.pkl.gz',
    # V16a: V16 + jockey×trainer/weight_slope/perf_score/gate_bias (4 new ability features)
    # paper shadow only. 採用基準: WF AUC >= V16 0.8677 かつ N≥30後 人気乖離ROI改善
    'v16a_ability': BASE_DIR / 'models' / 'v16a_candidate.pkl.gz',
}

# V20-specific feature fill values (global means from training data)
# Used when features_df does not contain V20-only columns.
_V20_FEATURE_FILLS = {
    'has_lap_data': 0.457,             # 45.7% fill rate overall
    'sire_shinba_top3r_exp': 0.242,    # global mean
    'sire_shinba_runs_exp': 20.0,      # approximate mean prior shinba runs
    'sib_top3_rate_exp': 0.274,        # global mean
    'sib_shinba_wr_exp': 0.100,        # global mean
    'sib_total_races_exp': 28.8,       # global mean
    'sib_total_offspring_exp': 3.1,    # global mean
    'track_index_nk': -7.7,           # global mean (neutral track)
    'jrdb_prev_ten_idx': -5.21,      # global mean
    'jrdb_prev_agari_idx': -5.32,    # global mean
}


def load_candidate_model(model_key: str):
    """Load a candidate model by key. Returns None if not available."""
    path = CANDIDATE_MODELS.get(model_key)
    if path is None or not path.exists():
        return None
    try:
        with gzip.open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"[paper_shadow] Failed to load {model_key}: {e}")
        return None


def _augment_v16a(features_df, race_info: dict):
    """V16a 推論用特徴量を features_df に付加。lookup がなければ default で埋める。"""
    try:
        _tools_dir = str(BASE_DIR / 'tools')
        import sys
        if _tools_dir not in sys.path:
            sys.path.insert(0, _tools_dir)
        from features_v16a_lookup import augment_features_v16a
        return augment_features_v16a(features_df.copy(), race_info)
    except Exception as e:
        print(f"[paper_shadow] v16a augment warn: {e}")
        from features_v16a_lookup import V16A_DEFAULTS
        df = features_df.copy()
        for feat, val in V16A_DEFAULTS.items():
            if feat not in df.columns:
                df[feat] = val
        return df


def predict_paper_shadow(features_df, model_key: str = 'v15_full_optuna',
                         race_info: dict | None = None):
    """
    Generate paper shadow prediction using candidate model.

    Args:
        features_df: pandas DataFrame with same features as V15
        model_key: which candidate model to use
        race_info: race metadata dict (course, distance, surface) for V16a lookup

    Returns:
        dict {index: score} or None if model unavailable / prediction fails
    """
    model = load_candidate_model(model_key)
    if model is None:
        return None

    try:
        import numpy as np
        import xgboost as xgb

        # V16 / V16a style: dict with 'model' (LGB Booster) + optional 'xgb_model' + 'features'
        if isinstance(model, dict) and 'model' in model and 'features' in model:
            model_feats = model['features']
            # V16a: augment with new ability features from lookup tables
            if model_key == 'v16a_ability':
                df_aug = _augment_v16a(features_df, race_info or {})
            else:
                df_aug = features_df
            df_in = df_aug.reindex(columns=model_feats).fillna(0.0)
            X = df_in.values.astype(np.float32)
            p_lgb = model['model'].predict(X)
            w = model.get('ensemble_weights', {'lgb': 0.5, 'xgb': 0.5})
            if 'xgb_model' in model:
                p_xgb = model['xgb_model'].predict(xgb.DMatrix(X, feature_names=model_feats))
                scores = w.get('lgb', 0.5) * p_lgb + w.get('xgb', 0.5) * p_xgb
            else:
                scores = p_lgb
            return {i: float(s) for i, s in enumerate(scores)}

        # V20/V22: dict with lgb_model + xgb_model + features
        if isinstance(model, dict) and 'lgb_model' in model and 'xgb_model' in model:
            model_feats = model.get('features', [])
            # Augment features_df with V20-specific fills for missing columns
            df_aug = features_df.copy()
            for col, fill_val in _V20_FEATURE_FILLS.items():
                if col not in df_aug.columns:
                    df_aug[col] = fill_val
            df_in = df_aug.reindex(columns=model_feats).fillna(0.0)
            X = df_in.values.astype(np.float32)
            p_lgb = model['lgb_model'].predict(X)
            # ★feature_names を渡す(V16経路と同様)。 未指定だと特徴名付きboosterが
            #   「data did not contain feature names...」で毎R失敗→ログ汚染+V20/V22 shadow欠測★
            p_xgb = model['xgb_model'].predict(xgb.DMatrix(X, feature_names=model_feats) if model_feats
                                               else xgb.DMatrix(X))
            scores = 0.5 * p_lgb + 0.5 * p_xgb
            return {i: float(s) for i, s in enumerate(scores)}

        # Try common prediction interfaces
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_df)
            scores = probs[:, 1] if probs.ndim > 1 else probs
        elif hasattr(model, 'predict'):
            scores = model.predict(features_df)
        else:
            return None

        return {i: float(s) for i, s in enumerate(scores)}
    except Exception as e:
        print(f"[paper_shadow] Prediction failed for {model_key}: {e}")
        return None


def run_paper_shadow_comparison(race_id, features_df, v15_predictions: list,
                                race_info: dict | None = None) -> dict:
    """
    Compare V15 production predictions with candidate model predictions.

    Args:
        race_id: race identifier (str)
        features_df: feature DataFrame used for V15 prediction
        v15_predictions: V15 predictions list of dicts with horse_num / score
        race_info: race metadata dict (course, distance, surface) for V16a lookup

    Returns:
        dict with comparison results
    """
    result = {
        'race_id': str(race_id),
        'timestamp': datetime.now().isoformat(),
        'v15_top3': [
            p.get('horse_num') or p.get('umaban')
            for p in sorted(v15_predictions,
                            key=lambda x: x.get('score', 0), reverse=True)[:3]
        ],
        'paper_shadows': {},
    }

    for model_key in CANDIDATE_MODELS:
        shadow = predict_paper_shadow(features_df, model_key, race_info=race_info)
        if shadow is None:
            continue
        top3 = sorted(shadow.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_nums = [h for h, _ in top3]
        result['paper_shadows'][model_key] = {
            'top3': top3_nums,
            'scores': {str(h): s for h, s in top3},
            'agree_with_v15': set(top3_nums) == set(result['v15_top3']),
        }

    return result


def log_paper_shadow(result: dict, log_dir=None) -> None:
    """Save paper shadow comparison to daily JSONL log file."""
    if log_dir is None:
        log_dir = BASE_DIR / 'data' / 'paper_shadow_log'

    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = Path(log_dir) / f'shadow_{date_str}.jsonl'

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def get_paper_shadow_stats(days: int = 7) -> dict:
    """
    Compute paper shadow agreement stats from log files.

    Returns dict {model_key: {n_agree, n_total}} for last `days` days.
    """
    log_dir = BASE_DIR / 'data' / 'paper_shadow_log'
    if not log_dir.exists():
        return {}

    cutoff = datetime.now() - timedelta(days=days)
    stats: dict = {}

    for log_file in sorted(log_dir.glob('shadow_*.jsonl')):
        try:
            date_str = log_file.stem.replace('shadow_', '')
            file_date = datetime.strptime(date_str, '%Y%m%d')
        except Exception:
            continue
        if file_date < cutoff:
            continue

        try:
            with open(log_file, encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        for model_key, shadow in rec.get('paper_shadows', {}).items():
                            if model_key not in stats:
                                stats[model_key] = {'n_agree': 0, 'n_total': 0}
                            stats[model_key]['n_total'] += 1
                            if shadow.get('agree_with_v15'):
                                stats[model_key]['n_agree'] += 1
                    except Exception:
                        pass
        except Exception:
            pass

    return stats


if __name__ == '__main__':
    # Status check
    print("=== Paper Shadow Model Status ===")
    for key, path in CANDIDATE_MODELS.items():
        exists = path.exists()
        size_mb = path.stat().st_size / 1e6 if exists else 0.0
        print(f"  {key}: {'OK' if exists else 'NOT FOUND'} {size_mb:.1f}MB")

    stats = get_paper_shadow_stats()
    print("\n=== Agreement Stats (last 7 days) ===")
    if not stats:
        print("  (no log data yet)")
    for key, s in stats.items():
        rate = s['n_agree'] / s['n_total'] * 100 if s['n_total'] > 0 else 0.0
        print(f"  {key}: {s['n_agree']}/{s['n_total']} = {rate:.1f}% agree with V15")
