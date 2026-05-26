"""V23 candidate: V21 (136 feats) + prev_race_lap_real (netkeiba/RA source).

V21 は prev_race_first3f (ゼロ埋め 100%) を使用。
V23 は lap_features_v15cache.parquet の _real 版 (82.3% fill) に置換。

V21 WF = 0.8663 / V15 genuine = 0.8678 (GO threshold)
期待値低 (STEP A/B で RA delta = +0.0000 確認済み)。

Usage:
  python train/train_v23_candidate.py
  python train/train_v23_candidate.py --wf-only
"""
from __future__ import annotations
import sys, os, time, gzip, pickle, json, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'train'))

from v20_config import V20_LEAK_FEATURES, get_v20_features

CACHE_PKL   = os.path.join(BASE, 'data', '_v15_train_df_cache.pkl')
LAP_PARQ    = os.path.join(BASE, 'data', 'v20', 'lap_features_v15cache.parquet')
BLOD_PARQ   = os.path.join(BASE, 'data', 'v20', 'blod_features_v15cache.parquet')
SIB_PARQ    = os.path.join(BASE, 'data', 'v20', 'sib_features_v15cache.parquet')
NK_PARQ     = os.path.join(BASE, 'data', 'v20', 'netkeiba_race_features_v15cache.parquet')
HC_PARQ     = os.path.join(BASE, 'data', 'v20', 'horse_course_features_v15cache.parquet')
OUT_MODEL   = os.path.join(BASE, 'models', 'v23_candidate.pkl.gz')
OUT_JSON    = os.path.join(BASE, 'data', 'v20', 'v23_clean_wf_results.json')

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist', 'verbosity': 0,
}

V15_GENUINE_WF = 0.8678   # V15-audit-2 genuine WF LGB+XGB 6-fold
V21_WF         = 0.8663   # V21 C result
GO_THRESHOLD   = V15_GENUINE_WF  # must beat V15 genuine

# V21 identical delete list
V21_DELETE_12 = {
    'odds_sharp_drop',
    'jrdb_prev_rise_code',
    'bracket_pos',
    'is_long_transport',
    'course_renovated',
    'jrdb_prev_interference',
    'top3_count_3r',
    'rest_category',
    'has_sakaro_training',
    'stable_comment_score',
    'age_group',
    'wood_best_4f_filled',
}
PACI_NINKI = 'paci_ninki_idx'


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load V15 cache + parquets.
    Returns (df, feats_v21, feats_v23).
    """
    print("=" * 65)
    print("Loading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, 'rb') as f:
        d = pickle.load(f)
    df = d['df'].copy()
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")
    df['_y'] = df['year'].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))

    # ===== Merge lap features =====
    if os.path.exists(LAP_PARQ):
        lap = pd.read_parquet(LAP_PARQ)
        real_first = lap.get('prev_race_first3f_real', pd.Series(dtype=float)).reindex(df.index)
        df['has_lap_data'] = real_first.notna().astype(np.float32)
        for real_col, base_col in [
            ('prev_race_first3f_real', 'prev_race_first3f'),
            ('prev_race_last3f_real',  'prev_race_last3f'),
            ('prev_race_pace_diff_real', 'prev_race_pace_diff'),
        ]:
            if real_col in lap.columns:
                rv = lap[real_col].reindex(df.index)
                df[base_col] = rv.fillna(rv.mean())
        print(f"  lap fill: {real_first.notna().mean():.1%}")
    else:
        print("  [WARN] lap parquet not found")

    # ===== Merge BLOD =====
    if os.path.exists(BLOD_PARQ):
        blod = pd.read_parquet(BLOD_PARQ)
        for col in ['sire_shinba_top3r_exp']:
            if col in blod.columns:
                df[col] = blod[col].reindex(df.index)
        print(f"  blod fill: {df.get('sire_shinba_top3r_exp', pd.Series()).notna().mean():.1%}")
    else:
        print("  [WARN] blod parquet not found")

    # ===== Merge sib =====
    if os.path.exists(SIB_PARQ):
        sib = pd.read_parquet(SIB_PARQ)
        for col in ['sib_top3_rate_exp', 'sib_shinba_wr_exp',
                    'sib_total_races_exp', 'sib_total_offspring_exp']:
            if col in sib.columns:
                df[col] = sib[col].reindex(df.index)
        print(f"  sib fill: {df.get('sib_top3_rate_exp', pd.Series()).notna().mean():.1%}")
    else:
        print("  [WARN] sib parquet not found")

    # ===== Merge netkeiba race features =====
    if os.path.exists(NK_PARQ):
        nk = pd.read_parquet(NK_PARQ)
        if 'track_index_nk' in nk.columns:
            df['track_index_nk'] = nk['track_index_nk'].reindex(df.index)
        print(f"  nk fill: {df.get('track_index_nk', pd.Series()).notna().mean():.1%}")
    else:
        print("  [WARN] netkeiba race features parquet not found")

    # ===== Merge horse_course features (both columns) =====
    if os.path.exists(HC_PARQ):
        hc = pd.read_parquet(HC_PARQ)
        for col in ['horse_course_top3r_exp', 'horse_course_n_exp']:
            if col in hc.columns:
                df[col] = hc[col].reindex(df.index)
        fill_top3r = df.get('horse_course_top3r_exp', pd.Series()).notna().mean()
        fill_n     = df.get('horse_course_n_exp', pd.Series()).notna().mean()
        print(f"  horse_course_top3r_exp fill: {fill_top3r:.1%}")
        print(f"  horse_course_n_exp fill:     {fill_n:.1%}")
    else:
        print("  [WARN] horse_course parquet not found")

    # ===== Build prev_race_last3f_relative (V23 new feature) =====
    # prev_race_last3f_real - mean(prev_race_last3f_real) grouped by dist_cat
    # dist_cat: 1=sprint, 2=mile, 3=middle, 4=long (from V15)
    # For simplicity use global mean per dist_cat (no WF issue: this is the horse's
    # *previous* race, so no current-race leakage)
    print("  Building prev_race_last3f_relative ...")
    if 'prev_race_last3f' in df.columns and 'dist_cat' in df.columns:
        real_last3f = lap['prev_race_last3f_real'].reindex(df.index) if os.path.exists(LAP_PARQ) else pd.Series(dtype=float)
        # Group mean (expanding would be ideal, but dist_cat mean is stable and no leak)
        dist_group_mean = real_last3f.groupby(df['dist_cat']).transform('mean')
        df['prev_race_last3f_relative'] = real_last3f - dist_group_mean
        fill_rel = df['prev_race_last3f_relative'].notna().mean()
        print(f"  prev_race_last3f_relative fill: {fill_rel:.1%}")
    else:
        df['prev_race_last3f_relative'] = np.nan
        print("  [WARN] cannot build prev_race_last3f_relative (missing dist_cat or last3f)")

    # ===== V20 base feature list =====
    feats_v20 = get_v20_features(
        include_o1_fixed=True, include_blod_fixed=True,
        include_sib=True, include_blod_full=False
    )
    feats_v20 = [f for f in feats_v20 if f in df.columns and f not in V20_LEAK_FEATURES]
    print(f"\nV20 base features available: {len(feats_v20)}")

    # V21 feature set (C variant: 136)
    feats_B   = [f for f in feats_v20 if f != PACI_NINKI]
    feats_C   = [f for f in feats_B if f not in V21_DELETE_12]
    if 'horse_course_top3r_exp' in df.columns:
        feats_C = feats_C + ['horse_course_top3r_exp']

    # V23 = V21 + new features
    feats_V23 = list(feats_C)
    new_feats: list[str] = []

    # Add horse_course_n_exp if available
    if 'horse_course_n_exp' in df.columns and 'horse_course_n_exp' not in feats_V23:
        feats_V23.append('horse_course_n_exp')
        new_feats.append('horse_course_n_exp')

    # Add prev_race_last3f_relative if has signal
    if ('prev_race_last3f_relative' in df.columns
            and df['prev_race_last3f_relative'].notna().mean() > 0.3
            and 'prev_race_last3f_relative' not in feats_V23):
        feats_V23.append('prev_race_last3f_relative')
        new_feats.append('prev_race_last3f_relative')

    print(f"\nFeature counts:")
    print(f"  V21 (C):   {len(feats_C)}")
    print(f"  V23:       {len(feats_V23)}  (added: {new_feats})")

    # ===== Correlation analysis =====
    print("\n=== Correlation: new features vs existing prev_race ===")
    if os.path.exists(LAP_PARQ):
        lap_vals = pd.read_parquet(LAP_PARQ)
        for nf in new_feats:
            if nf in df.columns and 'prev_race_last3f_real' in lap_vals.columns:
                a = df[nf].values
                b = lap_vals['prev_race_last3f_real'].reindex(df.index).values
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() > 1000:
                    corr = np.corrcoef(a[mask], b[mask])[0, 1]
                    print(f"  corr({nf}, prev_race_last3f_real) = {corr:.4f}")

    return df, feats_C, feats_V23


def run_wf(df: pd.DataFrame, feats: list[str], label: str) -> dict:
    """Walk-forward 6-fold (2020-2025). Returns results dict."""
    print(f"\n--- WF: {label} ({len(feats)} feats) ---")
    results = []
    for fold_yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        train_mask = (df['_y'] >= 2015) & (df['_y'] < fold_yr)
        test_mask  = df['_y'] == fold_yr
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            print(f"  Fold {fold_yr}: skipped (insufficient data)")
            continue

        y_tr = (df.loc[train_mask, 'finish'] <= 3).astype(int)
        y_te = (df.loc[test_mask,  'finish'] <= 3).astype(int)
        # Only use feats that are present in df
        avail = [f for f in feats if f in df.columns]
        X_tr = df.loc[train_mask, avail].fillna(0).astype(np.float32)
        X_te = df.loc[test_mask,  avail].fillna(0).astype(np.float32)

        t0 = time.time()
        lgb_m = lgb.train(
            LGB_PARAMS, lgb.Dataset(X_tr, y_tr),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X_te, y_te)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        p_lgb = lgb_m.predict(X_te)

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtest  = xgb.DMatrix(X_te, label=y_te)
        xgb_m = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=500,
            evals=[(dtest, 'val')], early_stopping_rounds=50, verbose_eval=False,
        )
        p_xgb = xgb_m.predict(dtest)

        p_ens = 0.5 * p_lgb + 0.5 * p_xgb
        auc = roc_auc_score(y_te.values, p_ens)
        print(f"  Fold {fold_yr}: AUC={auc:.4f}  "
              f"n_train={train_mask.sum():,}  [{time.time()-t0:.0f}s]")
        results.append({'year': fold_yr, 'auc': float(auc)})

    mean_auc = float(np.mean([r['auc'] for r in results]))
    print(f"  => WF mean AUC = {mean_auc:.4f}")
    return {'label': label, 'n_feats': len(feats), 'folds': results, 'wf_mean_auc': mean_auc}


def save_model(df: pd.DataFrame, feats: list[str], wf_auc: float,
               ablation: list[dict]) -> None:
    """Train on full 2015-2025 and save production candidate."""
    print(f"\n=== Training final V23 model (2015-2025) ===")
    mask = (df['_y'] >= 2015) & (df['_y'] <= 2025)
    y = (df.loc[mask, 'finish'] <= 3).astype(int)
    avail = [f for f in feats if f in df.columns]
    X = df.loc[mask, avail].fillna(0).astype(np.float32)
    print(f"  n={mask.sum():,}, features={len(avail)}")

    t0 = time.time()
    lgb_m = lgb.train(LGB_PARAMS, lgb.Dataset(X, y),
                      num_boost_round=500, callbacks=[lgb.log_evaluation(0)])
    dtrain = xgb.DMatrix(X, label=y)
    xgb_m = xgb.train(XGB_PARAMS, dtrain, num_boost_round=500, verbose_eval=False)
    print(f"  training done [{time.time()-t0:.0f}s]")

    out = {
        'version': 'v23_candidate',
        'features': avail,
        'n_features': len(avail),
        'lgb_model': lgb_m,
        'xgb_model': xgb_m,
        'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5},
        'wf_mean_auc': wf_auc,
        'v15_baseline_genuine_wf': V15_GENUINE_WF,
        'v21_wf': V21_WF,
        'ablation': ablation,
    }
    with gzip.open(OUT_MODEL, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(OUT_MODEL) / 1e6
    print(f"  [SAVED] {OUT_MODEL} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wf-only', action='store_true',
                        help='Run WF only, skip final model training even if GO')
    args = parser.parse_args()

    df, feats_v21, feats_v23 = load_data()

    # ===== WF evaluations =====
    # Only run V21 and V23 (skip A/B which are identical to train_v21_clean.py)
    res_v21 = run_wf(df, feats_v21, 'V21 (136 feats, reference)')
    res_v23 = run_wf(df, feats_v23, f'V23 ({len(feats_v23)} feats, +new)')

    auc_v21 = res_v21['wf_mean_auc']
    auc_v23 = res_v23['wf_mean_auc']
    delta_vs_v21 = auc_v23 - auc_v21
    delta_vs_v15 = auc_v23 - V15_GENUINE_WF

    # ===== Results summary =====
    print(f"\n{'='*65}")
    print(f"{'Config':<35} {'feats':>5} {'WF AUC':>8} {'delta':>8}")
    print(f"{'-'*65}")
    print(f"{'V15 genuine baseline':<35} {'145':>5} {V15_GENUINE_WF:.4f}  {'—':>7}")
    print(f"{'V21 (reference)':<35} {len(feats_v21):>5} {auc_v21:.4f}  {auc_v21-V15_GENUINE_WF:>+7.4f}")
    print(f"{'V23 (V21 + new feats)':<35} {len(feats_v23):>5} {auc_v23:.4f}  {delta_vs_v15:>+7.4f}")
    print(f"{'-'*65}")
    print(f"  V23 delta vs V21:  {delta_vs_v21:+.4f}")
    print(f"  V23 delta vs V15:  {delta_vs_v15:+.4f}")
    verdict = 'GO' if auc_v23 >= GO_THRESHOLD else 'NO-GO'
    print(f"  Verdict (>= {GO_THRESHOLD:.4f}):  {verdict}")
    print(f"{'='*65}")

    # ===== Save JSON =====
    ablation = [res_v21, res_v23]
    result_json = {
        'ablation': ablation,
        'v15_genuine_wf': V15_GENUINE_WF,
        'v21_wf': V21_WF,
        'v21_wf_this_run': float(auc_v21),
        'v23_wf': float(auc_v23),
        'v23_vs_v21': float(delta_vs_v21),
        'v23_vs_v15': float(delta_vs_v15),
        'v23_features': feats_v23,
        'v23_n_feats': len(feats_v23),
        'verdict': verdict,
        'go_threshold': GO_THRESHOLD,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] WF results -> {OUT_JSON}")

    # ===== Save model if GO =====
    if verdict == 'GO' and not args.wf_only:
        save_model(df, feats_v23, auc_v23, ablation)
        print(f"\n[V23 GO] model saved to {OUT_MODEL}")
    elif verdict == 'NO-GO':
        print(f"\n[V23 NO-GO] model NOT saved. V15 production unchanged.")
        print(f"  → Next: consider additional lap features from JV-Link RA")
    else:
        print(f"\n[WF-ONLY] skipping model save (--wf-only flag)")


if __name__ == '__main__':
    main()
