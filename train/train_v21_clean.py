"""V21 clean candidate: P0 fix + 12 dead feature removal + horse_course_top3r.

Ablation WF runs:
  A: V20 base (148 feats, incl. paci_ninki_idx) — reference / contamination check
  B: V20 - paci_ninki_idx (147 feats)           — P0 true AUC
  C: V21 = B - 12 dead + horse_course_top3r_exp (136 feats) — V21 candidate

V15 production: COMPLETELY UNCHANGED.
Output: models/v21_candidate.pkl.gz (overwrites old TYB-based v21)

Usage:
  python train/train_v21_clean.py
"""
from __future__ import annotations
import sys, os, time, pickle, gzip, json, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

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
MODEL_OUT   = os.path.join(BASE, 'models', 'v21_candidate.pkl.gz')
WF_JSON_OUT = os.path.join(BASE, 'data', 'v20', 'v21_clean_wf_results.json')

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
    'seed': 42, 'tree_method': 'hist',
}

V15_GENUINE_WF = 0.8678  # V15-audit-2 genuine WF LGB+XGB 6-fold
V20_BASE_WF    = 0.8664  # V20 base (includes paci_ninki_idx — potentially contaminated)
SAVE_THRESHOLD = 0.8660  # save if V21 >= this

# ===== 12 dead features to remove =====
V21_DELETE_12 = {
    'odds_sharp_drop',        # lgb=14, xgb=0
    'jrdb_prev_rise_code',    # lgb=35, 97.6% quasi-constant
    'bracket_pos',            # lgb=41
    'is_long_transport',      # lgb=65, transport_distance_km redundant
    'course_renovated',       # lgb=70, 98.7% quasi-constant
    'jrdb_prev_interference', # lgb=160, 98.2% quasi-constant
    'top3_count_3r',          # lgb=158, redundant with avg_finish_3r
    'rest_category',          # lgb=114, redundant with rest_days
    'has_sakaro_training',    # lgb=116, redundant with sakaro_best_4f_filled
    'stable_comment_score',   # lgb=201, 5.5% real data
    'age_group',              # lgb=0 (xgb=194), redundant with age
    'wood_best_4f_filled',    # lgb=1248, 18.1% real, collinear with training_time_filled
}

# paci_ninki_idx: odds-derived → Pattern A leak (exclude from WF eval)
PACI_NINKI = 'paci_ninki_idx'


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load V15 cache + all parquets. Returns (df, feats_v20_base)."""
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

    # ===== Merge horse_course features =====
    if os.path.exists(HC_PARQ):
        hc = pd.read_parquet(HC_PARQ)
        for col in ['horse_course_top3r_exp', 'horse_course_n_exp']:
            if col in hc.columns:
                df[col] = hc[col].reindex(df.index)
        fill = df.get('horse_course_top3r_exp', pd.Series()).notna().mean()
        print(f"  horse_course fill: {fill:.1%}")
    else:
        print("  [WARN] horse_course parquet not found — run build_horse_course_features.py first")

    # ===== V20 base feature list (148) =====
    feats_v20 = get_v20_features(
        include_o1_fixed=True, include_blod_fixed=True,
        include_sib=True, include_blod_full=False
    )
    feats_v20 = [f for f in feats_v20 if f in df.columns and f not in V20_LEAK_FEATURES]
    print(f"\nV20 base features available: {len(feats_v20)}")
    if PACI_NINKI in feats_v20:
        print(f"  paci_ninki_idx: IN V20 feature list (Pattern A contamination confirmed)")
    else:
        print(f"  paci_ninki_idx: NOT in V20 list")

    return df, feats_v20


def run_wf(df: pd.DataFrame, feats: list[str], label: str) -> dict:
    """Walk-forward 6-fold (2020-2025). Returns results dict."""
    print(f"\n--- WF: {label} ({len(feats)} feats) ---")
    results = []
    for fold_yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        train_mask = (df['_y'] >= 2015) & (df['_y'] < fold_yr)
        test_mask  = df['_y'] == fold_yr
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue
        y_tr = (df.loc[train_mask, 'finish'] <= 3).astype(int)
        y_te = (df.loc[test_mask,  'finish'] <= 3).astype(int)
        X_tr = df.loc[train_mask, feats].fillna(0).astype(np.float32)
        X_te = df.loc[test_mask,  feats].fillna(0).astype(np.float32)

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
        print(f"  Fold {fold_yr}: AUC={auc:.4f}  n_train={train_mask.sum():,}  [{time.time()-t0:.0f}s]")
        results.append({'year': fold_yr, 'auc': float(auc)})

    mean_auc = float(np.mean([r['auc'] for r in results]))
    print(f"  => WF mean AUC = {mean_auc:.4f}")
    return {'label': label, 'n_feats': len(feats), 'folds': results, 'wf_mean_auc': mean_auc}


def save_model(df: pd.DataFrame, feats: list[str], wf_auc: float,
               ablation_results: list[dict]) -> None:
    """Train on full 2015-2025 and save production candidate."""
    print(f"\n=== Training final V21 model (2015-2025) ===")
    mask = (df['_y'] >= 2015) & (df['_y'] <= 2025)
    y = (df.loc[mask, 'finish'] <= 3).astype(int)
    X = df.loc[mask, feats].fillna(0).astype(np.float32)
    print(f"  n={mask.sum():,}, features={len(feats)}")

    t0 = time.time()
    lgb_m = lgb.train(LGB_PARAMS, lgb.Dataset(X, y),
                      num_boost_round=500, callbacks=[lgb.log_evaluation(0)])
    dtrain = xgb.DMatrix(X, label=y)
    xgb_m = xgb.train(XGB_PARAMS, dtrain, num_boost_round=500, verbose_eval=False)
    print(f"  training done [{time.time()-t0:.0f}s]")

    out = {
        'version': 'v21_clean',
        'features': feats,
        'n_features': len(feats),
        'lgb_model': lgb_m,
        'xgb_model': xgb_m,
        'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5},
        'wf_mean_auc': wf_auc,
        'v15_baseline_genuine_wf': V15_GENUINE_WF,
        'v20_base_wf': V20_BASE_WF,
        'ablation': ablation_results,
        'changes_from_v20': {
            'removed_paci_ninki_idx': True,
            'removed_12_dead': sorted(V21_DELETE_12),
            'added': ['horse_course_top3r_exp'],
        },
    }
    with gzip.open(MODEL_OUT, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(MODEL_OUT) / 1e6
    print(f"  [SAVED] {MODEL_OUT} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wf-only', action='store_true')
    args = parser.parse_args()

    df, feats_v20 = load_data()

    # ===== Define 3 feature sets =====
    # A: V20 base (148, includes paci_ninki_idx)
    feats_A = feats_v20[:]

    # B: V20 - paci_ninki_idx (P0 true AUC)
    feats_B = [f for f in feats_v20 if f != PACI_NINKI]

    # C: V21 = B - 12 dead + horse_course_top3r_exp
    feats_C = [f for f in feats_B if f not in V21_DELETE_12]
    if 'horse_course_top3r_exp' in df.columns:
        feats_C = feats_C + ['horse_course_top3r_exp']
    else:
        print("[WARN] horse_course_top3r_exp not in df — skipping (run build_horse_course_features.py)")

    print(f"\nFeature counts:")
    print(f"  A (V20 base):          {len(feats_A)}")
    print(f"  B (V20 -ninki):        {len(feats_B)}")
    print(f"  C (V21 full):          {len(feats_C)}")
    print(f"  Removed (12 dead):     {sorted(V21_DELETE_12)}")
    print(f"  Added:                 horse_course_top3r_exp")

    # ===== P0-2: training_time_filled analysis =====
    print("\n=== P0-2: training_time_filled analysis ===")
    if 'has_wood_training' in df.columns and 'training_time_filled' in df.columns:
        corr = df['training_time_filled'].corr(df['has_wood_training'])
        print(f"  corr(training_time_filled, has_wood_training) = {corr:.4f}")
        mode_val = df['training_time_filled'].mode()[0]
        mode_pct = (df['training_time_filled'] == mode_val).mean()
        print(f"  mode value = {mode_val:.4f}, mode rate = {mode_pct:.1%}")
        if mode_pct > 0.5:
            print(f"  => {mode_pct:.1%} rows have identical value (imputed mean)")
            print(f"  => training_time_filled acts as quasi-binary via imputation")
        else:
            print(f"  => mode rate low ({mode_pct:.1%}) — individual imputation, not global mean")

    # ===== Run 3 WF evaluations =====
    res_A = run_wf(df, feats_A, 'A: V20 base (ref, incl paci_ninki_idx)')
    res_B = run_wf(df, feats_B, 'B: V20 -paci_ninki_idx (P0 true)')
    res_C = run_wf(df, feats_C, 'C: V21 = B -12dead +horse_course')

    auc_A = res_A['wf_mean_auc']
    auc_B = res_B['wf_mean_auc']
    auc_C = res_C['wf_mean_auc']

    # ===== Results table =====
    print(f"\n{'='*65}")
    print(f"{'Config':<40} {'feats':>5} {'WF AUC':>8} {'delta_vs_A':>11}")
    print(f"{'-'*65}")
    print(f"{'A: V20 base (ref, incl ninki)':<40} {len(feats_A):>5} {auc_A:.4f}  {'—':>10}")
    print(f"{'B: V20 -paci_ninki_idx (P0)':<40} {len(feats_B):>5} {auc_B:.4f}  {auc_B-auc_A:>+10.4f}")
    print(f"{'C: V21 -12dead +horse_course':<40} {len(feats_C):>5} {auc_C:.4f}  {auc_C-auc_A:>+10.4f}")
    print(f"{'-'*65}")
    print(f"  V15 genuine WF baseline:     {V15_GENUINE_WF:.4f}")
    print(f"  V20 reported WF (w/ ninki):  {V20_BASE_WF:.4f}")
    print(f"  V21 WF vs V15 0.8678:        {auc_C - V15_GENUINE_WF:+.4f}")
    verdict = 'GO' if auc_C >= V15_GENUINE_WF else 'NO-GO'
    print(f"  Verdict (V21 >= 0.8678):     {verdict}")
    print(f"{'='*65}")

    # Save WF results JSON
    ablation = [res_A, res_B, res_C]
    with open(WF_JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump({
            'ablation': ablation,
            'paci_ninki_contamination': auc_A - auc_B,
            'dead_features_delta': auc_C - auc_B + (auc_B - auc_A),
            'v15_genuine_wf': V15_GENUINE_WF,
            'v20_reported_wf': V20_BASE_WF,
            'v21_wf': auc_C,
            'v21_vs_v15': auc_C - V15_GENUINE_WF,
            'verdict': verdict,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {WF_JSON_OUT}")

    # ===== Save model =====
    if not args.wf_only:
        if auc_C >= SAVE_THRESHOLD:
            save_model(df, feats_C, auc_C, ablation)
        else:
            print(f"\n✗ V21 WF {auc_C:.4f} < threshold {SAVE_THRESHOLD}. Model NOT saved.")
    else:
        print("\n--wf-only: model save skipped.")

    # ===== Final summary =====
    print(f"\n=== SUMMARY ===")
    print(f"  paci_ninki_idx汚染:   {auc_A:.4f} → {auc_B:.4f} ({auc_B-auc_A:+.4f})")
    print(f"  削除12件 delta:       {auc_B:.4f} → ... intermediate")
    print(f"  horse_course追加:     V21 = {auc_C:.4f}")
    print(f"  vs V15 0.8678:        {auc_C - V15_GENUINE_WF:+.4f}  [{verdict}]")
    print(f"  V15 production:       UNCHANGED")


if __name__ == '__main__':
    main()
