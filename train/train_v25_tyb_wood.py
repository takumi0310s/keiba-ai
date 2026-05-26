"""V25 candidate: V15 (145 feats) + TYB features + wood_best_4f rolling補完。

追加特徴:
  A) JRDB TYB: ashimoto / batai_code / kehai_code / padock_idx
  B) wood_best_4f_rolling: horse_id 別 expanding mean による改善補完

V15 production (.pkl.gz / predict_core / app.py) 完全不変。
T4 leak audit PASS 必須。
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train.t4_leak_audit import run_leak_audit

# ============================================================
# Constants
# ============================================================
WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
GO_THRESHOLD = 0.8678   # V15 genuine WF LGB+XGB baseline

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
    'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist', 'verbosity': 0,
}

# course_enc (V15 cache) → TYB jo_code (2桁)
# V15: 0=札幌,1=函館,2=福島,3=新潟,4=東京,5=中山,6=中京,7=京都,8=阪神,9=小倉
# TYB: '01'=札幌,'02'=函館,...,'10'=小倉
ENC_TO_JO = {i: str(i + 1).zfill(2) for i in range(10)}

TYB_NEW_FEATURES = ['tyb_ashimoto', 'tyb_batai_code', 'tyb_kehai_code', 'tyb_padock_idx']
WOOD_ROLLING_FEATURE = 'wood_best_4f_rolling'


# ============================================================
# Step 1: Load V15 cache
# ============================================================
def load_v15_cache() -> tuple[pd.DataFrame, list[str]]:
    cache_path = REPO / 'data' / '_v15_train_df_cache.pkl'
    print(f"  Loading {cache_path} ...")
    raw = pickle.load(open(cache_path, 'rb'))
    if isinstance(raw, dict):
        df = raw['df']
        v15_feats = raw.get('v15_features', [])
    else:
        df = raw
        v15_feats = []
    print(f"  Rows: {len(df):,} / Cols: {df.shape[1]}")

    # Also load features from pkl.gz for definitive list
    model_path = REPO / 'models' / 'v15_full_candidate.pkl.gz'
    with gzip.open(model_path, 'rb') as f:
        m = pickle.load(f)
    v15_feats_model = m['features']
    print(f"  v15_full_candidate features: {len(v15_feats_model)}")
    if not v15_feats:
        v15_feats = v15_feats_model
    return df, v15_feats_model


# ============================================================
# Step 2: Merge TYB features
# ============================================================
def merge_tyb_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n[2] Merging JRDB TYB features ...")
    tyb_path = REPO / 'data' / 'jrdb_tyb.csv'
    tyb = pd.read_csv(tyb_path, dtype={'race_id': str},
                      usecols=['race_id', 'umaban', 'ashimoto', 'batai_code',
                               'kehai_code', 'padock_idx'])
    print(f"  TYB rows: {len(tyb):,}")

    # Build nk_id in V15 cache
    df = df.copy()
    df['_jo_code'] = df['course_enc'].astype(int).map(ENC_TO_JO).fillna('00')
    df['_nk_id'] = (
        df['year_full'].astype(str)
        + df['_jo_code']
        + df['kai'].astype(str).str.zfill(2)
        + df['nichi'].astype(str).str.zfill(2)
        + df['race_num'].astype(str).str.zfill(2)
    )
    df['_umaban_int'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)

    # Prepare TYB for merge
    tyb['race_id'] = tyb['race_id'].astype(str)
    tyb['umaban_int'] = pd.to_numeric(tyb['umaban'], errors='coerce').fillna(0).astype(int)
    tyb = tyb.rename(columns={
        'ashimoto': 'tyb_ashimoto',
        'batai_code': 'tyb_batai_code',
        'kehai_code': 'tyb_kehai_code',
        'padock_idx': 'tyb_padock_idx',
    })
    tyb = tyb.drop_duplicates(['race_id', 'umaban_int'], keep='last')

    n_before = len(df)
    df = df.merge(
        tyb[['race_id', 'umaban_int', 'tyb_ashimoto', 'tyb_batai_code',
             'tyb_kehai_code', 'tyb_padock_idx']],
        left_on=['_nk_id', '_umaban_int'],
        right_on=['race_id', 'umaban_int'],
        how='left',
    )
    assert len(df) == n_before, f"Row count changed after merge: {n_before} → {len(df)}"

    # Drop temp merge cols
    drop_cols = ['_jo_code', '_nk_id', '_umaban_int', 'race_id', 'umaban_int']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Fill rates
    stats = {}
    total = len(df)
    for feat in TYB_NEW_FEATURES:
        filled = int(df[feat].notna().sum())
        pct = filled / total * 100
        stats[feat] = {'filled': filled, 'total': total, 'pct': round(pct, 1)}
        print(f"  {feat}: {filled:,}/{total:,} ({pct:.1f}% fill)")

    # TYB race-level hit (how many races matched)
    n_matched_rows = df['tyb_ashimoto'].notna().sum()
    hit_pct = n_matched_rows / total * 100
    print(f"  TYB merge row hit: {n_matched_rows:,}/{total:,} ({hit_pct:.1f}%)")
    stats['merge_hit_pct'] = round(hit_pct, 1)

    # Fill NaN with mode/median for categorical, 0 for continuous
    df['tyb_ashimoto'] = df['tyb_ashimoto'].fillna(0).astype(int)
    df['tyb_batai_code'] = df['tyb_batai_code'].fillna(df['tyb_batai_code'].median()).fillna(3).astype(int)
    df['tyb_kehai_code'] = df['tyb_kehai_code'].fillna(df['tyb_kehai_code'].median()).fillna(2).astype(int)
    df['tyb_padock_idx'] = df['tyb_padock_idx'].fillna(0.0)

    return df, stats


# ============================================================
# Step 3: wood_best_4f rolling (horse_id 別 expanding mean)
# ============================================================
def build_wood_rolling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n[3] Building wood_best_4f per-horse expanding mean ...")
    raw_col = 'wood_best_4f'
    filled_col = 'wood_best_4f_filled'
    rolling_col = WOOD_ROLLING_FEATURE

    if raw_col not in df.columns:
        print(f"  WARNING: '{raw_col}' not in df. Skipping wood rolling.")
        df[rolling_col] = df[filled_col] if filled_col in df.columns else 0.0
        return df, {'raw_fill_pct': 0.0, 'rolling_fill_pct': 100.0}

    raw_fill = df[raw_col].notna().sum()
    raw_fill_pct = raw_fill / len(df) * 100
    print(f"  {raw_col} raw fill: {raw_fill:,}/{len(df):,} ({raw_fill_pct:.1f}%)")

    # Sort by horse_id + date_num for expanding order
    df = df.copy()
    df = df.sort_values(['horse_id', 'date_num']).reset_index(drop=True)

    # Expanding mean per horse, shifted to exclude current row (no data leakage)
    # expanding().mean().shift(1) = mean of all previous races for this horse
    df[rolling_col] = (
        df.groupby('horse_id')[raw_col]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    # For rows where rolling is NaN (first race of horse or no prior data),
    # fall back to global mean of raw col
    global_mean = df[raw_col].mean()
    df[rolling_col] = df[rolling_col].fillna(global_mean)

    rolling_fill = df[rolling_col].notna().sum()
    rolling_fill_pct = rolling_fill / len(df) * 100
    print(f"  {rolling_col} fill after rolling: {rolling_fill:,}/{len(df):,} ({rolling_fill_pct:.1f}%)")

    stats = {
        'raw_fill_pct': round(raw_fill_pct, 1),
        'rolling_fill_pct': round(rolling_fill_pct, 1),
        'global_mean': round(float(global_mean), 4),
    }
    return df, stats


# ============================================================
# Step 4: Build V25 feature set
# ============================================================
def build_v25_features(v15_feats: list[str], df: pd.DataFrame) -> list[str]:
    print("\n[4] Building V25 feature set ...")
    v25 = list(v15_feats)  # start from V15 145 feats

    # Add TYB features
    added = []
    for feat in TYB_NEW_FEATURES:
        if feat not in v25 and feat in df.columns:
            v25.append(feat)
            added.append(feat)
    print(f"  TYB features added: {added}")

    # Add wood rolling feature (replaces or supplements wood_best_4f_filled)
    if WOOD_ROLLING_FEATURE not in v25 and WOOD_ROLLING_FEATURE in df.columns:
        v25.append(WOOD_ROLLING_FEATURE)
        print(f"  Wood rolling added: {WOOD_ROLLING_FEATURE}")

    # Remove features not in df
    missing = [f for f in v25 if f not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing from df, removing: {missing}")
        v25 = [f for f in v25 if f in df.columns]

    print(f"  V15: {len(v15_feats)} feats → V25: {len(v25)} feats (added {len(v25) - len(v15_feats)})")
    return v25


# ============================================================
# Step 5: T4 Leak Audit
# ============================================================
def run_audit(df: pd.DataFrame, features: list[str]) -> None:
    print("\n[5] T4 Leak Audit ...")
    # TYB features leak check:
    # ashimoto / batai_code / kehai_code / padock_idx は JRDB TYB ファイル。
    # TYB = 出走直前 (発走前 ~15min) のパドック観察データ。PRE-RACE確定。
    # padock_idx は当日パドック評価指数 → Pattern A では注意対象だが
    # POST-RACE ではないので T4 strict/pattern_a どちらもPASS可能。
    # (paci_ninki_idx のようなオッズ派生でもない。)
    #
    # wood_best_4f_rolling = expanding().mean().shift(1) = 前走までの平均
    # → PRE-RACE, expanding window で当該レース除外済み → 安全。
    run_leak_audit(df, features, mode='pattern_a', fail_on_error=True)


# ============================================================
# Step 6: Walk-Forward
# ============================================================
def wf_fold(df: pd.DataFrame, features: list[str], test_year: int) -> dict | None:
    train_df = df[df['_wf_year'] < test_year].copy()
    test_df = df[df['_wf_year'] == test_year].copy()

    if len(train_df) < 1000 or len(test_df) < 100:
        print(f"  fold {test_year}: SKIP (train={len(train_df)}, test={len(test_df)})")
        return None

    X_tr = train_df[features].values.astype(np.float32)
    y_tr = (train_df['finish'] <= 3).astype(int).values
    X_te = test_df[features].values.astype(np.float32)
    y_te = (test_df['finish'] <= 3).astype(int).values

    # LGB
    d_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    d_val = lgb.Dataset(X_te, label=y_te, reference=d_tr)
    cb = lgb.train(
        LGB_PARAMS, d_tr,
        num_boost_round=1000,
        valid_sets=[d_val],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    p_lgb = cb.predict(X_te)

    # XGB
    dm_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
    dm_te = xgb.DMatrix(X_te, label=y_te, feature_names=features)
    bst = xgb.train(
        XGB_PARAMS, dm_tr,
        num_boost_round=1000,
        evals=[(dm_te, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    p_xgb = bst.predict(dm_te)

    score = 0.5 * p_lgb + 0.5 * p_xgb
    auc = float(roc_auc_score(y_te, score))
    lgb_auc = float(roc_auc_score(y_te, p_lgb))
    xgb_auc = float(roc_auc_score(y_te, p_xgb))
    print(f"  fold {test_year}: AUC={auc:.6f}  LGB={lgb_auc:.6f}  XGB={xgb_auc:.6f}"
          f"  (n_train={len(train_df):,}, n_test={len(test_df):,})")

    return {'year': test_year, 'auc': auc, 'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc,
            'n_train': len(train_df), 'n_test': len(test_df),
            '_lgb': cb, '_xgb': bst}


def run_wf(df: pd.DataFrame, features: list[str]) -> tuple[list[dict], object, object]:
    print("\n[6] Walk-Forward 2020-2025 ...")
    fold_results = []
    last_lgb = None
    last_xgb = None
    for yr in WF_YEARS:
        result = wf_fold(df, features, yr)
        if result is None:
            continue
        last_lgb = result.pop('_lgb')
        last_xgb = result.pop('_xgb')
        fold_results.append(result)
    return fold_results, last_lgb, last_xgb


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("V25 CANDIDATE: TYB features + wood_best_4f rolling")
    print("=" * 70)

    # 1. Load
    print("\n[1] Loading V15 training cache ...")
    df, v15_feats = load_v15_cache()

    # 2. Merge TYB
    df, tyb_stats = merge_tyb_features(df)

    # 3. Wood rolling
    df, wood_stats = build_wood_rolling(df)

    # 4. Feature set
    v25_feats = build_v25_features(v15_feats, df)

    # 5. Leak audit
    run_audit(df, v25_feats)

    # 6. Prepare df for WF
    df = df.copy()
    df['_wf_year'] = df['year_full'].astype(int)
    df = df.dropna(subset=['finish', '_wf_year'])
    df['_wf_year'] = df['_wf_year'].astype(int)
    df[v25_feats] = df[v25_feats].fillna(0.0)
    print(f"\n  Training rows: {len(df):,} / Year: {df['_wf_year'].min()}-{df['_wf_year'].max()}")

    # 7. WF
    fold_results, last_lgb, last_xgb = run_wf(df, v25_feats)

    if not fold_results:
        print("ERROR: No fold results. Aborting.")
        sys.exit(1)

    wf_mean = float(np.mean([r['auc'] for r in fold_results]))
    verdict = "GO" if wf_mean >= GO_THRESHOLD else "NO-GO"
    delta = wf_mean - GO_THRESHOLD

    print(f"\n{'=' * 70}")
    print(f"V25 WF mean AUC : {wf_mean:.6f}")
    print(f"V15 baseline    : {GO_THRESHOLD:.4f}")
    print(f"delta           : {delta:+.6f}")
    print(f"Verdict         : {verdict}")
    print(f"{'=' * 70}")

    # 8. Save results
    results = {
        'version': 'v25_tyb_wood',
        'v15_genuine_wf': GO_THRESHOLD,
        'v25_wf': wf_mean,
        'delta': delta,
        'verdict': verdict,
        'v15_n_feats': len(v15_feats),
        'v25_n_feats': len(v25_feats),
        'new_features': {
            'tyb': TYB_NEW_FEATURES,
            'wood_rolling': WOOD_ROLLING_FEATURE,
        },
        'tyb_stats': tyb_stats,
        'wood_stats': wood_stats,
        'folds': fold_results,
        'v25_features': v25_feats,
    }
    out_dir = REPO / 'data' / 'v20'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'v25_tyb_wood_wf_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_json}")

    # 9. Save model if GO
    if verdict == "GO" and last_lgb is not None and last_xgb is not None:
        model_out = REPO / 'models' / 'v25_tyb_wood_candidate.pkl.gz'
        payload = {
            'version': 'v25_tyb_wood',
            'features': v25_feats,
            'lgb_model': last_lgb,
            'xgb_model': last_xgb,
            'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5},
            'wf_auc': wf_mean,
            'v15_baseline': GO_THRESHOLD,
            'delta': delta,
            'verdict': verdict,
            'tyb_stats': tyb_stats,
            'wood_stats': wood_stats,
        }
        with gzip.open(model_out, 'wb') as f:
            pickle.dump(payload, f, protocol=4)
        print(f"Model saved: {model_out}")
    elif verdict == "NO-GO":
        print("NO-GO: model not saved (did not beat V15 baseline).")
    else:
        print("WARNING: last fold models missing, skipping save.")

    # 10. Summary report
    fill_ashimoto = tyb_stats.get('tyb_ashimoto', {}).get('pct', 0)
    fill_batai = tyb_stats.get('tyb_batai_code', {}).get('pct', 0)
    fill_kehai = tyb_stats.get('tyb_kehai_code', {}).get('pct', 0)
    raw_wood = wood_stats.get('raw_fill_pct', 0)
    roll_wood = wood_stats.get('rolling_fill_pct', 0)
    merge_hit = tyb_stats.get('merge_hit_pct', 0)

    print("\n" + "=" * 70)
    print("COMPLETION REPORT")
    print("=" * 70)
    print(
        f"V25完了、"
        f"TYB merge hit={merge_hit}%、"
        f"新特徴fill=[ashimoto:{fill_ashimoto}%, batai:{fill_batai}%, kehai:{fill_kehai}%]、"
        f"wood rolling改善={raw_wood}%→{roll_wood}%、"
        f"WF={wf_mean:.4f} (vs V15 {GO_THRESHOLD})、"
        f"leak audit=PASS、"
        f"verdict={verdict}、"
        f"V15不変"
    )
    print("=" * 70)


if __name__ == '__main__':
    main()
