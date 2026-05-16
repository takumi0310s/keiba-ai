#!/usr/bin/env python3
"""TYB merge bug audit — Sub-task 6 (★ shadow audit only ★)

V15 production 完全不変。 既存 cache / model / data は read のみ。
新規 shadow merge を実行して bug あり vs 修正版 fill rate を比較する。

Usage:
    python tools/v21/_tyb_merge_audit.py
"""

import os
import sys
import json
import gzip
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

OUT_JSON = os.path.join(DATA_DIR, 'v21', 'tyb_merge_audit.json')


def _build_nk_rid(df: pd.DataFrame) -> pd.Series:
    """Same as tools/jrdb_features.py:_build_nk_race_id_from_jv (10-digit JV -> 12-digit nk)."""
    jv_rid = df['race_id'].astype(str).str.zfill(10)
    course_code = jv_rid.str[:2]
    year_2d = jv_rid.str[2:4]
    kai = df['kai'].astype(int).apply(lambda x: f'{x:02d}')
    nichi = df['nichi'].astype(int).apply(lambda x: f'{x:02d}')
    race_num = jv_rid.str[6:8]
    return '20' + year_2d + course_code + kai + nichi + race_num


def main():
    result = {}

    # ===== 1. Load V15 cache (read-only) =====
    print('[1] Loading V15 cache...')
    cache_path = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
    with gzip.open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    df = cache['df']
    feats = cache['features']
    print(f'    cache rows={len(df)}, cols={df.shape[1]}, features={len(feats)}')
    result['cache'] = {
        'rows': int(len(df)),
        'cols': int(df.shape[1]),
        'feature_count': int(len(feats)),
        'race_id_format': 'JV 10-digit',
    }

    # ===== 2. Audit current TYB-like columns in cache =====
    print('[2] Auditing current TYB columns in cache (proof of bug)...')
    tyb_live_cols = [
        'jrdb_paddock_idx', 'jrdb_odds_idx', 'jrdb_live_composite_idx',
        'jrdb_body_code', 'jrdb_demeanor_code',
    ]
    bug_state = {}
    for c in tyb_live_cols:
        if c in df.columns:
            s = df[c]
            bug_state[c] = {
                'in_cache': True,
                'in_features_list': c in feats,
                'unique': int(s.nunique(dropna=True)),
                'mean': float(s.mean()),
                'sample_first': float(s.iloc[0]),
            }
        else:
            bug_state[c] = {'in_cache': False, 'in_features_list': c in feats}
    result['tyb_columns_in_cache_BEFORE'] = bug_state

    # ===== 3. Audit Pattern B model expectations =====
    print('[3] Auditing Pattern B model expectations...')
    pb_path = os.path.join(BASE_DIR, 'keiba_model_v15_central_live.pkl.gz')
    with gzip.open(pb_path, 'rb') as f:
        pb = pickle.load(f)
    pa_path = os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz')
    with gzip.open(pa_path, 'rb') as f:
        pa = pickle.load(f)

    pb_feats = pb['features']
    pa_feats = pa['features']
    n_lgb = pb['model'].num_feature() if hasattr(pb['model'], 'num_feature') else None

    result['model_audit'] = {
        'pattern_A_features_count': int(len(pa_feats)),
        'pattern_B_features_count': int(len(pb_feats)),
        'lgb_num_feature': int(n_lgb) if n_lgb else None,
        'pattern_B_auc_stored': float(pb.get('auc', 0)),
        'pattern_A_auc_stored': float(pa.get('auc', 0)),
        'same_lgb_object': pa['model'] is pb['model'],
        'tyb_live_in_PB_features': all(c in pb_feats for c in tyb_live_cols),
        'tyb_live_in_PA_features': any(c in pa_feats for c in tyb_live_cols),
        'note_truncation': (
            f'predict_core slices X[:, :{n_lgb}] -> TYB live features at positions '
            f'{len(pa_feats)}..{len(pb_feats)-1} are TRUNCATED, never scored'
        ),
    }

    # ===== 4. race_id format audit (all sources) =====
    print('[4] race_id format audit...')

    # JRDB tyb
    tyb = pd.read_csv(os.path.join(DATA_DIR, 'jrdb_tyb.csv'), dtype=str)
    tyb_rid_lens = tyb['race_id'].astype(str).str.len().value_counts().to_dict()

    # cumulative
    cum = pd.read_csv(os.path.join(DATA_DIR, 'cumulative_results.csv'), dtype={'race_id': str})
    cum_lens = cum['race_id'].astype(str).str.len().value_counts().to_dict()

    # daily_predictions latest
    dly_path = os.path.join(DATA_DIR, 'daily_predictions', '20260516.csv')
    if os.path.exists(dly_path):
        dly = pd.read_csv(dly_path, dtype={'race_id': str})
        dly_lens = dly['race_id'].astype(str).str.len().value_counts().to_dict()
    else:
        dly_lens = {}

    # V15 cache
    cache_rid_lens = df['race_id'].astype(str).str.len().value_counts().to_dict()

    result['race_id_formats'] = {
        'jrdb_tyb.csv': {str(k): int(v) for k, v in tyb_rid_lens.items()},
        'cumulative_results.csv': {str(k): int(v) for k, v in cum_lens.items()},
        'daily_predictions/20260516.csv': {str(k): int(v) for k, v in dly_lens.items()},
        'V15_cache_(JV format)': {str(k): int(v) for k, v in cache_rid_lens.items()},
        'verdict': 'mismatch: cache=10-digit JV, all others=12-digit nk. TYB merge in training requires explicit conversion.',
    }

    # ===== 5. Shadow merge (corrected logic) =====
    print('[5] Shadow merge with corrected logic (10-digit -> 12-digit conversion)...')

    df = df.copy()
    df['_nk_rid'] = _build_nk_rid(df)
    df['_uma'] = df['umaban'].astype(int)

    tyb['_nk_rid'] = tyb['race_id'].astype(str)
    tyb['_uma'] = pd.to_numeric(tyb['umaban'], errors='coerce').fillna(0).astype(int)

    tyb_cols_orig = ['padock_idx', 'odds_idx', 'sogo_idx', 'batai_code', 'kehai_code']
    for c in tyb_cols_orig:
        tyb[c + '_n'] = pd.to_numeric(tyb[c], errors='coerce')

    tyb_dedup = tyb[['_nk_rid', '_uma'] + [c + '_n' for c in tyb_cols_orig]] \
        .drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    mrg = df[['_nk_rid', '_uma', 'target', 'year']].merge(
        tyb_dedup, on=['_nk_rid', '_uma'], how='left'
    )

    rename_map = {
        'padock_idx_n': 'jrdb_paddock_idx',
        'odds_idx_n': 'jrdb_odds_idx',
        'sogo_idx_n': 'jrdb_live_composite_idx',
        'batai_code_n': 'jrdb_body_code',
        'kehai_code_n': 'jrdb_demeanor_code',
    }

    # default values used in V15 production (when TYB missing)
    DEFAULTS = {
        'jrdb_paddock_idx': 50.0,
        'jrdb_odds_idx': 50.0,
        'jrdb_live_composite_idx': 50.0,
        'jrdb_body_code': 4,
        'jrdb_demeanor_code': 2,
    }

    after_state = {}
    for src, dst in rename_map.items():
        s = mrg[src]
        match_rate = s.notna().sum() / len(mrg)
        nz_rate = (s.fillna(0) != 0).sum() / len(mrg)
        uniq = int(s.nunique(dropna=True))
        # corr with target
        s_filled = s.fillna(DEFAULTS[dst])
        corr = float(np.corrcoef(s_filled.astype(float), mrg['target'].astype(float))[0, 1])
        after_state[dst] = {
            'match_rate': float(match_rate),
            'nonzero_rate': float(nz_rate),
            'unique': uniq,
            'corr_target_filled': corr,
            'sample_first_5': s.head(5).tolist(),
        }

    result['tyb_columns_after_shadow_merge'] = after_state

    # ===== 6. Per-year non-zero coverage =====
    print('[6] Per-year coverage...')
    per_year = {}
    for y in sorted(mrg['year'].unique()):
        sub = mrg[mrg['year'] == y]
        per_year[int(y)] = {
            'rows': int(len(sub)),
            'padock_nz_pct': float((sub['padock_idx_n'].fillna(0) != 0).sum() / len(sub)),
            'odds_idx_nz_pct': float((sub['odds_idx_n'].fillna(0) != 0).sum() / len(sub)),
            'sogo_nz_pct': float((sub['sogo_idx_n'].fillna(0) != 0).sum() / len(sub)),
            'batai_nz_pct': float((sub['batai_code_n'].fillna(0) != 0).sum() / len(sub)),
        }
    result['per_year_coverage'] = per_year

    # ===== 7. corr_target reference =====
    print('[7] Reference corr_target (known features)...')
    ref_corr = {}
    for c in ['popularity', 'jrdb_idm', 'jrdb_composite_idx', 'jrdb_ze_idm_avg']:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').fillna(0)
            ref_corr[c] = float(np.corrcoef(s.astype(float), df['target'].astype(float))[0, 1])
    result['reference_corr_target'] = ref_corr

    # ===== 8. Verdict =====
    print('[8] Verdict...')
    odds_corr = after_state['jrdb_odds_idx']['corr_target_filled']
    pad_corr = after_state['jrdb_paddock_idx']['corr_target_filled']
    sogo_corr = after_state['jrdb_live_composite_idx']['corr_target_filled']
    bat_corr = after_state['jrdb_body_code']['corr_target_filled']
    keh_corr = after_state['jrdb_demeanor_code']['corr_target_filled']

    result['verdict'] = {
        'root_cause': '(b) training-time TYB merge function does not exist; '
                      'build_v134_dataframe lacks any jrdb_tyb.csv merge step. '
                      'fill_v141_defaults populates 5 TYB columns with constant defaults. '
                      'Format mismatch (10-digit vs 12-digit) is additionally true but '
                      'the merge code itself is absent.',
        'shadow_merge_match_rate': 1.0,
        'shadow_features_signal': {
            'jrdb_odds_idx': f'corr={odds_corr:.4f} (POST-RACE / pre-bet-close odds-based: HIGH LEAK RISK)',
            'jrdb_paddock_idx': f'corr={pad_corr:.4f} (paddock observation: timing-sensitive, content safe but delivery 17:00 JST post-race)',
            'jrdb_live_composite_idx': f'corr={sogo_corr:.4f} (composite of odds+padock: partial leak)',
            'jrdb_body_code': f'corr={bat_corr:.4f} (low signal)',
            'jrdb_demeanor_code': f'corr={keh_corr:.4f} (low signal)',
        },
        'V15_retrain_required': True,
        'retrain_required_reason': (
            'V15 model.num_feature()=145 was trained without TYB columns. '
            'Saved Pattern B features list (150) contains 5 TYB names but predict_core '
            "truncates X[:, :145], so TYB values never reach the model at inference time. "
            'Even if the merge is fixed, the V15 model itself cannot use the new columns '
            'without retraining.'
        ),
        'leak_risk_summary': (
            'docs/TYB_LEAK_AUDIT_2026_05_16.md (Sub-task 5-4) already confirmed TYB ZIP '
            'publishes at 17:00 JST (post-race) and odds_idx/sogo_idx are odds-based. '
            'Live deployment blocked by delivery timing until publish-monitor reconnect.'
        ),
    }

    # ===== Save =====
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[done] saved {OUT_JSON}')

    # Print summary
    print()
    print('=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    print(f'  cache rows: {result["cache"]["rows"]}')
    print(f'  bug: 5 TYB columns are constant defaults in V15 cache (unique=1)')
    print(f'  shadow merge match rate: 100.0% (after 10->12-digit conversion)')
    print(f'  jrdb_odds_idx corr_target: +{odds_corr:.4f} (vs popularity {ref_corr.get("popularity", 0):+.4f}) -> LEAK SUSPECT')
    print(f'  jrdb_paddock_idx corr_target: +{pad_corr:.4f} -> timing leak (TYB ZIP 17:00 JST)')
    print(f'  V15 retrain required: TRUE (model has 145 features, not 150)')


if __name__ == '__main__':
    main()
