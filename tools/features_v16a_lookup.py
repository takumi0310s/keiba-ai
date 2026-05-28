"""V16a 推論用 lookup 特徴量付加モジュール。

学習時に生成した data/v16a_lookup_tables.pkl を読み込み、
推論時の features_df に V16a 新特徴量を付加する。

Usage:
    from tools.features_v16a_lookup import augment_features_v16a
    features_df = augment_features_v16a(features_df, race_info)
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
LOOKUP_PATH = DATA_DIR / 'v16a_lookup_tables.pkl'

V16A_DEFAULTS = {
    'jockey_trainer_top3_rate_exp': 0.33,
    'jockey_trainer_n_exp': 0.0,
    'horse_weight_slope_5': 0.0,
    'horse_weight_std_5': 5.0,
    'horse_perf_score_exp': 0.33,
    'gate_bias_top3_exp': 0.33,
    'gate_bias_n_exp': 0.0,
}


@lru_cache(maxsize=1)
def _load_tables() -> dict:
    if not LOOKUP_PATH.exists():
        print(f"[v16a_lookup] Lookup not found: {LOOKUP_PATH}")
        return {}
    import pickle
    with open(LOOKUP_PATH, 'rb') as f:
        tables = pickle.load(f)
    built = tables.get('built_at', 'unknown')
    jt_n  = len(tables.get('jt', {}))
    pf_n  = len(tables.get('perf', {}))
    gt_n  = len(tables.get('gate', {}))
    wt_n  = len(tables.get('weight', {}))
    print(f"[v16a_lookup] Loaded (built: {built}): "
          f"jt={jt_n} perf={pf_n} weight={wt_n} gate={gt_n}")
    return tables


def _dist_bin(distance: int) -> int:
    return (distance // 400) * 400


def augment_features_v16a(features_df, race_info: dict):
    """features_df に V16a 特徴量カラムを追加して返す。

    Args:
        features_df: predict_core が生成した特徴量 DataFrame
        race_info:   race_info dict (course, distance, surface, condition 等)

    Returns:
        features_df に 7カラム追加したもの (inplace 変更)
    """
    import pandas as pd

    tables = _load_tables()
    if not tables:
        for feat, default in V16A_DEFAULTS.items():
            features_df[feat] = default
        return features_df

    jt_tbl    = tables.get('jt', {})
    perf_tbl  = tables.get('perf', {})
    wt_tbl    = tables.get('weight', {})
    gate_tbl  = tables.get('gate', {})

    course   = str(race_info.get('course', ''))
    distance = int(race_info.get('distance', 0))
    surface  = str(race_info.get('surface', ''))
    dist_bin = _dist_bin(distance)

    rows = []
    for idx, row in features_df.iterrows():
        horse_id  = str(row.get('horse_id', ''))
        jockey_id = str(row.get('jockey_id', ''))
        trainer_id = str(row.get('trainer_id', ''))
        umaban    = int(row.get('umaban', row.get('馬番', 0)))

        # 1. jockey × trainer combo
        jt_key = (jockey_id, trainer_id)
        jt = jt_tbl.get(jt_key, {})
        jt_rate = jt.get('top3_rate', V16A_DEFAULTS['jockey_trainer_top3_rate_exp'])
        jt_n    = jt.get('n', 0)

        # 2. 馬体重 slope / std
        wt = wt_tbl.get(horse_id, {})
        wt_slope = wt.get('slope', V16A_DEFAULTS['horse_weight_slope_5'])
        wt_std   = wt.get('std',   V16A_DEFAULTS['horse_weight_std_5'])

        # 3. 正規化パフォーマンス
        perf = perf_tbl.get(horse_id, V16A_DEFAULTS['horse_perf_score_exp'])

        # 4. 枠番バイアス
        gate_key = (course, dist_bin, surface, umaban)
        gt = gate_tbl.get(gate_key, {})
        gt_rate = gt.get('top3_rate', V16A_DEFAULTS['gate_bias_top3_exp'])
        gt_n    = gt.get('n', 0)

        rows.append({
            'jockey_trainer_top3_rate_exp': jt_rate,
            'jockey_trainer_n_exp':         float(jt_n),
            'horse_weight_slope_5':         wt_slope,
            'horse_weight_std_5':           wt_std,
            'horse_perf_score_exp':         perf,
            'gate_bias_top3_exp':           gt_rate,
            'gate_bias_n_exp':              float(gt_n),
        })

    aug = pd.DataFrame(rows, index=features_df.index)
    for col in aug.columns:
        features_df[col] = aug[col]

    return features_df


def invalidate_cache():
    """lookup table のキャッシュを破棄 (lookup 更新後に呼ぶ)。"""
    _load_tables.cache_clear()
