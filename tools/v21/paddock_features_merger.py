"""Phase A — Paddock 12 features merger.

Reads `data/video_ai_features/` (90 entries, 33 races, 85 horses) and
derives 12 paddock features per (race_id, horse_id), writing
`data/v21/paddock_12_features.csv`.

Schema: see `data/v21/phase_a_paddock_12_features_spec.md`.

V15 production unchanged: this script only reads data/video_ai_features
(read-only) and writes data/v21/paddock_12_features.csv (new file).
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / 'data' / 'video_ai_features'
OUT_DIR = BASE_DIR / 'data' / 'v21'
OUT_CSV = OUT_DIR / 'paddock_12_features.csv'

# 12 features and their derivation
# Each derivation: (feature_name, source_kind, json_path_dotted)
# source_kind: 'body_agg' = body_condition.aggregated.<key>
#              'gait_feat' = gait.features.<key>
PADDOCK_FEATURES = [
    ('pad_body_condition_score', 'body_agg', 'condition_score_mean'),
    ('pad_body_condition_std',   'body_agg', 'condition_score_std'),
    ('pad_coat_gloss',           'body_agg', 'coat_brightness_mean'),
    ('pad_coat_contrast',        'body_agg', 'coat_contrast_mean'),
    ('pad_body_aspect_mean',     'body_agg', 'body_aspect_mean'),
    ('pad_body_aspect_var',      'body_agg', 'body_aspect_std'),
    ('pad_body_compactness',     'body_agg', 'body_compactness_mean'),
    ('pad_gait_motion_speed',    'gait_feat', 'motion_speed_mean'),
    ('pad_gait_motion_std',      'gait_feat', 'motion_speed_std'),
    ('pad_gait_aspect_range',    'gait_feat', 'aspect_range'),
    ('pad_gait_area_change',     'gait_feat', 'area_change_mean'),
    ('pad_yolo_conf_mean',       'gait_feat', 'conf_mean'),
]


def _load_json(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def extract_for_entry(entry_dir: Path):
    """Return dict with race_id, horse_id, and 12 paddock features.

    Missing source files / keys → NaN for that feature.
    """
    parts = entry_dir.name.split('_')
    if len(parts) < 2:
        return None
    race_id = parts[0]
    horse_id = parts[1]

    body = _load_json(entry_dir / 'body_condition_features.json')
    gait = _load_json(entry_dir / 'gait_features.json')

    body_agg = (body or {}).get('aggregated', {}) if body else {}
    gait_feat = (gait or {}).get('features', {}) if gait else {}

    row = {'race_id': race_id, 'horse_id': horse_id}
    for feat_name, kind, key in PADDOCK_FEATURES:
        if kind == 'body_agg':
            val = body_agg.get(key)
        elif kind == 'gait_feat':
            val = gait_feat.get(key)
        else:
            val = None
        row[feat_name] = float(val) if val is not None else float('nan')
    return row


def build_paddock_csv():
    if not SRC_DIR.exists():
        print(f"[ERROR] source dir not found: {SRC_DIR}")
        return None

    entries = sorted(p for p in SRC_DIR.iterdir() if p.is_dir())
    rows = []
    skip_zero = 0
    for ent in entries:
        # skip "_0" placeholder entries (sanity-check / no horse_id)
        if ent.name.endswith('_0'):
            skip_zero += 1
            continue
        rec = extract_for_entry(ent)
        if rec is None:
            continue
        rows.append(rec)

    print(f"  scanned entries: {len(entries)}")
    print(f"  skipped placeholder (_0): {skip_zero}")
    print(f"  extracted rows: {len(rows)}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  wrote: {OUT_CSV} ({len(df)} rows × {len(df.columns)} cols)")

    # Coverage report per feature
    feat_cols = [f for f, _, _ in PADDOCK_FEATURES]
    print()
    print("  per-feature coverage:")
    for f in feat_cols:
        n_present = df[f].notna().sum()
        print(f"    {f:32s}: {n_present}/{len(df)} ({n_present/max(len(df),1)*100:.1f}%)")

    print()
    print(f"  distinct races: {df['race_id'].nunique()}")
    print(f"  distinct horses: {df['horse_id'].nunique()}")
    return df


def main():
    print("=" * 70)
    print("  Phase A - Paddock 12 features merger")
    print("=" * 70)
    df = build_paddock_csv()
    if df is None:
        return 1
    print()
    print("[OK] paddock_12_features.csv written")
    return 0


if __name__ == '__main__':
    sys.exit(main())
