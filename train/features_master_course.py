#!/usr/bin/env python
"""Master Course 個別ラップ特徴量 (scaffold)

netkeiba マスターコース（有料会員限定）の個別ラップデータから特徴量を生成。
Source CSV: `data/netkeiba_individual_lap.csv`
  （`tools/scrape_master_course.py --source laps` で取得）

特徴量:
  - individual_lap_first_3f   : 前走の個別前半3Fタイム
  - individual_lap_last_3f    : 前走の個別後半3Fタイム (netkeiba結果画面の上がり3Fとは別ソース)
  - individual_lap_mid_pace   : 前走の中間ラップ平均
  - individual_lap_std        : 前走ラップ標準偏差（ペース変化の大きさ）
  - individual_lap_max_f      : 前走の最遅ラップ値（失速度）

全て lag-1（前走値）なのでリークフリー。AM8:00で取得可能。

注意: 現在 netkeiba_individual_lap.csv は**未構築**。
`python tools/scrape_master_course.py --source laps --year 2024-2025` で取得が必要。
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')

FEATURE_NAMES = [
    'individual_lap_first_3f',
    'individual_lap_last_3f',
    'individual_lap_mid_pace',
    'individual_lap_std',
    'individual_lap_max_f',
]

FEATURE_DEFAULTS = {
    'individual_lap_first_3f': 36.0,
    'individual_lap_last_3f': 36.0,
    'individual_lap_mid_pace': 12.0,
    'individual_lap_std': 0.5,
    'individual_lap_max_f': 13.0,
}


def get_features():
    return list(FEATURE_NAMES)


def _compute_row_features(laps: list[float]) -> dict:
    """単一レース分のlapリストから特徴量を計算"""
    if not laps or len(laps) < 3:
        return {f: FEATURE_DEFAULTS[f] for f in FEATURE_NAMES}
    arr = np.array(laps, dtype=float)
    first_3f = float(arr[:3].sum())
    last_3f = float(arr[-3:].sum())
    mid = arr[3:-3] if len(arr) > 6 else arr[1:-1]
    mid_pace = float(mid.mean()) if len(mid) > 0 else FEATURE_DEFAULTS['individual_lap_mid_pace']
    std = float(arr.std())
    max_f = float(arr.max())
    return {
        'individual_lap_first_3f': first_3f,
        'individual_lap_last_3f': last_3f,
        'individual_lap_mid_pace': mid_pace,
        'individual_lap_std': std,
        'individual_lap_max_f': max_f,
    }


def merge_individual_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    """df に individual_lap_* 特徴量を追加 (lag-1)。

    Args:
        df: 学習/予測データ。`horse_id`, `date_num`, `umaban` カラム必須。
            nk_race_id (12桁) があれば直接マージ、なければ jv race_id から構築。
    """
    csv_path = os.path.join(DATA_DIR, 'netkeiba_individual_lap.csv')
    if not os.path.exists(csv_path):
        print(f"    [WARN] {csv_path} 未構築 -> デフォルト値で埋める")
        for f in FEATURE_NAMES:
            df[f] = FEATURE_DEFAULTS[f]
        return df

    lap_df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str)
    lap_cols = [c for c in lap_df.columns if c.startswith('lap_')]
    # 各行のlap配列 → 特徴量dict
    lap_records = []
    for _, r in lap_df.iterrows():
        laps = []
        for c in lap_cols:
            v = r.get(c)
            if v is None or v == '' or pd.isna(v):
                continue
            try:
                laps.append(float(v))
            except (ValueError, TypeError):
                continue
        feats = _compute_row_features(laps)
        feats['_nk_rid'] = str(r.get('race_id', '')).zfill(12)
        feats['_uma'] = pd.to_numeric(r.get('umaban'), errors='coerce')
        lap_records.append(feats)

    if not lap_records:
        for f in FEATURE_NAMES:
            df[f] = FEATURE_DEFAULTS[f]
        return df

    lap_merge = pd.DataFrame(lap_records)
    lap_merge = lap_merge.dropna(subset=['_uma']).drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    # df 側のキー準備
    if '_nk_rid' not in df.columns:
        from jrdb_features import _build_nk_race_id_from_jv
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    # 現走の値を一旦マージしてから lag-1 化
    current_cols = {f: f'{f}_cur' for f in FEATURE_NAMES}
    lap_merge_renamed = lap_merge.rename(columns=current_cols)
    df = df.merge(
        lap_merge_renamed[['_nk_rid', '_uma'] + list(current_cols.values())],
        on=['_nk_rid', '_uma'], how='left',
    )

    # horse_id + date_num で shift(1) して lag-1 値を生成
    if 'horse_id' in df.columns and 'date_num' in df.columns:
        df = df.sort_values(['horse_id', 'date_num']).reset_index()
        grp = df.groupby('horse_id')
        for f in FEATURE_NAMES:
            df[f] = grp[current_cols[f]].shift(1).fillna(FEATURE_DEFAULTS[f])
        df = df.sort_values('index').drop(columns='index').reset_index(drop=True)
    else:
        for f in FEATURE_NAMES:
            df[f] = FEATURE_DEFAULTS[f]

    # クリーンアップ
    df.drop(columns=list(current_cols.values()), inplace=True, errors='ignore')
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    matched = (df['individual_lap_first_3f'] != FEATURE_DEFAULTS['individual_lap_first_3f']).sum()
    print(f"    individual_lap features: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    return df
