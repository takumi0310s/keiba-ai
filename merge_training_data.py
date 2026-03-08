#!/usr/bin/env python
"""TARGET調教データ（坂路・ウッドチップ）をkeiba_data(target_odds.csv)と結合するスクリプト

想定する調教CSVファイル:
  data/training_slope.csv   - 坂路調教データ（TARGETエクスポート）
  data/training_wood.csv    - ウッドチップ調教データ（TARGETエクスポート）

出力:
  data/target_odds_with_training.csv - 結合済みデータ

TARGETの調教データCSV想定カラム:
  race_horse_key, 調教日, 場所(坂路/ウッド等), コース, 馬場, 強さ,
  4F, 3F, 1F, 調教評価(A-E), 騎乗者, 併せ結果, 備考

Usage:
  python merge_training_data.py
  python merge_training_data.py --slope data/training_slope.csv --wood data/training_wood.csv
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
BASE_CSV = os.path.join(DATA_DIR, 'target_odds.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'target_odds_with_training.csv')

# target_odds.csv column mapping (52 cols, no header, cp932)
COL = {
    'year': 0, 'month': 1, 'day': 2, 'race_num': 3,
    'course': 4, 'kai': 5, 'nichi': 6, 'class': 7,
    'waku_or_field': 8, 'surface': 9, 'obstacle': 10, 'distance': 11,
    'condition': 12, 'horse_name': 13, 'sex': 14, 'age': 15,
    'jockey': 16, 'weight_carry': 17, 'num_horses': 18, 'umaban': 19,
    'finish': 20, 'finish2': 21, 'margin_flag': 22, 'time_margin': 23,
    'pop': 24, 'pace': 25, 'time_x10': 26, 'col27': 27,
    'pass1': 28, 'pass2': 29, 'pass3': 30, 'pass4': 31,
    'agari': 32, 'horse_weight': 33, 'trainer': 34, 'location': 35,
    'prize': 36, 'horse_id': 37, 'jockey_id': 38, 'trainer_id': 39,
    'race_horse_key': 40, 'father': 41, 'mother': 42, 'bms': 43,
    'sire_sire': 44, 'col45': 45, 'origin': 46, 'birthday': 47,
    'odds': 48, 'empty1': 49, 'empty2': 50, 'training_time': 51,
}

# TARGET調教CSV想定カラム名（cp932, ヘッダーあり）
# TARGETのエクスポート形式に合わせて調整が必要な場合があります
TRAINING_COLS_EXPECTED = [
    'race_horse_key',  # レース馬キー（結合キー）
    'training_date',   # 調教日 (YYYYMMDD)
    'place',           # 場所（栗東坂路/美浦坂路/栗東ウッド/美浦ウッド等）
    'course',          # コース（坂路/ウッドチップ/ポリトラック/ダート等）
    'condition',       # 馬場状態（良/稍/重/不）
    'intensity',       # 強さ（馬なり/一杯/強め/G前仕掛け等）
    'time_4f',         # 4F タイム（秒）
    'time_3f',         # 3F タイム（秒）
    'time_1f',         # 1F タイム（秒）
    'evaluation',      # 調教評価（A/B/C/D/E）
    'rider',           # 騎乗者
    'awase_result',    # 併せ結果（先着/同入/遅れ等）
    'note',            # 備考
]


def load_base_data():
    """target_odds.csvを読み込み"""
    print(f"Loading base data: {BASE_CSV}")
    df = pd.read_csv(BASE_CSV, encoding='cp932', header=None, low_memory=False)
    inv_col = {v: k for k, v in COL.items()}
    df.columns = [inv_col.get(i, f'col{i}') for i in range(df.shape[1])]
    df['race_horse_key'] = df['race_horse_key'].astype(str).str.strip()
    print(f"  Loaded {len(df)} rows, {df['race_horse_key'].nunique()} unique horse-race keys")
    return df


def load_training_csv(filepath, training_type='slope'):
    """調教CSVを読み込み。TARGETエクスポート形式を想定。

    Args:
        filepath: CSVファイルパス
        training_type: 'slope' (坂路) or 'wood' (ウッドチップ)
    """
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found. Skipping.")
        return None

    print(f"Loading training data ({training_type}): {filepath}")

    # TARGETエクスポートはcp932が一般的
    try:
        df = pd.read_csv(filepath, encoding='cp932', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)

    print(f"  Raw columns: {list(df.columns)}")
    print(f"  Loaded {len(df)} rows")

    # カラム名の正規化（TARGET形式のバリエーションに対応）
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).strip()
        if 'レース馬キー' in col_lower or 'race_horse_key' in col_lower or 'キー' in col_lower:
            col_mapping[col] = 'race_horse_key'
        elif '調教日' in col_lower or 'training_date' in col_lower or '日付' in col_lower:
            col_mapping[col] = 'training_date'
        elif '場所' in col_lower or 'place' in col_lower:
            col_mapping[col] = 'place'
        elif '強さ' in col_lower or 'intensity' in col_lower or '追切' in col_lower:
            col_mapping[col] = 'intensity'
        elif '4F' in col_lower or '4f' in col_lower:
            col_mapping[col] = 'time_4f'
        elif '3F' in col_lower or '3f' in col_lower:
            col_mapping[col] = 'time_3f'
        elif '1F' in col_lower or '1f' in col_lower:
            col_mapping[col] = 'time_1f'
        elif '評価' in col_lower or 'evaluation' in col_lower:
            col_mapping[col] = 'evaluation'
        elif '併せ' in col_lower or 'awase' in col_lower:
            col_mapping[col] = 'awase_result'
        elif 'コース' in col_lower or 'course' in col_lower:
            col_mapping[col] = 'tr_course'

    if col_mapping:
        df = df.rename(columns=col_mapping)

    if 'race_horse_key' not in df.columns:
        print(f"  ERROR: 結合キー(race_horse_key)が見つかりません。カラム: {list(df.columns)}")
        print(f"  TARGET調教CSVのカラム名を確認してください。")
        return None

    df['race_horse_key'] = df['race_horse_key'].astype(str).str.strip()
    df['training_type'] = training_type
    return df


def aggregate_training_features(training_df):
    """調教データを馬ごとに集約して特徴量を生成。

    最終追い切り（レースに最も近い調教）のデータを中心に特徴量化。
    """
    if training_df is None or len(training_df) == 0:
        return None

    features = []

    for key, group in training_df.groupby('race_horse_key'):
        row = {'race_horse_key': key}

        # 調教回数
        row['training_count'] = len(group)

        # 坂路/ウッド別カウント
        if 'training_type' in group.columns:
            row['slope_count'] = (group['training_type'] == 'slope').sum()
            row['wood_count'] = (group['training_type'] == 'wood').sum()

        # 4Fタイム関連
        if 'time_4f' in group.columns:
            t4f = pd.to_numeric(group['time_4f'], errors='coerce').dropna()
            if len(t4f) > 0:
                row['best_4f'] = t4f.min()
                row['avg_4f'] = t4f.mean()
                row['last_4f'] = t4f.iloc[-1]  # 最終追い切り
            else:
                row['best_4f'] = np.nan
                row['avg_4f'] = np.nan
                row['last_4f'] = np.nan
        else:
            row['best_4f'] = np.nan
            row['avg_4f'] = np.nan
            row['last_4f'] = np.nan

        # 3Fタイム関連
        if 'time_3f' in group.columns:
            t3f = pd.to_numeric(group['time_3f'], errors='coerce').dropna()
            if len(t3f) > 0:
                row['best_3f'] = t3f.min()
                row['avg_3f'] = t3f.mean()
                row['last_3f'] = t3f.iloc[-1]
            else:
                row['best_3f'] = np.nan
                row['avg_3f'] = np.nan
                row['last_3f'] = np.nan
        else:
            row['best_3f'] = np.nan
            row['avg_3f'] = np.nan
            row['last_3f'] = np.nan

        # 1Fタイム関連
        if 'time_1f' in group.columns:
            t1f = pd.to_numeric(group['time_1f'], errors='coerce').dropna()
            if len(t1f) > 0:
                row['best_1f'] = t1f.min()
                row['last_1f'] = t1f.iloc[-1]
            else:
                row['best_1f'] = np.nan
                row['last_1f'] = np.nan
        else:
            row['best_1f'] = np.nan
            row['last_1f'] = np.nan

        # 調教強度エンコード
        if 'intensity' in group.columns:
            last_intensity = str(group['intensity'].iloc[-1]).strip()
            intensity_map = {
                '一杯': 3, '強め': 2, 'G前仕掛け': 2, 'G前仕掛': 2,
                '馬なり': 1, '馬ナリ': 1, '直線強め': 2,
            }
            row['last_intensity_enc'] = intensity_map.get(last_intensity, 1)
            # 一杯追い切り回数
            row['ippai_count'] = sum(1 for x in group['intensity']
                                     if '一杯' in str(x))
        else:
            row['last_intensity_enc'] = 1
            row['ippai_count'] = 0

        # 調教評価エンコード
        if 'evaluation' in group.columns:
            last_eval = str(group['evaluation'].iloc[-1]).strip().upper()
            eval_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
            row['last_eval_enc'] = eval_map.get(last_eval, 3)
        else:
            row['last_eval_enc'] = 3

        # 併せ結果エンコード
        if 'awase_result' in group.columns:
            last_awase = str(group['awase_result'].iloc[-1]).strip()
            awase_map = {'先着': 2, '同入': 1, '遅れ': 0}
            row['last_awase_enc'] = awase_map.get(last_awase, 1)
            row['awase_win_count'] = sum(1 for x in group['awase_result']
                                         if '先着' in str(x))
        else:
            row['last_awase_enc'] = 1
            row['awase_win_count'] = 0

        features.append(row)

    feature_df = pd.DataFrame(features)
    print(f"  Generated {len(feature_df)} horse-race training features")
    print(f"  Feature columns: {[c for c in feature_df.columns if c != 'race_horse_key']}")
    return feature_df


def merge_and_save(base_df, training_features_df):
    """ベースデータと調教特徴量を結合して保存"""

    if training_features_df is None or len(training_features_df) == 0:
        print("\n  No training features to merge. Saving base data as-is.")
        base_df.to_csv(OUTPUT_CSV, encoding='cp932', index=False)
        return base_df

    print(f"\nMerging training features...")
    print(f"  Base data: {len(base_df)} rows")
    print(f"  Training features: {len(training_features_df)} rows")

    merged = base_df.merge(training_features_df, on='race_horse_key', how='left')

    # 結合率の確認
    new_cols = [c for c in training_features_df.columns if c != 'race_horse_key']
    for col in new_cols:
        if col in merged.columns:
            non_null = merged[col].notna().sum()
            rate = non_null / len(merged) * 100
            print(f"    {col}: {non_null}/{len(merged)} ({rate:.1f}%)")

    # NaN埋め
    fill_defaults = {
        'training_count': 0, 'slope_count': 0, 'wood_count': 0,
        'last_intensity_enc': 1, 'last_eval_enc': 3, 'last_awase_enc': 1,
        'ippai_count': 0, 'awase_win_count': 0,
    }
    for col, default in fill_defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)

    # タイム系はfloatのまま（学習時にfillna）

    print(f"\n  Merged data: {len(merged)} rows, {merged.shape[1]} cols")
    merged.to_csv(OUTPUT_CSV, encoding='cp932', index=False)
    print(f"  Saved to: {OUTPUT_CSV}")

    return merged


def main():
    parser = argparse.ArgumentParser(description='TARGET調教データ結合スクリプト')
    parser.add_argument('--slope', default=os.path.join(DATA_DIR, 'training_slope.csv'),
                        help='坂路調教CSV (default: data/training_slope.csv)')
    parser.add_argument('--wood', default=os.path.join(DATA_DIR, 'training_wood.csv'),
                        help='ウッドチップ調教CSV (default: data/training_wood.csv)')
    parser.add_argument('--output', default=OUTPUT_CSV,
                        help='出力CSV (default: data/target_odds_with_training.csv)')
    args = parser.parse_args()

    global OUTPUT_CSV
    OUTPUT_CSV = args.output

    print("=" * 60)
    print("  TARGET調教データ結合スクリプト")
    print("=" * 60)

    # 1. ベースデータ読み込み
    base_df = load_base_data()

    # 2. 調教データ読み込み
    all_training = []

    slope_df = load_training_csv(args.slope, 'slope')
    if slope_df is not None:
        all_training.append(slope_df)

    wood_df = load_training_csv(args.wood, 'wood')
    if wood_df is not None:
        all_training.append(wood_df)

    # 3. 調教データが存在すれば結合
    if all_training:
        combined = pd.concat(all_training, ignore_index=True)
        print(f"\n  Combined training data: {len(combined)} rows")

        # 調教日でソート（最新が最後）
        if 'training_date' in combined.columns:
            combined['training_date'] = pd.to_numeric(
                combined['training_date'], errors='coerce')
            combined = combined.sort_values(
                ['race_horse_key', 'training_date']).reset_index(drop=True)

        training_features = aggregate_training_features(combined)
    else:
        print("\n  No training CSV files found in data/ directory.")
        print("  Expected files:")
        print(f"    - {args.slope}")
        print(f"    - {args.wood}")
        print("\n  TARGETから調教データをCSVエクスポートして配置してください。")
        print("  想定カラム: レース馬キー, 調教日, 場所, 強さ, 4F, 3F, 1F, 評価, 併せ結果")
        training_features = None

    # 4. 結合・保存
    merged = merge_and_save(base_df, training_features)

    print("\n" + "=" * 60)
    print("  完了")
    print("=" * 60)
    return merged


if __name__ == '__main__':
    main()
