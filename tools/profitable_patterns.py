#!/usr/bin/env python3
"""profitable_patterns.py - 多次元交差による収益パターン分析

ウォークフォワードバックテスト結果 + JRA公式配当データを用いて、
オッズ帯×会場×AI予測順位×条件の多次元交差でROIを算出し、
収益性の高いパターンを抽出する。

初回実行時はWFバックテスト(LGBのみ、軽量版)を実行してキャッシュを作成。
2回目以降はキャッシュから即座に分析結果を生成。

Usage:
    python tools/profitable_patterns.py              # フル分析
    python tools/profitable_patterns.py --min-n 30   # 最低サンプル数変更
    python tools/profitable_patterns.py --summary     # サマリーのみ表示
    python tools/profitable_patterns.py --rebuild     # キャッシュ再構築
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'wf_race_results_cache.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'profitable_patterns.json')

TEST_YEARS = list(range(2020, 2026))

# Course code mapping
COURSE_CODE_TO_NAME = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
}
COURSE_NAME_TO_CODE = {v: k for k, v in COURSE_CODE_TO_NAME.items()}

NICHI_TO_CHAR = {i: str(i) for i in range(1, 10)}
NICHI_TO_CHAR.update({10: 'A', 11: 'B', 12: 'C'})

# Odds bands
ODDS_BANDS = [
    ('1-3', 1.0, 3.0),
    ('3-10', 3.0, 10.0),
    ('10-30', 10.0, 30.0),
    ('30-100', 30.0, 100.0),
    ('100+', 100.0, 99999.0),
]

# AI rank labels
RANK_LABELS = {1: '1位', 2: '2位', 3: '3位'}
# 4-6 grouped
RANK_GROUP_LABEL = '4-6位'


def load_payouts():
    """jra_payouts.csv をロードして race_key -> payout のルックアップを作成"""
    path = os.path.join(DATA_DIR, 'jra_payouts.csv')
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return {}

    pay_df = pd.read_csv(path, encoding='utf-8', dtype=str)
    print(f"  Payouts loaded: {len(pay_df)} records")

    lookup = {}
    for _, row in pay_df.iterrows():
        date_str = str(row['race_date'])
        course_name = str(row['course'])
        kai_int = int(row['kai'])
        nichi_int = int(row['nichi'])
        race_num = str(row['race_num']).zfill(2)

        cc = COURSE_NAME_TO_CODE.get(course_name)
        if cc is None:
            continue

        year_2d = date_str[2:4]
        nichi_char = NICHI_TO_CHAR.get(nichi_int, str(nichi_int))
        key = f"{cc}{year_2d}{kai_int}{nichi_char}{race_num}"

        trio_payout = 0
        try:
            trio_payout = int(row.get('trio_payout', 0))
        except (ValueError, TypeError):
            pass

        umaren_payout = 0
        try:
            umaren_payout = int(row.get('umaren_payout', 0))
        except (ValueError, TypeError):
            pass

        tansho_payout = 0
        try:
            tansho_payout = int(row.get('tansho_payout', 0))
        except (ValueError, TypeError):
            pass

        lookup[key] = {
            'race_date': date_str,
            'trio_nums': str(row.get('trio_nums', '')),
            'trio_payout': trio_payout,
            'umaren_nums': str(row.get('umaren_nums', '')),
            'umaren_payout': umaren_payout,
            'tansho_nums': str(row.get('tansho_nums', '')),
            'tansho_payout': tansho_payout,
        }

    return lookup


def parse_trio_nums(s):
    """'1-5-7' -> frozenset({1, 5, 7})"""
    if not s or s == 'nan':
        return None
    try:
        return frozenset(int(x) for x in s.split('-'))
    except (ValueError, AttributeError):
        return None


def parse_umaren_nums(s):
    """'3-7' -> frozenset({3, 7})"""
    if not s or s == 'nan':
        return None
    try:
        return frozenset(int(x) for x in s.split('-'))
    except (ValueError, AttributeError):
        return None


def classify_condition(num_horses, distance, condition_enc):
    """条件クラス分類"""
    heavy = condition_enc >= 2
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


def calc_trio_bets(ranking):
    """三連複フォーメーション: TOP1軸-TOP2,3-TOP2~6 (最大7点)"""
    if len(ranking) < 3:
        return []
    nums = ranking[:min(6, len(ranking))]
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    return sorted(set(
        tuple(sorted({n1, s, t}))
        for s in second for t in third
        if len(set({n1, s, t})) == 3
    ))


def calc_umaren_bets(ranking):
    """馬連: TOP1-TOP2, TOP1-TOP3"""
    if len(ranking) < 3:
        return []
    return [tuple(sorted([ranking[0], ranking[1]])),
            tuple(sorted([ranking[0], ranking[2]]))]


def get_odds_band(odds):
    """オッズ帯を返す"""
    if odds is None or np.isnan(odds):
        return 'unknown'
    for label, lo, hi in ODDS_BANDS:
        if lo <= odds < hi:
            return label
    return 'unknown'


def get_ai_rank_label(rank):
    """AI予測順位ラベル"""
    if rank <= 3:
        return RANK_LABELS[rank]
    elif rank <= 6:
        return RANK_GROUP_LABEL
    return None  # 7位以下は除外


def get_distance_label(dist):
    """距離ラベル"""
    if dist <= 1200:
        return 'sprint'
    elif dist <= 1600:
        return 'mile'
    elif dist <= 2200:
        return 'middle'
    return 'long'


def get_surface_label(surface_enc):
    """馬場ラベル"""
    if surface_enc == 0:
        return '芝'
    elif surface_enc == 1:
        return 'ダート'
    return '障害'


# ============================================================
# Walk-Forward Backtest (lightweight LGB only)
# ============================================================

def run_wf_backtest():
    """LGB単体のウォークフォワードバックテストを実行し、
    レース単位の結果をキャッシュに保存する。"""

    print("=" * 70)
    print("Walk-Forward Backtest (LGB lightweight) for Pattern Analysis")
    print("=" * 70)

    # Import training infrastructure
    try:
        from train_v92_central import (
            load_data, encode_categoricals, encode_sires, load_training_times,
            merge_training_features, compute_jockey_wr, compute_trainer_stats,
            compute_horse_career, compute_sire_performance, load_lap_data,
            compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
        )
        from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A
    except ImportError as e:
        print(f"ERROR: Cannot import training modules: {e}")
        print("Make sure train/train_v92_central.py and train_v92_leakfree.py exist.")
        return None

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    COURSE_MAP_INV = {v: k for k, v in COURSE_MAP.items()}

    # Load and prepare data
    print("\n[1/4] Loading data...")
    t0 = time.time()
    df = load_data()
    print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

    print("[2/4] Feature engineering...")
    t0 = time.time()
    df = encode_categoricals(df)
    df, _, _ = encode_sires(df)

    training_times = load_training_times()
    if training_times is not None:
        df = merge_training_features(df, training_times)

    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)

    lap_df = load_lap_data()
    if lap_df is not None:
        merge_key = 'race_id_str' if 'race_id_str' in lap_df.columns else 'race_id'
        if merge_key != 'race_id':
            lap_df = lap_df.rename(columns={merge_key: 'race_id'})
        if 'race_id' in df.columns and 'race_id' in lap_df.columns:
            df['race_id'] = df['race_id'].astype(str)
            lap_df['race_id'] = lap_df['race_id'].astype(str)
            df = df.merge(lap_df, on='race_id', how='left', suffixes=('', '_lap'))

    df = compute_lag_features(df)
    df = build_features(df)
    print(f"  Features built in {time.time()-t0:.1f}s")

    # Load odds data
    print("[3/4] Loading odds data...")
    odds_path = os.path.join(DATA_DIR, 'odds_history.csv')
    if os.path.exists(odds_path):
        odds_df = pd.read_csv(odds_path, encoding='utf-8-sig',
                              usecols=['race_id', 'umaban', 'tansho_odds'])
        odds_df['race_id'] = odds_df['race_id'].astype(str).str.zfill(9)
        df['race_id_str_9'] = df['race_id'].astype(str).str.zfill(9)
        df = df.merge(odds_df, left_on=['race_id_str_9', 'umaban'],
                      right_on=['race_id', 'umaban'], how='left',
                      suffixes=('', '_odds'))
        if 'tansho_odds_odds' in df.columns:
            df['tansho_odds'] = df['tansho_odds_odds']
            df.drop(columns=['tansho_odds_odds'], inplace=True, errors='ignore')
        elif 'tansho_odds' not in df.columns:
            df['tansho_odds'] = np.nan
        df.drop(columns=['race_id_odds', 'race_id_str_9'], inplace=True, errors='ignore')
        print(f"  Odds merged. Non-null: {df['tansho_odds'].notna().sum()}/{len(df)}")
    else:
        print("  WARNING: odds_history.csv not found, tansho_odds will be NaN")
        df['tansho_odds'] = np.nan

    # Determine features
    features = [f for f in FEATURES_PATTERN_A if f in df.columns and f not in LEAK_FEATURES_A]
    print(f"  Using {len(features)} features")

    # Build race_id_str for payout matching
    # race_id format: XYYKKNNRR (9 digits, 0-padded)
    # X=course_code(1-2 digits), YY=year, KK=kai+nichi, NN=race_num, RR=horse_entry
    # For payout matching we need: CC(2)+YY(2)+K(1)+N(1)+RR(2) from race-level info
    # Actually race_id already encodes course+year+kai+nichi+race_num+umaban
    # Let's construct race-level id: race_id without umaban part
    df['race_id_padded'] = df['race_id'].astype(str).str.zfill(9)
    # race_id format: CYYKKNNHH where C=course, YY=year, KK=kai?, NN=racenum?, HH=umaban?
    # From data: 615110101 -> course=中山(06), year=15, kai=1, nichi=1, race_num=01, umaban=01
    # So: 0615110101 (10 digits) -> CC=06, YY=15, K=1, N=1, RR=01, HH=01
    # But stored as 9 digits: 615110101
    # Padded to 10: 0615110101
    df['race_id_10'] = df['race_id'].astype(str).str.zfill(10)
    # CC=first 2, YY=next 2, K=next 1, N=next 1, RR=next 2, HH=last 2
    df['race_id_str'] = df['race_id_10'].str[:8]  # CC+YY+K+N+RR (without umaban)

    # Target variable
    df['target'] = (df['finish'] <= 3).astype(int)

    # Load payouts
    payout_lookup = load_payouts()

    # Walk-forward
    print("\n[4/4] Walk-forward backtest...")
    all_race_results = []

    LGB_PARAMS = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }

    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    for test_year in TEST_YEARS:
        yr_2d = test_year % 100
        print(f"\n  --- Test year {test_year} ---")

        train_mask = df['year'] < yr_2d
        test_mask = df['year'] == yr_2d

        if test_mask.sum() == 0:
            print(f"    No test data for {test_year}")
            continue

        # Encode sires using only training data
        df_fold = df.copy()
        train_sires = df_fold[train_mask]['father'].value_counts()
        top_sires = train_sires.head(100).index.tolist()
        sire_map = {s: i for i, s in enumerate(top_sires)}
        df_fold['sire_enc'] = df_fold['father'].map(sire_map).fillna(100).astype(int)

        train_bms = df_fold[train_mask]['bms'].value_counts()
        top_bms = train_bms.head(100).index.tolist()
        bms_map = {s: i for i, s in enumerate(top_bms)}
        df_fold['bms_enc'] = df_fold['bms'].map(bms_map).fillna(100).astype(int)

        # Prepare features
        avail_feats = [f for f in features if f in df_fold.columns]
        X_train = df_fold[train_mask][avail_feats].values.astype(np.float32)
        y_train = df_fold[train_mask]['target'].values
        X_test = df_fold[test_mask][avail_feats].values.astype(np.float32)
        y_test = df_fold[test_mask]['target'].values

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Validation split (last 20% of training)
        n_train = len(X_train)
        n_valid = max(1, int(n_train * 0.2))
        X_tr = X_train[:n_train - n_valid]
        y_tr = y_train[:n_train - n_valid]
        X_va = X_train[n_train - n_valid:]
        y_va = y_train[n_train - n_valid:]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=avail_feats)
        dvalid = lgb.Dataset(X_va, label=y_va, feature_name=avail_feats, reference=dtrain)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, preds)
        print(f"    AUC: {auc:.4f}")

        # Collect per-race results
        test_df = df_fold[test_mask].copy()
        test_df['pred_score'] = preds

        race_ids = test_df['race_id_str'].unique()
        n_matched = 0
        n_total = 0

        for rid in race_ids:
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 3:
                continue

            n_total += 1
            row0 = race_df.iloc[0]

            num_horses = int(row0.get('num_horses', row0.get('num_horses_val', len(race_df))))
            distance = int(row0['distance'])
            condition_enc = int(row0.get('condition_enc', 0))
            course_enc = int(row0['course_enc'])
            surface_enc = int(row0['surface_enc'])

            cond_class = classify_condition(num_horses, distance, condition_enc)
            course_name = {v: k for k, v in COURSE_MAP.items()}.get(course_enc, 'unknown')
            dist_label = get_distance_label(distance)
            surf_label = get_surface_label(surface_enc)

            # AI ranking by prediction score (descending)
            race_df = race_df.sort_values('pred_score', ascending=False).reset_index(drop=True)
            ranking = race_df['umaban'].astype(int).tolist()

            # Per-horse info with AI rank and odds
            horse_info = []
            for idx, (_, hrow) in enumerate(race_df.iterrows()):
                ai_rank = idx + 1
                odds_val = float(hrow['tansho_odds']) if pd.notna(hrow.get('tansho_odds')) else None
                horse_info.append({
                    'umaban': int(hrow['umaban']),
                    'ai_rank': ai_rank,
                    'tansho_odds': odds_val,
                    'finish': int(hrow['finish']),
                    'pred_score': float(hrow['pred_score']),
                })

            # Actual top 3
            race_by_finish = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_by_finish.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue

            # Generate bets based on condition
            trio_bets = calc_trio_bets(ranking)
            umaren_bets = calc_umaren_bets(ranking)

            # Check against actual payouts
            payout_info = payout_lookup.get(rid)
            trio_hit = False
            trio_return = 0
            umaren_hit = False
            umaren_return = 0
            tansho_hit = False
            tansho_return = 0

            if payout_info:
                n_matched += 1

                # Trio check
                actual_trio = parse_trio_nums(payout_info['trio_nums'])
                if actual_trio:
                    for bet in trio_bets:
                        if frozenset(bet) == actual_trio:
                            trio_hit = True
                            trio_return = payout_info['trio_payout']
                            break

                # Umaren check
                actual_umaren = parse_umaren_nums(payout_info['umaren_nums'])
                if actual_umaren:
                    for bet in umaren_bets:
                        if frozenset(bet) == actual_umaren:
                            umaren_hit = True
                            umaren_return = payout_info['umaren_payout']
                            break

                # Tansho (単勝): AI 1位の馬が1着か
                tansho_winner = payout_info.get('tansho_nums', '')
                try:
                    if int(tansho_winner) == ranking[0]:
                        tansho_hit = True
                        tansho_return = payout_info.get('tansho_payout', 0)
                except (ValueError, TypeError):
                    pass

            race_result = {
                'race_id_str': rid,
                'year': test_year,
                'course': course_name,
                'distance': distance,
                'dist_label': dist_label,
                'surface': surf_label,
                'condition_enc': condition_enc,
                'num_horses': num_horses,
                'cond_class': cond_class,
                'horses': horse_info,
                'trio_hit': trio_hit,
                'trio_return': trio_return,
                'umaren_hit': umaren_hit,
                'umaren_return': umaren_return,
                'tansho_hit': tansho_hit,
                'tansho_return': tansho_return,
                'has_payout': payout_info is not None,
            }
            all_race_results.append(race_result)

        print(f"    Races: {n_total}, Payout matched: {n_matched}")

    print(f"\nTotal race results: {len(all_race_results)}")

    # Save cache
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_years': TEST_YEARS,
            'model': 'LGB_lightweight_WF',
            'n_races': len(all_race_results),
            'races': all_race_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Cache saved to {CACHE_PATH}")

    return all_race_results


def load_cache():
    """キャッシュからレース結果をロード"""
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Cache loaded: {data['n_races']} races, generated {data['generated']}")
    return data['races']


# ============================================================
# Pattern Analysis
# ============================================================

def analyze_patterns(races, min_n=20, summary_only=False):
    """多次元交差によるパターン分析"""

    # Only use races with payout data
    races_with_payout = [r for r in races if r.get('has_payout', False)]
    print(f"\nAnalyzing {len(races_with_payout)} races with payout data "
          f"(out of {len(races)} total)")

    if len(races_with_payout) == 0:
        print("ERROR: No races with payout data. Check jra_payouts.csv coverage.")
        return None

    # Build flat records for cross-analysis
    # Each record = one (race, horse) combination for horses ranked 1-6
    records = []
    for race in races_with_payout:
        for horse in race['horses']:
            ai_rank = horse['ai_rank']
            if ai_rank > 6:
                continue  # Only analyze top 6 predictions
            rank_label = get_ai_rank_label(ai_rank)
            if rank_label is None:
                continue
            odds_band = get_odds_band(horse['tansho_odds'])

            records.append({
                'race_id': race['race_id_str'],
                'year': race['year'],
                'course': race['course'],
                'distance': race['distance'],
                'dist_label': race['dist_label'],
                'surface': race['surface'],
                'cond_class': race['cond_class'],
                'num_horses': race['num_horses'],
                'umaban': horse['umaban'],
                'ai_rank': ai_rank,
                'rank_label': rank_label,
                'odds_band': odds_band,
                'tansho_odds': horse['tansho_odds'],
                'finish': horse['finish'],
                'is_top3': horse['finish'] <= 3,
                'is_winner': horse['finish'] == 1,
                # Race-level bet results
                'trio_hit': race['trio_hit'],
                'trio_return': race['trio_return'],
                'umaren_hit': race['umaren_hit'],
                'umaren_return': race['umaren_return'],
                'tansho_hit': race['tansho_hit'],
                'tansho_return': race['tansho_return'],
            })

    df = pd.DataFrame(records)
    print(f"  Records for analysis: {len(df)} (top-6 horses from {len(races_with_payout)} races)")

    # ---- Dimension definitions ----
    # For race-level bets (trio/umaren), we analyze by race attributes
    # For tansho, we analyze by individual horse attributes

    # --- Race-level analysis (trio/umaren) ---
    race_df = pd.DataFrame(races_with_payout)

    # Cross dimensions for race-level analysis
    dimensions_race = {
        'venue': 'course',
        'condition': 'cond_class',
        'distance': 'dist_label',
        'surface': 'surface',
    }

    # --- Horse-level analysis (tansho, individual accuracy) ---
    # For each AI rank, what's the win rate by odds band and venue

    all_patterns = []
    total_tested = 0

    # ============================
    # Pattern type 1: Race-level (trio) by venue x condition
    # ============================
    print("\n--- Pattern Analysis: Trio (三連複) ---")
    for venue in sorted(race_df['course'].unique()):
        for cond in sorted(race_df['cond_class'].unique()):
            subset = race_df[(race_df['course'] == venue) & (race_df['cond_class'] == cond)]
            total_tested += 1
            n = len(subset)
            if n < min_n:
                continue
            hits = int(subset['trio_hit'].sum())
            total_return = int(subset['trio_return'].sum())
            investment = n * 700  # 7点 x 100円
            roi = (total_return / investment * 100) if investment > 0 else 0
            avg_payout = total_return / hits if hits > 0 else 0

            if roi > 100:
                stars = 3 if roi > 200 and n >= 50 else (2 if roi > 150 and n >= 30 else 1)
                all_patterns.append({
                    'type': 'trio',
                    'odds_band': 'all',
                    'venue': venue,
                    'ai_rank': 'all',
                    'condition': cond,
                    'surface': 'all',
                    'distance': 'all',
                    'n': n,
                    'hits': hits,
                    'hit_rate': round(hits / n, 3) if n > 0 else 0,
                    'roi': round(roi, 1),
                    'avg_payout': round(avg_payout),
                    'investment': investment,
                    'total_return': total_return,
                    'stars': stars,
                })

    # ============================
    # Pattern type 2: Race-level (trio) by venue x condition x distance
    # ============================
    for venue in sorted(race_df['course'].unique()):
        for cond in sorted(race_df['cond_class'].unique()):
            for dist in sorted(race_df['dist_label'].unique()):
                subset = race_df[(race_df['course'] == venue) &
                                 (race_df['cond_class'] == cond) &
                                 (race_df['dist_label'] == dist)]
                total_tested += 1
                n = len(subset)
                if n < min_n:
                    continue
                hits = int(subset['trio_hit'].sum())
                total_return = int(subset['trio_return'].sum())
                investment = n * 700
                roi = (total_return / investment * 100) if investment > 0 else 0
                avg_payout = total_return / hits if hits > 0 else 0

                if roi > 100:
                    stars = 3 if roi > 200 and n >= 50 else (2 if roi > 150 and n >= 30 else 1)
                    all_patterns.append({
                        'type': 'trio',
                        'odds_band': 'all',
                        'venue': venue,
                        'ai_rank': 'all',
                        'condition': cond,
                        'surface': 'all',
                        'distance': dist,
                        'n': n,
                        'hits': hits,
                        'hit_rate': round(hits / n, 3) if n > 0 else 0,
                        'roi': round(roi, 1),
                        'avg_payout': round(avg_payout),
                        'investment': investment,
                        'total_return': total_return,
                        'stars': stars,
                    })

    # ============================
    # Pattern type 3: Race-level (umaren) by venue x condition (E condition)
    # ============================
    for venue in sorted(race_df['course'].unique()):
        for cond in sorted(race_df['cond_class'].unique()):
            subset = race_df[(race_df['course'] == venue) & (race_df['cond_class'] == cond)]
            total_tested += 1
            n = len(subset)
            if n < min_n:
                continue
            hits = int(subset['umaren_hit'].sum())
            total_return = int(subset['umaren_return'].sum())
            investment = n * 700  # 2点 x 350円 = 700円 (条件Eの場合)
            roi = (total_return / investment * 100) if investment > 0 else 0
            avg_payout = total_return / hits if hits > 0 else 0

            if roi > 100:
                stars = 3 if roi > 200 and n >= 50 else (2 if roi > 150 and n >= 30 else 1)
                all_patterns.append({
                    'type': 'umaren',
                    'odds_band': 'all',
                    'venue': venue,
                    'ai_rank': 'all',
                    'condition': cond,
                    'surface': 'all',
                    'distance': 'all',
                    'n': n,
                    'hits': hits,
                    'hit_rate': round(hits / n, 3) if n > 0 else 0,
                    'roi': round(roi, 1),
                    'avg_payout': round(avg_payout),
                    'investment': investment,
                    'total_return': total_return,
                    'stars': stars,
                })

    # ============================
    # Pattern type 4: Tansho by ai_rank x odds_band x venue
    # ============================
    print("--- Pattern Analysis: Tansho (単勝) by AI rank x odds x venue ---")
    # Only AI rank 1 matters for tansho (betting on AI top pick)
    ai1_df = df[df['ai_rank'] == 1].copy()
    for odds_band in [b[0] for b in ODDS_BANDS]:
        for venue in sorted(ai1_df['course'].unique()):
            subset = ai1_df[(ai1_df['odds_band'] == odds_band) & (ai1_df['course'] == venue)]
            total_tested += 1
            n = len(subset)
            if n < min_n:
                continue
            wins = int(subset['is_winner'].sum())
            # Tansho return: sum of odds * 100 for winners
            total_return = 0
            for _, row in subset[subset['is_winner']].iterrows():
                if row['tansho_odds'] and not np.isnan(row['tansho_odds']):
                    total_return += int(row['tansho_odds'] * 100)
                else:
                    total_return += 200  # fallback
            investment = n * 100
            roi = (total_return / investment * 100) if investment > 0 else 0
            avg_payout = total_return / wins if wins > 0 else 0

            if roi > 100:
                stars = 3 if roi > 200 and n >= 50 else (2 if roi > 150 and n >= 30 else 1)
                all_patterns.append({
                    'type': 'tansho',
                    'odds_band': odds_band,
                    'venue': venue,
                    'ai_rank': '1位',
                    'condition': 'all',
                    'surface': 'all',
                    'distance': 'all',
                    'n': n,
                    'hits': wins,
                    'hit_rate': round(wins / n, 3) if n > 0 else 0,
                    'roi': round(roi, 1),
                    'avg_payout': round(avg_payout),
                    'investment': investment,
                    'total_return': total_return,
                    'stars': stars,
                })

    # ============================
    # Pattern type 5: Trio by ai_rank_1_odds_band x condition
    # (Analyze trio hit rate when AI top pick has specific odds range)
    # ============================
    print("--- Pattern Analysis: Trio by AI-1 odds band x condition ---")
    # Enrich race_df with AI top pick odds band
    race_top1_odds = {}
    for race in races_with_payout:
        top1_horse = [h for h in race['horses'] if h['ai_rank'] == 1]
        if top1_horse:
            race_top1_odds[race['race_id_str']] = get_odds_band(top1_horse[0]['tansho_odds'])

    race_df['top1_odds_band'] = race_df['race_id_str'].map(race_top1_odds)

    for odds_band in [b[0] for b in ODDS_BANDS]:
        for cond in sorted(race_df['cond_class'].unique()):
            subset = race_df[(race_df['top1_odds_band'] == odds_band) &
                             (race_df['cond_class'] == cond)]
            total_tested += 1
            n = len(subset)
            if n < min_n:
                continue
            hits = int(subset['trio_hit'].sum())
            total_return = int(subset['trio_return'].sum())
            investment = n * 700
            roi = (total_return / investment * 100) if investment > 0 else 0
            avg_payout = total_return / hits if hits > 0 else 0

            if roi > 100:
                stars = 3 if roi > 200 and n >= 50 else (2 if roi > 150 and n >= 30 else 1)
                all_patterns.append({
                    'type': 'trio',
                    'odds_band': odds_band,
                    'venue': 'all',
                    'ai_rank': '1位オッズ帯',
                    'condition': cond,
                    'surface': 'all',
                    'distance': 'all',
                    'n': n,
                    'hits': hits,
                    'hit_rate': round(hits / n, 3) if n > 0 else 0,
                    'roi': round(roi, 1),
                    'avg_payout': round(avg_payout),
                    'investment': investment,
                    'total_return': total_return,
                    'stars': stars,
                })

    # ============================
    # Pattern type 6: Trio by surface x condition
    # ============================
    print("--- Pattern Analysis: Trio by surface x condition ---")
    for surface in sorted(race_df['surface'].unique()):
        for cond in sorted(race_df['cond_class'].unique()):
            subset = race_df[(race_df['surface'] == surface) &
                             (race_df['cond_class'] == cond)]
            total_tested += 1
            n = len(subset)
            if n < min_n:
                continue
            hits = int(subset['trio_hit'].sum())
            total_return = int(subset['trio_return'].sum())
            investment = n * 700
            roi = (total_return / investment * 100) if investment > 0 else 0
            avg_payout = total_return / hits if hits > 0 else 0

            if roi > 100:
                stars = 3 if roi > 200 and n >= 50 else (2 if roi > 150 and n >= 30 else 1)
                all_patterns.append({
                    'type': 'trio',
                    'odds_band': 'all',
                    'venue': 'all',
                    'ai_rank': 'all',
                    'condition': cond,
                    'surface': surface,
                    'distance': 'all',
                    'n': n,
                    'hits': hits,
                    'hit_rate': round(hits / n, 3) if n > 0 else 0,
                    'roi': round(roi, 1),
                    'avg_payout': round(avg_payout),
                    'investment': investment,
                    'total_return': total_return,
                    'stars': stars,
                })

    # ============================
    # Pattern type 7: AI top3 accuracy by rank x odds_band
    # ============================
    print("--- Pattern Analysis: AI prediction accuracy by rank x odds ---")
    for rank_label in ['1位', '2位', '3位', '4-6位']:
        for odds_band in [b[0] for b in ODDS_BANDS]:
            subset = df[(df['rank_label'] == rank_label) & (df['odds_band'] == odds_band)]
            total_tested += 1
            n = len(subset)
            if n < min_n:
                continue
            top3_count = int(subset['is_top3'].sum())
            top3_rate = top3_count / n if n > 0 else 0

            # For informational purposes (not directly a betting ROI pattern)
            # but useful for understanding model strength
            if top3_rate > 0.33:  # Better than random (1/3 for top 3)
                all_patterns.append({
                    'type': 'accuracy',
                    'odds_band': odds_band,
                    'venue': 'all',
                    'ai_rank': rank_label,
                    'condition': 'all',
                    'surface': 'all',
                    'distance': 'all',
                    'n': n,
                    'hits': top3_count,
                    'hit_rate': round(top3_rate, 3),
                    'roi': round(top3_rate * 100, 1),  # Use as percentage
                    'avg_payout': 0,
                    'investment': 0,
                    'total_return': 0,
                    'stars': 3 if top3_rate > 0.5 and n >= 50 else (
                        2 if top3_rate > 0.4 and n >= 30 else 1),
                })

    # ============================
    # Pattern type 8: Condition x year stability check
    # ============================
    print("--- Pattern Analysis: Year stability by condition ---")
    yearly_cond = {}
    for cond in sorted(race_df['cond_class'].unique()):
        yearly_cond[cond] = {}
        for year in TEST_YEARS:
            subset = race_df[(race_df['cond_class'] == cond) &
                             (race_df['year'] == year)]
            n = len(subset)
            if n == 0:
                continue
            hits = int(subset['trio_hit'].sum())
            total_return = int(subset['trio_return'].sum())
            investment = n * 700
            roi = (total_return / investment * 100) if investment > 0 else 0
            yearly_cond[cond][year] = {'n': n, 'roi': round(roi, 1)}

    # Sort patterns by ROI descending
    # Separate bet patterns from accuracy patterns
    bet_patterns = [p for p in all_patterns if p['type'] != 'accuracy']
    acc_patterns = [p for p in all_patterns if p['type'] == 'accuracy']

    bet_patterns.sort(key=lambda x: x['roi'], reverse=True)
    acc_patterns.sort(key=lambda x: x['hit_rate'], reverse=True)

    # Generate recommendations
    recommendations = []
    star3 = [p for p in bet_patterns if p['stars'] == 3]
    star2 = [p for p in bet_patterns if p['stars'] == 2]

    if star3:
        best = star3[0]
        recommendations.append(
            f"最高ROIパターン: {best['venue']} {best['condition']}条件 "
            f"{best['type']} ROI {best['roi']}% (N={best['n']})"
        )

    # Venue recommendations
    venue_roi = defaultdict(lambda: {'investment': 0, 'return': 0, 'n': 0})
    for p in bet_patterns:
        if p['venue'] != 'all' and p['type'] == 'trio':
            venue_roi[p['venue']]['investment'] += p['investment']
            venue_roi[p['venue']]['return'] += p['total_return']
            venue_roi[p['venue']]['n'] += p['n']

    for venue, data in sorted(venue_roi.items(), key=lambda x: -x[1]['return']):
        if data['investment'] > 0:
            roi = data['return'] / data['investment'] * 100
            if roi > 150:
                star_str = '★★★' if roi > 200 else '★★'
                recommendations.append(f"{venue}は{star_str} (集計ROI {roi:.0f}%)")

    # Condition recommendations
    for cond in sorted(yearly_cond.keys()):
        years_data = yearly_cond[cond]
        if len(years_data) >= 3:
            rois = [v['roi'] for v in years_data.values()]
            avg_roi = np.mean(rois)
            if avg_roi > 150 and min(rois) > 80:
                recommendations.append(
                    f"条件{cond}は安定高ROI (平均{avg_roi:.0f}%, "
                    f"最低{min(rois):.0f}%, {len(years_data)}年間)"
                )

    # Summary
    summary = {
        'total_profitable': len(bet_patterns),
        'total_accuracy_patterns': len(acc_patterns),
        'best_roi_pattern': bet_patterns[0] if bet_patterns else None,
        'star3_count': len(star3),
        'star2_count': len(star2),
        'yearly_condition_stability': yearly_cond,
        'recommendations': recommendations,
    }

    result = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_patterns_tested': total_tested,
        'min_n': min_n,
        'total_races_analyzed': len(races_with_payout),
        'profitable_patterns': bet_patterns,
        'accuracy_patterns': acc_patterns,
        'summary': summary,
    }

    return result


def print_results(result, summary_only=False):
    """結果を表示"""
    if result is None:
        print("No results to display.")
        return

    summary = result['summary']

    print("\n" + "=" * 70)
    print("PROFITABLE PATTERNS ANALYSIS")
    print("=" * 70)
    print(f"Generated: {result['generated']}")
    print(f"Total races analyzed: {result['total_races_analyzed']}")
    print(f"Total patterns tested: {result['total_patterns_tested']}")
    print(f"Profitable bet patterns: {summary['total_profitable']}")
    print(f"  ★★★ (ROI>200%, N>=50): {summary['star3_count']}")
    print(f"  ★★ (ROI>150%, N>=30): {summary['star2_count']}")
    print(f"  ★ (ROI>100%, N>=20): "
          f"{summary['total_profitable'] - summary['star3_count'] - summary['star2_count']}")

    # Recommendations
    print("\n--- Recommendations ---")
    for rec in summary.get('recommendations', []):
        print(f"  {rec}")

    # Year stability
    yearly = summary.get('yearly_condition_stability', {})
    if yearly:
        print("\n--- Year x Condition Trio ROI ---")
        years = sorted(set(y for cdata in yearly.values() for y in cdata.keys()))
        header = f"{'Cond':>5}" + "".join(f"{y:>10}" for y in years)
        print(header)
        for cond in sorted(yearly.keys()):
            row = f"{cond:>5}"
            for y in years:
                if y in yearly[cond]:
                    d = yearly[cond][y]
                    row += f"{d['roi']:>8.0f}% "
                else:
                    row += f"{'N/A':>10}"
            print(row)

    if summary_only:
        return

    # Top patterns table
    patterns = result.get('profitable_patterns', [])
    if patterns:
        print(f"\n--- Top 30 Profitable Bet Patterns (sorted by ROI) ---")
        print(f"{'#':>3} {'Type':>7} {'Venue':>5} {'Cond':>5} {'Odds':>7} "
              f"{'Dist':>8} {'Surf':>4} "
              f"{'N':>5} {'Hits':>5} {'Rate':>6} {'ROI':>8} {'AvgPay':>8} {'★':>3}")
        for i, p in enumerate(patterns[:30]):
            star_str = '★' * p['stars']
            print(f"{i+1:>3} {p['type']:>7} {p['venue']:>5} {p['condition']:>5} "
                  f"{p['odds_band']:>7} {p['distance']:>8} {p['surface']:>4} "
                  f"{p['n']:>5} {p['hits']:>5} {p['hit_rate']:>6.1%} "
                  f"{p['roi']:>7.1f}% {p['avg_payout']:>8} {star_str:>3}")

    # Accuracy patterns
    acc_patterns = result.get('accuracy_patterns', [])
    if acc_patterns:
        print(f"\n--- AI Prediction Accuracy (Top3 rate, > 33%) ---")
        print(f"{'Rank':>6} {'Odds':>7} {'N':>6} {'Top3':>6} {'Rate':>7}")
        for p in acc_patterns[:20]:
            print(f"{p['ai_rank']:>6} {p['odds_band']:>7} {p['n']:>6} "
                  f"{p['hits']:>6} {p['hit_rate']:>6.1%}")


def main():
    parser = argparse.ArgumentParser(
        description='多次元交差による収益パターン分析')
    parser.add_argument('--min-n', type=int, default=20,
                        help='最低サンプル数 (default: 20)')
    parser.add_argument('--summary', action='store_true',
                        help='サマリーのみ表示')
    parser.add_argument('--rebuild', action='store_true',
                        help='キャッシュを再構築')
    args = parser.parse_args()

    print("=" * 70)
    print("Profitable Pattern Analysis for Keiba AI")
    print("=" * 70)

    # Load or build cache
    races = None
    if not args.rebuild:
        races = load_cache()

    if races is None:
        print("\nNo cache found. Running walk-forward backtest (this takes ~10-30 min)...")
        races = run_wf_backtest()
        if races is None:
            print("ERROR: Backtest failed. Exiting.")
            sys.exit(1)

    # Analyze patterns
    result = analyze_patterns(races, min_n=args.min_n, summary_only=args.summary)
    if result is None:
        print("ERROR: Analysis failed.")
        sys.exit(1)

    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Display
    print_results(result, summary_only=args.summary)


if __name__ == '__main__':
    main()
