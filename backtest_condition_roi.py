#!/usr/bin/env python
"""条件別・クラス別ROI格付けバックテスト

Train: 2010-2022 (single model) → Test: 2023-2025
3券種（trio 7点 / umaren 1軸2流し / wide 1軸2流し）で条件A-X別・クラス別ROIを算出。
JRA公式配当データで実ROIを計算。
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A

from backtest_central_leakfree import (
    classify_condition, calc_trio_bets, calc_umaren_bets, calc_wide_bets,
    encode_sires_fold, train_lgb_fold,
)
from calc_actual_roi import (
    load_payouts, parse_trio_nums, parse_umaren_nums, parse_wide_data,
)

# TARGET JV class_code → グループ分類
# cc=15:新馬, 7:未勝利, 23:1勝(500万下), 43:2勝(1000万下),
# 67:3勝(1600万下), 131:OP特別, 115:リステッド,
# 163:G3, 179:G2, 195:G1, 147:障害重賞等

def classify_class_group(class_code):
    """TARGET JV class_code → 分析用グループ名"""
    try:
        cc = int(class_code)
    except (ValueError, TypeError):
        return 'その他'
    if cc == 15: return '新馬'
    if cc == 7: return '未勝利'
    if cc == 23: return '1勝'
    if cc == 43: return '2勝'
    if cc == 67: return '3勝'
    if cc in (131, 115): return 'OP/リステッド'
    if cc in (163, 179, 195): return '重賞'
    return 'その他'

CLASS_GROUP_ORDER = ['新馬', '未勝利', '1勝', '2勝', '3勝', 'OP/リステッド', '重賞']


def main():
    print("=" * 70)
    print("  条件別ROI格付けバックテスト")
    print(f"  Train: 2010-2022 → Test: 2023-2025")
    print(f"  Features: Pattern A ({len(FEATURES_PATTERN_A)} features)")
    print("=" * 70)

    # Load payout data
    print("\n[1] 配当データ読み込み...")
    payout_lookup = load_payouts()

    # Load and prepare data
    print("\n[2] データ準備...")
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')

    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    features = list(FEATURES_PATTERN_A)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"  Data ready: {len(df)} rows")

    # Train: 2010-2022, Test: 2023-2025
    train_mask = (df['year_full'] >= 2010) & (df['year_full'] <= 2022)
    test_mask = df['year_full'].isin([2023, 2024, 2025])

    print(f"\n[3] モデル学習 (train: {train_mask.sum():,} rows)...")

    # Encode sires using only train data
    df = encode_sires_fold(df, train_mask)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    train_df = df[train_mask]
    dates = train_df['date_num']
    valid_cutoff = dates.quantile(0.85)
    tr_idx = dates < valid_cutoff
    va_idx = dates >= valid_cutoff

    X_tr = train_df.loc[tr_idx, features].values
    y_tr = train_df.loc[tr_idx, 'target'].values
    X_va = train_df.loc[va_idx, features].values
    y_va = train_df.loc[va_idx, 'target'].values

    t0 = time.time()
    model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)
    elapsed = time.time() - t0
    va_auc = roc_auc_score(y_va, model.predict(X_va))
    print(f"  Valid AUC: {va_auc:.4f} ({elapsed:.0f}s)")

    # Predict on test
    test_df = df[test_mask].copy()
    test_df['pred'] = model.predict(test_df[features].values)
    test_auc = roc_auc_score(test_df['target'].values, test_df['pred'])
    print(f"  Test AUC (2023-2025): {test_auc:.4f}")

    # Per-year AUC
    year_aucs = {}
    for yr in [2023, 2024, 2025]:
        yr_mask = test_df['year_full'] == yr
        if yr_mask.sum() > 0:
            yr_auc = roc_auc_score(test_df.loc[yr_mask, 'target'], test_df.loc[yr_mask, 'pred'])
            year_aucs[yr] = round(yr_auc, 4)
            print(f"    {yr} AUC: {yr_auc:.4f}")

    # Evaluate each race
    print(f"\n[4] レース評価...")
    all_results = []
    test_races = test_df['race_id_str'].unique()

    for rid in test_races:
        race_df = test_df[test_df['race_id_str'] == rid].copy()
        if len(race_df) < 5:
            continue

        row0 = race_df.iloc[0]
        cond_key = classify_condition(row0)
        year = int(row0['year_full'])
        class_group = classify_class_group(row0.get('class_code', -1))

        # Actual top 3
        race_sorted = race_df.sort_values('finish')
        actual_top3 = {}
        for _, r in race_sorted.head(3).iterrows():
            actual_top3[int(r['finish'])] = int(r['umaban'])
        if len(actual_top3) < 3:
            continue

        # AI ranking
        race_df = race_df.sort_values('pred', ascending=False)
        ranking = race_df['umaban'].astype(int).tolist()

        # Generate bets
        trio_bets = calc_trio_bets(ranking)
        umaren_bets = calc_umaren_bets(ranking)
        wide_bets = calc_wide_bets(ranking)

        # Check actual payouts
        payout_info = payout_lookup.get(rid)
        if payout_info is None:
            continue

        # Trio
        actual_trio = parse_trio_nums(payout_info['trio_nums'])
        trio_hit = False
        trio_return = 0
        if actual_trio:
            for bet in trio_bets:
                if frozenset(bet) == actual_trio:
                    trio_hit = True
                    trio_return = payout_info['trio_payout']
                    break

        # Umaren
        actual_umaren = parse_umaren_nums(payout_info['umaren_nums'])
        umaren_hit = False
        umaren_return = 0
        if actual_umaren:
            for bet in umaren_bets:
                if frozenset(bet) == actual_umaren:
                    umaren_hit = True
                    umaren_return = payout_info['umaren_payout']
                    break

        # Wide (multiple can hit)
        actual_wide = parse_wide_data(payout_info['wide_nums'], payout_info['wide_payouts'])
        wide_hit_count = 0
        wide_return = 0
        for bet in wide_bets:
            bet_set = frozenset(bet)
            if bet_set in actual_wide:
                wide_hit_count += 1
                wide_return += actual_wide[bet_set]

        all_results.append({
            'race_id': rid,
            'year': year,
            'cond': cond_key,
            'class_group': class_group,
            'trio_hit': trio_hit,
            'trio_return': trio_return,
            'umaren_hit': umaren_hit,
            'umaren_return': umaren_return,
            'wide_hits': wide_hit_count,
            'wide_return': wide_return,
        })

    print(f"  評価完了: {len(all_results)} races")

    # === Analyze by condition ===
    print(f"\n[5] 条件別分析...")

    def analyze_condition(races):
        """Calculate ROI for 3 bet types."""
        n = len(races)
        if n == 0:
            return None

        result = {'n': n}
        # Trio: 7点 × 100円 = 700円
        trio_hits = sum(1 for r in races if r['trio_hit'])
        trio_inv = n * 700
        trio_pay = sum(r['trio_return'] for r in races)
        result['trio'] = {
            'hits': trio_hits,
            'hit_rate': round(trio_hits / n * 100, 1),
            'investment': trio_inv,
            'payout': trio_pay,
            'roi': round(trio_pay / trio_inv * 100, 1),
        }

        # Umaren: 2点 × 350円 = 700円 (配当は100円単位 → ×3.5)
        uma_hits = sum(1 for r in races if r['umaren_hit'])
        uma_inv = n * 700
        uma_pay = sum(r['umaren_return'] * 3.5 for r in races)
        result['umaren'] = {
            'hits': uma_hits,
            'hit_rate': round(uma_hits / n * 100, 1),
            'investment': uma_inv,
            'payout': int(uma_pay),
            'roi': round(uma_pay / uma_inv * 100, 1),
        }

        # Wide: 2点 × 350円 = 700円 (配当は100円単位 → ×3.5)
        wide_hits = sum(1 for r in races if r['wide_hits'] > 0)
        wide_inv = n * 700
        wide_pay = sum(r['wide_return'] * 3.5 for r in races)
        result['wide'] = {
            'hits': wide_hits,
            'hit_rate': round(wide_hits / n * 100, 1),
            'investment': wide_inv,
            'payout': int(wide_pay),
            'roi': round(wide_pay / wide_inv * 100, 1),
        }

        return result

    # Group by condition
    cond_groups = defaultdict(list)
    for r in all_results:
        cond_groups[r['cond']].append(r)

    # Group by condition × year
    cond_year_groups = defaultdict(list)
    for r in all_results:
        cond_year_groups[(r['cond'], r['year'])].append(r)

    # Condition analysis
    condition_ratings = {}
    for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = cond_groups.get(ck, [])
        if not races:
            continue
        analysis = analyze_condition(races)

        # Best bet type
        best_bt = max(['trio', 'umaren', 'wide'], key=lambda bt: analysis[bt]['roi'])
        best_roi = analysis[best_bt]['roi']

        # Rating
        if best_roi >= 120:
            stars = 3
            rating = '★★★'
            rating_label = '強く推奨'
        elif best_roi >= 100:
            stars = 2
            rating = '★★'
            rating_label = '推奨'
        elif best_roi >= 80:
            stars = 1
            rating = '★'
            rating_label = '様子見'
        else:
            stars = 0
            rating = '非推奨'
            rating_label = '非推奨'

        condition_ratings[ck] = {
            'n': analysis['n'],
            'best_bet': best_bt,
            'best_roi': best_roi,
            'stars': stars,
            'rating': rating,
            'rating_label': rating_label,
            'recommended': best_roi >= 100,
            'trio': analysis['trio'],
            'umaren': analysis['umaren'],
            'wide': analysis['wide'],
        }

    # Year stability
    year_stability = {}
    for yr in [2023, 2024, 2025]:
        year_stability[str(yr)] = {}
        for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            races = cond_year_groups.get((ck, yr), [])
            if len(races) < 5:
                continue
            analysis = analyze_condition(races)
            year_stability[str(yr)][ck] = analysis

    # Print results
    print(f"\n{'=' * 90}")
    print(f"  条件別ROI格付け (2023-2025, N={len(all_results)})")
    print(f"{'=' * 90}")
    print(f"  {'条件':<6} {'N':>5} | {'trio ROI':>9} {'trio的中':>7} | {'uma ROI':>9} {'uma的中':>7} | "
          f"{'wide ROI':>9} {'wide的中':>7} | {'Best':>5} {'格付'}")
    print(f"  {'-' * 85}")

    for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
        if ck not in condition_ratings:
            continue
        cr = condition_ratings[ck]
        t = cr['trio']
        u = cr['umaren']
        w = cr['wide']
        print(f"  {ck:<6} {cr['n']:>5} | {t['roi']:>8.1f}% {t['hit_rate']:>5.1f}% | "
              f"{u['roi']:>8.1f}% {u['hit_rate']:>5.1f}% | "
              f"{w['roi']:>8.1f}% {w['hit_rate']:>5.1f}% | "
              f"{cr['best_bet']:>5} {cr['rating']} ({cr['rating_label']})")

    # Year stability table
    print(f"\n  --- 年別安定性 ---")
    for yr in ['2023', '2024', '2025']:
        yd = year_stability.get(yr, {})
        for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            if ck not in yd:
                continue
            a = yd[ck]
            print(f"  {yr} {ck}: N={a['n']:>4} trio={a['trio']['roi']:>6.1f}% "
                  f"uma={a['umaren']['roi']:>6.1f}% wide={a['wide']['roi']:>6.1f}%")

    # === Class-based analysis ===
    print(f"\n[6] クラス別分析...")
    class_groups = defaultdict(list)
    for r in all_results:
        class_groups[r['class_group']].append(r)

    class_ratings = {}
    for cg in CLASS_GROUP_ORDER:
        races = class_groups.get(cg, [])
        if len(races) < 10:
            continue
        analysis = analyze_condition(races)

        best_bt = max(['trio', 'umaren', 'wide'], key=lambda bt: analysis[bt]['roi'])
        best_roi = analysis[best_bt]['roi']

        if best_roi >= 120:
            stars, rating, rating_label = 3, '★★★', '強く推奨'
        elif best_roi >= 100:
            stars, rating, rating_label = 2, '★★', '推奨'
        elif best_roi >= 80:
            stars, rating, rating_label = 1, '★', '様子見'
        else:
            stars, rating, rating_label = 0, '非推奨', '非推奨'

        class_ratings[cg] = {
            'n': analysis['n'],
            'best_bet': best_bt,
            'best_roi': best_roi,
            'stars': stars,
            'rating': rating,
            'rating_label': rating_label,
            'recommended': best_roi >= 100,
            'trio': analysis['trio'],
            'umaren': analysis['umaren'],
            'wide': analysis['wide'],
        }

    # Print class results
    print(f"\n{'=' * 90}")
    print(f"  クラス別ROI格付け (2023-2025)")
    print(f"{'=' * 90}")
    print(f"  {'クラス':<14} {'N':>5} | {'trio ROI':>9} {'trio的中':>7} | {'uma ROI':>9} {'uma的中':>7} | "
          f"{'wide ROI':>9} {'wide的中':>7} | {'Best':>5} {'格付'}")
    print(f"  {'-' * 85}")

    for cg in CLASS_GROUP_ORDER:
        if cg not in class_ratings:
            continue
        cr = class_ratings[cg]
        t = cr['trio']
        u = cr['umaren']
        w = cr['wide']
        print(f"  {cg:<14} {cr['n']:>5} | {t['roi']:>8.1f}% {t['hit_rate']:>5.1f}% | "
              f"{u['roi']:>8.1f}% {u['hit_rate']:>5.1f}% | "
              f"{w['roi']:>8.1f}% {w['hit_rate']:>5.1f}% | "
              f"{cr['best_bet']:>5} {cr['rating']} ({cr['rating_label']})")

    # Class × Condition cross analysis
    class_cond_groups = defaultdict(list)
    for r in all_results:
        class_cond_groups[(r['class_group'], r['cond'])].append(r)

    class_cond_ratings = {}
    for (cg, ck), races in class_cond_groups.items():
        if len(races) < 20:
            continue
        analysis = analyze_condition(races)
        best_bt = max(['trio', 'umaren', 'wide'], key=lambda bt: analysis[bt]['roi'])
        best_roi = analysis[best_bt]['roi']
        class_cond_ratings[f"{cg}_{ck}"] = {
            'class_group': cg, 'condition': ck,
            'n': analysis['n'], 'best_bet': best_bt, 'best_roi': best_roi,
            'trio': analysis['trio'], 'umaren': analysis['umaren'], 'wide': analysis['wide'],
        }

    print(f"\n  --- クラス×条件 クロス分析 (N>=20) ---")
    for cg in CLASS_GROUP_ORDER:
        for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            key = f"{cg}_{ck}"
            if key not in class_cond_ratings:
                continue
            ccr = class_cond_ratings[key]
            print(f"  {cg:<10} {ck}: N={ccr['n']:>4} trio={ccr['trio']['roi']:>6.1f}% "
                  f"uma={ccr['umaren']['roi']:>6.1f}% wide={ccr['wide']['roi']:>6.1f}% "
                  f"best={ccr['best_bet']}")

    # === Save condition JSON ===
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'v9.3_leakfree (Pattern A)',
        'n_features': len(FEATURES_PATTERN_A),
        'train_period': '2010-2022',
        'test_period': '2023-2025',
        'total_races': len(all_results),
        'test_auc': round(test_auc, 4),
        'year_aucs': year_aucs,
        'conditions': condition_ratings,
        'year_stability': year_stability,
    }

    out_path = os.path.join(BASE_DIR, 'data', 'condition_roi_ratings.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    # === Save class JSON ===
    class_output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'v9.3_leakfree (Pattern A)',
        'train_period': '2010-2022',
        'test_period': '2023-2025',
        'total_races': len(all_results),
        'classes': class_ratings,
        'class_condition_cross': class_cond_ratings,
    }

    class_path = os.path.join(BASE_DIR, 'data', 'class_roi_ratings.json')
    with open(class_path, 'w', encoding='utf-8') as f:
        json.dump(class_output, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {class_path}")

    # === Save markdown report ===
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    md_path = os.path.join(results_dir, 'BACKTEST_CONDITION_REPORT.md')

    md = []
    md.append("# 条件別・クラス別ROI格付けレポート")
    md.append(f"\n生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"\n## 概要")
    md.append(f"- モデル: Pattern A (リークフリー, {len(FEATURES_PATTERN_A)}特徴量)")
    md.append(f"- 学習期間: 2010-2022")
    md.append(f"- テスト期間: 2023-2025")
    md.append(f"- テストレース数: {len(all_results)}")
    md.append(f"- テストAUC: {test_auc:.4f}")
    for yr, auc in year_aucs.items():
        md.append(f"  - {yr}: {auc}")

    md.append(f"\n## 格付け基準")
    md.append("| 格付け | ROI | 判定 |")
    md.append("|--------|-----|------|")
    md.append("| ★★★ | 120%以上 | 強く推奨 |")
    md.append("| ★★ | 100-120% | 推奨 |")
    md.append("| ★ | 80-100% | 様子見 |")
    md.append("| 非推奨 | 80%未満 | 非推奨 |")

    md.append(f"\n## 条件別ROI (2023-2025)")
    md.append("| 条件 | N | 三連複ROI | 馬連ROI | ワイドROI | 最適券種 | 格付け |")
    md.append("|------|---|-----------|---------|-----------|----------|--------|")
    for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
        if ck not in condition_ratings:
            continue
        cr = condition_ratings[ck]
        desc = {'A': '8-14頭/1600m+/良稍', 'B': '8-14頭/1600m+/重不',
                'C': '15頭+/1600m+/良稍', 'D': '1200-1400m',
                'E': '7頭以下', 'X': '15頭+/重不'}.get(ck, '')
        md.append(f"| {ck} ({desc}) | {cr['n']} | {cr['trio']['roi']:.1f}% | "
                  f"{cr['umaren']['roi']:.1f}% | {cr['wide']['roi']:.1f}% | "
                  f"{cr['best_bet']} | {cr['rating']} ({cr['rating_label']}) |")

    md.append(f"\n## クラス別ROI (2023-2025)")
    md.append("| クラス | N | 三連複ROI | 三連複的中率 | 馬連ROI | ワイドROI | 最適券種 | 格付け |")
    md.append("|--------|---|-----------|-------------|---------|-----------|----------|--------|")
    for cg in CLASS_GROUP_ORDER:
        if cg not in class_ratings:
            continue
        cr = class_ratings[cg]
        md.append(f"| {cg} | {cr['n']} | {cr['trio']['roi']:.1f}% | {cr['trio']['hit_rate']:.1f}% | "
                  f"{cr['umaren']['roi']:.1f}% | {cr['wide']['roi']:.1f}% | "
                  f"{cr['best_bet']} | {cr['rating']} ({cr['rating_label']}) |")

    md.append(f"\n## クラス×条件 クロス分析")
    md.append("| クラス | 条件 | N | 三連複ROI | 馬連ROI | ワイドROI | 最適券種 |")
    md.append("|--------|------|---|-----------|---------|-----------|----------|")
    for cg in CLASS_GROUP_ORDER:
        for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            key = f"{cg}_{ck}"
            if key not in class_cond_ratings:
                continue
            ccr = class_cond_ratings[key]
            md.append(f"| {cg} | {ck} | {ccr['n']} | {ccr['trio']['roi']:.1f}% | "
                      f"{ccr['umaren']['roi']:.1f}% | {ccr['wide']['roi']:.1f}% | {ccr['best_bet']} |")

    md.append(f"\n## 年別安定性")
    for yr in ['2023', '2024', '2025']:
        md.append(f"\n### {yr}年")
        md.append("| 条件 | N | 三連複ROI | 馬連ROI | ワイドROI |")
        md.append("|------|---|-----------|---------|-----------|")
        yd = year_stability.get(yr, {})
        for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            if ck not in yd:
                continue
            a = yd[ck]
            md.append(f"| {ck} | {a['n']} | {a['trio']['roi']:.1f}% | "
                      f"{a['umaren']['roi']:.1f}% | {a['wide']['roi']:.1f}% |")

    md.append(f"\n## 結論")
    md.append(f"\n### 条件別")
    for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
        if ck not in condition_ratings:
            continue
        cr = condition_ratings[ck]
        md.append(f"- **条件{ck}**: {cr['rating']} {cr['best_bet']} ROI {cr['best_roi']:.1f}%")

    md.append(f"\n### クラス別")
    for cg in CLASS_GROUP_ORDER:
        if cg not in class_ratings:
            continue
        cr = class_ratings[cg]
        warning = ' **[注意]**' if cr['stars'] <= 1 else ''
        md.append(f"- **{cg}**: {cr['rating']} {cr['best_bet']} ROI {cr['best_roi']:.1f}%{warning}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"  Saved: {md_path}")

    print(f"\n  完了!")
    return output, class_output


if __name__ == '__main__':
    main()
