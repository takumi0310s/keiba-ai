#!/usr/bin/env python
"""詳細ROI分析（8テスト + サマリー）

jra_payouts.csv（JRA公式配当データ）×WFバックテストで以下を分析:
1. 月別・季節別ROI
2. 競馬場別ROI
3. クラス別ROI
4. 芝/ダート別ROI
5. 人気別ROI
6. 配当分布分析
7. 条件D細分化
8. ストレステスト
9. 全結果サマリー

出力: data/detailed_analysis.json
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A

from backtest_central_leakfree import (
    classify_condition, get_axes, calc_trio_bets, calc_umaren_bets,
    calc_wide_bets, check_bets, estimate_payouts, encode_sires_fold,
    train_lgb_fold, CLASS_LABELS,
)

# Import payout functions from calc_actual_roi
from calc_actual_roi import (
    load_payouts, parse_trio_nums, parse_umaren_nums, parse_wide_data,
    get_actual_payouts, calc_actual_returns,
)

COURSE_MAP_INV = {v: k for k, v in COURSE_MAP.items()}
TEST_YEARS = list(range(2020, 2026))


def run_backtest(df, features, payout_lookup):
    """WFバックテスト実行 + 配当マッチング。レース単位の拡張情報を返す。"""
    all_results = []
    fold_aucs = {}

    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year
        n_test = test_mask.sum()
        if n_test < 100:
            continue

        print(f"  Train 2010-{test_year-1} -> Test {test_year} ({n_test:,})")

        df_fold = df.copy()
        df_fold = encode_sires_fold(df_fold, train_mask)
        for f in features:
            if f not in df_fold.columns:
                df_fold[f] = 0
            df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

        train_df = df_fold[train_mask]
        y_train = train_df['target'].values
        dates = train_df['date_num']
        valid_cutoff = dates.quantile(0.85)
        tr_idx = dates < valid_cutoff
        va_idx = dates >= valid_cutoff

        X_tr = train_df.loc[tr_idx, features].values
        y_tr = y_train[tr_idx.values]
        X_va = train_df.loc[va_idx, features].values
        y_va = y_train[va_idx.values]

        t0 = time.time()
        model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)
        elapsed = time.time() - t0

        test_df_fold = df_fold[test_mask].copy()
        X_test = test_df_fold[features].values
        test_df_fold['pred'] = model.predict(X_test)
        test_auc = roc_auc_score(test_df_fold['target'].values, test_df_fold['pred'])
        fold_aucs[test_year] = test_auc
        print(f"    AUC: {test_auc:.4f} ({elapsed:.0f}s)")

        # Per-race evaluation
        test_races = test_df_fold['race_id_str'].unique()
        for rid in test_races:
            race_df = test_df_fold[test_df_fold['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            axes = get_axes(row0)

            # Extended metadata
            month = int(row0['month'])
            distance = int(row0['distance'])
            num_horses = int(row0.get('num_horses', row0.get('num_horses_val', 14)))
            surface_enc = int(row0['surface_enc'])
            cond_enc = int(row0.get('condition_enc', 0))
            class_code = int(row0.get('class_code', 0))

            # Actual top3
            race_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue

            # AI ranking
            race_df = race_df.sort_values('pred', ascending=False)
            ranking = race_df['umaban'].astype(int).tolist()

            # TOP1 horse popularity (tansho_odds rank)
            top1_umaban = ranking[0]
            odds_sorted = race_df.sort_values('tansho_odds')
            pop_rank = list(odds_sorted['umaban'].astype(int)).index(top1_umaban) + 1 if top1_umaban in odds_sorted['umaban'].astype(int).values else 8

            # Bets
            trio_bets = calc_trio_bets(ranking)
            umaren_bets = calc_umaren_bets(ranking)
            wide_bets = calc_wide_bets(ranking)

            # Estimated
            trio_hit, umaren_hits, wide_hits = check_bets(actual_top3, trio_bets, umaren_bets, wide_bets)
            trio_pay_est, umaren_pay_est, wide_pays_est = estimate_payouts(actual_top3, race_df)
            trio_return_est = trio_pay_est if trio_hit else 0

            # Actual from JRA payouts
            payout_info = get_actual_payouts(payout_lookup, rid)
            (actual_trio_hit, actual_trio_return,
             actual_umaren_hit, actual_umaren_return,
             actual_wide_hits, actual_wide_return) = calc_actual_returns(
                payout_info, trio_bets, umaren_bets, wide_bets)

            all_results.append({
                'race_id': rid, 'year': test_year,
                'month': month, 'distance': distance,
                'num_horses': num_horses, 'surface_enc': surface_enc,
                'condition_enc': cond_enc, 'class_code': class_code,
                'pop_rank_top1': pop_rank,
                'axes': axes,
                'has_payout': payout_info is not None,
                # Estimated
                'trio_hit': trio_hit, 'trio_return': trio_return_est,
                # Actual
                'actual_trio_hit': actual_trio_hit,
                'actual_trio_return': actual_trio_return,
                'actual_umaren_hit': actual_umaren_hit,
                'actual_umaren_return': actual_umaren_return,
                'actual_wide_hits': actual_wide_hits,
                'actual_wide_return': actual_wide_return,
            })

    return all_results, fold_aucs


def calc_roi(races, bet_type='trio'):
    """ROI計算ヘルパー"""
    n = len(races)
    if n == 0:
        return {'n': 0, 'hits': 0, 'hit_rate': 0, 'roi': 0, 'investment': 0, 'payout': 0}
    if bet_type == 'trio':
        hits = sum(1 for r in races if r.get('actual_trio_hit', False))
        payout = sum(r.get('actual_trio_return', 0) for r in races)
    elif bet_type == 'umaren':
        hits = sum(1 for r in races if r.get('actual_umaren_hit', False))
        payout = sum(r.get('actual_umaren_return', 0) * 3.5 for r in races)
    else:
        return {'n': n, 'hits': 0, 'hit_rate': 0, 'roi': 0, 'investment': 0, 'payout': 0}
    investment = n * 700
    roi = payout / investment * 100 if investment > 0 else 0
    hit_rate = hits / n * 100
    return {
        'n': n, 'hits': hits, 'hit_rate': round(hit_rate, 1),
        'roi': round(roi, 1), 'investment': investment, 'payout': int(payout),
    }


def get_condition_bet_type(cond_key):
    """条件に応じた最適券種"""
    return 'umaren' if cond_key == 'E' else 'trio'


def calc_condition_roi(races):
    """条件に応じた正しい券種でROI計算"""
    cond_key = races[0]['axes']['cond_key'] if races else 'A'
    bt = get_condition_bet_type(cond_key)
    return calc_roi(races, bt)


# ============================================================
# テスト1: 月別・季節別ROI
# ============================================================
def test1_monthly_seasonal(matched):
    print("\n" + "=" * 70)
    print("  [1] 月別・季節別ROI")
    print("=" * 70)

    MONTH_NAMES = {1:'1月',2:'2月',3:'3月',4:'4月',5:'5月',6:'6月',
                   7:'7月',8:'8月',9:'9月',10:'10月',11:'11月',12:'12月'}
    SEASONS = {'春(3-5月)':[3,4,5],'夏(6-8月)':[6,7,8],'秋(9-11月)':[9,10,11],'冬(12-2月)':[12,1,2]}

    # 月別
    monthly = {}
    print(f"\n  {'月':>6} {'N':>6} {'的中':>5} {'的中率':>7} {'ROI':>8} {'状態':>6}")
    print(f"  {'-'*45}")
    for m in range(1, 13):
        races = [r for r in matched if r['month'] == m]
        if not races:
            continue
        roi_info = calc_roi(races, 'trio')
        status = 'OK' if roi_info['roi'] >= 100 else 'WARN'
        print(f"  {MONTH_NAMES[m]:>6} {roi_info['n']:>6} {roi_info['hits']:>5} "
              f"{roi_info['hit_rate']:>6.1f}% {roi_info['roi']:>7.1f}% {status:>6}")
        monthly[MONTH_NAMES[m]] = roi_info

    # 季節別
    seasonal = {}
    print(f"\n  {'季節':>12} {'N':>6} {'的中':>5} {'的中率':>7} {'ROI':>8} {'状態':>6}")
    print(f"  {'-'*50}")
    for season_name, months in SEASONS.items():
        races = [r for r in matched if r['month'] in months]
        if not races:
            continue
        roi_info = calc_roi(races, 'trio')
        status = 'OK' if roi_info['roi'] >= 100 else 'WARN'
        print(f"  {season_name:>12} {roi_info['n']:>6} {roi_info['hits']:>5} "
              f"{roi_info['hit_rate']:>6.1f}% {roi_info['roi']:>7.1f}% {status:>6}")
        seasonal[season_name] = roi_info

    below_100 = [m for m, info in monthly.items() if info['roi'] < 100]
    if below_100:
        print(f"\n  [WARN] ROI 100%未満の月: {', '.join(below_100)}")
    else:
        print(f"\n  [OK] 全月ROI 100%以上")

    return {'monthly': monthly, 'seasonal': seasonal}


# ============================================================
# テスト2: 競馬場別ROI
# ============================================================
def test2_course(matched):
    print("\n" + "=" * 70)
    print("  [2] 競馬場別ROI")
    print("=" * 70)

    COURSE_NAMES = ['東京','中山','阪神','京都','小倉','新潟','福島','札幌','函館','中京']
    course_results = {}

    print(f"\n  {'競馬場':>6} {'N':>6} {'的中':>5} {'的中率':>7} {'ROI':>8} {'状態':>6}")
    print(f"  {'-'*45}")

    for cname in COURSE_NAMES:
        races = [r for r in matched if r['axes'].get('course') == cname]
        if not races:
            continue
        roi_info = calc_roi(races, 'trio')
        status = 'OK' if roi_info['roi'] >= 100 else 'WARN'
        print(f"  {cname:>6} {roi_info['n']:>6} {roi_info['hits']:>5} "
              f"{roi_info['hit_rate']:>6.1f}% {roi_info['roi']:>7.1f}% {status:>6}")
        course_results[cname] = roi_info

    if course_results:
        best = max(course_results, key=lambda c: course_results[c]['roi'])
        worst = min(course_results, key=lambda c: course_results[c]['roi'])
        print(f"\n  最高ROI: {best} ({course_results[best]['roi']:.1f}%)")
        print(f"  最低ROI: {worst} ({course_results[worst]['roi']:.1f}%)")

    return course_results


# ============================================================
# テスト3: クラス別ROI
# ============================================================
def test3_class(matched):
    print("\n" + "=" * 70)
    print("  [3] クラス別ROI")
    print("=" * 70)

    class_results = {}
    print(f"\n  {'クラス':>10} {'N':>6} {'的中':>5} {'的中率':>7} {'ROI':>8} {'状態':>6}")
    print(f"  {'-'*50}")

    # Sort by class order
    class_order = ['新馬','未勝利','1勝','2勝','3勝','OP','OP特別','リステッド','G3','G2','G1']
    all_classes = set(r['axes'].get('class', '') for r in matched)
    ordered_classes = [c for c in class_order if c in all_classes]
    remaining = sorted(all_classes - set(ordered_classes))
    ordered_classes.extend(remaining)

    for cls_name in ordered_classes:
        if not cls_name:
            continue
        races = [r for r in matched if r['axes'].get('class') == cls_name]
        if len(races) < 20:
            continue
        roi_info = calc_roi(races, 'trio')
        status = 'OK' if roi_info['roi'] >= 100 else 'WARN'
        print(f"  {cls_name:>10} {roi_info['n']:>6} {roi_info['hits']:>5} "
              f"{roi_info['hit_rate']:>6.1f}% {roi_info['roi']:>7.1f}% {status:>6}")
        class_results[cls_name] = roi_info

    return class_results


# ============================================================
# テスト4: 芝/ダート別ROI
# ============================================================
def test4_surface(matched):
    print("\n" + "=" * 70)
    print("  [4] 芝/ダート別ROI")
    print("=" * 70)

    SURF_MAP = {0: '芝', 1: 'ダート', 2: '障害'}
    surface_results = {}

    print(f"\n  {'馬場':>6} {'N':>6} {'的中':>5} {'的中率':>7} {'ROI':>8}")
    print(f"  {'-'*40}")
    for s_enc, s_name in SURF_MAP.items():
        races = [r for r in matched if r['surface_enc'] == s_enc]
        if not races:
            continue
        roi_info = calc_roi(races, 'trio')
        print(f"  {s_name:>6} {roi_info['n']:>6} {roi_info['hits']:>5} "
              f"{roi_info['hit_rate']:>6.1f}% {roi_info['roi']:>7.1f}%")
        surface_results[s_name] = roi_info

    # 条件別×芝ダート クロス集計
    cross_results = {}
    print(f"\n  条件別 x 芝/ダート:")
    print(f"  {'条件':>4} {'馬場':>4} {'N':>6} {'的中率':>7} {'ROI':>8}")
    print(f"  {'-'*40}")
    for cond in ['A','B','C','D','E','X']:
        for s_enc, s_name in [(0,'芝'),(1,'ダ')]:
            races = [r for r in matched
                     if r['axes']['cond_key'] == cond and r['surface_enc'] == s_enc]
            if len(races) < 20:
                continue
            bt = get_condition_bet_type(cond)
            roi_info = calc_roi(races, bt)
            key = f"{cond}_{s_name}"
            cross_results[key] = roi_info
            print(f"  {cond:>4} {s_name:>4} {roi_info['n']:>6} "
                  f"{roi_info['hit_rate']:>6.1f}% {roi_info['roi']:>7.1f}%")

    return {'surface': surface_results, 'cross': cross_results}


# ============================================================
# テスト5: 人気別ROI
# ============================================================
def test5_popularity(matched):
    print("\n" + "=" * 70)
    print("  [5] AI予測1位馬の人気別ROI")
    print("=" * 70)

    POP_GROUPS = {
        '1番人気': [1],
        '2-3番人気': [2, 3],
        '4-6番人気': [4, 5, 6],
        '7番人気以下': list(range(7, 30)),
    }

    pop_results = {}
    print(f"\n  {'人気':>12} {'N':>6} {'的中':>5} {'的中率':>7} {'ROI':>8}")
    print(f"  {'-'*45}")

    for label, pops in POP_GROUPS.items():
        races = [r for r in matched if r.get('pop_rank_top1', 99) in pops]
        if not races:
            continue
        roi_info = calc_roi(races, 'trio')
        print(f"  {label:>12} {roi_info['n']:>6} {roi_info['hits']:>5} "
              f"{roi_info['hit_rate']:>6.1f}% {roi_info['roi']:>7.1f}%")
        pop_results[label] = roi_info

    if pop_results:
        best = max(pop_results, key=lambda p: pop_results[p]['roi'])
        print(f"\n  最高ROI: {best} ({pop_results[best]['roi']:.1f}%)")

    return pop_results


# ============================================================
# テスト6: 配当分布分析
# ============================================================
def test6_payout_distribution(matched):
    print("\n" + "=" * 70)
    print("  [6] 配当分布分析")
    print("=" * 70)

    # 全体の三連複配当分布
    trio_payouts = [r['actual_trio_return'] for r in matched if r.get('actual_trio_hit', False)]
    n_hits = len(trio_payouts)
    n_total = len(matched)

    overall = {}
    if trio_payouts:
        payouts_arr = np.array(trio_payouts)
        overall = {
            'n_hits': n_hits,
            'n_total': n_total,
            'hit_rate': round(n_hits / n_total * 100, 1),
            'median': int(np.median(payouts_arr)),
            'mean': int(np.mean(payouts_arr)),
            'min': int(np.min(payouts_arr)),
            'max': int(np.max(payouts_arr)),
            'std': int(np.std(payouts_arr)),
            'p25': int(np.percentile(payouts_arr, 25)),
            'p75': int(np.percentile(payouts_arr, 75)),
        }
        man_baken = sum(1 for p in trio_payouts if p >= 10000)
        overall['man_baken_count'] = man_baken
        overall['man_baken_rate'] = round(man_baken / n_hits * 100, 1)

        print(f"\n  三連複的中配当（100円あたり）:")
        print(f"    的中数: {n_hits} / {n_total} ({overall['hit_rate']:.1f}%)")
        print(f"    中央値: {overall['median']:,}円")
        print(f"    平均値: {overall['mean']:,}円")
        print(f"    最小:   {overall['min']:,}円")
        print(f"    最大:   {overall['max']:,}円")
        print(f"    25%ile: {overall['p25']:,}円")
        print(f"    75%ile: {overall['p75']:,}円")
        print(f"    万馬券: {man_baken}回 ({overall['man_baken_rate']:.1f}%)")

    # 条件別配当分布
    cond_dist = {}
    print(f"\n  条件別 配当分布:")
    print(f"  {'条件':>4} {'的中':>5} {'中央値':>8} {'平均':>8} {'万馬券率':>8} {'ROI依存度':>10}")
    print(f"  {'-'*55}")

    for cond in ['A','B','C','D','E','X']:
        if cond == 'E':
            payouts_c = [r['actual_umaren_return'] for r in matched
                        if r['axes']['cond_key'] == cond and r.get('actual_umaren_hit', False)]
        else:
            payouts_c = [r['actual_trio_return'] for r in matched
                        if r['axes']['cond_key'] == cond and r.get('actual_trio_hit', False)]
        n_cond = sum(1 for r in matched if r['axes']['cond_key'] == cond)
        if not payouts_c or n_cond == 0:
            continue

        arr = np.array(payouts_c)
        med = int(np.median(arr))
        mean = int(np.mean(arr))
        man_count = sum(1 for p in payouts_c if p >= 10000)
        man_rate = man_count / len(payouts_c) * 100

        # ROI依存度: top10%の配当がROI全体の何%を占めるか
        sorted_pays = sorted(payouts_c, reverse=True)
        top10_n = max(1, len(sorted_pays) // 10)
        top10_sum = sum(sorted_pays[:top10_n])
        total_sum = sum(sorted_pays)
        dependency = top10_sum / total_sum * 100 if total_sum > 0 else 0

        cond_dist[cond] = {
            'n_hits': len(payouts_c), 'n_total': n_cond,
            'median': med, 'mean': mean,
            'man_baken_rate': round(man_rate, 1),
            'top10_dependency': round(dependency, 1),
        }
        print(f"  {cond:>4} {len(payouts_c):>5} {med:>7,}円 {mean:>7,}円 "
              f"{man_rate:>6.1f}% {dependency:>9.1f}%")

    # 少数高配当への依存チェック
    if trio_payouts:
        sorted_all = sorted(trio_payouts, reverse=True)
        top5_sum = sum(sorted_all[:5])
        all_sum = sum(sorted_all)
        dep5 = top5_sum / all_sum * 100 if all_sum > 0 else 0
        top10_sum = sum(sorted_all[:10])
        dep10 = top10_sum / all_sum * 100 if all_sum > 0 else 0
        top50_sum = sum(sorted_all[:50])
        dep50 = top50_sum / all_sum * 100 if all_sum > 0 else 0
        print(f"\n  高配当依存度:")
        print(f"    TOP5的中がROIに占める割合:  {dep5:.1f}%")
        print(f"    TOP10的中がROIに占める割合: {dep10:.1f}%")
        print(f"    TOP50的中がROIに占める割合: {dep50:.1f}%")
        if dep5 > 30:
            print(f"    [WARN] TOP5依存度が高い ({dep5:.1f}%)")
        else:
            print(f"    [OK] 少数高配当に過度に依存していない")
        overall['top5_dependency'] = round(dep5, 1)
        overall['top10_dependency'] = round(dep10, 1)
        overall['top50_dependency'] = round(dep50, 1)

    return {'overall': overall, 'by_condition': cond_dist}


# ============================================================
# テスト7: 条件D細分化
# ============================================================
def test7_condition_d(matched):
    print("\n" + "=" * 70)
    print("  [7] 条件D細分化分析")
    print("=" * 70)

    d_races = [r for r in matched if r['axes']['cond_key'] == 'D']
    print(f"\n  条件D全体: {len(d_races)}レース")

    results = {}

    # 距離別
    dist_groups = {
        '1000m以下': [r for r in d_races if r['distance'] <= 1000],
        '1200m': [r for r in d_races if r['distance'] == 1200],
        '1400m': [r for r in d_races if r['distance'] == 1400],
    }
    print(f"\n  --- 距離別 ---")
    print(f"  {'距離':>10} {'N':>6} {'的中率':>7} {'ROI':>8}")
    print(f"  {'-'*35}")
    dist_results = {}
    for label, races in dist_groups.items():
        if not races:
            continue
        roi = calc_roi(races, 'trio')
        print(f"  {label:>10} {roi['n']:>6} {roi['hit_rate']:>6.1f}% {roi['roi']:>7.1f}%")
        dist_results[label] = roi
    results['distance'] = dist_results

    # 頭数別
    nh_groups = {
        '8-14頭': [r for r in d_races if 8 <= r['num_horses'] <= 14],
        '15頭以上': [r for r in d_races if r['num_horses'] >= 15],
    }
    print(f"\n  --- 頭数別 ---")
    print(f"  {'頭数':>10} {'N':>6} {'的中率':>7} {'ROI':>8}")
    print(f"  {'-'*35}")
    nh_results = {}
    for label, races in nh_groups.items():
        if not races:
            continue
        roi = calc_roi(races, 'trio')
        print(f"  {label:>10} {roi['n']:>6} {roi['hit_rate']:>6.1f}% {roi['roi']:>7.1f}%")
        nh_results[label] = roi
    results['num_horses'] = nh_results

    # 芝/ダート別
    surf_groups = {
        '芝': [r for r in d_races if r['surface_enc'] == 0],
        'ダート': [r for r in d_races if r['surface_enc'] == 1],
    }
    print(f"\n  --- 芝/ダート別 ---")
    print(f"  {'馬場':>10} {'N':>6} {'的中率':>7} {'ROI':>8}")
    print(f"  {'-'*35}")
    surf_results = {}
    for label, races in surf_groups.items():
        if not races:
            continue
        roi = calc_roi(races, 'trio')
        print(f"  {label:>10} {roi['n']:>6} {roi['hit_rate']:>6.1f}% {roi['roi']:>7.1f}%")
        surf_results[label] = roi
    results['surface'] = surf_results

    # サブ条件特定
    above_100 = []
    below_100 = []
    for group_name, group_results in [('距離', dist_results), ('頭数', nh_results), ('馬場', surf_results)]:
        for label, info in group_results.items():
            tag = f"{group_name}:{label}"
            if info['roi'] >= 100:
                above_100.append((tag, info['roi']))
            else:
                below_100.append((tag, info['roi']))

    print(f"\n  ROI 100%以上のサブ条件:")
    for tag, roi in sorted(above_100, key=lambda x: -x[1]):
        print(f"    {tag}: {roi:.1f}%")
    if below_100:
        print(f"  ROI 100%未満のサブ条件:")
        for tag, roi in sorted(below_100, key=lambda x: x[1]):
            print(f"    {tag}: {roi:.1f}%")
    results['above_100'] = [{'sub': t, 'roi': r} for t, r in above_100]
    results['below_100'] = [{'sub': t, 'roi': r} for t, r in below_100]

    return results


# ============================================================
# テスト8: ストレステスト
# ============================================================
def test8_stress(matched):
    print("\n" + "=" * 70)
    print("  [8] ストレステスト")
    print("=" * 70)

    # 現行ROI(ベースライン)
    base_roi = calc_roi(matched, 'trio')
    print(f"\n  ベースライン: N={base_roi['n']}, 的中率={base_roi['hit_rate']:.1f}%, ROI={base_roi['roi']:.1f}%")

    results = {}
    results['baseline'] = base_roi

    trio_payouts = [r.get('actual_trio_return', 0) for r in matched]
    trio_hits = [r.get('actual_trio_hit', False) for r in matched]
    n = len(matched)
    investment = n * 700
    total_payout = sum(trio_payouts)

    # テスト1: 控除率25%→30% (配当が約(100-30)/(100-25) = 93.3%に減少)
    deduction_factor = (100 - 30) / (100 - 25)  # 0.9333
    stress1_payout = total_payout * deduction_factor
    stress1_roi = stress1_payout / investment * 100
    results['deduction_30pct'] = {
        'description': '控除率25%->30%',
        'factor': round(deduction_factor, 4),
        'roi': round(stress1_roi, 1),
        'pass': stress1_roi >= 100,
    }
    print(f"\n  [1] 控除率30%: ROI {stress1_roi:.1f}% {'PASS' if stress1_roi >= 100 else 'FAIL'}")

    # テスト2: 的中率を実績の80%に (ランダムに20%の的中を外す)
    np.random.seed(42)
    hit_indices = [i for i, h in enumerate(trio_hits) if h]
    n_remove = int(len(hit_indices) * 0.2)
    remove_set = set(np.random.choice(hit_indices, n_remove, replace=False))
    stress2_payout = sum(p for i, p in enumerate(trio_payouts) if i not in remove_set)
    stress2_roi = stress2_payout / investment * 100
    results['hit_rate_80pct'] = {
        'description': '的中率を80%に低下',
        'removed_hits': n_remove,
        'roi': round(stress2_roi, 1),
        'pass': stress2_roi >= 100,
    }
    print(f"  [2] 的中率80%: ROI {stress2_roi:.1f}% {'PASS' if stress2_roi >= 100 else 'FAIL'} "
          f"({n_remove}的中を除外)")

    # テスト3: 両方同時
    stress3_payout = stress2_payout * deduction_factor
    stress3_roi = stress3_payout / investment * 100
    results['both'] = {
        'description': '控除率30% + 的中率80%',
        'roi': round(stress3_roi, 1),
        'pass': stress3_roi >= 100,
    }
    print(f"  [3] 両方同時:  ROI {stress3_roi:.1f}% {'PASS' if stress3_roi >= 100 else 'FAIL'}")

    # 条件別ストレステスト
    print(f"\n  --- 条件別ストレステスト ---")
    print(f"  {'条件':>4} {'基準ROI':>8} {'控除30%':>8} {'的中80%':>8} {'両方':>8}")
    print(f"  {'-'*45}")
    cond_stress = {}
    for cond in ['A','B','C','D','E','X']:
        bt = get_condition_bet_type(cond)
        cond_races = [r for r in matched if r['axes']['cond_key'] == cond]
        if not cond_races:
            continue
        nc = len(cond_races)
        inv_c = nc * 700

        if bt == 'trio':
            pays_c = [r.get('actual_trio_return', 0) for r in cond_races]
            hits_c = [r.get('actual_trio_hit', False) for r in cond_races]
        else:
            pays_c = [r.get('actual_umaren_return', 0) * 3.5 for r in cond_races]
            hits_c = [r.get('actual_umaren_hit', False) for r in cond_races]

        base = sum(pays_c) / inv_c * 100
        s1 = sum(pays_c) * deduction_factor / inv_c * 100

        hit_idx_c = [i for i, h in enumerate(hits_c) if h]
        n_rm_c = int(len(hit_idx_c) * 0.2)
        if n_rm_c > 0:
            rm_set_c = set(np.random.choice(hit_idx_c, n_rm_c, replace=False))
        else:
            rm_set_c = set()
        s2_pay = sum(p for i, p in enumerate(pays_c) if i not in rm_set_c)
        s2 = s2_pay / inv_c * 100
        s3 = s2_pay * deduction_factor / inv_c * 100

        cond_stress[cond] = {
            'baseline': round(base, 1), 'deduction_30': round(s1, 1),
            'hit_80': round(s2, 1), 'both': round(s3, 1),
        }
        print(f"  {cond:>4} {base:>7.1f}% {s1:>7.1f}% {s2:>7.1f}% {s3:>7.1f}%")
    results['by_condition'] = cond_stress

    return results


# ============================================================
# テスト9: サマリー
# ============================================================
def test9_summary(all_tests, matched):
    print("\n" + "=" * 70)
    print("  [9] 全結果サマリー")
    print("=" * 70)

    warnings_list = []
    suggestions = []

    # 月別チェック
    monthly = all_tests.get('test1', {}).get('monthly', {})
    for m, info in monthly.items():
        if info['roi'] < 100:
            warnings_list.append(f"月別: {m} ROI {info['roi']:.1f}% (100%未満)")

    # 競馬場別チェック
    course = all_tests.get('test2', {})
    for c, info in course.items():
        if info['roi'] < 100:
            warnings_list.append(f"競馬場: {c} ROI {info['roi']:.1f}% (100%未満)")

    # クラス別チェック
    cls = all_tests.get('test3', {})
    for c, info in cls.items():
        if info['roi'] < 100:
            warnings_list.append(f"クラス: {c} ROI {info['roi']:.1f}% (100%未満)")

    # 芝ダートチェック
    surf = all_tests.get('test4', {}).get('surface', {})
    for s, info in surf.items():
        if info['roi'] < 100:
            warnings_list.append(f"馬場: {s} ROI {info['roi']:.1f}% (100%未満)")

    # 配当依存チェック
    dist_info = all_tests.get('test6', {}).get('overall', {})
    if dist_info.get('top5_dependency', 0) > 30:
        warnings_list.append(f"配当依存: TOP5的中が全体の{dist_info['top5_dependency']:.1f}%を占める")

    # 条件Dチェック
    d_info = all_tests.get('test7', {})
    for item in d_info.get('below_100', []):
        warnings_list.append(f"条件D: {item['sub']} ROI {item['roi']:.1f}%")

    # ストレステスト
    stress = all_tests.get('test8', {})
    if not stress.get('deduction_30pct', {}).get('pass', True):
        warnings_list.append(f"ストレス: 控除率30%でROI 100%未満 ({stress['deduction_30pct']['roi']:.1f}%)")
    if not stress.get('hit_rate_80pct', {}).get('pass', True):
        warnings_list.append(f"ストレス: 的中率80%でROI 100%未満 ({stress['hit_rate_80pct']['roi']:.1f}%)")
    if not stress.get('both', {}).get('pass', True):
        warnings_list.append(f"ストレス: 両方悪化でROI 100%未満 ({stress['both']['roi']:.1f}%)")

    # 改善提案
    if monthly:
        low_months = [(m, info['roi']) for m, info in monthly.items() if info['roi'] < 120]
        if low_months:
            suggestions.append(f"低ROI月({', '.join(m for m,_ in low_months)})のレース選別基準を厳格化")

    if course:
        low_courses = [(c, info['roi']) for c, info in course.items() if info['roi'] < 120]
        if low_courses:
            suggestions.append(f"低ROI場({', '.join(c for c,_ in low_courses)})でのベット削減を検討")

    d_below = d_info.get('below_100', [])
    if d_below:
        suggestions.append("条件D内でROI 100%未満のサブ条件を除外フィルタに追加検討")

    if dist_info.get('top5_dependency', 0) > 30:
        suggestions.append("少数の高配当に依存 - 安定的中重視の買い目に調整検討")

    # 表示
    print(f"\n  テスト完了: 8項目")
    print(f"\n  === 警告一覧 ({len(warnings_list)}件) ===")
    if warnings_list:
        for w in warnings_list:
            print(f"    [WARN] {w}")
    else:
        print(f"    警告なし")

    print(f"\n  === 改善提案 ({len(suggestions)}件) ===")
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s}")
    else:
        print(f"    提案なし")

    # 全条件ROI一覧
    print(f"\n  === 条件別ROI一覧 ===")
    for cond in ['A','B','C','D','E','X']:
        bt = get_condition_bet_type(cond)
        cond_races = [r for r in matched if r['axes']['cond_key'] == cond]
        if not cond_races:
            continue
        roi = calc_roi(cond_races, bt)
        status = 'OK' if roi['roi'] >= 100 else 'WARN'
        print(f"    {cond}: {bt} ROI {roi['roi']:>7.1f}% (N={roi['n']}, 的中{roi['hit_rate']:.1f}%) [{status}]")

    verdict = 'ALL_PASS' if not warnings_list else f'WARNINGS({len(warnings_list)})'
    print(f"\n  最終判定: {verdict}")

    return {
        'verdict': verdict,
        'warnings': warnings_list,
        'suggestions': suggestions,
        'n_warnings': len(warnings_list),
    }


# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 70)
    print("  詳細ROI分析 (JRA公式配当 x WFバックテスト)")
    print("  8テスト + サマリー")
    print("=" * 70)

    # Load data
    print("\n[0] データ準備...")
    payout_lookup = load_payouts()

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

    print(f"  Data: {len(df):,} rows, {df['race_id_str'].nunique():,} races")

    # WF Backtest
    print("\n[BT] ウォークフォワードバックテスト...")
    all_results, fold_aucs = run_backtest(df, features, payout_lookup)
    matched = [r for r in all_results if r['has_payout']]
    print(f"\n  結果: {len(all_results)} races total, {len(matched)} matched with payouts")
    print(f"  AUC: {np.mean(list(fold_aucs.values())):.4f}")

    # Run all tests
    all_tests = {}
    all_tests['test1'] = test1_monthly_seasonal(matched)
    all_tests['test2'] = test2_course(matched)
    all_tests['test3'] = test3_class(matched)
    all_tests['test4'] = test4_surface(matched)
    all_tests['test5'] = test5_popularity(matched)
    all_tests['test6'] = test6_payout_distribution(matched)
    all_tests['test7'] = test7_condition_d(matched)
    all_tests['test8'] = test8_stress(matched)
    all_tests['summary'] = test9_summary(all_tests, matched)

    # Save
    out_path = os.path.join(BASE_DIR, 'data', 'detailed_analysis.json')
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'v9.3_leakfree (Pattern A)',
        'n_features': len(FEATURES_PATTERN_A),
        'test_years': TEST_YEARS,
        'total_races': len(all_results),
        'matched_races': len(matched),
        'fold_aucs': {str(k): round(v, 4) for k, v in fold_aucs.items()},
        'avg_auc': round(np.mean(list(fold_aucs.values())), 4),
        'tests': all_tests,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存: {out_path}")
    print(f"\n  完了!")


if __name__ == '__main__':
    main()
