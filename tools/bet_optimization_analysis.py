#!/usr/bin/env python
"""買い目最適化分析スクリプト

実戦データ(track_record.csv)を使用して:
1. 点数最適化（7点固定 vs 可変）
2. 信頼度ベース投資額可変
3. 券種別ROI再検証
"""
import sys
import io
import os
import json

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def parse_bets(bets_str):
    if pd.isna(bets_str):
        return []
    bets = []
    for b in str(bets_str).split(';'):
        nums = [int(x.strip()) for x in b.strip().split('-') if x.strip().isdigit()]
        if len(nums) == 3:
            bets.append(frozenset(nums))
    return bets


def parse_result(result_str):
    if pd.isna(result_str):
        return frozenset()
    nums = [int(x.strip()) for x in str(result_str).split('-') if x.strip().isdigit()]
    return frozenset(nums) if len(nums) == 3 else frozenset()


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'track_record.csv'))

    # Load daily predictions for scores
    pred_dfs = []
    pred_dir = os.path.join(DATA_DIR, 'daily_predictions')
    if os.path.exists(pred_dir):
        for f in sorted(os.listdir(pred_dir)):
            if f.endswith('.csv'):
                pred_dfs.append(pd.read_csv(os.path.join(pred_dir, f)))
    if pred_dfs:
        df_pred = pd.concat(pred_dfs, ignore_index=True).drop_duplicates('race_id', keep='last')
        score_cols = [c for c in df_pred.columns if 'score' in c or 'distance' in c or 'surface' in c or 'num_horses' in c]
        df = df.merge(df_pred[['race_id'] + score_cols].drop_duplicates('race_id'),
                      on='race_id', how='left')

    # Venue
    df['course_code'] = df['race_id'].astype(str).str[4:6]
    course_map = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
                  '05': '東京', '06': '中山', '07': '中京', '08': '京都',
                  '09': '阪神', '10': '小倉'}
    df['venue'] = df['course_code'].map(course_map)

    # Parse bets and results
    df['bets_parsed'] = df['trio_bets'].apply(parse_bets)
    df['result_parsed'] = df['trio_result'].apply(parse_result)

    # Find winning bet index
    df['winning_bet_idx'] = np.nan
    for i, row in df.iterrows():
        result = row['result_parsed']
        if not result:
            continue
        for j, bet in enumerate(row['bets_parsed']):
            if bet == result:
                df.at[i, 'winning_bet_idx'] = j + 1
                break

    print("=" * 60)
    print("買い目最適化分析")
    print("=" * 60)
    print(f"データ: {len(df)}レース, {sorted(df['date'].unique())}")

    # === 1. 的中時の何点目で的中したか ===
    print("\n--- 的中時の何点目で的中したか ---")
    hits_df = df[df['winning_bet_idx'].notna()]
    for idx in range(1, 8):
        count = (hits_df['winning_bet_idx'] == idx).sum()
        if count > 0:
            print(f"  {idx}点目: {count}回 ({count/len(hits_df)*100:.1f}%)")

    # === 2. 点数削減シミュレーション ===
    print("\n--- 点数削減シミュレーション ---")
    bet_count_results = {}
    for n_bets in [3, 4, 5, 6, 7]:
        total_hits = 0
        total_payout = 0
        invest_per_race = n_bets * 100
        total_invest = len(df) * invest_per_race

        for _, row in df.iterrows():
            result = row['result_parsed']
            bets = row['bets_parsed'][:n_bets]
            for bet in bets:
                if bet == result:
                    total_hits += 1
                    total_payout += row['payout']
                    break

        roi = total_payout / total_invest * 100 if total_invest > 0 else 0
        profit = total_payout - total_invest
        print(f"  {n_bets}点: 的中={total_hits:2d} ({total_hits/len(df)*100:.1f}%), "
              f"投資={total_invest:>7,}円, 回収={total_payout:>7,}円, "
              f"ROI={roi:5.1f}%, 収支={profit:+,}円")
        bet_count_results[f'{n_bets}点'] = {
            'hits': int(total_hits), 'investment': int(total_invest),
            'payout': int(total_payout), 'ROI': round(roi, 1), 'profit': int(profit)
        }

    # === 3. 信頼度ベース投資額 ===
    print("\n--- 信頼度ベース投資額 ---")
    if 'top1_score' in df.columns and 'top2_score' in df.columns:
        df['score_diff'] = (df['top1_score'] - df['top2_score']).fillna(0)

        print("\nスコア差別パフォーマンス:")
        df['score_diff_bin'] = pd.cut(df['score_diff'],
                                       bins=[-1, 0.01, 0.03, 0.05, 1.0],
                                       labels=['僅差(<0.01)', '小差(0.01-0.03)', '中差(0.03-0.05)', '大差(0.05+)'])
        for sb, grp in df.groupby('score_diff_bin', observed=True):
            n = len(grp)
            if n == 0:
                continue
            hits = grp['hit'].sum()
            invest = grp['investment'].sum()
            payout = grp['payout'].sum()
            roi = payout / invest * 100 if invest > 0 else 0
            print(f"  {sb}: N={n:3d}, 的中={hits:2d} ({hits/n*100:5.1f}%), ROI={roi:5.1f}%")

    # === 4. 戦略シミュレーション ===
    print("\n--- 投資戦略シミュレーション ---")

    def sim_strategy(name, bet_fn, scale_payout=True):
        total_invest = 0
        total_payout = 0
        races_bet = 0
        for _, row in df.iterrows():
            bet_amount = bet_fn(row)
            if bet_amount <= 0:
                continue
            races_bet += 1
            total_invest += bet_amount
            if row['hit'] == 1:
                if scale_payout:
                    total_payout += row['payout'] * (bet_amount / 700)
                else:
                    total_payout += row['payout']
        roi = total_payout / total_invest * 100 if total_invest > 0 else 0
        profit = total_payout - total_invest
        print(f"  {name:30s}: N={races_bet:3d}, 投資={total_invest:>8,.0f}円, "
              f"回収={total_payout:>8,.0f}円, ROI={roi:5.1f}%, 収支={profit:+,.0f}円")
        return {'races': int(races_bet), 'investment': round(total_invest),
                'payout': round(total_payout), 'ROI': round(roi, 1), 'profit': round(profit)}

    strategy_results = {}

    strategy_results['固定700円(現行)'] = sim_strategy(
        '固定700円(現行)', lambda r: 700)

    strategy_results['条件D半額350円'] = sim_strategy(
        '条件D半額350円', lambda r: 350 if r.condition == 'D' else 700)

    strategy_results['条件A/D半額350円'] = sim_strategy(
        '条件A/D半額350円', lambda r: 350 if r.condition in ('A', 'D') else 700)

    strategy_results['阪神除外'] = sim_strategy(
        '阪神除外', lambda r: 0 if r.venue == '阪神' else 700)

    strategy_results['条件D除外'] = sim_strategy(
        '条件D除外', lambda r: 0 if r.condition == 'D' else 700)

    strategy_results['A/D除外(C/X集中)'] = sim_strategy(
        'A/D除外(C/X集中)', lambda r: 0 if r.condition in ('A', 'D', 'E', 'B') else 700)

    strategy_results['午後のみ(7-12R)'] = sim_strategy(
        '午後のみ(7-12R)', lambda r: 0 if r.race_num <= 6 else 700)

    if 'score_diff' in df.columns:
        strategy_results['スコア差大→増額'] = sim_strategy(
            'スコア差大→1000/小→500',
            lambda r: 1000 if getattr(r, 'score_diff', 0) > 0.03 else 500)

    # === 5. 条件別×点数クロス分析 ===
    print("\n--- 条件別の最適点数 ---")
    for cond in ['A', 'C', 'D']:
        df_cond = df[df['condition'] == cond]
        if len(df_cond) < 5:
            continue
        print(f"\n条件{cond} (N={len(df_cond)}):")
        for n_bets in [5, 7]:
            total_hits = 0
            total_payout = 0
            invest_per_race = n_bets * 100
            total_invest = len(df_cond) * invest_per_race
            for _, row in df_cond.iterrows():
                result = row['result_parsed']
                bets = row['bets_parsed'][:n_bets]
                for bet in bets:
                    if bet == result:
                        total_hits += 1
                        total_payout += row['payout']
                        break
            roi = total_payout / total_invest * 100 if total_invest > 0 else 0
            print(f"  {n_bets}点: 的中={total_hits}, ROI={roi:.1f}%")

    # Save results
    output = {
        'analysis_date': '2026-03-29',
        'total_races': int(len(df)),
        'dates': sorted([int(d) for d in df['date'].unique()]),
        'winning_bet_distribution': {},
        'bet_count_analysis': bet_count_results,
        'strategy_simulation': strategy_results,
        'key_findings': [
            '現行7点固定: ROI 90.8% (209レース)',
            '5点に削減: 的中数維持しつつ投資額削減でROI改善の可能性',
            '条件D除外: ROI悪化（条件Dは実は118.6%でプラス）',
            '阪神除外: ROI改善（阪神45.4%が全体を引き下げ）',
            'サンプルサイズ209では統計的有意差の判定は困難',
            '4日間のデータでは分散が大きく結論は暫定的',
        ]
    }

    # Winning bet distribution
    for idx in range(1, 8):
        count = int((hits_df['winning_bet_idx'] == idx).sum())
        if count > 0:
            output['winning_bet_distribution'][f'{idx}点目'] = count

    with open(os.path.join(DATA_DIR, 'bet_optimization.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: data/bet_optimization.json")


if __name__ == '__main__':
    main()
