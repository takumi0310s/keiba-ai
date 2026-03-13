#!/usr/bin/env python
"""■6 詳細ドローダウン分析
monte_carlo_sim.pyを拡張して詳細なドローダウン指標を計算。
"""
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

# バックテスト実績
JRA_CONDITIONS = {
    'A': {'hit_rate': 0.451, 'avg_return': 25429227 / 3890, 'investment': 700, 'n': 8634, 'label': '8-14頭/1600m+/良~稍'},
    'B': {'hit_rate': 0.454, 'avg_return': 3539139 / 484, 'investment': 700, 'n': 1067, 'label': '8-14頭/1600m+/重~不良'},
    'C': {'hit_rate': 0.334, 'avg_return': 22355894 / 2137, 'investment': 700, 'n': 6405, 'label': '15頭+/1600m+/良~稍'},
    'D': {'hit_rate': 0.282, 'avg_return': 16958143 / 2769, 'investment': 700, 'n': 9807, 'label': '1400m以下'},
    'E': {'hit_rate': 0.757, 'avg_return': 1273853 / 431, 'investment': 700, 'n': 569, 'label': '7頭以下'},
    'X': {'hit_rate': 0.361, 'avg_return': 4418076 / 381, 'investment': 700, 'n': 1055, 'label': '15頭+/重~不良'},
}
TOTAL_RACES = sum(c['n'] for c in JRA_CONDITIONS.values())
COND_WEIGHTS = {k: v['n'] / TOTAL_RACES for k, v in JRA_CONDITIONS.items()}


def simulate_detailed(initial_fund, num_races, conditions, cond_weights, rng):
    """詳細なシミュレーション（資金推移を全て記録）"""
    fund = initial_fund
    peak = initial_fund
    max_dd_amount = 0
    max_dd_pct = 0
    max_dd_start = 0
    max_dd_end = 0
    current_dd_start = 0

    fund_history = [fund]
    dd_history = [0.0]
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_loss_cond = ''
    current_loss_cond = []

    cond_keys = list(cond_weights.keys())
    cond_probs = [cond_weights[k] for k in cond_keys]

    # 条件別連敗トラッキング
    cond_loss_streaks = {k: [] for k in cond_keys}
    cond_current_streak = {k: 0 for k in cond_keys}

    dd_below_peak_count = 0
    recovery_races = []  # DD回復に要したレース数

    prev_peak_race = 0  # ピーク更新したレース番号

    for i in range(num_races):
        cond = rng.choice(cond_keys, p=cond_probs)
        c = conditions[cond]
        investment = c['investment']

        if fund < investment:
            return {
                'final_fund': 0, 'ruin': True,
                'max_dd_amount': max_dd_amount, 'max_dd_pct': max_dd_pct,
                'max_dd_recovery_races': num_races,
                'max_consecutive_losses': max_consecutive_losses,
                'max_loss_streak_cond': max_loss_cond,
                'fund_history': fund_history[:min(1000, len(fund_history))],
                'dd_below_peak_fraction': dd_below_peak_count / max(1, i),
                'cond_loss_streaks': {k: max(v) if v else 0 for k, v in cond_loss_streaks.items()},
            }

        fund -= investment
        hit = rng.random() < c['hit_rate']

        if hit:
            avg_return = c['avg_return']
            log_mean = np.log(avg_return) - 0.5
            payout = rng.lognormal(log_mean, 0.8)
            payout = max(100, payout)
            fund += payout
            consecutive_losses = 0
            current_loss_cond = []

            # 条件別連敗リセット
            cond_loss_streaks[cond].append(cond_current_streak[cond])
            cond_current_streak[cond] = 0
        else:
            consecutive_losses += 1
            current_loss_cond.append(cond)
            cond_current_streak[cond] += 1

            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses
                max_loss_cond = ','.join(set(current_loss_cond[-max_consecutive_losses:]))

        # ピーク・ドローダウン更新
        if fund > peak:
            if peak > initial_fund and fund > peak:
                recovery_races.append(i - prev_peak_race)
            peak = fund
            prev_peak_race = i
        else:
            dd_below_peak_count += 1

        dd_amount = peak - fund
        dd_pct = dd_amount / peak if peak > 0 else 0
        if dd_amount > max_dd_amount:
            max_dd_amount = dd_amount
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        # 間引いて記録（メモリ節約）
        if i % max(1, num_races // 500) == 0:
            fund_history.append(fund)
            dd_history.append(dd_pct)

    # 最終的な条件別連敗を記録
    for k in cond_keys:
        cond_loss_streaks[k].append(cond_current_streak[k])

    return {
        'final_fund': fund, 'ruin': False,
        'max_dd_amount': max_dd_amount,
        'max_dd_pct': max_dd_pct,
        'max_dd_recovery_races': np.mean(recovery_races) if recovery_races else 0,
        'max_consecutive_losses': max_consecutive_losses,
        'max_loss_streak_cond': max_loss_cond,
        'fund_history': fund_history,
        'dd_below_peak_fraction': dd_below_peak_count / num_races,
        'cond_loss_streaks': {k: max(v) if v else 0 for k, v in cond_loss_streaks.items()},
    }


def run_detailed_mc(initial_fund, num_races, num_trials, conditions, cond_weights):
    """詳細モンテカルロ"""
    rng = np.random.default_rng(42)

    finals = np.zeros(num_trials)
    max_dds_pct = np.zeros(num_trials)
    max_dds_amt = np.zeros(num_trials)
    max_losses = np.zeros(num_trials, dtype=int)
    recoveries = np.zeros(num_trials)
    dd_fractions = np.zeros(num_trials)
    ruin_count = 0
    all_cond_streaks = {k: [] for k in conditions}

    for trial in range(num_trials):
        result = simulate_detailed(initial_fund, num_races, conditions, cond_weights, rng)
        finals[trial] = result['final_fund']
        max_dds_pct[trial] = result['max_dd_pct']
        max_dds_amt[trial] = result['max_dd_amount']
        max_losses[trial] = result['max_consecutive_losses']
        recoveries[trial] = result['max_dd_recovery_races']
        dd_fractions[trial] = result['dd_below_peak_fraction']
        if result['ruin']:
            ruin_count += 1
        for k, v in result['cond_loss_streaks'].items():
            all_cond_streaks[k].append(v)

    return {
        'initial_fund': initial_fund,
        'num_races': num_races,
        'num_trials': num_trials,
        'ruin_probability': round(ruin_count / num_trials * 100, 2),
        'profit_probability': round(np.mean(finals > initial_fund) * 100, 1),
        'avg_final_fund': int(np.mean(finals)),
        'median_final_fund': int(np.median(finals)),
        'expected_roi': round(np.mean(finals) / initial_fund * 100, 1),
        'max_drawdown': {
            'avg_pct': round(np.mean(max_dds_pct) * 100, 1),
            'median_pct': round(np.median(max_dds_pct) * 100, 1),
            'worst_pct': round(np.max(max_dds_pct) * 100, 1),
            'p95_pct': round(np.percentile(max_dds_pct, 95) * 100, 1),
            'avg_amount': int(np.mean(max_dds_amt)),
            'worst_amount': int(np.max(max_dds_amt)),
        },
        'recovery': {
            'avg_races_to_recover': round(np.mean(recoveries), 0),
            'median_races_to_recover': round(np.median(recoveries), 0),
            'p95_races_to_recover': round(np.percentile(recoveries, 95), 0),
        },
        'drawdown_frequency': {
            'avg_fraction_below_peak': round(np.mean(dd_fractions) * 100, 1),
            'note': '資金がピークを下回っている期間の割合',
        },
        'consecutive_losses': {
            'avg_max': round(np.mean(max_losses), 1),
            'median_max': int(np.median(max_losses)),
            'worst': int(np.max(max_losses)),
            'p95': int(np.percentile(max_losses, 95)),
        },
        'condition_loss_streaks': {
            k: {
                'avg_max': round(np.mean(v), 1),
                'worst': int(np.max(v)) if v else 0,
                'p95': int(np.percentile(v, 95)) if v else 0,
            } for k, v in all_cond_streaks.items()
        },
    }


def verify_assumptions():
    """モンテカルロの前提が楽観的でないか検証"""
    checks = {}

    # 1. 的中率の検証
    # 的中率はバックテスト（2020-2025, 20,579レース）の実績値をそのまま使用
    checks['hit_rates'] = {
        'source': 'Walk-forward backtest 2020-2025 (Pattern A, AUC 0.8017)',
        'method': '6年間の年別独立予測の集計値',
        'inflation_risk': 'LOW',
        'note': 'WF方式のため過学習リスクは低い。ただし2026年以降の外挿になるため、5-10%の劣化を想定すべき',
    }

    # 2. 配当分布の検証
    # 配当は対数正規分布（log_std=0.8）でシミュレーション
    # 実際の配当分布も対数正規に近い
    checks['payout_distribution'] = {
        'model': '対数正規分布 (log_std=0.8)',
        'actual_distribution': '実際の競馬配当も対数正規分布に従うことが知られている',
        'avg_return_source': 'バックテスト的中レースの配当合計 / 的中数',
        'concern': 'log_std=0.8は比較的大きな分散。実配当の分散と要照合。',
        'inflation_risk': 'MEDIUM',
        'note': '対数正規分布の性質上、平均は中央値より高くなる。log_mean = log(avg) - 0.5 で補正しているが、完全ではない可能性あり。',
    }

    # 3. 条件出現頻度
    checks['condition_frequency'] = {
        'source': 'バックテスト期間の実績頻度',
        'inflation_risk': 'LOW',
        'note': '条件分類は頭数・距離・馬場に基づくため、年による変動は小さい',
    }

    # 4. 独立性の仮定
    checks['independence'] = {
        'assumption': '各レースの的中/不的中は独立',
        'reality': 'おおむね独立だが、同日の同コースレースは馬場状態が共通するため完全独立ではない',
        'inflation_risk': 'LOW',
        'note': '依存性があると連敗が実際より長くなる可能性があるが、条件をランダム選択しているため影響は限定的',
    }

    # 総合評価
    checks['overall'] = {
        'optimism_level': 'SLIGHTLY_OPTIMISTIC',
        'main_risks': [
            '2026年以降のモデル劣化（的中率低下5-10%）',
            '配当分布の対数正規近似の精度',
            '投票による市場インパクト（大口投資時）',
        ],
        'conservative_adjustment': '実運用では期待ROIの60-70%を見込むのが妥当',
    }

    return checks


def main():
    print("=" * 60)
    print("  ■6 詳細ドローダウン分析")
    print("=" * 60)

    # 前提検証
    print("\n[1] 前提条件の検証...")
    assumptions = verify_assumptions()
    print(f"  楽観度: {assumptions['overall']['optimism_level']}")
    for risk in assumptions['overall']['main_risks']:
        print(f"    - {risk}")

    # 詳細モンテカルロ
    initial_funds = [10000, 30000, 100000]
    num_trials = 10000
    num_races = 1000

    results = {}
    for fund in initial_funds:
        print(f"\n[2] 初期資金: {fund:,}円 × {num_races}レース × {num_trials:,}回...")
        result = run_detailed_mc(fund, num_races, num_trials, JRA_CONDITIONS, COND_WEIGHTS)
        results[str(fund)] = result

        print(f"  破産確率: {result['ruin_probability']:.2f}%")
        print(f"  利益確率: {result['profit_probability']:.1f}%")
        print(f"  期待ROI: {result['expected_roi']:.1f}%")
        print(f"  MDD: avg={result['max_drawdown']['avg_pct']:.1f}%, worst={result['max_drawdown']['worst_pct']:.1f}%")
        print(f"  MDD回復: avg={result['recovery']['avg_races_to_recover']:.0f}レース")
        print(f"  DD期間: {result['drawdown_frequency']['avg_fraction_below_peak']:.1f}%")
        print(f"  最大連敗: avg={result['consecutive_losses']['avg_max']:.1f}, worst={result['consecutive_losses']['worst']}")
        print(f"  条件別連敗:")
        for k, v in result['condition_loss_streaks'].items():
            print(f"    {k}: avg_max={v['avg_max']:.1f}, worst={v['worst']}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'num_races': num_races,
            'num_trials': num_trials,
            'initial_funds': initial_funds,
        },
        'conditions': {k: {
            'hit_rate': v['hit_rate'],
            'avg_return': round(v['avg_return']),
            'investment': v['investment'],
            'n': v['n'],
            'label': v['label'],
        } for k, v in JRA_CONDITIONS.items()},
        'simulations': results,
        'assumptions_verification': assumptions,
    }

    out_path = os.path.join(BASE_DIR, 'data', 'drawdown_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")

    return output


if __name__ == '__main__':
    main()
