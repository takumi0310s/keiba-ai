"""
分布乖離チェック: 学習データ内の年別ドリフトを検出。

実行:
    python tools/check_distribution_drift.py

出力:
    report/distribution_drift_20260423.md

ロジック:
    学習df (data/_v15_train_df_cache.pkl) を 2024年 vs 2025年 に分割し、
    150 v15 特徴量について平均・分散の乖離を計算。
    乖離大の特徴量を要注意としてリスト化。
"""
import pickle
import os
import gzip
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CACHE = os.path.join(BASE_DIR, 'data', '_v15_train_df_cache.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v15_central_live.pkl.gz')
REPORT_PATH = os.path.join(BASE_DIR, 'report', 'distribution_drift_20260423.md')


def safe_stats(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return None
    return {
        'n': len(s),
        'mean': float(s.mean()),
        'std': float(s.std(ddof=0)),
        'p50': float(s.median()),
        'p10': float(s.quantile(0.10)),
        'p90': float(s.quantile(0.90)),
        'zero_rate': float((s == 0).mean()),
    }


def drift_score(a, b):
    """簡易ドリフトスコア: 平均差を統合標準偏差で割る (Cohen's d 風)"""
    if a is None or b is None:
        return None
    pooled = np.sqrt((a['std'] ** 2 + b['std'] ** 2) / 2)
    if pooled < 1e-9:
        return 0.0 if abs(a['mean'] - b['mean']) < 1e-9 else float('inf')
    return abs(a['mean'] - b['mean']) / pooled


def main():
    print('Loading train cache...')
    d = pickle.load(open(TRAIN_CACHE, 'rb'))
    df = d['df']
    print(f'  {len(df):,} rows, {len(df.columns)} cols')

    print('Loading model features...')
    m = pickle.load(gzip.open(MODEL_PATH, 'rb'))
    feats = m['features']
    print(f'  {len(feats)} v15 features')

    # year_full から 2024 / 2025 抽出
    if 'year_full' not in df.columns:
        print('ERROR: year_full not found')
        return
    df_24 = df[df['year_full'] == 2024]
    df_25 = df[df['year_full'] == 2025]
    print(f'  2024: {len(df_24):,} rows / 2025: {len(df_25):,} rows')

    # ドリフトチェック
    results = []
    missing = []
    for f in feats:
        if f not in df.columns:
            missing.append(f)
            continue
        s24 = safe_stats(df_24[f])
        s25 = safe_stats(df_25[f])
        if s24 is None or s25 is None:
            continue
        d_mean = drift_score(s24, s25)
        zero_diff = abs(s24['zero_rate'] - s25['zero_rate'])
        results.append({
            'feature': f,
            'mean_24': s24['mean'],
            'mean_25': s25['mean'],
            'std_24': s24['std'],
            'std_25': s25['std'],
            'drift_d': d_mean if d_mean != float('inf') else 99.0,
            'zero_rate_24': s24['zero_rate'],
            'zero_rate_25': s25['zero_rate'],
            'zero_diff': zero_diff,
        })

    res_df = pd.DataFrame(results).sort_values('drift_d', ascending=False)

    # 区分
    big = res_df[res_df['drift_d'] > 0.3]
    mid = res_df[(res_df['drift_d'] > 0.1) & (res_df['drift_d'] <= 0.3)]
    small = res_df[res_df['drift_d'] <= 0.1]

    # zero_rate急変も別軸でチェック
    zero_changed = res_df[res_df['zero_diff'] > 0.10].sort_values('zero_diff', ascending=False)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = []
    lines.append(f'# 分布乖離チェック (2024 vs 2025)\n')
    lines.append(f'生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append(f'データソース: data/_v15_train_df_cache.pkl\n')
    lines.append(f'\n')
    lines.append(f'## サマリー\n')
    lines.append(f'- 比較対象特徴量: {len(res_df)}/{len(feats)}\n')
    lines.append(f'- 学習df cols未登載: {len(missing)} (予測時に動的生成される特徴量)\n')
    lines.append(f'- 2024年サンプル: {len(df_24):,} rows\n')
    lines.append(f'- 2025年サンプル: {len(df_25):,} rows\n')
    lines.append(f'\n')
    lines.append(f'## 乖離分布 (Cohen\'s d 風スコア)\n')
    lines.append(f'- 大 (d > 0.3): **{len(big)}** 個 — 要注意\n')
    lines.append(f'- 中 (0.1 < d <= 0.3): {len(mid)} 個 — 監視\n')
    lines.append(f'- 小 (d <= 0.1): {len(small)} 個 — 問題なし\n')
    lines.append(f'\n')
    lines.append(f'## 乖離大 TOP15 (d > 0.3)\n')
    lines.append(f'| feature | mean24 | mean25 | std24 | std25 | drift_d | zero24 | zero25 |\n')
    lines.append(f'|---------|--------|--------|-------|-------|---------|--------|--------|\n')
    for _, r in big.head(15).iterrows():
        lines.append(
            f'| {r["feature"]} | {r["mean_24"]:.3f} | {r["mean_25"]:.3f} | '
            f'{r["std_24"]:.3f} | {r["std_25"]:.3f} | {r["drift_d"]:.3f} | '
            f'{r["zero_rate_24"]:.2f} | {r["zero_rate_25"]:.2f} |\n'
        )

    if len(zero_changed) > 0:
        lines.append(f'\n## ゼロ率急変 TOP10 (|Δzero_rate| > 0.10)\n')
        lines.append(f'| feature | zero24 | zero25 | Δ |\n')
        lines.append(f'|---------|--------|--------|---|\n')
        for _, r in zero_changed.head(10).iterrows():
            lines.append(
                f'| {r["feature"]} | {r["zero_rate_24"]:.2f} | {r["zero_rate_25"]:.2f} | {r["zero_diff"]:.2f} |\n'
            )

    if missing:
        lines.append(f'\n## 学習dfに無い特徴量 ({len(missing)}個)\n')
        lines.append(f'予測時のみ動的生成 (build_features 内で計算):\n\n')
        lines.append(f'```\n')
        for f in missing:
            lines.append(f'{f}\n')
        lines.append(f'```\n')

    lines.append(f'\n## 判定\n')
    if len(big) == 0:
        verdict = '**問題なし**: 全特徴量の年次ドリフトが許容範囲内。'
    elif len(big) <= 5:
        verdict = f'**軽微な要注意**: {len(big)}個の特徴量に分布変化。本番への影響は限定的。'
    elif len(big) <= 15:
        verdict = f'**中程度の注意**: {len(big)}個の特徴量に分布変化。継続監視推奨。'
    else:
        verdict = f'**要対応**: {len(big)}個の特徴量に分布変化。再学習を検討。'
    lines.append(f'{verdict}\n')

    lines.append(f'\n## 採用基準と本日の判断\n')
    lines.append(f'- 採用基準: 改善率 >= 3pt かつ 予測変動 <= 20%\n')
    lines.append(f'- 本日の判断: **修正なし** (本作業はチェックのみ、副作用ある修正は次週以降に検討)\n')

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))

    print(f'\nReport saved: {REPORT_PATH}')
    print(f'Drift summary: 大={len(big)} 中={len(mid)} 小={len(small)}')
    print(f'Zero-rate changed: {len(zero_changed)}')
    if len(big) > 0:
        print('\nTop 5 drift:')
        for _, r in big.head(5).iterrows():
            print(f'  {r["feature"]:30s} d={r["drift_d"]:.3f} ({r["mean_24"]:.3f} -> {r["mean_25"]:.3f})')


if __name__ == '__main__':
    main()
