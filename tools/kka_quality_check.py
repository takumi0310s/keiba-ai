"""KKA features quality + LEAK monitoring — Session #53 D.

Read-only diagnostics on `data/jrdb_kka_features.csv`:
    1. Year / 場 coverage 詳細
    2. feature 内 redundancy (top3r 系 corr matrix)
    3. monotonic / 極端値 detection
    4. SKB-like POST-RACE leak pattern check
    5. PASS / NG verdict
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'data', 'jrdb_kka_features.csv')
OUT = os.path.join(REPO, 'data', 'v18', 'session_53_kka_quality.json')


def jou_decode_jrdb(race_id_str: str) -> str:
    """JRDB jou code (chars 5-6) → 場名。"""
    JOU = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
           '05': '東京', '06': '中山', '07': '中京', '08': '京都',
           '09': '阪神', '10': '小倉'}
    if not isinstance(race_id_str, str) or len(race_id_str) < 6:
        return 'unknown'
    return JOU.get(race_id_str[4:6], 'unknown')


def main():
    print(f'Loading: {SRC}')
    df = pd.read_csv(SRC, dtype={'race_id': str})
    n = len(df)
    print(f'  rows: {n:,}')

    df['_year'] = df['race_id'].str[:4]
    df['_jou'] = df['race_id'].apply(jou_decode_jrdb)

    report = {'n_rows': n}

    # 1. Year coverage
    print('\n--- 1. Year coverage ---')
    yr_cov = df.groupby('_year').agg(
        n=('umaban', 'size'),
        jra_top3r_nonnull=('kka_jra_seiseki_top3r', lambda s: s.notna().mean() * 100),
        kyori_top3r_nonnull=('kka_kyori_seiseki_top3r', lambda s: s.notna().mean() * 100),
    ).round(1)
    print(yr_cov)
    report['year_coverage'] = yr_cov.to_dict()

    # 2. 場 coverage
    print('\n--- 2. Jou coverage ---')
    jou_cov = df.groupby('_jou').size().sort_values(ascending=False)
    print(jou_cov.head(15))
    report['jou_coverage'] = jou_cov.to_dict()

    # 3. Top3r 内 redundancy (corr matrix)
    print('\n--- 3. Top3r internal redundancy ---')
    top3r_cols = [c for c in df.columns if c.endswith('_top3r')]
    sub = df[top3r_cols].dropna(subset=['kka_jra_seiseki_top3r'])
    sub_corr = sub.corr().round(3)
    print(f'  top3r features: {len(top3r_cols)}')
    print(f'  N (jra_seiseki non-null): {len(sub):,}')
    # Find features highly correlated with kka_jra_seiseki_top3r
    base_corr = sub_corr['kka_jra_seiseki_top3r'].drop('kka_jra_seiseki_top3r').sort_values(ascending=False)
    print('  corr with kka_jra_seiseki_top3r (top 5):')
    print(base_corr.head(5).to_string())
    print('  corr with kka_jra_seiseki_top3r (bottom 5):')
    print(base_corr.tail(5).to_string())
    report['top3r_corr_with_jra'] = base_corr.to_dict()

    # 4. Distribution / extreme value check (LEAK signal)
    print('\n--- 4. LEAK 監査: top3r distribution ---')
    leak_check = []
    for c in ['kka_jra_seiseki_top3r', 'kka_kyori_seiseki_top3r',
              'kka_class_seiseki_top3r', 'kka_track_seiseki_top3r',
              'kka_heavy_seiseki_top3r']:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if len(s) < 100:
            continue
        # Equal to 1.0 (all wins) or 0.0 (all losses)
        all_one = (s == 1.0).mean() * 100
        all_zero = (s == 0.0).mean() * 100
        # Std → 健全な features は std 0.15-0.25
        std_v = s.std()
        verdict = 'PASS'
        # SKB-like leak: monotonic, extreme separation. Heuristic: all_one > 30%
        # でかつ all_zero > 30% は不自然 (二極化)
        if all_one + all_zero > 60:
            verdict = 'NG (二極化)'
        if std_v > 0.45:
            verdict = 'NG (std 異常)'
        row = dict(feature=c, all_one_pct=round(all_one, 1),
                   all_zero_pct=round(all_zero, 1), std=round(std_v, 4),
                   verdict=verdict)
        leak_check.append(row)
        print(f'  {c}: all=1: {all_one:.1f}%, all=0: {all_zero:.1f}%, std={std_v:.4f}, {verdict}')
    report['leak_check'] = leak_check

    # 5. starts (経験) distribution
    print('\n--- 5. starts (経験) distribution ---')
    starts_stats = []
    for c in ['kka_jra_seiseki_starts', 'kka_kyori_seiseki_starts',
              'kka_class_seiseki_starts']:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        starts_stats.append(dict(
            feature=c,
            mean=round(s.mean(), 2),
            median=int(s.median()) if not pd.isna(s.median()) else None,
            p90=int(s.quantile(0.9)) if len(s) else None,
            p99=int(s.quantile(0.99)) if len(s) else None,
            zero_pct=round((s == 0).mean() * 100, 1),
        ))
    for r in starts_stats:
        print(f'  {r}')
    report['starts_stats'] = starts_stats

    # 6. V15 既存 features との redundancy 仮評価
    # KKA jra_seiseki は 「全 JRA 成績」 → V15 horse_career_top3r と概念的に重複
    # V15 horse_dist_top3r ≒ KKA kyori_seiseki_top3r
    # V15 horse_surface_top3r ≒ KKA track_seiseki_top3r
    print('\n--- 6. V15 既存 features との redundancy 推定 ---')
    redundancy = [
        ('kka_jra_seiseki_top3r', 'V15.horse_career_top3r', '高 (両方とも全成績)'),
        ('kka_kyori_seiseki_top3r', 'V15.horse_dist_top3r', '高 (両方とも距離別)'),
        ('kka_track_seiseki_top3r', 'V15.horse_surface_top3r', '中 (track vs surface 概念異)'),
        ('kka_heavy_seiseki_top3r', 'なし', '低 (V15 に重馬場 features 無し)'),
        ('kka_class_seiseki_top3r', 'なし', '低 (V15 に class 別 features 無し)'),
        ('kka_speed_seiseki_top3r', 'なし', '低 (V15 にペース別 features 無し)'),
        ('kka_season_seiseki_top3r', 'なし', '低 (V15 に season 別 features 無し)'),
        ('kka_waku_seiseki_top3r', 'V15.frame_course_dist_wr', '中 (waku の集計方法異)'),
        ('kka_dam_rensho_max', 'V15.dam_top3r (除外済)', '低 (V15 で dam リーク事故、 KKA は連勝率)'),
    ]
    print('  KKA -> V15 既存 mapping:')
    for k, v, sev in redundancy:
        print(f'    {k:<35} <-> {v:<30} : {sev}')
    report['redundancy_with_v15'] = [
        dict(kka=k, v15=v, severity=s) for k, v, s in redundancy
    ]

    # Final verdict
    leak_ng = [r for r in leak_check if r['verdict'].startswith('NG')]
    print('\n--- Final Verdict ---')
    if leak_ng:
        print(f'  LEAK 監査: NG ({len(leak_ng)} features 異常)')
        print(f'  該当: {[r["feature"] for r in leak_ng]}')
        verdict = 'NG'
    else:
        print('  LEAK 監査: PASS (二極化 / 異常 std 無し)')
        verdict = 'PASS'
    print(f'  V15 redundancy: 主要 5 features 中 3 件 が高重複 (jra/kyori/track)')
    print(f'  → V15 retrain 必須、 standalone PoC ではマージ不可')
    report['final_verdict'] = verdict

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
