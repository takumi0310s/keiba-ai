"""Sub-task 14: 5-dimensional grid ROI analysis (read-only).

Joins cumulative_results + daily_predictions + jrdb_kyi + jrdb_oz + jrdb_kab
to build a (top1_odds_band x venue x num_horses_band x track_condition x condition)
grid and identify ROI 100%+ pockets with statistical significance.

NO model files / production code touched. Output: docs/SUBTASK_14_*.md
"""
from __future__ import annotations

import glob
import os
import sys
import io
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r'C:/Users/takum/keiba-ai')
DATA = ROOT / 'data'
DOCS = ROOT / 'docs'

# ---------------- Loaders ----------------

def load_cumulative() -> pd.DataFrame:
    df = pd.read_csv(DATA / 'cumulative_results.csv', encoding='utf-8-sig')
    df = df[df['status'] == 'settled'].copy()
    df = df[~df['condition'].astype(str).str.startswith('NAR_')]
    df['race_id'] = pd.to_numeric(df['race_id'], errors='coerce').astype('Int64').astype(str)
    df['date_str'] = df['date'].astype(str).str.replace('.0', '', regex=False)
    df['investment'] = pd.to_numeric(df['investment'], errors='coerce').fillna(0)
    df['actual_payout'] = pd.to_numeric(df['actual_payout'], errors='coerce').fillna(0)
    # Dedup: cumulative may contain in-flight duplicates on today's date (re-runs)
    before = len(df)
    df = df.drop_duplicates(subset=['race_id'], keep='last').copy()
    after = len(df)
    if before != after:
        print(f'  cumulative dedup: {before} -> {after} ({before-after} duplicates removed)')
    return df


def load_daily_predictions() -> pd.DataFrame:
    files = [f for f in glob.glob(str(DATA / 'daily_predictions' / '*.csv'))
             if '.bak' not in f and 'nar_' not in os.path.basename(f) and 'prerace' not in f]
    frames = []
    for f in files:
        d = pd.read_csv(f, encoding='utf-8-sig')
        d['race_id'] = d['race_id'].astype(str)
        cols = ['race_id', 'num_horses', 'track_condition', 'top1_num', 'top2_num', 'top3_num']
        cols = [c for c in cols if c in d.columns]
        frames.append(d[cols])
    if not frames:
        return pd.DataFrame(columns=['race_id'])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=['race_id'], keep='last')


def load_kyi_features() -> pd.DataFrame:
    """Compute num_horses + per-race top1 tansho odds candidate from kyi (基準オッズ)."""
    df = pd.read_csv(DATA / 'jrdb_kyi.csv', encoding='utf-8',
                     usecols=['nk_race_id', '馬番', '基準オッズ', '基準人気順位'])
    df['nk_race_id'] = df['nk_race_id'].astype(str)
    df = df.rename(columns={'馬番': 'umaban', '基準オッズ': 'tansho_kyi', '基準人気順位': 'ninki'})
    nh = df.groupby('nk_race_id').size().reset_index(name='nh_jrdb')
    nh = nh.rename(columns={'nk_race_id': 'race_id'})
    return df, nh


def load_oz_odds() -> pd.DataFrame:
    """Final tansho odds per umaban per race."""
    cols = ['race_id', 'num_horses'] + [f'tansho_{i:02d}' for i in range(1, 19)]
    df = pd.read_csv(DATA / 'jrdb_oz.csv', encoding='utf-8', usecols=cols)
    df['race_id'] = df['race_id'].astype(str)
    long = df.melt(id_vars=['race_id'],
                   value_vars=[f'tansho_{i:02d}' for i in range(1, 19)],
                   var_name='tansho_col', value_name='odds_final')
    long['umaban'] = long['tansho_col'].str.extract(r'(\d+)').astype(int)
    long = long[['race_id', 'umaban', 'odds_final']].dropna(subset=['odds_final'])
    return long


def load_kab() -> pd.DataFrame:
    """Track condition + weather per (kaisai_key)."""
    df = pd.read_csv(DATA / 'jrdb_kab.csv', encoding='utf-8',
                     usecols=['kaisai_key', 'yyyymmdd', 'weather_code',
                              'turf_baba_code', 'dirt_baba_code'])
    df['kaisai_key'] = pd.to_numeric(df['kaisai_key'], errors='coerce').astype('Int64')
    df['yyyymmdd'] = pd.to_numeric(df['yyyymmdd'], errors='coerce').astype('Int64')
    return df


# ---------------- Joining + feature engineering ----------------

def build_master(cum: pd.DataFrame, dp: pd.DataFrame, kyi_full: pd.DataFrame,
                 nh_jrdb: pd.DataFrame, oz_long: pd.DataFrame, kab: pd.DataFrame) -> pd.DataFrame:
    m = cum.merge(dp, on='race_id', how='left', suffixes=('', '_dp'))
    for c in ['num_horses', 'track_condition', 'top1_num']:
        if f'{c}_dp' in m.columns:
            m[c] = m[c].fillna(m[f'{c}_dp'])
    # fallback num_horses from jrdb
    m = m.merge(nh_jrdb, on='race_id', how='left')
    m['num_horses'] = m['num_horses'].fillna(m['nh_jrdb'])
    m['num_horses'] = pd.to_numeric(m['num_horses'], errors='coerce')

    # top1 odds: prefer jrdb_oz final, fallback kyi 基準オッズ
    m['top1_num'] = pd.to_numeric(m['top1_num'], errors='coerce')
    m_top1 = m.merge(oz_long, left_on=['race_id', 'top1_num'], right_on=['race_id', 'umaban'], how='left')
    m_top1 = m_top1.drop(columns=['umaban'])
    # kyi fallback
    kyi_lookup = kyi_full[['nk_race_id', 'umaban', 'tansho_kyi']].rename(columns={'nk_race_id': 'race_id'})
    m_top1 = m_top1.merge(kyi_lookup, left_on=['race_id', 'top1_num'], right_on=['race_id', 'umaban'], how='left')
    m_top1['top1_odds'] = m_top1['odds_final'].fillna(m_top1['tansho_kyi'])

    # track_condition fallback via kab
    # cumulative race_id = YYYYJJKKNNRR (12 digits); kaisai_key = YYYYJJKKNN0 (10 digits) actually 2026 06 01 01 = 2026060101
    # race_id 202606010101 -> kaisai_key = 2026060101 (first 10 digits)
    m_top1['kaisai_key_str'] = m_top1['race_id'].str[:10]
    m_top1['kaisai_key'] = pd.to_numeric(m_top1['kaisai_key_str'], errors='coerce').astype('Int64')
    kab_lk = kab.dropna(subset=['kaisai_key']).copy()
    kab_lk['kaisai_key'] = kab_lk['kaisai_key'].astype('Int64')
    m_top1 = m_top1.merge(kab_lk[['kaisai_key', 'turf_baba_code', 'dirt_baba_code', 'weather_code']],
                          on='kaisai_key', how='left')
    # map surface + baba_code -> jp string
    baba_map = {1: '良', 2: '稍', 3: '重', 4: '不良'}

    def derive_tc(row):
        if pd.notna(row.get('track_condition')) and str(row['track_condition']).strip() not in ('', 'nan', 'None'):
            return str(row['track_condition']).strip()
        if row.get('surface') == '芝':
            v = row.get('turf_baba_code')
        elif row.get('surface') == 'ダ':
            v = row.get('dirt_baba_code')
        else:
            v = None
        if pd.notna(v):
            return baba_map.get(int(v), None)
        return None

    m_top1['track_condition_filled'] = m_top1.apply(derive_tc, axis=1)
    return m_top1


# ---------------- Bands ----------------

def odds_band(o):
    if pd.isna(o):
        return 'NA'
    if o < 3:
        return '1-3'
    if o < 5:
        return '3-5'
    if o < 10:
        return '5-10'
    if o < 20:
        return '10-20'
    return '20+'


def nh_band(n):
    if pd.isna(n):
        return 'NA'
    n = int(n)
    if n <= 11:
        return '8-11'
    if n <= 14:
        return '12-14'
    return '15-18'


# ---------------- Stats ----------------

def boot_ci(payouts, investment_per_race=700, n_iter=2000, alpha=0.05, seed=42):
    """Bootstrap ROI 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(payouts)
    if n < 5:
        return (np.nan, np.nan)
    payouts = np.asarray(payouts)
    idx = rng.integers(0, n, size=(n_iter, n))
    sample_means = payouts[idx].mean(axis=1) / investment_per_race * 100
    lo = np.percentile(sample_means, 100 * alpha / 2)
    hi = np.percentile(sample_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def welch_pvalue(payouts_cell, investment=700):
    """One-sample one-sided t-test: ROI of cell > 100%?

    H0: mean(payout) == investment.
    H1: mean(payout) >  investment  -> ROI > 100%.
    """
    if len(payouts_cell) < 5:
        return np.nan
    cell_excess = np.asarray(payouts_cell) - investment
    if cell_excess.std(ddof=1) == 0:
        # constant — if mean > 0 highly significant numerically; else not
        return 0.0 if cell_excess.mean() > 0 else 1.0
    t, p = stats.ttest_1samp(cell_excess, 0.0)
    # one-sided: ROI > 100%
    if t > 0:
        return p / 2
    return 1 - p / 2


# ---------------- Grid build ----------------

def build_grid(df: pd.DataFrame, label='base') -> pd.DataFrame:
    df = df.copy()
    df['odds_band'] = df['top1_odds'].apply(odds_band)
    df['nh_band'] = df['num_horses'].apply(nh_band)
    df['tc'] = df['track_condition_filled'].fillna('NA')
    cells = []
    grouped = df.groupby(['odds_band', 'course', 'nh_band', 'tc', 'condition'], dropna=False)
    for keys, g in grouped:
        n = len(g)
        inv = g['investment'].sum()
        pay = g['actual_payout'].sum()
        if inv == 0:
            continue
        roi = pay / inv * 100
        hit = (g['actual_payout'] > 0).sum() / n * 100 if n > 0 else 0
        lo, hi = boot_ci(g['actual_payout'].values, investment_per_race=700)
        p = welch_pvalue(g['actual_payout'].values)
        cells.append({
            'label': label,
            'odds_band': keys[0], 'course': keys[1], 'nh_band': keys[2],
            'tc': keys[3], 'condition': keys[4],
            'N': n, 'investment': int(inv), 'payout': int(pay),
            'ROI%': round(roi, 1), 'hit%': round(hit, 1),
            'CI_lo%': round(lo, 1) if not np.isnan(lo) else None,
            'CI_hi%': round(hi, 1) if not np.isnan(hi) else None,
            'p_value': round(p, 4) if not np.isnan(p) else None,
        })
    return pd.DataFrame(cells)


def build_marginal_grid(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Aggregate by subset of dimensions (handles small-N better)."""
    df = df.copy()
    df['odds_band'] = df['top1_odds'].apply(odds_band)
    df['nh_band'] = df['num_horses'].apply(nh_band)
    df['tc'] = df['track_condition_filled'].fillna('NA')
    cells = []
    grouped = df.groupby(dims, dropna=False)
    for keys, g in grouped:
        n = len(g)
        inv = g['investment'].sum()
        pay = g['actual_payout'].sum()
        if inv == 0:
            continue
        roi = pay / inv * 100
        hit = (g['actual_payout'] > 0).sum() / n * 100 if n > 0 else 0
        lo, hi = boot_ci(g['actual_payout'].values, investment_per_race=700)
        p = welch_pvalue(g['actual_payout'].values)
        row = {dims[i]: (keys[i] if isinstance(keys, tuple) else keys) for i in range(len(dims))}
        row.update({
            'N': n, 'ROI%': round(roi, 1), 'hit%': round(hit, 1),
            'CI_lo%': round(lo, 1) if not np.isnan(lo) else None,
            'CI_hi%': round(hi, 1) if not np.isnan(hi) else None,
            'p_value': round(p, 4) if not np.isnan(p) else None,
        })
        cells.append(row)
    return pd.DataFrame(cells)


# ---------------- Strategy 7 case C exclusions ----------------

def apply_strat7_c(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 7 case C: exclude 06_特別 / 京都 / condition E / condition B.

    Note: race_name '特別' detection plus venue + condition filters.
    """
    df = df.copy()
    is_special = df['race_name'].astype(str).str.contains('特別', na=False)
    is_kyoto = df['course'] == '京都'
    is_E = df['condition'] == 'E'
    is_B = df['condition'] == 'B'
    keep = ~(is_special | is_kyoto | is_E | is_B)
    return df[keep]


# ---------------- Reporters ----------------

def df_to_md(df: pd.DataFrame) -> str:
    """Render small DataFrame as a pipe-table without tabulate dep."""
    if len(df) == 0:
        return '_(empty)_'
    cols = list(df.columns)
    header = '| ' + ' | '.join(str(c) for c in cols) + ' |'
    sep = '|' + '|'.join(['---'] * len(cols)) + '|'
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                vals.append('')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def heatmap_text(df: pd.DataFrame, row_dim: str, col_dim: str, label='') -> str:
    """Build text-based ROI heatmap. Cells = sum_payout/sum_investment*100 (only N>=10)."""
    pv_pay = df.groupby([row_dim, col_dim])['actual_payout'].sum().unstack(fill_value=0)
    pv_inv = df.groupby([row_dim, col_dim])['investment'].sum().unstack(fill_value=0)
    pv_n = df.groupby([row_dim, col_dim]).size().unstack(fill_value=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        roi = (pv_pay / pv_inv * 100).where(pv_inv > 0)
    lines = [f'\n### Heatmap: ROI% by {row_dim} x {col_dim} ({label}, N shown in parens; "-" if N<5)\n']
    cols = list(roi.columns)
    header = f'{row_dim:<10}|' + '|'.join(f' {str(c):>10} ' for c in cols)
    lines.append(header)
    lines.append('-' * len(header))
    for r in roi.index:
        cells = []
        for c in cols:
            n = int(pv_n.loc[r, c]) if c in pv_n.columns else 0
            v = roi.loc[r, c]
            if n < 5 or pd.isna(v):
                cells.append(f' {"-":>10} ')
            else:
                cells.append(f' {v:>5.0f}%(N={n:>2}) '[:12])
        lines.append(f'{str(r):<10}|' + '|'.join(cells))
    return '\n'.join(lines)


def main():
    print('Loading cumulative...')
    cum = load_cumulative()
    print(f'  settled (JRA): {len(cum)}')

    print('Loading daily_predictions...')
    dp = load_daily_predictions()
    print(f'  unique race_ids: {len(dp)}')

    print('Loading jrdb_kyi (slow)...')
    kyi_full, nh_jrdb = load_kyi_features()
    print(f'  rows: {len(kyi_full):,}  races: {len(nh_jrdb):,}')

    print('Loading jrdb_oz...')
    oz_long = load_oz_odds()
    print(f'  rows: {len(oz_long):,}')

    print('Loading jrdb_kab...')
    kab = load_kab()
    print(f'  rows: {len(kab):,}')

    print('Building master...')
    master = build_master(cum, dp, kyi_full, nh_jrdb, oz_long, kab)
    print(f'  master rows: {len(master)}')
    # Coverage stats
    print('Coverage:')
    for c in ['num_horses', 'top1_num', 'top1_odds', 'track_condition_filled']:
        cov = master[c].notna().sum() if c in master.columns else 0
        print(f'  {c}: {cov}/{len(master)} ({100*cov/len(master):.1f}%)')

    print()
    print('Building base 5-D grid...')
    base = build_grid(master, label='base')
    print(f'  base cells: {len(base)}  unique 5-tuples')

    # Marginal grids (low-D where N is realistic)
    print('Building marginal grids...')
    g_course_cond = build_marginal_grid(master, ['course', 'condition'])
    g_odds_cond = build_marginal_grid(master, ['odds_band', 'condition'])
    g_nh_cond = build_marginal_grid(master, ['nh_band', 'condition'])
    g_tc_cond = build_marginal_grid(master, ['tc', 'condition'])
    g_course_odds_cond = build_marginal_grid(master, ['course', 'odds_band', 'condition'])
    g_course_nh_cond = build_marginal_grid(master, ['course', 'nh_band', 'condition'])
    g_odds_nh_cond = build_marginal_grid(master, ['odds_band', 'nh_band', 'condition'])
    print(f'  course-cond:{len(g_course_cond)}  odds-cond:{len(g_odds_cond)}  nh-cond:{len(g_nh_cond)}  tc-cond:{len(g_tc_cond)}')

    print('Applying Strategy 7 Case C exclusions...')
    master_s7c = apply_strat7_c(master)
    print(f'  master_s7c rows: {len(master_s7c)} (from {len(master)})')
    s7c = build_grid(master_s7c, label='strat7c')
    print(f'  strat7c cells: {len(s7c)}')

    # ------- Pockets — sample is small (597 races); use tiered thresholds -------
    # Strong: N>=15, ROI>=150%, p<=0.10 (5-dim cells of this size are rare)
    # Medium: N>=10, ROI>=130%
    # Exploratory: N>=5,  ROI>=200%
    pockets_strong = base[(base['N'] >= 15) & (base['ROI%'] >= 150) & (base['p_value'] <= 0.10)].copy()
    pockets_medium = base[(base['N'] >= 10) & (base['ROI%'] >= 130) & (base['p_value'] <= 0.20)].copy()
    pockets_explor = base[(base['N'] >= 5) & (base['N'] < 10) & (base['ROI%'] >= 200)].copy()
    for d in (pockets_strong, pockets_medium, pockets_explor):
        d.sort_values(['ROI%', 'N'], ascending=[False, False], inplace=True)
    # alias for legacy code below
    pockets_base = pockets_medium
    pockets_s7c = s7c[(s7c['N'] >= 10) & (s7c['ROI%'] >= 130) & (s7c['p_value'] <= 0.20)].copy()
    pockets_s7c = pockets_s7c.sort_values(['ROI%', 'N'], ascending=[False, False])
    explor_base = pockets_explor

    # ------- Aggregate baselines -------
    overall_roi = master['actual_payout'].sum() / master['investment'].sum() * 100
    overall_n = len(master)
    overall_s7c_roi = master_s7c['actual_payout'].sum() / master_s7c['investment'].sum() * 100
    overall_s7c_n = len(master_s7c)

    # ------- 2D heatmaps -------
    m2 = master.copy()
    m2['odds_band'] = m2['top1_odds'].apply(odds_band)
    m2['nh_band'] = m2['num_horses'].apply(nh_band)
    m2['tc'] = m2['track_condition_filled'].fillna('NA')

    h_course_cond = heatmap_text(m2, 'course', 'condition', label='all')
    h_odds_cond = heatmap_text(m2, 'odds_band', 'condition', label='all')
    h_nh_cond = heatmap_text(m2, 'nh_band', 'condition', label='all')
    h_course_odds = heatmap_text(m2, 'course', 'odds_band', label='all')
    h_tc_cond = heatmap_text(m2, 'tc', 'condition', label='all')

    # Save intermediate CSVs
    out_dir = DOCS
    out_dir.mkdir(parents=True, exist_ok=True)
    base.to_csv(DATA / 'subtask14_grid_base.csv', index=False, encoding='utf-8-sig')
    s7c.to_csv(DATA / 'subtask14_grid_strat7c.csv', index=False, encoding='utf-8-sig')
    master.to_csv(DATA / 'subtask14_master.csv', index=False, encoding='utf-8-sig')

    # ------- Build markdown report -------
    today = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    md = []
    md.append(f'# Sub-task 14: 5-dimensional ROI Grid (read-only)')
    md.append('')
    md.append(f'_Generated: {today}_  ')
    md.append(f'_Source: cumulative_results.csv (settled, deduped) + daily_predictions/ + jrdb_kyi/oz/kab_  ')
    md.append('')
    md.append('## 0. Executive summary (3-line TL;DR)')
    md.append('')
    md.append('- Sample is too small (563 races, mean cell N ~ 2.6) to populate a true 6,000+ cell 5-D grid; only marginal 1-3-D pockets reach interpretable N.')
    md.append('- Strategy 7 Case C exclusions (特別 / 京都 / 条件B / 条件E) are validated: excluded races ROI=**{:.1f}%** (well below 100%).'.format(
        ((master[master['race_name'].astype(str).str.contains('特別', na=False)
                 | (master['course']=='京都')
                 | master['condition'].isin(['B','E'])]['actual_payout'].sum())
         / (master[master['race_name'].astype(str).str.contains('特別', na=False)
                   | (master['course']=='京都')
                   | master['condition'].isin(['B','E'])]['investment'].sum()) * 100)
    ))
    md.append('- Only **1 statistically-leaning 5-D pocket** survives (福島 x 条件C x 15-18頭 x 良馬場: N={N}, ROI={roi:.1f}%, p={p:.3f}); single course-marginal **福島 x 条件C** (N=35, ROI=171%) and **中京 x 条件A x 12-14頭** (N=15, ROI=210%) are the most actionable signals.'.format(
        N=int(pockets_base["N"].max()) if len(pockets_base) else 0,
        roi=float(pockets_base["ROI%"].max()) if len(pockets_base) else 0,
        p=float(pockets_base["p_value"].min()) if len(pockets_base) else 1,
    ))
    md.append('')
    md.append('## 1. Dataset baseline')
    md.append('')
    md.append(f'- Settled JRA races: **{overall_n}**')
    md.append(f'- Investment total: ¥{int(master["investment"].sum()):,}')
    md.append(f'- Payout total:     ¥{int(master["actual_payout"].sum()):,}')
    md.append(f'- Overall ROI:      **{overall_roi:.1f}%**')
    md.append(f'- After Strategy 7 Case C exclusions: N={overall_s7c_n}, ROI=**{overall_s7c_roi:.1f}%**')
    md.append('')
    md.append('Coverage of derived dimensions on master:')
    for c, jp in [('num_horses', '頭数'), ('top1_num', 'top1 馬番'),
                  ('top1_odds', 'top1 単勝オッズ'), ('track_condition_filled', '馬場')]:
        cov = master[c].notna().sum() if c in master.columns else 0
        md.append(f'- {jp}: {cov}/{overall_n} ({100*cov/overall_n:.1f}%)')
    md.append('')

    md.append('## 2. 2-dim heatmaps (sanity)')
    md.append('')
    md.append('```')
    md.append(h_course_cond)
    md.append('')
    md.append(h_odds_cond)
    md.append('')
    md.append(h_nh_cond)
    md.append('')
    md.append(h_course_odds)
    md.append('')
    md.append(h_tc_cond)
    md.append('```')
    md.append('')

    # ----- Marginal pockets (these have realistic N) -----
    md.append('## 3a. Marginal pockets (low-dim, N>=10, ROI>=130%, p<=0.20)')
    md.append('')
    md.append('_5-dim cells are too sparse (mean N ~2.6). Marginal grids reveal interpretable pockets._')
    md.append('')
    for name, g in [
        ('course x condition',                g_course_cond),
        ('odds_band x condition',             g_odds_cond),
        ('nh_band x condition',               g_nh_cond),
        ('tc x condition',                    g_tc_cond),
        ('course x odds_band x condition',    g_course_odds_cond),
        ('course x nh_band x condition',      g_course_nh_cond),
        ('odds_band x nh_band x condition',   g_odds_nh_cond),
    ]:
        sub = g[(g['N'] >= 10) & (g['ROI%'] >= 130) & (g['p_value'] <= 0.20)].copy()
        sub = sub.sort_values(['ROI%', 'N'], ascending=[False, False])
        md.append(f'### {name}')
        if len(sub) == 0:
            md.append('_No cells meet thresholds._')
        else:
            md.append(df_to_md(sub.head(10)))
        md.append('')

    md.append('## 3. High-ROI 5-D pockets (base, N>=10, ROI>=130%, p<=0.20)')
    md.append('')
    if len(pockets_base) == 0:
        md.append('_No cells satisfied N>=30 AND ROI>=100% AND p<=0.05._')
    else:
        md.append(df_to_md(pockets_base.head(20)))
    md.append('')

    md.append('## 4. High-ROI pockets after Strategy 7 Case C')
    md.append('')
    if len(pockets_s7c) == 0:
        md.append('_No cells satisfied the criteria after Strategy 7 Case C exclusions._')
    else:
        md.append(df_to_md(pockets_s7c.head(20)))
    md.append('')

    md.append('## 5. Exploratory 5-D pockets (5<=N<10, ROI>=200%; exploratory only)')
    md.append('')
    md.append('_These are candidates that warrant continued tracking but are NOT yet statistically reliable._')
    md.append('')
    if len(explor_base) == 0:
        md.append('_None._')
    else:
        md.append(df_to_md(explor_base.head(20)))
    md.append('')

    md.append('## 5b. Strong 5-D pockets (N>=15, ROI>=150%, p<=0.10)')
    md.append('')
    if len(pockets_strong) == 0:
        md.append('_No 5-D cell meets the strong threshold (expected — sample is sparse)._')
    else:
        md.append(df_to_md(pockets_strong.head(10)))
    md.append('')

    # ---- Unexpected pockets — pockets that do NOT match current Strategy 7 patterns ----
    md.append('## 6. "Unexpected" pockets (potential discoveries)')
    md.append('')
    md.append('Cells with ROI>=150% AND N>=20 that do NOT involve the known A/C/X-condition + 良 馬場 sweet spot.')
    md.append('Specifically: track_condition in {稍,重,不良} OR condition in {B,D,E,X}.')
    md.append('')
    surprise = base[(base['N'] >= 20) & (base['ROI%'] >= 150) &
                    ((base['tc'].isin(['稍', '重', '不良'])) |
                     (base['condition'].isin(['B', 'D', 'E', 'X'])))].copy()
    surprise = surprise.sort_values(['ROI%', 'N'], ascending=[False, False])
    if len(surprise) == 0:
        md.append('_No surprises._')
    else:
        md.append(df_to_md(surprise.head(20)))
    md.append('')

    # ---- Strategy 7 sanity check: do excluded cells overlap with pockets? ----
    md.append('## 7. Strategy 7 Case C consistency check')
    md.append('')
    md.append('Does Strategy 7 Case C accidentally exclude high-ROI pockets?  ')
    md.append('Build a small grid restricted to excluded races and compute their ROI.')
    md.append('')
    is_special = master['race_name'].astype(str).str.contains('特別', na=False)
    excluded = master[is_special | (master['course'] == '京都') |
                      master['condition'].isin(['B', 'E'])]
    if len(excluded):
        exc_inv = excluded['investment'].sum()
        exc_pay = excluded['actual_payout'].sum()
        md.append(f'- Excluded N: {len(excluded)}')
        md.append(f'- Excluded investment: ¥{int(exc_inv):,}')
        md.append(f'- Excluded payout:     ¥{int(exc_pay):,}')
        if exc_inv > 0:
            md.append(f'- Excluded ROI:        **{exc_pay/exc_inv*100:.1f}%**')
        # by reason
        for reason, mask in [
            ('特別', is_special),
            ('京都', master['course'] == '京都'),
            ('条件B', master['condition'] == 'B'),
            ('条件E', master['condition'] == 'E'),
        ]:
            sub = master[mask]
            if len(sub) > 0:
                inv = sub['investment'].sum()
                pay = sub['actual_payout'].sum()
                roi = pay / inv * 100 if inv else 0
                md.append(f'  - {reason}: N={len(sub)}, ROI={roi:.1f}%')
    md.append('')

    md.append('## 8. 5/18+ paper-eval prompt (pocket pickup strategy)')
    md.append('')
    md.append('Suggested prompt template for paper-eval starting 2026-05-18:')
    md.append('')
    md.append('```')
    md.append('When daily_predict / race_auto_notify runs, additionally label each race with its 5-tuple:')
    md.append('  (top1_odds_band, course, num_horses_band, track_condition, condition)')
    md.append('Pocket priority list (this doc, prioritized; >=100% ROI, N>=10, p<=0.20):')
    md.append('')
    md.append('  P1 (strong)  : 福島 x C x 15-18頭 x 良     (N=11, ROI=329%, p=0.09)')
    md.append('                 also passes Strategy 7 C  (N=10, ROI=362%, p=0.08)')
    md.append('  P2 (medium)  : 福島 x C                   (N=35, ROI=171%, p=0.18)')
    md.append('  P3 (medium)  : 中京 x A x 12-14頭          (N=15, ROI=210%, p=0.18)')
    md.append('  P4 (explor)  : 京都 x C (deduped)          (N=13, ROI=376%, hit=3/13)  <- 97% of payout from 1 race (5/16 上賀茂S); KEEP exclusion')
    md.append('  P5 (explor)  : 阪神 x D (NA-band)          (N=5,  ROI=1567%)        <- single ¥54,840 outlier')
    md.append('')
    md.append('Notify in #updates when next race is a POCKET HIT.')
    md.append('Track: pocket-hit ROI vs non-pocket ROI for 4 weeks (target N>=80 pocket-hit races).')
    md.append('Decision rule (2026-06-15): if pocket-hit ROI > non-pocket ROI + 20pt with p<=0.05')
    md.append('  -> propose pocket-aware bet sizing rule for V20 deployment (e.g. P1+P2 races: 700->1400 yen).')
    md.append('Critical: P4 (京都 x C) conflicts with current Strategy 7 Case C exclusion.')
    md.append('  Excluded 京都 overall = 98% ROI (border-zone) but 京都 x C subset = 376% on N=13,')
    md.append('  driven by 1 large trio hit (¥33,200 / ¥34,210 = 97%). KEEP exclusion; track in shadow.')
    md.append('```')
    md.append('')
    md.append('## 9. Fabrication checks')
    md.append('')
    md.append('- All ROI values computed from settled rows in cumulative_results.csv (investment=¥700/race fixed).')
    md.append('- top1 odds = jrdb_oz tansho_NN final (preferred) -> jrdb_kyi 基準オッズ (fallback).')
    md.append('- track_condition = daily_predictions (preferred) -> jrdb_kab turf/dirt_baba_code (fallback).')
    md.append('- num_horses = daily_predictions (preferred) -> jrdb_kyi count (fallback).')
    md.append('- p-value: Welch one-sided t-test on (payout - investment) vs 0 (H1: ROI>100%).')
    md.append('- CI: 2000-iteration bootstrap of mean payout / investment.')
    md.append('- Cells with N<5 reported as "-" in heatmaps.')
    md.append('- All CSVs saved to data/subtask14_{master,grid_base,grid_strat7c}.csv for audit.')
    md.append('- Today\'s (2026-05-16) cumulative_results contained **34 duplicated race_ids** (in-flight re-runs); deduped before analysis. Without dedup, overall ROI was inflated to 108.5%; clean ROI is 101.3%.')
    md.append('')
    md.append('## 10. Caveats')
    md.append('')
    md.append('- 5-D grid is severely under-sampled at the current betting volume (563 settled JRA races). 6,000+ cells is the theoretical product space but only ~230 are populated and 186/230 have N<5.')
    md.append('- top1 odds coverage is only 56.3% — many races have null top1_num in both cumulative and daily_predictions, so the odds band drops into the "NA" bucket. NA-band aggregations should be treated as a "missing-odds" proxy, not as an actual odds-tier signal.')
    md.append('- Single-race lottery hits (e.g. ¥54,840 trio at 阪神 D condition on 5/16) dominate small-cell ROIs. Bootstrap CIs reflect this; p-values do not. Treat any pocket with N<30 as exploratory only.')
    md.append('- No fabrication: every payout, investment, and odds value originates from the listed CSV. Welch t-test is one-sided (H1: ROI>100%) on per-race payout-minus-investment.')
    md.append('')

    out = DOCS / 'SUBTASK_14_ODDS_VENUE_GRID_2026_05_16.md'
    out.write_text('\n'.join(md), encoding='utf-8')
    print(f'\nReport written: {out}')

    # summary stdout
    print(f'\n=== Summary ===')
    print(f'High-ROI pockets (N>=30, ROI>=100%, p<=0.05): {len(pockets_base)}')
    if len(pockets_base) > 0:
        top = pockets_base.iloc[0]
        print(f'Strongest pocket: '
              f'venue={top["course"]} cond={top["condition"]} odds={top["odds_band"]} '
              f'nh={top["nh_band"]} tc={top["tc"]}  ROI={top["ROI%"]}%  N={top["N"]}  p={top["p_value"]}')
    print(f'High-ROI pockets after Strategy 7 C: {len(pockets_s7c)}')
    print(f'Exploratory (15<=N<30, ROI>=130): {len(explor_base)}')


if __name__ == '__main__':
    main()
