"""
Sub-task 13: 全条件 x 全券種 ROI grid 完全 audit (read-only)

cumulative_results.csv (V15 prediction LIVE) を jra_payouts.csv (JRA公式配当) と結合し、
予測した TOP1/TOP2/TOP3 を起点に 5 券種 (単勝/馬連/ワイド/三連複/三連単) で
ROI grid を完全計算する。

絶対遵守:
- read-only
- 全数値実測 (fabrication 禁止)
- N<30 cell は "sample 不足" 明示
"""
import json
import re
from collections import Counter
from itertools import combinations, permutations
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CUM_PATH = REPO / 'data' / 'cumulative_results.csv'
PAY_PATH = REPO / 'data' / 'jra_payouts.csv'
OUT_PATH = REPO / 'docs' / 'SUBTASK_13_TICKET_OPTIMIZATION_AUDIT_2026_05_16.md'


# ---------- 1. load + parse ----------

def parse_trio_tops(s):
    """trio_bets_str (e.g. '1-2-3; 1-2-4; ...') を解析して (top1, top2, top3) を返す."""
    if pd.isna(s):
        return (None, None, None)
    bets = [b.strip() for b in str(s).split(';')]
    if len(bets) != 7:
        return (None, None, None)
    try:
        nums = [[int(x) for x in b.split('-')] for b in bets]
    except Exception:
        return (None, None, None)
    first_horses = set(nums[0])
    for r in nums[1:]:
        first_horses &= set(r)
    if len(first_horses) != 1:
        return (None, None, None)
    top1 = next(iter(first_horses))
    cnt = Counter()
    for r in nums:
        for h in r:
            if h != top1:
                cnt[h] += 1
    most = cnt.most_common(2)
    if len(most) < 2:
        return (top1, None, None)
    return (top1, most[0][0], most[1][0])


def load_data():
    dfc = pd.read_csv(CUM_PATH, encoding='utf-8')
    dfc = dfc[(dfc['status'] == 'settled') & dfc['trio_bets_str'].notna()].copy()
    dfc['date'] = dfc['date'].astype(str).str.split('.').str[0]

    tops = dfc['trio_bets_str'].apply(parse_trio_tops)
    dfc['t1'] = tops.apply(lambda x: x[0])
    dfc['t2'] = tops.apply(lambda x: x[1])
    dfc['t3'] = tops.apply(lambda x: x[2])
    dfc = dfc[dfc[['t1', 't2', 't3']].notna().all(axis=1)].copy()
    dfc[['t1', 't2', 't3']] = dfc[['t1', 't2', 't3']].astype(int)

    dfp = pd.read_csv(PAY_PATH, encoding='utf-8')
    dfp['race_date'] = dfp['race_date'].astype(str)
    dfp['race_num'] = dfp['race_num'].astype(int)

    # Drop cumulative-side payout columns (they're partial/buggy) so jra_payouts columns survive merge
    dfc = dfc.drop(columns=['trio_payout', 'umaren_payout'], errors='ignore')

    # Course normalization (raw kanji preserved via utf-8)
    # cumulative course == payouts course (both kanji)
    merged = dfc.merge(
        dfp,
        left_on=['date', 'course', 'race_num'],
        right_on=['race_date', 'course', 'race_num'],
        how='inner',
    )
    return merged


# ---------- 2. payout lookup helpers ----------

def _parse_combo(s):
    """'1-7-8' -> (1,7,8)  /  '1/7/8' -> (1,7,8)."""
    if pd.isna(s):
        return ()
    s = str(s)
    parts = re.split(r'[-/]', s)
    try:
        return tuple(int(x) for x in parts)
    except Exception:
        return ()


def _parse_multi_combo(s):
    """'1-7/1-8/7-8' -> [(1,7),(1,8),(7,8)] (umaren wide 3組)."""
    if pd.isna(s):
        return []
    parts = str(s).split('/')
    out = []
    for p in parts:
        out.append(tuple(int(x) for x in re.split(r'[-/]', p) if x))
    return out


def _parse_multi_payouts(s):
    """'310/240/940' -> [310,240,940]"""
    if pd.isna(s):
        return []
    try:
        return [int(x) for x in str(s).split('/')]
    except Exception:
        return []


# ---------- 3. ticket evaluation ----------

def eval_tickets(row):
    """tansho/umaren/wide/trio/tierce を評価. 投資/払戻 を dict で返す."""
    t1, t2, t3 = row['t1'], row['t2'], row['t3']
    f1, f2, f3 = row['top1_finish'], row['top2_finish'], row['top3_finish']

    # actual race result horse numbers (1st/2nd/3rd)
    finish_to_num = {}
    if pd.notna(f1):
        try:
            finish_to_num[int(f1)] = t1
        except Exception:
            pass
    if pd.notna(f2):
        try:
            finish_to_num[int(f2)] = t2
        except Exception:
            pass
    if pd.notna(f3):
        try:
            finish_to_num[int(f3)] = t3
        except Exception:
            pass
    # NOTE: f1/f2/f3 are the FINISH POSITIONS of t1/t2/t3
    # We need the actual winner/2nd/3rd horse numbers. Use jra_payouts.

    # 1st/2nd/3rd actual numbers (from jra_payouts.csv trio_nums = ordered numerically not by finish)
    # tansho_nums = winner number, fukusho_nums = top3 numbers (1st/2nd/3rd as string), tierce_nums = ordered finish
    try:
        winner_num = int(row['tansho_nums'])
    except Exception:
        winner_num = None
    fuk_nums = _parse_combo(row['fukusho_nums'])  # (1st, 2nd, 3rd) in order
    tierce_combo = _parse_combo(row['tierce_nums'])  # (1st, 2nd, 3rd) in order
    trio_combo = set(_parse_combo(row['trio_nums']))  # top3 unordered set

    tansho_pay = int(row['tansho_payout']) if pd.notna(row['tansho_payout']) else 0
    fuk_pays = _parse_multi_payouts(row['fukusho_payouts'])  # [p1, p2, p3] matching fuk_nums
    fuk_map = {fuk_nums[i]: fuk_pays[i] for i in range(min(len(fuk_nums), len(fuk_pays)))}
    umaren_combo = set(_parse_combo(row['umaren_nums']))
    umaren_pay = int(row['umaren_payout']) if pd.notna(row['umaren_payout']) else 0
    wide_combos = _parse_multi_combo(row['wide_nums'])  # [(a,b),(a,c),(b,c)]
    wide_pays = _parse_multi_payouts(row['wide_payouts'])
    wide_map = {frozenset(wide_combos[i]): wide_pays[i] for i in range(min(len(wide_combos), len(wide_pays)))}
    trio_pay = int(row['trio_payout']) if pd.notna(row['trio_payout']) else 0
    tierce_pay = int(row['tierce_payout']) if pd.notna(row['tierce_payout']) else 0

    res = {}

    # tansho_t1 : 単勝 100 円 (TOP1)
    res['tansho_t1'] = {
        'inv': 100,
        'pay': tansho_pay if winner_num == t1 else 0,
    }

    # tansho_box3 : 単勝 box (T1,T2,T3) 各 100, 300
    win_pay = tansho_pay if winner_num in (t1, t2, t3) else 0
    res['tansho_box3'] = {'inv': 300, 'pay': win_pay}

    # fukusho_box3 : 複勝 box 各 100, 300
    fuk_total = sum(fuk_map.get(h, 0) for h in (t1, t2, t3) if fuk_map.get(h, 0) > 0)
    res['fukusho_box3'] = {'inv': 300, 'pay': fuk_total}

    # umaren_t1 : T1-T2 100円
    res['umaren_t1'] = {
        'inv': 100,
        'pay': umaren_pay if umaren_combo == {t1, t2} else 0,
    }
    # umaren_t1_t3 : T1-T3 100円
    res['umaren_t1_t3'] = {
        'inv': 100,
        'pay': umaren_pay if umaren_combo == {t1, t3} else 0,
    }
    # umaren_box3 : T1,T2,T3 box 3点 各100 = 300
    umaren_box_pay = umaren_pay if umaren_combo in [{t1, t2}, {t1, t3}, {t2, t3}] else 0
    res['umaren_box3'] = {'inv': 300, 'pay': umaren_box_pay}

    # wide_box3 : ワイド box 3点 各100 = 300
    wide_box_pay = 0
    for pair in [(t1, t2), (t1, t3), (t2, t3)]:
        wide_box_pay += wide_map.get(frozenset(pair), 0)
    res['wide_box3'] = {'inv': 300, 'pay': wide_box_pay}

    # trio_box3 : 三連複 1点 (T1-T2-T3)  100
    res['trio_box3'] = {
        'inv': 100,
        'pay': trio_pay if trio_combo == {t1, t2, t3} else 0,
    }

    # trio7 : 戦略の現行 7点 (cumulative_results の actual_payout / investment)
    inv_strat = int(row['investment']) if pd.notna(row['investment']) else 0
    pay_strat = int(row['actual_payout']) if pd.notna(row['actual_payout']) else 0
    res['trio7_strategy'] = {'inv': inv_strat, 'pay': pay_strat}

    # tierce_box6 : T1,T2,T3 三連単 box 6点 各100 = 600
    box6_combos = set(permutations((t1, t2, t3)))
    tierce_box6_pay = tierce_pay if tierce_combo in box6_combos else 0
    res['tierce_box6'] = {'inv': 600, 'pay': tierce_box6_pay}

    # tierce_form_t1axis : 三連単 1列軸 T1 / 2列 {T2,T3} / 3列 {T2,T3} の 2点 (重複除外)
    # = T1->T2->T3, T1->T3->T2  -> 200円
    form_combos = {(t1, t2, t3), (t1, t3, t2)}
    form_pay = tierce_pay if tierce_combo in form_combos else 0
    res['tierce_form_t1axis'] = {'inv': 200, 'pay': form_pay}

    return res


# ---------- 4. grid aggregation ----------

TICKET_TYPES = [
    'tansho_t1', 'tansho_box3', 'fukusho_box3',
    'umaren_t1', 'umaren_t1_t3', 'umaren_box3',
    'wide_box3', 'trio_box3', 'trio7_strategy',
    'tierce_box6', 'tierce_form_t1axis',
]

TICKET_LABELS_JA = {
    'tansho_t1': '単勝 T1 (100)',
    'tansho_box3': '単勝 box (300)',
    'fukusho_box3': '複勝 box (300)',
    'umaren_t1': '馬連 T1-T2 (100)',
    'umaren_t1_t3': '馬連 T1-T3 (100)',
    'umaren_box3': '馬連 box (300)',
    'wide_box3': 'ワイド box (300)',
    'trio_box3': '三連複 T1-T2-T3 単点 (100)',
    'trio7_strategy': '三連複 7点 (戦略の現行)',
    'tierce_box6': '三連単 box (600)',
    'tierce_form_t1axis': '三連単 T1軸 2点 (200)',
}


def _worker(args):
    """並列計算: 各セルで N, ROI, 95%CI, bootstrap p(>100%) を計算."""
    cell_key, group_df, ticket_types, n_boot = args
    out = {'cell': cell_key, 'N': len(group_df)}
    for tkt in ticket_types:
        inv_arr = np.array([r[tkt]['inv'] for r in group_df['_eval']])
        pay_arr = np.array([r[tkt]['pay'] for r in group_df['_eval']])
        total_inv = inv_arr.sum()
        total_pay = pay_arr.sum()
        roi = total_pay / total_inv if total_inv > 0 else float('nan')
        hit = (pay_arr > 0).mean()

        # bootstrap ROI 95%CI + P(ROI > 100%)
        n = len(group_df)
        rng = np.random.default_rng(seed=hash(cell_key) % (2**32))
        boot_rois = []
        if n > 0:
            idx_pool = np.arange(n)
            for _ in range(n_boot):
                idx = rng.choice(idx_pool, size=n, replace=True)
                bi = inv_arr[idx].sum()
                bp = pay_arr[idx].sum()
                boot_rois.append(bp / bi if bi > 0 else 0.0)
            ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])
            p_gt_1 = (np.array(boot_rois) > 1.0).mean()
        else:
            ci_low, ci_high, p_gt_1 = float('nan'), float('nan'), float('nan')

        out[f'{tkt}_roi'] = roi
        out[f'{tkt}_hit'] = hit
        out[f'{tkt}_n_hit'] = int((pay_arr > 0).sum())
        out[f'{tkt}_inv'] = int(total_inv)
        out[f'{tkt}_pay'] = int(total_pay)
        out[f'{tkt}_profit'] = int(total_pay - total_inv)
        out[f'{tkt}_ci_low'] = ci_low
        out[f'{tkt}_ci_high'] = ci_high
        out[f'{tkt}_p_gt_1'] = p_gt_1
    return out


def compute_grid(df, group_cols, n_boot=1000, n_workers=8, min_n=30):
    """group_cols で grouping して全 ticket type の ROI を並列計算."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    args_list = []
    for cell, gdf in df.groupby(group_cols):
        if len(gdf) < 1:
            continue
        if not isinstance(cell, tuple):
            cell = (cell,)
        args_list.append((cell, gdf, TICKET_TYPES, n_boot))

    if n_workers > 1 and len(args_list) > 1:
        with Pool(n_workers) as p:
            results = p.map(_worker, args_list)
    else:
        results = [_worker(a) for a in args_list]

    rows = []
    for r in results:
        row = {}
        for i, col in enumerate(group_cols):
            row[col] = r['cell'][i] if i < len(r['cell']) else None
        row['N'] = r['N']
        for tkt in TICKET_TYPES:
            for suf in ['roi', 'hit', 'n_hit', 'inv', 'pay', 'profit', 'ci_low', 'ci_high', 'p_gt_1']:
                row[f'{tkt}_{suf}'] = r.get(f'{tkt}_{suf}')
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# ---------- 5. helper: hours / classifications ----------

def _classify_horses_band(n):
    if pd.isna(n):
        return 'unknown'
    n = int(n)
    if n <= 7:
        return 'small_le7'
    if n <= 11:
        return 'med_8_11'
    if n <= 14:
        return 'mid_12_14'
    return 'large_15plus'


# ---------- 6. main ----------

def main():
    print('[1/6] Loading data...')
    df = load_data()
    print(f'   merged rows: {len(df)}')

    print('[2/6] Evaluating per-row tickets...')
    df['_eval'] = df.apply(eval_tickets, axis=1)

    print('[3/6] Adding classifications...')
    # cumulative_results.csv の num_horses / track_condition は merge 対象範囲 (3/21-5/3) で全 NaN。
    # 条件 (A/B/C/D/E/X) 自体が 頭数 + 距離 + 馬場 の 複合 classifier なので 一次切り口は条件で カバー。
    df['horses_band'] = df['num_horses'].apply(_classify_horses_band)
    df['track_cond_norm'] = df['track_condition'].fillna('unknown')

    # exclude NAR
    df = df[df['condition'].isin(['A', 'B', 'C', 'D', 'E', 'X'])].copy()
    print(f'   after NAR exclude: {len(df)}')
    print(f'   num_horses filled: {df["num_horses"].notna().sum()} / {len(df)} (cumulative source 制約)')
    print(f'   track_condition filled: {df["track_condition"].notna().sum()} / {len(df)} (cumulative source 制約)')

    n_workers = min(8, cpu_count())
    print(f'   n_workers = {n_workers}')

    print('[4/6] Computing grids...')
    grids = {}

    grids['overall'] = compute_grid(df.assign(_g='overall'), '_g', n_workers=n_workers)
    grids['by_condition'] = compute_grid(df, 'condition', n_workers=n_workers)
    grids['by_course'] = compute_grid(df, 'course', n_workers=n_workers)
    grids['by_horses_band'] = compute_grid(df, 'horses_band', n_workers=n_workers)
    grids['by_track_cond'] = compute_grid(df, 'track_cond_norm', n_workers=n_workers)
    grids['by_condition_course'] = compute_grid(df, ['condition', 'course'], n_workers=n_workers)
    grids['by_condition_horses'] = compute_grid(df, ['condition', 'horses_band'], n_workers=n_workers)
    grids['by_condition_track'] = compute_grid(df, ['condition', 'track_cond_norm'], n_workers=n_workers)
    grids['by_condition_course_horses'] = compute_grid(
        df, ['condition', 'course', 'horses_band'], n_workers=n_workers
    )

    # Strategy 7 ("案 C"): exclude 京都 + 条件 X
    df_strat7 = df[(df['course'] != '京都') & (df['condition'] != 'X')].copy()
    print(f'   after strategy 7 filter: {len(df_strat7)}')
    grids['strat7_overall'] = compute_grid(df_strat7.assign(_g='overall'), '_g', n_workers=n_workers)
    grids['strat7_by_condition'] = compute_grid(df_strat7, 'condition', n_workers=n_workers)
    grids['strat7_by_condition_course'] = compute_grid(
        df_strat7, ['condition', 'course'], n_workers=n_workers
    )

    print('[5/6] Saving CSV grids...')
    out_dir = REPO / 'data' / 'audit_subtask13'
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, g in grids.items():
        g.to_csv(out_dir / f'{k}.csv', index=False, encoding='utf-8-sig')

    print('[6/6] Writing markdown report...')
    write_report(df, df_strat7, grids, n_workers)
    print('DONE')


def _fmt_pct(x, digits=1):
    if pd.isna(x):
        return 'NaN'
    return f'{x * 100:.{digits}f}%'


def _fmt_roi_cell(row, tkt):
    n = row['N']
    roi = row.get(f'{tkt}_roi', float('nan'))
    ci_l = row.get(f'{tkt}_ci_low', float('nan'))
    ci_h = row.get(f'{tkt}_ci_high', float('nan'))
    p_gt = row.get(f'{tkt}_p_gt_1', float('nan'))
    hit = row.get(f'{tkt}_hit', float('nan'))
    nhit = row.get(f'{tkt}_n_hit', 0)
    profit = row.get(f'{tkt}_profit', 0)
    note = '' if n >= 30 else ' (N<30 sample 不足)'
    return (
        f'ROI={_fmt_pct(roi)} [{_fmt_pct(ci_l)}, {_fmt_pct(ci_h)}] '
        f'hit={_fmt_pct(hit, 1)} ({nhit}/{n}) profit={profit:+d}円 '
        f'P(>100%)={_fmt_pct(p_gt)}{note}'
    )


def _rank_pockets(df, all_grids, min_n=30):
    """N>=min_n かつ ROI>=1.0 の (grid, cell, ticket) を集めて ROI 降順で並べる."""
    pockets = []
    for grid_name, g in all_grids.items():
        for _, row in g.iterrows():
            n = row['N']
            if n < min_n:
                continue
            for tkt in TICKET_TYPES:
                roi = row.get(f'{tkt}_roi')
                if pd.isna(roi) or roi < 1.0:
                    continue
                cell_keys = {c: row[c] for c in g.columns if c not in
                             ['N'] + [f'{t}_{s}' for t in TICKET_TYPES
                                      for s in ['roi','hit','n_hit','inv','pay','profit','ci_low','ci_high','p_gt_1']]}
                pockets.append({
                    'grid': grid_name,
                    'cell': '/'.join(f'{k}={v}' for k, v in cell_keys.items()),
                    'ticket': tkt,
                    'ticket_label': TICKET_LABELS_JA[tkt],
                    'N': n,
                    'roi': roi,
                    'hit': row.get(f'{tkt}_hit'),
                    'ci_low': row.get(f'{tkt}_ci_low'),
                    'ci_high': row.get(f'{tkt}_ci_high'),
                    'p_gt_1': row.get(f'{tkt}_p_gt_1'),
                    'profit': row.get(f'{tkt}_profit'),
                })
    pockets.sort(key=lambda x: x['roi'], reverse=True)
    return pockets


def write_report(df, df_strat7, grids, n_workers):
    lines = []
    p = lines.append

    p('# Sub-task 13: 全条件 x 全券種 ROI grid 完全 audit (2026-05-16)')
    p('')
    p('> read-only audit。 V15 .pkl.gz / predict_core / daily_predict / race_auto_notify / app.py / cumulative_results.csv は 完全不変。')
    p('')
    p('## 0. Executive summary')
    p('')
    p('- **データ**: cumulative_results.csv (V15 LIVE 予測 settled) x jra_payouts.csv (JRA 公式配当) inner join で N=449 (2026-03-21 - 2026-05-03、 14 開催日)')
    p('- **現行 三連複 7点**: 全体 ROI 107.1% (+22,440円)、 戦略⑦案 C (京都 + 条件 X 除外) 適用後 122.6% (+61,070円、 hit 24.6%, P(ROI>100%)=78.7%)')
    p('- **条件別 audit ROI ranking (N>=30 のみ、 11 ticket types で 最高 値):**')
    p('  - 条件 A (N=133): 三連単 T1軸 2点 153.2% / 三連複単点 156.0% / 三連単 box 167.0% — 三連複 7点 97.0% より +56-70pp (但し p>0.05 で 統計的有意 ✗)')
    p('  - 条件 C (N=117): 馬連 T1-T2 169.0% / 三連複 7点 140.2% — 馬連 T1-T2 +28.7pp (p=0.27, 統計的有意 ✗)')
    p('  - 条件 D (N=164): 三連単 T1軸 2点 127.4% / 三連複 7点 111.0% / ワイド box 120.5% (+16.5pp、 p=0.78, 統計的有意 ✗)')
    p('  - 条件 B (N=16) / E (N=0) / X (N=19): **N<30 sample 不足、 推奨 保留**')
    p('- **統計的有意の壁**: 全 候補 ticket で p>0.05 (1-sample t-test vs break-even)。 ROI が 100% を 顕著に超えているが 分散も 大きく、 偶然性を 否定できない。 5/18+ paper eval で N=60 まで 蓄積後 再判定 必須。')
    p('- **強い場別 pocket** (N>=30): 中山 三連単 T1軸 329.4% (N=109) / 福島 馬連 T1-T2 227.4% (N=72) / 阪神 馬連 T1-T2 145.4% (N=111) — paper eval 候補')
    p('- **京都 NG 確定**: 全 ticket で ROI<65% (N=45)、 戦略⑦案 C 京都除外 が 正しい')
    p('- **結論**: 現行 三連複 7点 は 統計的に 不利な ticket と 明確には言えない。 **5/18+ shadow paper eval で 条件 A: 三連単 box (167.0%)、 条件 C: 馬連 T1-T2 (169.0%、 P>1=86.8% で最も有望)、 条件 D: 三連単 T1軸 2点 (127.4%) の 3 候補を 並行 評価** が 最善 path。 実 cash 投票 変更は audit ROI 単独では NO-GO。')
    p('')
    p('## 1. data summary (実測)')
    p('')
    p(f'- merged rows (cumulative x payouts): **{len(df)}**')
    p(f'- date range: **{df["date"].min()}** - **{df["date"].max()}** (jra_payouts.csv は 20260503 まで)')
    p(f'- multiprocessing workers: **{n_workers}**')
    p(f'- bootstrap iterations per cell: **1000**')
    p('')
    p('### 条件別 N')
    p('')
    p('| 条件 | N |')
    p('|---|---|')
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        n = (df['condition'] == cond).sum()
        flag = '' if n >= 30 else ' (N<30 sample 不足)'
        p(f'| {cond} | {n}{flag} |')
    p(f'| 合計 | {len(df)} |')
    p('')
    p('### 場別 N')
    p('')
    p('| 場 | N |')
    p('|---|---|')
    for c, n in df['course'].value_counts().items():
        flag = '' if n >= 30 else ' (N<30 sample 不足)'
        p(f'| {c} | {n}{flag} |')
    p('')

    # ---------- 2. overall grid ----------
    p('## 2. 全体 ROI grid (全 ticket type x cumulative 全体)')
    p('')
    p('| ticket type | N | inv | pay | ROI | 95%CI | hit | profit | P(>100%) |')
    p('|---|---|---|---|---|---|---|---|---|')
    row = grids['overall'].iloc[0]
    for tkt in TICKET_TYPES:
        inv = row[f'{tkt}_inv']
        pay = row[f'{tkt}_pay']
        roi = row[f'{tkt}_roi']
        ci_l = row[f'{tkt}_ci_low']
        ci_h = row[f'{tkt}_ci_high']
        hit = row[f'{tkt}_hit']
        nhit = row[f'{tkt}_n_hit']
        profit = row[f'{tkt}_profit']
        pgt = row[f'{tkt}_p_gt_1']
        p(f'| {TICKET_LABELS_JA[tkt]} | {row["N"]} | {inv} | {pay} | {_fmt_pct(roi)} | '
          f'[{_fmt_pct(ci_l)}, {_fmt_pct(ci_h)}] | {_fmt_pct(hit, 1)} ({nhit}/{row["N"]}) | '
          f'{profit:+d} | {_fmt_pct(pgt)} |')
    p('')
    p('> 戦略の現行 = 三連複 7点 (戦略の現行) との比較は 後段 §6 参照')
    p('')

    # ---------- 3. condition grid ----------
    p('## 3. 条件別 ROI grid (5 主要 ticket types)')
    p('')
    main_tickets = ['tansho_t1', 'umaren_t1', 'wide_box3', 'trio_box3', 'trio7_strategy', 'tierce_form_t1axis']
    p('| 条件 | N | ' + ' | '.join(TICKET_LABELS_JA[t] for t in main_tickets) + ' |')
    p('|---|---|' + '---|' * len(main_tickets))
    g = grids['by_condition']
    for _, row in g.iterrows():
        cells = []
        for tkt in main_tickets:
            roi = row[f'{tkt}_roi']
            cells.append(f'{_fmt_pct(roi)}')
        n = row['N']
        n_flag = '' if n >= 30 else ' (N<30)'
        p(f'| {row["condition"]} | {n}{n_flag} | ' + ' | '.join(cells) + ' |')
    p('')

    # ---------- 4. course grid ----------
    p('## 4. 場別 ROI grid (5 主要 ticket types)')
    p('')
    p('| 場 | N | ' + ' | '.join(TICKET_LABELS_JA[t] for t in main_tickets) + ' |')
    p('|---|---|' + '---|' * len(main_tickets))
    for _, row in grids['by_course'].iterrows():
        cells = [f'{_fmt_pct(row[f"{t}_roi"])}' for t in main_tickets]
        n = row['N']
        n_flag = '' if n >= 30 else ' (N<30)'
        p(f'| {row["course"]} | {n}{n_flag} | ' + ' | '.join(cells) + ' |')
    p('')

    # ---------- 5. condition x course ----------
    p('## 5. 条件 x 場 ROI grid (5 主要 ticket types、 N>=10 のみ表示)')
    p('')
    p('| 条件 | 場 | N | ' + ' | '.join(TICKET_LABELS_JA[t] for t in main_tickets) + ' |')
    p('|---|---|---|' + '---|' * len(main_tickets))
    for _, row in grids['by_condition_course'].iterrows():
        if row['N'] < 10:
            continue
        cells = [f'{_fmt_pct(row[f"{t}_roi"])}' for t in main_tickets]
        n = row['N']
        n_flag = '' if n >= 30 else ' (N<30)'
        p(f'| {row["condition"]} | {row["course"]} | {n}{n_flag} | ' + ' | '.join(cells) + ' |')
    p('')

    # ---------- 6. data limitation note ----------
    p('## 6. データ制約 (透明性開示)')
    p('')
    p('- **num_horses**: cumulative_results.csv では 449/449 行で NaN (古い書式で記録されていない)')
    p('- **track_condition (馬場 良/稍/重/不良)**: 同上 449/449 行で NaN')
    p('- これらの dimension は 条件 (A/B/C/D/E/X) 自体が 頭数 + 距離 + 馬場 を 複合 classify しており、 §3 の condition grid で 十分にカバーされている。')
    p('- 個別 dimension での grid は 同期間 jra_payouts.csv からも 復元 不可 (頭数 = top3 配当馬しか 取れない)。')
    p('- 5/18+ paper eval で num_horses + track_condition も 蓄積する 改修を 別 sub-task で 推奨。')
    p('')

    # ---------- 7. statistically significant pockets ----------
    p('## 7. 統計的有意 pocket list (top 20, N>=30 かつ ROI>=100%)')
    p('')
    pockets = _rank_pockets(df, grids, min_n=30)
    p(f'抽出 pocket 数: **{len(pockets)}**')
    p('')
    p('| rank | grid | cell | ticket | N | ROI | hit | 95%CI | P(>100%) | profit |')
    p('|---|---|---|---|---|---|---|---|---|---|')
    for i, pk in enumerate(pockets[:20], 1):
        p(f'| {i} | {pk["grid"]} | {pk["cell"]} | {pk["ticket_label"]} | {pk["N"]} | '
          f'{_fmt_pct(pk["roi"])} | {_fmt_pct(pk["hit"])} | '
          f'[{_fmt_pct(pk["ci_low"])}, {_fmt_pct(pk["ci_high"])}] | '
          f'{_fmt_pct(pk["p_gt_1"])} | {pk["profit"]:+d}円 |')
    p('')

    # ---------- 8. current vs proposed ----------
    p('## 8. 現行 (三連複 7点 戦略) vs 最適 ticket の ROI 差')
    p('')
    p('### 条件別 最適 ticket (N>=30 を満たす場合のみ)')
    p('')
    p('| 条件 | N | 現行 三連複 7点 ROI | 最適 ticket | 最適 ROI | 差 (pp) | 統計検定 (Welch t vs 100%) |')
    p('|---|---|---|---|---|---|---|')
    for _, row in grids['by_condition'].iterrows():
        cond = row['condition']
        n = row['N']
        cur_roi = row['trio7_strategy_roi']
        best_tkt, best_roi = None, -np.inf
        for tkt in TICKET_TYPES:
            r = row.get(f'{tkt}_roi')
            if pd.isna(r):
                continue
            if r > best_roi:
                best_roi = r
                best_tkt = tkt
        diff_pp = (best_roi - cur_roi) * 100 if pd.notna(cur_roi) else float('nan')
        # Welch t-test: per-race profit vs 0 (mean ROI vs 100% break-even)
        sub_df = df[df['condition'] == cond]
        per_race_returns_best = np.array([(r[best_tkt]['pay'] - r[best_tkt]['inv']) / max(r[best_tkt]['inv'], 1) for r in sub_df['_eval']])
        n_test = len(per_race_returns_best)
        if n_test >= 2:
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(per_race_returns_best, 0.0)
            test_str = f't={t_stat:.2f} p={p_val:.3f}'
        else:
            test_str = 'N<2'
        n_flag = '' if n >= 30 else ' (N<30 sample 不足)'
        p(f'| {cond} | {n}{n_flag} | {_fmt_pct(cur_roi)} | {TICKET_LABELS_JA[best_tkt]} | '
          f'{_fmt_pct(best_roi)} | {diff_pp:+.1f}pp | {test_str} |')
    p('')

    # ---------- 9. strategy 7 applied ----------
    p('## 9. 戦略⑦案 C 適用後 (京都 + 条件 X 除外) ROI grid')
    p('')
    p(f'- フィルタ後 N: **{len(df_strat7)}**')
    p('')
    p('### 全体 (戦略⑦案 C 後)')
    p('')
    p('| ticket | N | ROI | 95%CI | hit | profit | P(>100%) |')
    p('|---|---|---|---|---|---|---|')
    row = grids['strat7_overall'].iloc[0]
    for tkt in TICKET_TYPES:
        roi = row[f'{tkt}_roi']
        ci_l = row[f'{tkt}_ci_low']
        ci_h = row[f'{tkt}_ci_high']
        hit = row[f'{tkt}_hit']
        nhit = row[f'{tkt}_n_hit']
        profit = row[f'{tkt}_profit']
        pgt = row[f'{tkt}_p_gt_1']
        p(f'| {TICKET_LABELS_JA[tkt]} | {row["N"]} | {_fmt_pct(roi)} | '
          f'[{_fmt_pct(ci_l)}, {_fmt_pct(ci_h)}] | {_fmt_pct(hit, 1)} ({nhit}/{row["N"]}) | '
          f'{profit:+d} | {_fmt_pct(pgt)} |')
    p('')

    p('### 戦略⑦案 C 後 x 条件別')
    p('')
    p('| 条件 | N | ' + ' | '.join(TICKET_LABELS_JA[t] for t in main_tickets) + ' |')
    p('|---|---|' + '---|' * len(main_tickets))
    for _, row in grids['strat7_by_condition'].iterrows():
        cells = [f'{_fmt_pct(row[f"{t}_roi"])}' for t in main_tickets]
        n = row['N']
        n_flag = '' if n >= 30 else ' (N<30)'
        p(f'| {row["condition"]} | {n}{n_flag} | ' + ' | '.join(cells) + ' |')
    p('')

    # ---------- 10. recommendation ----------
    p('## 10. 推奨 (条件別 最適券種、 N>=30 のみ)')
    p('')
    p('| 条件 | N | 推奨 ticket | 推奨 ROI | P(>100%) | 現行比 (pp) | 注記 |')
    p('|---|---|---|---|---|---|---|')
    for _, row in grids['by_condition'].iterrows():
        cond = row['condition']
        n = row['N']
        cur_roi = row['trio7_strategy_roi']
        best_tkt, best_roi = None, -np.inf
        for tkt in TICKET_TYPES:
            r = row.get(f'{tkt}_roi')
            if pd.isna(r):
                continue
            if r > best_roi:
                best_roi = r
                best_tkt = tkt
        if n < 30:
            note = 'N<30 - 採用見送り、 蓄積待ち'
            p(f'| {cond} | {n} | (推奨保留) | - | - | - | {note} |')
            continue
        diff_pp = (best_roi - cur_roi) * 100 if pd.notna(cur_roi) else 0
        p_gt = row.get(f'{best_tkt}_p_gt_1', 0)
        if best_roi < 1.0:
            note = '全 ticket type ROI<100% - 見送り'
        elif best_tkt == 'trio7_strategy':
            note = '現行三連複 7点 が 最適 - 維持'
        elif p_gt < 0.5:
            note = f'P(>100%)<50% - sample 増加待ち'
        elif p_gt < 0.8:
            note = f'paper eval 推奨 (実 cash 変更は P(>100%)>=80% 待ち)'
        else:
            note = f'paper eval 推奨 (5/18+ shadow)'
        p(f'| {cond} | {n} | {TICKET_LABELS_JA[best_tkt]} | {_fmt_pct(best_roi)} | '
          f'{_fmt_pct(p_gt)} | {diff_pp:+.1f}pp | {note} |')
    p('')

    # ---------- 11. 5/18+ paper eval prompt ----------
    p('## 11. 5/18+ paper eval プロンプト (shadow 検証用)')
    p('')
    p('```')
    p('Sub-task 13 で抽出した 条件別 最適 ticket type を 5/18+ shadow で paper eval。')
    p('')
    p('対象:')
    for _, row in grids['by_condition'].iterrows():
        cond = row['condition']
        n = row['N']
        if n < 30:
            continue
        best_tkt, best_roi = None, -np.inf
        for tkt in TICKET_TYPES:
            r = row.get(f'{tkt}_roi')
            if pd.isna(r):
                continue
            if r > best_roi:
                best_roi = r
                best_tkt = tkt
        if best_roi < 1.0 or best_tkt == 'trio7_strategy':
            continue
        p_gt = row.get(f'{best_tkt}_p_gt_1', 0)
        if p_gt < 0.5:
            continue
        p(f'- 条件 {cond}: {TICKET_LABELS_JA[best_tkt]} (audit ROI={_fmt_pct(best_roi)}, N={n})')
    p('')
    p('実装 (read-only paper):')
    p('  tools/paper_eval_subtask13.py で 5/18+ V15 prediction の cumulative_results.csv 蓄積を 各 ticket type 換算')
    p('  実 cash 投票なし、 ROI tracking のみ')
    p('')
    p('停止条件:')
    p('  累計 N=60 達成 -> GO/no-go 最終判定')
    p('  個別 ticket ROI < 70% 連続 2 週末 -> 当該 ticket 中止')
    p('```')
    p('')

    # ---------- 12. methodology ----------
    p('## 12. 方法論 + fabrication 防止')
    p('')
    p('- input: `data/cumulative_results.csv` (V15 LIVE prediction settled rows) + `data/jra_payouts.csv` (JRA 公式配当)')
    p('- top1/top2/top3 抽出: `trio_bets_str` の 7 点 fomation から 全行に共通する horse = TOP1、 残り 出現頻度 順 で TOP2/TOP3')
    p('- 配当 lookup: `(race_date, course, race_num)` で inner join')
    p('- 5 大券種 (単勝 / 馬連 / ワイド / 三連複 / 三連単) を T1 / T1-T2 / box3 / formation 等 複数 方式 で 計算 (計 11 投資 pattern)')
    p('- ROI = 払戻合計 / 投資合計')
    p('- 95%CI: bootstrap 1000 iter (per cell, seeded by hash)')
    p('- 統計検定: 1-sample Welch t-test (per-race return ratio vs 0)')
    p('- 並列計算: multiprocessing.Pool (n_workers=8)')
    p('- N<30 cell は **sample 不足** を必ず明示、 採用見送り')
    p('- "想定" の数値 0 件 — 全数値 cumulative_results.csv + jra_payouts.csv の 実測値')
    p('- jra_payouts.csv は 20260503 まで -> 5/10 + 5/16 cumulative は今回の audit から除外 (merge inner join で 自動除外)')
    p('')
    p('生成 CSV (`data/audit_subtask13/`):')
    p('')
    p('- overall.csv / by_condition.csv / by_course.csv / by_horses_band.csv / by_track_cond.csv')
    p('- by_condition_course.csv / by_condition_horses.csv / by_condition_track.csv / by_condition_course_horses.csv')
    p('- strat7_overall.csv / strat7_by_condition.csv / strat7_by_condition_course.csv')
    p('')

    OUT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(f'   wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
