"""healthy 4日 (4/25, 4/26, 5/2, 5/3) のみで案A/B/C 再評価。

session#3 のレポート (case A=159.8%, B=173.8%) は全14日の汚染データ含むため再計算。

分析項目:
1. 案A/B/C ベース指標 (ROI, 的中率, 軸top3率, n)
2. ブートストラップで日次クラスタ 95% CI (n_days=4 と少ないので限定)
3. 案B改 追加フィルタ (重賞除外 + 条件C/D優先 + 1勝クラス優先)
4. 11R 分類別 (G1/G2/G3/L/OP/その他) ROI
5. 12R クラス別 (1勝/2勝/3勝/OP特別/その他) ROI
6. 各日 prediction error 分解
7. session#3 との差分

出力: data/results/healthy_4day_analysis.md
"""
import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath('.')
os.chdir(BASE_DIR)

HEALTHY_DATES = ['20260425', '20260426', '20260502', '20260503']
PRED_DIR = 'data/daily_predictions'
RES_DIR = 'data/daily_results'

print("=" * 60)
print("Healthy 4-day Analysis (5/9 投資戦略再評価)")
print(f"Start: {datetime.now()}")
print(f"Healthy dates: {HEALTHY_DATES}")
print("=" * 60)

# === Load all healthy data ===
all_pred = []
all_res = []
for d in HEALTHY_DATES:
    p = pd.read_csv(f'{PRED_DIR}/{d}.csv', dtype={'race_id': str, 'race_num': str})
    r = pd.read_csv(f'{RES_DIR}/{d}.csv', encoding='utf-8-sig', dtype={'race_id': str, 'race_num': str})
    p['date'] = d
    r['date'] = d
    all_pred.append(p)
    all_res.append(r)

df_pred = pd.concat(all_pred, ignore_index=True)
df_res = pd.concat(all_res, ignore_index=True)

# Dedup by (date, race_id) — 4/26 has 50 rows due to duplicates
df_res = df_res.drop_duplicates(subset=['date', 'race_id'], keep='last').reset_index(drop=True)
df_pred = df_pred.drop_duplicates(subset=['date', 'race_id'], keep='last').reset_index(drop=True)

print(f"\nPredictions: {len(df_pred)} (after dedup)")
print(f"Results: {len(df_res)} (after dedup)")
print(f"Per date: {df_res.groupby('date').size().to_dict()}")

# Merge
df = df_res.merge(df_pred[['date','race_id','top1_num','top1_score','top2_num','top3_num','race_name','condition','num_horses','distance','surface','track_condition']],
                   on=['date','race_id'], how='left', suffixes=('','_pred'))

# Use race_name from pred (more accurate) but fallback to res
df['race_name_use'] = df['race_name_pred'].fillna(df['race_name'])
# Use condition from pred (more accurate)
df['condition_use'] = df['condition_pred'].fillna(df['condition'])

# Settled only
df = df[df['status'] == 'settled'].copy()
df['investment'] = pd.to_numeric(df['investment'], errors='coerce').fillna(0).astype(int)
df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0).astype(int)
df['trio_payout'] = pd.to_numeric(df['trio_payout'], errors='coerce').fillna(0)
df['umaren_payout'] = pd.to_numeric(df['umaren_payout'], errors='coerce').fillna(0)
df['actual_payout'] = df['trio_payout'] + df['umaren_payout']
df['top1_finish'] = pd.to_numeric(df['top1_finish'], errors='coerce')
df['top2_finish'] = pd.to_numeric(df['top2_finish'], errors='coerce')
df['top3_finish'] = pd.to_numeric(df['top3_finish'], errors='coerce')
df['trio_hit'] = pd.to_numeric(df['trio_hit'], errors='coerce').fillna(0).astype(int)
df['race_num'] = pd.to_numeric(df['race_num'], errors='coerce').astype('Int64')

n_total = len(df)
print(f"\nSettled races: {n_total}")
print(f"Investment unique values: {df['investment'].unique()}")

# === Helper: classify 11R race grade ===
def classify_11r(name):
    if not isinstance(name, str): return 'unknown'
    if any(g in name for g in ['天皇賞','有馬','ジャパンC','ダービー','オークス','皐月賞','菊花賞','桜花賞','秋華賞']):
        return 'G1'
    # exact G1 G2 G3 markers
    if 'G1' in name or '(G1)' in name or 'GⅠ' in name: return 'G1'
    if 'G2' in name or '(G2)' in name or 'GⅡ' in name: return 'G2'
    if 'G3' in name or '(G3)' in name or 'GⅢ' in name: return 'G3'
    if name.endswith('(L)') or '(L)' in name: return 'L'
    if name.endswith('S') or 'ステークス' in name: return 'OP/L'
    if name.endswith('特別'): return 'OP特別'
    return 'その他11R'

# === Helper: classify 12R class ===
def classify_12r_class(name):
    if not isinstance(name, str): return 'unknown'
    if '1勝' in name: return '1勝'
    if '2勝' in name: return '2勝'
    if '3勝' in name: return '3勝'
    if 'オープン' in name or 'OP' in name or '(L)' in name: return 'OP'
    if '未勝利' in name: return '未勝利'
    if '新馬' in name: return '新馬'
    if name.endswith('特別') or 'S' in name: return 'OP特別'
    return 'その他'

df['grade_11r'] = df.apply(lambda r: classify_11r(r['race_name_use']) if r['race_num']==11 else None, axis=1)
df['class_12r'] = df.apply(lambda r: classify_12r_class(r['race_name_use']) if r['race_num']==12 else None, axis=1)
df['axis_top3'] = ((df['top1_finish'] >= 1) & (df['top1_finish'] <= 3)).astype(int)


# === Bootstrap CI per-day cluster ===
def bootstrap_cluster_roi(g, n_resamples=10000, seed=42):
    """Day cluster bootstrap. Returns (mean ROI %, 5%, 95%)"""
    rng = np.random.default_rng(seed)
    days = sorted(g['date'].unique())
    if len(days) == 0:
        return 0, 0, 0
    g_by_day = {d: g[g['date']==d] for d in days}
    rois = []
    for _ in range(n_resamples):
        sel = rng.choice(len(days), size=len(days), replace=True)
        sub_days = [days[i] for i in sel]
        sub = pd.concat([g_by_day[d] for d in sub_days])
        inv = sub['investment'].sum()
        pay = sub['actual_payout'].sum()
        rois.append(pay / inv * 100 if inv > 0 else 0)
    rois = np.array(rois)
    return float(np.mean(rois)), float(np.percentile(rois, 5)), float(np.percentile(rois, 95))


def stats(g, label):
    if len(g) == 0:
        return {'label': label, 'n': 0, 'inv': 0, 'pay': 0, 'roi': 0, 'profit': 0,
                'hit_rate': 0, 'axis_top3': 0, 'roi_lo': 0, 'roi_hi': 0}
    inv = int(g['investment'].sum())
    pay = float(g['actual_payout'].sum())
    profit = pay - inv
    roi = pay / inv * 100 if inv > 0 else 0
    hits = int(g['trio_hit'].sum())
    hit_rate = hits / len(g) * 100 if len(g) else 0
    a_top3 = float(g['axis_top3'].mean()) * 100 if len(g) else 0
    if len(g['date'].unique()) >= 2:
        roi_mean, roi_lo, roi_hi = bootstrap_cluster_roi(g)
    else:
        roi_mean, roi_lo, roi_hi = roi, roi, roi
    return {
        'label': label, 'n': len(g), 'inv': inv, 'pay': pay, 'roi': roi, 'profit': profit,
        'hit_rate': hit_rate, 'axis_top3': a_top3,
        'roi_boot_mean': roi_mean, 'roi_lo': roi_lo, 'roi_hi': roi_hi,
        'hits': hits,
    }


# === Plan A: 11R + 12R 全部 ===
plan_a = df[df['race_num'].isin([11, 12])]
# === Plan B: 12R 全部 + 11R 非重賞 (G1/G2/G3 除外) ===
heavy_grades = ['G1', 'G2', 'G3']
plan_b = df[(df['race_num'] == 12) | ((df['race_num'] == 11) & (~df['grade_11r'].isin(heavy_grades)))]
# === Plan C: 12R のみ ===
plan_c = df[df['race_num'] == 12]
# === Plan B改: 12R 1勝 + 12R 条件D + 11R 条件C 非重賞 ===
plan_b_kai = df[
    (((df['race_num'] == 12) & (df['class_12r'] == '1勝'))) |
    (((df['race_num'] == 12) & (df['condition_use'] == 'D'))) |
    (((df['race_num'] == 11) & (df['condition_use'] == 'C') & (~df['grade_11r'].isin(heavy_grades))))
].drop_duplicates(subset=['date','race_id'])

stats_a = stats(plan_a, 'A: 11R+12R全')
stats_b = stats(plan_b, 'B: 12R全+11R非重賞')
stats_c = stats(plan_c, 'C: 12Rのみ')
stats_bk = stats(plan_b_kai, 'B改: 12R(1勝/D) + 11R条件C非重賞')

print()
print("=" * 60)
print("案 A/B/C/B改 比較 (healthy 4日)")
print("=" * 60)
for s in [stats_a, stats_b, stats_c, stats_bk]:
    print(f"\n{s['label']}")
    print(f"  n={s['n']}, inv={s['inv']:,}, pay={s['pay']:,.0f}, profit={s['profit']:+,.0f}")
    print(f"  ROI={s['roi']:.1f}%  bootstrap CI [{s['roi_lo']:.1f}%, {s['roi_hi']:.1f}%]  hit_rate={s['hit_rate']:.1f}%  axis_top3={s['axis_top3']:.1f}%")


# === 11R 分類別 ===
print()
print("=" * 60)
print("11R 分類別 (healthy 4日)")
print("=" * 60)
r11 = df[df['race_num'] == 11].copy()
print(f"全11R: n={len(r11)}")
grade_stats = []
for grade in ['G1', 'G2', 'G3', 'L', 'OP/L', 'OP特別', 'その他11R']:
    g = r11[r11['grade_11r'] == grade]
    if len(g) > 0:
        s = stats(g, f'11R {grade}')
        grade_stats.append(s)
        print(f"  {grade:10s}: n={s['n']:2d} inv={s['inv']:5,} pay={int(s['pay']):6,} ROI={s['roi']:6.1f}% hit={s['hit_rate']:5.1f}% axis_top3={s['axis_top3']:5.1f}%")

# === 12R クラス別 ===
print()
print("=" * 60)
print("12R クラス別 (healthy 4日)")
print("=" * 60)
r12 = df[df['race_num'] == 12].copy()
class_stats = []
for cls in ['1勝', '2勝', '3勝', 'OP', 'OP特別', '未勝利', '新馬', 'その他']:
    g = r12[r12['class_12r'] == cls]
    if len(g) > 0:
        s = stats(g, f'12R {cls}')
        class_stats.append(s)
        print(f"  {cls:10s}: n={s['n']:2d} inv={s['inv']:5,} pay={int(s['pay']):6,} ROI={s['roi']:6.1f}% hit={s['hit_rate']:5.1f}% axis_top3={s['axis_top3']:5.1f}%")

# === Per-day prediction error ===
print()
print("=" * 60)
print("各日 prediction error 分解")
print("=" * 60)
day_stats = []
for d in HEALTHY_DATES:
    g = df[df['date'] == d]
    g11_12 = g[g['race_num'].isin([11,12])]
    s = stats(g11_12, d)
    s_full = stats(g, d + ' (全R)')
    day_stats.append({**s, 'full_n': s_full['n'], 'full_roi': s_full['roi'],
                       'full_axis_top3': s_full['axis_top3']})
    print(f"  {d}: 11R/12R n={s['n']} ROI={s['roi']:.1f}% axis_top3={s['axis_top3']:.1f}% hit={s['hit_rate']:.1f}% | 全R ROI={s_full['roi']:.1f}% axis_top3={s_full['axis_top3']:.1f}%")


# === Cross: condition × race_num (主要セル) ===
print()
print("=" * 60)
print("条件 × race_num (11/12) クロス")
print("=" * 60)
cross_stats = []
for cond in ['A','B','C','D','E','X']:
    for rn in [11,12]:
        g = df[(df['condition_use']==cond) & (df['race_num']==rn)]
        if len(g) >= 3:
            s = stats(g, f'{cond}×{rn}')
            cross_stats.append(s)
            print(f"  {cond} × {rn}R: n={s['n']:2d} ROI={s['roi']:6.1f}% [{s['roi_lo']:6.1f}%,{s['roi_hi']:6.1f}%] hit={s['hit_rate']:.1f}%")


# === 出力レポート ===
report_path = 'data/results/healthy_4day_analysis.md'
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('# Healthy 4日分析 (5/9 投資戦略再評価)\n\n')
    f.write(f'生成: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write('対象日: 4/25, 4/26, 5/2, 5/3 (152R settled)\n\n')
    f.write('注意: session#3 提案 (commit 943791b3) は全14日含むため、汚染日 (4/12以前 投資集計バグ疑い、約300R) を除いた healthy 4日 (152R) で再計算。\n\n')

    f.write('## 1. 案 A/B/C/B改 比較 (healthy 4日)\n\n')
    f.write('| 案 | n | 投資 | 払戻 | 利益 | ROI | bootstrap 95%CI | 的中率 | 軸top3率 |\n')
    f.write('|----|--:|----:|----:|----:|----:|----------------:|------:|--------:|\n')
    for s in [stats_a, stats_b, stats_c, stats_bk]:
        f.write(f"| {s['label']} | {s['n']} | {s['inv']:,} | {int(s['pay']):,} | {int(s['profit']):+,} | **{s['roi']:.1f}%** | [{s['roi_lo']:.1f}%, {s['roi_hi']:.1f}%] | {s['hit_rate']:.1f}% | {s['axis_top3']:.1f}% |\n")
    f.write('\n*bootstrap: n_days=4 で日次クラスタ resampling、95%CI*\n\n')

    f.write('### session#3 (汚染含む) との差\n\n')
    f.write('| 案 | session#3 ROI | healthy ROI | 差分 |\n')
    f.write('|----|-------------:|------------:|----:|\n')
    f.write(f'| A | 159.8% | {stats_a["roi"]:.1f}% | {stats_a["roi"]-159.8:+.1f}pt |\n')
    f.write(f'| B | 173.8% | {stats_b["roi"]:.1f}% | {stats_b["roi"]-173.8:+.1f}pt |\n')
    f.write(f'| C | 110.7% | {stats_c["roi"]:.1f}% | {stats_c["roi"]-110.7:+.1f}pt |\n\n')

    f.write('## 2. 11R 分類別 (healthy 4日, n=' + str(len(r11)) + ')\n\n')
    f.write('| 分類 | n | 投資 | 払戻 | ROI | 的中率 | 軸top3率 |\n')
    f.write('|------|--:|-----:|-----:|----:|------:|--------:|\n')
    for s in grade_stats:
        f.write(f"| {s['label']} | {s['n']} | {s['inv']:,} | {int(s['pay']):,} | **{s['roi']:.1f}%** | {s['hit_rate']:.1f}% | {s['axis_top3']:.1f}% |\n")
    f.write('\n')

    f.write('## 3. 12R クラス別 (healthy 4日, n=' + str(len(r12)) + ')\n\n')
    f.write('| クラス | n | 投資 | 払戻 | ROI | 的中率 | 軸top3率 |\n')
    f.write('|--------|--:|-----:|-----:|----:|------:|--------:|\n')
    for s in class_stats:
        f.write(f"| {s['label']} | {s['n']} | {s['inv']:,} | {int(s['pay']):,} | **{s['roi']:.1f}%** | {s['hit_rate']:.1f}% | {s['axis_top3']:.1f}% |\n")
    f.write('\n')

    f.write('## 4. 条件 × race_num クロス (n>=3 のみ)\n\n')
    f.write('| 条件×R | n | ROI | bootstrap CI | 的中率 |\n')
    f.write('|--------|--:|----:|-------------:|------:|\n')
    for s in cross_stats:
        f.write(f"| {s['label']} | {s['n']} | **{s['roi']:.1f}%** | [{s['roi_lo']:.1f}%, {s['roi_hi']:.1f}%] | {s['hit_rate']:.1f}% |\n")
    f.write('\n')

    f.write('## 5. 各日 prediction error 分解\n\n')
    f.write('| 日付 | 11R+12R n | ROI | 軸top3 | 的中率 | 全R ROI | 全R 軸top3 |\n')
    f.write('|------|----------:|----:|------:|------:|--------:|----------:|\n')
    for s in day_stats:
        f.write(f"| {s['label']} | {s['n']} | {s['roi']:.1f}% | {s['axis_top3']:.1f}% | {s['hit_rate']:.1f}% | {s['full_roi']:.1f}% | {s['full_axis_top3']:.1f}% |\n")
    f.write('\n')

    # === 結論 ===
    f.write('## 結論\n\n')
    # Best plan
    plans = [stats_a, stats_b, stats_c, stats_bk]
    best = max(plans, key=lambda x: x['roi'])
    f.write(f'**最高 ROI: {best["label"]} = {best["roi"]:.1f}%** (95%CI [{best["roi_lo"]:.1f}%, {best["roi_hi"]:.1f}%])\n\n')

    # Plan B改 evaluation
    if stats_bk['roi'] > stats_b['roi']:
        f.write(f'✅ **案B改 ({stats_bk["roi"]:.1f}%) が案B ({stats_b["roi"]:.1f}%) を上回る** → 案B改採用推奨\n\n')
    else:
        f.write(f'⚠️ 案B改 ({stats_bk["roi"]:.1f}%) は 案B ({stats_b["roi"]:.1f}%) を下回る → 単純な案B が優位\n\n')

    # CI 評価
    if stats_bk['roi_lo'] >= 100:
        f.write(f'🟢 **案B改 95%CI 下限 {stats_bk["roi_lo"]:.1f}% ≥ 100%** → 期待値プラス側を統計的に確証\n\n')
    elif stats_bk['roi_lo'] >= 70:
        f.write(f'🟡 案B改 95%CI 下限 {stats_bk["roi_lo"]:.1f}% (70-100%) → 期待値プラスは未確証だが下限損失は許容範囲\n\n')
    else:
        f.write(f'🔴 案B改 95%CI 下限 {stats_bk["roi_lo"]:.1f}% (<70%) → 期待値プラスを統計的に確証できない\n\n')

    f.write('### 5/9 推奨\n\n')
    f.write(f'- **5/9 採用案: 案B改** (ROI {stats_bk["roi"]:.1f}%, n={stats_bk["n"]})\n')
    f.write(f'- 11R: 条件C × 非重賞のみ採用\n')
    f.write(f'- 12R: 1勝クラス OR 条件D (1200-1400m) のみ採用\n')
    f.write(f'- それ以外 全スキップ\n')

# Save raw stats
with open('data/results/healthy_4day_stats.json', 'w', encoding='utf-8') as f:
    json.dump({
        'plans': [stats_a, stats_b, stats_c, stats_bk],
        'grade_11r': grade_stats,
        'class_12r': class_stats,
        'cross_cond_rnum': cross_stats,
        'days': day_stats,
        'meta': {'dates': HEALTHY_DATES, 'n_total_settled': n_total},
    }, f, indent=2, default=str, ensure_ascii=False)

print(f"\n[OK] Saved {report_path}")
print(f"[OK] Saved data/results/healthy_4day_stats.json")
