"""V15-audit-4: read-only cumulative ROI audit after 5/17 G1 day."""
import pandas as pd
import numpy as np
import json
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CSV = r'C:\Users\takum\keiba-ai\data\cumulative_results.csv'
DAILY_517 = r'C:\Users\takum\keiba-ai\data\daily_results\20260517.csv'

cum = pd.read_csv(CSV)
print(f"raw rows: {len(cum)}")
print(f"columns: {list(cum.columns)}")
print()

# settled only
cum = cum[cum['status'] == 'settled'].copy()
cum['inv'] = pd.to_numeric(cum['investment'], errors='coerce').fillna(0)
cum['pay'] = pd.to_numeric(cum['actual_payout'], errors='coerce').fillna(0)
cum['pnl'] = cum['pay'] - cum['inv']
cum['date_str'] = cum['date'].astype(str).str.replace('.0', '', regex=False).str.replace('nan', '', regex=False)

print(f"settled rows: {len(cum)}")
print(f"settled with inv>0: {(cum['inv']>0).sum()}")
print()

def stat(df, label):
    n_all = len(df)
    df_v = df[df['inv'] > 0]
    n = len(df_v)
    inv = df_v['inv'].sum()
    pay = df_v['pay'].sum()
    pnl = df_v['pnl'].sum()
    roi = (pay / inv * 100) if inv > 0 else 0.0
    hits = (df_v['pay'] > 0).sum()
    hr = (hits / n * 100) if n > 0 else 0.0
    print(f"[{label}] n_all={n_all} n_voted={n} hits={hits} hit_rate={hr:.2f}% inv=¥{inv:,.0f} pay=¥{pay:,.0f} ROI={roi:.2f}% PnL=¥{pnl:+,.0f}")
    return dict(label=label, n_all=n_all, n_voted=n, hits=int(hits), hit_rate=hr, inv=float(inv), pay=float(pay), roi=roi, pnl=float(pnl))

results = {}
results['total'] = stat(cum, 'TOTAL ALL')
results['pre_515'] = stat(cum[cum['date_str'] <= '20260515'], '<=5/15 (pre-案C)')
results['post_516'] = stat(cum[cum['date_str'] >= '20260516'], '>=5/16 (post-案C)')
results['only_516'] = stat(cum[cum['date_str'] == '20260516'], '5/16 only')
results['only_517'] = stat(cum[cum['date_str'] == '20260517'], '5/17 only')
results['ex_516'] = stat(cum[cum['date_str'] != '20260516'], 'TOTAL ex 5/16')
print()

# 5/17 detail
print("=== 5/17 detail ===")
day = cum[cum['date_str'] == '20260517'].copy()
print(f"races on 5/17: {len(day)}")
voted = day[day['inv'] > 0]
print(f"voted: {len(voted)}")
print(f"hits: {(voted['pay']>0).sum()}")
print()

# Tokyo 11R - Victoria Mile
print("=== Victoria Mile (Tokyo 11R) ===")
vm = day[day['race_id'].astype(str).str.contains('11', regex=False) & day['course'].astype(str).str.contains('東京', regex=False)]
print(vm[['race_id','course','race_num','race_name','top1_num','top1_name','top2_num','top3_num','top1_finish','top2_finish','top3_finish','trio_result','trio_hit','actual_payout','investment','profit']].to_string())
print()

# All Tokyo races on 5/17
print("=== All Tokyo on 5/17 ===")
tok = day[day['course'] == '東京']
print(tok[['race_num','race_name','condition','top1_num','top1_finish','trio_hit','actual_payout','investment','profit']].to_string())
print()

# Bootstrap CI (10K iter) using TOTAL voted set
voted_all = cum[cum['inv'] > 0].copy()
np.random.seed(42)
n_iter = 10000
roi_samples = []
arr_inv = voted_all['inv'].values
arr_pay = voted_all['pay'].values
N = len(arr_inv)
for _ in range(n_iter):
    idx = np.random.randint(0, N, size=N)
    inv_s = arr_inv[idx].sum()
    pay_s = arr_pay[idx].sum()
    if inv_s > 0:
        roi_samples.append(pay_s / inv_s * 100)
roi_samples = np.array(roi_samples)
ci_low, ci_high = np.percentile(roi_samples, [2.5, 97.5])
mean_roi = roi_samples.mean()
median_roi = np.median(roi_samples)
print(f"=== Bootstrap CI (TOTAL, n_iter=10000) ===")
print(f"N_voted={N} mean={mean_roi:.2f}% median={median_roi:.2f}% CI95=[{ci_low:.2f}%, {ci_high:.2f}%]")
includes_100 = ci_low <= 100 <= ci_high
print(f"95% CI includes 100%: {includes_100}")
print()

# Also CI excluding 5/16 jackpot
ex516 = cum[(cum['inv'] > 0) & (cum['date_str'] != '20260516')].copy()
arr_inv2 = ex516['inv'].values
arr_pay2 = ex516['pay'].values
N2 = len(arr_inv2)
np.random.seed(42)
roi_samples2 = []
for _ in range(n_iter):
    idx = np.random.randint(0, N2, size=N2)
    inv_s = arr_inv2[idx].sum()
    pay_s = arr_pay2[idx].sum()
    if inv_s > 0:
        roi_samples2.append(pay_s / inv_s * 100)
roi_samples2 = np.array(roi_samples2)
ci_low2, ci_high2 = np.percentile(roi_samples2, [2.5, 97.5])
print(f"=== Bootstrap CI (EX 5/16 jackpot) ===")
print(f"N={N2} mean={roi_samples2.mean():.2f}% CI95=[{ci_low2:.2f}%, {ci_high2:.2f}%]")
print(f"100% in CI: {ci_low2 <= 100 <= ci_high2}")

results['ci'] = dict(N=N, mean=float(mean_roi), median=float(median_roi), low=float(ci_low), high=float(ci_high), includes_100=bool(includes_100))
results['ci_ex516'] = dict(N=N2, low=float(ci_low2), high=float(ci_high2), includes_100=bool(ci_low2 <= 100 <= ci_high2))

with open(r'C:\Users\takum\keiba-ai\data\audit_v15_4_results.json','w',encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n[ok] saved data/audit_v15_4_results.json")
