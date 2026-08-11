# -*- coding: utf-8 -*-
"""無人供給の初回検証 (2026-08-13 朝実行・8/15前の予行演習)。
GO/NG表を機械生成。偽装成功 (ログDone×行数不変=死因#2型) を明示チェック。
ハード検出器 = jv_hc (JV調教は平日も毎日配信 → 行数が進まなければ偽装確定)。"""
import json, os, subprocess, sys, glob
from datetime import datetime, timedelta
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if sys.platform=='win32': sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

today = datetime.now().strftime('%Y%m%d')
base = json.load(open(os.path.join(BASE,'research/v15r/supply_check_baseline.json'),encoding='utf-8'))
rows=[]
def add(item, ok, detail): rows.append((item, 'GO' if ok else '★NG★', detail))

def task_result(tn):
    r = subprocess.run(['schtasks','/query','/tn',tn,'/v','/fo','csv'],
                       capture_output=True, text=True, encoding='cp932', errors='replace')
    try:
        import csv as _csv, io
        rec = list(_csv.DictReader(io.StringIO(r.stdout)))[0]
        keys = list(rec.keys())
        lr = rec.get('Last Run Time') or rec.get('前回の実行時刻')
        rc = rec.get('Last Result') or rec.get('前回の結果')
        return lr, rc
    except Exception as e:
        return None, f'query失敗:{e}'

# 1. タスク実行結果
for tn, jsttime in [(r'keiba-ai\JrdbSupplyDaily','03:20'), (r'keiba-ai\JvlinkSupplyDaily','03:40'),
                    (r'keiba-ai\T1v2Audit','08:50')]:
    lr, rc = task_result(tn)
    ok = (lr is not None) and (today[:4] in str(lr) or str(datetime.now().day) in str(lr)) and str(rc) in ('0','267009')
    add(f'task {tn.split(chr(92))[-1]} ({jsttime})', ok, f'last={lr} rc={rc}')

# 2. ログのタイムスタンプ+Done
for name, pat, done_tok in [('JRDBログ', f'logs/daily_jrdb_supply_{today}.log', 'ALL OK'),
                            ('JVログ',  f'logs/daily_jvlink_supply_{today}.log', 'ALL OK')]:
    p=os.path.join(BASE,pat)
    if os.path.exists(p):
        txt=open(p,encoding='utf-8',errors='replace').read()
        add(name, done_tok in txt, f"存在 / '{done_tok}'{'あり' if done_tok in txt else 'なし'}")
    else:
        add(name, False, '当日ログなし (タスク未実行)')

# 3. ★実データ鮮度★ + 偽装成功チェック
def rowcount(p): return sum(1 for _ in open(p,encoding='utf-8-sig',errors='replace'))-1
hc_now = rowcount(os.path.join(BASE,'data/jvlink/jv_hc.csv'))
hc_delta = hc_now - base['jv_hc']['rows']
hc_latest = str(pd.read_csv(os.path.join(BASE,'data/jvlink/jv_hc.csv'),usecols=['train_date'],dtype=str)['train_date'].max())
yday=(datetime.now()-timedelta(days=1)).strftime('%Y%m%d')
add('★偽装検査(死因#2)★ jv_hc行数', hc_delta>0, f'{base["jv_hc"]["rows"]:,}→{hc_now:,} (+{hc_delta}) 調教は毎日配信=+0なら偽装')
add('jv_hc 内容最新日', hc_latest>=yday, f'{hc_latest} (要≥前日{yday})')
ck = json.load(open(os.path.join(BASE,'data/jvlink/daily/checkpoint.json'),encoding='utf-8-sig'))
ck_adv = any(str(ck.get(k,''))>str(base['jv_checkpoint'].get(k,'')) for k in ck)
add('JV checkpoint 前進', ck_adv, f"{ {k:ck[k][:8] for k in ck} }")
# JRDB: 平日はレース系新規なしが正常 → extracted数の非減少 + mtime更新(ジョブが再構築した)で判定
ext_delta = {d: len(glob.glob(os.path.join(BASE,f'data/jrdb/extracted/{d}/*'))) - base[f'ext_{d}']
             for d in ['Kyi','Sed','Kka','Kta','Cha','Jo','Oz']}
kyi_mt = os.path.getmtime(os.path.join(BASE,'data/jrdb_kyi.csv'))
add('JRDB extracted 非減少', all(v>=0 for v in ext_delta.values()), f'{ext_delta} (平日+0=正常: 新規publishなし)')
add('JRDB CSV 再構築実行 (mtime更新)', kyi_mt>base['jrdb_kyi']['mtime'],
    f'kyi.csv mtime {"更新" if kyi_mt>base["jrdb_kyi"]["mtime"] else "★不変=ジョブ未実行★"}')
jrdb_rows_now={t:rowcount(os.path.join(BASE,f'data/jrdb_{t}.csv')) for t in ['kyi','sed','kka']}
add('JRDB 行数 非減少', all(jrdb_rows_now[t]>=base[f'jrdb_{t}']['rows'] for t in jrdb_rows_now),
    f'{ {t:(base[f"jrdb_{t}"]["rows"],jrdb_rows_now[t]) for t in jrdb_rows_now} }')

# 4. T1v2 source-check が両系を見たか
t1log=os.path.join(BASE,'logs/t1v2_audit.log')
tail=open(t1log,encoding='utf-8',errors='replace').read()[-2000:] if os.path.exists(t1log) else ''
saw_pass='[T1v2 source] PASS' in tail
add('T1v2 08:50 source-check PASS', saw_pass, tail.strip().splitlines()[-1][:90] if tail else 'ログなし')
# jv_health 当日生成 = JV系も見た証拠 (source_checkはjv_health必須化済)
jvh=os.path.exists(os.path.join(BASE,f'data/T1v2_audit/jv_health_{today}.json'))
sph=os.path.exists(os.path.join(BASE,f'data/T1v2_audit/supply_health_{today}.json'))
add('両系ヘルスJSON当日生成', jvh and sph, f'supply_health={sph} jv_health={jvh}')

print('='*72)
print(f'無人供給 初回検証 GO/NG表  ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
print('='*72)
w=max(len(r[0]) for r in rows)
for item,ok,detail in rows:
    print(f'{item:<{w}}  {ok:^6}  {detail}')
ngs=[r for r in rows if r[1]!='GO']
print('-'*72)
print(f'総合: {"★GO★ (8/15 本番チェーンへ)" if not ngs else f"★NG {len(ngs)}件★ → 要修理"}')
