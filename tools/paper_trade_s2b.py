#!/usr/bin/env python3
"""前向き paper trading: s2b(穴特化候補)の予測を発走前に記録 → 結果照合 → ROI継続測定。

★完全分離・本番非影響★:
  - 本番 V15/V16 .pkl.gz / predict_core / daily_predict / app.py / race_auto_notify は一切改変しない。
  - 投票・買い目通知・Discord(買い目)には一切出さない。記録のみ。
  - 発走前データ(本番liveが使えるデータ)のみで予測 = リーク構造的に不可能。
  - 出力は新ログ data/paper_s2b/ (.gitignore対象)。

モード:
  predict  --date YYYYMMDD : 当日全レースを s2b で予測 → top6+人気を記録 (発走前に実行)
  results  --date YYYYMMDD : JRA払戻と照合 → 券種別ROIを計算・追記
  report                    : 累積ROI/的中率/N を券種別に表示 (leak-freeバックテストと乖離確認)
  from-oof --date YYYYMMDD : (検証用) leak-free OOF から擬似pred-logを生成しresults配線をテスト

s2b予測は predict_core.build_features(本番と同じ発走前特徴) → s2b特徴(人気代理族除去+レース相対) → s2b候補。
ze4特徴は live では過去ZEDのみ=元々 leak-free。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, argparse, time
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
LOG_DIR = os.path.join(DATA, 'paper_s2b')
S2B_PATH = os.path.join(BASE, 'models', 'v16_anaba_s2b_candidate.pkl.gz')

ODDS_REMOVE = ['paci_ninki_idx','odds_change_rate','odds_sharp_drop','oz_base_pop_rank','oz_fukusho_base_log','oz_tansho_base_log','pop_rank_change','prev_odds_log']
PROXY_FAMILY = ['paci_jockey_exp_wr','paci_jockey_exp_3rd','paci_jockey_mark','paci_sogo_mark','paci_train_mark','paci_idm_mark','jrdb_cid_idx','jrdb_ls_idx','jrdb_training_idx','jrdb_stable_idx']
EXTRA = ['paci_goal_rank','paci_goal_diff','paci_dochu_rank']
RAW_REPLACE = ['jrdb_running_style','jrdb_dist_apt']

_S2B = None
def _load_s2b():
    global _S2B
    if _S2B is None:
        with gzip.open(S2B_PATH,'rb') as f: _S2B = pickle.load(f)
    return _S2B


def build_s2b_features(race_df: pd.DataFrame) -> pd.DataFrame:
    """1レース分の V15特徴df(全馬) から s2b 特徴(one-hot脚質/距離適性 + レース相対 + 交互)を構築。
    v16_anaba_s2_eval.build_features を流用(race_id_unique を定数化=1レース内で集計)。"""
    from v16_anaba_s2_eval import build_features as _bf
    df = race_df.copy()
    if 'race_id_unique' not in df.columns: df['race_id_unique'] = 'PAPER_RACE'
    # 必須列の存在保証(無ければ0)
    for c in ['jrdb_running_style','jrdb_dist_apt','distance','num_horses_val','horse_num','jrdb_tb_homestr_inner']:
        if c not in df.columns: df[c] = 0
    return _bf(df)


def score_s2b(race_df: pd.DataFrame) -> dict:
    """race_df(V15特徴・全馬) → {horse_num: s2b_score}。"""
    m = _load_s2b(); feats = m['features']
    df = build_s2b_features(race_df)
    for f in feats:
        if f not in df.columns: df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    import xgboost as xgb
    X = df[feats].values
    p = 0.5*m['model'].predict(X) + 0.5*m['xgb_model'].predict(xgb.DMatrix(X))
    hn = pd.to_numeric(df['horse_num'], errors='coerce').fillna(0).astype(int).values
    return {int(h): float(s) for h, s in zip(hn, p)}


# ============ 払戻 index (date, course, race_num) ============
def _to_i(x):
    try: return int(float(x))
    except Exception: return None

def load_payout_index():
    p = pd.read_csv(os.path.join(DATA,'jra_payouts.csv'), low_memory=False)
    idx = {}
    for _, r in p.iterrows():
        key = f"{int(r['race_date'])}_{r['course']}_{int(r['race_num'])}"
        fuk = {}
        try:
            for n,pay in zip(str(r['fukusho_nums']).replace(' ','').split('/'), str(r['fukusho_payouts']).replace(' ','').split('/')):
                if _to_i(n) is not None: fuk[_to_i(n)] = _to_i(pay) or 0
        except Exception: pass
        def pset(nums,pay):
            try:
                a=str(nums).replace(' ','').split('-')
                if all(_to_i(x) is not None for x in a): return (frozenset(_to_i(x) for x in a), _to_i(pay) or 0)
            except Exception: pass
            return None
        def pord(nums,pay):
            try:
                a=str(nums).replace(' ','').split('-')
                if len(a)==3 and all(_to_i(x) is not None for x in a): return (tuple(_to_i(x) for x in a), _to_i(pay) or 0)
            except Exception: pass
            return None
        idx[key] = {'tan':(_to_i(r['tansho_nums']),_to_i(r['tansho_payout']) or 0),'fuk':fuk,
                    'umaren':pset(r['umaren_nums'],r['umaren_payout']),'trio':pset(r['trio_nums'],r['trio_payout']),
                    'tierce':pord(r['tierce_nums'],r['tierce_payout'])}
    return idx


# ============ 券種(top6順位 o から (return, points)) ============
import itertools
def bt_tan(o,pm): return (pm['tan'][1] if pm['tan'][0]==o[0] else 0, 1)
def bt_fuku1(o,pm): return (pm['fuk'].get(o[0],0), 1)
def bt_umaren_t3box(o,pm):
    r=sum(pm['umaren'][1] for a,b in itertools.combinations(o[:3],2) if pm['umaren'] and pm['umaren'][0]==frozenset((a,b))); return (r,3)
def bt_trio4(o,pm):
    if not pm['trio']: return (0,0)
    won=pm['trio'][0]; combos=list(itertools.combinations(o[:4],3))
    return (pm['trio'][1] if any(frozenset(c)==won for c in combos) else 0, len(combos))
def bt_tierce125(o,pm):
    if not pm['tierce']: return (0,0)
    bets=set()
    for b in o[1:3]:
        for c in o[1:6]:
            if len({o[0],b,c})==3: bets.add((o[0],b,c))
    return (pm['tierce'][1] if pm['tierce'][0] in bets else 0, len(bets))
BETS = [('単勝',bt_tan),('複勝top1',bt_fuku1),('馬連top3box',bt_umaren_t3box),('三連複top4box',bt_trio4),('三連単form1-2-5',bt_tierce125)]


def _pred_path(date): return os.path.join(LOG_DIR, f'{date}_pred.jsonl')
def _res_path(date):  return os.path.join(LOG_DIR, f'{date}_results.jsonl')


# ============ predict (live・発走前) ============
def predict_date(date):
    os.makedirs(LOG_DIR, exist_ok=True)
    from predict_core import (load_models, parse_shutuba, build_features, get_horse_stats,
                              apply_horse_stats, set_horse_defaults, classify_race_condition)
    from jrdb_features import merge_jrdb_predict_features
    import daily_predict as dp
    model_data = load_models()
    races = dp.fetch_race_list(date)
    if not races:
        print(f"[paper_s2b] {date} レースなし(発走前に再実行 or 非開催)"); return
    out = open(_pred_path(date), 'w', encoding='utf-8')
    n_ok = 0
    for r in races:
        rid = r['race_id']
        try:
            horses = parse_shutuba(rid)
            if not horses: continue
            race_info = horses[0].get('race_info') if isinstance(horses[0], dict) and 'race_info' in horses[0] else r
            # daily_predict と同じ発走前構築(get_horse_stats → build_features → JRDBマージ)
            df = build_features(horses, r, model_data, race_id=rid)
            try: df = merge_jrdb_predict_features(df, rid)
            except Exception: pass
            scores = score_s2b(df)
            order = [h for h,_ in sorted(scores.items(), key=lambda x:-x[1])]
            rec = {'date':date, 'race_id':rid, 'course':r.get('course',''), 'race_num':r.get('race_num',0),
                   'rk':f"{date}_{r.get('course','')}_{r.get('race_num',0)}",
                   's2b_top6':order[:6], 'ts':time.strftime('%Y-%m-%dT%H:%M:%S')}
            out.write(json.dumps(rec, ensure_ascii=False)+'\n'); n_ok += 1
            print(f"  {r.get('course','')}{r.get('race_num','')}R s2b top6={order[:6]}")
        except Exception as e:
            print(f"  [skip] {rid}: {e}")
    out.close()
    print(f"[paper_s2b] {date}: {n_ok}R 記録 → {_pred_path(date)} (投票・通知なし)")


# ============ results 照合 ============
def results_date(date):
    pp = _pred_path(date)
    if not os.path.exists(pp): print(f"[paper_s2b] {pp} なし"); return
    pay = load_payout_index()
    out = open(_res_path(date),'w',encoding='utf-8'); n=0
    for line in open(pp,encoding='utf-8'):
        rec = json.loads(line); key = rec['rk']
        if key not in pay: continue
        o = [int(x) for x in rec['s2b_top6']]
        if len(o) < 5: continue
        pm = pay[key]; bets = {}
        for name, fn in BETS:
            ret, pts = fn(o, pm); bets[name] = {'ret':ret, 'pts':pts}
        out.write(json.dumps({'rk':key,'bets':bets}, ensure_ascii=False)+'\n'); n+=1
    out.close()
    print(f"[paper_s2b] {date}: {n}R 照合 → {_res_path(date)}")


# ============ report 累積 ============
def report():
    files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_results.jsonl')]) if os.path.isdir(LOG_DIR) else []
    agg = {name:{'ret':0,'stake':0,'hit':0,'n':0} for name,_ in BETS}
    for fn in files:
        for line in open(os.path.join(LOG_DIR,fn),encoding='utf-8'):
            rec = json.loads(line)
            for name,_ in BETS:
                b = rec['bets'].get(name)
                if not b or b['pts']==0: continue
                a = agg[name]; a['ret']+=b['ret']; a['stake']+=100*b['pts']; a['hit']+=(b['ret']>0); a['n']+=1
    print(f"=== paper s2b 累積ROI (照合日数={len(files)}) ===")
    print(f"  leak-freeバックテスト参照: 単勝111.6% / 三連複top4box194.3% / 三連単1-2-5 229% / 馬連top3box146%")
    print(f"  {'券種':16s}{'ROI':>8s}{'的中率':>8s}{'N':>7s}{'収支':>12s}")
    for name,_ in BETS:
        a = agg[name]
        if a['n']==0: print(f"  {name:16s}{'--':>8s}{'--':>8s}{0:>7d}"); continue
        roi=a['ret']/a['stake'] if a['stake'] else 0; pnl=a['ret']-a['stake']
        warn = ' ★N<100信用不可' if a['n']<100 else ''
        print(f"  {name:16s}{roi*100:7.1f}%{a['hit']/a['n']*100:7.1f}%{a['n']:7d}{pnl:>11,}円{warn}")


# ============ from-oof (検証配線) ============
def from_oof(date):
    """検証用: leak-free OOF(s_s2b)から擬似pred-log生成(過去レースで results/report 配線テスト)。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    oof = pd.read_parquet(os.path.join(DATA,'v16_leakfree_s2b_oof.parquet'))  # leak-free s_s2b, _rk, horse_num, finish
    # _rk = date_course_kai_nichi_racenum → paper rk = date_course_racenum へ変換
    def conv(rk):
        p = rk.split('_'); return f"{p[0]}_{p[1]}_{p[4]}" if len(p)>=5 else rk
    oof = oof.copy(); oof['prk'] = oof['_rk'].map(conv)
    if date != 'ALL':
        oof = oof[oof['prk'].str.startswith(date)]
    out = open(_pred_path(date),'w',encoding='utf-8'); n=0
    for prk,g in oof.groupby('prk'):
        order=[int(x) for x in g.sort_values('s_s2b',ascending=False)['horse_num'].tolist()]
        out.write(json.dumps({'date':date,'rk':prk,'s2b_top6':order[:6]},ensure_ascii=False)+'\n'); n+=1
    out.close(); print(f"[from-oof] {n}R 擬似pred生成({date})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['predict','results','report','from-oof'])
    ap.add_argument('--date', default='')
    a = ap.parse_args()
    if a.mode=='predict': predict_date(a.date or time.strftime('%Y%m%d'))
    elif a.mode=='results': results_date(a.date or time.strftime('%Y%m%d'))
    elif a.mode=='report': report()
    elif a.mode=='from-oof': from_oof(a.date or 'ALL')

if __name__=='__main__':
    main()
