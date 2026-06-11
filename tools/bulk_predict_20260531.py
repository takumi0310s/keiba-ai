#!/usr/bin/env python3
"""5/31 一括予測 (V15+V16 全頭スコア、 allscores形式)。検証/予想用・Discord送信なし。
未走レース=ライブ発走前オッズ(リークなし)、 終了レース=オッズ無し(結果オッズ回避)。
障害除外。 V15/V16 model不変・predict_core不変。 レジューム対応。
出力: data/allscores/20260531_v2/<race_id>.json
"""
from __future__ import annotations
import os,io,sys,json,time
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
if sys.platform=="win32":
    sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8"); sys.stderr=io.TextIOWrapper(sys.stderr.buffer,encoding="utf-8")
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"tools"))
import pandas as pd, gzip, pickle
from predict_core import (parse_shutuba,build_features,get_horse_stats,apply_horse_stats,
    set_horse_defaults,fetch_jra_and_weather,fetch_realtime_odds_full,is_race_started,load_models,predict_race)
from jrdb_features import merge_jrdb_once  # 二重マージ禁止ガード (6/11 Fable sweep)。歴史的出力(5/31)は二重マージで生成
DATE="20260531"
OUT=os.path.join(BASE,"data","allscores",f"{DATE}_v2"); os.makedirs(OUT,exist_ok=True)
DONE=os.path.join(BASE,"data",f"_bulk_done_{DATE}.txt")
done=set(open(DONE).read().split()) if os.path.exists(DONE) else set()
races=json.load(open(os.path.join(BASE,"data",f"_races_{DATE}.json"),encoding="utf-8"))
md=load_models()
v16=pickle.load(gzip.open(os.path.join(BASE,"models","v16_ability_candidate.pkl.gz"),"rb"))
import xgboost as xgb
def v16score(df):
    X=df.reindex(columns=v16["features"]).fillna(0).values.astype("float32")
    p=v16["model"].predict(X); w=v16.get("ensemble_weights",{"lgb":.5,"xgb":.5})
    if v16.get("xgb_model") is not None: p=w.get("lgb",.5)*p+w.get("xgb",.5)*v16["xgb_model"].predict(xgb.DMatrix(X))
    return {str(int(df.iloc[i]["馬番"])):float(p[i]) for i in range(len(df))}
jw={}
print(f"[{time.strftime('%H:%M:%S')}] bulk predict {DATE}: {len(races)}R (済{len(done)})",flush=True)
for i,rc in enumerate(sorted(races,key=lambda x:(x['course'],x['race_num'])),1):
    rid=rc["race_id"]
    if rid in done: print(f"  [{i}] {rid} skip(済)",flush=True); continue
    try:
        rn,horses,hids,rinfo=parse_shutuba(rid)
        if not horses: print(f"  [{i}] {rid} 出馬表空 skip",flush=True); continue
        if rinfo.get("surface")=="障":
            open(DONE,"a").write(rid+"\n"); print(f"  [{i}] {rc['course']}{rc['race_num']}R 障害除外",flush=True); continue
        started=is_race_started(rid)
        od={}
        if not started:
            try: of=fetch_realtime_odds_full(rid); od={u:v["odds"] for u,v in of.items()}
            except Exception: od={}
        for h in horses:
            u=h.get("馬番",0)
            if u in od: h["単勝オッズ"]=od[u]
        c=rinfo.get("course","")
        if md.get("is_live") and c:
            if c not in jw:
                try: jw[c]=fetch_jra_and_weather(c)
                except Exception: jw[c]=({},{})
            jra,wx=jw[c]
        else: jra,wx={},{}
        for k,(h,hid) in enumerate(zip(horses,hids)):
            try: apply_horse_stats(h,get_horse_stats(hid,rinfo["distance"],rinfo["surface"],c),rinfo) if hid else set_horse_defaults(h)
            except Exception: set_horse_defaults(h)
            if k<len(horses)-1: time.sleep(0.22)
        df=build_features(horses,rinfo,md,race_id=rid,odds_dict=od,jra_track_info=jra,weather_info=wx)
        try: df=merge_jrdb_once(df,rid)
        except Exception as e: print(f"    [JRDB] {e}",flush=True)
        df=predict_race(df,md,bool(od),race_info=rinfo)
        df=df.sort_values("スコア",ascending=False).reset_index(drop=True)
        v15={str(int(r["馬番"])):float(r["スコア"]) for _,r in df.iterrows() if pd.notna(r.get("馬番"))}
        try: v16s=v16score(df)
        except Exception as e: print(f"    [V16] {e}",flush=True); v16s={}
        rec={"race_id":rid,"race_name":rn,"course":c,"race_num":rc["race_num"],"start":rc.get("start",""),
             "started":bool(started),"odds_used":bool(od),
             "horses":[{"馬番":int(r["馬番"]),"馬名":str(r.get("馬名",""))} for _,r in df.iterrows() if pd.notna(r.get("馬番"))],
             "v15_scores":v15,"v16_scores":v16s}
        json.dump(rec,open(os.path.join(OUT,f"{rid}.json"),"w",encoding="utf-8"),ensure_ascii=False)
        open(DONE,"a").write(rid+"\n")
        print(f"  [{i}] {c}{rc['race_num']}R {rn[:14]} OK ({len(df)}頭 {'未走/live odds' if not started else '終了/odds無'})",flush=True)
    except Exception as e:
        print(f"  [{i}] {rid} ERR {e}",flush=True)
print(f"[{time.strftime('%H:%M:%S')}] done (計{len(set(open(DONE).read().split()))})",flush=True)
