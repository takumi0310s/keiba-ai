#!/usr/bin/env python3
"""今日23Rの特徴量行列を再構築 → 特徴ごとの充足率を測定 (2026-05-30 検証専用)。
昼の予想と同条件(朝オッズ固定・健全paci/ZE・各馬過去成績)。V15/V16 model不変・読み取りのみ。
出力: data/_feat_matrix_20260530.csv (全馬 × 145特徴)
"""
from __future__ import annotations
import os, io, sys, json, glob, time
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
if sys.platform=="win32":
    sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8"); sys.stderr=io.TextIOWrapper(sys.stderr.buffer,encoding="utf-8")
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"tools"))
import pandas as pd, gzip, pickle
from predict_core import (parse_shutuba,build_features,get_horse_stats,apply_horse_stats,
                          set_horse_defaults,fetch_jra_and_weather,load_odds_base,load_models)
from jrdb_features import merge_jrdb_once  # 二重マージ禁止ガード (6/11 Fable sweep)。歴史的出力(5/30)は二重マージで生成
DATE="20260530"
MORN=os.path.join(BASE,"data","allscores",DATE)
OUT=os.path.join(BASE,"data","_feat_matrix_20260530.csv")
v15=pickle.load(gzip.open(os.path.join(BASE,"keiba_model_v15_central_live.pkl.gz"),"rb"))
FEATS=v15["features"][:v15["model"].num_feature()]
md=load_models()
rows=[]; jw={}
DONE_F=os.path.join(BASE,"data","_feat_done_20260530.txt")
done=set(open(DONE_F).read().split()) if os.path.exists(DONE_F) else set()
files=sorted(glob.glob(os.path.join(MORN,"*.json")))
print(f"[{time.strftime('%H:%M:%S')}] feat reconstruct: {len(files)}R (済{len(done)})",flush=True)
for i,f in enumerate(files,1):
    rid=os.path.basename(f)[:-5]
    if rid in done:
        print(f"  [{i}] {rid} skip(済)",flush=True); continue
    try:
        race_name,horses,horse_ids,rinfo=parse_shutuba(rid)
        if not horses or rinfo.get("surface")=="障": print(f"  {rid} skip"); continue
        ob=load_odds_base(rid,DATE); odds={u:v["odds"] for u,v in ob.items()}
        for h in horses:
            u=h.get("馬番",0)
            if u in ob: h["単勝オッズ"]=ob[u]["odds"]; h["人気順位"]=ob[u]["pop_rank"]
        c=rinfo.get("course","")
        if md.get("is_live") and c:
            if c not in jw:
                try: jw[c]=fetch_jra_and_weather(c)
                except Exception: jw[c]=({},{})
            jra,wx=jw[c]
        else: jra,wx={},{}
        for k,(h,hid) in enumerate(zip(horses,horse_ids)):
            try: apply_horse_stats(h,get_horse_stats(hid,rinfo["distance"],rinfo["surface"],c),rinfo) if hid else set_horse_defaults(h)
            except Exception: set_horse_defaults(h)
            if k<len(horses)-1: time.sleep(0.2)
        df=build_features(horses,rinfo,md,race_id=rid,odds_dict=odds,jra_track_info=jra,weather_info=wx)
        try: df=merge_jrdb_once(df,rid)
        except Exception as e: print(f"  [JRDB] {e}")
        for col in FEATS:
            if col not in df.columns: df[col]=0.0
        sub=df[FEATS].copy()
        # 逐次追記(途中終了でも残る)
        hdr = not os.path.exists(OUT)
        sub.to_csv(OUT,mode="a",index=False,header=hdr,encoding="utf-8-sig")
        open(DONE_F,"a").write(rid+"\n")
        print(f"  [{i}] {c} {rid} OK ({len(df)}頭) → 追記",flush=True)
    except Exception as e:
        print(f"  {rid} ERR {e}",flush=True)
print(f"[{time.strftime('%H:%M:%S')}] done",flush=True)
