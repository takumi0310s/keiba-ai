# -*- coding: utf-8 -*-
"""
Phase C-2: オッズ結合後の本評価（市場ベースライン比較）。
既存V15運用ログ(読み取り専用)を流用:
  - data/daily_predictions_full/<date>.csv : per-horse odds(朝) → 人気=オッズ順位
  - data/daily_results_full/<date>.json    : 全馬 finish_order + 確定配当(単勝/複勝)
両方揃う4日(6/7,6/14,6/20,6/21)で評価。ツール特徴は raw JSON から再計算。
data/ へは書き込まない。
"""
import sys, os, glob, json, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np
from tool_filters import (calc_sim_turf, calc_sim_dirt,
    R_DR, R_RES, R_TD, R_AG3, R_AGR, R_CV, R_MO)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "keiba-ai", "data"))
if not os.path.isdir(DATA):
    DATA = os.path.abspath(os.path.join(HERE, "..", "..", "data"))
PLACE = {'札幌':'01','函館':'02','福島':'03','新潟':'04','東京':'05',
         '中山':'06','中京':'07','京都':'08','阪神':'09','小倉':'10'}
DATES = ["20260607", "20260614", "20260620", "20260621"]
def nz(s): return unicodedata.normalize("NFKC", str(s or "")).strip()


def horse_feat(race, h):
    is_turf = race["s"] == "芝"; tx, ty = race.get("tx"), race.get("ty"); date = int(race["date"])
    runs = [r for r in (h.get("runs") or []) if len(r) >= 15 and int(r[R_DR]) < date]
    pcts = []; good = tot = 0; agr_ranks = []
    for r in runs[:10]:
        if r[R_MO] is None or (is_turf and r[R_CV] is None): continue
        pct = calc_sim_turf(tx, ty, r[R_CV], r[R_MO]) if is_turf else calc_sim_dirt(ty, r[R_MO])
        pcts.append(pct)
        if pct >= 50:
            tot += 1
            if (r[R_RES] is not None and r[R_RES] <= 3) or (r[R_TD] is not None and r[R_TD] <= 0.6):
                good += 1
    for r in runs:
        if r[R_AGR] is not None: agr_ranks.append(r[R_AGR])
    num = str(h.get("num"))
    ag3 = (race.get("rsim") or {}).get("ag3", {}).get(num)
    return dict(sim_pct_max=max(pcts) if pcts else np.nan,
                simgood_rate=(good/tot) if tot else np.nan,
                past_agari_rank_mean=(np.mean(agr_ranks) if agr_ranks else np.nan),
                rsim_ag3=ag3)


def load_tool():
    rows = []
    for dt in DATES:
        for fp in glob.glob(os.path.join(HERE, f"raw/{dt}/scatter_*.json")):
            d = json.load(open(fp, encoding="utf-8")); rc = d["race"]
            rsim = d.get("rsim") or {}; ag3 = rsim.get("ag3") or {}
            order = sorted([(k, v) for k, v in ag3.items() if v is not None], key=lambda kv: kv[1])
            rank = {k: i+1 for i, (k, v) in enumerate(order)}
            for h in d.get("horses", []):
                num = str(h.get("num"))
                if not num.isdigit(): continue
                f = horse_feat(rc, h)
                rows.append(dict(date=dt, place=PLACE.get(nz(rc["v"])), rno=int(rc["no"]),
                                 umaban=int(num), surf=rc["s"], dist=rc["dist"],
                                 rsim_ag3_rank=rank.get(num), **f))
    return pd.DataFrame(rows)


def load_odds():
    rows = []
    for dt in DATES:
        fp = os.path.join(DATA, "daily_predictions_full", f"{dt}.csv")
        p = pd.read_csv(fp, dtype=str)
        p["place"] = p["race_id"].str[4:6]; p["rno"] = p["race_id"].str[-2:].astype(int)
        p["umaban"] = pd.to_numeric(p["horse_num"], errors="coerce")
        p["odds"] = pd.to_numeric(p["odds"], errors="coerce")
        p["date"] = dt
        rows.append(p[["date", "place", "rno", "umaban", "odds", "race_id"]])
    o = pd.concat(rows, ignore_index=True).dropna(subset=["umaban", "odds"])
    o["umaban"] = o["umaban"].astype(int)
    o["pop"] = o.groupby("race_id")["odds"].rank(method="min").astype(int)
    return o


def load_results():
    rows = []
    for dt in DATES:
        fp = os.path.join(DATA, "daily_results_full", f"{dt}.json")
        arr = json.load(open(fp, encoding="utf-8"))
        for r in arr:
            rid = str(r["race_id"]); place = rid[4:6]; rno = int(rid[-2:])
            fo = r.get("finish_order") or {}
            pay = r.get("payouts") or {}
            tansho = pay.get("tansho"); fuku = pay.get("fukusho") or {}
            if not isinstance(fuku, dict): fuku = {}
            for umaban_s, fin in fo.items():
                ub = int(umaban_s)
                rows.append(dict(date=dt, place=place, rno=rno, umaban=ub, finish=int(fin),
                                 tansho_ret=(tansho if int(fin) == 1 and tansho else 0),
                                 fukusho_ret=int(fuku.get(umaban_s, 0) or 0)))
    return pd.DataFrame(rows)


def band(p):
    return ("1" if p == 1 else "2-3" if p <= 3 else "4-6" if p <= 6 else "7-9" if p <= 9 else "10+")


def main():
    tool = load_tool(); odds = load_odds(); res = load_results()
    df = tool.merge(odds, on=["date", "place", "rno", "umaban"], how="inner") \
             .merge(res, on=["date", "place", "rno", "umaban"], how="inner")
    print(f"評価対象: {DATES}  結合 per-horse={len(df)}  レース={df.groupby(['date','place','rno']).ngroups}")
    df["top3"] = (df["finish"] <= 3).astype(int); df["win"] = (df["finish"] == 1).astype(int)
    df["pop_band"] = df["pop"].map(band)
    df["tan_roi"] = df["tansho_ret"] / 100.0   # 100円賭けのリターン倍率
    df["fuku_roi"] = df["fukusho_ret"] / 100.0
    print(f"全体: 複勝率={df['top3'].mean()*100:.1f}%  勝率={df['win'].mean()*100:.1f}%  "
          f"単回={df['tan_roi'].mean()*100:.1f}%  複回={df['fuku_roi'].mean()*100:.1f}%")

    # 人気帯ベースライン
    print("\n=== 人気帯ベースライン ===")
    g = df.groupby("pop_band").agg(n=("top3","size"), 複勝率=("top3","mean"),
        勝率=("win","mean"), 単回=("tan_roi","mean"), 複回=("fuku_roi","mean"))
    for c in ["複勝率","勝率","単回","複回"]: g[c]=(g[c]*100).round(1)
    print(g.reindex(["1","2-3","4-6","7-9","10+"]).to_string())

    # シグナル×人気帯: 上乗せ検証
    def signal_lift(col, flag_fn, label):
        s = df[df[col].notna()].copy(); s["sig"] = s[col].map(flag_fn)
        print(f"\n=== {label}: 人気帯内での上乗せ(シグナル群 vs 帯全体) ===")
        out=[]
        for b in ["1","2-3","4-6","7-9","10+"]:
            sub=s[s["pop_band"]==b]; sig=sub[sub["sig"]]; base=sub
            if len(sig)<10:
                out.append((b,len(sig),None,None,None,None)); continue
            out.append((b,len(sig),
                round(sig["top3"].mean()*100,1), round(base["top3"].mean()*100,1),
                round(sig["fuku_roi"].mean()*100,1), round(base["fuku_roi"].mean()*100,1)))
        r=pd.DataFrame(out,columns=["人気帯","シグnaln","シグ複勝率","帯複勝率","シグ複回","帯複回"])
        print(r.to_string(index=False))

    signal_lift("rsim_ag3_rank", lambda v: v<=3, "rsim予測上がり 上位3位内")
    signal_lift("simgood_rate", lambda v: v>=0.5, "類似馬場好走率 ≥0.5")

    # 単回/複回: シグナル分位別(n付き)
    def roi_by_q(col,label,q=5):
        s=df[df[col].notna()].copy()
        try: s["q"]=pd.qcut(s[col].rank(method="first"),q,labels=False)
        except: print(f"[{label}] 分位不可"); return
        g=s.groupby("q").agg(n=("top3","size"),複勝率=("top3","mean"),単回=("tan_roi","mean"),
                             複回=("fuku_roi","mean"),平均=(col,"mean"))
        for c in ["複勝率","単回","複回"]: g[c]=(g[c]*100).round(1)
        g["平均"]=g["平均"].round(2)
        print(f"\n=== {label} 分位別 単回/複回 ===\n"+g.to_string())
    roi_by_q("rsim_ag3_rank","rsim予測上がり順位")
    roi_by_q("simgood_rate","類似馬場好走率")

    # rsim残余予測力: 過去平均上がり順位を統制
    print("\n=== rsim予測上がりの残余予測力(過去平均上がり順位を統制) ===")
    s=df[df["rsim_ag3_rank"].notna() & df["past_agari_rank_mean"].notna()].copy()
    print(f"corr(rsim_ag3_rank, past_agari_rank_mean) = {s['rsim_ag3_rank'].corr(s['past_agari_rank_mean']):.3f}  n={len(s)}")
    # past を3分位で層化し、その中で rsim上位3 vs 他の複勝率差
    s["past_q"]=pd.qcut(s["past_agari_rank_mean"].rank(method="first"),3,labels=["速い","中","遅い"])
    s["rsim_top"]=s["rsim_ag3_rank"]<=3
    piv=s.groupby(["past_q","rsim_top"]).agg(n=("top3","size"),複勝率=("top3","mean")).reset_index()
    piv["複勝率"]=(piv["複勝率"]*100).round(1)
    print(piv.to_string(index=False))
    # 各past層でのrsim上位の上乗せ
    print("\n過去平均上がり層ごとの rsim上位3の複勝率上乗せ:")
    for pq in ["速い","中","遅い"]:
        a=s[(s["past_q"]==pq)&s["rsim_top"]]["top3"]; b=s[(s["past_q"]==pq)&~s["rsim_top"]]["top3"]
        if len(a)>=10 and len(b)>=10:
            print(f"  過去{pq}: rsim上位 {a.mean()*100:.1f}%(n={len(a)}) vs 非上位 {b.mean()*100:.1f}%(n={len(b)})  差 {(a.mean()-b.mean())*100:+.1f}pt")


if __name__ == "__main__":
    main()
