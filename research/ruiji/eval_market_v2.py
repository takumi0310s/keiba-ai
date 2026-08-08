# -*- coding: utf-8 -*-
"""
Phase C-3 本評価: 全20日・確定オッズでクリーン信号の市場超過を検証。
rsim は §10.5 のリークで除外。評価対象=過去走結果のみ使うクリーン信号:
  simgood_rate / 各フィルタ(好走/類似度/上がり順/同脚質/1着のみ) / 組合せ。
入力:
  research/ruiji/raw_results/<date>.json  … 確定 着順/単勝オッズ/人気/複勝払戻(fetch_results.py)
  research/ruiji/raw/<date>/scatter_*.json … ツール特徴
data/ は読まない。research/ruiji 完結。
"""
import sys, os, glob, json, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from tool_filters import (build_sim, calc_sim_turf, calc_sim_dirt, filter_horses, DEFAULT_F,
    R_DR, R_RES, R_TD, R_AG3, R_AGR, R_CV, R_MO)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "raw_results")
PLACE = {'札幌':'01','函館':'02','福島':'03','新潟':'04','東京':'05',
         '中山':'06','中京':'07','京都':'08','阪神':'09','小倉':'10'}
def nz(s): return unicodedata.normalize("NFKC", str(s or "")).strip()


def horse_feat(race, h):
    is_turf = race["s"] == "芝"; tx, ty = race.get("tx"), race.get("ty"); date = int(race["date"])
    runs = [r for r in (h.get("runs") or []) if len(r) >= 15 and int(r[R_DR]) < date]
    pcts = []; good = tot = 0
    for r in runs[:10]:
        if r[R_MO] is None or (is_turf and r[R_CV] is None): continue
        pct = calc_sim_turf(tx, ty, r[R_CV], r[R_MO]) if is_turf else calc_sim_dirt(ty, r[R_MO])
        pcts.append(pct)
        if pct >= 50:
            tot += 1
            if (r[R_RES] is not None and r[R_RES] <= 3) or (r[R_TD] is not None and r[R_TD] <= 0.6):
                good += 1
    return dict(sim_pct_max=max(pcts) if pcts else np.nan,
                simgood_rate=(good/tot) if tot else np.nan, n_similar=tot)


# 評価する単一フィルタ(全てクリーン=過去走結果のみ)
FILTERS = {
    "好走":        {"good": True},
    "類似70↑":     {"sim": 70},
    "上がり3位内":  {"agari": 3},
    "同脚質":       {"ky": True},
    "1着のみ":      {"oth": {"first": 1}},
    "好走+類似70":  {"good": True, "sim": 70},
    "好走+同脚質":  {"good": True, "ky": True},
}


def load_tool_with_filters():
    rows = []
    for fp in glob.glob(os.path.join(HERE, "raw/*/scatter_*.json")):
        d = json.load(open(fp, encoding="utf-8")); rc = d["race"]
        date = str(rc["date"]); place = PLACE.get(nz(rc["v"])); rno = int(rc["no"])
        horses = d.get("horses", [])
        # フィルタ別の合格馬番集合
        passed = {}
        for name, fd in FILTERS.items():
            F = {**DEFAULT_F, **{k: v for k, v in fd.items() if k != "oth"}}
            F["oth"] = fd.get("oth", {})
            F["grades"] = {}
            try:
                passed[name] = set(filter_horses(rc, horses, F))
            except Exception:
                passed[name] = set()
        for h in horses:
            num = str(h.get("num"))
            if not num.isdigit(): continue
            feat = horse_feat(rc, h)
            row = dict(date=date, place=place, rno=rno, umaban=int(num),
                       surf=rc["s"], dist=rc["dist"], **feat)
            for name in FILTERS:
                row[f"f_{name}"] = num in passed[name]
            rows.append(row)
    return pd.DataFrame(rows)


def load_results():
    rows = []
    for fp in glob.glob(os.path.join(RES, "*.json")):
        date = os.path.basename(fp)[:8]
        arr = json.load(open(fp, encoding="utf-8"))
        for r in arr:
            rid = str(r["race_id"]); place = rid[4:6]; rno = int(rid[-2:])
            fo = r.get("finish_order") or {}; odds = r.get("odds") or {}
            pop = r.get("popularity") or {}; fuku = (r.get("payouts") or {}).get("fukusho") or {}
            if not isinstance(fuku, dict): fuku = {}
            for ub_s, fin in fo.items():
                ub = int(ub_s)
                rows.append(dict(date=date, place=place, rno=rno, umaban=ub, finish=int(fin),
                                 odds=odds.get(ub_s) or odds.get(int(ub_s)),
                                 pop=pop.get(ub_s) or pop.get(int(ub_s)),
                                 fuku=int(fuku.get(str(ub), fuku.get(ub, 0)) or 0)))
    return pd.DataFrame(rows)


def band(p):
    if pd.isna(p): return "?"
    p = int(p)
    return ("1" if p == 1 else "2-3" if p <= 3 else "4-6" if p <= 6 else "7-9" if p <= 9 else "10+")


def boot_ci(df, value_col, race_key=["date","place","rno"], n=2000, seed=42):
    """レースクラスタ bootstrap で平均のCIを返す(0-1スケール→%)。"""
    rng = np.random.RandomState(seed)
    groups = [g[value_col].values for _, g in df.groupby(race_key)]
    if not groups: return (np.nan, np.nan)
    means = []
    for _ in range(n):
        idx = rng.randint(0, len(groups), len(groups))
        vals = np.concatenate([groups[i] for i in idx])
        means.append(vals.mean())
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))


def main():
    tool = load_tool_with_filters(); res = load_results()
    print(f"tool 馬={len(tool)}  results 馬={len(res)}  results 日={res['date'].nunique()}")
    df = tool.merge(res, on=["date", "place", "rno", "umaban"], how="inner")
    nR = df.groupby(["date","place","rno"]).ngroups
    print(f"結合 per-horse={len(df)}  レース={nR}  結合率={len(df)/len(tool)*100:.1f}%")
    df["top3"] = (df["finish"] <= 3).astype(int); df["win"] = (df["finish"] == 1).astype(int)
    df["pop_band"] = df["pop"].map(band)
    df["tan_roi"] = np.where(df["win"] == 1, df["odds"].fillna(0), 0.0)      # 単勝: 100円→odds倍
    df["fuku_roi"] = df["fuku"].fillna(0) / 100.0                            # 複勝: 払戻/100
    print(f"全体: 複勝率={df['top3'].mean()*100:.1f}% 勝率={df['win'].mean()*100:.1f}% "
          f"単回={df['tan_roi'].mean()*100:.1f}% 複回={df['fuku_roi'].mean()*100:.1f}%")

    print("\n=== 人気帯ベースライン ===")
    g = df.groupby("pop_band").agg(n=("top3","size"),複勝率=("top3","mean"),
        単回=("tan_roi","mean"),複回=("fuku_roi","mean"))
    for c in ["複勝率","単回","複回"]: g[c]=(g[c]*100).round(1)
    print(g.reindex(["1","2-3","4-6","7-9","10+"]).to_string())

    # 各フィルタ: 全体 & CI
    print("\n=== 各フィルタ選択馬の成績(全体) ===")
    out=[]
    for name in FILTERS:
        s = df[df[f"f_{name}"]]
        if len(s) < 20:
            out.append((name,len(s),None,None,None,None,None)); continue
        ci = boot_ci(s, "fuku_roi")
        out.append((name,len(s),round(s["top3"].mean()*100,1),round(s["tan_roi"].mean()*100,1),
                    round(s["fuku_roi"].mean()*100,1),round(ci[0]*100,1),round(ci[1]*100,1)))
    r=pd.DataFrame(out,columns=["フィルタ","n","複勝率","単回","複回","複回CI下","複回CI上"])
    print(r.to_string(index=False))
    print("※ ベース複回 %.1f%% / 複回CIが100%%を上抜けなら市場超過の候補" % (df['fuku_roi'].mean()*100))

    # simgood_rate の人気帯内上乗せ
    print("\n=== simgood_rate(≥0.5) 人気帯内上乗せ ===")
    s = df[df["simgood_rate"].notna()].copy(); s["sig"] = s["simgood_rate"] >= 0.5
    rows=[]
    for b in ["1","2-3","4-6","7-9","10+"]:
        sub=s[s["pop_band"]==b]; sig=sub[sub["sig"]]
        if len(sig)<10: rows.append((b,len(sig),None,None,None,None)); continue
        rows.append((b,len(sig),round(sig["top3"].mean()*100,1),round(sub["top3"].mean()*100,1),
                     round(sig["fuku_roi"].mean()*100,1),round(sub["fuku_roi"].mean()*100,1)))
    print(pd.DataFrame(rows,columns=["人気帯","シグn","シグ複勝率","帯複勝率","シグ複回","帯複回"]).to_string(index=False))

    # フィルタ×人気帯: 複回上乗せ(選択 vs 帯)
    print("\n=== 好走+類似70 の人気帯内 複回上乗せ ===")
    s=df.copy()
    rows=[]
    for b in ["1","2-3","4-6","7-9","10+"]:
        sub=s[s["pop_band"]==b]; sig=sub[sub["f_好走+類似70"]]
        if len(sig)<10: rows.append((b,len(sig),None,None)); continue
        rows.append((b,len(sig),round(sig["fuku_roi"].mean()*100,1),round(sub["fuku_roi"].mean()*100,1)))
    print(pd.DataFrame(rows,columns=["人気帯","n","シグ複回","帯複回"]).to_string(index=False))

    df.to_parquet(os.path.join(HERE,"merged","ruiji_market_eval.parquet"), index=False)
    print("\n保存: merged/ruiji_market_eval.parquet", len(df),"行")


if __name__ == "__main__":
    main()
