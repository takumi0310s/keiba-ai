# -*- coding: utf-8 -*-
"""
Phase C: 結果突合 + idx13/14 予測力の一次検証。
外部データに依存せず、ツール自身の20ファイルを跨いだ過去走記録から
各対象馬の「実着順」を復元する（ある日の走は、次走時の後続ファイルに
過去走として着順込みで記録される）。

出力:
  merged/ruiji_horses.parquet  … per-horse 突合テーブル
  merged/ruiji_pastruns.parquet… 全過去走(59k, idx13/14 非予測性の検証用)
  標準出力に分位表
制約: data/ は読まない(ツール内で完結)。research/ruiji 専用。
"""
import sys, glob, json, unicodedata, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from tool_filters import (build_sim, calc_sim_turf, calc_sim_dirt, style_of_run,
    R_DR, R_VEN, R_RN, R_DIST, R_SURF, R_RES, R_NH, R_TD, R_WIN, R_PASS,
    R_MAE, R_AG3, R_AGR, R_CV, R_MO)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "merged"); os.makedirs(OUT, exist_ok=True)
def nz(s): return unicodedata.normalize("NFKC", str(s or "")).strip()
FILES = sorted(glob.glob(os.path.join(HERE, "raw/*/scatter_*.json")))


def load_all():
    races = []
    for fp in FILES:
        d = json.load(open(fp, encoding="utf-8"))
        races.append(d)
    return races


def build_result_db(races):
    """全過去走を (date,ven,rn,dist,surf,horse) -> run に集約(実着順DB)。"""
    db = {}
    allruns = []
    for d in races:
        for h in d.get("horses", []):
            hn = nz(h.get("name"))
            for r in h.get("runs", []):
                if len(r) < 15:
                    continue
                key = (str(r[R_DR]), nz(r[R_VEN]), nz(r[R_RN]), str(r[R_DIST]), nz(r[R_SURF]), hn)
                db[key] = r
                allruns.append(dict(
                    date=str(r[R_DR]), ven=nz(r[R_VEN]), rn=nz(r[R_RN]),
                    dist=int(r[R_DIST]) if str(r[R_DIST]).isdigit() else None,
                    surf=nz(r[R_SURF]), horse=hn,
                    finish=r[R_RES], n_horses=r[R_NH], time_diff=r[R_TD],
                    front3f=r[R_MAE], agari3f=r[R_AG3], agari_rank=r[R_AGR],
                    cushion=r[R_CV], moisture=r[R_MO], passing=r[R_PASS]))
    return db, allruns


def horse_features(race, h):
    """当該レースの馬 h についてツール由来シグナルを計算。"""
    is_turf = race["s"] == "芝"
    tx, ty = race.get("tx"), race.get("ty")
    date = int(race["date"])
    runs = [r for r in (h.get("runs") or []) if len(r) >= 15 and int(r[R_DR]) < date]
    # 類似度 pct（各過去走→当該馬場）
    pcts = []
    good_sim = tot_sim = 0
    for r in runs[:10]:
        if r[R_MO] is None: continue
        if is_turf and r[R_CV] is None: continue
        pct = calc_sim_turf(tx, ty, r[R_CV], r[R_MO]) if is_turf else calc_sim_dirt(ty, r[R_MO])
        pcts.append(pct)
        if pct >= 50:  # 類似馬場とみなす
            tot_sim += 1
            if (r[R_RES] is not None and r[R_RES] <= 3) or (r[R_TD] is not None and r[R_TD] <= 0.6):
                good_sim += 1
    num = str(h.get("num"))
    ag3map = (race.get("rsim") or {}).get("ag3") or {}
    rsim_ag3 = ag3map.get(num)
    return dict(
        sim_pct_max=max(pcts) if pcts else None,
        n_similar=tot_sim,
        simgood_rate=(good_sim / tot_sim) if tot_sim else None,
        n_runs=len(runs),
        pred_ky=h.get("ky", ""),
        rsim_ag3=rsim_ag3,
    )


def main():
    races = load_all()
    db, allruns = build_result_db(races)
    print(f"過去走DB: {len(db)} ユニーク走 / allruns {len(allruns)}")

    rows = []
    for d in races:
        rc = d["race"]
        date, ven = str(rc["date"]), nz(rc["v"])
        rn = nz(rc.get("nfull") or rc.get("n")); dist = str(rc["dist"]); surf = nz(rc["s"])
        rsim = d.get("rsim") or {}
        ag3map = rsim.get("ag3") or {}
        # rsim 予測上がり順位（速いほど上位）
        ag3_sorted = sorted([(k, v) for k, v in ag3map.items() if v is not None], key=lambda kv: kv[1])
        ag3_rank = {k: i + 1 for i, (k, v) in enumerate(ag3_sorted)}
        for h in d.get("horses", []):
            hn = nz(h.get("name"))
            if not hn: continue
            feat = horse_features(rc, h)
            num = str(h.get("num"))
            key = (date, ven, rn, dist, surf, hn)
            act = db.get(key)
            rows.append(dict(
                date=date, ven=ven, race_no=rc["no"], race_name=rn,
                surf=surf, dist=int(dist) if dist.isdigit() else None,
                n_horses=rc.get("h"), race_cushion=rc.get("tx"), race_moisture=rc.get("ty"),
                num=num, horse=hn, waku=h.get("waku"),
                **feat,
                rsim_ag3_rank=ag3_rank.get(num),
                # 復元した実結果
                act_finish=(act[R_RES] if act else None),
                act_agari=(act[R_AG3] if act else None),
                act_agari_rank=(act[R_AGR] if act else None),
                recovered=bool(act),
            ))
    df = pd.DataFrame(rows)
    dfp = pd.DataFrame(allruns)
    df.to_parquet(os.path.join(OUT, "ruiji_horses.parquet"), index=False)
    dfp.to_parquet(os.path.join(OUT, "ruiji_pastruns.parquet"), index=False)
    print(f"保存: ruiji_horses.parquet {len(df)}行 / ruiji_pastruns.parquet {len(dfp)}行")

    cov = df["recovered"].mean()
    print(f"\n=== 実着順 復元カバレッジ: {df['recovered'].sum()}/{len(df)} = {cov*100:.1f}% ===")

    # top3(複勝圏)フラグ
    r = df[df["recovered"]].copy()
    r["top3"] = (r["act_finish"] <= 3).astype(int)
    r["win"] = (r["act_finish"] == 1).astype(int)
    print(f"復元サブセット n={len(r)}  ベース複勝率(top3)={r['top3'].mean()*100:.1f}%  勝率={r['win'].mean()*100:.1f}%")

    def decile(col, label):
        s = r[r[col].notna()].copy()
        if len(s) < 100:
            print(f"\n[{label}] n={len(s)} 少数のためスキップ"); return
        try:
            s["q"] = pd.qcut(s[col], 10, labels=False, duplicates="drop")
        except Exception:
            print(f"\n[{label}] 分位化不可(値の分散不足)"); return
        g = s.groupby("q").agg(n=("top3", "size"), 複勝率=("top3", "mean"),
                               勝率=("win", "mean"), 平均値=(col, "mean"))
        g["複勝率"] = (g["複勝率"] * 100).round(1); g["勝率"] = (g["勝率"] * 100).round(1)
        g["平均値"] = g["平均値"].round(2)
        print(f"\n[{label}] 分位×複勝率  (n={len(s)})")
        print(g.to_string())

    # ツール由来 per-horse シグナルの予測力
    decile("simgood_rate", "類似馬場での好走率(simgood_rate)")
    decile("sim_pct_max", "最大類似度(sim_pct_max)")
    decile("rsim_ag3_rank", "rsim予測上がり順位(小=速い)")
    decile("rsim_ag3", "rsim予測上がり3F(小=速い)")

    # idx13/14 の非予測性を「全過去走」で直接証明
    print("\n=== idx13/14(クッション/含水) 非予測性の直接検証(全過去走) ===")
    p = dfp[dfp["finish"].notna()].copy()
    p["top3"] = (p["finish"] <= 3).astype(int)
    for col, lab in [("cushion", "idx13 クッション値"), ("moisture", "idx14 含水率")]:
        s = p[p[col].notna()].copy()
        s["q"] = pd.qcut(s[col].rank(method="first"), 5, labels=False)
        g = s.groupby("q").agg(n=("top3", "size"), 複勝率=("top3", "mean"), 平均=(col, "mean"))
        g["複勝率"] = (g["複勝率"] * 100).round(1); g["平均"] = g["平均"].round(2)
        print(f"\n[{lab}] 5分位×複勝率(=その馬場を走った際の複勝率. 相関あれば意味あり)")
        print(g.to_string())


if __name__ == "__main__":
    main()
