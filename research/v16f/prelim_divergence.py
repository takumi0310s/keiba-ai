# -*- coding: utf-8 -*-
"""
v16-fundamental 先行実験(コスト最小): 新学習の前に、既存 Pattern A で
「モデル確率 vs 市場確率の乖離上位馬」を V15 三連複フォーメーションに
注入/入替した場合の ROI 変化を 640R(実際は JRDB生存6日=~186R)でバックテスト。

方針(V15 pkl・運用系 不変・research/v16f 完結):
- 特徴量は既存 data/v15_feat_dump/<date>/<race_id>.parquet を流用(再計算なし)。
- ★JRDB生存日のみ★: 6/27以降は JRDB全滅で Pattern A も V15本番も劣化 → 除外。
  使用可能= 6/6,6/7,6/13,6/14,6/20,6/21 (Pattern A corr(本番)=0.98-0.99で健全)。
- 結果/オッズ/配当は research/ruiji/raw_results/<date>.json を流用。
- V15フォーメーション= 本番スコア上位6の三連複7点(TOP1軸-[T2,T3]-[T2..T6])。
  乖離注入= 3列目に乖離上位馬を「入替(n6→乖離馬)」または「追加(8点化)」。
data/ へは書き込まない。
"""
import sys, os, glob, json, gzip, pickle
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, xgboost as xgbm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DUMP = os.path.join(ROOT, "data", "v15_feat_dump")
RES = os.path.join(ROOT, "research", "ruiji", "raw_results")
DATES = ["20260606", "20260607", "20260613", "20260614", "20260620", "20260621"]

with gzip.open(os.path.join(ROOT, "keiba_model_v15_central.pkl.gz"), "rb") as f:
    M = pickle.load(f)
FEATS = M["features"]; LGB = M["model"]; XGB = M["xgb_model"]; W = M["ensemble_weights"]


def predict_pA(df):
    X = df[FEATS]
    return W["lgb"] * LGB.predict(X) + W["xgb"] * XGB.predict(xgbm.DMatrix(X))


def load_results():
    res = {}
    for dt in DATES:
        fp = os.path.join(RES, f"{dt}.json")
        if not os.path.exists(fp): continue
        for r in json.load(open(fp, encoding="utf-8")):
            fo = {int(k): int(v) for k, v in (r.get("finish_order") or {}).items()}
            top3 = tuple(sorted([u for u, f in fo.items() if f <= 3]))
            res[str(r["race_id"])] = {
                "top3": top3 if len(top3) == 3 else None,
                "trio_pay": (r.get("payouts") or {}).get("trio", 0),
            }
    return res


def trio_bets(nums):
    """TOP1軸-[n2,n3]-[n2..n6] 7点(集合)。nums=上位6馬番。"""
    n1 = nums[0]; second = nums[1:3]; third = nums[1:6]
    bets = set()
    for s in second:
        for t in third:
            c = tuple(sorted({n1, s, t}))
            if len(c) == 3: bets.add(c)
    return bets


def eval_formation(bets, top3, trio_pay):
    """1レースの三連複: 投資=100*点数, 払戻=当該winが含まれれば trio_pay。"""
    inv = 100 * len(bets)
    ret = trio_pay if (top3 is not None and top3 in bets) else 0
    hit = 1 if ret > 0 else 0
    return inv, ret, hit


def main():
    RESULTS = load_results()
    rows = []
    for dt in DATES:
        for fp in sorted(glob.glob(os.path.join(DUMP, dt, "*.parquet"))):
            if os.path.getsize(fp) == 0: continue
            try: df = pd.read_parquet(fp)
            except Exception: continue
            if len(df) < 3 or "スコア" not in df.columns: continue
            rid = str(df["race_id"].iloc[0])
            if rid not in RESULTS or RESULTS[rid]["top3"] is None: continue
            df = df.copy()
            df["pA"] = predict_pA(df)
            df["umaban"] = pd.to_numeric(df["馬番"], errors="coerce")
            df = df.dropna(subset=["umaban"]); df["umaban"] = df["umaban"].astype(int)
            if "人気順位" in df.columns:
                df["pop"] = pd.to_numeric(df["人気順位"], errors="coerce")
            else:
                df["pop"] = pd.to_numeric(df["単勝オッズ"], errors="coerce").rank(method="min")
            df["pA_rank"] = df["pA"].rank(ascending=False, method="first")
            # V15本番 上位6(スコア降順)
            v15 = df.sort_values("スコア", ascending=False)
            base_nums = v15["umaban"].tolist()[:6]
            if len(base_nums) < 6: continue
            # 乖離馬: Pattern A が高評価(pA_rank<=6)だが 市場薄い(pop>=6)、V15上位6外
            div = df[(df["pA_rank"] <= 6) & (df["pop"] >= 6) & (~df["umaban"].isin(base_nums))]
            div = div.sort_values("pA_rank")  # Pattern A で最良の乖離馬
            dh = int(div["umaban"].iloc[0]) if len(div) else None

            tp = RESULTS[rid]; top3 = tp["top3"]; pay = tp["trio_pay"]
            # A. 現行 V15 formation(7点)
            base_b = trio_bets(base_nums)
            bi, br, bh = eval_formation(base_b, top3, pay)
            # B. 入替: n6 → 乖離馬(dhがあれば)
            if dh is not None:
                swap_nums = base_nums[:5] + [dh]
                swap_b = trio_bets(swap_nums)
            else:
                swap_b = base_b
            si, sr, sh = eval_formation(swap_b, top3, pay)
            # C. 追加: 3列目に乖離馬(8点化) = top1軸-[n2,n3]-[n2..n6,dh]
            if dh is not None:
                add_nums = base_nums + [dh]  # 7要素
                n1 = add_nums[0]; second = add_nums[1:3]; third = add_nums[1:7]
                add_b = set()
                for s in second:
                    for t in third:
                        c = tuple(sorted({n1, s, t}))
                        if len(c) == 3: add_b.add(c)
            else:
                add_b = base_b
            ai, ar, ah = eval_formation(add_b, top3, pay)

            rows.append(dict(date=dt, rid=rid, has_div=(dh is not None),
                             base_inv=bi, base_ret=br, base_hit=bh,
                             swap_inv=si, swap_ret=sr, swap_hit=sh,
                             add_inv=ai, add_ret=ar, add_hit=ah))
    d = pd.DataFrame(rows)
    print(f"評価レース: {len(d)}  乖離馬あり: {d['has_div'].sum()}\n")

    def summ(name, inv, ret, hit):
        I, R = d[inv].sum(), d[ret].sum()
        print(f"  {name:14} n={len(d)} 的中={d[hit].sum():3d}/{len(d)} ({d[hit].mean()*100:4.1f}%) "
              f"投資¥{I:,} 払戻¥{R:,} ROI={R/I*100:5.1f}%")
    print("=== 全レース ===")
    summ("現行V15(7点)", "base_inv", "base_ret", "base_hit")
    summ("入替(n6→乖離)", "swap_inv", "swap_ret", "swap_hit")
    summ("追加(8点化)", "add_inv", "add_ret", "add_hit")

    # 乖離馬があったレースのみ(injectが実際に効いた部分集合)
    dd = d[d["has_div"]]
    if len(dd):
        print(f"\n=== 乖離馬ありレースのみ (n={len(dd)}) ===")
        for name, inv, ret, hit in [("現行V15","base_inv","base_ret","base_hit"),
                                     ("入替","swap_inv","swap_ret","swap_hit"),
                                     ("追加","add_inv","add_ret","add_hit")]:
            I, R = dd[inv].sum(), dd[ret].sum()
            print(f"  {name:10} 的中={dd[hit].sum():3d}/{len(dd)} ({dd[hit].mean()*100:4.1f}%) ROI={R/I*100:5.1f}%")
        # 乖離馬が実際にtop3に来た率
        print(f"\n  ※ 乖離馬あり {len(dd)}R / 全{len(d)}R")


if __name__ == "__main__":
    main()
