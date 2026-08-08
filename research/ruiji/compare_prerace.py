# -*- coding: utf-8 -*-
"""
§12 真rsimの一次観測: 8/9 の事前rsim(snapshots/20260809_prerace/=8/8取得=発走前=リーク不能)
と、発走後の実結果を突合。リーク監査の唯一の対照。
n不足(1日~35R)で結論は出ない前提の参考記録。驚くほど良くても単独で再開判断はしない。

実行(結果確定後の夜): python compare_prerace.py
  → 8/9 結果を netkeiba から取得(fetch_results流用)、事前rsimと比較、
    research/ruiji/_section12.md に §12 マークダウンを書き出す。
"""
import sys, os, glob, json, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from tool_filters import R_DR, R_AG3
import fetch_results as FR

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "snapshots", "20260809_prerace")
PLACE = {'札幌':'01','函館':'02','福島':'03','新潟':'04','東京':'05',
         '中山':'06','中京':'07','京都':'08','阪神':'09','小倉':'10'}
def nz(s): return unicodedata.normalize("NFKC", str(s or "")).strip()


def fetch_0809_results():
    """8/9 の確定結果を取得(発走後・夜に実行)。"""
    rids = FR.race_ids_for("20260809")
    out = {}
    for rid in rids:
        try:
            r = FR.parse_result(rid)
        except Exception:
            r = None
        if r and r.get("finish_order"):
            out[rid] = r
    return out


def main():
    res = fetch_0809_results()
    nres = sum(len(r["finish_order"]) for r in res.values())
    print(f"8/9 結果取得: {len(res)}レース {nres}頭")
    if not res:
        print("結果未確定 — 夜に再実行してください")
        return

    # 事前rsim(snapshot) を読む
    rows = []
    for fp in glob.glob(os.path.join(SNAP, "scatter_*.json")):
        d = json.load(open(fp, encoding="utf-8")); rc = d["race"]
        place = PLACE.get(nz(rc["v"])); rno = int(rc["no"])
        ag3 = (d.get("rsim") or {}).get("ag3") or {}
        # 該当race_id を結果から特定(place+rno一致)
        rr = next((r for rid, r in res.items() if rid[4:6] == place and int(rid[-2:]) == rno), None)
        if rr is None:
            continue
        for h in d.get("horses", []):
            num = str(h.get("num"))
            if not num.isdigit(): continue
            ub = int(num)
            pred = ag3.get(num)
            act = (rr.get("agari") or {}).get(str(ub))
            fin = (rr.get("finish_order") or {}).get(str(ub)) or (rr.get("finish_order") or {}).get(ub)
            pop = (rr.get("popularity") or {}).get(str(ub))
            pa = [r[R_AG3] for r in h.get("runs", [])
                  if len(r) >= 15 and str(r[R_DR]).isdigit() and int(r[R_DR]) < 20260809 and r[R_AG3] is not None]
            rows.append(dict(place=place, rno=rno, ub=ub, pred=pred, actual=act,
                             finish=fin, pop=pop, past_avg=(np.mean(pa) if pa else np.nan)))
    df = pd.DataFrame(rows)
    d2 = df.dropna(subset=["pred", "actual"])
    # rsim予測上がり順位 vs 実上がり順位・複勝率
    # 順位はレース内
    df["pred_rank"] = df.groupby(["place","rno"])["pred"].rank(method="first")
    df["top3"] = (pd.to_numeric(df["finish"], errors="coerce") <= 3).astype("Int64")

    c_pa = d2["pred"].corr(d2["actual"])
    sd = d2.dropna(subset=["past_avg"])
    c_dev = (sd["pred"] - sd["past_avg"]).corr(sd["actual"] - sd["past_avg"]) if len(sd) > 5 else np.nan
    mae = (d2["pred"] - d2["actual"]).abs().mean()
    pastmae = (sd["past_avg"] - sd["actual"]).abs().mean() if len(sd) > 5 else np.nan

    # rsim上位3(予測最速) の複勝率
    top = df[df["pred_rank"] <= 3].dropna(subset=["top3"])
    rest = df[df["pred_rank"] > 3].dropna(subset=["top3"])
    base = df.dropna(subset=["top3"])["top3"].mean()

    lines = []
    lines.append("## 12. 真rsimの一次観測（8/9・prospective・参考記録）\n")
    lines.append("8/9 の rsim は 8/8（発走前）取得の snapshot＝**リーク不能な真の事前予測**。発走後の実結果と突合。")
    lines.append(f"※ n={len(d2)}頭 / {df.groupby(['place','rno']).ngroups}R（1日のみ）。**n不足で結論は出さない参考記録**。\n")
    lines.append("### rsim.ag3(真・事前) vs 実上がり")
    lines.append(f"- corr(予測, 実際) = **{c_pa:.3f}**（過去汚染日は0.79–0.93）")
    lines.append(f"- corr(予測偏差, 実偏差) = **{c_dev:.3f}**（過去汚染日は0.67–0.89 / 8/8当日=0.44）")
    lines.append(f"- 平均誤差 |予測−実際| = **{mae:.2f}秒**（過去平均ベースライン {pastmae:.2f}秒）")
    lines.append("")
    lines.append("### rsim予測上位3頭(最速想定)の複勝率")
    lines.append(f"- 上位3: **{top['top3'].mean()*100:.1f}%**（n={len(top)}） / 非上位: {rest['top3'].mean()*100:.1f}%（n={len(rest)}） / 全体ベース {base*100:.1f}%")
    lines.append("")
    lines.append("### 解釈")
    lines.append(f"- dev相関が過去汚染日(0.67–0.89)より**大きく低い**ほど、真の事前rsimは結果を知らない＝汚染確定の傍証。")
    lines.append(f"- ただし n={len(d2)} は極小。予測力の有無は**この1日では判定不能**。単独で再開判断はしない（本リサーチはこれで終了）。")
    open(os.path.join(HERE, "_section12.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("\n→ _section12.md に書き出し完了")


if __name__ == "__main__":
    main()
