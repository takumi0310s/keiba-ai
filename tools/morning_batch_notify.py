# -*- coding: utf-8 -*-
"""朝の一括通知 (2026-08-11 通知再整備 item2。09:30、旧 RaceAutoNotify 08:45 の置換)。

★予測ロジック不変★ — daily_predict (08:00) が生成済みの feat_dump を読むだけの表示層。
送信条件: T1v2 監査 PASS を確認してから送信。BLOCK フラグありなら
「本日は監査NGのため予測停止」を通知して終了。

内容:
  冒頭サマリー: 総レース数 / 買い・見送り数 / T1v2 結果 / 供給鮮度 (KYI/SED)
  1レースごと (買い対象のみ個別 embed、見送りはサマリーに理由付き一覧):
    a. 発走時刻・場・R・クラス・頭数
    b. V15スコア上位6頭 (馬番/馬名/score)
    c. フォーメーション買い目 (📝 PAPER観察中バッジ・9月ゲートまで実投票なし)
    d. 特徴量詳細: 非ゼロ特徴数/145・KYI結合率・gain上位6特徴 (s2b同形式)・premium欠損数

usage:
  python tools/morning_batch_notify.py [--date YYYYMMDD] [--dry-run] [--test]
    --dry-run: 送信せず stdout のみ / --test: test チャンネルに【TEST】付きで送信
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, pickle, sys, time
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tools"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

_keep = [sys.stdout, sys.stderr]
from strategy_filters import evaluate_bet_decision, build_trio_bets  # noqa: E402
from notify import send_discord  # noqa: E402
_keep += [sys.stdout, sys.stderr]

DUMP = os.path.join(BASE, "data", "v15_feat_dump")
AUDIT = os.path.join(BASE, "data", "T1v2_audit")
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
PAPER_BADGE = "📝 PAPER観察中 — 9月ゲートまで実投票なし"
# netkeiba premium 系 (解約で恒常default。DEPENDENCY_AUDIT §1)
PREMIUM_FEATS = {
    'index_max_filled', 'index_run1_filled', 'index_avg5_filled', 'stable_comment_score',
    'wood_best_4f_filled', 'sakaro_best_4f_filled', 'sakaro_best_3f_filled', 'time_1f_last_filled',
    'training_intensity_enc', 'wood_count_2w', 'total_training_count', 'training_per_dist',
    'has_training', 'has_wood_training', 'has_sakaro_training', 'training_time_filled',
}


def _name_ok(nm):
    return all(0x3040 <= ord(c) <= 0x30ff or 0x4e00 <= ord(c) <= 0x9fff
               or c.isascii() or c in "ー・" for c in str(nm)) and str(nm) not in ("", "nan")


_KYI_NAMES = None


def kyi_name(rid, umaban):
    """dump の馬名が mojibake (6/20-8/9 の EUC-JP バグ期) の場合の KYI フォールバック。"""
    global _KYI_NAMES
    if _KYI_NAMES is None:
        try:
            k = pd.read_csv(os.path.join(BASE, "data", "jrdb_kyi.csv"),
                            usecols=["nk_race_id", "馬番", "馬名"], dtype=str,
                            encoding="utf-8-sig")
            k = k[k["nk_race_id"].astype(str).str.startswith("2026")]
            _KYI_NAMES = {(str(r["nk_race_id"]), str(int(float(r["馬番"])))): str(r["馬名"]).strip()
                          for _, r in k.iterrows() if pd.notna(r["馬番"])}
        except Exception:
            _KYI_NAMES = {}
    return _KYI_NAMES.get((str(rid), str(int(umaban))), None)


def load_model_meta():
    with gzip.open(os.path.join(BASE, "keiba_model_v15_central.pkl.gz"), "rb") as f:
        m = pickle.load(f)
    feats = m["features"]
    gi = m["model"].feature_importance(importance_type="gain")
    gain = {feats[i]: float(gi[i]) for i in range(len(feats))}  # 位置対応 (Column_N 対策)
    top6 = sorted(gain, key=lambda k: -gain[k])[:6]
    return feats, top6


def classify_condition(nh, dist, ce):
    heavy = (ce is not None) and (ce >= 2)
    if nh <= 7: return "E"
    if dist <= 1400: return "D"
    if 8 <= nh <= 14 and dist >= 1600 and not heavy: return "A"
    if 8 <= nh <= 14 and dist >= 1600 and heavy: return "B"
    if nh >= 15 and dist >= 1600 and not heavy: return "C"
    return "X"


def class_label(race_name):
    rn = str(race_name or "")
    for tok in ["G1", "GⅠ", "G2", "GⅡ", "G3", "GⅢ"]:
        if tok in rn: return tok.replace("Ⅰ", "1").replace("Ⅱ", "2").replace("Ⅲ", "3")
    for tok in ["(L)", "オープン", "OP"]:
        if tok in rn: return "OP/L"
    for tok in ["3勝", "2勝", "1勝", "未勝利", "新馬"]:
        if tok in rn: return tok
    return "—"


def fmt_race(df, feats, gain_top6):
    """1レース分の dump df → 表示要素 dict。"""
    rid = str(df["race_id"].iloc[0])
    course = str(df.get("course", pd.Series([np.nan])).iloc[0])
    if course in ("nan", "", "None") or pd.isna(course):
        course = PLACE.get(rid[4:6], "?")
    _rv = pd.to_numeric(df["race_num"].iloc[0], errors="coerce") if "race_num" in df.columns else np.nan
    rno = int(_rv) if pd.notna(_rv) else int(rid[-2:])
    rn = str(df["race_name"].iloc[0]) if "race_name" in df.columns else ""
    st = str(df["start_time"].iloc[0]) if "start_time" in df.columns else "?"
    _dv = pd.to_numeric(df.get("距離(m)", df.get("distance")).iloc[0], errors="coerce")
    dist = int(_dv) if pd.notna(_dv) else 0
    nh = len(df)
    ce = pd.to_numeric(df["condition_enc"].iloc[0], errors="coerce") if "condition_enc" in df.columns else None
    ce = None if pd.isna(ce) else int(ce)
    cond = classify_condition(nh, dist, ce)
    should, reason = evaluate_bet_decision(rn, course, dist, cond)
    # top6 (AI順位優先。馬番/スコア欠損行=取消等は除外)
    s = df.copy()
    s["_sc"] = pd.to_numeric(s["スコア"], errors="coerce")
    s["_ub"] = pd.to_numeric(s["馬番"], errors="coerce")
    s = s.dropna(subset=["_sc", "_ub"])
    if "AI順位" in s.columns and s["AI順位"].notna().all():
        s = s.sort_values("AI順位", key=lambda x: pd.to_numeric(x, errors="coerce"))
    else:
        s = s.sort_values("_sc", ascending=False)
    top = s.head(6)
    nums = pd.to_numeric(top["馬番"], errors="coerce").dropna().astype(int).tolist()
    # 買い目
    bets_str = ""
    if should and len(nums) >= 6:
        if cond == "E":
            bets_str = f"馬連 {nums[0]}-{nums[1]} / {nums[0]}-{nums[2]} (2点)"
        else:
            bets = build_trio_bets(*nums[:6], apply_c3=False)
            bets_str = "三連複 " + "; ".join("-".join(map(str, b)) for b in bets) + " (7点)"
    # 特徴量詳細 (d)
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    nonzero = int((X.fillna(0) != 0).any(axis=0).sum())
    kyi_join = float((pd.to_numeric(df["jrdb_idm"], errors="coerce") != 50).mean()) if "jrdb_idm" in df.columns else np.nan
    prem_dead = int(sum(1 for f in PREMIUM_FEATS if f in X.columns and X[f].nunique() <= 1))
    g6 = "\n".join(f"  {f} = {pd.to_numeric(top[f].iloc[0], errors='coerce'):.2f}"
                   if f in top.columns else f"  {f} = n/a" for f in gain_top6)
    def _disp_name(r):
        nm = str(r.get("馬名", ""))
        if not _name_ok(nm):
            nm = kyi_name(rid, r["_ub"]) or nm  # KYI フォールバック (mojibake dump 対策)
        return nm[:12]
    top6_str = "\n".join(f"  {int(r['_ub'])} {_disp_name(r)}  {r['_sc']:.3f}"
                         for _, r in top.iterrows())
    return dict(rid=rid, course=course, rno=rno, rn=rn, st=st, dist=dist, nh=nh,
                cond=cond, should=should, reason=reason, cls=class_label(rn),
                top6=top6_str, bets=bets_str, nonzero=nonzero, kyi_join=kyi_join,
                prem_dead=prem_dead, gain6=g6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    date = a.date
    channel = "test" if a.test else "bets"
    prefix = "【TEST】" if a.test else ""

    def send(title, msg, color="green"):
        if a.dry_run:
            print(f"\n----- [{channel}] {title} -----\n{msg}")
        else:
            send_discord(prefix + title, msg, color=color, channel=channel, dedup_window_sec=0)
            time.sleep(0.8)

    # T1v2 ゲート
    flag = os.path.join(AUDIT, "BLOCK_NEXT.flag")
    if os.path.exists(flag):
        reason = open(flag, encoding="utf-8").read()[:400]
        send(f"⛔ {date} 本日は監査NGのため予測停止", f"T1v2 CRITICAL:\n{reason}", color="red")
        print("BLOCK flag → 停止通知のみ送信")
        return 0
    audit_p = os.path.join(AUDIT, f"{date}.json")
    t1v2 = "未実施"
    if os.path.exists(audit_p):
        aj = json.load(open(audit_p, encoding="utf-8"))
        t1v2 = f"{aj.get('verdict')} (jrdb定数 {aj.get('jrdb_const_n')}/40)"
    sh = sorted(glob.glob(os.path.join(AUDIT, "supply_health_*.json")))
    kyi_d = sed_d = "?"
    if sh:
        h = json.load(open(sh[-1], encoding="utf-8"))
        kyi_d, sed_d = h.get("kyi_latest_file", "?"), h.get("sed_latest", "?")

    files = [f for f in sorted(glob.glob(os.path.join(DUMP, date, "*.parquet")))
             if os.path.getsize(f) > 0]
    if not files:
        send(f"{date} 朝通知", "本日の feat_dump なし (非開催 or 予測未実行)", color="yellow")
        return 0
    feats, gain_top6 = load_model_meta()
    races = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if len(df) < 2 or "スコア" not in df.columns:
            continue
        try:
            races.append(fmt_race(df, feats, gain_top6))
        except Exception as e:
            print(f"[WARN] fmt失敗 {os.path.basename(fp)}: {e}")
    races.sort(key=lambda r: r["st"])
    buys = [r for r in races if r["should"]]
    skips = [r for r in races if not r["should"]]

    # 冒頭サマリー
    skip_lines = "\n".join(f"  ⚪ {r['course']}{r['rno']}R {r['st']} — {r['reason']}" for r in skips)
    send(f"🏇 {date} 朝一括通知 (09:30)",
         f"{PAPER_BADGE}\n"
         f"総レース {len(races)}R = 🟢買い対象 {len(buys)} / ⚪見送り {len(skips)}\n"
         f"T1v2: {t1v2}\n供給鮮度: KYI={kyi_d} / SED={sed_d}\n"
         f"\n--- 見送り一覧 ---\n{skip_lines or '  なし'}",
         color="blue")

    for r in buys:
        kyi_pct = f"{r['kyi_join']*100:.0f}%" if not np.isnan(r["kyi_join"]) else "n/a"
        msg = (f"{r['rn']} {r['dist']}m {r['nh']}頭 条件{r['cond']} [{r['cls']}]\n"
               f"{PAPER_BADGE}\n"
               f"\n【V15上位6頭】\n{r['top6']}\n"
               f"\n【買い目(paper)】\n{r['bets']}\n"
               f"\n【特徴量】非ゼロ {r['nonzero']}/145・KYI結合 {kyi_pct}・premium欠損 {r['prem_dead']}\n"
               f"gain上位6 (top1馬の値):\n{r['gain6']}")
        send(f"🟢 {r['course']}{r['rno']}R 発走{r['st']}", msg, color="green")
    print(f"送信完了: サマリー1 + 買い{len(buys)}embed (見送り{len(skips)}はサマリー内)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
