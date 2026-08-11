# -*- coding: utf-8 -*-
"""T1v2 特徴量監視 (2026-08-10 実装。設計=research/v15_autopsy/T1v2_DESIGN.md)。

現行 T1 (T1_features_audit) は学習キャッシュ(静的)を監査していたため
全日 byte-identical のゾンビと化し、ライブの JRDB 全滅 (6/27 40/40定数) を
一度も検知できなかった (死因解剖 §3)。T1v2 は監査対象を
★当日ライブ feat_dump (data/v15_feat_dump/<date>/)★ に変更し、
前開催日との値ハッシュ一致 (=生成停止/ゾンビ) を即 CRITICAL とする。

チェック項目 (設計書準拠):
  A. ゾンビ検知     : 当日145特徴の値ハッシュ == 前開催日 → CRITICAL
  B. NO_DUMP        : 土日祝相当で dump 不在 → CRITICAL (平日不在は INFO skip)
  C. JRDB 定数化    : jrdb_* 特徴の定数化数 N>=10 WARN / N>=25 CRITICAL
  D. 欠損急増       : null_rate が前回監査比 +0.30 超 → WARN
  E. スコア圧縮     : 本番スコア range < 0.30 → WARN (識別能力喪失の代理)

CRITICAL 時: data/T1v2_audit/BLOCK_NEXT.flag を作成 (daily_predict が参照し
翌日予測をブロック) + Discord 通知。OK 判定の監査が通ると flag は自動クリア。

usage:
  python tools/t1v2_feature_audit.py                 # 今日を監査
  python tools/t1v2_feature_audit.py --date 20260808 # 日付指定
exit code: 0=OK/WARN/SKIP, 2=CRITICAL
"""
from __future__ import annotations
import argparse, glob, gzip, hashlib, json, os, pickle, subprocess, sys
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP_DIR = os.path.join(BASE, "data", "v15_feat_dump")
AUDIT_DIR = os.path.join(BASE, "data", "T1v2_audit")
BLOCK_FLAG = os.path.join(AUDIT_DIR, "BLOCK_NEXT.flag")
MODEL_PKL = os.path.join(BASE, "keiba_model_v15_central.pkl.gz")

JRDB_WARN_N = 10
JRDB_CRIT_N = 25
NULL_SPIKE = 0.30
SCORE_RANGE_MIN = 0.30


def load_feature_list():
    with gzip.open(MODEL_PKL, "rb") as f:
        return pickle.load(f)["features"]  # 145


def load_day(date_str, feats):
    """当日 feat_dump を連結して返す (race_id ソート・空/壊れparquetはスキップ)。"""
    ddir = os.path.join(DUMP_DIR, date_str)
    if not os.path.isdir(ddir):
        return None
    parts = []
    for fp in sorted(glob.glob(os.path.join(ddir, "*.parquet"))):
        try:
            if os.path.getsize(fp) == 0:
                continue
            p = pd.read_parquet(fp)
        except Exception:
            continue
        if len(p) == 0 or "race_id" not in p.columns:
            continue
        parts.append(p)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["race_id"] + (["horse_num"] if "horse_num" in df.columns else [])).reset_index(drop=True)


def value_hash(df, feats):
    """145特徴の値 + race_id 連結の md5。生成が動いていれば日々必ず変わる。"""
    X = df[feats].to_numpy(dtype="float64", na_value=np.nan)
    h = hashlib.md5()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update("|".join(df["race_id"].astype(str)).encode())
    return h.hexdigest()


def prev_open_day(date_str):
    """自分より前の最新 dump 日付。"""
    days = sorted(d for d in os.listdir(DUMP_DIR)
                  if os.path.isdir(os.path.join(DUMP_DIR, d)) and d.isdigit() and d < date_str)
    return days[-1] if days else None


def prev_audit(date_str):
    outs = sorted(glob.glob(os.path.join(AUDIT_DIR, "2*.json")))
    outs = [o for o in outs if os.path.basename(o)[:8] < date_str]
    if not outs:
        return None
    try:
        return json.load(open(outs[-1], encoding="utf-8"))
    except Exception:
        return None


def notify(title, msg, color=None):
    try:
        cmd = [sys.executable, os.path.join(BASE, "tools", "notify_done.py"), title, msg]
        if color:
            cmd += ["--color", color]
        subprocess.run(cmd, timeout=30, capture_output=True)
    except Exception:
        pass  # 通知失敗で監査自体は止めない


def source_check(date_str):
    """供給レベル監査 (平日/非開催日用。2026-08-11 供給復旧 item3)。
    daily_jrdb_supply の supply_health JSON + KYI 内容鮮度 + 馬名解決率≥99% を検査し、
    PASS なら BLOCK flag をクリアする (dump ベースの完全確認は次の開催日の通常監査)。"""
    reasons = []
    hp = os.path.join(AUDIT_DIR, f"supply_health_{date_str}.json")
    if not os.path.exists(hp):
        cands = sorted(glob.glob(os.path.join(AUDIT_DIR, "supply_health_*.json")))
        hp = cands[-1] if cands else None
    if not hp:
        print("[T1v2 source] supply_health なし → 供給ジョブ未稼働")
        return 2
    h = json.load(open(hp, encoding="utf-8"))
    # 直近開催日 (直前の土日) まで KYI 内容があるか
    d = datetime.strptime(date_str, "%Y%m%d")
    back = d
    while back.weekday() < 5:  # 直前の日曜まで戻る
        back = back - __import__("datetime").timedelta(days=1)
    need = back.strftime("%Y%m%d")
    ok = True
    if (h.get("kyi_latest_file") or "0") < need:
        ok = False; reasons.append(f"KYI内容 {h.get('kyi_latest_file')} < 直近開催日 {need}")
    nr = h.get("name_resolution_2026")
    if nr is None or nr < 0.99:
        ok = False; reasons.append(f"馬名解決率 {nr} < 0.99")
    if not h.get("sed_latest") or h["sed_latest"] < need:
        ok = False; reasons.append(f"SED内容 {h.get('sed_latest')} < {need}")
    # JV-Link 供給 (2026-08-12 第1弾-2 で常設化。jv_health の SE/HR 鮮度も必須化)
    jh_files = sorted(glob.glob(os.path.join(AUDIT_DIR, "jv_health_*.json")))
    if not jh_files:
        ok = False; reasons.append("JV-Link jv_health なし (JvlinkSupplyDaily 未稼働)")
    else:
        jh = json.load(open(jh_files[-1], encoding="utf-8"))
        for k, lab in [("se_latest", "JV-SE"), ("hr_latest", "JV-HR")]:
            v = jh.get(k)
            if not v or str(v) < need:
                ok = False; reasons.append(f"{lab}内容 {v} < {need}")
    if ok:
        print(f"[T1v2 source] PASS  KYI={h.get('kyi_latest_file')} SED={h.get('sed_latest')} "
              f"馬名解決={nr:.2%} (基準日 {need})")
        if os.path.exists(BLOCK_FLAG):
            os.remove(BLOCK_FLAG)
            notify("T1v2 供給復旧", f"{date_str}: source-check PASS (KYI/SED={h.get('kyi_latest_file')}, "
                   f"馬名解決{nr:.0%}) → 予測ブロック解除。次開催日の dump 監査で最終確認")
            print("[T1v2 source] BLOCK flag クリア")
        return 0
    print(f"[T1v2 source] FAIL: {'; '.join(reasons)}")
    notify("T1v2 供給NG", f"{date_str}: " + "; ".join(reasons), color="red")
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--source-check", action="store_true",
                    help="供給レベル監査 (平日用)。PASS で BLOCK flag をクリア")
    args = ap.parse_args()
    date_str = args.date
    if args.source_check:
        os.makedirs(AUDIT_DIR, exist_ok=True)
        return source_check(date_str)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    feats = load_feature_list()
    jrdb_feats = [f for f in feats if f.startswith("jrdb_")]

    reasons, verdict = [], "OK"

    df = load_day(date_str, feats)
    if df is None:
        # 土日 = 開催日相当。dump が無いのは生成停止の疑い。平日は非開催として skip。
        wd = datetime.strptime(date_str, "%Y%m%d").weekday()
        if wd >= 5:
            verdict = "CRITICAL"
            reasons.append("NO_DUMP_ON_RACEDAY: 週末なのに feat_dump が存在しない (生成停止の疑い)")
        else:
            print(f"[T1v2] {date_str}: dump なし (平日=非開催とみなし skip)")
            return 0
        out = {"date": date_str, "verdict": verdict, "reasons": reasons,
               "n_races": 0, "n_horses": 0, "value_md5": None,
               "jrdb_const_n": None, "features": {}}
    else:
        vh = value_hash(df, feats)
        n_races = df["race_id"].nunique()

        # A. ゾンビ検知
        prev_day = prev_open_day(date_str)
        prev_hash = None
        pa = prev_audit(date_str)
        if pa and pa.get("value_md5"):
            prev_hash = pa["value_md5"]
            prev_src = f"audit {pa['date']}"
        elif prev_day:
            pdf = load_day(prev_day, feats)
            if pdf is not None:
                prev_hash = value_hash(pdf, feats)
                prev_src = f"dump {prev_day}"
        if prev_hash and vh == prev_hash:
            verdict = "CRITICAL"
            reasons.append(f"ZOMBIE: 特徴値ハッシュが前開催日({prev_src})と完全一致 = 生成停止")

        # C. JRDB 定数化
        jr_n = int((df[jrdb_feats].nunique() <= 2).sum())
        if jr_n >= JRDB_CRIT_N:
            verdict = "CRITICAL"
            reasons.append(f"JRDB_DEAD: jrdb特徴 {jr_n}/{len(jrdb_feats)} が定数化 (>= {JRDB_CRIT_N})")
        elif jr_n >= JRDB_WARN_N:
            if verdict == "OK":
                verdict = "WARN"
            reasons.append(f"JRDB_DEGRADED: jrdb特徴 {jr_n}/{len(jrdb_feats)} が定数化 (>= {JRDB_WARN_N})")

        # D. 欠損急増 (前回監査比)
        null_rates = df[feats].isna().mean().round(4).to_dict()
        if pa and pa.get("features"):
            spikes = [f for f in feats
                      if null_rates.get(f, 0) - pa["features"].get(f, {}).get("null_rate", 0) > NULL_SPIKE]
            if spikes:
                if verdict == "OK":
                    verdict = "WARN"
                reasons.append(f"NULL_SPIKE: {len(spikes)}特徴で欠損率が前回比+{NULL_SPIKE}超 ({spikes[:5]}...)")

        # E. スコア圧縮
        if "スコア" in df.columns:
            rng = float(df["スコア"].max() - df["スコア"].min())
            if rng < SCORE_RANGE_MIN:
                if verdict == "OK":
                    verdict = "WARN"
                reasons.append(f"SCORE_COMPRESSED: 本番スコア range {rng:.3f} < {SCORE_RANGE_MIN} (識別能力喪失の疑い)")

        out = {"date": date_str, "verdict": verdict, "reasons": reasons,
               "n_races": int(n_races), "n_horses": int(len(df)),
               "value_md5": vh, "jrdb_const_n": jr_n,
               "features": {f: {"null_rate": float(null_rates[f]),
                                "nunique": int(df[f].nunique())} for f in feats}}

    with open(os.path.join(AUDIT_DIR, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # flag の作成 / 自動クリア
    if verdict == "CRITICAL":
        with open(BLOCK_FLAG, "w", encoding="utf-8") as f:
            json.dump({"date": date_str, "reasons": reasons,
                       "created_at": datetime.now().isoformat()}, f, ensure_ascii=False)
        notify("T1v2 CRITICAL", f"{date_str}: " + " / ".join(reasons) + " → 翌日予測をブロック", color="red")
    elif verdict == "OK" and os.path.exists(BLOCK_FLAG):
        os.remove(BLOCK_FLAG)
        notify("T1v2 回復", f"{date_str}: 監査OK。予測ブロックを解除しました")
    elif verdict == "WARN":
        notify("T1v2 WARN", f"{date_str}: " + " / ".join(reasons))

    print(f"[T1v2] {date_str}: {verdict}"
          + (f"  ({'; '.join(reasons)})" if reasons else "")
          + (f"  races={out['n_races']} horses={out['n_horses']} jrdb_const={out['jrdb_const_n']}"
             if out.get("n_horses") else ""))
    return 2 if verdict == "CRITICAL" else 0


if __name__ == "__main__":
    sys.exit(main())
