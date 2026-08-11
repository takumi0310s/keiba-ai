# -*- coding: utf-8 -*-
"""特徴量健全性 日次レポート (2026-08-11 通知再整備 item3。T1v2 の詳細版・毎朝の健康診断)。

09:30 一括通知とは別に 1通、その日の全レース平均の特徴量健全性を #アップデート へ:
  - カテゴリ別 生存率 (BASE / JRDB系54 / premium系16)
  - 死亡中特徴リスト (カテゴリ別)
  - 前回監査 (T1v2 audit JSON) との差分 (新規死亡 / 復活)

usage: python tools/feature_health_report.py [--date YYYYMMDD] [--dry-run] [--test]
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, pickle, sys
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tools"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

_keep = [sys.stdout, sys.stderr]
from notify import send_discord  # noqa: E402
_keep += [sys.stdout, sys.stderr]

DUMP = os.path.join(BASE, "data", "v15_feat_dump")
AUDIT = os.path.join(BASE, "data", "T1v2_audit")
PREMIUM = {
    'index_max_filled', 'index_run1_filled', 'index_avg5_filled', 'stable_comment_score',
    'wood_best_4f_filled', 'sakaro_best_4f_filled', 'sakaro_best_3f_filled', 'time_1f_last_filled',
    'training_intensity_enc', 'wood_count_2w', 'total_training_count', 'training_per_dist',
    'has_training', 'has_wood_training', 'has_sakaro_training', 'training_time_filled',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    with gzip.open(os.path.join(BASE, "keiba_model_v15_central.pkl.gz"), "rb") as f:
        feats = pickle.load(f)["features"]
    jrdb_g = [c for c in feats if c.startswith(("jrdb_", "paci_", "oz_"))]
    prem_g = [c for c in feats if c in PREMIUM]
    base_g = [c for c in feats if c not in jrdb_g and c not in prem_g]

    files = [f for f in sorted(glob.glob(os.path.join(DUMP, a.date, "*.parquet")))
             if os.path.getsize(f) > 0]
    if not files:
        print(f"{a.date}: dump なし (非開催)"); return 0
    big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    X = big[feats].apply(pd.to_numeric, errors="coerce")
    dead = [c for c in feats if X[c].nunique() <= 2]

    def rate(group):
        d = sum(1 for c in group if c in dead)
        return f"{len(group)-d}/{len(group)} 生存 ({(len(group)-d)/len(group)*100:.0f}%)"

    # 前回監査との差分
    prev_lines = ""
    prevs = sorted(p for p in glob.glob(os.path.join(AUDIT, "2*.json"))
                   if os.path.basename(p)[:8] < a.date)
    if prevs:
        pj = json.load(open(prevs[-1], encoding="utf-8"))
        pfeat = pj.get("features", {})
        if pfeat:
            prev_dead = {f for f, v in pfeat.items() if v.get("nunique", 99) <= 2}
            new_dead = sorted(set(dead) - prev_dead)
            revived = sorted(prev_dead - set(dead))
            prev_lines = (f"\n【前回 {pj['date']} との差分】\n"
                          f"  新規死亡 {len(new_dead)}: {', '.join(new_dead[:8]) or 'なし'}\n"
                          f"  復活 {len(revived)}: {', '.join(revived[:8]) or 'なし'}")
    dead_jrdb = [c for c in dead if c in jrdb_g]
    dead_prem = [c for c in dead if c in prem_g]
    dead_base = [c for c in dead if c in base_g]
    msg = (f"対象 {len(files)}R / {len(big)}頭\n"
           f"【カテゴリ別生存】\n"
           f"  BASE({len(base_g)}): {rate(base_g)}\n"
           f"  JRDB系({len(jrdb_g)}): {rate(jrdb_g)}\n"
           f"  premium系({len(prem_g)}): {rate(prem_g)} ← netkeiba解約で恒常default想定\n"
           f"【死亡中 {len(dead)}/145】\n"
           f"  JRDB系({len(dead_jrdb)}): {', '.join(dead_jrdb[:10]) or 'なし'}\n"
           f"  premium系({len(dead_prem)}): {', '.join(dead_prem[:16]) or 'なし'}\n"
           f"  BASE({len(dead_base)}): {', '.join(dead_base[:10]) or 'なし'}"
           f"{prev_lines}")
    title = f"🩺 {a.date} 特徴量健全性レポート"
    if a.dry_run:
        print(f"----- {title} -----\n{msg}")
    else:
        send_discord(("【TEST】" if a.test else "") + title, msg,
                     color="blue", channel=("test" if a.test else "updates"),
                     dedup_window_sec=0)
        print("送信完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
