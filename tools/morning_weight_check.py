"""morning_weight_check — 当日朝 馬体重 公開後 の予測補正チェック

朝 08:00 の予測 (data/daily_predictions/{ymd}.csv、馬体重デフォルト) と
本機構実行時 (例 09:30、馬体重公開済み) の予測を比較し、
変化が大きいレースを Discord アラート。

V15 model 内に馬体重 features 実装済み (predict_core.py L1530-1838) のため、
predict_one_race を再実行するだけで自動的に新値で予測される。

Usage:
    python tools/morning_weight_check.py                     # 今日の案B改採用R
    python tools/morning_weight_check.py --date 20260509     # 指定日
    python tools/morning_weight_check.py --races 202604010312,202605020512  # 特定R
    python tools/morning_weight_check.py --all               # 全R (重い、12R x ~30秒)
    python tools/morning_weight_check.py --dry-run           # Discord 通知しない
    python tools/morning_weight_check.py --silent            # 同上 alias
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

# Windows cp932 対策
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ===== 比較閾値 =====
TOP1_PROB_DIFF_NOTE = 0.05      # ±5% で注意
TOP1_PROB_DIFF_ALERT = 0.10     # ±10% で alert
WEIGHT_DIFF_ALERT = 15          # ±15kg で alert
WEIGHT_DIFF_NOTE = 10           # ±10kg で注意


def load_morning_predictions(date_str: str) -> dict[str, dict]:
    """朝 08:00 の予測 CSV を読み込み race_id -> 行 dict"""
    path = BASE / f"data/daily_predictions/{date_str}.csv"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("race_id", "").strip()
            if rid:
                out[rid] = row
    return out


def parse_int_safe(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def parse_float_safe(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def select_target_races(date_str: str,
                        morning: dict[str, dict],
                        races_arg: str | None = None,
                        all_flag: bool = False) -> list[str]:
    """対象レース選定。 default = 案B改 採用候補 (12R 1勝クラス、recommended)"""
    if races_arg:
        return [r.strip() for r in races_arg.split(",") if r.strip()]
    if all_flag:
        return list(morning.keys())
    # 案B改: 12R + 1勝クラス + recommended (bet_type = trio or umaren)
    targets = []
    for rid, row in morning.items():
        race_num = parse_int_safe(row.get("race_num"))
        race_name = row.get("race_name", "")
        bet_type = row.get("bet_type", "")
        if race_num != 12:
            continue
        if "1勝" not in race_name:
            continue
        if bet_type not in ("trio", "umaren"):
            continue
        targets.append(rid)
    return targets


def compare(rid: str, morning_row: dict, current_result, current_race_name: str, current_rinfo: dict) -> dict:
    """朝予測 vs 補正後予測 比較結果 dict を返す"""
    out = {
        "race_id": rid,
        "course": morning_row.get("course", ""),
        "race_num": morning_row.get("race_num", ""),
        "race_name": current_race_name or morning_row.get("race_name", ""),
        "morning_top1_num": morning_row.get("top1_num", ""),
        "morning_top1_name": morning_row.get("top1_name", ""),
        "morning_top1_score": parse_float_safe(morning_row.get("top1_score", 0)),
        "morning_top2_num": morning_row.get("top2_num", ""),
        "morning_top3_num": morning_row.get("top3_num", ""),
        "current_top1_num": "",
        "current_top1_name": "",
        "current_top1_score": 0.0,
        "current_top2_num": "",
        "current_top3_num": "",
        "current_top1_weight": 0,
        "current_top1_weight_diff": 0,
        "alerts": [],
        "notes": [],
        "max_weight_change_horse": "",
        "max_weight_change_kg": 0,
    }

    # current_result: predict_one_race 戻り値 ('馬番', '馬名', 'スコア', '馬体重' columns)
    if current_result is None or len(current_result) == 0:
        out["alerts"].append("補正後予測 取得失敗")
        return out
    df = current_result.head(8)  # top8 で十分
    out["current_top1_num"] = str(df.iloc[0].get("馬番", ""))
    out["current_top1_name"] = str(df.iloc[0].get("馬名", ""))
    out["current_top1_score"] = float(df.iloc[0].get("スコア", 0))
    if len(df) >= 2:
        out["current_top2_num"] = str(df.iloc[1].get("馬番", ""))
    if len(df) >= 3:
        out["current_top3_num"] = str(df.iloc[2].get("馬番", ""))

    # top1 馬体重 + 場体重増減
    top1_weight = parse_int_safe(df.iloc[0].get("馬体重", 0))
    top1_diff = parse_int_safe(df.iloc[0].get("場体重増減", 0))
    out["current_top1_weight"] = top1_weight
    out["current_top1_weight_diff"] = top1_diff

    # 全頭で馬体重変化最大の馬を抽出
    max_change_kg = 0
    max_change_name = ""
    for _, row in df.iterrows():
        w_diff = parse_int_safe(row.get("場体重増減", 0))
        if abs(w_diff) > abs(max_change_kg):
            max_change_kg = w_diff
            max_change_name = str(row.get("馬名", ""))
    out["max_weight_change_horse"] = max_change_name
    out["max_weight_change_kg"] = max_change_kg

    # === 判定 ===

    # 1. top1 確率 diff
    score_diff = out["current_top1_score"] - out["morning_top1_score"]
    if abs(score_diff) >= TOP1_PROB_DIFF_ALERT:
        out["alerts"].append(f"TOP1 確率 {score_diff:+.3f} (>±{TOP1_PROB_DIFF_ALERT})")
    elif abs(score_diff) >= TOP1_PROB_DIFF_NOTE:
        out["notes"].append(f"TOP1 確率 {score_diff:+.3f}")

    # 2. top1 入替
    if str(out["morning_top1_num"]) != str(out["current_top1_num"]):
        out["alerts"].append(f"TOP1 入替 #{out['morning_top1_num']} → #{out['current_top1_num']}")

    # 3. top1-3 メンバー入替
    morning_top3 = {str(out["morning_top1_num"]), str(out["morning_top2_num"]), str(out["morning_top3_num"])}
    current_top3 = {str(out["current_top1_num"]), str(out["current_top2_num"]), str(out["current_top3_num"])}
    diff_set = morning_top3.symmetric_difference(current_top3)
    swaps = len(diff_set) // 2
    if swaps >= 2:
        out["alerts"].append(f"TOP1-3 で {swaps} 馬入替")
    elif swaps == 1:
        out["notes"].append(f"TOP1-3 で {swaps} 馬入替")

    # 4. 馬体重 ±15kg 以上
    if abs(max_change_kg) >= WEIGHT_DIFF_ALERT:
        target = "軸馬" if max_change_name == out["current_top1_name"] else "相手馬"
        out["alerts"].append(f"{target} {max_change_name} 馬体重 {max_change_kg:+d}kg")
    elif abs(max_change_kg) >= WEIGHT_DIFF_NOTE:
        out["notes"].append(f"{max_change_name} 馬体重 {max_change_kg:+d}kg")

    # 5. 同時条件 (確率 ±5% かつ 馬体重 ±10kg)
    if abs(score_diff) >= TOP1_PROB_DIFF_NOTE and abs(top1_diff) >= WEIGHT_DIFF_NOTE:
        out["notes"].append(f"TOP1 #{out['current_top1_num']} 確率 {score_diff:+.3f} + 体重 {top1_diff:+d}kg 同時")

    return out


def format_discord_body(results: list[dict]) -> tuple[str, str, str]:
    """Discord 通知のタイトル / 本文 / color"""
    n_alerts = sum(1 for r in results if r.get("alerts"))
    n_notes = sum(1 for r in results if r.get("notes"))
    n_total = len(results)

    if n_alerts > 0:
        title = f"馬体重チェック ALERT ({n_alerts}/{n_total} R)"
        color = "red"
    elif n_notes > 0:
        title = f"馬体重チェック 注意 ({n_notes}/{n_total} R)"
        color = "yellow"
    else:
        title = f"馬体重チェック OK ({n_total} R 全て安定)"
        color = "green"

    lines = []
    for r in results:
        course = r.get("course", "?")
        race_num = r.get("race_num", "?")
        race_name = r.get("race_name", "")[:20]
        morning_top1 = f"#{r.get('morning_top1_num', '?')} {r.get('morning_top1_name', '')}"
        current_top1 = f"#{r.get('current_top1_num', '?')} {r.get('current_top1_name', '')}"
        score_morning = r.get("morning_top1_score", 0)
        score_current = r.get("current_top1_score", 0)
        weight = r.get("current_top1_weight", 0)
        weight_diff = r.get("current_top1_weight_diff", 0)

        lines.append(f"━━━ {course}{race_num}R {race_name} ━━━")
        lines.append(f"  朝 TOP1: {morning_top1} (score={score_morning:.3f})")
        lines.append(f"  現 TOP1: {current_top1} (score={score_current:.3f}) 馬体重 {weight}({weight_diff:+d}kg)")
        if r.get("alerts"):
            for a in r["alerts"]:
                lines.append(f"  🚨 {a}")
        if r.get("notes"):
            for n in r["notes"]:
                lines.append(f"  ⚠ {n}")
        if not r.get("alerts") and not r.get("notes"):
            lines.append(f"  ✅ 朝の買い目維持で OK")

    body = "\n".join(lines) if lines else "対象レースなし"
    if n_alerts > 0:
        body = "🚨 修正検討してください\n\n" + body
    return title, body, color


def notify_discord(title: str, body: str, color: str) -> None:
    """tools/notify_done.py 経由で Discord 通知"""
    import subprocess
    try:
        subprocess.run(
            [sys.executable, str(BASE / "tools/notify_done.py"),
             title, body[:1800], "--color", color],
            check=False, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}", file=sys.stderr)


def save_result_csv(date_str: str, results: list[dict]) -> Path:
    """data/morning_weight_check/{ymd}.csv に保存"""
    out_dir = BASE / "data/morning_weight_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.csv"

    cols = [
        "race_id", "course", "race_num", "race_name",
        "morning_top1_num", "morning_top1_name", "morning_top1_score",
        "current_top1_num", "current_top1_name", "current_top1_score",
        "current_top1_weight", "current_top1_weight_diff",
        "max_weight_change_horse", "max_weight_change_kg",
        "alerts", "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in cols}
            row["alerts"] = "; ".join(r.get("alerts", []))
            row["notes"] = "; ".join(r.get("notes", []))
            w.writerow(row)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None,
                   help="対象日 YYYYMMDD (default: today)")
    p.add_argument("--races", type=str, default=None,
                   help="特定 race_id をカンマ区切り (例: 202604010312,202605020512)")
    p.add_argument("--all", action="store_true",
                   help="朝予測 全レース対象 (~30 R x 30秒、重い)")
    p.add_argument("--dry-run", action="store_true",
                   help="Discord 通知しない (内部処理のみ)")
    p.add_argument("--silent", action="store_true",
                   help="--dry-run alias")
    args = p.parse_args()

    silent = args.dry_run or args.silent

    today = datetime.date.today() if not args.date else datetime.datetime.strptime(args.date, "%Y%m%d").date()
    ymd = today.strftime("%Y%m%d")

    print(f"=== morning_weight_check {ymd} ===\n")

    # 1. 朝予測 読み込み
    morning = load_morning_predictions(ymd)
    if not morning:
        msg = f"朝予測 CSV 未生成: data/daily_predictions/{ymd}.csv\nDailyPredict (08:00) が完了していない可能性"
        print(f"[WARN] {msg}")
        if not silent:
            notify_discord("morning_weight_check スキップ", msg, "yellow")
        return 1

    print(f"[OK] 朝予測 読み込み: {len(morning)} R")

    # 2. 対象レース選定
    targets = select_target_races(ymd, morning, races_arg=args.races, all_flag=args.all)
    if not targets:
        msg = f"対象レース 0 件 (案B改 採用候補が見当たらない)。 morning csv に 12R 1勝クラス trio/umaren が含まれているか確認"
        print(f"[WARN] {msg}")
        if not silent:
            notify_discord("morning_weight_check 対象なし", msg, "yellow")
        return 0

    print(f"[OK] 対象 {len(targets)} R: {targets}\n")

    # 3. predict_one_race import
    try:
        import predict_one_race as por
    except Exception as e:
        print(f"[NG] predict_one_race import 失敗: {e}", file=sys.stderr)
        if not silent:
            notify_discord("morning_weight_check 起動失敗", f"predict_one_race import: {e}", "red")
        return 2

    # 4. 各 R で予測 + 比較
    results = []
    for i, rid in enumerate(targets, 1):
        print(f"--- [{i}/{len(targets)}] {rid} ---")
        try:
            ret = por.predict_one_race(rid)
            if ret is None:
                print(f"  [NG] 予測失敗")
                results.append({
                    "race_id": rid,
                    "course": morning[rid].get("course", ""),
                    "race_num": morning[rid].get("race_num", ""),
                    "race_name": morning[rid].get("race_name", ""),
                    "alerts": ["補正後予測 取得失敗"],
                    "notes": [],
                })
                continue
            current_result, current_race_name, current_rinfo = ret
            cmp = compare(rid, morning[rid], current_result, current_race_name, current_rinfo)
            results.append(cmp)
            n_alerts = len(cmp.get("alerts", []))
            n_notes = len(cmp.get("notes", []))
            print(f"  [OK] alerts={n_alerts} notes={n_notes}")
        except Exception as e:
            print(f"  [NG] 例外: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "race_id": rid,
                "course": morning[rid].get("course", ""),
                "race_num": morning[rid].get("race_num", ""),
                "race_name": morning[rid].get("race_name", ""),
                "alerts": [f"例外: {type(e).__name__}"],
                "notes": [],
            })
        # 連続実行 で netkeiba ban 回避
        if i < len(targets):
            time.sleep(2)
    print()

    # 5. CSV 保存
    csv_path = save_result_csv(ymd, results)
    print(f"[OK] CSV 保存: {csv_path}")

    # 6. Discord 通知
    title, body, color = format_discord_body(results)
    print(f"\n=== Discord 通知 ===")
    print(f"title: {title}")
    print(f"color: {color}")
    print(f"body:\n{body}\n")

    if not silent:
        notify_discord(title, body, color)
        print(f"[OK] Discord 通知送信")
    else:
        print(f"[SKIP] dry-run / silent")

    # exit code: alert あれば 2、note あれば 1、なければ 0
    n_alerts = sum(1 for r in results if r.get("alerts"))
    n_notes = sum(1 for r in results if r.get("notes"))
    return 2 if n_alerts > 0 else (1 if n_notes > 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
