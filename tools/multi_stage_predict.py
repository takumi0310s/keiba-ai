"""multi_stage_predict — 当日 3 段階自動予測機構 (5/9 本番)

土日本番で 10:00 / 14:50 / 15:45 の 3 段階で全開催場予測 + Discord 通知。

設計: data/v18/sat_sun_3stage_prediction_design.md
config: tools/multi_stage_predict_config.json

Stage 仕様:
  test10        (10:00): 2R 実値補正 + 3R-12R 朝予測のまま (情報提供のみ)
  race11_1450   (14:50): 全 11R 予測、重賞は採用外、1勝のみ買い目
  race12_1545   (15:45): 全 12R 予測、案B改 1勝のみ買い目 (主戦場)

Usage:
    python tools/multi_stage_predict.py --stage test10
    python tools/multi_stage_predict.py --stage race11_1450 --date 20260509
    python tools/multi_stage_predict.py --stage race12_1545 --date 20260509 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
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

CONFIG_PATH = BASE / "tools/multi_stage_predict_config.json"


# ===== Helpers =====

def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def load_morning_predictions(date_str: str) -> dict[str, dict]:
    """朝 08:00 予測 CSV を読み race_id → 行 dict"""
    path = BASE / f"data/daily_predictions/{date_str}.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("race_id") or "").strip()
            if rid:
                out[rid] = row
    return out


# ===== 案B改 フィルタ =====

def classify_11r(race_name: str) -> tuple[bool, str]:
    """11R 採用判定。 戻り値: (採用?, 理由)"""
    rn = (race_name or "").replace(" ", "").replace("　", "")
    # 重賞 (G1/G2/G3 の明示)
    for g in ["(G1)", "(G2)", "(G3)", "G１", "G２", "G３"]:
        if g in rn:
            return False, "重賞"
    # 重賞名前パターン (記念/カップ/ステークス上位/-S/-C 等のみで判定するのは難しい)
    # 11R で "1勝" が入っていれば 1勝クラス
    if "1勝" in rn:
        return True, "1勝クラス"
    # 11R は通常 重賞 or OP 特別、その他は採用外
    if "未勝利" in rn:
        return False, "未勝利"
    if "2勝" in rn or "3勝" in rn:
        return False, "2勝/3勝"
    # 重賞推定: 賞名カタカナ + 漢字「賞」「記念」「杯」など
    if any(s in rn for s in ["杯", "記念", "賞", "ステークス", "カップ", "C", "クラシック"]):
        return False, "重賞/特別"
    if any(s in rn for s in ["S", "L", "OP", "リステッド"]):
        return False, "OP/特別"
    return False, "条件外"


def classify_12r(race_name: str) -> tuple[bool, str]:
    """12R 採用判定 (案B改 主戦場)。 戻り値: (採用?, 理由)"""
    rn = (race_name or "").replace(" ", "").replace("　", "")
    if "1勝" in rn:
        return True, "1勝クラス"
    if "未勝利" in rn:
        return False, "未勝利"
    if "2勝" in rn or "3勝" in rn:
        return False, "2勝/3勝"
    if any(s in rn for s in ["特別", "S", "OP", "リステッド", "L", "杯", "記念", "賞"]):
        return False, "特別/OP"
    return False, "その他"


def apply_filter(race_name: str, filter_mode: str) -> tuple[bool, str]:
    if filter_mode == "11r_planb":
        return classify_11r(race_name)
    elif filter_mode == "12r_planb":
        return classify_12r(race_name)
    elif filter_mode == "none":
        return True, "情報提供"
    return False, "未定義 filter"


# ===== R 選定 =====

def select_target_races(stage: dict, morning: dict[str, dict]) -> dict[str, list[str]]:
    """{ 'with_weight': [race_id...], 'morning_only': [race_id...] }"""
    targets_w = stage["target_races"]  # [2] / [11] / [12]
    targets_mo = stage.get("morning_only_races", [])  # [3..12] for test10

    with_weight = []
    morning_only = []
    for rid, row in morning.items():
        try:
            r_num = int(row.get("race_num", 0))
        except (ValueError, TypeError):
            continue
        if r_num in targets_w:
            with_weight.append(rid)
        elif r_num in targets_mo:
            morning_only.append(rid)
    return {"with_weight": sorted(with_weight), "morning_only": sorted(morning_only)}


# ===== 予測 =====

def fetch_predict_with_weight(rid: str):
    """predict_one_race を呼び (current_result, race_name, rinfo) を返す"""
    try:
        import predict_one_race as por
        return por.predict_one_race(rid)
    except Exception as e:
        print(f"  [NG] predict_one_race {rid} 例外: {e}")
        return None


def compare_with_morning(morning_row: dict, current_result, race_name: str) -> dict:
    """現状予測 vs 朝予測 の比較 dict"""
    out = {
        "morning_top1_num": morning_row.get("top1_num", ""),
        "morning_top1_name": morning_row.get("top1_name", ""),
        "morning_top1_score": parse_float_safe(morning_row.get("top1_score", 0)),
        "current_top1_num": "",
        "current_top1_name": "",
        "current_top1_score": 0.0,
        "current_top1_weight": 0,
        "current_top1_weight_diff": 0,
        "diff_score": 0.0,
        "top1_changed": False,
        "current_top2_num": "",
        "current_top3_num": "",
        "current_top4_num": "",
        "current_top5_num": "",
        "current_top6_num": "",
    }
    if current_result is None or len(current_result) == 0:
        return out
    df = current_result.head(8)
    out["current_top1_num"] = str(df.iloc[0].get("馬番", ""))
    out["current_top1_name"] = str(df.iloc[0].get("馬名", ""))
    out["current_top1_score"] = float(df.iloc[0].get("スコア", 0))
    out["current_top1_weight"] = parse_int_safe(df.iloc[0].get("馬体重", 0))
    out["current_top1_weight_diff"] = parse_int_safe(df.iloc[0].get("場体重増減", 0))
    if len(df) >= 2:
        out["current_top2_num"] = str(df.iloc[1].get("馬番", ""))
    if len(df) >= 3:
        out["current_top3_num"] = str(df.iloc[2].get("馬番", ""))
    if len(df) >= 4:
        out["current_top4_num"] = str(df.iloc[3].get("馬番", ""))
    if len(df) >= 5:
        out["current_top5_num"] = str(df.iloc[4].get("馬番", ""))
    if len(df) >= 6:
        out["current_top6_num"] = str(df.iloc[5].get("馬番", ""))
    out["diff_score"] = out["current_top1_score"] - out["morning_top1_score"]
    out["top1_changed"] = str(out["morning_top1_num"]) != str(out["current_top1_num"])
    return out


def gen_trio_buying(cmp: dict) -> str:
    """三連複 7 点 1×2×5 のフォーメーション文字列"""
    top1 = cmp.get("current_top1_num", "")
    top2 = cmp.get("current_top2_num", "")
    top3 = cmp.get("current_top3_num", "")
    top4 = cmp.get("current_top4_num", "")
    top5 = cmp.get("current_top5_num", "")
    top6 = cmp.get("current_top6_num", "")
    if not all([top1, top2, top3]):
        return ""
    second = [t for t in [top2, top3] if t]
    third = [t for t in [top2, top3, top4, top5, top6] if t]
    bets = set()
    for s in second:
        for t in third:
            if len({top1, s, t}) == 3:
                trio = sorted([top1, s, t], key=lambda x: int(x))
                bets.add("-".join(trio))
    return "; ".join(sorted(bets, key=lambda b: tuple(int(x) for x in b.split("-"))))


# ===== Discord 通知 format =====

def format_test10(results: list[dict], morning_only_results: list[dict], date_str: str) -> tuple[str, str, str]:
    """test10 stage 用 message (title, body, color)"""
    weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][datetime.datetime.strptime(date_str, "%Y%m%d").weekday()]
    title = f"🔍 [10:00 テスト予測] {date_str[:4]}/{date_str[4:6]}/{date_str[6:8]} ({weekday_jp})"

    courses = sorted({r.get("course", "?") for r in results} | {r.get("course", "?") for r in morning_only_results})
    n_total = len(results) + len(morning_only_results)

    lines = [f"開催: {'/'.join(courses)} ({len(courses)}場 {n_total}R)"]
    lines.append("")
    lines.append("★ 2R 馬体重補正後 (実値):")
    if not results:
        lines.append("  対象なし (全 R 取得失敗 or 開催 0)")
    for r in results:
        cmp = r.get("cmp", {})
        course = r.get("course", "?")
        s_morning = cmp.get("morning_top1_score", 0)
        s_current = cmp.get("current_top1_score", 0)
        diff = cmp.get("diff_score", 0)
        wd = cmp.get("current_top1_weight_diff", 0)
        wd_str = f"{wd:+d}" if wd != 0 else "+0"
        lines.append(
            f"  {course} 2R: TOP1 馬{cmp.get('current_top1_num', '?')} score {s_current:.3f} "
            f"(朝{s_morning:.3f} {diff:+.3f}) 馬体重 {wd_str}kg"
        )

    lines.append("")
    lines.append(f"★ 3R-12R 朝予測 (馬体重未公開、{len(morning_only_results)} R 参考):")
    lines.append("  全 R 朝予測通り、馬体重 公開時刻まで待機")

    lines.append("")
    lines.append("機構: 動作正常")
    lines.append("次回: 14:50 (11R 一括) / 15:45 (12R 一括)")

    body = "\n".join(lines)
    color = "blue" if results else "yellow"
    return title, body, color


def format_race11_1450(results: list[dict], date_str: str) -> tuple[str, str, str]:
    """race11_1450 stage 用 (重賞含む全表示、買い目は 1勝のみ)"""
    weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][datetime.datetime.strptime(date_str, "%Y%m%d").weekday()]
    title = f"🏇 [14:50 11R 一括予測] {date_str[:4]}/{date_str[4:6]}/{date_str[6:8]} ({weekday_jp})"

    n_adopted = sum(1 for r in results if r.get("adopted"))
    courses = sorted({r.get("course", "?") for r in results})
    lines = [f"全 {len(courses)}場 11R 予測 (重賞含む、採用 {n_adopted}/{len(results)}):"]
    lines.append("")

    investment_total = 0
    for r in results:
        cmp = r.get("cmp", {})
        course = r.get("course", "?")
        rname = r.get("race_name", "")[:30]
        adopted = r.get("adopted", False)
        reason = r.get("reason", "")
        s_current = cmp.get("current_top1_score", 0)
        wd = cmp.get("current_top1_weight_diff", 0)
        wd_str = f"{wd:+d}" if wd != 0 else "+0"

        prefix = "★" if adopted else "・"
        lines.append(f"{prefix} {course} 11R {rname}:")
        lines.append(f"   軸 馬{cmp.get('current_top1_num', '?')} (TOP1 score {s_current:.3f}, 馬体重 {wd_str}kg)")
        if adopted:
            buying = gen_trio_buying(cmp)
            lines.append(f"   買い目: 三連複 7点 {buying} = 700円")
            investment_total += 700
        else:
            lines.append(f"   採用外 ({reason})、観察用予測")

    lines.append("")
    lines.append(f"投資合計: {investment_total:,}円 ({'案B改 採用あり' if investment_total > 0 else '案B改 フィルタ全 NG'})")

    body = "\n".join(lines)
    color = "green" if investment_total > 0 else "yellow"
    return title, body, color


def format_race12_1545(results: list[dict], date_str: str) -> tuple[str, str, str]:
    """race12_1545 stage 用 (案B改 主戦場、1勝のみ買い目)"""
    weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][datetime.datetime.strptime(date_str, "%Y%m%d").weekday()]
    title = f"🏇 [15:45 12R 一括予測] {date_str[:4]}/{date_str[4:6]}/{date_str[6:8]} ({weekday_jp})"

    n_adopted = sum(1 for r in results if r.get("adopted"))
    lines = [f"採用 R: {n_adopted}/{len(results)}"]
    lines.append("")

    investment_total = 0
    for r in results:
        cmp = r.get("cmp", {})
        course = r.get("course", "?")
        rname = r.get("race_name", "")[:30]
        adopted = r.get("adopted", False)
        reason = r.get("reason", "")
        s_current = cmp.get("current_top1_score", 0)
        wd = cmp.get("current_top1_weight_diff", 0)
        wd_str = f"{wd:+d}" if wd != 0 else "+0"

        prefix = "★" if adopted else "・"
        lines.append(f"{prefix} {course} 12R {rname}:")
        lines.append(f"   軸 馬{cmp.get('current_top1_num', '?')} (TOP1 score {s_current:.3f}, 馬体重 {wd_str}kg)")
        if adopted:
            buying = gen_trio_buying(cmp)
            lines.append(f"   買い目: 三連複 7点 {buying} = 700円")
            investment_total += 700
        else:
            lines.append(f"   採用外 ({reason})、観察用予測")

    lines.append("")
    lines.append(f"投資合計: {investment_total:,}円")

    # 累計余裕は MEMORY.md / cumulative_results.csv から取得 (簡易)
    base_cum = 13_530  # 5/6 真相確定 (data/v18/may_2_3_truth_audit_5_6.md)
    margin = base_cum - (-50_000) - investment_total  # 撤退ライン余裕 (最悪 -investment 時)
    lines.append(f"累計余裕: +{margin:,}円 (撤退ライン超まで、最悪時)")

    body = "\n".join(lines)
    color = "green" if investment_total > 0 else "yellow"
    return title, body, color


def notify_discord(title: str, body: str, color: str) -> None:
    """tools/notify_done.py 経由"""
    try:
        subprocess.run(
            [sys.executable, str(BASE / "tools/notify_done.py"),
             title, body[:1900], "--color", color],
            check=False, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}", file=sys.stderr)


def save_result_csv(stage_name: str, date_str: str, results: list[dict]) -> Path:
    out_dir = BASE / "data/multi_stage_predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}_{stage_name}.csv"
    cols = [
        "race_id", "course", "race_num", "race_name", "adopted", "reason",
        "morning_top1_num", "morning_top1_score",
        "current_top1_num", "current_top1_score", "current_top1_weight_diff",
        "diff_score", "top1_changed",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            cmp = r.get("cmp", {})
            row = {
                "race_id": r.get("race_id", ""),
                "course": r.get("course", ""),
                "race_num": r.get("race_num", ""),
                "race_name": r.get("race_name", ""),
                "adopted": r.get("adopted", False),
                "reason": r.get("reason", ""),
                "morning_top1_num": cmp.get("morning_top1_num", ""),
                "morning_top1_score": cmp.get("morning_top1_score", 0),
                "current_top1_num": cmp.get("current_top1_num", ""),
                "current_top1_score": cmp.get("current_top1_score", 0),
                "current_top1_weight_diff": cmp.get("current_top1_weight_diff", 0),
                "diff_score": cmp.get("diff_score", 0),
                "top1_changed": cmp.get("top1_changed", False),
            }
            w.writerow(row)
    return out_path


# ===== Main =====

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, required=True,
                   choices=["test10", "race11_1450", "race12_1545"])
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--silent", action="store_true", help="--dry-run alias")
    args = p.parse_args()

    silent = args.dry_run or args.silent
    today = datetime.date.today() if not args.date else datetime.datetime.strptime(args.date, "%Y%m%d").date()
    date_str = today.strftime("%Y%m%d")

    config = load_config()
    stage = config["stages"].get(args.stage)
    if not stage:
        print(f"[NG] unknown stage: {args.stage}")
        return 1

    print(f"=== multi_stage_predict {args.stage} {date_str} ===")
    print(f"label: {stage['label']}")
    print()

    # 1. 朝予測 読み込み
    morning = load_morning_predictions(date_str)
    if not morning:
        msg = f"朝予測 CSV 未生成: data/daily_predictions/{date_str}.csv"
        print(f"[WARN] {msg}")
        if not silent:
            notify_discord(f"{stage['label']} スキップ", msg, "yellow")
        return 1
    print(f"[OK] 朝予測 {len(morning)} R")

    # 2. R 選定
    targets = select_target_races(stage, morning)
    print(f"[OK] target with_weight: {len(targets['with_weight'])} / morning_only: {len(targets['morning_only'])}")
    print()

    # 3. 補正対象 R で predict_one_race
    results = []
    for i, rid in enumerate(targets["with_weight"], 1):
        morning_row = morning[rid]
        course = morning_row.get("course", "")
        race_num = morning_row.get("race_num", "")
        rname = morning_row.get("race_name", "")
        print(f"--- [{i}/{len(targets['with_weight'])}] {course} {race_num}R {rid} ---")
        ret = fetch_predict_with_weight(rid)
        if ret is None:
            cmp = {"current_top1_num": morning_row.get("top1_num", ""),
                   "current_top1_score": parse_float_safe(morning_row.get("top1_score", 0)),
                   "morning_top1_num": morning_row.get("top1_num", ""),
                   "morning_top1_score": parse_float_safe(morning_row.get("top1_score", 0))}
            results.append({"race_id": rid, "course": course, "race_num": race_num,
                            "race_name": rname, "cmp": cmp, "adopted": False,
                            "reason": "予測失敗 (朝予測使用)"})
            continue
        current_result, current_race_name, current_rinfo = ret
        cmp = compare_with_morning(morning_row, current_result, current_race_name)

        # filter
        adopted, reason = apply_filter(rname or current_race_name, stage["filter_mode"])
        if not stage.get("buying_enabled", False):
            adopted = False
            reason = "情報提供のみ"
        results.append({"race_id": rid, "course": course, "race_num": race_num,
                        "race_name": current_race_name or rname,
                        "cmp": cmp, "adopted": adopted, "reason": reason})
        if i < len(targets["with_weight"]):
            time.sleep(2)

    # morning_only (test10 で 3R-12R)
    morning_only_results = []
    for rid in targets["morning_only"]:
        row = morning[rid]
        morning_only_results.append({
            "race_id": rid,
            "course": row.get("course", ""),
            "race_num": row.get("race_num", ""),
            "race_name": row.get("race_name", ""),
        })

    # 4. CSV 保存
    csv_path = save_result_csv(args.stage, date_str, results)
    print(f"\n[OK] CSV 保存: {csv_path}")

    # 5. Discord format
    if args.stage == "test10":
        title, body, color = format_test10(results, morning_only_results, date_str)
    elif args.stage == "race11_1450":
        title, body, color = format_race11_1450(results, date_str)
    else:  # race12_1545
        title, body, color = format_race12_1545(results, date_str)

    print(f"\n=== Discord 通知 ===")
    print(f"title: {title}")
    print(f"color: {color}")
    print(f"body:\n{body}\n")

    if not silent:
        notify_discord(title, body, color)
        print("[OK] Discord 通知送信")
    else:
        print("[SKIP] dry-run / silent")

    return 0


if __name__ == "__main__":
    sys.exit(main())
