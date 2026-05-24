#!/usr/bin/env python
"""phase1ログ (top5スコア) + daily_predictions (formation) で一括通知

netkeiba 400 レート制限中でも動作する。
race_notify_log の既通知レースをスキップ (二重通知防止)。
strategy7 フィルタ適用 (京都非重賞・06_特別 除外)。

Usage:
    python tools/batch_notify_phase1.py               # #買い目 に送信
    python tools/batch_notify_phase1.py --preview     # #アップデート で確認
    python tools/batch_notify_phase1.py --dry-run     # 送信なし (内容表示のみ)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tools'))

# Windows コンソール cp932 → UTF-8 強制
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

DATE_STR = datetime.now().strftime('%Y%m%d')

ALREADY_NOTIFIED_CHANNELS = {'bets'}   # skip は二重通知OK
IS_GRADED_MARKS = ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ']
IS_OPEN_MARKS   = ['杯', '賞', 'ステークス', 'カップ', 'ハンデ', 'Stakes', '(L)', '(OP)']

COND_ROI = {
    'A': ('205.3%', '44.5%'),
    'B': ('236.9%', '45.2%'),
    'C': ('285.6%', '33.7%'),
    'D': ('136.0%', '27.0%'),
    'E': ('118.0%', '53.4%'),
    'X': ('330.5%', '35.5%'),
}


def _is_strategy7_skip(race_name: str, course: str) -> tuple[bool, str]:
    is_graded = any(g in race_name for g in IS_GRADED_MARKS)
    is_open = any(s in race_name for s in IS_OPEN_MARKS)

    if course == '京都' and not is_graded:
        return True, 'strategy7_kyoto'

    if '特別' in race_name and not (is_graded or is_open):
        return True, 'strategy7_06_tokubetsu'

    return False, ''


def load_notify_log(date_str: str) -> set[str]:
    """race_notify_log から既通知レース ID を返す (channel=bets のみ)。"""
    log_path = BASE_DIR / 'data' / 'race_notify_log' / f'{date_str}.json'
    if not log_path.exists():
        return set()
    try:
        logs = json.loads(log_path.read_text(encoding='utf-8'))
        return {e['race_id'] for e in logs if e.get('channel') in ALREADY_NOTIFIED_CHANNELS}
    except Exception:
        return set()


def load_phase1(race_id: str, date_str: str) -> dict | None:
    """phase1 ログを読み込む。なければ None。"""
    path = BASE_DIR / 'data' / 'race_notify_log_v2' / date_str / 'phase1' / f'{race_id}.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def load_daily_predictions(date_str: str) -> dict[str, dict]:
    """daily_predictions CSV を {race_id: row_dict} で返す。"""
    import csv
    path = BASE_DIR / 'data' / 'daily_predictions' / f'{date_str}.csv'
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            result[row['race_id']] = row
    return result


def parse_formation_horses(trio_bets_str: str) -> tuple[int | None, list[int]]:
    """'1-2-10; 1-6-10; ...' → (axis, 全馬番list sorted)

    axis = 全組に含まれる馬番。なければ先頭の第1要素。
    """
    bets = [
        tuple(int(n) for n in combo.strip().split('-'))
        for combo in trio_bets_str.split(';')
        if combo.strip()
    ]
    if not bets:
        return None, []
    from collections import Counter
    count = Counter(n for b in bets for n in b)
    n_bets = len(bets)
    axis_list = [n for n, c in count.items() if c == n_bets]
    axis = axis_list[0] if axis_list else bets[0][0]
    all_horses = sorted(count.keys())
    return axis, all_horses


def build_message(csv_row: dict, phase1: dict | None, idx: int, total: int) -> str:
    """1レース分のDiscordメッセージを構築。"""
    race_id   = csv_row['race_id']
    course    = csv_row['course']
    race_num  = csv_row['race_num']
    race_name = csv_row['race_name']
    condition = csv_row['condition']
    distance  = csv_row['distance']
    surface   = csv_row['surface']
    track_cond= csv_row.get('track_condition', '')
    bet_type  = csv_row.get('bet_type', 'trio')
    investment= csv_row.get('investment', '700')
    trio_bets = csv_row.get('trio_bets', '')
    top1_num  = csv_row.get('top1_num', '?')
    top1_name = csv_row.get('top1_name', '?')[:10]
    top1_score= float(csv_row.get('top1_score', 0))
    top2_num  = csv_row.get('top2_num', '?')
    top3_num  = csv_row.get('top3_num', '?')

    roi_str, hit_str = COND_ROI.get(condition, ('?', '?'))

    # --- Header ---
    lines = [
        f"**{course}{race_num}R** {surface}{distance}m {track_cond} {race_name[:14]}",
        f"条件{condition} ROI {roi_str} (的中{hit_str})",
        "",
        "**━【V15 本番・投票対象】━**",
        "",
    ]

    # --- Formation ---
    if bet_type == 'trio':
        axis, all_horses = parse_formation_horses(trio_bets)
        # phase1 top5 があれば col2=top2/top3, col3=top2〜top6 で表示
        if phase1 and phase1.get('ranking_top5') and axis:
            top5 = phase1['ranking_top5']
            r_by_uma = {h['umaban']: h for h in top5}
            col2_horses = [h['umaban'] for h in top5[1:3]]   # top2, top3
            col3_horses = sorted(n for n in all_horses if n != axis)
            lines.append(f"三連複フォーメーション 7点")
            lines.append(f"1列目: {axis}")
            lines.append(f"2列目: {', '.join(str(n) for n in col2_horses)}")
            lines.append(f"3列目: {', '.join(str(n) for n in col3_horses)}")
        elif axis:
            others = [n for n in all_horses if n != axis]
            lines.append(f"三連複フォーメーション 7点")
            lines.append(f"1列目: {axis} / 2〜3列目: {', '.join(str(n) for n in others)}")
        else:
            lines.append(f"三連複: {trio_bets}")
    elif bet_type == 'umaren':
        lines.append(f"馬連: {top1_num}-{top2_num} / {top1_num}-{top3_num}")

    lines.append(f"軸: {top1_num}番 {top1_name} (スコア{top1_score:.3f})")
    lines.append(f"投資額: {investment}円")

    # --- Top5 ranking (from phase1) ---
    lines.append("")
    lines.append("──────────────────────")

    if phase1 and phase1.get('ranking_top5'):
        top5 = phase1['ranking_top5']
        lines.append("**[《V15 上位スコア》]** (▶=買い目対象)")
        # 買い目に含まれる馬番
        formation_nums = set()
        for combo in trio_bets.split(';'):
            for n in combo.strip().split('-'):
                try:
                    formation_nums.add(int(n))
                except ValueError:
                    pass
        if bet_type == 'umaren':
            formation_nums = {int(top1_num), int(top2_num), int(top3_num)}

        for h in top5:
            rank  = h.get('rank', '?')
            uma   = h.get('umaban', '?')
            name  = str(h.get('name', ''))[:10]
            score = float(h.get('score', 0))
            mark  = '▶' if int(uma) in formation_nums else ' '
            lines.append(f"{mark} {rank}位 {str(uma):>2} {name:<10} {score:.3f}")

        lines.append("_V16能力馬券内率: ネットワーク制限中のため省略_")
    else:
        lines.append(f"軸: {top1_num}番 {top1_name}")
        lines.append("_全馬スコア: データ取得済 (top3のみ)_")

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=DATE_STR, help='対象日 YYYYMMDD')
    ap.add_argument('--preview', action='store_true', help='#アップデートで確認')
    ap.add_argument('--dry-run', action='store_true', help='送信しない')
    ap.add_argument('--channel', default='bets', help='送信チャンネル')
    args = ap.parse_args()

    date_str = args.date
    channel  = 'updates' if args.preview else args.channel

    # Load data
    already_notified = load_notify_log(date_str)
    predictions      = load_daily_predictions(date_str)

    print(f"[batch] date={date_str}, already_notified={already_notified}")
    print(f"[batch] total predictions={len(predictions)}")

    # Collect races to notify
    races_to_notify = []
    skipped_why: dict[str, str] = {}

    # Sort by venue + race_num
    sorted_rows = sorted(predictions.values(), key=lambda r: (r['course'], int(r['race_num'])))

    for row in sorted_rows:
        rid   = row['race_id']
        course= row['course']
        rname = row['race_name']
        cond  = row['condition']

        # Skip 京都 (strategy7)
        skip7, reason7 = _is_strategy7_skip(rname, course)
        if skip7:
            skipped_why[rid] = reason7
            continue

        # Skip 条件B (strategy7)
        if cond == 'B':
            skipped_why[rid] = 'strategy7_condB'
            continue

        # Skip already notified (bets channel)
        if rid in already_notified:
            skipped_why[rid] = 'already_notified'
            continue

        races_to_notify.append(row)

    print(f"[batch] to notify={len(races_to_notify)}, skipped={len(skipped_why)}")
    for rid, reason in skipped_why.items():
        print(f"  skip {rid}: {reason}")

    if not races_to_notify:
        print("[batch] 通知対象なし")
        return 0

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for row in races_to_notify:
            rid = row['race_id']
            p1  = load_phase1(rid, date_str)
            msg = build_message(row, p1, 0, len(races_to_notify))
            print(f"\n--- {rid} ---")
            print(msg)
        return 0

    # Send
    from notify import send_discord

    # Header
    dt = datetime.strptime(date_str, '%Y%m%d')
    wd = ['月','火','水','木','金','土','日'][dt.weekday()]
    header_title = f"📋 {dt.month}/{dt.day}({wd}) 残レース一括通知 [phase1データ]"
    header_body  = (
        f"**V15 朝予測 → 全馬スコア上位5頭で通知**\n"
        f"対象: {len(races_to_notify)}R (既通知={len(already_notified)}R スキップ)\n"
        f"⚠️ netkeiba 制限中: V16能力馬券内率・オッズ省略\n"
        f"━━━━━━━━━━━━━━━━"
    )
    ok = send_discord(header_title, header_body, color='blue', channel=channel)
    sent = 1 if ok else 0

    # Per race
    for i, row in enumerate(races_to_notify):
        rid    = row['race_id']
        course = row['course']
        rnum   = row['race_num']
        rname  = row['race_name'][:14]

        p1  = load_phase1(rid, date_str)
        msg = build_message(row, p1, i, len(races_to_notify))

        title = f"🏇 {course}{rnum}R {rname}"
        ok2   = send_discord(title, msg, color='green', channel=channel)
        if ok2:
            sent += 1
        print(f"  [{i+1}/{len(races_to_notify)}] {rid} {'OK' if ok2 else 'SKIPPED'}")

    # Footer
    send_discord(
        f"一括通知完了 ({len(races_to_notify)}R)",
        f"V15 production 不変 / V16・オッズは netkeiba 回復後に race_auto_notify で自動通知",
        color='blue', channel=channel
    )
    sent += 1

    print(f"\n[batch] sent={sent} messages (channel={channel})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
