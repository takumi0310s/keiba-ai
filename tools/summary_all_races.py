#!/usr/bin/env python
"""今日の全レース予測サマリーを Discord に分割配信

phase1 ログ (top5 スコア+馬名) + daily_predictions を使って
コンパクトな「全レース予測一覧」を #買い目 に送る。

Usage:
    python tools/summary_all_races.py              # 本番
    python tools/summary_all_races.py --preview    # #アップデートで確認
    python tools/summary_all_races.py --dry-run    # 表示のみ
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

DATE_STR = datetime.now().strftime('%Y%m%d')
CHAR_LIMIT = 1900  # 2000字制限に対してマージン

IS_GRADED_MARKS = ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ']
IS_OPEN_MARKS   = ['杯', '賞', 'ステークス', 'カップ', 'ハンデ', 'Stakes', '(L)', '(OP)']
COND_LABEL = {'A':'芝長', 'B':'芝重', 'C':'芝多', 'D':'短', 'E':'少頭', 'X':'芝多重'}


def _shorten(name: str, n: int = 5) -> str:
    """馬名を n文字に短縮。"""
    return name[:n] if name else '?'


def _is_skip(race_name: str, course: str, cond: str) -> tuple[bool, str]:
    is_graded = any(g in race_name for g in IS_GRADED_MARKS)
    is_open   = any(s in race_name for s in IS_OPEN_MARKS)
    if course == '京都' and not is_graded:
        return True, '京都(非重賞)'
    if '特別' in race_name and not (is_graded or is_open):
        return True, '06_特別'
    if cond == 'B':
        return True, '条件B'
    if cond == 'E':
        return True, '条件E'
    return False, ''


def load_phase1(race_id: str, date_str: str) -> list[dict]:
    path = BASE_DIR / 'data' / 'race_notify_log_v2' / date_str / 'phase1' / f'{race_id}.json'
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
        return d.get('ranking_top5', [])
    except Exception:
        return []


def load_daily(date_str: str) -> dict[str, dict]:
    import csv
    path = BASE_DIR / 'data' / 'daily_predictions' / f'{date_str}.csv'
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            result[row['race_id']] = row
    return result


def build_race_line(row: dict, top5: list[dict]) -> str:
    """1レース分のコンパクト行を生成 (3行以内)。"""
    course = row['course']
    rnum   = int(row['race_num'])
    cond   = row['condition']
    rname  = row['race_name'][:10]
    skip, skip_reason = _is_skip(row['race_name'], course, cond)

    # 投票マーク
    vote_mark = '  ' if skip else '🏇'

    # top5 の丸数字+短縮馬名
    circle = ['①','②','③','④','⑤']
    if top5:
        horse_parts = []
        for i, h in enumerate(top5[:5]):
            uma   = h.get('umaban', '?')
            name  = _shorten(str(h.get('name', '?')), 4)
            horse_parts.append(f"{circle[i]}{uma}{name}")
        horses_str = ' '.join(horse_parts)
    else:
        horses_str = '(データなし)'

    if skip:
        hdr = f"{vote_mark}{course}{rnum}R[{cond}] {rname} 《参考のみ/{skip_reason}》"
    else:
        hdr = f"{vote_mark}{course}{rnum}R[{cond}] {rname}"

    return f"{hdr}\n    {horses_str}"


def chunk_messages(header: str, lines: list[str], limit: int) -> list[str]:
    """lines を limit字以内のメッセージに分割。"""
    messages = []
    current = header
    for line in lines:
        candidate = current + '\n' + line if current else line
        if len(candidate) > limit:
            messages.append(current)
            current = header + '\n' + line
        else:
            current = candidate
    if current:
        messages.append(current)
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date',    default=DATE_STR)
    ap.add_argument('--preview', action='store_true', help='#アップデートで確認')
    ap.add_argument('--dry-run', action='store_true', help='表示のみ、送信なし')
    ap.add_argument('--channel', default='bets')
    args = ap.parse_args()

    date_str = args.date
    channel  = 'updates' if args.preview else args.channel

    daily = load_daily(date_str)
    if not daily:
        print(f"[summary] daily_predictions/{date_str}.csv なし")
        return 1

    # 全レースをコース・R番順にソート
    sorted_rows = sorted(daily.values(), key=lambda r: (r['course'], int(r['race_num'])))

    # コースごとにグループ化: 新潟→東京→京都 の順
    course_order = ['新潟', '東京', '京都']
    by_course: dict[str, list] = {c: [] for c in course_order}
    for row in sorted_rows:
        c = row['course']
        if c not in by_course:
            by_course[c] = []
        by_course[c].append(row)

    # コースの実際の出現順 (既知→未知)
    courses_present = [c for c in course_order if by_course.get(c)]
    for c in sorted(by_course):
        if c not in courses_present and by_course.get(c):
            courses_present.append(c)

    # 全レース行を構築
    all_race_lines: list[str] = []
    total_races  = 0
    vote_count   = 0
    ref_count    = 0

    for course in courses_present:
        rows = by_course.get(course, [])
        if not rows:
            continue
        all_race_lines.append(f'━━ {course} ━━')
        for row in rows:
            rid  = row['race_id']
            top5 = load_phase1(rid, date_str)
            line = build_race_line(row, top5)
            all_race_lines.append(line)
            total_races += 1
            skip, _ = _is_skip(row['race_name'], row['course'], row['condition'])
            if skip:
                ref_count += 1
            else:
                vote_count += 1

    # dry-run
    if args.dry_run:
        print(f"\n[summary] 全{total_races}R (投票対象={vote_count}R、参考={ref_count}R)")
        for line in all_race_lines:
            print(line)
        return 0

    # ヘッダー
    dt  = datetime.strptime(date_str, '%Y%m%d')
    wd  = ['月','火','水','木','金','土','日'][dt.weekday()]
    hdr = (
        f"📋 **{dt.month}/{dt.day}({wd}) 全レース予測サマリー (V15)**\n"
        f"🏇=投票対象  =参考のみ | 各R: 上位5頭 (V15スコア順)\n"
        f"投票対象={vote_count}R / 参考={ref_count}R / 計{total_races}R\n"
        f"━━━━━━━━━━━━━━━━"
    )

    # 分割
    chunks = chunk_messages(hdr, all_race_lines, CHAR_LIMIT)
    total_parts = len(chunks)

    print(f"[summary] {total_races}R → {total_parts}通に分割")
    for i, chunk in enumerate(chunks, 1):
        print(f"  part {i}: {len(chunk)}字")

    if args.dry_run:
        return 0

    # 送信
    from notify import send_discord

    sent = 0
    for i, body in enumerate(chunks, 1):
        title = (f"📋 {dt.month}/{dt.day}({wd}) 全レース予測サマリー"
                 + (f" ({i}/{total_parts})" if total_parts > 1 else ''))
        ok = send_discord(title, body, color='blue', channel=channel)
        if ok:
            sent += 1
        print(f"  [{i}/{total_parts}] {'OK' if ok else 'FAIL'} ({len(body)}字)")

    print(f"\n[summary] sent={sent}/{total_parts} (channel={channel})")
    return 0 if sent > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
