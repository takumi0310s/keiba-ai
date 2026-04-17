#!/usr/bin/env python
"""整形済み買い目Discord通知スクリプト

data/daily_predictions/YYYYMMDD.csv を読み込み、Discord #買い目 チャンネルへ
見やすい形式で全レース買い目を送信する。

Usage:
    python tools/notify_bets_formatted.py                    # 今日
    python tools/notify_bets_formatted.py --date 2026-04-18  # 指定日
    python tools/notify_bets_formatted.py --mode pre_race    # 前夜予測マーク付き

注意:
    - 投資額・条件分布をヘッダーに含む
    - 競馬場ごとにメッセージ分割 (Discord 2000字/msg制限)
    - ★評価は data/profitable_patterns.json の最大stars
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

from notify import send_discord  # noqa: E402


WEEKDAY_JP = ['月', '火', '水', '木', '金', '土', '日']
BET_JP = {'trio': '三連複', 'umaren': '馬連'}

DISCORD_CHAR_LIMIT = 1900  # safety margin below 2000


def normalize_date(s: str) -> tuple[str, str]:
    """受け入れ: '20260418' / '2026-04-18' / '' → (YYYYMMDD, weekday_jp)"""
    if not s:
        s = datetime.now().strftime('%Y%m%d')
    s = s.replace('-', '').replace('/', '')
    dt = datetime.strptime(s, '%Y%m%d')
    return s, WEEKDAY_JP[dt.weekday()]


def dist_bucket(dist: int) -> str:
    if dist <= 1400:
        return 'short'
    if dist <= 1800:
        return 'middle'
    return 'long'


def load_patterns():
    p = os.path.join(BASE_DIR, 'data', 'profitable_patterns.json')
    if not os.path.exists(p):
        return []
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f).get('profitable_patterns', [])
    except Exception:
        return []


def match_stars(patterns, course: str, cond: str, dist: int, surface: str) -> int:
    """対象レースに合致する profitable_patterns の最大 stars。"""
    dbkt = dist_bucket(int(dist))
    sm = {'芝': '芝', 'ダ': 'ダ'}.get(str(surface), 'all')
    best = 0
    for p in patterns:
        if p.get('type') != 'trio':
            continue
        pv = p.get('venue', 'all')
        if pv != 'all' and pv != course:
            continue
        pc = p.get('condition', 'all')
        if pc != 'all' and pc != cond:
            continue
        pd_ = p.get('distance', 'all')
        if pd_ != 'all' and pd_ != dbkt:
            continue
        ps = p.get('surface', 'all')
        if ps != 'all' and ps != sm:
            continue
        s = int(p.get('stars', 0) or 0)
        if s > best:
            best = s
    return best


def ai_comment(cond: str, surface: str, dist: int, num_horses: int) -> str:
    """粗い AI 見解ラベル (条件から推定)。"""
    heavy = cond in ('B', 'X')
    if num_horses <= 7:
        return '少頭数・差し不要'
    if dist <= 1200:
        return '前残り警戒' if not heavy else '内有利'
    if dist <= 1400:
        return '先行有利'
    if dist >= 2200:
        return '持久力勝負'
    if heavy:
        return 'タフ馬場・末脚鈍化'
    return '差し届く展開'


def turbulence(cond: str, num_horses: int) -> str:
    """波乱度 Lv1/Lv2/Lv3 (ざっくり)。"""
    if cond == 'X':
        return 'Lv3'
    if cond in ('C',) and num_horses >= 15:
        return 'Lv2'
    if cond in ('B',):
        return 'Lv2'
    return 'Lv1'


def _wrap_trio_bets(bets: list[str], per_line: int = 3) -> str:
    """三連複7点を3点×3行で見やすく。"""
    chunks = [bets[i:i + per_line] for i in range(0, len(bets), per_line)]
    return '\n       '.join(', '.join(chunks_i) for chunks_i in chunks)


def build_race_block(r, stars: int) -> str:
    bets = [b.strip() for b in str(r.trio_bets).split(';') if b.strip()]
    bt_jp = BET_JP.get(r.bet_type, r.bet_type)
    star_str = '★' * stars if stars > 0 else '-'
    race_name_short = str(r.race_name)[:14]

    # L1: 基本情報
    l1 = (f"**{int(r.race_num)}R** {r.surface}{int(r.distance)}m "
          f"{race_name_short} [{r.condition}] {star_str}")

    # L2: 買い目
    if r.bet_type == 'umaren':
        # 400/300円
        amounts = [400, 300]
        parts = [f"`{b}` {amt}円" for b, amt in zip(bets, amounts)]
        l2 = f"  {bt_jp}: " + ' / '.join(parts)
    else:
        wrapped = _wrap_trio_bets(bets)
        l2 = f"  {bt_jp}: {wrapped}"
        l2 += f"\n       各100円 / 合計{int(r.investment)}円"

    # L3: AI・軸
    tb = turbulence(r.condition, int(r.num_horses))
    ac = ai_comment(r.condition, r.surface, int(r.distance), int(r.num_horses))
    l3 = (f"  軸: {int(r.top1_num)}番 {str(r.top1_name)[:10]} "
          f"(score {float(r.top1_score):.3f}) | "
          f"波乱度 {tb} | AI: {ac}")

    return '\n'.join([l1, l2, l3])


def _chunk_and_send(title: str, header_line: str, blocks: list[str],
                    color: str, channel: str) -> int:
    """blocks を 1900字以内に分割して送信。送信数を返す。"""
    if not blocks:
        return 0
    parts: list[str] = []
    current = header_line
    for b in blocks:
        sep = '\n\n' if current and current != header_line else '\n'
        candidate = current + sep + b
        if len(candidate) > DISCORD_CHAR_LIMIT:
            parts.append(current)
            current = header_line + '\n' + b
        else:
            current = candidate
    if current:
        parts.append(current)

    total = len(parts)
    sent = 0
    for i, body in enumerate(parts, 1):
        t = f"{title} ({i}/{total})" if total > 1 else title
        ok = send_discord(t, body, color=color, channel=channel)
        if ok:
            sent += 1
    return sent


def notify_formatted(date_str_yyyymmdd: str, mode: str = 'morning',
                     channel: str = 'bets') -> int:
    """指定日の買い目CSVをDiscord整形送信。送信メッセージ総数を返す。"""
    csv_path = os.path.join(BASE_DIR, 'data', 'daily_predictions',
                            f'{date_str_yyyymmdd}.csv')
    if not os.path.exists(csv_path):
        send_discord('買い目通知エラー',
                     f"CSV not found: {csv_path}",
                     color='red', channel='updates')
        return 0

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if df.empty:
        send_discord('買い目通知',
                     f"{date_str_yyyymmdd} 予測結果0件",
                     color='yellow', channel='updates')
        return 0

    patterns = load_patterns()

    # ヘッダー情報
    dt = datetime.strptime(date_str_yyyymmdd, '%Y%m%d')
    date_disp = f"{dt.month}/{dt.day}({WEEKDAY_JP[dt.weekday()]})"
    total_inv = int(df['investment'].sum())
    cond_counts = df['condition'].value_counts().to_dict()
    cond_str = ' '.join(f'{k}:{v}' for k, v in sorted(cond_counts.items()))

    prefix = '🔮 前夜予測 ' if mode == 'pre_race' else '📋 '
    header_title = f"{prefix}【{date_disp} 買い目一覧】"
    header_body = (
        f"{header_title}\n"
        f"投資額: **¥{total_inv:,}** | {len(df)}R | 条件 {cond_str}\n"
        f"━━━━━━━━━━━━━━━━"
    )

    total_sent = 0
    ok = send_discord(header_title, header_body, color='green', channel=channel)
    if ok:
        total_sent += 1

    # 競馬場順ソート (既知3場優先、未知場は後ろ)
    known_order = ['中山', '東京', '阪神', '京都', '福島', '新潟',
                   '中京', '小倉', '札幌', '函館']
    courses = [c for c in known_order if c in set(df['course'].unique())]
    for c in sorted(set(df['course'].unique())):
        if c not in courses:
            courses.append(c)

    for course in courses:
        sub = df[df['course'] == course].sort_values('race_num')
        if sub.empty:
            continue
        blocks = []
        for _, r in sub.iterrows():
            stars = match_stars(patterns, course, r.condition,
                                int(r.distance), r.surface)
            blocks.append(build_race_block(r, stars))

        venue_inv = int(sub['investment'].sum())
        header_line = (
            f"━━━━━━━━━━━━━━━━\n"
            f"🏇 **{course}（{len(sub)}R / ¥{venue_inv:,}）**\n"
            f"━━━━━━━━━━━━━━━━"
        )
        title = f"{prefix}{date_disp} {course}"
        total_sent += _chunk_and_send(title, header_line, blocks,
                                      color='blue', channel=channel)

    # サマリー
    trio_df = df[df['bet_type'] == 'trio']
    uma_df = df[df['bet_type'] == 'umaren']
    trio_sum = int(trio_df['investment'].sum())
    uma_sum = int(uma_df['investment'].sum())
    summary = (
        f"【サマリー】\n"
        f"三連複: {len(trio_df)}R × 700円 = ¥{trio_sum:,}\n"
        f"馬連: {len(uma_df)}R × 700円 = ¥{uma_sum:,}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"**合計: ¥{total_inv:,}**"
    )
    if mode == 'pre_race':
        summary += '\n\n⚠️ 前夜予測はオッズ未確定。当日AM8:00本予測で再確認してください。'
    ok = send_discord(f"{prefix}{date_disp} サマリー", summary,
                      color='green', channel=channel)
    if ok:
        total_sent += 1

    return total_sent


def main():
    ap = argparse.ArgumentParser(description='整形済み買い目Discord通知')
    ap.add_argument('--date', type=str, default='',
                    help='対象日 YYYY-MM-DD or YYYYMMDD (default: today)')
    ap.add_argument('--mode', choices=['morning', 'pre_race'], default='morning',
                    help='morning (当日朝) / pre_race (前夜)')
    ap.add_argument('--channel', default='bets',
                    help='Discord channel (default: bets)')
    args = ap.parse_args()

    date_yyyymmdd, _ = normalize_date(args.date)
    n = notify_formatted(date_yyyymmdd, mode=args.mode, channel=args.channel)
    print(f'[notify_bets_formatted] sent {n} Discord messages '
          f'(date={date_yyyymmdd}, mode={args.mode})')
    return 0 if n > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
