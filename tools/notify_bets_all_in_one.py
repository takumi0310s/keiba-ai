#!/usr/bin/env python
"""全レース一括通知 Discordスクリプト

data/daily_predictions/YYYYMMDD.csv を読み込み、当日の全レース買い目を
1-3 メッセージにまとめて Discord #買い目 チャンネルへ送信する。

既存の通知 (race_auto_notify.py の 8分割通知、各レース5分前通知) に加えて
全体把握用のサマリー通知を追加する。

Usage:
    python tools/notify_bets_all_in_one.py                    # 今日
    python tools/notify_bets_all_in_one.py --date 2026-04-18  # 指定日

公開API:
    send_all_in_one(date: str, channel: str = 'bets') -> int
        送信メッセージ数を返す。
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

from notify import send_discord  # noqa: E402
from show_bets import compute_formation  # noqa: E402
from notify_bets_formatted import (  # noqa: E402
    fetch_umaren_pair_odds, load_patterns, match_stars, normalize_date,
    WEEKDAY_JP,
)

DISCORD_CHAR_LIMIT = 1900  # safety margin below 2000


def _short_name(name: str, max_len: int = 12) -> str:
    s = str(name or '').strip()
    if len(s) <= max_len:
        return s
    return s[:max_len]


def build_compact_block(r, stars: int, pair_odds: dict | None = None) -> str:
    """コンパクトな 2 行ブロックを返す (race_num + surface/distance + name + 条件 / bets)。"""
    row = r.to_dict() if hasattr(r, 'to_dict') else dict(r)
    form = compute_formation(row, pair_odds=pair_odds)

    star_str = ('★' * stars) if stars > 0 else '-'
    race_name = _short_name(r.race_name, 12)

    # L1: race_num surface+dist name 条件X ★★
    l1 = (f"{int(r.race_num)}R {r.surface}{int(r.distance)}m "
          f"{race_name} 条件{r.condition} {star_str}")

    if form is None:
        return l1 + "\n  (買い目データ異常)"

    investment = int(form['investment']) if form else int(r.investment or 0)

    if form['bet_type'] == 'umaren':
        parts = []
        for partner, amt, odds in form['umaren_pairs']:
            axis = form['axis']
            if odds and odds > 0:
                parts.append(f"{axis}-{partner} @{odds:.1f}倍 {amt}円")
            else:
                parts.append(f"{axis}-{partner} {amt}円")
        l2 = f"  馬連: {' / '.join(parts)} = {investment}円"
    else:
        col2 = ','.join(str(n) for n in form['second_col'])
        col3 = ','.join(str(n) for n in form['third_col'])
        l2 = f"  三連複F: {form['axis']} / {col2} / {col3} = {investment}円"

    return l1 + "\n" + l2


def _chunk_by_venue(header_body: str, venue_chunks: list[str],
                    footer: str) -> list[str]:
    """競馬場境界で分割。ヘッダーは各メッセージ先頭、フッターは最終メッセージ末尾のみ。"""
    messages: list[str] = []
    current = header_body
    for vc in venue_chunks:
        candidate = current + "\n\n" + vc
        # 最終メッセージならフッター余地も考慮
        if len(candidate) > DISCORD_CHAR_LIMIT:
            messages.append(current)
            current = header_body + "\n\n" + vc
        else:
            current = candidate

    # footer を付けてみる。はみ出るなら footer 単独メッセージ
    final_candidate = current + "\n\n" + footer
    if len(final_candidate) > DISCORD_CHAR_LIMIT:
        messages.append(current)
        messages.append(footer)
    else:
        messages.append(final_candidate)
    return messages


def send_all_in_one(date: str = '', channel: str = 'bets') -> int:
    """指定日の買い目を1-3メッセージにまとめてDiscord送信。送信数を返す。"""
    date_yyyymmdd, _ = normalize_date(date)
    csv_path = os.path.join(BASE_DIR, 'data', 'daily_predictions',
                            f'{date_yyyymmdd}.csv')
    if not os.path.exists(csv_path):
        send_discord('全レース一括通知エラー',
                     f"CSV not found: {csv_path}",
                     color='red', channel='updates')
        return 0

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if df.empty:
        send_discord('全レース一括通知',
                     f"{date_yyyymmdd} 予測結果0件",
                     color='yellow', channel='updates')
        return 0

    # 念のためフィルタ (予測CSVは通常既に除外済み)
    df = df[df['surface'] != '障']
    df = df[df['distance'] > 1000]
    if df.empty:
        send_discord('全レース一括通知',
                     f"{date_yyyymmdd} 対象レース0件 (障害/短距離除外後)",
                     color='yellow', channel='updates')
        return 0

    patterns = load_patterns()

    # 競馬場はCSV出現順を保持
    course_order: list[str] = []
    for c in df['course']:
        if c not in course_order:
            course_order.append(c)

    # ヘッダー
    dt = datetime.strptime(date_yyyymmdd, '%Y%m%d')
    date_disp = (f"{dt.year}/{dt.month:02d}/{dt.day:02d}"
                 f"({WEEKDAY_JP[dt.weekday()]})")
    total_inv = int(df['investment'].sum())
    n_races = len(df)
    header = (
        f"🏇 {date_disp} 全{n_races}R 買い目一覧\n"
        f"💰 投資額合計: {total_inv:,}円"
    )

    # 競馬場ごとブロックを構築
    venue_chunks: list[str] = []
    for course in course_order:
        sub = df[df['course'] == course].sort_values('race_num')
        if sub.empty:
            continue
        lines = [f"━━━ {course} {len(sub)}R開催 ━━━"]
        for _, r in sub.iterrows():
            stars = match_stars(patterns, course, r.condition,
                                int(r.distance), r.surface)
            pair_odds = None
            if str(r.bet_type).lower() == 'umaren':
                try:
                    pair_odds = fetch_umaren_pair_odds(str(r.race_id))
                except Exception:
                    pair_odds = None
            lines.append(build_compact_block(r, stars, pair_odds=pair_odds))
        venue_chunks.append('\n'.join(lines))

    # サマリー
    trio_df = df[df['bet_type'] == 'trio']
    uma_df = df[df['bet_type'] == 'umaren']
    trio_sum = int(trio_df['investment'].sum())
    uma_sum = int(uma_df['investment'].sum())
    footer_parts = ["━━━━━━━━━━━━━━",
                    f"投資額合計: {total_inv:,}円"]
    breakdown = []
    if len(trio_df):
        breakdown.append(f"三連複{len(trio_df)}R × 700円 = {trio_sum:,}円")
    if len(uma_df):
        breakdown.append(f"馬連{len(uma_df)}R × 700円 = {uma_sum:,}円")
    if breakdown:
        footer_parts.append("内訳: " + " / ".join(breakdown))
    footer_parts.append("━━━━━━━━━━━━━━")
    footer = '\n'.join(footer_parts)

    messages = _chunk_by_venue(header, venue_chunks, footer)

    total_parts = len(messages)
    title_base = f"📋 {date_disp} 全{n_races}R一括"
    sent = 0
    import time as _time
    for i, body in enumerate(messages, 1):
        title = f"{title_base} ({i}/{total_parts})" if total_parts > 1 else title_base
        ok = send_discord(title, body, color='green', channel=channel)
        if ok:
            sent += 1
        if i < total_parts:
            _time.sleep(0.5)  # Discord rate limit (5 req/sec) 対策
    return sent


def main():
    ap = argparse.ArgumentParser(description='全レース一括 Discord 通知')
    ap.add_argument('--date', type=str, default='',
                    help='対象日 YYYY-MM-DD or YYYYMMDD (default: today)')
    ap.add_argument('--channel', default='bets',
                    help='Discord channel (default: bets)')
    args = ap.parse_args()

    n = send_all_in_one(args.date, channel=args.channel)
    print(f'[notify_bets_all_in_one] sent {n} Discord messages')
    return 0 if n > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
