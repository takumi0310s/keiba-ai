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

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

from notify import send_discord  # noqa: E402
from show_bets import compute_formation  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_umaren_pair_odds(race_id: str) -> dict:
    """馬連(type=4)オッズを取得。{(n1,n2): odds} を返す。失敗時は空dict。

    netkeiba JRA odds API はJSON (`data.odds.4.XXYY = [odds, ?, pop_rank]`) を
    返す。発売前・結果未確定は `"data":""` や `"result odds empty"` になる。
    """
    pair_odds = {}
    try:
        url = (
            f"https://race.netkeiba.com/api/api_get_jra_odds.html"
            f"?race_id={race_id}&type=4"
        )
        resp = requests.get(url, headers=HEADERS, timeout=10)
        # JSON optimistic path
        try:
            data = resp.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            d = data.get('data')
            if isinstance(d, dict):
                odds_bag = d.get('odds', {})
                pairs = odds_bag.get('4', odds_bag) if isinstance(odds_bag, dict) else {}
                if isinstance(pairs, dict):
                    for combo_key, vals in pairs.items():
                        if not isinstance(combo_key, str) or len(combo_key) != 4:
                            continue
                        try:
                            n1 = int(combo_key[:2])
                            n2 = int(combo_key[2:])
                        except ValueError:
                            continue
                        if not (n1 and n2):
                            continue
                        key = tuple(sorted([n1, n2]))
                        try:
                            odds_str = vals[0] if isinstance(vals, list) else vals
                            v = float(str(odds_str).replace(',', ''))
                            if v >= 1.0:
                                pair_odds[key] = v
                        except (ValueError, IndexError, TypeError):
                            continue
        # HTML fallback (古い応答形式の互換)
        if not pair_odds:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for row in soup.find_all('tr'):
                tds = row.find_all('td')
                if len(tds) < 2:
                    continue
                combo_text = tds[0].get_text(strip=True)
                nums = re.findall(r'\d+', combo_text)
                if len(nums) == 2:
                    key = tuple(sorted(int(n) for n in nums))
                    for td in tds[1:]:
                        try:
                            v = float(td.get_text(strip=True).replace(',', ''))
                            if v >= 1.0:
                                pair_odds[key] = v
                                break
                        except Exception:
                            continue
    except Exception:
        pass
    return pair_odds


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


def build_race_block(r, stars: int, pair_odds: dict | None = None) -> str:
    """1レース分のDiscord表示ブロックを返す。

    show_bets.compute_formation() と完全一致するロジックで
    三連複=1-2-5フォーメーション、馬連=2点オッズ連動配分を出力。
    pair_odds が渡された場合、馬連はオッズ高→400円 / 低→300円 に振り分ける。
    """
    # pandas.Series -> dict (CSV由来の型で compute_formation が解釈)
    row = r.to_dict() if hasattr(r, 'to_dict') else dict(r)
    form = compute_formation(row, pair_odds=pair_odds)

    star_str = '★' * stars if stars > 0 else '-'
    race_name_short = str(r.race_name)[:14]
    bt_jp = BET_JP.get(form['bet_type'] if form else r.bet_type, r.bet_type)

    # L1: 基本情報
    l1 = (f"**{r.course} {int(r.race_num)}R** "
          f"{r.surface}{int(r.distance)}m {race_name_short} "
          f"({int(r.num_horses)}頭)")
    # L2: 条件 / 馬券種 / 星
    l2 = f"  条件 **{r.condition}** / {bt_jp} / {star_str}"
    # L3: 軸
    l3 = (f"  軸: {int(r.top1_num)}番 {str(r.top1_name)[:10]} "
          f"(score {float(r.top1_score):.3f})")

    lines = [l1, l2, l3]

    if form is None:
        lines.append('  (買い目データ異常)')
        return '\n'.join(lines)

    if form['bet_type'] == 'umaren':
        # 馬連: オッズ連動配分 (取得失敗時は TOP2=400/TOP3=300)
        lines.append('  買い目:')
        exp_parts = []
        for partner, amt, odds in form['umaren_pairs']:
            if odds and odds > 0:
                odds_txt = f" (オッズ{odds:.1f}倍)"
                exp_parts.append(f"{amt}×{odds:.1f}={int(amt * odds):,}円")
            else:
                odds_txt = ''
            lines.append(
                f"    馬連 `{form['axis']}-{partner}`{odds_txt}: **{amt}円**"
            )
        lines.append(f"  合計: **¥{form['investment']}**")
        if exp_parts:
            lines.append(f"  期待払戻: {' or '.join(exp_parts)}")
    else:
        # 三連複: 1-2-5フォーメーション
        col2 = ', '.join(str(n) for n in form['second_col'])
        col3 = ', '.join(str(n) for n in form['third_col'])
        lines.append('  フォーメーション 1-2-5:')
        lines.append(f"    1列目: `{form['axis']}`")
        lines.append(f"    2列目: `{col2}`")
        lines.append(f"    3列目: `{col3}`")
        lines.append(f"  {form['n_points']}点 × 100円 = **¥{form['investment']}**")

    # L末: 波乱度・AI見解
    tb = turbulence(r.condition, int(r.num_horses))
    ac = ai_comment(r.condition, r.surface, int(r.distance), int(r.num_horses))
    lines.append(f"  波乱度 {tb} | AI: {ac}")

    return '\n'.join(lines)


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
            # 馬連のみリアルタイムオッズ取得(失敗時はフォールバック)
            pair_odds = None
            if str(r.bet_type).lower() == 'umaren':
                try:
                    pair_odds = fetch_umaren_pair_odds(str(r.race_id))
                except Exception:
                    pair_odds = None
            blocks.append(build_race_block(r, stars, pair_odds=pair_odds))

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
