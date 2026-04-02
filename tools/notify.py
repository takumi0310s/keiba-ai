"""Discord通知モジュール（チャンネル振り分け対応）

Usage:
    from tools.notify import send_discord, build_rich_bet_message
    send_discord("予測完了", "...", color="green", channel="bets")
    send_discord("スクレイピング完了", "...", color="blue", channel="updates")

    # リッチ買い目通知
    title, msg, color = build_rich_bet_message(df, race_name, race_info, cond_key,
                                                cond_profile, bets, odds_dict, horses)
    send_discord(title, msg, color=color, channel="bets")

Channels:
    "bets"    → DISCORD_WEBHOOK_BETS (買い目通知)
    "updates" → DISCORD_WEBHOOK_UPDATES (システム通知)
    未指定     → DISCORD_WEBHOOK_URL (フォールバック)
"""
import os
import re
from datetime import datetime
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {"green": 0x4ade80, "yellow": 0xf0c040, "red": 0xff4060, "blue": 0x60b0ff}

_ENV_CACHE = None


def _load_env():
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    _ENV_CACHE = {}
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return _ENV_CACHE
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                _ENV_CACHE[key.strip()] = val.strip('"').strip("'")
    return _ENV_CACHE


def _get_webhook_url(channel="updates"):
    env = _load_env()
    if channel == "bets":
        url = env.get('DISCORD_WEBHOOK_BETS', '')
        if url.startswith('https://'):
            return url
    if channel == "updates":
        url = env.get('DISCORD_WEBHOOK_UPDATES', '')
        if url.startswith('https://'):
            return url
    # Fallback
    url = env.get('DISCORD_WEBHOOK_URL', '')
    return url if url.startswith('https://') else None


def send_discord(title, message, color="green", fields=None, channel="updates"):
    """Discord Webhook通知を送信。URLが未設定ならスキップ。

    Args:
        channel: "bets" (買い目) or "updates" (システム通知)
    """
    url = _get_webhook_url(channel)
    if not url:
        return False

    embed = {
        "title": title[:256],
        "description": message[:2000],
        "color": COLORS.get(color, COLORS["blue"]),
    }
    if fields:
        embed["fields"] = [{"name": str(k)[:256], "value": str(v)[:200], "inline": True}
                           for k, v in list(fields.items())[:10]]

    try:
        resp = requests.post(url, json={"embeds": [embed]}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False


def build_rich_bet_message(df, race_name, race_info, cond_key, cond_profile,
                           bets, odds_dict=None, horses=None, date_str=None):
    """リッチな買い目通知メッセージを構築。全通知元で共通フォーマット。

    Args:
        df: スコアでソート済みの予測DataFrame (columns: 馬番, 馬名, スコア, etc.)
        race_name: レース名
        race_info: dict with keys: course, race_num, distance, surface, condition,
                   start_time(optional), weather(optional), grade(optional)
        cond_key: 条件キー (A-E, X)
        cond_profile: CONDITION_PROFILESの値
        bets: 買い目リスト
        odds_dict: {馬番: 単勝オッズ} (optional)
        horses: 出走馬リスト (premium data用, optional)
        date_str: 日付文字列 YYYYMMDD (optional, default=today)

    Returns:
        (title, message, color) tuple
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')

    # 日付フォーマット: 3/28(土)
    weekday_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        date_disp = f"{dt.month}/{dt.day}({weekday_map[dt.weekday()]})"
    except Exception:
        date_disp = date_str

    course = race_info.get('course', '')
    race_num = race_info.get('race_num', '')
    start_time = race_info.get('start_time', '')
    distance = race_info.get('distance', 0)
    surface = race_info.get('surface', '')
    condition = race_info.get('condition', '')
    num_horses = len(df)
    bet_type = cond_profile.get('bet_type', 'trio')
    roi = cond_profile.get('roi', 0)
    hit_rate = cond_profile.get('hit_rate', 0)
    investment = cond_profile.get('investment', 700)

    stars = '★★★' if roi >= 200 else ('★★' if roi >= 100 else '★')

    # Title: 🏇 3/28(土) 中山1R 10:00発走
    time_part = f" {start_time}発走" if start_time else ""
    title = f"🏇 {date_disp} {course}{race_num}{time_part}"

    # Line 1: レース名 surface+distance condition 頭数
    lines = [f"**{race_name}** {surface}{distance}m {condition} {num_horses}頭"]
    # Line 2: 条件 + stars + ROI
    lines.append(f"条件{cond_key} {stars} ROI {roi:.1f}% (的中{hit_rate:.1f}%)")
    lines.append("")

    # 買い目
    if bet_type == 'umaren':
        n1 = int(df.iloc[0]['馬番'])
        n2 = int(df.iloc[1]['馬番'])
        n3 = int(df.iloc[2]['馬番'])
        lines.append(f"馬連 1軸2流し")
        lines.append(f"軸: {n1} → {n2}, {n3}")
    else:
        top6 = df.head(6)
        n1 = int(top6.iloc[0]['馬番'])
        col2 = sorted([int(top6.iloc[1]['馬番']), int(top6.iloc[2]['馬番'])])
        col3 = sorted([int(top6.iloc[i]['馬番']) for i in range(1, min(6, len(top6)))])
        lines.append(f"三連複フォーメーション {len(bets)}点")
        lines.append(f"1列目: {n1}")
        lines.append(f"2列目: {', '.join(str(n) for n in col2)}")
        lines.append(f"3列目: {', '.join(str(n) for n in col3)}")
    lines.append("")

    # TOP3
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        num = int(row['馬番'])
        name = row.get('馬名', '?')
        score = row.get('スコア', 0)
        rank_label = ['軸', '2位', '3位'][i]
        lines.append(f"{rank_label}: {num} {name} (スコア{score:.2f})")
    lines.append("")

    # 配当レンジ
    try:
        ro = odds_dict or {}
        if ro and bet_type != 'umaren' and len(bets) > 0:
            payouts_est = []
            for b in bets:
                o = [ro.get(int(x), 10.0) for x in b]
                est = o[0] * o[1] * o[2] * 0.6
                payouts_est.append(max(100, int(est * 100)))
            if payouts_est:
                lines.append(f"💰 配当レンジ: {min(payouts_est):,}円〜{max(payouts_est):,}円")
        elif ro and bet_type == 'umaren':
            n1v = int(df.iloc[0]['馬番'])
            n2v = int(df.iloc[1]['馬番'])
            n3v = int(df.iloc[2]['馬番'])
            o1 = ro.get(n1v, 10.0)
            o2 = ro.get(n2v, 10.0)
            o3 = ro.get(n3v, 10.0)
            est1 = max(100, int(o1 * o2 * 5))
            est2 = max(100, int(o1 * o3 * 5))
            lines.append(f"💰 配当目安: {est1:,}円 / {est2:,}円")
    except Exception:
        pass
    lines.append(f"投資額: {investment}円")

    # Premium data
    premium_parts = []
    if horses and len(horses) > 0:
        h0 = horses[0] if isinstance(horses[0], dict) else {}
        si = h0.get('タイム指数', 0) or h0.get('speed_index', 0)
        if si and si > 1000:
            premium_parts.append(f"指数: {si}")
        rank = h0.get('調教ランク', '') or h0.get('training_rank', '')
        if rank:
            premium_parts.append(f"調教: {rank}")
        cs = h0.get('厩舎スコア', 0) or h0.get('stable_score', 0)
        if cs and cs > 0:
            premium_parts.append("厩舎: 好調")
    # Also check df columns for premium data
    if not premium_parts and len(df) > 0:
        top1 = df.iloc[0]
        si = top1.get('タイム指数', 0)
        if si and si > 1000:
            premium_parts.append(f"指数: {si}")
    if premium_parts:
        lines.append(f"\n📊 Premium ✓  {' / '.join(premium_parts)}")

    # JRDB指数（TOP3馬のIDM・パドック指数・オッズ指数）
    jrdb_lines = []
    if horses and len(horses) > 0:
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            uma = int(row.get('馬番', 0))
            # horses dictからJRDB指数を取得
            h_jrdb = None
            for h in horses:
                if isinstance(h, dict) and int(h.get('馬番', 0)) == uma:
                    h_jrdb = h
                    break
            if h_jrdb is None:
                continue
            idm = h_jrdb.get('JRDB_IDM', 0)
            paddock = h_jrdb.get('JRDB_パドック指数', 0)
            odds_idx = h_jrdb.get('JRDB_オッズ指数', 0)
            # デフォルト値(50.0)以外なら表示
            parts_j = []
            if idm and float(idm) != 50.0:
                parts_j.append(f"IDM:{idm:.0f}")
            if paddock and float(paddock) != 50.0:
                parts_j.append(f"パド:{paddock:.0f}")
            if odds_idx and float(odds_idx) != 50.0:
                parts_j.append(f"ｵｯﾂﾞ:{odds_idx:.0f}")
            if parts_j:
                name = row.get('馬名', '?')
                jrdb_lines.append(f"  {uma} {name}: {' / '.join(parts_j)}")
    if jrdb_lines:
        lines.append("\n🎯 JRDB指数")
        lines.extend(jrdb_lines)

    # 新馬評価（新馬戦の場合）
    race_name_str = str(race_name) if race_name else ''
    if '新馬' in race_name_str and horses:
        shinba_lines = []
        for h in (horses[:3] if len(horses) >= 3 else horses):
            if not isinstance(h, dict):
                continue
            se = h.get('新馬厩舎評価', '') or h.get('shinba_stable_eval', '')
            tr = h.get('新馬調教ランク', '') or h.get('shinba_training_rank', '')
            cs = h.get('新馬スコア', None)
            if cs is None:
                cs = h.get('shinba_comment_score', None)
            if se or tr:
                name = h.get('馬名', '?')
                parts_s = []
                if se:
                    parts_s.append(f"厩舎{se}")
                if tr:
                    parts_s.append(f"調教{tr}")
                if cs is not None:
                    sign = '+' if cs > 0 else ''
                    parts_s.append(f"スコア{sign}{cs}")
                shinba_lines.append(f"  {name}: {'/'.join(parts_s)}")
        if shinba_lines:
            lines.append("\n🐴 新馬評価")
            lines.extend(shinba_lines)

    msg = "\n".join(lines)
    color = "green" if roi >= 200 else ("blue" if roi >= 100 else "yellow")
    return title, msg, color
