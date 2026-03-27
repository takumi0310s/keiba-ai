#!/usr/bin/env python
"""レース5分前自動予測＆Discord通知システム

起動後、当日の全JRAレースの発走5分前に自動で予測→Discord通知→DB保存。

Usage:
    python tools/race_auto_notify.py              # 本番
    python tools/race_auto_notify.py --test        # ダミーテスト
    python tools/race_auto_notify.py --date 20260328
"""
import os
import sys
import io
import re
import json
import time
import argparse
import threading
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MINUTES_BEFORE = 5


def get_todays_races(date_str):
    """netkeibaから当日全レースの発走時刻を取得。
    Returns: [(race_id, race_name, course, race_num, start_time_str), ...]
    """
    try:
        from scrape_training import _make_session
        session = _make_session() or requests.Session()
        session.headers.update(HEADERS)
    except Exception:
        session = requests.Session()
        session.headers.update(HEADERS)

    races = []
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = session.get(url, timeout=15)
        resp.encoding = 'EUC-JP'
        soup = BeautifulSoup(resp.text, 'html.parser')

        for a in soup.find_all('a', href=True):
            m = re.search(r'race_id=(\d{12})', a['href'])
            if not m:
                continue
            race_id = m.group(1)
            # Get race info from the link context
            parent = a.find_parent('li') or a.find_parent('div')
            text = parent.get_text(strip=True) if parent else a.get_text(strip=True)

            # Extract start time (HH:MM format)
            tm = re.search(r'(\d{1,2}:\d{2})', text)
            start_time = tm.group(1) if tm else ''

            # Extract race number
            rn = re.search(r'(\d{1,2})R', text)
            race_num = int(rn.group(1)) if rn else 0

            # Course from race_id: positions 4-5 are course code
            course_code = race_id[4:6]
            course_map = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
                          '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
            course = course_map.get(course_code, '?')

            races.append({
                'race_id': race_id,
                'course': course,
                'race_num': race_num,
                'start_time': start_time,
                'text': text[:50],
            })
    except Exception as e:
        print(f"  ERROR: {e}")

    # Deduplicate by race_id
    seen = set()
    unique = []
    for r in races:
        if r['race_id'] not in seen:
            seen.add(r['race_id'])
            unique.append(r)
    return sorted(unique, key=lambda r: r.get('start_time', ''))


def predict_and_notify(race_info, date_str):
    """1レースの予測→Discord通知→DB保存"""
    race_id = race_info['race_id']
    print(f"\n  >>> Predicting: {race_info['course']}{race_info['race_num']}R ({race_id})")

    try:
        # Import prediction functions from daily_predict
        from daily_predict import (
            load_models, parse_shutuba, fetch_realtime_odds,
            classify_condition, generate_trio_bets, generate_umaren_bets,
            build_feature_df, is_race_started, fetch_result_odds,
            CONDITION_PROFILES,
        )
        from notify import send_discord

        model_data = load_models()
        if model_data['model'] is None:
            print("    Model not found")
            return

        # Parse shutuba
        race_name, horses, horse_ids, rinfo = parse_shutuba(race_id)
        if not horses:
            print("    No horse data")
            return

        # Skip obstacle races
        if rinfo.get('surface') == '障':
            print("    Skipping obstacle race")
            return

        num_horses = len(horses)
        distance = rinfo.get('distance', 0)

        # Skip 1000m or less
        if distance <= 1000:
            print("    Skipping <=1000m")
            return

        # Fetch odds
        odds_dict = fetch_realtime_odds(race_id)
        time.sleep(1)

        # Build features and predict
        df = build_feature_df(horses, horse_ids, rinfo, model_data, odds_dict)
        if df is None or len(df) == 0:
            print("    Feature build failed")
            return

        # Sort by score
        df = df.sort_values('スコア', ascending=False).reset_index(drop=True)

        # Condition
        cond_key, cond_profile = classify_condition(rinfo, num_horses)
        bet_type = cond_profile['bet_type']
        roi = cond_profile['roi']

        # Stars
        stars = '★★★' if roi >= 200 else ('★★' if roi >= 100 else '★')

        # Generate bets
        if bet_type == 'umaren':
            bets = generate_umaren_bets(df)
            n1 = int(df.iloc[0]['馬番'])
            n2 = int(df.iloc[1]['馬番'])
            n3 = int(df.iloc[2]['馬番'])
            bet_text = f"馬連 1軸2流し\n軸: {n1}\n相手: {n2}, {n3}\n投資額: 700円（400+300円）"
        else:
            bets = generate_trio_bets(df)
            top6 = df.head(6)
            n1 = int(top6.iloc[0]['馬番'])
            col2 = sorted([int(top6.iloc[1]['馬番']), int(top6.iloc[2]['馬番'])])
            col3 = sorted([int(top6.iloc[i]['馬番']) for i in range(1, min(6, len(top6)))])
            bet_text = (f"三連複フォーメーション {len(bets)}点\n"
                        f"1列目: {n1}\n"
                        f"2列目: {', '.join(str(n) for n in col2)}\n"
                        f"3列目: {', '.join(str(n) for n in col3)}\n"
                        f"投資額: {len(bets)*100}円")

        # Top horse info
        top1 = df.iloc[0]
        top1_name = top1.get('馬名', '?')
        top1_score = top1.get('スコア', 0)

        # Premium data hints
        premium_parts = []
        si = top1.get('タイム指数', 0)
        if si and si > 1000:
            premium_parts.append(f"指数: {si}")
        rank = ''
        if len(horses) > 0:
            rank = horses[0].get('調教ランク', '')
        if rank:
            premium_parts.append(f"調教: {rank}")
        comment_score = horses[0].get('厩舎スコア', 0) if len(horses) > 0 else 0
        if comment_score > 0:
            premium_parts.append("厩舎: 好調")
        premium_line = " / ".join(premium_parts)

        surface = rinfo.get('surface', '?')
        condition = rinfo.get('condition', '?')
        start_time = race_info.get('start_time', '?')

        # Payout range estimate from tansho odds
        payout_range = ''
        try:
            if bet_type != 'umaren' and len(bets) > 0:
                payouts_est = []
                for b in bets:
                    o = [odds_dict.get(int(x), 10.0) for x in b]
                    est = o[0] * o[1] * o[2] * 0.6  # trio ≈ product × 0.6
                    payouts_est.append(max(100, int(est * 100)))
                payout_range = f"\n💰 配当レンジ: {min(payouts_est):,}円〜{max(payouts_est):,}円"
            elif bet_type == 'umaren' and odds_dict:
                o1 = odds_dict.get(n1, 10.0)
                o2 = odds_dict.get(n2, 10.0)
                o3 = odds_dict.get(n3, 10.0)
                est1 = max(100, int(o1 * o2 * 5))
                est2 = max(100, int(o1 * o3 * 5))
                payout_range = f"\n💰 配当目安: {est1:,}円 / {est2:,}円"
        except Exception:
            pass

        msg = (f"**{race_name}** {surface}{distance}m {condition} 条件{cond_key} {stars}\n\n"
               f"{bet_text}\n\n"
               f"軸: {top1_name} ({n1}) スコア{top1_score:.2f}"
               f"{payout_range}")
        if premium_line:
            msg += f"\n📊 {premium_line}"

        color = "green" if roi >= 200 else ("blue" if roi >= 100 else "yellow")
        send_discord(f"🏇 {race_info['course']}{race_info['race_num']}R 発走{start_time}",
                     msg, color=color, channel="bets")
        print(f"    Notified: {race_name} [{cond_key}] {bet_type} {len(bets)}点")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            from notify import send_discord
            send_discord("予測エラー", f"{race_info['course']}{race_info['race_num']}R: {str(e)[:100]}", color="red")
        except Exception:
            pass


def schedule_race(race, date_str):
    """指定レースの5分前にpredict_and_notifyを実行するタイマーをセット"""
    start_time = race.get('start_time', '')
    if not start_time or ':' not in start_time:
        return None

    try:
        today = datetime.strptime(date_str, '%Y%m%d')
        h, m = start_time.split(':')
        race_time = today.replace(hour=int(h), minute=int(m), second=0)
        notify_time = race_time - timedelta(minutes=MINUTES_BEFORE)
        now = datetime.now()
        delay = (notify_time - now).total_seconds()

        if delay < 0:
            print(f"  {race['course']}{race['race_num']}R {start_time}: already passed")
            return None

        timer = threading.Timer(delay, predict_and_notify, args=(race, date_str))
        timer.daemon = True
        timer.start()
        print(f"  {race['course']}{race['race_num']}R {start_time}: notify at {notify_time.strftime('%H:%M')} (in {delay/60:.0f}min)")
        return timer
    except Exception as e:
        print(f"  {race['course']}{race['race_num']}R: schedule error: {e}")
        return None


def run_test():
    """テストモード: 両チャンネルにテスト通知"""
    print("=== TEST MODE ===")
    try:
        from notify import send_discord

        # Bets channel
        ok1 = send_discord("🏇 テスト買い目",
                           "**テストレース** 芝2000m 良 条件A ★★★\n\n"
                           "三連複フォーメーション 7点\n"
                           "1列目: 5\n"
                           "2列目: 2, 8\n"
                           "3列目: 2, 3, 8, 11, 14\n"
                           "投資額: 700円\n\n"
                           "軸: テストホース (5) スコア0.85\n"
                           "💰 配当レンジ: 1,200円〜15,600円\n"
                           "📊 指数: 1127 / 調教: A / 厩舎: 好調",
                           color="green", channel="bets")
        print(f"  Bets channel: {'OK' if ok1 else 'SKIPPED (URL未設定)'}")

        # Updates channel
        ok2 = send_discord("テスト通知",
                           "システム通知のテストです。",
                           color="blue", channel="updates")
        print(f"  Updates channel: {'OK' if ok2 else 'SKIPPED (URL未設定)'}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="レース5分前自動予測＆Discord通知")
    parser.add_argument('--date', type=str, default='')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        run_test()
        return

    date_str = args.date or datetime.now().strftime('%Y%m%d')

    print("=" * 60)
    print(f"  Race Auto-Notify: {date_str}")
    print(f"  Notify: {MINUTES_BEFORE}min before each race")
    print("=" * 60)

    # Get today's races
    print("\n  Fetching race list...")
    races = get_todays_races(date_str)
    print(f"  Found: {len(races)} races")

    if not races:
        print("  No races today.")
        try:
            from notify import send_discord
            send_discord("Auto-Notify", f"{date_str}: レースなし", color="yellow")
        except Exception:
            pass
        return

    # Schedule all races
    print("\n  Scheduling notifications...")
    timers = []
    for race in races:
        t = schedule_race(race, date_str)
        if t:
            timers.append(t)

    active = len(timers)
    print(f"\n  Active timers: {active}")

    if active == 0:
        print("  All races already passed or no valid times.")
        return

    # Startup notification
    try:
        from notify import send_discord
        first = races[0].get('start_time', '?')
        last = races[-1].get('start_time', '?')
        send_discord("Auto-Notify起動",
                     f"{date_str}: {len(races)}R ({first}〜{last})\n{active}件の通知をスケジュール済み",
                     color="blue")
    except Exception:
        pass

    # Keep running until all timers fire
    print(f"\n  Waiting for races... (Ctrl+C to stop)")
    try:
        while any(t.is_alive() for t in timers):
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        for t in timers:
            t.cancel()

    print("\n  All races processed. Exiting.")

    # End-of-day notification
    try:
        from notify import send_discord
        send_discord("Auto-Notify終了", f"{date_str}: 全レース完了", color="blue")
    except Exception:
        pass


if __name__ == '__main__':
    main()
