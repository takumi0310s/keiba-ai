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

# Windows encoding: PYTHONIOENCODING=utf-8 (set in bat) handles this.
# Do NOT re-wrap sys.stdout here — it crashes when stdout is redirected to a file.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MINUTES_BEFORE = 5
JRDB_MINUTES_BEFORE = 20  # TYB取得タイミング（発走20分前）

_cached_model_data = None


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


def fetch_jrdb_tyb(date_str):
    """JRDB TYB(直前データ)をダウンロード・パース・CSV追記。
    パドック指数・オッズ指数・馬体コード・気配コード等を取得。
    """
    try:
        from scrape_jrdb import fetch_and_parse, save_csv
        jrdb_date = date_str[2:]  # YYYYMMDD → YYMMDD
        print(f"  [JRDB] TYB取得中... ({jrdb_date})")
        df = fetch_and_parse('TYB', jrdb_date)
        if df is not None and len(df) > 0:
            save_csv(df, 'TYB', append=True)
            print(f"  [JRDB] TYB: {len(df)} records saved")
            return len(df)
        else:
            print(f"  [JRDB] TYB: データなし")
            return 0
    except Exception as e:
        print(f"  [JRDB] TYB取得失敗: {e}")
        return -1


_tyb_fetched_dates = set()  # 同一日の二重取得防止


def predict_and_notify(race_info, date_str):
    """1レースの予測→Discord通知→DB保存"""
    race_id = race_info['race_id']
    print(f"\n  >>> Predicting: {race_info['course']}{race_info['race_num']}R ({race_id})")

    # TYB取得（発走前に1回だけ、同一日内で共有）
    if date_str not in _tyb_fetched_dates:
        _tyb_fetched_dates.add(date_str)
        fetch_jrdb_tyb(date_str)

    try:
        # Import prediction functions from predict_core (共通予測モジュール)
        from predict_core import (
            load_models, parse_shutuba, fetch_realtime_odds, fetch_realtime_odds_full, save_odds_base,
            classify_race_condition, generate_trio_bets, generate_umaren_bets,
            build_features, predict_race, is_race_started, fetch_result_odds,
            CONDITION_PROFILES, get_horse_stats, fetch_jra_and_weather,
            apply_horse_stats, set_horse_defaults,
        )
        from notify import send_discord

        global _cached_model_data
        if _cached_model_data is None:
            _cached_model_data = load_models()
        model_data = _cached_model_data
        if model_data['model'] is None:
            print("    Model not found")
            return

        # Parse shutuba
        race_name, horses, horse_ids, rinfo = parse_shutuba(race_id)
        if not horses:
            print("    No horse data")
            _p0_5_notify_log(race_id, None, datetime.now().isoformat(), channel='skip', strategy_7c_skip=False, strategy_7c_reason='no_horse_data')
            _v2_log_phase2_safe(race_id, None, None, None, None, None, None, channel='skip', strategy_7c_skip=False, strategy_7c_reason='no_horse_data')
            return

        # Skip obstacle races
        if rinfo.get('surface') == '障':
            print("    Skipping obstacle race")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=False, strategy_7c_reason='obstacle_race')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, None, None, None, channel='skip', strategy_7c_skip=False, strategy_7c_reason='obstacle_race')
            return

        num_horses = len(horses)
        distance = rinfo.get('distance', 0)

        # Skip 1000m or less
        if distance <= 1000:
            print("    Skipping <=1000m")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=False, strategy_7c_reason='distance_le_1000')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, None, None, None, channel='skip', strategy_7c_skip=False, strategy_7c_reason='distance_le_1000')
            return

        # ===== 戦略⑦ フィルタ (race_name + course) =====
        race_name_str = str(race_name)
        course_str = str(rinfo.get('course', ''))

        # 1. 06_特別 (G/L/OPEN特別 ではない平場特別) を除外
        is_graded = any(g in race_name_str for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'])
        is_listed = any(s in race_name_str for s in ['L)', '(L)', 'OP)', '(OP)'])
        is_open_tokubetsu = any(s in race_name_str for s in ['杯', '賞', 'ステークス', 'カップ', 'ハンデ'])
        if '特別' in race_name_str and not (is_graded or is_listed or is_open_tokubetsu):
            print(f"    [STRATEGY7] Skip 06_特別: {race_name_str}")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_06_tokubetsu')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, None, None, None, channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_06_tokubetsu')
            return

        # 2. 京都 filter (P0-2 案 C、 5/17 適用、 docs/P0_2_EXTENSION_DESIGN_2026_05_16.md)
        #    Kyoto×A (N=27、 p<0.001) + Kyoto×D (N=25、 p=0.021) 統計的に baseline 下回る
        #    G/L/OPEN特別 + Graded 重賞は除外しない (Victoria Mile 等 5/17 G1 day 影響回避)
        if course_str == '京都' and not (is_graded or is_listed):
            print(f"    [STRATEGY7] Skip 京都 (P0-2 案 C、 5/17 適用): {race_name_str}")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_kyoto_p0_2_5_17')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, None, None, None, channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_kyoto_p0_2_5_17')
            return
        # ===== 戦略⑦ フィルタ ここまで =====

        # Fetch odds (full = odds + pop_rank, save base cache for change features)
        odds_full = fetch_realtime_odds_full(race_id)
        odds_dict = {u: v['odds'] for u, v in odds_full.items()}
        if odds_full:
            save_odds_base(race_id, odds_full)
        time.sleep(1)

        # Fetch JRA track & weather
        jra_info, weather_info = {}, {}
        try:
            jra_info, weather_info = fetch_jra_and_weather(rinfo.get('course', ''))
        except Exception:
            pass

        # Get horse stats
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                try:
                    stats = get_horse_stats(hid, rinfo['distance'], rinfo['surface'], rinfo.get('course', ''))
                    apply_horse_stats(horse, stats, rinfo)
                except Exception:
                    set_horse_defaults(horse)
            else:
                set_horse_defaults(horse)
            if i < num_horses - 1:
                time.sleep(0.3)

        # 新馬評価（新馬戦の場合）
        if '新馬' in str(race_name):
            try:
                from scrape_shinba_eval import scrape_newspaper as _scrape_shinba
                shinba_rows = _scrape_shinba(race_id)
                if shinba_rows and shinba_rows != 'blocked' and len(shinba_rows) > 0:
                    for row in shinba_rows:
                        uma = str(row['umaban'])
                        for horse in horses:
                            if str(horse.get('馬番', '')) == uma:
                                horse['新馬厩舎評価'] = row['stable_eval']
                                horse['新馬調教ランク'] = row['training_rank']
                                horse['新馬スコア'] = row['comment_score']
                                break
                    rinfo['race_name'] = race_name  # predict_core用
            except Exception:
                pass

        # Build features and predict
        odds_available = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        df = build_features(horses, rinfo, model_data, race_id=race_id,
                            odds_dict=odds_dict, jra_track_info=jra_info, weather_info=weather_info)
        if df is None or len(df) == 0:
            print("    Feature build failed")
            return

        # JRDB特徴量マージ（KYI前日データ + TYB直前データ）
        try:
            from jrdb_features import merge_jrdb_predict_features
            df = merge_jrdb_predict_features(df, race_id)
        except Exception as e:
            print(f"    [JRDB] feature merge skipped: {e}")

        df = predict_race(df, model_data, odds_available, race_info=rinfo)

        # Sort by score
        df = df.sort_values('スコア', ascending=False).reset_index(drop=True)

        # JRDB指数をhorses dictにも転記（Discord通知用）
        for _, row in df.iterrows():
            uma = int(row.get('馬番', 0))
            for horse in horses:
                if int(horse.get('馬番', 0)) == uma:
                    horse['JRDB_IDM'] = row.get('jrdb_idm', 0)
                    horse['JRDB_パドック指数'] = row.get('jrdb_paddock_idx', 0)
                    horse['JRDB_オッズ指数'] = row.get('jrdb_odds_idx', 0)
                    horse['JRDB_激走指数'] = row.get('jrdb_upset_idx', 0)
                    horse['JRDB_総合指数'] = row.get('jrdb_composite_idx', 0)
                    break

        # Condition
        cond_key, cond_profile = classify_race_condition(rinfo, num_horses)

        # ===== 戦略⑦ フィルタ続き (条件判定後) =====

        # === STRATEGY_C4: Cond-A 1600-1800m drag 除外 (production active、重-2 +8.62pt confirmed) ===
        STRATEGY_C4_ENABLED = True
        if STRATEGY_C4_ENABLED and cond_key == 'A' and 1600 <= distance <= 1800:
            print(f"    [STRATEGY_C4] Skip Cond-A 1600-1800m: {race_name_str} dist={distance}")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_c4_condA_1600_1800')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, odds_dict, cond_key, None, channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_c4_condA_1600_1800')
            return

        if cond_key == 'E':
            print(f"    [STRATEGY7] Skip 条件E (頭数<=7)")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_cond_E')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, odds_dict, cond_key, None, channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_cond_E')
            return
        if cond_key == 'B':
            print(f"    [STRATEGY7] Skip 条件B (重~不馬場)")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_cond_B')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, odds_dict, cond_key, None, channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_cond_B')
            return
        # 条件 X (P0-2 案 C、 5/17 適用、 docs/P0_2_EXTENSION_DESIGN_2026_05_16.md)
        # 単一次元 N=19 ROI 8.72% 95% CI [0.00, 26.17] 統計的に baseline 下回る
        # Graded race 重賞は除外しない (G1/G2/G3 + L = 期待値高)
        if cond_key == 'X' and not (is_graded or is_listed):
            print(f"    [STRATEGY7] Skip 条件X (P0-2 案 C、 5/17 適用)")
            _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_cond_X_p0_2_5_17')
            _v2_log_phase2_safe(race_id, race_name, rinfo, None, odds_dict, cond_key, None, channel='skip', strategy_7c_skip=True, strategy_7c_reason='strategy_7_cond_X_p0_2_5_17')
            return
        # ===== 戦略⑦ フィルタ続き ここまで =====

        bet_type = cond_profile['bet_type']

        # Generate bets
        if bet_type == 'umaren':
            bets = generate_umaren_bets(df)
        else:
            bets = generate_trio_bets(df)
            # === STRATEGY_C3: pos2 (T1-T2-T4) bet 除外 trio 7→6点 (production active) ===
            STRATEGY_C3_ENABLED = True
            if STRATEGY_C3_ENABLED and bet_type == 'trio' and len(bets) >= 1:
                top4 = [int(df.iloc[i]['馬番']) for i in range(min(4, len(df)))]
                if len(top4) >= 4:
                    n1, n2, n4 = top4[0], top4[1], top4[3]
                    bet2_target = tuple(sorted([n1, n2, n4]))
                    bets_before = len(bets)
                    bets = [b for b in bets if tuple(sorted(b)) != bet2_target]
                    if len(bets) < bets_before:
                        print(f"    [STRATEGY_C3] Removed bet2 (T1-T2-T4)={list(bet2_target)} → {len(bets)}点")

        # 週末限定データ（波乱度・AI予測）— キャッシュ→リアルタイム取得フォールバック
        _upset_data, _newspaper_data = {}, {}
        try:
            from scrape_weekend_thisweek import load_thisweek_upset, load_thisweek_newspaper
            _upset_data = load_thisweek_upset().get(race_id, {})
            _newspaper_data = load_thisweek_newspaper().get(race_id, {})
        except Exception:
            pass
        # キャッシュに波乱度がない or Lv0の場合、リアルタイム取得
        if not _upset_data.get('upset_level'):
            try:
                from scrape_newspaper_ai import _make_session as _nai_session, scrape_shutuba_upset
                _nai_sess = _nai_session()
                if _nai_sess:
                    _upset_data = scrape_shutuba_upset(_nai_sess, race_id)
                    time.sleep(2)
            except Exception:
                pass
        if not _newspaper_data.get('ai_horse_times'):
            try:
                from scrape_newspaper_ai import _make_session as _nai_session2, scrape_newspaper as _scrape_np
                _nai_sess2 = _nai_session2()
                if _nai_sess2:
                    _newspaper_data = _scrape_np(_nai_sess2, race_id)
                    time.sleep(2)
            except Exception:
                pass

        # === STRATEGY_B1: V15 top1 = 市場1番人気のとき skip (paper eval only、N=41 N不足、6/17 判定) ===
        STRATEGY_B1_PAPER_ONLY = True
        _b1_top1_pop = int(df.iloc[0].get('pop_rank', 0)) if len(df) > 0 else 0
        _b1_skip = (_b1_top1_pop == 1)
        if STRATEGY_B1_PAPER_ONLY and _b1_skip:
            print(f"    [STRATEGY_B1][PAPER] would skip: top1 pop_rank=1 → {race_name_str}")

        # === STRATEGY_B2: V15-市場 divergence: top1 pop_rank >= 3 のみ (paper eval only) ===
        # +10-20pt候補 N不足のため paper。 top1_pop_rank < MIN のとき skip (実際は log only)
        STRATEGY_B2_PAPER_ONLY = True
        STRATEGY_B2_MIN_POP_RANK = 3  # pop_rank 1-2 = V15 が市場と一致 → skip
        _b2_skip = (_b1_top1_pop > 0 and _b1_top1_pop < STRATEGY_B2_MIN_POP_RANK)
        if STRATEGY_B2_PAPER_ONLY and _b2_skip:
            print(f"    [STRATEGY_B2][PAPER] would skip: top1 pop_rank={_b1_top1_pop} < {STRATEGY_B2_MIN_POP_RANK} → {race_name_str}")

        # === STRATEGY_C1: EV>1 trio フィルタ (paper eval only、bet-level EV 計算) ===
        STRATEGY_C1_PAPER_ONLY = True
        STRATEGY_C1_EV_THRESHOLD = 1.0
        STRATEGY_C1_DEFAULT_PAYOUT = 5000  # trio 平均想定 (円)
        if STRATEGY_C1_PAPER_ONLY and bet_type == 'trio' and len(bets) > 0:
            _scores = {}
            for _, _row in df.iterrows():
                _scores[int(_row.get('馬番', 0))] = float(_row.get('スコア', 0.01))
            _total_score = max(sum(_scores.values()), 1e-6)
            _c1_high_ev = []
            for _b in bets:
                _p = (_scores.get(_b[0], 0.01) * _scores.get(_b[1], 0.01) * _scores.get(_b[2], 0.01)) / (_total_score ** 3 + 1e-9)
                _ev = _p * STRATEGY_C1_DEFAULT_PAYOUT / 100.0
                if _ev >= STRATEGY_C1_EV_THRESHOLD:
                    _c1_high_ev.append(_b)
            print(f"    [STRATEGY_C1][PAPER] EV>={STRATEGY_C1_EV_THRESHOLD} bets: {len(_c1_high_ev)}/{len(bets)}")

        # === STRATEGY_C2: odds 帯フィルタ (paper eval only、過剰人気/極値/東京帯) ===
        STRATEGY_C2_PAPER_ONLY = True
        _c2_top1_odds = 0.0
        if len(df) > 0:
            _uma1 = int(df.iloc[0].get('馬番', 0))
            _c2_top1_odds = float(odds_dict.get(_uma1, odds_dict.get(str(_uma1), 0.0)))
        _c2_skip = False
        _c2_reason = ''
        if _c2_top1_odds > 0:
            if _c2_top1_odds < 1.5:
                _c2_skip, _c2_reason = True, f'odds<1.5({_c2_top1_odds})'
            elif _c2_top1_odds > 20.0:
                _c2_skip, _c2_reason = True, f'odds>20({_c2_top1_odds})'
            elif '東京' in course_str and 5.0 <= _c2_top1_odds <= 10.0:
                _c2_skip, _c2_reason = True, f'Tokyo 5-10x({_c2_top1_odds})'
        if STRATEGY_C2_PAPER_ONLY and _c2_skip:
            print(f"    [STRATEGY_C2][PAPER] would skip: {_c2_reason} → {race_name_str}")

        # 収益パターンマッチ
        _pp_stars, _pp_matched = 0, []
        try:
            from predict_core import match_profitable_patterns
            _pp_top1 = min((v for v in odds_dict.values() if v > 0), default=0)
            _pp_stars, _pp_matched = match_profitable_patterns(
                cond_key, rinfo.get('course', ''), rinfo.get('surface', '芝'),
                rinfo.get('distance', 1600), _pp_top1)
        except Exception:
            pass

        # リッチ通知（共通フォーマット）
        from notify import build_rich_bet_message
        # race_infoにstart_timeを追加（race listから取得した情報をマージ）
        rinfo['start_time'] = race_info.get('start_time', rinfo.get('start_time', ''))
        rinfo['course'] = rinfo.get('course', '') or race_info.get('course', '')

        title, msg, color = build_rich_bet_message(
            df, race_name, rinfo, cond_key, cond_profile,
            bets, odds_dict=odds_dict, horses=horses, date_str=date_str,
            upset_data=_upset_data, newspaper_data=_newspaper_data,
            pp_stars=_pp_stars, pp_matched=_pp_matched)
        send_discord(title, msg, color=color, channel="bets")
        print(f"    Notified: {race_name} [{cond_key}] {bet_type} {len(bets)}点")
        _p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel='bets', strategy_7c_skip=False, strategy_7c_reason=None)
        _v2_log_phase2_safe(race_id, race_name, rinfo, bets, odds_dict, cond_key, bet_type, channel='bets', strategy_7c_skip=False, strategy_7c_reason=None)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            from notify import send_discord
            send_discord("予測エラー", f"{race_info['course']}{race_info['race_num']}R: {str(e)[:100]}", color="red")
        except Exception:
            pass
        try:
            _p0_5_notify_log(race_id, None, datetime.now().isoformat(), channel='error', strategy_7c_skip=False, strategy_7c_reason=f'exception:{str(e)[:100]}')
        except Exception:
            pass
        try:
            _v2_log_phase2_safe(race_id, None, None, None, None, None, None, channel='error', strategy_7c_skip=False, strategy_7c_reason=f'exception:{str(e)[:100]}')
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
    """テストモード: 両チャンネルにテスト通知（リッチ版フォーマット）"""
    print("=== TEST MODE ===")
    try:
        from notify import send_discord
        from datetime import datetime as _dt
        _now = _dt.now()
        _wdmap = {0:'月',1:'火',2:'水',3:'木',4:'金',5:'土',6:'日'}
        _date = f"{_now.month}/{_now.day}({_wdmap[_now.weekday()]})"

        # Bets channel - リッチ版フォーマット
        ok1 = send_discord(f"🏇 {_date} 中山11R 15:45発走",
                           f"**アネモネS** 芝1600m 良 14頭\n"
                           f"条件A ★★★ ROI 205.3% (的中44.5%)\n\n"
                           f"三連複フォーメーション 7点\n"
                           f"1列目: 5\n"
                           f"2列目: 2, 8\n"
                           f"3列目: 2, 3, 8, 11, 14\n\n"
                           f"軸: 5 ホワイトオーキッド (スコア0.85)\n"
                           f"2位: 2 テストホースB (スコア0.78)\n"
                           f"3位: 8 テストホースC (スコア0.71)\n\n"
                           f"💰 配当レンジ: 1,200円〜15,600円\n"
                           f"投資額: 700円\n\n"
                           f"📊 Premium ✓  指数: 1127 / 調教: A / 厩舎: 好調",
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

    # 全35R一括通知（1-3メッセージ、全体把握用）
    try:
        from notify_bets_all_in_one import send_all_in_one
        sent_aio = send_all_in_one(date_str, channel='bets')
        print(f"  全レース一括通知: {sent_aio} messages")
    except Exception as e:
        print(f"  全レース一括通知失敗: {e}")

    # 当日一括の整形済み買い目通知（#買い目）を1回だけ送信
    # AM8:00 daily_predict で既に送信済みの場合もあるが、レース当日朝の
    # 8:45起動時点でオッズが更新されている可能性があるため再送する
    try:
        from notify_bets_formatted import notify_formatted
        sent = notify_formatted(date_str, mode='morning', channel='bets')
        print(f"  整形済み買い目通知: {sent} messages")
    except Exception as e:
        print(f"  整形済み買い目通知失敗: {e}")

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


def _v2_log_phase2_safe(race_id, race_name, rinfo, bets, odds_dict, cond_key, bet_type,
                         channel='bets', strategy_7c_skip=False, strategy_7c_reason=None):
    """race_notify_log v2 phase 2 (pre_vote) safe wrapper。

    ★ 既存 logic 完全不変、 log 出力 file IO のみ ★

    log fail / import fail で例外を投げない (race_auto_notify の logic に影響なし)。
    """
    try:
        from race_notify_log_v2 import log_phase2 as _v2_log_phase2

        race_meta = {}
        if rinfo:
            try:
                race_meta = {
                    'race_name': str(race_name) if race_name else '',
                    'course': str(rinfo.get('course', '')),
                    'distance': rinfo.get('distance', 0),
                    'surface': rinfo.get('surface', ''),
                    'condition': rinfo.get('condition', ''),
                    'start_time': str(rinfo.get('start_time', '')),
                }
            except Exception:
                race_meta = {'race_name': str(race_name) if race_name else ''}
        elif race_name:
            race_meta = {'race_name': str(race_name)}

        _v2_log_phase2(
            race_id=race_id,
            race_meta=race_meta,
            formation_actual=bets,
            vote_time_odds=odds_dict or {},
            strategy_7c_skip=strategy_7c_skip,
            strategy_7c_reason=strategy_7c_reason,
            channel=channel,
            cond_key=cond_key,
            bet_type=bet_type,
        )
    except Exception as _e:
        import sys as _sys
        print(f"[race_notify_log_v2 wrapper fail] {_e}", file=_sys.stderr)


def _p0_5_notify_log(race_id, race_name, notified_at, channel='bets', strategy_7c_skip=False, strategy_7c_reason=None):
    """P0-5 用 通知済 race log 出力 (★ 既存 logic 完全不変、 file IO のみ ★)

    出力: data/race_notify_log/YYYYMMDD.json (append)

    fail 時は log エラーのみ stderr 出力、 通知 logic に影響なし。
    """
    try:
        from pathlib import Path
        import json as _json
        from datetime import datetime as _dt

        log_dir = Path(__file__).resolve().parents[1] / 'data' / 'race_notify_log'
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = _dt.now().strftime('%Y%m%d')
        log_file = log_dir / f'{date_str}.json'

        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = _json.load(f)
                if not isinstance(logs, list):
                    logs = []
            except Exception:
                logs = []
        else:
            logs = []

        logs.append({
            'race_id': str(race_id) if race_id is not None else None,
            'race_name': str(race_name) if race_name else None,
            'notified_at': str(notified_at),
            'channel': channel,
            'strategy_7c_skip': bool(strategy_7c_skip),
            'strategy_7c_reason': str(strategy_7c_reason) if strategy_7c_reason else None,
        })

        with open(log_file, 'w', encoding='utf-8') as f:
            _json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as _e:
        import sys as _sys
        print(f"[P0-5 log fail] {_e}", file=_sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            from notify import send_discord
            send_discord("Auto-Notify CRASH", f"```{traceback.format_exc()[-500:]}```", color="red")
        except Exception:
            pass
