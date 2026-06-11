#!/usr/bin/env python
"""V15/V21 並行予測スクリプト (paper trading)

V15 production logic COMPLETELY UNCHANGED.
V21 は [V21 paper] 明示 / 実 cash 投票は V15 のみ。

Usage:
    python tools/v21_paper_predict.py --date 20260524
    python tools/v21_paper_predict.py --race 202605240511
    python tools/v21_paper_predict.py --date 20260524 --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tools'))

# ── paths ──────────────────────────────────────────────────────────────
V15_MODEL_PATH = BASE_DIR / 'keiba_model_v15_central.pkl.gz'
V21_MODEL_PATH = BASE_DIR / 'models' / 'v21_candidate.pkl.gz'
V21_PAPER_LOG_DIR = BASE_DIR / 'data' / 'v21_paper_log'

# ── Discord ─────────────────────────────────────────────────────────────
def _load_env():
    env_path = BASE_DIR / '.env'
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, _, v = line.partition('=')
                env[k.strip()] = v.strip().strip('"')
    return env

_ENV = _load_env()

def _get_webhook_v21() -> str:
    return (
        os.environ.get('DISCORD_WEBHOOK_V21_PAPER')
        or _ENV.get('DISCORD_WEBHOOK_V21_PAPER')
        or os.environ.get('DISCORD_WEBHOOK_UPDATES')
        or _ENV.get('DISCORD_WEBHOOK_UPDATES')
        or ''
    )

def _send_discord(webhook_url: str, content: str, dry_run: bool = False) -> bool:
    if dry_run:
        print(f"[DRY-RUN] Discord message:\n{content}")
        return True
    if not webhook_url:
        print("[WARN] V21 Discord webhook not set; skipping send")
        return False
    try:
        import requests as _req
        resp = _req.post(webhook_url, json={'content': content}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception as e:
        print(f"[WARN] Discord send failed: {e}")
        return False


# ── model loading ────────────────────────────────────────────────────────
def _load_pkl_gz(path: Path):
    """gzip + pickle load. Returns None on any failure (graceful)."""
    if not path.exists():
        return None
    try:
        with gzip.open(str(path), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] model load failed ({path.name}): {e}")
        return None


def load_v15_model():
    """Load V15 via predict_core.load_models() — production path."""
    try:
        from predict_core import load_models
        data = load_models()
        if data and data.get('model') is not None:
            return data
    except Exception as e:
        print(f"[WARN] predict_core.load_models failed: {e}")
    # direct fallback
    data = _load_pkl_gz(V15_MODEL_PATH)
    if data and isinstance(data, dict) and 'model' in data:
        print(f"[MODEL] V15 direct load OK ({V15_MODEL_PATH.name})")
        return data
    print("[ERROR] V15 model not found")
    return None


def load_v21_model():
    """Load V21 candidate. Returns None gracefully if file missing."""
    data = _load_pkl_gz(V21_MODEL_PATH)
    if data is None:
        print(f"[V21] model not found at {V21_MODEL_PATH} -- V21 paper skipped")
        return None
    if isinstance(data, dict) and 'model' in data:
        print(f"[MODEL] V21 candidate load OK ({V21_MODEL_PATH.name})")
        return data
    print(f"[WARN] V21 model file format unexpected — skipping")
    return None


# ── race list ─────────────────────────────────────────────────────────────
def get_races_for_date(date_str: str) -> list[dict]:
    """netkeibaから当日レース一覧取得。失敗時は空リスト。"""
    try:
        import requests as _req
        from bs4 import BeautifulSoup
        import re
        session = _req.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
        resp = session.get(url, timeout=15)
        resp.encoding = 'EUC-JP'
        soup = BeautifulSoup(resp.text, 'html.parser')
        races = []
        seen = set()
        course_map = {
            '01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
            '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉',
        }
        for a in soup.find_all('a', href=True):
            m = re.search(r'race_id=(\d{12})', a['href'])
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue
            seen.add(race_id)
            parent = a.find_parent('li') or a.find_parent('div')
            text = parent.get_text(strip=True) if parent else a.get_text(strip=True)
            tm = re.search(r'(\d{1,2}:\d{2})', text)
            start_time = tm.group(1) if tm else ''
            rn = re.search(r'(\d{1,2})R', text)
            race_num = int(rn.group(1)) if rn else 0
            course_code = race_id[4:6]
            course = course_map.get(course_code, '?')
            races.append({
                'race_id': race_id,
                'course': course,
                'race_num': race_num,
                'start_time': start_time,
            })
        return sorted(races, key=lambda r: r.get('start_time', ''))
    except Exception as e:
        print(f"[WARN] get_races_for_date failed: {e}")
        return []


# ── strategy filters ──────────────────────────────────────────────────────
def _passes_strategy7c(race_name: str, course: str, cond_key: str, distance: int) -> tuple[bool, str]:
    """Return (passes, reason). Mirrors race_auto_notify strategy7C + C4."""
    race_name_str = str(race_name)
    is_graded = any(g in race_name_str for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'])
    is_listed = any(s in race_name_str for s in ['L)', '(L)', 'OP)', '(OP)'])
    is_open_tokubetsu = any(s in race_name_str for s in ['杯', '賞', 'ステークス', 'カップ', 'ハンデ'])

    if '特別' in race_name_str and not (is_graded or is_listed or is_open_tokubetsu):
        return False, 'strategy_7_06_tokubetsu'
    if course == '京都' and not (is_graded or is_listed):
        return False, 'strategy_7_kyoto'
    if cond_key == 'E':
        return False, 'strategy_7_cond_E'
    if cond_key == 'B':
        return False, 'strategy_7_cond_B'
    if cond_key == 'X' and not (is_graded or is_listed):
        return False, 'strategy_7_cond_X'
    # C4: Cond-A 1600-1800m
    if cond_key == 'A' and 1600 <= distance <= 1800:
        return False, 'strategy_c4_condA_1600_1800'
    return True, ''


# ── prediction wrapper ─────────────────────────────────────────────────────
def _predict_with_model(race_id: str, model_data: dict, label: str) -> dict | None:
    """Run prediction using predict_core routines. Returns result dict or None."""
    try:
        from predict_core import (
            parse_shutuba, fetch_realtime_odds_full, save_odds_base,
            classify_race_condition, generate_trio_bets, generate_umaren_bets,
            build_features, predict_race, fetch_jra_and_weather,
            get_horse_stats, apply_horse_stats, set_horse_defaults,
        )
    except Exception as e:
        print(f"[ERROR] predict_core import failed: {e}")
        return None

    try:
        race_name, horses, horse_ids, rinfo = parse_shutuba(race_id)
        if not horses:
            print(f"    [{label}] no horse data")
            return None
        if rinfo.get('surface') == '障':
            print(f"    [{label}] obstacle race — skip")
            return None

        distance = rinfo.get('distance', 0)
        if distance <= 1000:
            print(f"    [{label}] distance <=1000m — skip")
            return None

        num_horses = len(horses)

        # odds
        try:
            odds_full = fetch_realtime_odds_full(race_id)
            odds_dict = {u: v['odds'] for u, v in odds_full.items()}
            if odds_full:
                save_odds_base(race_id, odds_full)
            time.sleep(0.5)
        except Exception:
            odds_dict = {}

        # jra + weather
        jra_info, weather_info = {}, {}
        try:
            jra_info, weather_info = fetch_jra_and_weather(rinfo.get('course', ''))
        except Exception:
            pass

        # horse stats
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
                time.sleep(0.2)

        # features
        odds_available = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        df = build_features(horses, rinfo, model_data, race_id=race_id,
                            odds_dict=odds_dict, jra_track_info=jra_info,
                            weather_info=weather_info)
        if df is None or len(df) == 0:
            print(f"    [{label}] feature build failed")
            return None

        # JRDB merge (best effort)
        # ★二重マージ禁止: build_features 内で適用済みのため merge_jrdb_once でガード (6/11 Fable sweep)★
        try:
            from jrdb_features import merge_jrdb_once
            df = merge_jrdb_once(df, race_id)
        except Exception:
            pass

        df = predict_race(df, model_data, odds_available, race_info=rinfo)
        df = df.sort_values('スコア', ascending=False).reset_index(drop=True)

        # condition
        cond_key, cond_profile = classify_race_condition(rinfo, num_horses)
        bet_type = cond_profile.get('bet_type', 'trio')

        # bets
        if bet_type == 'umaren':
            bets = generate_umaren_bets(df)
        else:
            bets = generate_trio_bets(df)

        top5 = [int(df.iloc[i]['馬番']) for i in range(min(5, len(df)))]
        top5_names = [str(df.iloc[i].get('馬名', '?')) for i in range(min(5, len(df)))]
        top1 = top5[0] if top5 else 0
        top1_name = top5_names[0] if top5_names else '?'
        top1_score = float(df.iloc[0].get('スコア', 0)) if len(df) > 0 else 0.0
        top1_pop_rank = int(df.iloc[0].get('pop_rank', 0)) if len(df) > 0 else 0

        passes, skip_reason = _passes_strategy7c(
            race_name, rinfo.get('course', ''), cond_key, distance
        )

        return {
            'race_id': race_id,
            'race_name': race_name,
            'rinfo': rinfo,
            'cond_key': cond_key,
            'bet_type': bet_type,
            'bets': bets,
            'top5': top5,
            'top5_names': top5_names,
            'top1': top1,
            'top1_name': top1_name,
            'top1_score': top1_score,
            'top1_pop_rank': top1_pop_rank,
            'distance': distance,
            'num_horses': num_horses,
            'strategy_pass': passes,
            'strategy_skip_reason': skip_reason,
            'formation': bets,
        }
    except Exception as e:
        print(f"    [{label}] prediction error: {e}")
        return None


# ── Discord message builder ────────────────────────────────────────────────
def _build_v21_discord_message(v15: dict, v21: dict) -> str:
    race_name = v21['race_name'] or '?'
    course = v21['rinfo'].get('course', '?')
    race_num = v21['rinfo'].get('race_num', '?')
    cond_key = v21['cond_key']
    bet_type = v21['bet_type']
    bets = v21['bets']
    distance = v21.get('distance', 0)
    surface = v21['rinfo'].get('surface', '?')
    condition = v21['rinfo'].get('condition', '?')

    v15_top1 = v15['top1'] if v15 else '?'
    v15_top1_name = v15['top1_name'] if v15 else '?'
    v21_top1 = v21['top1']
    v21_top1_name = v21['top1_name']
    same_top1 = (v15_top1 == v21_top1) if v15 else None

    # Formation lines
    if bet_type == 'umaren' and bets:
        bet_lines = "馬連フォーメーション 2点 (V21 paper)\n"
        for b in bets:
            bet_lines += f"  {' - '.join(str(x) for x in b)}\n"
    elif bets:
        top6 = v21['top5'][:5]  # up to 5
        n1 = v21['top1']
        second = v21['top5'][1:3] if len(v21['top5']) >= 3 else v21['top5'][1:]
        third = v21['top5'][1:] if len(v21['top5']) >= 2 else []
        second_str = ', '.join(str(x) for x in second) if second else '?'
        third_str = ', '.join(str(x) for x in third) if third else '?'
        bet_lines = (
            f"三連複フォーメーション {len(bets)}点 (V21 paper)\n"
            f"1列目: {n1}\n"
            f"2列目: {second_str}\n"
            f"3列目: {third_str}\n"
        )
    else:
        bet_lines = "(買い目なし)\n"

    if same_top1 is None:
        diff_line = "V15予測なし"
    elif same_top1:
        diff_line = f"軸馬: V15={v15_top1}({v15_top1_name}) V21={v21_top1}({v21_top1_name}) → 一致"
    else:
        diff_line = (
            f"軸馬: V15={v15_top1}({v15_top1_name}) V21={v21_top1}({v21_top1_name}) "
            f"→ 相違 ← 注目"
        )

    msg = (
        f"【V21 paper】 投票しないでください\n"
        f"📝 {course}{race_num}R {race_name}\n"
        f"   {surface}{distance}m {condition} 条件{cond_key} "
        f"({v21['num_horses']}頭)\n"
        f"V21候補モデル (TYB全部入り) の予測 / 実 cash 投票は V15 のみ\n\n"
        f"{bet_lines}\n"
        f"V15との差異:\n"
        f"  {diff_line}\n\n"
        f"⚠ これはpaper予測です。実際の投票は V15 買い目のみ使用してください。"
    )
    return msg


# ── paper log ─────────────────────────────────────────────────────────────
def record_v21_paper(race_id: str, v15_pred: dict | None, v21_pred: dict, date_str: str):
    log_dir = V21_PAPER_LOG_DIR / date_str
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        'race_id': race_id,
        'timestamp': datetime.now().isoformat(),
        'v15_top5': v15_pred['top5'] if v15_pred else None,
        'v15_top5_names': v15_pred['top5_names'] if v15_pred else None,
        'v21_top5': v21_pred['top5'],
        'v21_top5_names': v21_pred['top5_names'],
        'v15_formation': v15_pred['formation'] if v15_pred else None,
        'v21_formation': v21_pred['formation'],
        'v15_strategy_pass': v15_pred['strategy_pass'] if v15_pred else None,
        'v21_strategy_pass': v21_pred['strategy_pass'],
        'v15_strategy_skip_reason': v15_pred['strategy_skip_reason'] if v15_pred else None,
        'v21_strategy_skip_reason': v21_pred['strategy_skip_reason'],
        'top1_match': (v15_pred['top1'] == v21_pred['top1']) if v15_pred else None,
        'cond_key': v21_pred['cond_key'],
        'bet_type': v21_pred['bet_type'],
        'distance': v21_pred.get('distance'),
        'num_horses': v21_pred.get('num_horses'),
        'race_name': str(v21_pred.get('race_name', '')),
        'course': v21_pred['rinfo'].get('course', ''),
    }
    out_file = log_dir / f'{race_id}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    print(f"    [V21 log] {out_file}")


# ── per-race entry ────────────────────────────────────────────────────────
def process_race(
    race_id: str,
    v15_model_data: dict | None,
    v21_model_data: dict | None,
    date_str: str,
    dry_run: bool = False,
) -> dict:
    """Process one race. Returns summary dict."""
    print(f"\n  >>> [{date_str}] {race_id}")
    summary = {
        'race_id': race_id,
        'v15_pass': None,
        'v21_pass': None,
        'top1_match': None,
        'v21_sent': False,
    }

    # ── V15 predict ──────────────────────────────────────────────────────
    v15_pred = None
    if v15_model_data is not None:
        v15_pred = _predict_with_model(race_id, v15_model_data, 'V15')
        if v15_pred:
            summary['v15_pass'] = v15_pred['strategy_pass']
            print(f"    [V15] top1={v15_pred['top1']}({v15_pred['top1_name']}) "
                  f"cond={v15_pred['cond_key']} pass={v15_pred['strategy_pass']}")
        else:
            print(f"    [V15] prediction failed")

    # ── V21 predict ──────────────────────────────────────────────────────
    if v21_model_data is None:
        print(f"    [V21] model not loaded — skip")
        return summary

    v21_pred = _predict_with_model(race_id, v21_model_data, 'V21')
    if v21_pred is None:
        print(f"    [V21] prediction failed")
        return summary

    summary['v21_pass'] = v21_pred['strategy_pass']
    summary['top1_match'] = (v15_pred['top1'] == v21_pred['top1']) if v15_pred else None

    print(f"    [V21] top1={v21_pred['top1']}({v21_pred['top1_name']}) "
          f"cond={v21_pred['cond_key']} pass={v21_pred['strategy_pass']}")

    if v15_pred and summary['top1_match'] is False:
        print(f"    [V21] ← 注目: 軸馬相違 "
              f"V15={v15_pred['top1']}({v15_pred['top1_name']}) "
              f"V21={v21_pred['top1']}({v21_pred['top1_name']})")

    # ── Discord (V21 only, when passes strategy filter) ───────────────────
    if v21_pred['strategy_pass']:
        msg = _build_v21_discord_message(v15_pred, v21_pred)
        webhook = _get_webhook_v21()
        sent = _send_discord(webhook, msg, dry_run=dry_run)
        summary['v21_sent'] = sent
        if sent:
            print(f"    [V21] Discord sent (V21 paper channel)")
    else:
        print(f"    [V21] strategy filter → skip Discord "
              f"(reason: {v21_pred['strategy_skip_reason']})")

    # ── log ──────────────────────────────────────────────────────────────
    try:
        record_v21_paper(race_id, v15_pred, v21_pred, date_str)
    except Exception as e:
        print(f"    [V21 log] write failed: {e}")

    return summary


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='V15/V21 並行予測 (paper trading)')
    parser.add_argument('--date', type=str, default='',
                        help='対象日 YYYYMMDD (例: 20260524)')
    parser.add_argument('--race', type=str, default='',
                        help='単一 race_id 12桁 (例: 202605240511)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Discord を実際には送信しない (テスト用)')
    args = parser.parse_args()

    if not args.date and not args.race:
        parser.print_help()
        sys.exit(1)

    # ── load models ───────────────────────────────────────────────────────
    print("[INIT] Loading V15 model...")
    v15_model_data = load_v15_model()
    if v15_model_data is None:
        print("[ERROR] V15 model load failed. Aborting.")
        sys.exit(1)

    print("[INIT] Loading V21 candidate model...")
    v21_model_data = load_v21_model()
    # V21 failure is graceful — continue with V15 only for logging

    # ── resolve race list ─────────────────────────────────────────────────
    if args.race:
        date_str = args.race[:8]
        races = [{'race_id': args.race, 'course': '?', 'race_num': 0, 'start_time': ''}]
    else:
        date_str = args.date
        print(f"[INFO] Fetching race list for {date_str}...")
        races = get_races_for_date(date_str)
        if not races:
            print(f"[WARN] No races found for {date_str}")
            sys.exit(0)
        print(f"[INFO] {len(races)} races found")

    # ── process ───────────────────────────────────────────────────────────
    summaries: list[dict] = []
    for race_info in races:
        race_id = race_info['race_id']
        try:
            s = process_race(race_id, v15_model_data, v21_model_data, date_str, dry_run=args.dry_run)
            summaries.append(s)
        except Exception as e:
            print(f"  [ERROR] {race_id}: {e}")
        time.sleep(1)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[SUMMARY] {date_str}  ({len(summaries)} races processed)")
    v15_pass = sum(1 for s in summaries if s['v15_pass'])
    v21_pass = sum(1 for s in summaries if s['v21_pass'])
    v21_sent = sum(1 for s in summaries if s['v21_sent'])
    both_pass = [s for s in summaries if s['v15_pass'] and s['v21_pass']]
    match_count = sum(1 for s in both_pass if s['top1_match'] is True)
    diff_count = sum(1 for s in both_pass if s['top1_match'] is False)
    total_both = len(both_pass)
    agree_rate = (match_count / total_both * 100) if total_both > 0 else float('nan')

    print(f"  V15 strategy pass : {v15_pass}")
    print(f"  V21 strategy pass : {v21_pass}")
    print(f"  V21 Discord sent  : {v21_sent}")
    print(f"  Both pass (N={total_both})")
    print(f"    top-1 一致: {match_count} / 相違: {diff_count}")
    print(f"    top-1 agreement rate: {agree_rate:.1f}%")
    print("=" * 60)

    if v21_model_data is None:
        print("[NOTE] V21 model was not loaded. "
              "Place models/v21_candidate.pkl.gz and re-run.")


if __name__ == '__main__':
    main()
