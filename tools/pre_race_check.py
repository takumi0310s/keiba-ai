#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""実戦前チェックリスト自動実行スクリプト

毎週土曜朝、予測開始前に実行して全項目を確認する。
Usage: python tools/pre_race_check.py [--date YYYYMMDD]
"""
import sys
import os
import io
import argparse
import gzip
import pickle
import json
import re
import sqlite3
import time
from datetime import datetime

# Windows環境でのUTF-8出力
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
COURSE_MAP = {'札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,'中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9}
SURFACE_MAP = {'芝':0,'ダ':1,'障':2}
COND_MAP = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}
SEX_MAP = {'牡':0,'牝':1,'セ':2,'騸':2}

# ── 結果集計 ──
results = []

def ok(name, detail=""):
    results.append(('OK', name, detail))
    print(f"  \u2705 {name}: {detail}")

def warn(name, detail=""):
    results.append(('WARN', name, detail))
    print(f"  \u26a0\ufe0f  {name}: {detail}")

def fail(name, detail=""):
    results.append(('FAIL', name, detail))
    print(f"  \u274c {name}: {detail}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. モデルファイル読み込み確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_models():
    print("\n[1] モデルファイル読み込み確認")
    model_data = {}
    for label, candidates in [
        ("Pattern A", ["keiba_model_v9_central.pkl.gz", "keiba_model_v9_central.pkl"]),
        ("Pattern B", ["keiba_model_v9_central_live.pkl.gz", "keiba_model_v9_central_live.pkl"]),
    ]:
        loaded = False
        for fname in candidates:
            fpath = os.path.join(BASE_DIR, fname)
            if not os.path.exists(fpath):
                continue
            try:
                opener = gzip.open if fname.endswith('.gz') else open
                with opener(fpath, 'rb') as f:
                    data = pickle.load(f)
                size_mb = os.path.getsize(fpath) / 1024 / 1024
                version = data.get('version', '?')
                n_feat = len(data.get('features', []))
                auc = data.get('auc', data.get('pattern_a_auc', 0))
                ok(f"{label}", f"{fname} ({size_mb:.1f}MB) ver={version} features={n_feat} AUC={auc:.4f}")
                model_data[label] = data
                loaded = True
                break
            except Exception as e:
                fail(f"{label}", f"{fname} 読み込みエラー: {e}")
                loaded = True
                break
        if not loaded:
            fail(f"{label}", "ファイルが見つかりません")

    # V8 fallback
    v8_path = os.path.join(BASE_DIR, "keiba_model_v8.pkl")
    if os.path.exists(v8_path):
        ok("V8 Fallback", f"keiba_model_v8.pkl ({os.path.getsize(v8_path)/1024/1024:.1f}MB)")
    else:
        warn("V8 Fallback", "keiba_model_v8.pkl が見つかりません")

    return model_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Feature Lookups確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_feature_lookups():
    print("\n[2] Feature Lookups確認")
    candidates = [
        ("data/feature_lookups.pkl", "フル版"),
        ("data/feature_lookups.pkl.gz", "フル版gz"),
        ("data/feature_lookups_lite.pkl.gz", "lite版"),
    ]
    for fname, label in candidates:
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            opener = gzip.open if fname.endswith('.gz') else open
            with opener(fpath, 'rb') as f:
                data = pickle.load(f)
            size_kb = os.path.getsize(fpath) / 1024
            n_keys = len(data) if isinstance(data, dict) else 0
            horse_stats_n = len(data.get('horse_stats', {})) if isinstance(data, dict) else 0
            if n_keys >= 5:
                ok(f"Feature Lookups ({label})", f"{fname} ({size_kb:.0f}KB) keys={n_keys} horse_stats={horse_stats_n}")
            else:
                warn(f"Feature Lookups ({label})", f"キー数が少ない: {n_keys}")
            return data
        except Exception as e:
            fail(f"Feature Lookups ({label})", f"読み込みエラー: {e}")
            return None

    fail("Feature Lookups", "フル版/lite版いずれも見つかりません")
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. netkeibaアクセス確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_netkeiba():
    print("\n[3] netkeibaアクセス確認")
    # 出馬表ページ
    try:
        r = requests.get("https://race.netkeiba.com/top/", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            ok("netkeiba トップ", f"HTTP {r.status_code}")
        else:
            fail("netkeiba トップ", f"HTTP {r.status_code}")
    except Exception as e:
        fail("netkeiba トップ", f"接続エラー: {e}")

    # オッズAPI
    try:
        r = requests.get(
            "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=202606020701&type=1",
            headers=HEADERS, timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            has_odds = 'data' in data and 'odds' in data.get('data', {})
            if has_odds:
                ok("オッズAPI", "JSON正常取得")
            else:
                warn("オッズAPI", f"HTTP 200だがデータ構造が異なる")
        else:
            fail("オッズAPI", f"HTTP {r.status_code}")
    except Exception as e:
        fail("オッズAPI", f"接続エラー: {e}")

    # 追い切りデータ（調教ランク + プレミアム実タイム + 厩舎コメント）テスト
    try:
        sys.path.insert(0, BASE_DIR)
        from scrape_training import check_premium_access, fetch_training_times, fetch_stable_comments
        premium_ok, premium_msg = check_premium_access()
        if premium_ok:
            ok("調教タイム(Premium)", premium_msg)
        else:
            warn("調教タイム(Premium)", premium_msg)

        # Test rank + time fetch
        test_race = "202606020501"
        data = fetch_training_times(test_race)
        rank_count = sum(1 for d in data.values() if d.get('rank'))
        time_count = sum(1 for d in data.values() if d.get('time_4f', 0) > 0)
        if time_count > 0:
            ok("追い切りデータ", f"{time_count}馬の実タイム + {rank_count - time_count}馬のランク取得")
        elif rank_count > 0:
            ok("追い切りデータ", f"{rank_count}馬のランク取得（実タイムはPremium Cookie要）")
        else:
            warn("追い切りデータ", "取得0件（レース未発走の可能性）")

        # Test stable comments
        comments = fetch_stable_comments(test_race)
        if comments:
            ok("厩舎コメント", f"{len(comments)}馬のコメント取得")
        else:
            warn("厩舎コメント", "取得0件")
    except Exception as e:
        warn("追い切りデータ", f"テスト失敗: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. JRA馬場データ確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_jra_track():
    print("\n[4] JRA馬場データ確認")
    try:
        r = requests.get("https://www.jra.go.jp/keiba/baba/", headers=HEADERS, timeout=10)
        if r.status_code != 200:
            fail("JRA馬場ページ", f"HTTP {r.status_code}")
            return

        enc = r.apparent_encoding or 'shift_jis'
        r.encoding = enc
        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text()

        if 'クッション' in text and '含水率' in text:
            ok("JRA馬場ページ", f"HTTP 200, encoding={enc}, テキスト正常")
        else:
            warn("JRA馬場ページ", f"HTTP 200だが日本語テキストが検出できない (encoding={enc})")

        # scrape_jra_track.pyのテスト
        from scrape_jra_track import fetch_jra_track_info
        info = fetch_jra_track_info('東京')
        if isinstance(info, dict) and 'source' in info:
            ok("scrape_jra_track", "関数正常動作（エラーなし）")
        else:
            warn("scrape_jra_track", "予期しない戻り値")
    except Exception as e:
        fail("JRA馬場データ", f"エラー: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 気象庁API確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_weather_api():
    print("\n[5] 気象庁API確認")
    try:
        from scrape_weather import get_weather_features
        weather = get_weather_features('東京')
        if isinstance(weather, dict) and weather.get('temperature') is not None:
            t = weather['temperature']
            h = weather.get('humidity', '?')
            w = weather.get('wind_speed', '?')
            ok("気象庁API", f"temp={t}C, humidity={h}%, wind={w}m/s")
        else:
            warn("気象庁API", f"データ不完全: {weather}")
    except Exception as e:
        fail("気象庁API", f"エラー: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. DB確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_db():
    print("\n[6] DB確認")
    for db_name in ["keiba_predictions.db", "keiba_race_results.db"]:
        db_path = os.path.join(BASE_DIR, db_name)
        if not os.path.exists(db_path):
            warn(db_name, "ファイルが見つかりません")
            continue
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # テーブル一覧
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cur.fetchall()]
            # 各テーブルのレコード数
            counts = {}
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM [{t}]")
                counts[t] = cur.fetchone()[0]
            conn.close()
            size_kb = os.path.getsize(db_path) / 1024
            detail = ", ".join(f"{t}={n}" for t, n in counts.items())
            ok(db_name, f"{size_kb:.0f}KB | {detail}")
        except Exception as e:
            fail(db_name, f"エラー: {e}")

    # jockey_wr.json
    jwr_path = os.path.join(BASE_DIR, "jockey_wr.json")
    if os.path.exists(jwr_path):
        age_days = (time.time() - os.path.getmtime(jwr_path)) / 86400
        with open(jwr_path, 'r', encoding='utf-8') as f:
            jwr = json.load(f)
        if age_days > 30:
            warn("jockey_wr.json", f"{len(jwr)}騎手, {age_days:.0f}日前更新 (30日超)")
        else:
            ok("jockey_wr.json", f"{len(jwr)}騎手, {age_days:.0f}日前更新")
    else:
        warn("jockey_wr.json", "ファイルが見つかりません")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7-8. 特徴量テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_today_races(date_str):
    """今日の開催レース一覧を取得"""
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.encoding = 'euc-jp'
    soup = BeautifulSoup(r.text, 'html.parser')
    race_ids = []
    for a in soup.find_all('a', href=True):
        m = re.search(r'race_id=(\d{12})', a.get('href', ''))
        if m:
            rid = m.group(1)
            if rid not in [r[0] for r in race_ids]:
                venue_code = rid[4:6]
                venue_name = {v: k for k, v in {
                    '札幌':'01','函館':'02','福島':'03','新潟':'04','東京':'05',
                    '中山':'06','中京':'07','京都':'08','阪神':'09','小倉':'10'
                }.items()}.get(venue_code, f'会場{venue_code}')
                race_num = int(rid[10:12])
                race_ids.append((rid, venue_name, race_num))
    return race_ids


def check_features(date_str, model_data):
    print("\n[7-8] 特徴量テスト")

    # 今日のレースを取得
    try:
        races = fetch_today_races(date_str)
    except Exception as e:
        fail("レース一覧取得", f"エラー: {e}")
        return

    if not races:
        warn("レース一覧", f"{date_str}の開催レースが見つかりません")
        return

    # 会場ごとの最初のレースを取得
    venues_seen = set()
    test_races = []
    for rid, venue, rnum in races:
        if venue not in venues_seen:
            venues_seen.add(venue)
            test_races.append((rid, venue, rnum))
    ok("レース一覧", f"{len(races)}レース, {len(test_races)}会場: {', '.join(v for _, v, _ in test_races)}")

    # Pattern Bのfeatureリストを取得
    pb_data = model_data.get("Pattern B")
    if not pb_data:
        warn("特徴量テスト", "Pattern Bモデルがないためスキップ")
        return
    feature_list = pb_data.get('features', [])
    if not feature_list:
        fail("特徴量テスト", "モデルにfeatures定義がありません")
        return

    # 最初の会場のR1でテスト
    test_rid, test_venue, test_rnum = test_races[0]
    print(f"  テスト対象: {test_venue}R{test_rnum} (race_id={test_rid})")

    try:
        # 出馬表取得
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={test_rid}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = 'EUC-JP'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # RaceName
        tag = soup.find(class_='RaceName')
        race_name = tag.get_text(strip=True) if tag and tag.get_text(strip=True) else 'Unknown'

        # Race info
        d01 = soup.find('div', class_='RaceData01')
        d01t = d01.get_text(strip=True) if d01 else ''
        dm = re.search(r'(\d{3,4})m', d01t)
        distance = int(dm.group(1)) if dm else 0
        surface_jp = '芝' if '芝' in d01t else ('ダ' if 'ダ' in d01t else '芝')
        cm = re.search(r'馬場:(\S+)', d01t)
        if not cm:
            cm = re.search(r'(良|稍重|稍|重|不良)', d01t)
        condition = cm.group(1).rstrip('/') if cm else '良'

        # Parse horses
        rows = soup.select('tr.HorseList')
        horses = []
        for row in rows:
            if 'Cancel' in row.get('class', []):
                continue
            umaban = 0
            for td in row.find_all('td'):
                cls = ' '.join(td.get('class', []))
                if 'Umaban' in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
            waku = 0
            for td in row.find_all('td'):
                cls = ' '.join(td.get('class', []))
                if 'Waku' in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 8:
                        waku = int(t)
                        break
            nt = row.select_one('span.HorseName a')
            if not nt:
                continue
            horse_name = nt.get_text(strip=True)

            it = row.select_one('td.Barei') or row.select_one('span.Barei')
            sa = it.get_text(strip=True) if it else ''
            if not sa:
                for td in row.find_all('td'):
                    t = td.get_text(strip=True)
                    if re.match(r'^[牡牝セ騸]\d+$', t):
                        sa = t
                        break
            sex = sa[0] if sa else '牡'
            age = int(sa[1:]) if sa and sa[1:].isdigit() else 3

            kinryo = 55.0
            for td in row.find_all('td'):
                try:
                    v = float(td.get_text(strip=True))
                    if 48.0 <= v <= 62.0:
                        kinryo = v
                        break
                except:
                    continue

            jt = row.select_one('td.Jockey a') or row.select_one('a[href*="jockey"]')
            jockey_name = jt.get_text(strip=True) if jt else ''

            horse_weight, weight_diff = 0, 0
            for td in row.find_all('td'):
                bm = re.search(r'(\d{3,})\(([\+\-]?\d+)\)', td.get_text(strip=True))
                if bm:
                    w = int(bm.group(1))
                    if 350 <= w <= 600:
                        horse_weight = w
                        weight_diff = int(bm.group(2))
                        break
            if horse_weight == 0:
                horse_weight = 480

            # Load jockey WR
            jwr = 0.08
            jwr_path = os.path.join(BASE_DIR, 'jockey_wr.json')
            if os.path.exists(jwr_path):
                with open(jwr_path, 'r', encoding='utf-8') as f:
                    jockey_wr_data = json.load(f)
                jwr = jockey_wr_data.get(jockey_name, 0.08)

            horses.append({
                '馬名': horse_name, '馬体重': horse_weight, '場体重増減': weight_diff,
                '斤量': kinryo, '馬齢': age, '距離(m)': distance,
                '競馬場コード_enc': COURSE_MAP.get(test_venue, 5),
                '芝ダート_enc': SURFACE_MAP.get(surface_jp, 0),
                '馬場状態_enc': COND_MAP.get(condition, 0),
                '性別_enc': SEX_MAP.get(sex, 0),
                '騎手勝率': jwr, '騎手名': jockey_name,
                '枠番': waku, '馬番': umaban, '単勝オッズ': 0.0,
                '前走着順': 5, '通過順平均': 8.0, '通過順4': 8,
                '前走オッズ': 15.0, '前走人気': 8, '所属地': '',
                '上がり3F': 35.5, '父': '', '母の父': '',
                'prev2_finish': 5, 'prev3_finish': 5,
                'avg_finish_3r': 5.0, 'best_finish_3r': 5,
                'top3_count_3r': 0, 'finish_trend': 0, 'prev2_last3f': 35.5,
                '前走間隔': 30,
            })

        if not horses:
            fail("出馬表パース", "馬データを取得できませんでした")
            return

        # Fetch odds
        odds_url = f'https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={test_rid}&type=1'
        r_odds = requests.get(odds_url, headers=HEADERS, timeout=10)
        odds_dict = {}
        try:
            od = r_odds.json()
            tansho = od.get('data', {}).get('odds', {}).get('1', {})
            for ustr, vals in tansho.items():
                if ustr.isdigit() and isinstance(vals, list) and len(vals) >= 1:
                    try:
                        ov = float(str(vals[0]).replace(',', ''))
                        if 1.0 <= ov <= 9999.9:
                            odds_dict[int(ustr)] = ov
                    except:
                        pass
        except:
            pass

        for h in horses:
            if h['馬番'] in odds_dict:
                h['単勝オッズ'] = odds_dict[h['馬番']]

        # Build DataFrame & all 87 features
        df = pd.DataFrame(horses)
        num_horses = len(df)
        now = datetime.now()

        sire_map = pb_data.get('sire_map', {})
        bms_map = pb_data.get('bms_map', {})
        n_top = pb_data.get('n_top_encode', 80)

        # Feature lookups
        _lookups = {}
        for cand in ['data/feature_lookups.pkl', 'data/feature_lookups.pkl.gz', 'data/feature_lookups_lite.pkl.gz']:
            fp = os.path.join(BASE_DIR, cand)
            if os.path.exists(fp):
                opener = gzip.open if cand.endswith('.gz') else open
                with opener(fp, 'rb') as f:
                    _lookups = pickle.load(f)
                break
        _training_mean = _lookups.get('training_mean', 49.0)

        # ── Build all features ──
        df['頭数'] = num_horses
        df['斤量平均差'] = df['斤量'] - df['斤量'].mean()
        df['horse_weight'] = df['馬体重']
        df['weight_carry'] = df['斤量']
        df['age'] = df['馬齢']
        df['distance'] = df['距離(m)']
        df['course_enc'] = df['競馬場コード_enc']
        df['surface_enc'] = df['芝ダート_enc']
        df['condition_enc'] = df['馬場状態_enc']
        df['sex_enc'] = df['性別_enc']
        df['num_horses'] = df['頭数']
        df['horse_num'] = df['馬番']
        df['bracket'] = df['枠番']
        df['jockey_wr_calc'] = df['騎手勝率']
        df['jockey_course_wr_calc'] = df['騎手勝率']
        df['carry_diff'] = df['斤量平均差']
        df['dist_cat'] = pd.cut(df['距離(m)'], bins=[0,1200,1400,1800,2200,9999], labels=[0,1,2,3,4]).astype(float).fillna(2)
        df['weight_cat'] = pd.cut(df['馬体重'], bins=[0,440,480,520,9999], labels=[0,1,2,3]).astype(float).fillna(1)
        df['age_sex'] = df['馬齢'] * 10 + df['性別_enc']
        df['bracket_pos'] = pd.cut(df['枠番'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
        df['season'] = 0 if now.month in [3,4,5] else (1 if now.month in [6,7,8] else (2 if now.month in [9,10,11] else 3))
        df['age_season'] = df['馬齢'] * 10 + df['season']
        df['age_group'] = df['馬齢'].clip(2, 7)
        df['prev_finish'] = df['前走着順']
        df['prev_odds_log'] = np.log1p(df['前走オッズ'].clip(1, 999).fillna(15.0))
        df['prev_last3f'] = df['上がり3F'].fillna(35.5)
        df['prev_pass4'] = df['通過順4'].fillna(8)
        df['prev_prize'] = 0
        df['prev2_finish'] = df['prev2_finish'].fillna(5)
        df['prev3_finish'] = df['prev3_finish'].fillna(5)
        df['avg_finish_3r'] = df['avg_finish_3r'].fillna(5.0)
        df['best_finish_3r'] = df['best_finish_3r'].fillna(5)
        df['finish_trend'] = df['finish_trend'].fillna(0)
        df['top3_count_3r'] = df['top3_count_3r'].fillna(0)
        df['avg_last3f_3r'] = df['上がり3F'].fillna(35.5)
        df['prev2_last3f'] = df['prev2_last3f'].fillna(35.5)
        df['dist_change'] = 0
        df['dist_change_abs'] = 0
        df['rest_days'] = 30
        df['rest_category'] = 2
        df['horse_num_ratio'] = df['馬番'] / df['頭数'].clip(1)
        df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
        df['surface_dist_enc'] = df['芝ダート_enc'] * 10 + df['dist_cat']
        df['cond_surface'] = df['馬場状態_enc'] * 10 + df['芝ダート_enc']
        df['course_surface'] = df['競馬場コード_enc'] * 10 + df['芝ダート_enc']
        df['is_nar'] = 0
        df['sire_enc'] = df['父'].apply(lambda x: sire_map.get(x, n_top) if sire_map else n_top)
        df['bms_enc'] = df['母の父'].apply(lambda x: bms_map.get(x, n_top) if bms_map else n_top)
        df['location_enc'] = 3
        df['odds_log'] = np.log1p(df['単勝オッズ'].replace(0, 15.0).clip(1, 999))
        df['weight_change'] = df['場体重増減'].fillna(0)
        df['weight_change_abs'] = df['weight_change'].abs()
        weather_map_enc = {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3}
        wm = re.search(r'天候:(\S+)', d01t)
        weather_str = wm.group(1).rstrip('/') if wm else '晴'
        df['weather_enc'] = weather_map_enc.get(weather_str, 0)
        df['pop_rank'] = df['単勝オッズ'].replace(0, 9999).rank(method='min')

        # JRA track
        try:
            from scrape_jra_track import fetch_jra_track_info, get_moisture_rate
            jra_info = fetch_jra_track_info(test_venue)
            cv = jra_info.get('cushion_value')
            mr = get_moisture_rate(jra_info, surface_jp)
            df['cushion_value'] = cv if cv is not None else 0
            df['moisture_rate'] = mr if mr is not None else 0
        except:
            df['cushion_value'] = 0
            df['moisture_rate'] = 0

        # Weather
        try:
            from scrape_weather import get_weather_features
            wi = get_weather_features(test_venue)
            df['temperature'] = wi.get('temperature', 0)
            df['humidity'] = wi.get('humidity', 0)
            df['wind_speed'] = wi.get('wind_speed', 0)
            df['precipitation'] = wi.get('precipitation', 0)
        except:
            df['temperature'] = df['humidity'] = df['wind_speed'] = df['precipitation'] = 0

        # Lookup-based features (defaults)
        df['training_time_filled'] = _training_mean
        df['has_training'] = 0
        df['training_per_dist'] = _training_mean / max(distance / 200, 1)
        df['trainer_top3_calc'] = 0.25
        df['jockey_surface_wr'] = df['騎手勝率']
        df['horse_career_races'] = 0
        df['horse_career_wr'] = 0.1
        df['horse_career_top3r'] = 0.25
        df['sire_surface_wr'] = 0.1
        df['sire_dist_wr'] = 0.1
        df['bms_surface_wr'] = 0.1
        df['wood_best_4f_filled'] = 52.0
        df['has_wood_training'] = 0
        df['wood_count_2w'] = 0
        df['sakaro_best_4f_filled'] = 53.0
        df['sakaro_best_3f_filled'] = 39.0
        df['has_sakaro_training'] = 0
        df['total_training_count'] = 0
        df['prev_race_first3f'] = 35.0
        df['prev_race_last3f'] = 35.5
        df['prev_race_pace_diff'] = 0.5
        df['prev_agari_relative'] = 0.0
        df['horse_dist_top3r'] = 0.25
        df['horse_surface_top3r'] = 0.25
        df['frame_course_dist_wr'] = 0.08
        df['running_style'] = 2.0
        df['predicted_pace'] = 1.0
        df['pace_advantage'] = 0.5
        df['style_vs_pace'] = 21.0

        # Ensure all features exist
        missing = []
        for f in feature_list:
            if f not in df.columns:
                df[f] = 0
                missing.append(f)
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

        # Count features
        total = len(feature_list)
        present = total - len(missing)

        if present == total:
            ok(f"特徴量生成", f"{present}/{total} 全特徴量生成成功")
        else:
            fail(f"特徴量生成", f"{present}/{total} (不足: {', '.join(missing)})")

        # Zero feature check
        # Features that are legitimately zero
        # 正当にゼロになりうる特徴量（エンコード値0=有効値、またはデフォルト値として妥当）
        ZERO_OK = {'is_nar', 'dist_change', 'dist_change_abs', 'prev_prize',
                   'precipitation', 'has_training', 'has_wood_training',
                   'has_sakaro_training', 'total_training_count', 'wood_count_2w',
                   'finish_trend', 'top3_count_3r', 'prev_agari_relative',
                   'cushion_value', 'moisture_rate',
                   # エンコード値0が有効値の特徴量
                   'condition_enc',   # 良=0
                   'surface_enc',     # 芝=0
                   'sex_enc',         # 牡=0
                   'season',          # 春=0
                   'weather_enc',     # 晴=0
                   'cond_surface',    # 良×芝=0
                   'dist_cat',        # <=1200m=0
                   'bracket_pos',     # 内枠=0
                   'age_sex',         # age*10+0
                   # テスト時デフォルト（app.pyではlookup/scrapeで補完される）
                   'horse_career_races', 'horse_career_wr', 'horse_career_top3r',
                   'location_enc',    # enc_loc未定=3だが計算結果で0もあり得る
                   }

        zero_features = []
        for f in feature_list:
            vals = df[f].values
            if (vals == 0).all() and f not in ZERO_OK:
                zero_features.append(f)

        if len(zero_features) == 0:
            ok("ゼロ特徴量チェック", "ゼロ特徴量なし（正常ゼロ除く）")
        elif len(zero_features) < 5:
            warn("ゼロ特徴量チェック", f"{len(zero_features)}個がゼロ")
            for f in zero_features:
                print(f"    - {f}")
        else:
            fail("ゼロ特徴量チェック", f"{len(zero_features)}個がゼロ（5個以上: 要調査）")
            for f in zero_features:
                print(f"    - {f}")

        # データ取得状況
        odds_n = sum(1 for h in horses if h['単勝オッズ'] > 0)
        wt_n = sum(1 for h in horses if h['馬体重'] != 480)
        ub_n = sum(1 for h in horses if h['馬番'] > 0)

        detail = f"{race_name} {distance}m{surface_jp} {condition} {num_horses}頭"
        detail += f" | 馬番={ub_n}/{num_horses} 体重={wt_n}/{num_horses} オッズ={odds_n}/{num_horses}"
        ok("レースデータ", detail)

        if odds_n == 0:
            warn("オッズ", "全馬オッズ未取得（レース前は正常）")

    except Exception as e:
        import traceback
        fail("特徴量テスト", f"エラー: {e}")
        traceback.print_exc()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description='実戦前チェックリスト')
    parser.add_argument('--date', type=str, default=None, help='チェック日 (YYYYMMDD)')
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime('%Y%m%d')

    print("=" * 60)
    print(f"  実戦前チェックリスト - {date_str}")
    print(f"  実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model_data = check_models()
    check_feature_lookups()
    check_netkeiba()
    check_jra_track()
    check_weather_api()
    check_db()
    check_features(date_str, model_data)

    # ── 最終判定 ──
    n_ok = sum(1 for s, _, _ in results if s == 'OK')
    n_warn = sum(1 for s, _, _ in results if s == 'WARN')
    n_fail = sum(1 for s, _, _ in results if s == 'FAIL')

    print("\n" + "=" * 60)
    print(f"  最終判定")
    print("=" * 60)
    print(f"  \u2705 OK: {n_ok}  \u26a0\ufe0f  WARN: {n_warn}  \u274c FAIL: {n_fail}")

    if n_fail > 0:
        print(f"\n  \u274c 判定: エラーあり → 修正するまで実戦禁止")
        print(f"  失敗項目:")
        for s, name, detail in results:
            if s == 'FAIL':
                print(f"    - {name}: {detail}")
        return 1
    elif n_warn > 0:
        print(f"\n  \u26a0\ufe0f  判定: 警告あり → 影響確認してから判断")
        print(f"  警告項目:")
        for s, name, detail in results:
            if s == 'WARN':
                print(f"    - {name}: {detail}")
        return 0
    else:
        print(f"\n  \u2705 判定: 全項目OK → 実戦開始")
        return 0


if __name__ == '__main__':
    sys.exit(main())
