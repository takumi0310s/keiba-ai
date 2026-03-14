"""
実運用テスト用 予測ログシステム
netkeibaのURLを入力すると予測を実行し、結果をCSVに保存する。

Usage:
    python predict_and_log.py "https://race.netkeiba.com/race/shutuba.html?race_id=202406050811"
    python predict_and_log.py 202406050811
    python predict_and_log.py 202406050811 --nar
"""
import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import os
import sys
from datetime import datetime
from itertools import combinations

# === 定数 ===
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
LOG_PATH = "data/predictions_log.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COURSE_MAP = {
    '札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,'中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9,
    '大井':10,'川崎':11,'船橋':12,'浦和':13,'園田':14,'姫路':15,'門別':16,
    '盛岡':17,'水沢':18,'金沢':19,'笠松':20,'名古屋':21,'高知':22,'佐賀':23,
    '帯広':24,'旭川':25,
}
SURFACE_MAP = {'芝':0,'ダ':1,'障':2}
COND_MAP = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}
SEX_MAP = {'牡':0,'牝':1,'セ':2,'騸':2}

CONDITION_PROFILES = {
    'A': {'bet_type':'trio','label':'条件A','desc':'8-14頭/1600m+/良~稍','investment':700,'roi':420.7,'hit_rate':45.1,'recommended':True},
    'B': {'bet_type':'trio','label':'条件B','desc':'8-14頭/1600m+/重~不良','investment':700,'roi':473.8,'hit_rate':45.4,'recommended':True},
    'C': {'bet_type':'trio','label':'条件C','desc':'15頭+/1600m+/良~稍','investment':700,'roi':498.6,'hit_rate':33.4,'recommended':True},
    'D': {'bet_type':'trio','label':'条件D','desc':'1200-1400m','investment':700,'roi':247.0,'hit_rate':28.2,'recommended':True},
    'E': {'bet_type':'umaren','label':'条件E','desc':'7頭以下','investment':200,'roi':118.0,'hit_rate':53.4,'recommended':True},
    'X': {'bet_type':'trio','label':'条件X','desc':'15頭+/重~不良','investment':700,'roi':598.2,'hit_rate':36.1,'recommended':True},
}
NAR_CONDITION_PROFILES = {
    'A': {'bet_type':'trio','label':'NAR条件A','desc':'8-14頭/1600m+/良~稍','investment':700,'roi':366.0,'hit_rate':65.2,'recommended':True},
    'B': {'bet_type':'trio','label':'NAR条件B','desc':'8-14頭/1600m+/重~不良','investment':700,'roi':431.9,'hit_rate':49.4,'recommended':True},
    'C': {'bet_type':'wide','label':'NAR条件C','desc':'1600m+/15頭+','investment':700,'roi':0,'hit_rate':0,'recommended':False},
    'D': {'bet_type':'wide','label':'NAR条件D','desc':'短距離/1-4頭','investment':700,'roi':0,'hit_rate':93.9,'recommended':True},
    'E': {'bet_type':'umaren','label':'NAR条件E','desc':'1600m+/7頭以下','investment':700,'roi':349.8,'hit_rate':60.0,'recommended':True},
    'F': {'bet_type':'wide','label':'NAR条件F','desc':'短距離/5-7頭','investment':700,'roi':0,'hit_rate':83.0,'recommended':True},
    'G': {'bet_type':'wide','label':'NAR条件G','desc':'短距離/8頭+','investment':700,'roi':0,'hit_rate':0,'recommended':False},
    'X': {'bet_type':'trio','label':'NAR条件X','desc':'その他','investment':700,'roi':0,'hit_rate':0,'recommended':False},
}


def load_models():
    """モデルをロード"""
    models = {'central': None, 'nar': None, 'default': None}
    for key, fname in [('central', 'keiba_model_v9_central.pkl'), ('nar', 'keiba_model_v9_nar.pkl')]:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                models[key] = pickle.load(f)
    # fallback
    for fname in ['keiba_model_v8.pkl']:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                models['default'] = pickle.load(f)
            break
    return models


def load_jockey_wr():
    fpath = os.path.join(BASE_DIR, "jockey_wr.json")
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


MODERN_JOCKEY_WR = {
    'ルメール':0.220,'C.ルメール':0.220,'川田将雅':0.210,'川田':0.210,'武豊':0.171,
    '戸崎圭太':0.140,'横山武史':0.130,'松山弘平':0.120,'池添謙一':0.110,
    '岩田望来':0.100,'岩田康誠':0.090,'吉田隼人':0.090,'三浦皇成':0.080,
    '横山和生':0.100,'横山典弘':0.121,'坂井瑠星':0.100,'鮫島克駿':0.080,
    '西村淳也':0.090,'佐々木大':0.080,'M.デムーロ':0.140,'R.ムーア':0.250,
    '森泰斗':0.180,'矢野貴之':0.170,'御神本訓史':0.160,'笹川翼':0.150,
    '張田昂':0.130,'吉原寛人':0.140,'赤岡修次':0.150,'永森大智':0.130,
    '山口勲':0.140,
}


def find_jockey_wr(name, jockey_wr_data):
    if name in MODERN_JOCKEY_WR:
        return MODERN_JOCKEY_WR[name]
    if name in jockey_wr_data:
        return jockey_wr_data[name]
    # 部分一致
    for k, v in MODERN_JOCKEY_WR.items():
        if k in name or name in k:
            return v
    return 0.05


def classify_race_condition(race_info, num_horses, is_nar=False):
    dist = race_info.get('distance', 0)
    cond = str(race_info.get('condition', '良'))
    heavy_track = any(c in cond for c in ['重', '不'])
    good_track = not heavy_track

    if is_nar:
        if dist >= 1600:
            if num_horses <= 7:
                cond_key = 'E'
            elif 8 <= num_horses <= 14 and good_track:
                cond_key = 'A'
            elif 8 <= num_horses <= 14 and heavy_track:
                cond_key = 'B'
            elif num_horses >= 15:
                cond_key = 'C'
            else:
                cond_key = 'X'
        else:
            if num_horses <= 4:
                cond_key = 'D'
            elif num_horses <= 7:
                cond_key = 'F'
            else:
                cond_key = 'G'
        return cond_key, NAR_CONDITION_PROFILES.get(cond_key, NAR_CONDITION_PROFILES['X'])

    if num_horses <= 7:
        cond_key = 'E'
    elif dist <= 1400:
        cond_key = 'D'
    elif 8 <= num_horses <= 14 and dist >= 1600 and good_track:
        cond_key = 'A'
    elif 8 <= num_horses <= 14 and dist >= 1600 and heavy_track:
        cond_key = 'B'
    elif num_horses >= 15 and dist >= 1600 and good_track:
        cond_key = 'C'
    else:
        cond_key = 'X'

    profile = dict(CONDITION_PROFILES[cond_key])
    if cond_key == 'D' and dist <= 1000:
        profile['recommended'] = False
        profile['desc'] = '1000m以下（非推奨：ROI 85%）'
    return cond_key, profile


def parse_shutuba(race_id, is_nar=False):
    """出馬表を解析"""
    if is_nar:
        url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    else:
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")

    race_name = "レース"
    tag = soup.find("div", class_="RaceName")
    if tag and tag.get_text(strip=True):
        race_name = tag.get_text(strip=True)
        race_name = re.sub(r'\s*\(G[I123]+\)\s*', '', race_name).strip()

    d01 = soup.find("div", class_="RaceData01")
    d01t = d01.get_text(strip=True) if d01 else soup.get_text()
    dm = re.search(r'(\d{3,4})m', d01t)
    distance = int(dm.group(1)) if dm else 0
    surface = '芝' if '芝' in d01t else 'ダ'
    cm = re.search(r'馬場:(\S+)', d01t)
    if not cm:
        cm = re.search(r'(良|稍重|稍|重|不良)', d01t)
    condition = cm.group(1) if cm else '良'

    d02 = soup.find("div", class_="RaceData02")
    d02t = d02.get_text(strip=True) if d02 else d01t
    all_text = d01t + " " + d02t

    course_name = ""
    for cn in COURSE_MAP:
        if cn in all_text:
            course_name = cn
            break
    if not course_name and is_nar:
        title_tag = soup.find("title")
        if title_tag:
            tt = title_tag.get_text(strip=True)
            for cn in COURSE_MAP:
                if cn in tt:
                    course_name = cn
                    break

    race_info = dict(distance=distance, surface=surface, condition=condition, course=course_name)

    rows = soup.select("tr.HorseList")
    horses, horse_ids = [], []
    jockey_wr_data = load_jockey_wr()

    for row in rows:
        rc = row.get("class", [])
        if "Cancel" in rc:
            continue
        waku, umaban = 0, 0
        wt = row.select_one("td.Waku span")
        if wt:
            w = wt.get_text(strip=True)
            if w.isdigit():
                waku = int(w)
        ut = row.select_one("td.Umaban")
        if ut:
            u = ut.get_text(strip=True)
            if u.isdigit():
                umaban = int(u)
        if umaban == 0:
            for td in row.find_all("td"):
                cls = " ".join(td.get("class", []))
                if "Num" in cls or "Umaban" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
        if umaban == 0:
            umaban = len(horses) + 1

        nt = row.select_one("span.HorseName a")
        if not nt:
            continue
        horse_name = nt.get_text(strip=True)
        href = nt.get("href", "")
        hm = re.search(r'/horse/(\d+)', href)
        horse_id = hm.group(1) if hm else None

        it = row.select_one("td.Barei") or row.select_one("span.Barei")
        sa = it.get_text(strip=True) if it else ""
        if not sa:
            for td in row.find_all("td"):
                t = td.get_text(strip=True)
                if re.match(r'^[牡牝セ騸]\d+$', t):
                    sa = t
                    break
        sex = sa[0] if sa else '牡'
        age = int(sa[1:]) if sa and sa[1:].isdigit() else 3

        kinryo = 55.0
        for td in row.find_all("td"):
            try:
                v = float(td.get_text(strip=True))
                if 48.0 <= v <= 62.0:
                    kinryo = v
                    break
            except Exception:
                continue

        jt = row.select_one("td.Jockey a") or row.select_one("a[href*='jockey']")
        jockey_name = jt.get_text(strip=True) if jt else ""

        horse_weight, weight_diff = 480, 0
        for td in row.find_all("td"):
            bm = re.search(r'(\d{3,})\(([\+\-]?\d+)\)', td.get_text(strip=True))
            if bm:
                w = int(bm.group(1))
                if 350 <= w <= 600:
                    horse_weight = w
                    weight_diff = int(bm.group(2))
                    break

        horses.append({
            '馬名': horse_name, '馬体重': horse_weight, '場体重増減': weight_diff,
            '斤量': kinryo, '馬齢': age, '距離(m)': distance,
            '競馬場コード_enc': COURSE_MAP.get(course_name, 4),
            '芝ダート_enc': SURFACE_MAP.get(surface, 0),
            '馬場状態_enc': COND_MAP.get(condition, 0),
            '性別_enc': SEX_MAP.get(sex, 0),
            '騎手勝率': find_jockey_wr(jockey_name, jockey_wr_data),
            '騎手名': jockey_name, '枠番': waku, '馬番': umaban,
            '性別': sex,
            '前走着順': 5, '距離適性': 0.5, '馬場適性': 0.5, '人気傾向': 0.5,
            'コース適性': 0.5, '前走間隔': 30, '脚質': 0, '上がり3F': 35.5,
            '複勝率': 0.0, '父': '', '母の父': '', '血統スコア': 0.5,
            '前走オッズ': 15.0, '前走人気': 8, '所属地': '',
            '通過順平均': 8.0, '通過順4': 8,
            'prev2_finish': 5, 'prev3_finish': 5, 'prev4_finish': 5, 'prev5_finish': 5,
            'avg_finish_3r': 5.0, 'avg_finish_5r': 5.0,
            'best_finish_3r': 5, 'best_finish_5r': 5,
            'top3_count_3r': 0, 'top3_count_5r': 0,
            'finish_trend': 0, 'prev2_last3f': 35.5,
        })
        horse_ids.append(horse_id)

    return race_name, horses, horse_ids, race_info


def get_horse_stats_simple(horse_id, distance, surface):
    """馬の成績を簡易取得"""
    if not horse_id:
        return {}
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.select_one("table.db_h_race_results")
        if not table:
            return {}
        rows = table.select("tr")[1:6]  # 直近5走
        finishes = []
        last3fs = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 12:
                continue
            finish_text = cells[11].get_text(strip=True)
            try:
                finishes.append(int(finish_text))
            except ValueError:
                finishes.append(10)
            try:
                last3f = float(cells[17].get_text(strip=True)) if len(cells) > 17 else 35.5
                last3fs.append(last3f)
            except (ValueError, IndexError):
                last3fs.append(35.5)

        stats = {}
        if finishes:
            stats['prev_finish'] = finishes[0]
            stats['avg_finish_3r'] = np.mean(finishes[:3]) if len(finishes) >= 3 else np.mean(finishes)
            stats['avg_finish_5r'] = np.mean(finishes[:5]) if len(finishes) >= 5 else np.mean(finishes)
            stats['best_finish_3r'] = min(finishes[:3]) if len(finishes) >= 3 else min(finishes)
            stats['best_finish_5r'] = min(finishes[:5]) if len(finishes) >= 5 else min(finishes)
            stats['top3_count_3r'] = sum(1 for f in finishes[:3] if f <= 3)
            stats['top3_count_5r'] = sum(1 for f in finishes[:5] if f <= 3)
            if len(finishes) >= 2:
                stats['prev2_finish'] = finishes[1]
            if len(finishes) >= 3:
                stats['prev3_finish'] = finishes[2]
        if last3fs:
            stats['last3f'] = last3fs[0]
            stats['prev2_last3f'] = last3fs[1] if len(last3fs) >= 2 else 35.5

        # 血統
        prof = soup.select_one("table.db_prof_table")
        if prof:
            rows_p = prof.find_all("tr")
            for rp in rows_p:
                th = rp.find("th")
                td = rp.find("td")
                if th and td:
                    label = th.get_text(strip=True)
                    if '父' in label and '母' not in label:
                        a = td.find("a")
                        if a:
                            stats['father'] = a.get_text(strip=True)

        return stats
    except Exception:
        return {}


def fetch_realtime_odds(race_id, is_nar=False):
    """リアルタイムオッズ取得"""
    if is_nar:
        url = f"https://nar.netkeiba.com/odds/odds_get_form.html?type=b1&race_id={race_id}"
    else:
        url = f"https://race.netkeiba.com/odds/odds_get_form.html?type=b1&race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        odds_dict = {}
        rows = soup.select("tr.OddsTableData, tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 3:
                try:
                    num = int(cells[0].get_text(strip=True))
                    odds_text = cells[2].get_text(strip=True) if len(cells) > 2 else cells[1].get_text(strip=True)
                    odds_val = float(odds_text.replace(',', ''))
                    if 1.0 <= odds_val <= 9999.0:
                        odds_dict[num] = odds_val
                except (ValueError, IndexError):
                    continue
        return odds_dict
    except Exception:
        return {}


def generate_trio_bets(df_sorted):
    """三連複7点生成"""
    if len(df_sorted) < 3:
        return []
    nums = [int(df_sorted.iloc[i]['馬番']) for i in range(min(6, len(df_sorted)))]
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    bets = set()
    for s in second:
        for t in third:
            combo = tuple(sorted({n1, s, t}))
            if len(combo) == 3:
                bets.add(combo)
    return [list(b) for b in sorted(bets)]


def generate_wide_bets(df_sorted):
    if len(df_sorted) < 3:
        return []
    nums = [int(df_sorted.iloc[i]['馬番']) for i in range(min(3, len(df_sorted)))]
    bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
    if bets[0] == bets[1]:
        return [bets[0]]
    return bets


def generate_umaren_bets(df_sorted):
    if len(df_sorted) < 3:
        return []
    nums = [int(df_sorted.iloc[i]['馬番']) for i in range(min(3, len(df_sorted)))]
    bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
    if bets[0] == bets[1]:
        return [bets[0]]
    return bets


def predict_race(race_id, is_nar=False, fetch_stats=True):
    """予測を実行してログに保存"""
    print(f"{'NAR' if is_nar else 'JRA'} レース {race_id} を予測中...")

    # モデルロード
    models = load_models()
    if is_nar:
        model_data = models.get('nar') or models.get('default')
    else:
        model_data = models.get('central') or models.get('default')

    if not model_data:
        print("ERROR: モデルファイルが見つかりません")
        return None

    model = model_data.get('model') if isinstance(model_data, dict) else model_data
    features = model_data.get('features') if isinstance(model_data, dict) else None
    version = model_data.get('version', 'v1') if isinstance(model_data, dict) else 'v1'
    sire_map = model_data.get('sire_map', {}) if isinstance(model_data, dict) else {}

    # 出馬表取得
    print("  出馬表を取得中...")
    race_name, horses, horse_ids, race_info = parse_shutuba(race_id, is_nar=is_nar)
    if not horses:
        print("ERROR: 馬データを取得できませんでした")
        return None

    print(f"  {race_info['course']} {race_name} {race_info['distance']}m {race_info['surface']} {race_info['condition']} {len(horses)}頭")

    # オッズ取得
    print("  オッズを取得中...")
    realtime_odds = fetch_realtime_odds(race_id, is_nar=is_nar)
    for horse in horses:
        umaban = horse.get('馬番', 0)
        horse['単勝オッズ'] = realtime_odds.get(umaban, 0.0)

    # 馬成績取得
    if fetch_stats:
        print("  各馬の成績を取得中...")
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                stats = get_horse_stats_simple(hid, race_info['distance'], race_info['surface'])
                if stats:
                    horse['前走着順'] = stats.get('prev_finish', 5)
                    horse['avg_finish_3r'] = stats.get('avg_finish_3r', 5.0)
                    horse['avg_finish_5r'] = stats.get('avg_finish_5r', 5.0)
                    horse['best_finish_3r'] = stats.get('best_finish_3r', 5)
                    horse['best_finish_5r'] = stats.get('best_finish_5r', 5)
                    horse['top3_count_3r'] = stats.get('top3_count_3r', 0)
                    horse['top3_count_5r'] = stats.get('top3_count_5r', 0)
                    horse['prev2_finish'] = stats.get('prev2_finish', 5)
                    horse['prev3_finish'] = stats.get('prev3_finish', 5)
                    horse['上がり3F'] = stats.get('last3f', 35.5)
                    horse['prev2_last3f'] = stats.get('prev2_last3f', 35.5)
                    if stats.get('father'):
                        horse['父'] = stats['father']
                print(f"    [{i+1}/{len(horses)}] {horse['馬名']}", end="\r")
                time.sleep(0.8)
        print()

    # DataFrame構築 & 特徴量
    df = pd.DataFrame(horses)
    num_horses = len(df)
    df['頭数'] = num_horses
    df['斤量平均差'] = df['斤量'] - df['斤量'].mean()
    dist = race_info['distance']
    df['距離カテゴリ'] = 0 if dist <= 1400 else (1 if dist <= 1800 else (2 if dist <= 2200 else 3))
    df['体重カテゴリ'] = df['馬体重'].apply(lambda w: 0 if w <= 440 else (1 if w <= 480 else (2 if w <= 520 else 3)))
    df['体重変動abs'] = df['場体重増減'].abs()
    df['年齢性別'] = df['馬齢'] * 10 + df['性別_enc']
    surf_enc = df['芝ダート_enc'].iloc[0] if len(df) > 0 else 0
    df['距離馬場'] = df['距離カテゴリ'] * 10 + surf_enc
    df['枠位置'] = df['枠番'].apply(lambda w: 0 if w <= 3 else (1 if w <= 6 else 2))
    now = datetime.now()
    df['月'] = now.month
    m = now.month
    df['季節'] = 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3))
    df['枠馬場'] = df['枠位置'] * 10 + df['馬場状態_enc']
    df['馬齢グループ'] = df['馬齢'].clip(2, 7)

    # v5+ features
    if version in ('v5', 'v6', 'v8', 'v9'):
        n_top = model_data.get('n_top_encode', 80) if isinstance(model_data, dict) else 80
        df['sire_enc'] = df['父'].apply(lambda x: sire_map.get(x, n_top) if sire_map else n_top)
        bms_map = model_data.get('bms_map', {}) if isinstance(model_data, dict) else {}
        df['bms_enc'] = df['母の父'].apply(lambda x: bms_map.get(x, n_top) if bms_map else n_top)
        df['location_enc'] = 2 if is_nar else 3
        df['horse_weight'] = df['馬体重']
        df['weight_diff'] = df['場体重増減'].fillna(0)
        df['weight_carry'] = df['斤量']
        df['age'] = df['馬齢']
        df['distance'] = df['距離(m)']
        df['course_enc'] = df['競馬場コード_enc']
        df['turf_dirt_enc'] = df['芝ダート_enc']
        df['condition_enc'] = df['馬場状態_enc']
        df['sex_enc'] = df['性別_enc']
        df['jockey_wr'] = df['騎手勝率']
        df['prev_finish'] = df['前走着順']
        df['bracket'] = df['枠番']
        df['horse_num'] = df['馬番']
        df['num_horses'] = df['頭数']
        df['carry_diff'] = df['斤量平均差']
        df['dist_cat'] = pd.cut(df['距離(m)'], bins=[0,1200,1400,1800,2200,9999], labels=[0,1,2,3,4]).astype(float).fillna(2)
        df['weight_cat'] = pd.cut(df['馬体重'], bins=[0,440,480,520,9999], labels=[0,1,2,3]).astype(float).fillna(1)
        df['age_sex'] = df['馬齢'] * 10 + df['性別_enc']
        df['dist_surface'] = df['dist_cat'] * 10 + df['芝ダート_enc']
        df['bracket_pos'] = pd.cut(df['枠番'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
        df['month_val'] = now.month
        df['season'] = df['季節']
        df['bracket_cond'] = df['bracket_pos'] * 10 + df['馬場状態_enc']
        df['age_group'] = df['馬齢'].clip(2, 7)
        df['prev_pop'] = df['前走人気'].fillna(8)
        df['prev_odds_log'] = np.log1p(df['前走オッズ'].clip(1, 999).fillna(15.0))
        df['prev_last3f'] = df['上がり3F'].fillna(35.5)
        df['prev_pass1'] = df['通過順平均'].fillna(8.0)
        df['prev_pass4'] = df['通過順4'].fillna(8)
        df['prev_margin'] = 0
        df['prev_prize'] = 0
        df['prev2_finish'] = df['prev2_finish'].fillna(5)
        df['prev3_finish'] = df['prev3_finish'].fillna(5)
        df['prev4_finish'] = df['prev4_finish'].fillna(5)
        df['prev5_finish'] = df['prev5_finish'].fillna(5)
        df['prev2_last3f'] = df['prev2_last3f'].fillna(35.5)
        df['avg_finish_3r'] = df['avg_finish_3r'].fillna(5.0)
        df['avg_finish_5r'] = df['avg_finish_5r'].fillna(5.0)
        df['avg_last3f_3r'] = df['上がり3F'].fillna(35.5)
        df['best_finish_3r'] = df['best_finish_3r'].fillna(5)
        df['best_finish_5r'] = df['best_finish_5r'].fillna(5)
        df['top3_count_3r'] = df['top3_count_3r'].fillna(0)
        df['top3_count_5r'] = df['top3_count_5r'].fillna(0)
        df['finish_trend'] = df['finish_trend'].fillna(0)
        df['dist_change'] = 0
        df['dist_change_abs'] = 0
        df['rest_days'] = 30
        df['rest_category'] = 2
        df['same_dist_rate'] = 0.3
        df['same_course_rate'] = 0.3
        df['same_surface_rate'] = 0.3
        df['horse_win_rate'] = 0.1
        df['horse_top3_rate'] = 0.3
        df['horse_race_count'] = 5
        df['jockey_course_wr'] = df['騎手勝率']
        df['jockey_dist_wr'] = df['騎手勝率']
        df['jockey_top3'] = df['騎手勝率'] * 3
        df['trainer_wr'] = 0.08
        df['trainer_top3'] = 0.25
        df['weight_dist'] = df['馬体重'] * df['距離(m)'] / 10000.0
        df['age_season'] = df['馬齢'] * 10 + df['季節']
        df['carry_per_weight'] = df['斤量'] / df['馬体重'].clip(1) * 100
        df['horse_num_ratio'] = df['馬番'] / df['頭数'].clip(1)
        df['weight_diff_abs'] = 0
        df['surface_enc'] = df['芝ダート_enc']
        df['jockey_wr_calc'] = df['騎手勝率']
        df['jockey_course_wr_calc'] = df['騎手勝率']
        df['trainer_top3_calc'] = df['trainer_top3']
        df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
        df['surface_dist_enc'] = df['芝ダート_enc'] * 10 + df['dist_cat']
        df['cond_surface'] = df['馬場状態_enc'] * 10 + df['芝ダート_enc']
        df['course_surface'] = df['競馬場コード_enc'] * 10 + df['芝ダート_enc']
        df['is_nar'] = 1 if is_nar else 0

    # オッズ特徴量
    odds_available = len(realtime_odds) > 0
    if odds_available and '単勝オッズ' in df.columns:
        df['odds_log'] = np.log1p(df['単勝オッズ'].clip(1, 999).replace(0, 15.0))
        has_odds = df['単勝オッズ'] > 0
        if has_odds.any():
            if 'prev_odds_log' in df.columns:
                df.loc[has_odds, 'prev_odds_log'] = df.loc[has_odds, 'odds_log']
    else:
        df['odds_log'] = np.log1p(pd.Series([15.0] * len(df)))

    # 予測
    use_features = features if features else ['馬体重','場体重増減','斤量','馬齢','距離(m)',
                                                '競馬場コード_enc','芝ダート_enc','馬場状態_enc',
                                                '性別_enc','騎手勝率','前走着順']
    for f in use_features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[use_features].values
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        ai_scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, :3].sum(axis=1)
    else:
        ai_scores = model.predict(X)

    df['スコア'] = ai_scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')

    # 条件分類
    cond_key, profile = classify_race_condition(race_info, num_horses, is_nar=is_nar)
    bet_type = profile['bet_type']

    # 買い目生成
    if bet_type == 'trio':
        bets = generate_trio_bets(df)
        bet_label = '三連複7点'
    elif bet_type == 'wide':
        bets = generate_wide_bets(df)
        bet_label = 'ワイド2点'
    elif bet_type == 'umaren':
        bets = generate_umaren_bets(df)
        bet_label = '馬連2点'
    else:
        bets = []
        bet_label = 'なし'

    # 結果表示
    print(f"\n===== 予測結果 =====")
    print(f"レース: {race_info['course']} {race_name}")
    print(f"距離: {race_info['distance']}m {race_info['surface']} {race_info['condition']}")
    print(f"頭数: {num_horses}")
    print(f"条件: {cond_key} ({profile['desc']})")
    print(f"推奨: {bet_label}")
    print(f"BT的中率: {profile['hit_rate']:.1f}%, BT ROI: {profile['roi']:.1f}%")
    print()

    top5 = df.head(5)
    print(f"{'順位':>4} {'馬番':>4} {'馬名':<12} {'スコア':>8} {'オッズ':>7} {'騎手':<8}")
    print("-" * 50)
    for _, h in top5.iterrows():
        odds_str = f"{h.get('単勝オッズ', 0):.1f}" if h.get('単勝オッズ', 0) > 0 else "-"
        print(f"{int(h['AI順位']):>4} {int(h['馬番']):>4} {h['馬名']:<12} {h['スコア']:>8.4f} {odds_str:>7} {h.get('騎手名',''):>8}")

    print(f"\n買い目 ({bet_label}):")
    for b in bets:
        print(f"  {', '.join(str(n) for n in b)}")
    print(f"投資額: {profile['investment']}円")

    # ログ保存
    os.makedirs("data", exist_ok=True)
    log_entry = {
        'predicted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'race_id': race_id,
        'race_name': race_name,
        'course': race_info['course'],
        'distance': race_info['distance'],
        'surface': race_info['surface'],
        'condition': race_info['condition'],
        'num_horses': num_horses,
        'is_nar': 1 if is_nar else 0,
        'cond_key': cond_key,
        'bet_type': bet_type,
        'bet_label': bet_label,
        'bets': json.dumps(bets),
        'investment': profile['investment'],
        'top1_num': int(df.iloc[0]['馬番']),
        'top1_name': df.iloc[0]['馬名'],
        'top1_score': round(float(df.iloc[0]['スコア']), 4),
        'top2_num': int(df.iloc[1]['馬番']) if len(df) >= 2 else 0,
        'top2_name': df.iloc[1]['馬名'] if len(df) >= 2 else '',
        'top2_score': round(float(df.iloc[1]['スコア']), 4) if len(df) >= 2 else 0,
        'top3_num': int(df.iloc[2]['馬番']) if len(df) >= 3 else 0,
        'top3_name': df.iloc[2]['馬名'] if len(df) >= 3 else '',
        'top3_score': round(float(df.iloc[2]['スコア']), 4) if len(df) >= 3 else 0,
        'result_status': 'pending',
        'actual_top3': '',
        'hit': '',
        'payout': 0,
        'roi': 0,
    }

    log_df = pd.DataFrame([log_entry])
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        # 同じrace_idの古い予測を除去
        existing = existing[existing['race_id'].astype(str) != str(race_id)]
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✓ 予測ログを {LOG_PATH} に保存しました")
    return log_entry


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="競馬AI予測ログシステム")
    parser.add_argument("input", help="netkeibaのURL or race_id")
    parser.add_argument("--nar", action="store_true", help="地方競馬")
    parser.add_argument("--no-stats", action="store_true", help="馬成績取得をスキップ")
    args = parser.parse_args()

    # URLからrace_idを抽出
    inp = args.input
    is_nar = args.nar or "nar" in inp
    rid_match = re.search(r'race_id=(\d+)', inp)
    if rid_match:
        race_id = rid_match.group(1)
    elif re.match(r'^\d{10,12}$', inp):
        race_id = inp
    else:
        rid_match = re.search(r'/race/(\d{10,12})', inp)
        if rid_match:
            race_id = rid_match.group(1)
        else:
            print(f"ERROR: race_idを取得できません: {inp}")
            sys.exit(1)

    predict_race(race_id, is_nar=is_nar, fetch_stats=not args.no_stats)
