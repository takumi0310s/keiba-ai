"""
毎朝の自動予測スクリプト
当日のJRA全レースを取得し、AI予測→買い目生成→CSV保存する。

Usage:
    python tools/daily_predict.py                  # 今日の予測
    python tools/daily_predict.py --date 20260315  # 日付指定
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
import argparse
from datetime import datetime, timedelta
from itertools import combinations

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# === 定数 ===
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
INVESTMENT_PER_RACE = 700

COURSE_MAP = {
    '札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,'中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9,
}
SURFACE_MAP = {'芝':0,'ダ':1,'障':2}
COND_MAP = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}
SEX_MAP = {'牡':0,'牝':1,'セ':2,'騸':2}

CONDITION_PROFILES = {
    'A': {'bet_type':'trio','label':'条件A','desc':'8-14頭/1600m+/良~稍','investment':700,'roi':205.3,'hit_rate':44.5,'recommended':True},
    'B': {'bet_type':'trio','label':'条件B','desc':'8-14頭/1600m+/重~不良','investment':700,'roi':236.9,'hit_rate':45.2,'recommended':True},
    'C': {'bet_type':'trio','label':'条件C','desc':'15頭+/1600m+/良~稍','investment':700,'roi':285.6,'hit_rate':33.7,'recommended':True},
    'D': {'bet_type':'trio','label':'条件D','desc':'1400m以下','investment':700,'roi':136.0,'hit_rate':27.0,'recommended':True},
    'E': {'bet_type':'umaren','label':'条件E','desc':'7頭以下','investment':200,'roi':118.0,'hit_rate':53.4,'recommended':True},
    'X': {'bet_type':'trio','label':'条件X','desc':'15頭+/重~不良','investment':700,'roi':330.5,'hit_rate':35.5,'recommended':True},
}

MODERN_JOCKEY_WR = {
    'ルメール':0.220,'C.ルメール':0.220,'川田将雅':0.210,'川田':0.210,'武豊':0.171,
    '戸崎圭太':0.140,'横山武史':0.130,'松山弘平':0.120,'池添謙一':0.110,
    '岩田望来':0.100,'岩田康誠':0.090,'吉田隼人':0.090,'三浦皇成':0.080,
    '横山和生':0.100,'横山典弘':0.121,'坂井瑠星':0.100,'鮫島克駿':0.080,
    '西村淳也':0.090,'佐々木大':0.080,'M.デムーロ':0.140,'R.ムーア':0.250,
    '角田大和':0.070,'団野大成':0.070,'藤岡佑介':0.080,'幸英明':0.060,
    '和田竜二':0.070,'浜中俊':0.090,'菅原明良':0.080,'田辺裕信':0.090,
    '石橋脩':0.080,'北村友一':0.080,'丹内祐次':0.060,'津村明秀':0.070,
    '永野猛蔵':0.060,'荻野極':0.050,'松岡正海':0.060,
}

SIRE_APT = {
    'ディープインパクト':{'turf':1.0,'dirt':0.3,'sprint':0.5,'mile':0.9,'mid':1.0,'long':0.8},
    'キングカメハメハ':{'turf':0.8,'dirt':0.7,'sprint':0.6,'mile':0.8,'mid':0.9,'long':0.7},
    'ロードカナロア':{'turf':0.9,'dirt':0.5,'sprint':1.0,'mile':0.8,'mid':0.5,'long':0.2},
    'ドゥラメンテ':{'turf':0.9,'dirt':0.5,'sprint':0.4,'mile':0.8,'mid':1.0,'long':0.8},
    'エピファネイア':{'turf':0.9,'dirt':0.4,'sprint':0.3,'mile':0.7,'mid':1.0,'long':0.9},
    'ハーツクライ':{'turf':0.9,'dirt':0.3,'sprint':0.2,'mile':0.6,'mid':0.9,'long':1.0},
    'キタサンブラック':{'turf':0.9,'dirt':0.4,'sprint':0.3,'mile':0.7,'mid':0.9,'long':1.0},
    'モーリス':{'turf':0.8,'dirt':0.5,'sprint':0.5,'mile':0.9,'mid':0.8,'long':0.5},
    'ヘニーヒューズ':{'turf':0.2,'dirt':1.0,'sprint':1.0,'mile':0.7,'mid':0.4,'long':0.1},
    'ホッコータルマエ':{'turf':0.2,'dirt':1.0,'sprint':0.7,'mile':0.9,'mid':0.8,'long':0.4},
    'コントレイル':{'turf':0.9,'dirt':0.3,'sprint':0.3,'mile':0.7,'mid':1.0,'long':0.9},
    'ドレフォン':{'turf':0.5,'dirt':0.8,'sprint':0.8,'mile':0.8,'mid':0.6,'long':0.3},
    'スワーヴリチャード':{'turf':0.8,'dirt':0.5,'sprint':0.3,'mile':0.7,'mid':0.9,'long':0.9},
}


# ===== ユーティリティ =====

def load_jockey_wr():
    fpath = os.path.join(BASE_DIR, "jockey_wr.json")
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_jockey_wr = load_jockey_wr()


def find_jockey_wr(name):
    if name in MODERN_JOCKEY_WR:
        return MODERN_JOCKEY_WR[name]
    if name in _jockey_wr:
        return _jockey_wr[name]
    for k, v in MODERN_JOCKEY_WR.items():
        if k in name or name in k:
            return v
    for k, v in _jockey_wr.items():
        if k.startswith(name) or name.startswith(k):
            return v
    return 0.05


def calc_sire_score(father, surface, distance):
    apt = SIRE_APT.get(father)
    if not apt:
        return 0.5
    ss = apt.get('turf', 0.5) if surface == '芝' else apt.get('dirt', 0.5)
    if distance <= 1400:
        ds = apt.get('sprint', 0.5)
    elif distance <= 1800:
        ds = apt.get('mile', 0.5)
    elif distance <= 2200:
        ds = apt.get('mid', 0.5)
    else:
        ds = apt.get('long', 0.5)
    return ss * 0.5 + ds * 0.5


def classify_race_condition(race_info, num_horses):
    dist = race_info.get('distance', 0)
    cond = str(race_info.get('condition', '良'))
    heavy_track = any(c in cond for c in ['重', '不'])
    good_track = not heavy_track

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
    return cond_key, CONDITION_PROFILES[cond_key]


def generate_trio_bets(df_sorted):
    """TOP1軸 - TOP2,3 - TOP2~6 の三連複7点"""
    if len(df_sorted) < 3:
        return []
    top6 = df_sorted.head(min(6, len(df_sorted)))
    nums = [int(top6.iloc[i]['馬番']) for i in range(len(top6))]
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
    """TOP1軸 - TOP2,TOP3 のワイド1軸2流し(2点)"""
    if len(df_sorted) < 3:
        return []
    nums = [int(df_sorted.iloc[i]['馬番']) for i in range(min(3, len(df_sorted)))]
    bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
    if bets[0] == bets[1]:
        return [bets[0]]
    return bets


def generate_umaren_bets(df_sorted):
    """TOP1軸 - TOP2,TOP3 の馬連1軸2流し(2点)"""
    if len(df_sorted) < 3:
        return []
    nums = [int(df_sorted.iloc[i]['馬番']) for i in range(min(3, len(df_sorted)))]
    bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
    if bets[0] == bets[1]:
        return [bets[0]]
    return bets


# ===== モデルロード =====

def load_models():
    """Pattern B優先、Pattern Aフォールバック"""
    result = {'model': None, 'features': None, 'sire_map': {}, 'bms_map': {},
              'version': 'v9', 'n_top_encode': 80, 'is_live': False}
    # Pattern B
    fpath_b = os.path.join(BASE_DIR, 'keiba_model_v9_central_live.pkl')
    if os.path.exists(fpath_b):
        try:
            with open(fpath_b, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data:
                result['model'] = data['model']
                result['features'] = data.get('features')
                result['sire_map'] = data.get('sire_map', {})
                result['bms_map'] = data.get('bms_map', {})
                result['version'] = data.get('version', 'v9')
                result['n_top_encode'] = data.get('n_top_encode', 80)
                result['is_live'] = True
                print(f"[MODEL] Pattern B (当日情報込み) ロード完了")
                return result
        except Exception as e:
            print(f"[WARN] Pattern Bロード失敗: {e}")

    # Pattern A
    fpath_a = os.path.join(BASE_DIR, 'keiba_model_v9_central.pkl')
    if os.path.exists(fpath_a):
        try:
            with open(fpath_a, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data:
                result['model'] = data['model']
                result['features'] = data.get('features')
                result['sire_map'] = data.get('sire_map', {})
                result['bms_map'] = data.get('bms_map', {})
                result['version'] = data.get('version', 'v9')
                result['n_top_encode'] = data.get('n_top_encode', 80)
                result['is_live'] = False
                print(f"[MODEL] Pattern A (リークフリー) ロード完了")
                return result
        except Exception as e:
            print(f"[WARN] Pattern Aロード失敗: {e}")

    # V8 fallback
    fpath_v8 = os.path.join(BASE_DIR, 'keiba_model_v8.pkl')
    if os.path.exists(fpath_v8):
        try:
            with open(fpath_v8, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data:
                result['model'] = data['model']
                result['features'] = data.get('features')
                result['sire_map'] = data.get('sire_map', {})
                result['bms_map'] = data.get('bms_map', {})
                result['version'] = data.get('version', 'v8')
                result['is_live'] = False
                print(f"[MODEL] V8フォールバックロード完了")
                return result
        except Exception as e:
            print(f"[ERROR] V8ロード失敗: {e}")

    return result


# ===== レース一覧取得 =====

def fetch_race_list(date_str):
    """netkeibaからその日のレース一覧を取得
    date_str: YYYYMMDD形式

    Returns: list of dict {'race_id': str, 'course': str, 'race_num': int}
    """
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[ERROR] レース一覧取得失敗: {e}")
        return []

    races = []
    # dl.RaceList_DataList ごとにコースが分かれている
    for dl in soup.find_all("dl", class_="RaceList_DataList"):
        # コース名はdtに含まれる
        dt = dl.find("dt")
        course_name = ""
        if dt:
            dt_text = dt.get_text(strip=True)
            for cn in COURSE_MAP:
                if cn in dt_text:
                    course_name = cn
                    break

        # 各レースのリンク
        for a in dl.find_all("a", href=True):
            href = a.get("href", "")
            m = re.search(r'race_id=(\d{12})', href)
            if not m:
                m = re.search(r'/race/(\d{12})/', href)
            if m:
                race_id = m.group(1)
                # レース番号を推定
                race_num = 0
                nm = re.search(r'(\d{1,2})R', a.get_text(strip=True))
                if nm:
                    race_num = int(nm.group(1))
                else:
                    # race_idの末尾2桁がレース番号
                    try:
                        race_num = int(race_id[-2:])
                    except ValueError:
                        pass

                # 重複チェック
                if not any(r['race_id'] == race_id for r in races):
                    races.append({
                        'race_id': race_id,
                        'course': course_name,
                        'race_num': race_num,
                    })

    # レース番号でソート
    races.sort(key=lambda x: (x['course'], x['race_num']))
    return races


# ===== 出馬表スクレイピング =====

def parse_shutuba(race_id):
    """出馬表を解析。app.pyのparse_shutubaを簡略化した版"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")

    race_name = "レース"
    tag = soup.find("div", class_="RaceName")
    if tag and tag.get_text(strip=True):
        race_name = tag.get_text(strip=True)
        race_name = re.sub(r'\s*\(G[I123]+\)\s*', '', race_name).strip()
        race_name = re.sub(r'\s*G[I123]+\s*$', '', race_name).strip()

    race_num = ""
    num_tag = soup.find("span", class_="RaceNum")
    if num_tag:
        race_num = num_tag.get_text(strip=True)
    if not race_num:
        nm = re.search(r'(\d{1,2})R', race_name)
        if nm:
            race_num = nm.group(0)

    d01 = soup.find("div", class_="RaceData01")
    d01t = d01.get_text(strip=True) if d01 else soup.get_text()
    dm = re.search(r'(\d{3,4})m', d01t)
    distance = int(dm.group(1)) if dm else 0
    surface = '芝' if '芝' in d01t else ('ダ' if 'ダ' in d01t else 'ダ')
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

    race_info = dict(distance=distance, surface=surface, condition=condition,
                     course=course_name, race_num=race_num)

    rows = soup.select("tr.HorseList")
    horses, horse_ids = [], []

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
            except:
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

        tt = row.select_one("td.Trainer a") or row.select_one("a[href*='trainer']")
        trainer = tt.get_text(strip=True) if tt else ""

        horses.append({
            '馬名': horse_name, '馬体重': horse_weight, '場体重増減': weight_diff,
            '斤量': kinryo, '馬齢': age, '距離(m)': distance,
            '競馬場コード_enc': COURSE_MAP.get(course_name, 4),
            '芝ダート_enc': SURFACE_MAP.get(surface, 0),
            '馬場状態_enc': COND_MAP.get(condition, 0),
            '性別_enc': SEX_MAP.get(sex, 0),
            '騎手勝率': find_jockey_wr(jockey_name),
            '騎手名': jockey_name, '枠番': waku, '馬番': umaban,
            '調教師': trainer, '性別': sex,
        })
        horse_ids.append(horse_id)

    return race_name, horses, horse_ids, race_info


# ===== 馬成績取得 =====

def get_horse_stats(horse_id, target_distance, target_surface, target_course=""):
    """netkeibaから馬の過去成績を取得（app.pyのget_horse_stats簡略版）"""
    result = {
        'last_finish': 5, 'dist_apt': 0.5, 'surf_apt': 0.5, 'pop_score': 0.5,
        'course_apt': 0.5, 'interval_days': 30, 'running_style': 0,
        'avg_agari': 35.5, 'father': '', 'mother_father': '', 'fukusho_rate': 0.0,
        'avg_pass_pos': 8.0, 'last_pass4': 8, 'last_odds': 15.0, 'last_pop': 8,
        'trainer_loc': '',
        'prev2_finish': 5, 'prev3_finish': 5, 'prev4_finish': 5, 'prev5_finish': 5,
        'avg_finish_3r': 5.0, 'avg_finish_5r': 5.0,
        'best_finish_3r': 5, 'best_finish_5r': 5,
        'top3_count_3r': 0, 'top3_count_5r': 0,
        'finish_trend': 0, 'prev2_last3f': 35.5,
    }
    try:
        url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 血統情報
        prof = soup.find("table", class_="db_prof_table")
        if prof:
            for td in prof.find_all("td"):
                a = td.find("a", href=re.compile(r"/horse/sire/"))
                if a:
                    result['father'] = a.get_text(strip=True)
                    break
            for tr in prof.find_all("tr"):
                th = tr.find("th")
                if th and '調教師' in th.get_text():
                    td_text = tr.find("td").get_text(strip=True) if tr.find("td") else ""
                    if '美浦' in td_text or '(美)' in td_text:
                        result['trainer_loc'] = '美浦'
                    elif '栗東' in td_text or '(栗)' in td_text:
                        result['trainer_loc'] = '栗東'

        bt = soup.find("table", summary=re.compile(".*血統.*"))
        if bt:
            all_links = bt.find_all("a", href=re.compile(r"/horse/"))
            if not result['father'] and all_links:
                result['father'] = all_links[0].get_text(strip=True)
            if len(all_links) >= 5:
                result['mother_father'] = all_links[4].get_text(strip=True)
            elif len(all_links) >= 3:
                result['mother_father'] = all_links[2].get_text(strip=True)

        table = soup.find("table", class_="db_h_race_results")
        if not table:
            return result
        tbody = table.find("tbody")
        if not tbody:
            return result
        rows = tbody.find_all("tr")

        dist_results, surf_results, course_results = [], [], []
        pop_list, pass_list, agari_list, finish_list = [], [], [], []
        race_dates = []
        odds_list = []
        pass4_list = []
        today_date = datetime.now().date()

        for ri, row in enumerate(rows):
            tds = row.find_all("td")
            if len(tds) < 15:
                continue
            ft = tds[11].get_text(strip=True)
            if not ft.isdigit():
                continue
            finish = int(ft)

            row_date = None
            for td_idx in range(min(5, len(tds))):
                td_text = tds[td_idx].get_text(strip=True)
                dm_match = re.search(r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', td_text)
                if dm_match:
                    try:
                        y, m_val, d_val = int(dm_match.group(1)), int(dm_match.group(2)), int(dm_match.group(3))
                        if 2000 <= y <= 2030 and 1 <= m_val <= 12 and 1 <= d_val <= 31:
                            row_date = datetime(y, m_val, d_val)
                    except:
                        pass
                if row_date:
                    break

            if row_date and row_date.date() >= today_date:
                continue

            finish_list.append(finish)
            if row_date:
                race_dates.append(row_date)

            if len(finish_list) == 1:
                result['last_finish'] = finish

            pt = tds[10].get_text(strip=True)
            if pt.isdigit():
                pop_list.append(int(pt))

            if len(tds) > 9:
                odds_text = tds[9].get_text(strip=True)
                try:
                    odds_val = float(odds_text)
                    if 1.0 <= odds_val <= 999.9:
                        odds_list.append(odds_val)
                except:
                    pass

            dc = tds[14].get_text(strip=True)
            ddm = re.match(r'([芝ダ障])(\d+)', dc)
            if ddm:
                sc, dv = ddm.group(1), int(ddm.group(2))
                if target_distance > 0 and abs(dv - target_distance) <= 200:
                    dist_results.append(finish)
                sn = '芝' if sc == '芝' else 'ダ'
                if sn == target_surface:
                    surf_results.append(finish)

            if target_course and len(tds) > 1:
                if target_course in tds[1].get_text(strip=True):
                    course_results.append(finish)

            # 通過順・上がり
            for tdi, td in enumerate(tds):
                txt = td.get_text(strip=True)
                cleaned = txt.replace(' ', '').replace('-', '-').replace('\uff0d', '-')
                if re.match(r'^\d{1,2}-\d{1,2}(-\d{1,2})*$', cleaned):
                    pn = re.findall(r'\d+', cleaned)
                    if pn and len(pn) >= 2:
                        pass_list.append(int(pn[0]))
                        pass4_list.append(int(pn[-1]))
                    break

            for tdi, td in enumerate(tds):
                if tdi >= 10:
                    cleaned_a = td.get_text(strip=True).strip()
                    if re.match(r'^\d{2}\.\d{1,2}$', cleaned_a):
                        try:
                            av = float(cleaned_a)
                            if 30.0 < av < 45.0:
                                agari_list.append(av)
                        except:
                            pass
                        break

            if ri >= 9:
                break

        def to_score(lst):
            if not lst:
                return 0.5
            return max(0.0, min(1.0, 1.0 - (sum(lst) / len(lst) - 1) / 17.0))

        result['dist_apt'] = to_score(dist_results)
        result['surf_apt'] = to_score(surf_results)
        result['course_apt'] = to_score(course_results)
        if pop_list:
            result['pop_score'] = max(0.0, min(1.0, 1.0 - (sum(pop_list) / len(pop_list) - 1) / 17.0))
            result['last_pop'] = pop_list[0]
        if odds_list:
            result['last_odds'] = odds_list[0]
        if race_dates:
            diff = (datetime.now() - race_dates[0]).days
            result['interval_days'] = max(diff, 1)
        if pass_list:
            ap = sum(pass_list) / len(pass_list)
            result['avg_pass_pos'] = ap
            if ap <= 2.0:
                result['running_style'] = 1
            elif ap <= 5.0:
                result['running_style'] = 2
            elif ap <= 10.0:
                result['running_style'] = 3
            else:
                result['running_style'] = 4
        if pass4_list:
            result['last_pass4'] = pass4_list[0]
        if agari_list:
            result['avg_agari'] = sum(agari_list) / len(agari_list)
        if finish_list:
            result['fukusho_rate'] = sum(1 for f in finish_list if f <= 3) / len(finish_list)
            fl = finish_list
            if len(fl) >= 2:
                result['prev2_finish'] = fl[1]
            if len(fl) >= 3:
                result['prev3_finish'] = fl[2]
            if len(fl) >= 4:
                result['prev4_finish'] = fl[3]
            if len(fl) >= 5:
                result['prev5_finish'] = fl[4]
            fl3 = fl[:min(3, len(fl))]
            result['avg_finish_3r'] = sum(fl3) / len(fl3)
            result['best_finish_3r'] = min(fl3)
            result['top3_count_3r'] = sum(1 for f in fl3 if f <= 3)
            fl5 = fl[:min(5, len(fl))]
            result['avg_finish_5r'] = sum(fl5) / len(fl5)
            result['best_finish_5r'] = min(fl5)
            result['top3_count_5r'] = sum(1 for f in fl5 if f <= 3)
            if len(fl) >= 3:
                result['finish_trend'] = fl[2] - fl[0]
            elif len(fl) >= 2:
                result['finish_trend'] = fl[1] - fl[0]
        if len(agari_list) >= 2:
            result['prev2_last3f'] = agari_list[1]

    except Exception as e:
        pass  # デフォルト値を返す

    return result


# ===== 特徴量構築 =====

def build_features(horses, race_info, model_data, odds_dict=None,
                   jra_track_info=None, weather_info=None):
    """馬リストから特徴量DataFrameを構築（app.pyの特徴量構築を再現）"""
    df = pd.DataFrame(horses)
    num_horses = len(df)
    version = model_data.get('version', 'v9')
    is_live = model_data.get('is_live', False)
    n_top = model_data.get('n_top_encode', 80)
    use_sire_map = model_data.get('sire_map', {})
    use_bms_map = model_data.get('bms_map', {})

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
    df['季節'] = 0 if m in [3, 4, 5] else (1 if m in [6, 7, 8] else (2 if m in [9, 10, 11] else 3))
    df['枠馬場'] = df['枠位置'] * 10 + df['馬場状態_enc']
    df['馬齢グループ'] = df['馬齢'].clip(2, 7)

    # v5+ 英語名特徴量
    if version in ('v5', 'v6', 'v8', 'v9'):
        df['sire_enc'] = df['父'].apply(lambda x: use_sire_map.get(x, n_top) if use_sire_map else n_top)
        df['bms_enc'] = df['母の父'].apply(lambda x: use_bms_map.get(x, n_top) if use_bms_map else n_top)

        def enc_loc(loc):
            s = str(loc)
            if '美浦' in s or '美' == s:
                return 0
            if '栗東' in s or '栗' == s:
                return 1
            return 3
        df['location_enc'] = df['所属地'].apply(enc_loc)

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
        df['dist_cat'] = pd.cut(df['距離(m)'], bins=[0, 1200, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
        df['weight_cat'] = pd.cut(df['馬体重'], bins=[0, 440, 480, 520, 9999], labels=[0, 1, 2, 3]).astype(float).fillna(1)
        df['age_sex'] = df['馬齢'] * 10 + df['性別_enc']
        df['dist_surface'] = df['dist_cat'] * 10 + df['芝ダート_enc']
        df['bracket_pos'] = pd.cut(df['枠番'], bins=[0, 3, 6, 8], labels=[0, 1, 2]).astype(float).fillna(1)
        month_now = now.month
        df['month_val'] = month_now
        df['season'] = 0 if month_now in [3, 4, 5] else (1 if month_now in [6, 7, 8] else (2 if month_now in [9, 10, 11] else 3))
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
        df['rest_days'] = df.get('前走間隔', pd.Series([30] * len(df))).fillna(30)
        if '前走間隔' in df.columns:
            df['rest_days'] = df['前走間隔']
        df['rest_category'] = pd.cut(df['rest_days'], bins=[-1, 6, 14, 35, 63, 180, 9999], labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(2)

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
        df['age_season'] = df['馬齢'] * 10 + df['season']
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
        df['is_nar'] = 0

    # リアルタイムオッズ
    odds_available = odds_dict and len(odds_dict) > 0
    if odds_available and '単勝オッズ' in df.columns:
        df['odds_log'] = np.log1p(df['単勝オッズ'].clip(1, 999).replace(0, 15.0))
    else:
        df['odds_log'] = np.log1p(pd.Series([15.0] * len(df)))

    # Pattern B 当日特徴量
    if is_live:
        df['weight_change'] = df['場体重増減'].fillna(0)
        df['weight_change_abs'] = df['weight_change'].abs()
        weather_str = str(race_info.get('weather', '晴'))
        weather_map = {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3}
        df['weather_enc'] = weather_map.get(weather_str, 0)

        if odds_available and '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any():
            df['pop_rank'] = df['単勝オッズ'].replace(0, 9999).rank(method='min')
        else:
            df['pop_rank'] = 8

        if jra_track_info:
            df['cushion_value'] = jra_track_info.get('cushion_value') or 0
            surface_type = race_info.get('surface', '芝')
            try:
                from scrape_jra_track import get_moisture_rate
                mr = get_moisture_rate(jra_track_info, surface_type)
                df['moisture_rate'] = mr if mr is not None else 0
            except Exception:
                df['moisture_rate'] = 0
        else:
            df['cushion_value'] = 0
            df['moisture_rate'] = 0

        if weather_info:
            df['temperature'] = weather_info.get('temperature', 0)
            df['humidity'] = weather_info.get('humidity', 0)
            df['wind_speed'] = weather_info.get('wind_speed', 0)
            df['precipitation'] = weather_info.get('precipitation', 0)
        else:
            df['temperature'] = 0
            df['humidity'] = 0
            df['wind_speed'] = 0
            df['precipitation'] = 0

    # 必要な特徴量の確保
    use_features = model_data.get('features')
    if use_features:
        for f in use_features:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    return df


# ===== 予測実行 =====

def predict_race(df, model_data, odds_available=False):
    """予測を実行してスコア・順位を付与"""
    use_features = model_data.get('features')
    if not use_features:
        print("[ERROR] モデルに特徴量リストがありません")
        return df
    use_model = model_data['model']
    X = df[use_features].values

    if hasattr(use_model, 'predict_proba'):
        proba = use_model.predict_proba(X)
        ai_scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, :3].sum(axis=1)
    else:
        ai_scores = use_model.predict(X)

    # 補助スコア
    pop_scores = df['人気傾向'].values if '人気傾向' in df.columns else np.full(len(df), 0.5)
    apt_scores = np.full(len(df), 0.5)
    if '距離適性' in df.columns and '馬場適性' in df.columns:
        apt_scores = (df['距離適性'].values + df['馬場適性'].values) / 2.0

    if odds_available and '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any():
        odds_vals = df['単勝オッズ'].replace(0, 15.0)
        odds_scores = np.clip(1.0 - np.log1p(odds_vals) / np.log1p(100.0), 0.0, 1.0)
        final_scores = (
            ai_scores * 0.65 + odds_scores * 0.08 + apt_scores * 0.06
            + pop_scores * 0.03 + np.full(len(df), 0.03) * 0.18  # pace etc. simplified
        )
    else:
        final_scores = (
            ai_scores * 0.70 + pop_scores * 0.06 + apt_scores * 0.06
            + np.full(len(df), 0.5) * 0.18  # simplified
        )

    df['スコア'] = final_scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    return df


# ===== オッズ取得 =====

def fetch_realtime_odds(race_id):
    """単勝リアルタイムオッズを取得"""
    odds_dict = {}
    try:
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        try:
            data = resp.json()
            if isinstance(data, dict) and 'data' in data:
                odds_data = data['data'].get('odds', data['data'])
                if isinstance(odds_data, dict):
                    tansho = odds_data.get('1', odds_data)
                    if isinstance(tansho, dict):
                        for umaban_str, vals in tansho.items():
                            if not umaban_str.isdigit():
                                continue
                            umaban = int(umaban_str)
                            if isinstance(vals, list) and len(vals) >= 1:
                                try:
                                    odds_val = float(str(vals[0]).replace(',', ''))
                                    if 1.0 <= odds_val <= 9999.9:
                                        odds_dict[umaban] = odds_val
                                except:
                                    pass
                            elif isinstance(vals, (int, float, str)):
                                try:
                                    odds_val = float(str(vals).replace(',', ''))
                                    if 1.0 <= odds_val <= 9999.9:
                                        odds_dict[umaban] = odds_val
                                except:
                                    pass
        except:
            pass
    except Exception:
        pass
    return odds_dict


# ===== JRA馬場・天候取得 =====

def fetch_jra_and_weather(course_name):
    """JRA馬場情報と天候データを取得"""
    jra_info = {}
    weather_info = {}
    try:
        from scrape_jra_track import fetch_jra_track_info
        jra_info = fetch_jra_track_info(course_name)
    except Exception:
        pass
    try:
        from scrape_weather import get_weather_features
        weather_info = get_weather_features(course_name)
    except Exception:
        pass
    return jra_info, weather_info


# ===== メイン処理 =====

def run_daily_predict(date_str):
    """指定日のJRA全レースを予測"""
    print(f"=" * 60)
    print(f"KEIBA AI 日次予測 - {date_str}")
    print(f"=" * 60)

    # モデルロード
    model_data = load_models()
    if model_data['model'] is None:
        print("[ERROR] モデルが見つかりません。終了します。")
        return

    # レース一覧取得
    print(f"\n[STEP 1] レース一覧取得中...")
    races = fetch_race_list(date_str)
    if not races:
        print(f"[INFO] {date_str} のレースが見つかりません（非開催日の可能性）")
        return

    print(f"  -> {len(races)}レース検出")
    for r in races:
        print(f"     {r['course']} {r['race_num']}R (race_id={r['race_id']})")

    # 各レースを予測
    results = []
    jra_weather_cache = {}  # コースごとにキャッシュ

    for idx, race in enumerate(races):
        race_id = race['race_id']
        print(f"\n[STEP 2-{idx+1}/{len(races)}] {race['course']} {race['race_num']}R (ID={race_id})")

        try:
            # 出馬表取得
            race_name, horses, horse_ids, race_info = parse_shutuba(race_id)
            if not horses:
                print(f"  [WARN] 馬データなし、スキップ")
                continue
            num_horses = len(horses)
            print(f"  レース名: {race_name} / {race_info['surface']}{race_info['distance']}m / {race_info['condition']} / {num_horses}頭")
            time.sleep(0.5)

            # オッズ取得
            odds_dict = fetch_realtime_odds(race_id)
            odds_available = len(odds_dict) > 0
            if odds_available:
                for horse in horses:
                    umaban = horse.get('馬番', 0)
                    if umaban in odds_dict:
                        horse['単勝オッズ'] = odds_dict[umaban]
                    else:
                        horse['単勝オッズ'] = 0.0
                print(f"  オッズ取得: {len(odds_dict)}頭分")
            else:
                for horse in horses:
                    horse['単勝オッズ'] = 0.0
                print(f"  オッズ: 未取得（レース前オッズ未発表の可能性）")
            time.sleep(0.3)

            # JRA馬場・天候（コースごとにキャッシュ）
            course_name = race_info.get('course', '')
            jra_info, weather_info = {}, {}
            if model_data['is_live'] and course_name:
                if course_name not in jra_weather_cache:
                    jra_info, weather_info = fetch_jra_and_weather(course_name)
                    jra_weather_cache[course_name] = (jra_info, weather_info)
                else:
                    jra_info, weather_info = jra_weather_cache[course_name]

            # 各馬の過去成績取得
            print(f"  各馬成績取得中...", end="", flush=True)
            for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
                if hid:
                    try:
                        stats = get_horse_stats(hid, race_info['distance'], race_info['surface'], course_name)
                        horse['前走着順'] = stats.get('last_finish', 5)
                        horse['距離適性'] = stats.get('dist_apt', 0.5)
                        horse['馬場適性'] = stats.get('surf_apt', 0.5)
                        horse['人気傾向'] = stats.get('pop_score', 0.5)
                        horse['コース適性'] = stats.get('course_apt', 0.5)
                        horse['前走間隔'] = stats.get('interval_days', 30)
                        horse['脚質'] = stats.get('running_style', 0)
                        horse['上がり3F'] = stats.get('avg_agari', 35.5)
                        horse['複勝率'] = stats.get('fukusho_rate', 0.0)
                        horse['父'] = stats.get('father', '')
                        horse['母の父'] = stats.get('mother_father', '')
                        horse['血統スコア'] = calc_sire_score(stats.get('father', ''), race_info['surface'], race_info['distance'])
                        horse['通過順平均'] = stats.get('avg_pass_pos', 8.0)
                        horse['通過順4'] = stats.get('last_pass4', 8)
                        horse['前走オッズ'] = stats.get('last_odds', 15.0)
                        horse['前走人気'] = stats.get('last_pop', 8)
                        horse['所属地'] = stats.get('trainer_loc', '')
                        horse['prev2_finish'] = stats.get('prev2_finish', 5)
                        horse['prev3_finish'] = stats.get('prev3_finish', 5)
                        horse['prev4_finish'] = stats.get('prev4_finish', 5)
                        horse['prev5_finish'] = stats.get('prev5_finish', 5)
                        horse['avg_finish_3r'] = stats.get('avg_finish_3r', 5.0)
                        horse['avg_finish_5r'] = stats.get('avg_finish_5r', 5.0)
                        horse['best_finish_3r'] = stats.get('best_finish_3r', 5)
                        horse['best_finish_5r'] = stats.get('best_finish_5r', 5)
                        horse['top3_count_3r'] = stats.get('top3_count_3r', 0)
                        horse['top3_count_5r'] = stats.get('top3_count_5r', 0)
                        horse['finish_trend'] = stats.get('finish_trend', 0)
                        horse['prev2_last3f'] = stats.get('prev2_last3f', 35.5)
                    except Exception:
                        _set_defaults(horse)
                else:
                    _set_defaults(horse)
                if i < num_horses - 1:
                    time.sleep(0.5)
            print(f" 完了")

            # 特徴量構築 & 予測
            df = build_features(horses, race_info, model_data, odds_dict, jra_info, weather_info)
            df = predict_race(df, model_data, odds_available)

            # 条件分類
            cond_key, cond_profile = classify_race_condition(race_info, num_horses)

            # 買い目生成（条件に応じた買い目種別）
            sorted_df = df.sort_values('スコア', ascending=False).reset_index(drop=True)
            bet_type = cond_profile['bet_type']
            if bet_type == 'umaren':
                bets = generate_umaren_bets(sorted_df)
                bet_label = '馬連'
            elif bet_type == 'wide':
                bets = generate_wide_bets(sorted_df)
                bet_label = 'ワイド'
            else:
                bets = generate_trio_bets(sorted_df)
                bet_label = '三連複'

            top1 = sorted_df.iloc[0] if len(sorted_df) > 0 else None
            top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else None
            top3 = sorted_df.iloc[2] if len(sorted_df) > 2 else None

            bets_str = "; ".join(["-".join(str(n) for n in b) for b in bets])
            race_num_int = 0
            nm = re.search(r'(\d+)', str(race_info.get('race_num', '')))
            if nm:
                race_num_int = int(nm.group(1))

            row = {
                'race_id': race_id,
                'course': course_name,
                'race_num': race_num_int,
                'race_name': race_name,
                'condition': cond_key,
                'num_horses': num_horses,
                'distance': race_info['distance'],
                'surface': race_info['surface'],
                'track_condition': race_info['condition'],
                'top1_num': int(top1['馬番']) if top1 is not None else 0,
                'top1_name': top1['馬名'] if top1 is not None else '',
                'top1_score': float(top1['スコア']) if top1 is not None else 0,
                'top2_num': int(top2['馬番']) if top2 is not None else 0,
                'top2_name': top2['馬名'] if top2 is not None else '',
                'top3_num': int(top3['馬番']) if top3 is not None else 0,
                'top3_name': top3['馬名'] if top3 is not None else '',
                'trio_bets': bets_str,
                'bet_type': cond_profile['bet_type'],
                'investment': cond_profile['investment'],
            }
            results.append(row)

            # コンソール出力
            print(f"  条件: {cond_key} ({cond_profile['desc']})")
            print(f"  TOP3: {top1['馬名']}({int(top1['馬番'])}) / {top2['馬名']}({int(top2['馬番'])}) / {top3['馬名']}({int(top3['馬番'])})")
            print(f"  {bet_label} {len(bets)}点: {bets_str}")

        except Exception as e:
            print(f"  [ERROR] 予測失敗: {e}")
            import traceback
            traceback.print_exc()
            continue

    # CSV保存
    if results:
        out_dir = os.path.join(BASE_DIR, "data", "daily_predictions")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.csv")
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n{'=' * 60}")
        print(f"予測完了: {len(results)}レース")
        print(f"保存先: {out_path}")
        print(f"総投資額: {sum(r['investment'] for r in results):,}円")
        print(f"{'=' * 60}")
    else:
        print(f"\n[INFO] 予測結果なし")


def _set_defaults(horse):
    """馬成績のデフォルト値を設定"""
    horse.update({
        '前走着順': 5, '距離適性': 0.5, '馬場適性': 0.5, '人気傾向': 0.5,
        'コース適性': 0.5, '前走間隔': 30, '脚質': 0, '上がり3F': 35.5,
        '複勝率': 0.0, '父': '', '母の父': '', '血統スコア': 0.5,
        '通過順平均': 8.0, '通過順4': 8, '前走オッズ': 15.0, '前走人気': 8, '所属地': '',
        'prev2_finish': 5, 'prev3_finish': 5, 'prev4_finish': 5, 'prev5_finish': 5,
        'avg_finish_3r': 5.0, 'avg_finish_5r': 5.0, 'best_finish_3r': 5, 'best_finish_5r': 5,
        'top3_count_3r': 0, 'top3_count_5r': 0, 'finish_trend': 0, 'prev2_last3f': 35.5,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEIBA AI 日次予測")
    parser.add_argument("--date", type=str, default=None,
                        help="予測日 YYYYMMDD (デフォルト: 今日)")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime("%Y%m%d")

    # 日付バリデーション
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"[ERROR] 日付形式が不正です: {date_str} (YYYYMMDD)")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_predict.py 開始")
    run_daily_predict(date_str)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_predict.py 終了")
