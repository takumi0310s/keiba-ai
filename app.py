import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import re
import time

st.set_page_config(page_title="競馬AI予想", page_icon="🏇", layout="wide")

CSS = """
<style>
.top-card-gold {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    border-left: 5px solid #FFD700; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.top-card-silver {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    border-left: 5px solid #C0C0C0; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.top-card-bronze {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    border-left: 5px solid #CD7F32; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.rank-gold { font-size: 2.0em; font-weight: bold; color: #FFD700; margin-right: 10px; }
.rank-silver { font-size: 2.0em; font-weight: bold; color: #C0C0C0; margin-right: 10px; }
.rank-bronze { font-size: 2.0em; font-weight: bold; color: #CD7F32; margin-right: 10px; }
.horse-name-card { font-size: 1.4em; font-weight: bold; color: #ffffff; }
.details-card { color: #aaaaaa; font-size: 0.95em; margin-top: 5px; }
.bar-bg { background: #333; border-radius: 10px; height: 12px; margin-top: 8px; overflow: hidden; }
.bar-gold { height: 12px; border-radius: 10px; background: linear-gradient(90deg, #FFD700, #FFA500); }
.bar-silver { height: 12px; border-radius: 10px; background: linear-gradient(90deg, #C0C0C0, #A0A0A0); }
.bar-bronze { height: 12px; border-radius: 10px; background: linear-gradient(90deg, #CD7F32, #A0522D); }
.race-hdr { background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); border-radius: 12px; padding: 15px 20px; margin: 15px 0; text-align: center; }
.race-hdr h2 { color: #e0e0e0; margin: 0; font-size: 1.6em; }
.race-hdr p { color: #aaa; margin: 5px 0 0 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    import os, gzip
    if os.path.exists("keiba_model.pkl.gz"):
        with gzip.open("keiba_model.pkl.gz", "rb") as f:
            return pickle.load(f)
    with open("keiba_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_jockey_wr():
    with open("jockey_wr.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
jockey_wr = load_jockey_wr()

FEATURES = [
    '馬体重', '場体重増減', '斤量', '馬齢', '距離(m)',
    '競馬場コード_enc', '芝ダート_enc', '馬場状態_enc',
    '性別_enc', '騎手勝率', '前走着順'
]

COURSE_MAP = {
    '札幌': 0, '函館': 1, '福島': 2, '新潟': 3, '東京': 4,
    '中山': 5, '中京': 6, '京都': 7, '阪神': 8, '小倉': 9
}
SURFACE_MAP = {'芝': 0, 'ダ': 1, '障': 2}
COND_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
SEX_MAP = {'牡': 0, '牝': 1, 'セ': 2, '騸': 2}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

MODERN_JOCKEY_WR = {
    'ルメール': 0.220, 'C.ルメール': 0.220,
    '川田将雅': 0.210, '川田': 0.210, '武豊': 0.171,
    '戸崎圭太': 0.140, '戸崎圭': 0.140,
    '横山武史': 0.130, '横山武': 0.130,
    '松山弘平': 0.120, '松山弘': 0.120, '松山': 0.120,
    '池添謙一': 0.110, '池添謙': 0.110,
    '福永祐一': 0.160, '福永祐': 0.160,
    '岩田望来': 0.100, '岩田望': 0.100,
    '岩田康誠': 0.090, '岩田康': 0.090,
    '吉田隼人': 0.090, '吉田隼': 0.090,
    '三浦皇成': 0.080, '三浦皇': 0.080, '三浦': 0.110,
    '田辺裕信': 0.090, '田辺裕': 0.090, '田辺': 0.090,
    '横山和生': 0.100, '横山和': 0.100,
    '横山典弘': 0.121, '横山典': 0.121,
    '大野拓弥': 0.070, '大野拓': 0.070, '大野': 0.070,
    '菅原明良': 0.080, '菅原明': 0.080,
    '北村宏司': 0.070, '北村宏': 0.070,
    '北村友一': 0.080, '北村友': 0.080,
    '丹内祐次': 0.060, '丹内祐': 0.060,
    '柴田大知': 0.060, '柴田大': 0.060,
    '石橋脩': 0.080, '木幡巧也': 0.060, '木幡巧': 0.060,
    'M.デムーロ': 0.140, 'デムーロ': 0.140,
    'R.ムーア': 0.250, 'ムーア': 0.250,
    '坂井瑠星': 0.100, '坂井瑠': 0.100,
    '鮫島克駿': 0.080, '鮫島克': 0.080, '鮫島駿': 0.080,
    '西村淳也': 0.090, '西村淳': 0.090,
    '角田大和': 0.070, '角田大': 0.070,
    '佐々木大': 0.080, '佐々木': 0.080,
    '津村明秀': 0.070, '津村明': 0.070, '津村': 0.070,
    '団野大成': 0.070, '団野大': 0.070,
    '藤岡佑介': 0.080, '藤岡佑': 0.080,
    '幸英明': 0.060, '幸英': 0.060,
    '和田竜二': 0.070, '和田竜': 0.070, '浜中俊': 0.090,
    '柴田善臣': 0.103, '柴田善': 0.103,
    '内田博幸': 0.060, '内田博': 0.060,
    '江田照男': 0.081, '江田照': 0.081,
    '丸山元気': 0.070, '丸山元': 0.070, '菅原': 0.080,
    '永野猛蔵': 0.060, '永野猛': 0.060,
    '野中悠太': 0.050, '野中': 0.050,
    '原優介': 0.050, '原優': 0.050, '原': 0.050,
    '長浜鷹人': 0.040, '長浜': 0.040,
    '上里涼': 0.030, '上里': 0.030,
    '遠藤裕喜': 0.030, '遠藤': 0.030,
    '佐藤翔馬': 0.030, '佐藤翔': 0.030, '佐藤': 0.030,
    '石神道大': 0.040, '石神道': 0.040,
    '原田和真': 0.050, '原田和': 0.050,
    '太宰啓介': 0.050, '太宰': 0.050,
}

def find_jockey_wr(name):
    if name in MODERN_JOCKEY_WR:
        return MODERN_JOCKEY_WR[name]
    if name in jockey_wr:
        return jockey_wr[name]
    for key, val in jockey_wr.items():
        if key.startswith(name) or name.startswith(key):
            return val
    return 0.05

TRAINER_WR = {
    '国枝栄': 0.150, '堀宣行': 0.145, '藤沢和雄': 0.140, '木村哲也': 0.135,
    '中内田充': 0.140, '友道康夫': 0.135, '矢作芳人': 0.130, '手塚貴久': 0.125,
    '池江泰寿': 0.120, '須貝尚介': 0.115, '西村真幸': 0.110, '武幸四郎': 0.115,
    '斎藤崇史': 0.110, '田中博康': 0.105, '萩原清': 0.100, '鹿戸雄一': 0.100,
    '高柳瑞樹': 0.095, '高野友和': 0.100, '音無秀孝': 0.095, '松永幹夫': 0.090,
    '池添学': 0.095, '安田隆行': 0.090, '清水久詞': 0.085, '尾関知人': 0.090,
    '杉山晴紀': 0.090, '宮田敬介': 0.100, '黒岩陽一': 0.085, '加藤征弘': 0.085,
    '栗田徹': 0.085, '美浦田中博': 0.080, '藤岡健一': 0.085, '大竹正博': 0.085,
    '奥村武': 0.080, '田村康仁': 0.080, '小島茂之': 0.080, '武井亮': 0.080,
    '和田正一': 0.075, '相沢郁': 0.075, '伊坂重信': 0.070, '小手川準': 0.075,
    '古賀慎明': 0.080, '上原博之': 0.075, '小桧山悟': 0.070, '竹内正洋': 0.070,
    '勢司和浩': 0.065, '小野次郎': 0.065, '菊沢隆徳': 0.070, '牧浦充徳': 0.080,
    '中竹和也': 0.075, '松下武士': 0.085, '岩戸孝樹': 0.075, '伊坂': 0.070,
    '美浦・田坂': 0.070, '美浦・鈴木': 0.070, '栗東・松永': 0.090,
}

def find_trainer_wr(name):
    if name in TRAINER_WR:
        return TRAINER_WR[name]
    for key, val in TRAINER_WR.items():
        if key in name or name in key:
            return val
    return 0.06

# 種牡馬の距離・馬場適性辞書
SIRE_APTITUDE = {
    'ディープインパクト': {'turf': 1.0, 'dirt': 0.3, 'sprint': 0.5, 'mile': 0.9, 'mid': 1.0, 'long': 0.8},
    'キングカメハメハ': {'turf': 0.8, 'dirt': 0.7, 'sprint': 0.6, 'mile': 0.8, 'mid': 0.9, 'long': 0.7},
    'ロードカナロア': {'turf': 0.9, 'dirt': 0.5, 'sprint': 1.0, 'mile': 0.8, 'mid': 0.5, 'long': 0.2},
    'ドゥラメンテ': {'turf': 0.9, 'dirt': 0.5, 'sprint': 0.4, 'mile': 0.8, 'mid': 1.0, 'long': 0.8},
    'エピファネイア': {'turf': 0.9, 'dirt': 0.4, 'sprint': 0.3, 'mile': 0.7, 'mid': 1.0, 'long': 0.9},
    'ハーツクライ': {'turf': 0.9, 'dirt': 0.3, 'sprint': 0.2, 'mile': 0.6, 'mid': 0.9, 'long': 1.0},
    'キタサンブラック': {'turf': 0.9, 'dirt': 0.4, 'sprint': 0.3, 'mile': 0.7, 'mid': 0.9, 'long': 1.0},
    'モーリス': {'turf': 0.8, 'dirt': 0.5, 'sprint': 0.5, 'mile': 0.9, 'mid': 0.8, 'long': 0.5},
    'サトノクラウン': {'turf': 0.7, 'dirt': 0.5, 'sprint': 0.3, 'mile': 0.7, 'mid': 0.9, 'long': 0.8},
    'オルフェーヴル': {'turf': 0.8, 'dirt': 0.6, 'sprint': 0.3, 'mile': 0.6, 'mid': 0.9, 'long': 1.0},
    'ゴールドシップ': {'turf': 0.8, 'dirt': 0.4, 'sprint': 0.2, 'mile': 0.5, 'mid': 0.8, 'long': 1.0},
    'ヘニーヒューズ': {'turf': 0.2, 'dirt': 1.0, 'sprint': 1.0, 'mile': 0.7, 'mid': 0.4, 'long': 0.1},
    'ダイワメジャー': {'turf': 0.8, 'dirt': 0.5, 'sprint': 0.7, 'mile': 1.0, 'mid': 0.6, 'long': 0.3},
    'スクリーンヒーロー': {'turf': 0.7, 'dirt': 0.6, 'sprint': 0.3, 'mile': 0.7, 'mid': 1.0, 'long': 0.8},
    'ルーラーシップ': {'turf': 0.8, 'dirt': 0.5, 'sprint': 0.3, 'mile': 0.7, 'mid': 0.9, 'long': 0.9},
    'ジャスタウェイ': {'turf': 0.8, 'dirt': 0.4, 'sprint': 0.4, 'mile': 0.9, 'mid': 0.9, 'long': 0.6},
    'シルバーステート': {'turf': 0.8, 'dirt': 0.5, 'sprint': 0.4, 'mile': 0.8, 'mid': 0.9, 'long': 0.7},
    'サトノダイヤモンド': {'turf': 0.8, 'dirt': 0.4, 'sprint': 0.2, 'mile': 0.6, 'mid': 0.9, 'long': 1.0},
    'ミッキーアイル': {'turf': 0.8, 'dirt': 0.4, 'sprint': 0.9, 'mile': 0.9, 'mid': 0.5, 'long': 0.2},
    'リアルスティール': {'turf': 0.8, 'dirt': 0.4, 'sprint': 0.3, 'mile': 0.8, 'mid': 0.9, 'long': 0.7},
    'ホッコータルマエ': {'turf': 0.2, 'dirt': 1.0, 'sprint': 0.7, 'mile': 0.9, 'mid': 0.8, 'long': 0.4},
    'シニスターミニスター': {'turf': 0.1, 'dirt': 1.0, 'sprint': 0.8, 'mile': 0.9, 'mid': 0.6, 'long': 0.3},
    'パイロ': {'turf': 0.1, 'dirt': 1.0, 'sprint': 0.9, 'mile': 0.8, 'mid': 0.5, 'long': 0.2},
    'コントレイル': {'turf': 0.9, 'dirt': 0.3, 'sprint': 0.3, 'mile': 0.7, 'mid': 1.0, 'long': 0.9},
    'イスラボニータ': {'turf': 0.7, 'dirt': 0.5, 'sprint': 0.4, 'mile': 0.9, 'mid': 0.8, 'long': 0.5},
    'ドレフォン': {'turf': 0.5, 'dirt': 0.8, 'sprint': 0.8, 'mile': 0.8, 'mid': 0.6, 'long': 0.3},
    'マインドユアビスケッツ': {'turf': 0.3, 'dirt': 0.9, 'sprint': 0.9, 'mile': 0.7, 'mid': 0.4, 'long': 0.1},
    'サウスヴィグラス': {'turf': 0.1, 'dirt': 1.0, 'sprint': 1.0, 'mile': 0.6, 'mid': 0.2, 'long': 0.0},
    'ブリックスアンドモルタル': {'turf': 0.8, 'dirt': 0.5, 'sprint': 0.4, 'mile': 0.8, 'mid': 0.9, 'long': 0.7},
    'サートゥルナーリア': {'turf': 0.9, 'dirt': 0.4, 'sprint': 0.3, 'mile': 0.7, 'mid': 1.0, 'long': 0.8},
    'スワーヴリチャード': {'turf': 0.8, 'dirt': 0.5, 'sprint': 0.3, 'mile': 0.7, 'mid': 0.9, 'long': 0.9},
    'レイデオロ': {'turf': 0.8, 'dirt': 0.5, 'sprint': 0.3, 'mile': 0.7, 'mid': 0.9, 'long': 0.8},
}

def calc_sire_score(father, surface, distance):
    """血統スコア: 父馬の距離・馬場適性"""
    apt = SIRE_APTITUDE.get(father, None)
    if not apt:
        return 0.5
    surf_score = apt.get('turf', 0.5) if surface == '芝' else apt.get('dirt', 0.5)
    if distance <= 1400:
        dist_score = apt.get('sprint', 0.5)
    elif distance <= 1800:
        dist_score = apt.get('mile', 0.5)
    elif distance <= 2200:
        dist_score = apt.get('mid', 0.5)
    else:
        dist_score = apt.get('long', 0.5)
    return (surf_score * 0.5 + dist_score * 0.5)

def get_horse_stats(horse_id, target_distance, target_surface, target_course=""):
    """全要素取得: 前走着順/適性/人気/コース/間隔/騎手/脚質/上がり3F/着差/連対率/血統/休み明け"""
    last_finish = 5
    dist_results = []
    surf_results = []
    popularity_list = []
    course_results = []
    race_dates = []
    prev_jockey = ""
    passing_positions = []
    agari3f_list = []
    margin_text = ""
    finish_list = []
    rest_finishes = []
    father_name = ""
    try:
        url = "https://db.netkeiba.com/horse/result/" + horse_id + "/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        # 父名取得（血統テーブルから）
        prof_table = soup.find("table", class_="db_prof_table")
        if prof_table:
            tds_prof = prof_table.find_all("td")
            for td in tds_prof:
                a_tag = td.find("a", href=re.compile(r"/horse/sire/"))
                if a_tag:
                    father_name = a_tag.get_text(strip=True)
                    break
        if not father_name:
            blood_table = soup.find("table", summary=re.compile(".*血統.*"))
            if blood_table:
                a_tags = blood_table.find_all("a", href=re.compile(r"/horse/"))
                if a_tags:
                    father_name = a_tags[0].get_text(strip=True)
        table = soup.find("table", class_="db_h_race_results")
        if table:
            tbody = table.find("tbody")
            if tbody:
                rows = tbody.find_all("tr")
                prev_date = None
                for row_idx, row in enumerate(rows):
                    tds = row.find_all("td")
                    if len(tds) < 15:
                        continue
                    finish_text = tds[11].get_text(strip=True)
                    if not finish_text.isdigit():
                        continue
                    finish = int(finish_text)
                    finish_list.append(finish)
                    if row_idx == 0:
                        last_finish = finish
                    # 日付取得 (tds[0])
                    date_text = tds[0].get_text(strip=True)
                    date_m = re.search(r'(\d{4})/(\d{1,2})/(\d{1,2})', date_text)
                    cur_date = None
                    if date_m:
                        try:
                            from datetime import datetime
                            cur_date = datetime(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)))
                            race_dates.append(cur_date)
                        except:
                            pass
                    # 休み明け成績（前走との間隔が60日以上なら休み明け）
                    if prev_date and cur_date:
                        gap = (prev_date - cur_date).days
                        if gap >= 60:
                            rest_finishes.append(finish_list[-2] if len(finish_list) >= 2 else finish)
                    prev_date = cur_date
                    # 前走騎手 (tds[12])
                    if row_idx == 0 and len(tds) > 12:
                        prev_jockey = tds[12].get_text(strip=True)
                    # 前走着差 (tds[18])
                    if row_idx == 0 and len(tds) > 18:
                        margin_text = tds[18].get_text(strip=True)
                    pop_text = tds[10].get_text(strip=True)
                    if pop_text.isdigit():
                        popularity_list.append(int(pop_text))
                    dist_col = tds[14].get_text(strip=True)
                    dm = re.match(r'([芝ダ障])(\d+)', dist_col)
                    if dm:
                        surf_ch = dm.group(1)
                        dist_val = int(dm.group(2))
                        if target_distance > 0 and abs(dist_val - target_distance) <= 200:
                            dist_results.append(finish)
                        surf_name = '芝' if surf_ch == '芝' else 'ダ'
                        if surf_name == target_surface:
                            surf_results.append(finish)
                    # コース適性
                    if target_course and len(tds) > 1:
                        course_text = tds[1].get_text(strip=True)
                        if target_course in course_text:
                            course_results.append(finish)
                    # 通過順位 (tds[20])
                    if len(tds) > 20:
                        pass_text = tds[20].get_text(strip=True)
                        pass_nums = re.findall(r'\d+', pass_text)
                        if pass_nums:
                            passing_positions.append(int(pass_nums[0]))
                    # 上がり3F (tds[22])
                    if len(tds) > 22:
                        agari_text = tds[22].get_text(strip=True)
                        try:
                            agari_val = float(agari_text)
                            if 30.0 < agari_val < 45.0:
                                agari3f_list.append(agari_val)
                        except:
                            pass
                    if row_idx >= 9:
                        break
        if last_finish == 5:
            all_text = soup.get_text()
            m = re.search(r'(\d{1,2})\(\d+人気\)', all_text)
            if m:
                last_finish = int(m.group(1))
    except Exception:
        pass
    # 適性スコア
    dist_apt = 0.5
    surf_apt = 0.5
    if dist_results:
        avg = sum(dist_results) / len(dist_results)
        dist_apt = max(0.0, min(1.0, 1.0 - (avg - 1) / 17.0))
    if surf_results:
        avg = sum(surf_results) / len(surf_results)
        surf_apt = max(0.0, min(1.0, 1.0 - (avg - 1) / 17.0))
    pop_score = 0.5
    if popularity_list:
        avg_pop = sum(popularity_list) / len(popularity_list)
        pop_score = max(0.0, min(1.0, 1.0 - (avg_pop - 1) / 17.0))
    course_apt = 0.5
    if course_results:
        avg = sum(course_results) / len(course_results)
        course_apt = max(0.0, min(1.0, 1.0 - (avg - 1) / 17.0))
    # 前走間隔
    interval_days = 30
    if len(race_dates) >= 1:
        from datetime import datetime
        today = datetime.now()
        diff = (today - race_dates[0]).days
        if diff > 0:
            interval_days = diff
    # 脚質判定
    running_style = 0
    if passing_positions:
        avg_pass = sum(passing_positions) / len(passing_positions)
        if avg_pass <= 2.0:
            running_style = 1
        elif avg_pass <= 5.0:
            running_style = 2
        elif avg_pass <= 10.0:
            running_style = 3
        else:
            running_style = 4
    # 上がり3F平均
    avg_agari = 35.5
    if agari3f_list:
        avg_agari = sum(agari3f_list) / len(agari3f_list)
    # 前走着差スコア (ハナ/クビ/アタマ→僅差、大差→離された)
    margin_score = 0.5
    if margin_text:
        if margin_text in ['ハナ', 'クビ', 'アタマ', '同着']:
            margin_score = 0.8 if last_finish <= 3 else 0.4
        elif re.match(r'^[\d\.\/]+$', margin_text):
            try:
                parts = margin_text.replace('/', '.').split('.')
                val = float(parts[0])
                if val <= 0.5:
                    margin_score = 0.7 if last_finish <= 3 else 0.4
                elif val <= 1.0:
                    margin_score = 0.6 if last_finish <= 5 else 0.4
                else:
                    margin_score = 0.3
            except:
                pass
        elif '大' in margin_text:
            margin_score = 0.2
    # 連対率・複勝率
    rentai_rate = 0.0
    fukusho_rate = 0.0
    if finish_list:
        rentai_rate = sum(1 for f in finish_list if f <= 2) / len(finish_list)
        fukusho_rate = sum(1 for f in finish_list if f <= 3) / len(finish_list)
    # 休み明け適性
    rest_apt = 0.5
    if rest_finishes:
        avg_rest = sum(rest_finishes) / len(rest_finishes)
        rest_apt = max(0.0, min(1.0, 1.0 - (avg_rest - 1) / 17.0))
    return {
        'last_finish': last_finish,
        'dist_apt': dist_apt,
        'surf_apt': surf_apt,
        'pop_score': pop_score,
        'course_apt': course_apt,
        'interval_days': interval_days,
        'prev_jockey': prev_jockey,
        'running_style': running_style,
        'avg_agari': avg_agari,
        'margin_score': margin_score,
    }

def parse_shutuba(race_id):
    url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + race_id
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")
    race_name = "レース"
    tag = soup.find("div", class_="RaceName")
    if tag and tag.get_text(strip=True):
        race_name = tag.get_text(strip=True)
    else:
        tag = soup.find("h1")
        if tag and tag.get_text(strip=True):
            race_name = tag.get_text(strip=True)
        else:
            tag = soup.find("title")
            if tag:
                title_text = tag.get_text(strip=True)
                m = re.search(r'(\S+\d+R)', title_text)
                if m:
                    race_name = m.group(1)
    race_data01 = soup.find("div", class_="RaceData01")
    data01_text = race_data01.get_text(strip=True) if race_data01 else ""
    if not data01_text:
        data01_text = soup.get_text()
    dist_match = re.search(r'(\d{4})m', data01_text)
    distance = int(dist_match.group(1)) if dist_match else 0
    surface = '芝' if '芝' in data01_text else ('ダ' if 'ダ' in data01_text else '芝')
    cond_match = re.search(r'馬場:(\S+)', data01_text)
    condition = cond_match.group(1) if cond_match else '良'
    race_data02 = soup.find("div", class_="RaceData02")
    data02_text = race_data02.get_text(strip=True) if race_data02 else data01_text
    course_name = ""
    for cname in COURSE_MAP.keys():
        if cname in data02_text:
            course_name = cname
            break
    race_info = dict(distance=distance, surface=surface, condition=condition, course=course_name)
    rows = soup.select("tr.HorseList")
    horses = []
    horse_ids = []
    for row in rows:
        row_class = row.get("class", [])
        if "Cancel" in row_class:
            continue
        waku_tag = row.select_one("td.Waku span")
        waku = 0
        if waku_tag:
            wt = waku_tag.get_text(strip=True)
            if wt.isdigit():
                waku = int(wt)
        umaban_tag = row.select_one("td.Umaban")
        umaban = 0
        if umaban_tag:
            ut = umaban_tag.get_text(strip=True)
            if ut.isdigit():
                umaban = int(ut)
        name_tag = row.select_one("span.HorseName a")
        if name_tag is None:
            continue
        horse_name = name_tag.get_text(strip=True)
        href = name_tag.get("href", "")
        hid_match = re.search(r'/horse/(\d+)', href)
        horse_id = hid_match.group(1) if hid_match else None
        info_tag = row.select_one("td.Barei") or row.select_one("span.Barei")
        if info_tag:
            sex_age_text = info_tag.get_text(strip=True)
        else:
            tds_all = row.find_all("td")
            sex_age_text = ""
            for td in tds_all:
                t = td.get_text(strip=True)
                if re.match(r'^[牡牝セ騸]\d+$', t):
                    sex_age_text = t
                    break
        sex = sex_age_text[0] if sex_age_text else '牡'
        age = int(sex_age_text[1:]) if sex_age_text and sex_age_text[1:].isdigit() else 3
        kinryo = 55.0
        tds = row.find_all("td")
        for td in tds:
            t = td.get_text(strip=True)
            try:
                val = float(t)
                if 48.0 <= val <= 62.0:
                    kinryo = val
                    break
            except ValueError:
                continue
        jockey_tag = row.select_one("td.Jockey a") or row.select_one("a[href*='jockey']")
        jockey_name = jockey_tag.get_text(strip=True) if jockey_tag else ""
        horse_weight = 480
        weight_diff = 0
        for td in tds:
            bw_text = td.get_text(strip=True)
            bw_match = re.search(r'(\d{3,})\(([\+\-]?\d+)\)', bw_text)
            if bw_match:
                w = int(bw_match.group(1))
                if 350 <= w <= 600:
                    horse_weight = w
                    weight_diff = int(bw_match.group(2))
                    break
        horses.append({
            '馬名': horse_name,
            '馬体重': horse_weight, '場体重増減': weight_diff,
            '斤量': kinryo, '馬齢': age,
            '距離(m)': distance,
            '競馬場コード_enc': COURSE_MAP.get(course_name, 4),
            '芝ダート_enc': SURFACE_MAP.get(surface, 0),
            '馬場状態_enc': COND_MAP.get(condition, 0),
            '性別_enc': SEX_MAP.get(sex, 0),
            '騎手勝率': find_jockey_wr(jockey_name),
            '騎手名': jockey_name,
            '枠番': waku,
            '馬番': umaban,
        })
        # 調教師名を取得（別途）
        trainer_tag = row.select_one("td.Trainer a") or row.select_one("a[href*='trainer']")
        trainer_name = trainer_tag.get_text(strip=True) if trainer_tag else ""
        horses[-1]['調教師'] = trainer_name
        horses[-1]['調教師勝率'] = find_trainer_wr(trainer_name)
        horse_ids.append(horse_id)
    return race_name, horses, horse_ids, race_info

def render_top3(rank, name, jockey, finish, score, max_score, pop_label, apt_str):
    medal_map = {1: '🥇', 2: '🥈', 3: '🥉'}
    color_map = {1: 'gold', 2: 'silver', 3: 'bronze'}
    medal = medal_map.get(rank, '')
    color = color_map.get(rank, 'bronze')
    bw = int((score / max_score) * 100) if max_score > 0 else 0
    html = '<div class="top-card-' + color + '">'
    html += '<div style="display:flex;align-items:center;">'
    html += '<span class="rank-' + color + '">' + medal + '</span>'
    html += '<div>'
    html += '<div class="horse-name-card">' + name + '</div>'
    html += '<div class="details-card">騎手: ' + jockey
    html += '｜前走: ' + str(finish) + '着'
    html += '｜人気: ' + pop_label
    html += '｜適性: ' + apt_str
    html += '｜スコア: ' + str(round(score, 3)) + '</div>'
    html += '</div></div>'
    html += '<div class="bar-bg"><div class="bar-' + color + '" style="width:' + str(bw) + '%"></div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def apt_label(d, s):
    avg = (d + s) / 2
    if avg >= 0.7:
        return '◎'
    elif avg >= 0.5:
        return '○'
    elif avg >= 0.3:
        return '△'
    else:
        return '✕'

def pop_label(score):
    if score >= 0.8:
        return '★★★'
    elif score >= 0.6:
        return '★★'
    elif score >= 0.4:
        return '★'
    else:
        return '-'

# ===== メイン =====
st.markdown("# 🏇 競馬AI予想")
url_input = st.text_input("netkeibaの出馬表URLを入力")

if st.button("🔍 予想する") and url_input:
    url_input = url_input.replace("race.sp.netkeiba.com", "race.netkeiba.com")
    rid_match = re.search(r'race_id=(\d+)', url_input)
    if not rid_match:
        st.error("URLからrace_idを取得できませんでした")
        st.stop()
    race_id = rid_match.group(1)
    with st.spinner("出馬表を取得中..."):
        race_name, horses, horse_ids, race_info = parse_shutuba(race_id)
    if not horses:
        st.error("馬データを取得できませんでした。URLを確認してください。")
        st.stop()
    surf = '🟢 芝' if race_info['surface'] == '芝' else '🟤 ダート'
    hdr = '<div class="race-hdr">'
    hdr += '<h2>' + race_info['course'] + ' ' + race_name + '</h2>'
    hdr += '<p>' + surf + ' ' + str(race_info['distance']) + 'm ｜ 馬場: '
    hdr += race_info['condition'] + ' ｜ ' + str(len(horses)) + '頭立て</p>'
    hdr += '</div>'
    st.markdown(hdr, unsafe_allow_html=True)
    with st.spinner("各馬の成績を分析中..."):
        progress_bar = st.progress(0)
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                try:
                    stats = get_horse_stats(
                        hid, race_info['distance'], race_info['surface'], race_info['course']
                    )
                    horse['前走着順'] = stats.get('last_finish', 5)
                    horse['距離適性'] = stats.get('dist_apt', 0.5)
                    horse['馬場適性'] = stats.get('surf_apt', 0.5)
                    horse['人気傾向'] = stats.get('pop_score', 0.5)
                    horse['コース適性'] = stats.get('course_apt', 0.5)
                    horse['前走間隔'] = stats.get('interval_days', 30)
                    horse['前走騎手'] = stats.get('prev_jockey', '')
                    horse['脚質'] = stats.get('running_style', 0)
                    horse['上がり3F'] = stats.get('avg_agari', 35.5)
                    horse['着差スコア'] = stats.get('margin_score', 0.5)
                    horse['連対率'] = stats.get('rentai_rate', 0.0)
                    horse['複勝率'] = stats.get('fukusho_rate', 0.0)
                    horse['休み明け適性'] = stats.get('rest_apt', 0.5)
                    horse['父'] = stats.get('father', '')
                    horse['血統スコア'] = calc_sire_score(
                        stats.get('father', ''), race_info['surface'], race_info['distance']
                    )
                except Exception:
                    horse['前走着順'] = 5
                    horse['距離適性'] = 0.5
                    horse['馬場適性'] = 0.5
                    horse['人気傾向'] = 0.5
                    horse['コース適性'] = 0.5
                    horse['前走間隔'] = 30
                    horse['前走騎手'] = ""
                    horse['脚質'] = 0
                    horse['上がり3F'] = 35.5
                    horse['着差スコア'] = 0.5
                    horse['連対率'] = 0.0
                    horse['複勝率'] = 0.0
                    horse['休み明け適性'] = 0.5
                    horse['父'] = ""
                    horse['血統スコア'] = 0.5
            else:
                horse['前走着順'] = 5
                horse['距離適性'] = 0.5
                horse['馬場適性'] = 0.5
                horse['人気傾向'] = 0.5
                horse['コース適性'] = 0.5
                horse['前走間隔'] = 30
                horse['前走騎手'] = ""
                horse['脚質'] = 0
                horse['上がり3F'] = 35.5
                horse['着差スコア'] = 0.5
                horse['連対率'] = 0.0
                horse['複勝率'] = 0.0
                horse['休み明け適性'] = 0.5
                horse['父'] = ""
                horse['血統スコア'] = 0.5
            progress_bar.progress((i + 1) / len(horses))
            if i < len(horses) - 1:
                time.sleep(0.5)
        progress_bar.empty()
    df = pd.DataFrame(horses)
    X = df[FEATURES].values
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        ai_scores = proba[:, 1]
    else:
        ai_scores = proba[:, :3].sum(axis=1) if proba.shape[1] >= 3 else proba[:, 0]
    apt_scores = (df['距離適性'].values + df['馬場適性'].values) / 2.0
    pop_scores = df['人気傾向'].values
    course_scores = df['コース適性'].values
    # 枠番スコア
    num_horses = len(horses)
    waku_scores = []
    for _, h in df.iterrows():
        ub = h.get('馬番', 0)
        if ub > 0 and num_horses > 0:
            pos = ub / num_horses
            dist = h['距離(m)']
            if dist <= 1400:
                ws = max(0.0, 1.0 - pos * 0.6)
            elif dist <= 1800:
                ws = max(0.0, 1.0 - pos * 0.3)
            else:
                ws = 0.5
        else:
            ws = 0.5
        waku_scores.append(ws)
    waku_scores = np.array(waku_scores)
    # 前走間隔スコア
    interval_scores = []
    for _, h in df.iterrows():
        days = h.get('前走間隔', 30)
        if 14 <= days <= 35:
            iscore = 0.7
        elif 7 <= days < 14:
            iscore = 0.5
        elif 35 < days <= 90:
            iscore = 0.5
        elif days > 90:
            iscore = 0.3
        else:
            iscore = 0.4
        interval_scores.append(iscore)
    interval_scores = np.array(interval_scores)
    # 馬体重変動スコア
    weight_scores = []
    for _, h in df.iterrows():
        wd = abs(h.get('場体重増減', 0))
        if wd <= 4:
            wscore = 0.7
        elif wd <= 8:
            wscore = 0.5
        elif wd <= 14:
            wscore = 0.3
        else:
            wscore = 0.2
        weight_scores.append(wscore)
    weight_scores = np.array(weight_scores)
    # 斤量相対スコア
    avg_kinryo = df['斤量'].mean()
    kinryo_scores = np.clip(1.0 - (df['斤量'].values - avg_kinryo) / 10.0, 0.0, 1.0) * 0.5 + 0.25
    # 騎手乗り替わりスコア
    jchange_scores = []
    for _, h in df.iterrows():
        cur_j = h.get('騎手名', '')
        prev_j = h.get('前走騎手', '')
        if not prev_j or not cur_j:
            jscore = 0.5
        elif prev_j in cur_j or cur_j in prev_j:
            jscore = 0.6
        else:
            cur_wr = find_jockey_wr(cur_j)
            prev_wr = find_jockey_wr(prev_j)
            if cur_wr > prev_wr + 0.02:
                jscore = 0.7
            elif cur_wr < prev_wr - 0.02:
                jscore = 0.3
            else:
                jscore = 0.5
        jchange_scores.append(jscore)
    jchange_scores = np.array(jchange_scores)
    # 脚質展開スコア
    pace_scores = []
    for _, h in df.iterrows():
        rs = h.get('脚質', 0)
        dist = h['距離(m)']
        cond = race_info.get('condition', '良')
        nh = num_horses
        if rs == 0:
            pscore = 0.5
        elif rs == 1:
            pscore = 0.6
            if nh <= 10: pscore += 0.1
            if cond in ['重', '不良']: pscore += 0.05
            if dist <= 1400: pscore += 0.05
            if nh >= 16: pscore -= 0.1
        elif rs == 2:
            pscore = 0.65
            if nh <= 12: pscore += 0.05
            if cond in ['重', '不良']: pscore += 0.05
        elif rs == 3:
            pscore = 0.5
            if nh >= 14: pscore += 0.1
            if dist >= 2000: pscore += 0.05
        else:
            pscore = 0.35
            if nh >= 16: pscore += 0.1
            if dist >= 2400: pscore += 0.1
        pace_scores.append(min(1.0, max(0.0, pscore)))
    pace_scores = np.array(pace_scores)
    # 上がり3Fスコア (速いほど高い: 33秒台=1.0, 37秒台=0.3)
    agari_scores = np.clip(1.0 - (df['上がり3F'].values - 33.0) / 5.0, 0.0, 1.0)
    # 着差スコア
    margin_scores = df['着差スコア'].values
    # 連対率・複勝率スコア
    rentai_scores = df['連対率'].values
    fukusho_scores = df['複勝率'].values
    # 血統スコア
    sire_scores = df['血統スコア'].values
    # 休み明け適性（休み明けの場合のみ適用）
    rest_scores = []
    for _, h in df.iterrows():
        days = h.get('前走間隔', 30)
        if days >= 60:
            rest_scores.append(h.get('休み明け適性', 0.5))
        else:
            rest_scores.append(0.5)
    rest_scores = np.array(rest_scores)
    # 季節補正
    from datetime import datetime
    month = datetime.now().month
    season_scores = []
    for _, h in df.iterrows():
        ss = 0.5
        bw = h.get('馬体重', 480)
        if month in [12, 1, 2]:
            if bw >= 500: ss = 0.6
            if h.get('脚質', 0) in [1, 2]: ss += 0.05
        elif month in [6, 7, 8]:
            if bw <= 470: ss = 0.6
        season_scores.append(min(1.0, ss))
    season_scores = np.array(season_scores)
    # === 最終スコア (15要素) ===
    # 調教師勝率スコア
    trainer_scores = []
    for _, h in df.iterrows():
        twr = h.get('調教師勝率', 0.06)
        tscore = min(1.0, twr / 0.15)
        trainer_scores.append(tscore)
    trainer_scores = np.array(trainer_scores)
    # 枠順×脚質コンボスコア
    combo_scores = []
    for _, h in df.iterrows():
        ub = h.get('馬番', 0)
        rs = h.get('脚質', 0)
        if ub == 0 or rs == 0 or num_horses == 0:
            combo_scores.append(0.5)
            continue
        pos = ub / num_horses
        inner = pos <= 0.35
        outer = pos >= 0.65
        if rs == 1 and inner:
            cs = 0.8
        elif rs == 1 and outer:
            cs = 0.4
        elif rs == 2 and inner:
            cs = 0.7
        elif rs == 2:
            cs = 0.6
        elif rs == 3 and outer:
            cs = 0.55
        elif rs == 3:
            cs = 0.5
        elif rs == 4 and outer:
            cs = 0.45
        elif rs == 4:
            cs = 0.35
        else:
            cs = 0.5
        combo_scores.append(cs)
    combo_scores = np.array(combo_scores)
    # トラックバイアス推定（馬場状態から内外有利を推定）
    bias_scores = []
    cond = race_info.get('condition', '良')
    surf = race_info.get('surface', '芝')
    for _, h in df.iterrows():
        ub = h.get('馬番', 0)
        if ub == 0 or num_horses == 0:
            bias_scores.append(0.5)
            continue
        pos = ub / num_horses
        inner = pos <= 0.35
        if surf == '芝' and cond in ['良', '稍', '稍重']:
            bs = 0.6 if inner else 0.45
        elif surf == '芝' and cond in ['重', '不良', '不']:
            bs = 0.5
        elif surf != '芝' and cond in ['良']:
            bs = 0.55 if inner else 0.45
        elif surf != '芝' and cond in ['重', '不良', '不']:
            bs = 0.55 if inner else 0.5
        else:
            bs = 0.5
        bias_scores.append(bs)
    bias_scores = np.array(bias_scores)
    # AI 23% + 人気 9% + 適性 6% + コース 6% + 脚質 6% + 上がり 6%
    # + 血統 5% + 複勝率 5% + 枠番 4% + 間隔 4% + 着差 4%
    # + 調教師 4% + 枠脚質コンボ 4% + トラックバイアス 3%
    # + 体重 3% + 騎手替 2% + 斤量 2% + 季節 2% + 休み明け 1% + 連対率 1%
    final_scores = (
        ai_scores * 0.23 + pop_scores * 0.09 + apt_scores * 0.06
        + course_scores * 0.06 + pace_scores * 0.06 + agari_scores * 0.06
        + sire_scores * 0.05 + fukusho_scores * 0.05 + waku_scores * 0.04
        + interval_scores * 0.04 + margin_scores * 0.04
        + trainer_scores * 0.04 + combo_scores * 0.04 + bias_scores * 0.03
        + weight_scores * 0.03 + jchange_scores * 0.02 + kinryo_scores * 0.02
        + season_scores * 0.02 + rest_scores * 0.01 + rentai_scores * 0.01
    )
    df['スコア'] = final_scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    df['適性'] = df.apply(lambda r: apt_label(r['距離適性'], r['馬場適性']), axis=1)
    df['人気'] = df['人気傾向'].map(pop_label)
    # TOP3
    st.markdown("### 🏆 AI推奨 TOP3")
    max_score = df['スコア'].max()
    for _, row in df.head(3).iterrows():
        render_top3(
            int(row['AI順位']), row['馬名'], row['騎手名'],
            int(row['前走着順']), row['スコア'], max_score,
            row['人気'], row['適性']
        )
    # グラフ
    st.markdown("### 📊 全馬スコア")
    chart_df = df[['馬名', 'スコア']].copy()
    chart_df = chart_df.set_index('馬名')
    st.bar_chart(chart_df, color='#FFD700')
    # テーブル
    st.markdown("### 📋 全馬データ")
    display_cols = ['AI順位', '馬名', '騎手名', '前走着順', '人気', '適性', '騎手勝率', 'スコア']
    result_df = df[display_cols].copy()
    result_df['スコア'] = result_df['スコア'].map(lambda x: str(round(x, 3)))
    result_df['騎手勝率'] = result_df['騎手勝率'].map(lambda x: str(round(x, 3)))
    result_df = result_df.reset_index(drop=True)
    st.table(result_df)
