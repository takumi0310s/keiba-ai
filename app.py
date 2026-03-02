import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime

st.set_page_config(page_title="KEIBA AI", page_icon="🏇", layout="wide")

# ===== CSS =====
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Oswald:wght@400;600;700&family=Noto+Sans+JP:wght@300;500;700;900&display=swap');
[data-testid="stAppViewContainer"] { background: #0a0a0f; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 520px; margin: 0 auto; padding: 1rem 1rem 3rem; }
h1, h2, h3, p, span, div, td, th { color: #e8e8f0 !important; }

/* Header */
.site-header {
    text-align: center; padding: 28px 0 18px; position: relative;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}
.site-logo {
    font-family: 'Bebas Neue', sans-serif; font-size: 3em; letter-spacing: 6px;
    background: linear-gradient(135deg, #f0c040, #fff, #f0c040);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;
}
.site-sub {
    font-family: 'Oswald', sans-serif; font-size: 0.82em; color: #6a6a80 !important;
    letter-spacing: 8px; margin-top: 2px;
}

/* Race Card */
.race-card {
    background: linear-gradient(135deg, #0f2847 0%, #0a1628 60%, #12082a 100%);
    border-radius: 16px; padding: 22px; margin: 16px 0;
    border: 1px solid rgba(255,255,255,0.06); position: relative; overflow: hidden;
}
.race-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-family: 'Oswald', sans-serif; font-size: 0.8em; letter-spacing: 2px; margin-bottom: 10px;
}
.badge-dirt { background: linear-gradient(90deg, #c0783a, #a06030); color: #fff; }
.badge-turf { background: linear-gradient(90deg, #2ecc71, #1a9050); color: #fff; }
.race-name { font-weight: 900; font-size: 1.5em; margin-bottom: 6px; }
.race-meta { display: flex; gap: 14px; flex-wrap: wrap; color: #6a6a80 !important; font-size: 0.85em; }

/* Pace Panel */
.pace-panel {
    margin-top: 16px; padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.pace-title {
    font-family: 'Oswald', sans-serif; font-size: 0.75em; color: #6a6a80 !important;
    letter-spacing: 2px; margin-bottom: 10px;
}
.pace-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.pace-item {
    text-align: center; padding: 10px 4px; border-radius: 10px;
    background: rgba(255,255,255,0.03); position: relative; overflow: hidden;
}
.pace-item::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.pace-best::before { background: #00e87b; }
.pace-good::before { background: #f0c040; }
.pace-fair::before { background: #6a6a80; }
.pace-bad::before { background: #ff4060; }
.pace-best .ps { color: #00e87b; }
.pace-good .ps { color: #f0c040; }
.pace-fair .ps { color: #b0b8c8; }
.pace-bad .ps { color: #ff4060; }
.ps { font-weight: 700; font-size: 0.9em; }
.pr { font-family: 'Oswald', sans-serif; font-size: 1.3em; font-weight: 700; }
.preason { font-size: 0.6em; color: #6a6a80 !important; margin-top: 2px; }

/* Section Title */
.sec-title {
    font-family: 'Oswald', sans-serif; font-size: 1.2em; letter-spacing: 3px;
    margin: 24px 0 14px; display: flex; align-items: center; gap: 8px;
}
.sec-line { flex: 1; height: 1px; background: linear-gradient(90deg, #6a6a80, transparent); }

/* Horse Card */
.hcard {
    background: #12121c; border-radius: 14px; padding: 18px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.04); position: relative; overflow: hidden;
}
.hcard::before { content: ''; position: absolute; top: 0; left: 0; bottom: 0; width: 4px; }
.hcard-g::before { background: linear-gradient(180deg, #f0c040, #c89020); }
.hcard-s::before { background: linear-gradient(180deg, #b0b8c8, #808898); }
.hcard-b::before { background: linear-gradient(180deg, #c87840, #905020); }
.hcard-top { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.hrank {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.6em; line-height: 1;
    min-width: 36px; text-align: center;
}
.hrank-g { color: #f0c040 !important; }
.hrank-s { color: #b0b8c8 !important; }
.hrank-b { color: #c87840 !important; }
.hname { font-weight: 900; font-size: 1.3em; line-height: 1.2; }
.hjockey { color: #6a6a80 !important; font-size: 0.8em; margin-top: 2px; }
.hscore {
    font-family: 'Oswald', sans-serif; font-size: 1.5em; font-weight: 700;
    padding: 5px 12px; border-radius: 10px; background: rgba(255,255,255,0.05);
}
.hscore-g { color: #f0c040 !important; }
.hscore-s { color: #b0b8c8 !important; }
.hscore-b { color: #c87840 !important; }

/* Stats Grid */
.sgrid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
.sitem { text-align: center; padding: 7px 2px; background: rgba(255,255,255,0.03); border-radius: 8px; }
.slbl { font-size: 0.6em; color: #6a6a80 !important; letter-spacing: 1px; margin-bottom: 2px; }
.sval { font-family: 'Oswald', sans-serif; font-size: 0.92em; font-weight: 600; }
.sg { color: #00e87b !important; }
.sw { color: #f0c040 !important; }
.sr { color: #ff4060 !important; }

/* Style & Pace Match */
.stag { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.72em; font-weight: 700; }
.st-nige { background: rgba(255,64,96,0.15); color: #ff4060 !important; }
.st-senk { background: rgba(0,232,123,0.15); color: #00e87b !important; }
.st-sasi { background: rgba(0,212,255,0.15); color: #00d4ff !important; }
.st-oiko { background: rgba(168,85,247,0.15); color: #a855f7 !important; }
.pmatch { display: inline-block; font-size: 0.68em; padding: 1px 5px; border-radius: 8px; margin-left: 3px; font-weight: 700; }
.pm-best { background: rgba(0,232,123,0.2); color: #00e87b !important; }
.pm-good { background: rgba(240,192,64,0.2); color: #f0c040 !important; }
.pm-fair { background: rgba(176,184,200,0.15); color: #b0b8c8 !important; }
.pm-bad { background: rgba(255,64,96,0.2); color: #ff4060 !important; }

/* Weight */
.wbadge { font-family: 'Oswald'; font-size: 0.78em; padding: 2px 5px; border-radius: 4px; }
.w-plus { background: rgba(0,232,123,0.12); color: #00e87b !important; }
.w-minus { background: rgba(255,64,96,0.12); color: #ff4060 !important; }
.w-flat { background: rgba(255,255,255,0.06); color: #6a6a80 !important; }
.w-danger { background: rgba(255,64,96,0.25); color: #ff4060 !important; }

/* Tag */
.tagrow { display: flex; gap: 5px; margin-top: 9px; flex-wrap: wrap; }
.tag { font-size: 0.68em; padding: 3px 9px; border-radius: 12px; background: rgba(255,255,255,0.06); color: #6a6a80 !important; }
.tag-sire { border: 1px solid rgba(0,212,255,0.3); color: #00d4ff !important; }

/* Score Bar */
.sbar-w { margin-top: 10px; height: 4px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }
.sbar { height: 100%; border-radius: 3px; }
.sbar-g { background: linear-gradient(90deg, #f0c040, #ffe070); }
.sbar-s { background: linear-gradient(90deg, #b0b8c8, #d0d8e8); }
.sbar-b { background: linear-gradient(90deg, #c87840, #e0a070); }

/* Buy Section */
.buy-card {
    background: #12121c; border-radius: 14px; padding: 18px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.04); position: relative; overflow: hidden;
}
.buy-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.buy-honmei::before { background: linear-gradient(90deg, #f0c040, #00e87b); }
.buy-hiroku::before { background: linear-gradient(90deg, #00d4ff, #a855f7); }
.buy-ana::before { background: linear-gradient(90deg, #ff4060, #f0c040); }
.buy-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.buy-type { font-family: 'Oswald'; font-size: 0.8em; letter-spacing: 2px; padding: 3px 12px; border-radius: 6px; }
.bt-hon { background: rgba(240,192,64,0.15); color: #f0c040 !important; }
.bt-hir { background: rgba(0,212,255,0.15); color: #00d4ff !important; }
.bt-ana { background: rgba(255,64,96,0.15); color: #ff4060 !important; }
.buy-conf { font-family: 'Oswald'; font-size: 0.82em; color: #6a6a80 !important; }
.buy-row {
    display: flex; align-items: center; gap: 10px; padding: 9px 12px;
    background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 6px;
}
.buy-lbl { font-size: 0.7em; color: #6a6a80 !important; font-family: 'Oswald'; min-width: 55px; }
.buy-horses { font-weight: 700; font-size: 1em; flex: 1; }
.buy-note {
    font-size: 0.73em; color: #6a6a80 !important; padding: 8px 12px;
    background: rgba(255,255,255,0.02); border-radius: 8px;
    border-left: 2px solid rgba(255,255,255,0.08); margin-top: 4px;
}

/* Invest Bar */
.inv-bar { display: flex; height: 30px; border-radius: 10px; overflow: hidden; margin: 10px 0 8px; }
.inv-seg {
    display: flex; align-items: center; justify-content: center;
    font-family: 'Oswald'; font-size: 0.78em; font-weight: 600;
}
.inv-hon { background: linear-gradient(90deg, #c89020, #f0c040); color: #000 !important; }
.inv-hir { background: linear-gradient(90deg, #0090b0, #00d4ff); color: #000 !important; }
.inv-ana { background: linear-gradient(90deg, #c03050, #ff4060); color: #fff !important; }
.inv-legend { display: flex; gap: 16px; justify-content: center; }
.inv-legend span { font-size: 0.7em; color: #6a6a80 !important; }

/* Table */
.htable { width: 100%; border-collapse: separate; border-spacing: 0 5px; }
.htable th { font-size: 0.62em; color: #6a6a80 !important; letter-spacing: 1px; padding: 6px 3px; text-align: center; font-weight: 500; }
.htable td { background: #12121c; padding: 9px 5px; text-align: center; font-size: 0.78em; }
.htable tr td:first-child { border-radius: 8px 0 0 8px; }
.htable tr td:last-child { border-radius: 0 8px 8px 0; }
.trank { font-family: 'Oswald'; font-weight: 700; font-size: 1.05em; }
.tname { font-weight: 700; text-align: left; white-space: nowrap; }
.tscore { font-family: 'Oswald'; font-weight: 600; }

.disclaimer {
    margin: 20px 0; padding: 12px; border-radius: 10px;
    background: rgba(255,64,96,0.06); border: 1px solid rgba(255,64,96,0.15);
    font-size: 0.7em; color: #6a6a80 !important; text-align: center;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===== Model & Data =====
@st.cache_resource
def load_model():
    import os, gzip
    for fname in ["keiba_model_v2.pkl.gz", "keiba_model_v2_pkl.gz", "keiba_model_v2.pkl", "keiba_model.pkl.gz"]:
        if os.path.exists(fname):
            try:
                if fname.endswith('.gz'):
                    with gzip.open(fname, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(fname, "rb") as f:
                        data = pickle.load(f)
                if isinstance(data, dict):
                    return data
                return {'model': data, 'features': None, 'version': 'v1'}
            except:
                continue
    with open("keiba_model.pkl", "rb") as f:
        return {'model': pickle.load(f), 'features': None, 'version': 'v1'}

@st.cache_resource
def load_jockey_wr():
    with open("jockey_wr.json", "r", encoding="utf-8") as f:
        return json.load(f)

_loaded = load_model()
if isinstance(_loaded, dict):
    model = _loaded['model']
    model_features = _loaded.get('features', None)
    model_version = _loaded.get('version', 'v1')
else:
    model = _loaded
    model_features = None
    model_version = 'v1'
jockey_wr = load_jockey_wr()

FEATURES_V1 = [
    '馬体重', '場体重増減', '斤量', '馬齢', '距離(m)',
    '競馬場コード_enc', '芝ダート_enc', '馬場状態_enc',
    '性別_enc', '騎手勝率', '前走着順'
]
FEATURES_V2 = [
    '馬体重', '場体重増減', '斤量', '馬齢', '距離(m)',
    '競馬場コード_enc', '芝ダート_enc', '馬場状態_enc',
    '性別_enc', '騎手勝率', '前走着順',
    '枠番', '馬番', '頭数', '斤量平均差',
    '距離カテゴリ', '体重カテゴリ', '体重変動abs',
    '年齢性別', '距離馬場', '枠位置',
    '月', '季節', '枠馬場', '馬齢グループ'
]
FEATURES = model_features if model_features else FEATURES_V1
COURSE_MAP = {
    '札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,'中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9,
    # 地方
    '大井':10,'川崎':11,'船橋':12,'浦和':13,'園田':14,'姫路':15,'門別':16,
    '盛岡':17,'水沢':18,'金沢':19,'笠松':20,'名古屋':21,'高知':22,'佐賀':23,
    '帯広':24,'旭川':25,
}
SURFACE_MAP = {'芝':0,'ダ':1,'障':2}
COND_MAP = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}
SEX_MAP = {'牡':0,'牝':1,'セ':2,'騸':2}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

MODERN_JOCKEY_WR = {
    'ルメール':0.220,'C.ルメール':0.220,'川田将雅':0.210,'川田':0.210,'武豊':0.171,
    '戸崎圭太':0.140,'戸崎圭':0.140,'横山武史':0.130,'横山武':0.130,
    '松山弘平':0.120,'松山弘':0.120,'松山':0.120,'池添謙一':0.110,'池添謙':0.110,
    '福永祐一':0.160,'福永祐':0.160,'岩田望来':0.100,'岩田望':0.100,
    '岩田康誠':0.090,'岩田康':0.090,'吉田隼人':0.090,'吉田隼':0.090,
    '三浦皇成':0.080,'三浦皇':0.080,'三浦':0.110,'田辺裕信':0.090,'田辺裕':0.090,'田辺':0.090,
    '横山和生':0.100,'横山和':0.100,'横山典弘':0.121,'横山典':0.121,
    '大野拓弥':0.070,'大野拓':0.070,'大野':0.070,'菅原明良':0.080,'菅原明':0.080,'菅原':0.080,
    '北村宏司':0.070,'北村宏':0.070,'北村友一':0.080,'北村友':0.080,
    '丹内祐次':0.060,'丹内祐':0.060,'柴田大知':0.060,'柴田大':0.060,
    '石橋脩':0.080,'木幡巧也':0.060,'木幡巧':0.060,
    'M.デムーロ':0.140,'デムーロ':0.140,'R.ムーア':0.250,'ムーア':0.250,
    '坂井瑠星':0.100,'坂井瑠':0.100,'鮫島克駿':0.080,'鮫島克':0.080,
    '西村淳也':0.090,'西村淳':0.090,'角田大和':0.070,'角田大':0.070,
    '佐々木大':0.080,'佐々木':0.080,'津村明秀':0.070,'津村明':0.070,'津村':0.070,
    '団野大成':0.070,'団野大':0.070,'藤岡佑介':0.080,'藤岡佑':0.080,
    '幸英明':0.060,'和田竜二':0.070,'浜中俊':0.090,
    '柴田善臣':0.103,'柴田善':0.103,'内田博幸':0.060,'内田博':0.060,
    '江田照男':0.081,'江田照':0.081,'丸山元気':0.070,'丸山元':0.070,
    '永野猛蔵':0.060,'野中悠太':0.050,'野中':0.050,
    '原優介':0.050,'長浜鷹人':0.040,'長浜':0.040,
    '荻野極':0.050,'荻野':0.050,'松岡正海':0.060,'松岡':0.060,
    '吉田豊':0.060,'吉田':0.060,
    # 地方トップ騎手
    '森泰斗':0.180,'森泰':0.180,'矢野貴之':0.170,'矢野貴':0.170,'矢野':0.170,
    '御神本訓史':0.160,'御神本':0.160,'笹川翼':0.150,'笹川':0.150,
    '本橋孝太':0.120,'本橋':0.120,'真島大輔':0.110,'真島大':0.110,'真島':0.110,
    '張田昂':0.130,'張田':0.130,'吉原寛人':0.140,'吉原寛':0.140,'吉原':0.140,
    '山崎誠士':0.120,'山崎誠':0.120,'藤田凌':0.110,'藤田':0.110,
    '瀧川寿希也':0.100,'瀧川':0.100,'今野忠成':0.110,'今野':0.110,
    '的場文男':0.090,'的場':0.090,'繁田健一':0.100,'繁田':0.100,
    '岡村健司':0.100,'岡村健':0.100,'岡村':0.100,
    '町田直希':0.090,'町田':0.090,'左海誠二':0.090,'左海':0.090,
    '川島正太郎':0.100,'川島正':0.100,
    '赤岡修次':0.150,'赤岡':0.150,'永森大智':0.130,'永森':0.130,
    '下原理':0.120,'下原':0.120,'田中学':0.110,'田中':0.110,
    '山口勲':0.140,'山口':0.140,'石崎駿':0.100,'石崎':0.100,
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

SIRE_APT = {
    'ディープインパクト':{'turf':1.0,'dirt':0.3,'sprint':0.5,'mile':0.9,'mid':1.0,'long':0.8},
    'キングカメハメハ':{'turf':0.8,'dirt':0.7,'sprint':0.6,'mile':0.8,'mid':0.9,'long':0.7},
    'ロードカナロア':{'turf':0.9,'dirt':0.5,'sprint':1.0,'mile':0.8,'mid':0.5,'long':0.2},
    'ドゥラメンテ':{'turf':0.9,'dirt':0.5,'sprint':0.4,'mile':0.8,'mid':1.0,'long':0.8},
    'エピファネイア':{'turf':0.9,'dirt':0.4,'sprint':0.3,'mile':0.7,'mid':1.0,'long':0.9},
    'ハーツクライ':{'turf':0.9,'dirt':0.3,'sprint':0.2,'mile':0.6,'mid':0.9,'long':1.0},
    'キタサンブラック':{'turf':0.9,'dirt':0.4,'sprint':0.3,'mile':0.7,'mid':0.9,'long':1.0},
    'モーリス':{'turf':0.8,'dirt':0.5,'sprint':0.5,'mile':0.9,'mid':0.8,'long':0.5},
    'オルフェーヴル':{'turf':0.8,'dirt':0.6,'sprint':0.3,'mile':0.6,'mid':0.9,'long':1.0},
    'ヘニーヒューズ':{'turf':0.2,'dirt':1.0,'sprint':1.0,'mile':0.7,'mid':0.4,'long':0.1},
    'ダイワメジャー':{'turf':0.8,'dirt':0.5,'sprint':0.7,'mile':1.0,'mid':0.6,'long':0.3},
    'ルーラーシップ':{'turf':0.8,'dirt':0.5,'sprint':0.3,'mile':0.7,'mid':0.9,'long':0.9},
    'ホッコータルマエ':{'turf':0.2,'dirt':1.0,'sprint':0.7,'mile':0.9,'mid':0.8,'long':0.4},
    'シニスターミニスター':{'turf':0.1,'dirt':1.0,'sprint':0.8,'mile':0.9,'mid':0.6,'long':0.3},
    'パイロ':{'turf':0.1,'dirt':1.0,'sprint':0.9,'mile':0.8,'mid':0.5,'long':0.2},
    'コントレイル':{'turf':0.9,'dirt':0.3,'sprint':0.3,'mile':0.7,'mid':1.0,'long':0.9},
    'ドレフォン':{'turf':0.5,'dirt':0.8,'sprint':0.8,'mile':0.8,'mid':0.6,'long':0.3},
    'サウスヴィグラス':{'turf':0.1,'dirt':1.0,'sprint':1.0,'mile':0.6,'mid':0.2,'long':0.0},
    'ブリックスアンドモルタル':{'turf':0.8,'dirt':0.5,'sprint':0.4,'mile':0.8,'mid':0.9,'long':0.7},
    'スワーヴリチャード':{'turf':0.8,'dirt':0.5,'sprint':0.3,'mile':0.7,'mid':0.9,'long':0.9},
}

def calc_sire_score(father, surface, distance):
    apt = SIRE_APT.get(father)
    if not apt:
        return 0.5
    ss = apt.get('turf',0.5) if surface == '芝' else apt.get('dirt',0.5)
    if distance <= 1400: ds = apt.get('sprint',0.5)
    elif distance <= 1800: ds = apt.get('mile',0.5)
    elif distance <= 2200: ds = apt.get('mid',0.5)
    else: ds = apt.get('long',0.5)
    return ss * 0.5 + ds * 0.5

# ===== Horse Stats =====
def get_horse_stats(horse_id, target_distance, target_surface, target_course=""):
    result = {
        'last_finish': 5, 'dist_apt': 0.5, 'surf_apt': 0.5, 'pop_score': 0.5,
        'course_apt': 0.5, 'interval_days': 30, 'running_style': 0,
        'avg_agari': 35.5, 'father': '', 'fukusho_rate': 0.0,
        'margin_text': '', 'weight_diff': 0, 'prev_jockey': '',
    }
    try:
        url = "https://db.netkeiba.com/horse/result/" + horse_id + "/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        # Father
        prof = soup.find("table", class_="db_prof_table")
        if prof:
            for td in prof.find_all("td"):
                a = td.find("a", href=re.compile(r"/horse/sire/"))
                if a:
                    result['father'] = a.get_text(strip=True)
                    break
        if not result['father']:
            bt = soup.find("table", summary=re.compile(".*血統.*"))
            if bt:
                aa = bt.find_all("a", href=re.compile(r"/horse/"))
                if aa:
                    result['father'] = aa[0].get_text(strip=True)
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
        best_time_sec = 999.9  # 同距離ベストタイム（秒）
        best_time_date = ""
        best_time_dist = 0
        best_time_str = ""
        for ri, row in enumerate(rows):
            tds = row.find_all("td")
            if len(tds) < 15:
                continue
            ft = tds[11].get_text(strip=True)
            if not ft.isdigit():
                continue
            finish = int(ft)
            finish_list.append(finish)
            if ri == 0:
                result['last_finish'] = finish
                if len(tds) > 18:
                    result['margin_text'] = tds[18].get_text(strip=True)
                if len(tds) > 12:
                    result['prev_jockey'] = tds[12].get_text(strip=True)
            # Date
            dm = re.search(r'(\d{4})/(\d{1,2})/(\d{1,2})', tds[0].get_text(strip=True))
            if dm:
                try:
                    race_dates.append(datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3))))
                except: pass
            # Pop
            pt = tds[10].get_text(strip=True)
            if pt.isdigit():
                pop_list.append(int(pt))
            # Distance/Surface
            dc = tds[14].get_text(strip=True)
            ddm = re.match(r'([芝ダ障])(\d+)', dc)
            if ddm:
                sc, dv = ddm.group(1), int(ddm.group(2))
                if target_distance > 0 and abs(dv - target_distance) <= 200:
                    dist_results.append(finish)
                    # タイム取得: "1:55.2" パターンのみ（分:秒.コンマ）
                    # 短距離(~1000m)でも最低50秒以上のはずなので、"XX.X"形式は除外
                    for td in tds:
                        tt = td.get_text(strip=True)
                        # "1:55.2" or "2:05.3" 形式（1分以上）
                        tm = re.match(r'^(\d):(\d{2})\.(\d)$', tt)
                        if tm:
                            secs = int(tm.group(1))*60 + int(tm.group(2)) + int(tm.group(3))*0.1
                            # 距離に応じた妥当性チェック（1000mで55秒、2400mで150秒くらい）
                            min_time = dv * 0.05
                            max_time = dv * 0.09
                            if min_time < secs < max_time and secs < best_time_sec:
                                best_time_sec = secs
                                best_time_str = tt
                                best_time_dist = dv
                                dt = tds[0].get_text(strip=True)
                                dtm = re.search(r'(\d{4}/\d{1,2}/\d{1,2})', dt)
                                best_time_date = dtm.group(1) if dtm else ""
                            break
                sn = '芝' if sc == '芝' else 'ダ'
                if sn == target_surface:
                    surf_results.append(finish)
            # Course
            if target_course and len(tds) > 1:
                if target_course in tds[1].get_text(strip=True):
                    course_results.append(finish)
            # Pass & Agari: scan all tds for patterns
            found_pass = False
            found_agari = False
            for tdi, td in enumerate(tds):
                txt = td.get_text(strip=True)
                # Pass: "3-3-2-1" / "03-03" / "3－3－2" (zenkaku) / "3 - 3 - 2"
                if not found_pass:
                    cleaned = txt.replace(' ', '').replace('-', '-').replace('－', '-')
                    if re.match(r'^\d{1,2}-\d{1,2}(-\d{1,2})*$', cleaned):
                        pn = re.findall(r'\d+', cleaned)
                        if pn and len(pn) >= 2:
                            pass_list.append(int(pn[0]))
                            found_pass = True
                # Agari: "35.8" / "339" (=33.9) etc
                if not found_agari and tdi >= 10:
                    cleaned_a = txt.strip()
                    if re.match(r'^\d{2}\.\d{1,2}$', cleaned_a):
                        try:
                            av = float(cleaned_a)
                            if 30.0 < av < 45.0:
                                agari_list.append(av)
                                found_agari = True
                        except: pass
            if ri >= 9:
                break
        # Calc scores
        def to_score(lst):
            if not lst: return 0.5
            return max(0.0, min(1.0, 1.0 - (sum(lst)/len(lst) - 1) / 17.0))
        result['dist_apt'] = to_score(dist_results)
        result['surf_apt'] = to_score(surf_results)
        result['course_apt'] = to_score(course_results)
        if pop_list:
            result['pop_score'] = max(0.0, min(1.0, 1.0 - (sum(pop_list)/len(pop_list) - 1) / 17.0))
        if race_dates:
            diff = (datetime.now() - race_dates[0]).days
            if diff > 0:
                result['interval_days'] = diff
        if pass_list:
            ap = sum(pass_list) / len(pass_list)
            if ap <= 2.0: result['running_style'] = 1
            elif ap <= 5.0: result['running_style'] = 2
            elif ap <= 10.0: result['running_style'] = 3
            else: result['running_style'] = 4
        if agari_list:
            result['avg_agari'] = sum(agari_list) / len(agari_list)
        if finish_list:
            result['fukusho_rate'] = sum(1 for f in finish_list if f <= 3) / len(finish_list)
        result['best_time'] = best_time_sec if best_time_sec < 999.0 else 0.0
        result['best_time_str'] = best_time_str
        result['best_time_date'] = best_time_date
        result['best_time_dist'] = best_time_dist
        if not finish_list:
            at = soup.get_text()
            m = re.search(r'(\d{1,2})\(\d+人気\)', at)
            if m:
                result['last_finish'] = int(m.group(1))
    except Exception:
        pass
    return result

# ===== Parse Shutuba =====
def parse_shutuba(race_id, is_nar=False):
    if is_nar:
        url = "https://nar.netkeiba.com/race/shutuba.html?race_id=" + race_id
    else:
        url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + race_id
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")
    race_name = "レース"
    tag = soup.find("div", class_="RaceName")
    if tag and tag.get_text(strip=True):
        race_name = tag.get_text(strip=True)
    else:
        tag = soup.find("title")
        if tag:
            m = re.search(r'(\S+\d+R)', tag.get_text(strip=True))
            if m: race_name = m.group(1)
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
    # 地方の場合: RaceData02がない時はtitleやh1から探す
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
    for row in rows:
        rc = row.get("class", [])
        if "Cancel" in rc: continue
        # Waku/Umaban
        waku, umaban = 0, 0
        wt = row.select_one("td.Waku span")
        if wt:
            w = wt.get_text(strip=True)
            if w.isdigit(): waku = int(w)
        ut = row.select_one("td.Umaban")
        if ut:
            u = ut.get_text(strip=True)
            if u.isdigit(): umaban = int(u)
        # Fallback 1: td with class containing 'Num' or 'num'
        if umaban == 0:
            for td in row.find_all("td"):
                cls = " ".join(td.get("class", []))
                if "Num" in cls or "num" in cls or "Umaban" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
        # Fallback 2: second td in row (typically waku=1st, umaban=2nd)
        if umaban == 0:
            all_tds = row.find_all("td")
            for tdi in range(min(3, len(all_tds))):
                t = all_tds[tdi].get_text(strip=True)
                if t.isdigit() and 1 <= int(t) <= 18:
                    if waku == 0:
                        waku = int(t)
                    elif umaban == 0:
                        umaban = int(t)
                        break
        # Fallback 3: use index+1 as umaban
        if umaban == 0:
            umaban = len(horses) + 1
        nt = row.select_one("span.HorseName a")
        if not nt: continue
        horse_name = nt.get_text(strip=True)
        href = nt.get("href", "")
        hm = re.search(r'/horse/(\d+)', href)
        horse_id = hm.group(1) if hm else None
        # Sex/Age
        it = row.select_one("td.Barei") or row.select_one("span.Barei")
        sa = it.get_text(strip=True) if it else ""
        if not sa:
            for td in row.find_all("td"):
                t = td.get_text(strip=True)
                if re.match(r'^[牡牝セ騸]\d+$', t):
                    sa = t; break
        sex = sa[0] if sa else '牡'
        age = int(sa[1:]) if sa and sa[1:].isdigit() else 3
        # Kinryo
        kinryo = 55.0
        for td in row.find_all("td"):
            try:
                v = float(td.get_text(strip=True))
                if 48.0 <= v <= 62.0: kinryo = v; break
            except: continue
        # Jockey
        jt = row.select_one("td.Jockey a") or row.select_one("a[href*='jockey']")
        jockey_name = jt.get_text(strip=True) if jt else ""
        # Weight
        horse_weight, weight_diff = 480, 0
        for td in row.find_all("td"):
            bm = re.search(r'(\d{3,})\(([\+\-]?\d+)\)', td.get_text(strip=True))
            if bm:
                w = int(bm.group(1))
                if 350 <= w <= 600:
                    horse_weight = w
                    weight_diff = int(bm.group(2))
                    break
        # Trainer
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

# ===== Pace Advantage =====
def calc_pace_advantage(distance, surface, condition, num_horses, is_nar=False):
    """Returns dict: {1:逃げ, 2:先行, 3:差し, 4:追込} -> (rank, reason)"""
    scores = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    reasons = {1: "", 2: "", 3: "", 4: ""}
    # NAR boost: 地方は小回りで逃げ・先行が圧倒的有利
    if is_nar:
        scores[1] += 0.15; scores[2] += 0.2; scores[3] -= 0.05; scores[4] -= 0.2
        reasons[1] = "地方小回り逃げ有利"
        reasons[2] = "地方先行圧倒的有利"
    # Base by surface/distance
    if surface == 'ダ':
        scores[1] += 0.1; scores[2] += 0.15; scores[3] -= 0.05; scores[4] -= 0.15
        reasons[2] = "ダート先行有利"
    else:
        if distance <= 1400:
            scores[1] += 0.15; scores[2] += 0.1; scores[3] -= 0.05; scores[4] -= 0.2
            reasons[1] = "芝短距離逃げ有利"
        elif distance >= 2400:
            scores[3] += 0.1; scores[4] += 0.05; scores[1] -= 0.1
            reasons[3] = "芝長距離差し有利"
    # Condition
    if condition in ['重', '不良', '不']:
        scores[1] += 0.05; scores[2] += 0.08; scores[4] -= 0.1
        reasons[2] += " 重馬場前残り" if reasons[2] else "重馬場前残り"
    elif condition in ['稍', '稍重']:
        scores[2] += 0.04; scores[4] -= 0.05
        if not reasons[2]: reasons[2] = "稍重で前残り"
    # Num horses
    if num_horses <= 10:
        scores[1] += 0.1; scores[2] += 0.05; scores[4] -= 0.1
        if not reasons[1]: reasons[1] = "少頭数逃げ有利"
    elif num_horses >= 16:
        scores[3] += 0.08; scores[4] += 0.05; scores[1] -= 0.05
        if not reasons[3]: reasons[3] = "多頭数で届く"
    # Rank
    sorted_s = sorted(scores.items(), key=lambda x: -x[1])
    rank_map = {}
    labels = ['◎', '○', '△', '×']
    css_labels = ['best', 'good', 'fair', 'bad']
    for i, (style, sc) in enumerate(sorted_s):
        rank_map[style] = (labels[i], css_labels[i], reasons.get(style, ""))
    return rank_map, scores

STYLE_NAMES = {0: '不明', 1: '逃げ', 2: '先行', 3: '差し', 4: '追込'}
STYLE_CSS = {0: 'st-senk', 1: 'st-nige', 2: 'st-senk', 3: 'st-sasi', 4: 'st-oiko'}

# ===== Render Functions =====
def render_pace_panel(rank_map):
    html = '<div class="pace-panel"><div class="pace-title">&#9889; PACE ADVANTAGE</div><div class="pace-grid">'
    for style_id in [1, 2, 3, 4]:
        sname = STYLE_NAMES[style_id]
        rank, css, reason = rank_map.get(style_id, ('△', 'fair', ''))
        if not reason: reason = "-"
        html += f'<div class="pace-item pace-{css}"><div class="ps">{sname}</div>'
        html += f'<div class="pr">{rank}</div>'
        html += f'<div class="preason">{reason}</div></div>'
    html += '</div></div>'
    return html

def get_pace_match_html(running_style, rank_map):
    if running_style == 0:
        return '<span class="pmatch pm-fair">-</span>'
    rank, css, _ = rank_map.get(running_style, ('△', 'fair', ''))
    return f'<span class="pmatch pm-{css}">{rank}</span>'

def weight_html(diff):
    if diff == 0: return '<span class="wbadge w-flat">&plusmn;0</span>'
    if diff > 0:
        cls = 'w-danger' if abs(diff) >= 10 else 'w-plus'
        return f'<span class="wbadge {cls}">+{diff}</span>'
    cls = 'w-danger' if abs(diff) >= 10 else 'w-minus'
    return f'<span class="wbadge {cls}">{diff}</span>'

def interval_text(days):
    weeks = days // 7
    if weeks <= 0: return "連闘"
    if weeks <= 8: return f"中{weeks}週"
    months = days // 30
    return f"{months}ヶ月"

def interval_cls(days):
    if 14 <= days <= 35: return 'sg'
    if 7 <= days < 14: return 'sw'
    if 35 < days <= 63: return 'sw'
    return 'sr'

def finish_cls(f):
    if f <= 2: return 'sg'
    if f <= 5: return 'sw'
    return 'sr'

def render_horse_card(rank, h, max_score, rank_map):
    colors = {1: ('g', 'gold'), 2: ('s', 'silver'), 3: ('b', 'bronze')}
    c, _ = colors.get(rank, ('b', 'bronze'))
    style_name = STYLE_NAMES.get(h.get('脚質', 0), '不明')
    style_css = STYLE_CSS.get(h.get('脚質', 0), 'st-senk')
    pm_html = get_pace_match_html(h.get('脚質', 0), rank_map)
    pct = int(h['スコア'] / max_score * 100) if max_score > 0 else 0
    sex_age = h.get('性別', '牡') + str(h.get('馬齢', 3))
    fr = h.get('複勝率', 0)
    father = h.get('父', '')
    html = f'<div class="hcard hcard-{c}"><div class="hcard-top">'
    html += f'<div class="hrank hrank-{c}">{rank}</div>'
    html += f'<div style="flex:1"><div class="hname">{h["馬名"]}</div>'
    html += f'<div class="hjockey">🏇 {h["騎手名"]} ｜ {sex_age} ｜ {h["斤量"]}kg</div></div>'
    html += f'<div class="hscore hscore-{c}">{h["スコア"]:.3f}</div></div>'
    # Stats
    html += '<div class="sgrid">'
    html += f'<div class="sitem"><div class="slbl">前走</div><div class="sval {finish_cls(h["前走着順"])}">{h["前走着順"]}着</div></div>'
    html += f'<div class="sitem"><div class="slbl">脚質</div><div class="sval"><span class="stag {style_css}">{style_name}</span>{pm_html}</div></div>'
    html += f'<div class="sitem"><div class="slbl">間隔</div><div class="sval {interval_cls(h.get("前走間隔",30))}">{interval_text(h.get("前走間隔",30))}</div></div>'
    html += f'<div class="sitem"><div class="slbl">体重</div><div class="sval">{weight_html(h.get("場体重増減",0))}</div></div>'
    html += f'<div class="sitem"><div class="slbl">上がり</div><div class="sval">{h.get("上がり3F",35.5):.1f}</div></div>'
    html += '</div>'
    # Tags
    html += '<div class="tagrow">'
    if father: html += f'<span class="tag tag-sire">父: {father}</span>'
    if fr > 0: html += f'<span class="tag">複勝率 {int(fr*100)}%</span>'
    bt_str = h.get('タイム表示', '')
    bt_date = h.get('タイム日付', '')
    bt_dist = h.get('タイム距離', 0)
    if bt_str:
        date_short = bt_date.replace('/', '.') if bt_date else ''
        html += f'<span class="tag" style="border:1px solid rgba(240,192,64,0.3);color:#f0c040 !important">⏱ {bt_dist}m {bt_str} ({date_short})</span>'
    html += '</div>'
    html += f'<div class="sbar-w"><div class="sbar sbar-{c}" style="width:{pct}%"></div></div></div>'
    return html

def render_buy_section(df, race_info, rank_map):
    sorted_df = df.sort_values('スコア', ascending=False).reset_index(drop=True)
    top = sorted_df.head(6)
    t1 = top.iloc[0]; t2 = top.iloc[1]; t3 = top.iloc[2]
    t4 = top.iloc[3] if len(top) > 3 else None
    t5 = top.iloc[4] if len(top) > 4 else None
    t6 = top.iloc[5] if len(top) > 5 else None
    def horse_label(h):
        num = int(h['馬番'])
        name = h['馬名'][:5]
        if num > 0:
            return f'{num} {name}'
        return name
    n1 = horse_label(t1); n2 = horse_label(t2); n3 = horse_label(t3)
    def horse_num(h):
        n = int(h['馬番'])
        return str(n) if n > 0 else h['馬名'][:3]
    # Confidence
    gap12 = t1['スコア'] - t2['スコア']
    conf_stars = '★★★★' if gap12 > 0.03 else ('★★★' if gap12 > 0.015 else '★★')
    # Honmei
    html = '<div class="buy-card buy-honmei"><div class="buy-header">'
    html += '<span class="buy-type bt-hon">&#128293; 本命</span>'
    html += f'<span class="buy-conf">信頼度 {conf_stars}</span></div>'
    html += f'<div class="buy-row"><span class="buy-lbl">ワイド</span><span class="buy-horses">{n1} ― {n3}</span></div>'
    html += f'<div class="buy-row"><span class="buy-lbl">馬連</span><span class="buy-horses">{n1} ― {n2}</span></div>'
    # Reason
    s1_style = STYLE_NAMES.get(int(t1.get('脚質', 0)), '不明')
    s1_match = rank_map.get(int(t1.get('脚質', 0)), ('△','fair',''))[0]
    html += f'<div class="buy-note">TOP1 {t1["馬名"]}は{s1_style}({s1_match})で展開向き。前走{int(t1["前走着順"])}着の安定感。</div>'
    html += '</div>'
    # Hiroku
    himo = []
    if t4 is not None: himo.append(horse_num(t4))
    if t5 is not None: himo.append(horse_num(t5))
    if t6 is not None: himo.append(horse_num(t6))
    html += '<div class="buy-card buy-hiroku"><div class="buy-header">'
    html += '<span class="buy-type bt-hir">&#9889; 手広く</span>'
    html += '<span class="buy-conf">信頼度 ★★★</span></div>'
    html += f'<div class="buy-row"><span class="buy-lbl">3連複</span><span class="buy-horses">{horse_num(t1)} ― {horse_num(t2)},{horse_num(t3)} ― {",".join(himo)}</span></div>'
    pt = len(himo) * 2 if himo else 3
    html += f'<div class="buy-note">軸{t1["馬名"]}固定。2着候補にTOP2-3、3着候補にTOP4-6。計{pt}点</div></div>'
    # Ana - find best value horse (lower ranked but pace match)
    ana_horse = None
    for i in range(5, min(len(sorted_df), 12)):
        h = sorted_df.iloc[i]
        rs = int(h.get('脚質', 0))
        if rs > 0:
            rm = rank_map.get(rs, ('△','fair',''))
            if rm[0] in ['◎', '○']:
                ana_horse = h
                break
    if ana_horse is not None:
        ana_name = ana_horse['馬名']
        ana_num = horse_num(ana_horse)
        ana_style = STYLE_NAMES.get(int(ana_horse.get('脚質',0)), '')
        ana_match = rank_map.get(int(ana_horse.get('脚質',0)), ('△','',''))[0]
        html += '<div class="buy-card buy-ana"><div class="buy-header">'
        html += '<span class="buy-type bt-ana">&#128142; 穴狙い</span>'
        html += '<span class="buy-conf">信頼度 ★★</span></div>'
        html += f'<div class="buy-row"><span class="buy-lbl">単勝</span><span class="buy-horses" style="color:#ff4060">{ana_num} {ana_name}</span></div>'
        html += f'<div class="buy-row"><span class="buy-lbl">ワイド</span><span class="buy-horses">{ana_num} ― {horse_num(t1)},{horse_num(t3)}</span></div>'
        html += f'<div class="buy-note">{ana_style}({ana_match})で展開◎。人気薄で妙味あり。</div></div>'
    # Invest bar
    html += '<div style="margin-top:14px"><div class="pace-title">&#128176; INVESTMENT BALANCE</div>'
    html += '<div class="inv-bar"><div class="inv-seg inv-hon" style="width:50%">50%</div>'
    html += '<div class="inv-seg inv-hir" style="width:30%">30%</div>'
    html += '<div class="inv-seg inv-ana" style="width:20%">20%</div></div>'
    html += '<div class="inv-legend"><span>&#9679; 本命</span><span>&#9679; 手広く</span><span>&#9679; 穴狙い</span></div></div>'
    return html

def render_table(df, rank_map):
    sorted_df = df.sort_values('AI順位')
    html = '<table class="htable"><tr><th>#</th><th>馬名</th><th>騎手</th><th>脚質</th><th>前走</th><th>間隔</th><th>体重</th><th>⏱ Best</th><th>SCORE</th></tr>'
    for _, h in sorted_df.iterrows():
        rank = int(h['AI順位'])
        rc = '#f0c040' if rank == 1 else ('#b0b8c8' if rank == 2 else ('#c87840' if rank == 3 else '#e8e8f0'))
        style_name = STYLE_NAMES.get(int(h.get('脚質', 0)), '?')
        style_css = STYLE_CSS.get(int(h.get('脚質', 0)), 'st-senk')
        pm = get_pace_match_html(int(h.get('脚質', 0)), rank_map)
        bt = h.get('タイム表示', '')
        bt_display = bt if bt else '-'
        html += f'<tr><td><span class="trank" style="color:{rc}">{rank}</span></td>'
        html += f'<td class="tname">{h["馬名"]}</td>'
        html += f'<td>{h["騎手名"][:3]}</td>'
        html += f'<td><span class="stag {style_css}">{style_name}</span>{pm}</td>'
        html += f'<td class="{finish_cls(int(h["前走着順"]))}">{int(h["前走着順"])}着</td>'
        html += f'<td class="{interval_cls(h.get("前走間隔",30))}">{interval_text(h.get("前走間隔",30))}</td>'
        html += f'<td>{weight_html(int(h.get("場体重増減",0)))}</td>'
        html += f'<td style="font-family:Oswald;font-size:0.82em">{bt_display}</td>'
        html += f'<td class="tscore" style="color:{rc}">{h["スコア"]:.3f}</td></tr>'
    html += '</table>'
    return html

# ===== MAIN =====
st.markdown('<div class="site-header"><div class="site-logo">KEIBA AI</div><div class="site-sub">PREDICTION SYSTEM</div></div>', unsafe_allow_html=True)

url_input = st.text_input("netkeibaの出馬表URLを入力（中央・地方対応）")

if st.button("🔍 予想する") and url_input:
    is_nar = "nar" in url_input
    url_input = url_input.replace("nar.sp.netkeiba.com", "nar.netkeiba.com")
    url_input = url_input.replace("race.sp.netkeiba.com", "race.netkeiba.com")
    rid_match = re.search(r'race_id=(\d+)', url_input)
    if not rid_match:
        st.error("URLからrace_idを取得できませんでした")
        st.stop()
    race_id = rid_match.group(1)
    with st.spinner("出馬表を取得中..."):
        race_name, horses, horse_ids, race_info = parse_shutuba(race_id, is_nar=is_nar)
    if not horses:
        st.error("馬データを取得できませんでした")
        st.stop()
    # Race card
    surf_badge = 'badge-turf' if race_info['surface'] == '芝' else 'badge-dirt'
    surf_icon = '🟢 TURF' if race_info['surface'] == '芝' else '🟤 DIRT'
    num_horses = len(horses)
    rc_html = f'<div class="race-card"><span class="race-badge {surf_badge}">{surf_icon}</span>'
    rc_html += f'<div class="race-name">{race_info["course"]} {race_name}</div>'
    rc_html += f'<div class="race-meta"><span>📏 {race_info["distance"]}m</span>'
    rc_html += f'<span>🏟️ {race_info["course"]}</span>'
    rc_html += f'<span>💧 {race_info["condition"]}</span>'
    rc_html += f'<span>🐎 {num_horses}頭</span></div>'
    # Pace advantage
    rank_map, pace_scores_map = calc_pace_advantage(
        race_info['distance'], race_info['surface'], race_info['condition'], num_horses, is_nar=is_nar
    )
    rc_html += render_pace_panel(rank_map)
    rc_html += '</div>'
    st.markdown(rc_html, unsafe_allow_html=True)
    # Get horse stats
    with st.spinner("各馬の成績を分析中..."):
        progress_bar = st.progress(0)
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                try:
                    stats = get_horse_stats(hid, race_info['distance'], race_info['surface'], race_info['course'])
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
                    horse['血統スコア'] = calc_sire_score(stats.get('father',''), race_info['surface'], race_info['distance'])
                    horse['持ちタイム'] = stats.get('best_time', 0.0)
                    horse['タイム表示'] = stats.get('best_time_str', '')
                    horse['タイム日付'] = stats.get('best_time_date', '')
                    horse['タイム距離'] = stats.get('best_time_dist', 0)
                except Exception:
                    horse.update({'前走着順':5,'距離適性':0.5,'馬場適性':0.5,'人気傾向':0.5,
                                  'コース適性':0.5,'前走間隔':30,'脚質':0,'上がり3F':35.5,
                                  '複勝率':0.0,'父':'','血統スコア':0.5,'持ちタイム':0.0,
                                  'タイム表示':'','タイム日付':'','タイム距離':0})
            else:
                horse.update({'前走着順':5,'距離適性':0.5,'馬場適性':0.5,'人気傾向':0.5,
                              'コース適性':0.5,'前走間隔':30,'脚質':0,'上がり3F':35.5,
                              '複勝率':0.0,'父':'','血統スコア':0.5,'持ちタイム':0.0,
                              'タイム表示':'','タイム日付':'','タイム距離':0})
            progress_bar.progress((i + 1) / num_horses)
            if i < num_horses - 1:
                time.sleep(0.5)
        progress_bar.empty()
    # Score calculation - SIMPLE 7 elements
    df = pd.DataFrame(horses)
    # === v2 新特徴量計算 ===
    if model_version == 'v2':
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
        import datetime
        now = datetime.datetime.now()
        df['月'] = now.month
        m = now.month
        df['季節'] = 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3))
        df['枠馬場'] = df['枠位置'] * 10 + df['馬場状態_enc']
        df['馬齢グループ'] = df['馬齢'].clip(2, 7)
    # === 特徴量のデフォルト値を保証 ===
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
    X = df[FEATURES].values
    proba = model.predict_proba(X)
    ai_scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, :3].sum(axis=1)
    pop_scores = df['人気傾向'].values
    apt_scores = (df['距離適性'].values + df['馬場適性'].values) / 2.0
    # Pace scores (from rank_map)
    pace_scores = []
    for _, h in df.iterrows():
        rs = int(h.get('脚質', 0))
        if rs == 0:
            pace_scores.append(0.5)
        else:
            base = pace_scores_map.get(rs, 0.5)
            pace_scores.append(max(0.0, min(1.0, base)))
    pace_scores = np.array(pace_scores)
    # Agari score
    agari_scores = np.clip(1.0 - (df['上がり3F'].values - 33.0) / 5.0, 0.0, 1.0)
    # Course scores
    course_scores = df['コース適性'].values
    # Other (blood + fukusho)
    other_scores = (df['血統スコア'].values + df['複勝率'].values) / 2.0
    # Jockey score (for NAR)
    jockey_scores = np.clip(df['騎手勝率'].values / 0.18, 0.0, 1.0)
    # Best time score (relative within race: fastest=1.0, slowest=0.0)
    times = df['持ちタイム'].values
    valid_times = times[times > 0]
    if len(valid_times) >= 2:
        t_min, t_max = valid_times.min(), valid_times.max()
        if t_max > t_min:
            time_scores = np.where(times > 0, 1.0 - (times - t_min) / (t_max - t_min), 0.5)
        else:
            time_scores = np.where(times > 0, 0.7, 0.5)
    else:
        time_scores = np.full(len(times), 0.5)
    time_scores = np.clip(time_scores, 0.0, 1.0)
    # ===== FINAL SCORE =====
    if is_nar:
        # 地方: AI 27% + 人気 13% + 脚質 18% + 上がり 8% + 騎手 10% + 適性 8% + 持ちタイム 8% + コース+他 8%
        final_scores = (
            ai_scores * 0.27 + pop_scores * 0.13 + pace_scores * 0.18
            + agari_scores * 0.08 + jockey_scores * 0.10 + apt_scores * 0.08
            + time_scores * 0.08 + course_scores * 0.04 + other_scores * 0.04
        )
    else:
        # 中央: AI 45% + 人気 15% + 適性 10% + 脚質 10% + 上がり 10% + コース 5% + 他 5%
        final_scores = (
            ai_scores * 0.45 + pop_scores * 0.15 + apt_scores * 0.10
            + pace_scores * 0.10 + agari_scores * 0.10 + course_scores * 0.05
            + other_scores * 0.05
        )
    df['スコア'] = final_scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    # Render TOP3
    st.markdown('<div class="sec-title">🏆 AI TOP 3<span class="sec-line"></span></div>', unsafe_allow_html=True)
    max_score = df['スコア'].max()
    for _, row in df.head(3).iterrows():
        st.markdown(render_horse_card(int(row['AI順位']), row, max_score, rank_map), unsafe_allow_html=True)
    # Buy section
    st.markdown('<div class="sec-title">🎯 AI推奨 買い目<span class="sec-line"></span></div>', unsafe_allow_html=True)
    st.markdown(render_buy_section(df, race_info, rank_map), unsafe_allow_html=True)
    # Chart
    st.markdown('<div class="sec-title">📊 全馬スコア<span class="sec-line"></span></div>', unsafe_allow_html=True)
    chart_df = df[['馬名', 'スコア']].copy().set_index('馬名')
    st.bar_chart(chart_df, color='#f0c040')
    # Table
    st.markdown('<div class="sec-title">📋 ALL HORSES<span class="sec-line"></span></div>', unsafe_allow_html=True)
    st.markdown(render_table(df, rank_map), unsafe_allow_html=True)
    # Disclaimer
    st.markdown('<div class="disclaimer">&#9888;&#65039; 本予想はAIによる統計分析です。馬券の購入は自己責任でお願いします。</div>', unsafe_allow_html=True)
