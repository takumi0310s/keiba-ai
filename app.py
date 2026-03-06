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

/* Model Version Badge */
.model-badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 0.7em;
    font-family: 'Oswald'; letter-spacing: 1px; margin-left: 8px;
}
.badge-v5 { background: rgba(255,215,0,0.15); color: #ffd700 !important; border: 1px solid rgba(255,215,0,0.3); }
.badge-v3 { background: rgba(0,232,123,0.15); color: #00e87b !important; border: 1px solid rgba(0,232,123,0.3); }
.badge-v2 { background: rgba(0,212,255,0.15); color: #00d4ff !important; border: 1px solid rgba(0,212,255,0.3); }
.badge-v1 { background: rgba(255,255,255,0.08); color: #6a6a80 !important; }

.disclaimer {
    margin: 20px 0; padding: 12px; border-radius: 10px;
    background: rgba(255,64,96,0.06); border: 1px solid rgba(255,64,96,0.15);
    font-size: 0.7em; color: #6a6a80 !important; text-align: center;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===== Model & Data =====
@st.cache_resource(ttl=3600)
def load_model():
    import os, gzip
    # v5 → v3 → v2 → v1 の順で検索
    for fname in [
        "keiba_model_v5.pkl.gz", "keiba_model_v5_pkl.gz", "keiba_model_v5.pkl",
        "keiba_model_v3.pkl.gz", "keiba_model_v3_pkl.gz", "keiba_model_v3.pkl",
        "keiba_model_v2.pkl.gz", "keiba_model_v2_pkl.gz", "keiba_model_v2.pkl",
        "keiba_model.pkl.gz",
    ]:
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
    try:
        with open("keiba_model.pkl", "rb") as f:
            return {'model': pickle.load(f), 'features': None, 'version': 'v1'}
    except:
        return None

@st.cache_resource
def load_jockey_wr():
    try:
        with open("jockey_wr.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

_loaded = load_model()
if isinstance(_loaded, dict):
    model = _loaded['model']
    model_features = _loaded.get('features', None)
    model_version = _loaded.get('version', 'v1')
    sire_map = _loaded.get('sire_map', {})
    bms_map = _loaded.get('bms_map', {})
    model_auc = _loaded.get('auc', 0.0)
    model_leak_free = _loaded.get('leak_free', False)
else:
    model = _loaded
    model_features = None
    model_version = 'v1'
    sire_map = {}
    bms_map = {}
    model_auc = 0.0
    model_leak_free = False
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
# v3: 学習時のAPP_FEATURES（モデル保存時のfeaturesキー）
FEATURES_V3 = [
    '馬体重', '場体重増減', '斤量', '馬齢', '距離(m)',
    '競馬場コード_enc', '芝ダート_enc', '馬場状態_enc',
    '性別_enc', '騎手勝率', '前走着順',
    '枠番', '馬番', '頭数', '斤量平均差',
    '距離カテゴリ', '体重カテゴリ', '体重変動abs',
    '年齢性別', '距離馬場', '枠位置',
    '月', '季節', '枠馬場', '馬齢グループ',
    '父馬_enc', '母父_enc', '所属_enc',
    '前走人気', '前走オッズlog', '前走上がり',
    '前走通過順1', '前走通過順4',
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

# ===== 所属判定マップ =====
TRAINER_LOCATION_MAP = {
    '美浦': 0, '栗東': 1, '美': 0, '栗': 1,
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
        'avg_agari': 35.5, 'father': '', 'mother_father': '', 'fukusho_rate': 0.0,
        'margin_text': '', 'weight_diff': 0, 'prev_jockey': '',
        'avg_pass_pos': 8.0, 'last_pass4': 8, 'last_odds': 15.0, 'last_pop': 8,
        'trainer_loc': '',
    }
    try:
        url = "https://db.netkeiba.com/horse/result/" + horse_id + "/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        # Father & Mother's Father
        prof = soup.find("table", class_="db_prof_table")
        if prof:
            for td in prof.find_all("td"):
                a = td.find("a", href=re.compile(r"/horse/sire/"))
                if a:
                    result['father'] = a.get_text(strip=True)
                    break
            # 調教師所属を取得
            for tr in prof.find_all("tr"):
                th = tr.find("th")
                if th and '調教師' in th.get_text():
                    td_text = tr.find("td").get_text(strip=True) if tr.find("td") else ""
                    if '美浦' in td_text or '(美)' in td_text:
                        result['trainer_loc'] = '美浦'
                    elif '栗東' in td_text or '(栗)' in td_text:
                        result['trainer_loc'] = '栗東'
        if not result['father']:
            bt = soup.find("table", summary=re.compile(".*血統.*"))
            if bt:
                aa = bt.find_all("a", href=re.compile(r"/horse/"))
                if aa:
                    result['father'] = aa[0].get_text(strip=True)
        # Mother's father (母の父)
        bt = soup.find("table", summary=re.compile(".*血統.*"))
        if bt:
            all_links = bt.find_all("a", href=re.compile(r"/horse/"))
            # 典型的な血統テーブル: 父, 父父, 父母, 母, 母父, 母母
            # 母の父は通常5番目（0-indexed: 4）だが構造によって異なる
            # テキストで"母の父"を探す方が確実
            for tr in bt.find_all("tr"):
                tds = tr.find_all("td")
                for td in tds:
                    text = td.get_text(strip=True)
                    if text and td.find("a", href=re.compile(r"/horse/")):
                        # 母の父を特定するのは複雑なので、all_linksの4番目を使う
                        pass
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
        best_time_sec = 999.9
        best_time_date = ""
        best_time_dist = 0
        best_time_str = ""
        today_date = datetime.now().date()
        for ri, row in enumerate(rows):
            tds = row.find_all("td")
            if len(tds) < 15:
                continue
            ft = tds[11].get_text(strip=True)
            if not ft.isdigit():
                continue
            finish = int(ft)

            # この行のレース日付を取得
            row_date = None
            for td_idx in range(min(5, len(tds))):
                td_text = tds[td_idx].get_text(strip=True)
                dm = re.search(r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', td_text)
                if dm:
                    try:
                        y, m_val, d_val = int(dm.group(1)), int(dm.group(2)), int(dm.group(3))
                        if 2000 <= y <= 2030 and 1 <= m_val <= 12 and 1 <= d_val <= 31:
                            row_date = datetime(y, m_val, d_val)
                    except: pass
                if row_date: break

            # 今日以降のレース結果はスキップ
            if row_date and row_date.date() >= today_date:
                continue

            # ここから先は過去レースのみ
            finish_list.append(finish)
            if row_date:
                race_dates.append(row_date)

            if len(finish_list) == 1:
                result['last_finish'] = finish
                if len(tds) > 18:
                    result['margin_text'] = tds[18].get_text(strip=True)
                if len(tds) > 12:
                    result['prev_jockey'] = tds[12].get_text(strip=True)
            # Pop
            pt = tds[10].get_text(strip=True)
            if pt.isdigit():
                pop_list.append(int(pt))
            # Odds (単勝オッズ) - 通常9番目のtd
            if len(tds) > 9:
                odds_text = tds[9].get_text(strip=True)
                try:
                    odds_val = float(odds_text)
                    if 1.0 <= odds_val <= 999.9:
                        odds_list.append(odds_val)
                except: pass
            # Distance/Surface
            dc = tds[14].get_text(strip=True)
            ddm = re.match(r'([芝ダ障])(\d+)', dc)
            if ddm:
                sc, dv = ddm.group(1), int(ddm.group(2))
                if target_distance > 0 and abs(dv - target_distance) <= 200:
                    dist_results.append(finish)
                    for td in tds:
                        tt = td.get_text(strip=True)
                        tm = re.match(r'^(\d):(\d{2})\.(\d)$', tt)
                        if tm:
                            secs = int(tm.group(1))*60 + int(tm.group(2)) + int(tm.group(3))*0.1
                            min_time = dv * 0.05
                            max_time = dv * 0.09
                            if min_time < secs < max_time and secs < best_time_sec:
                                best_time_sec = secs
                                best_time_str = tt
                                best_time_dist = dv
                                best_time_date = ""
                                for td_idx2 in range(min(5, len(tds))):
                                    td_text2 = tds[td_idx2].get_text(strip=True)
                                    dtm = re.search(r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', td_text2)
                                    if dtm:
                                        y2 = int(dtm.group(1))
                                        if 2000 <= y2 <= 2030:
                                            best_time_date = dtm.group(0)
                                            break
                                    dtm2 = re.search(r'(\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})', td_text2)
                                    if dtm2:
                                        yr2 = int(dtm2.group(1))
                                        yr2 = yr2 + 2000 if yr2 < 80 else yr2 + 1900
                                        best_time_date = f"{yr2}/{dtm2.group(2)}/{dtm2.group(3)}"
                                        break
                            break
                sn = '芝' if sc == '芝' else 'ダ'
                if sn == target_surface:
                    surf_results.append(finish)
            # Course
            if target_course and len(tds) > 1:
                if target_course in tds[1].get_text(strip=True):
                    course_results.append(finish)
            # Pass & Agari
            found_pass = False
            found_agari = False
            for tdi, td in enumerate(tds):
                txt = td.get_text(strip=True)
                if not found_pass:
                    cleaned = txt.replace(' ', '').replace('-', '-').replace('－', '-')
                    if re.match(r'^\d{1,2}-\d{1,2}(-\d{1,2})*$', cleaned):
                        pn = re.findall(r'\d+', cleaned)
                        if pn and len(pn) >= 2:
                            pass_list.append(int(pn[0]))
                            # 最終通過順（4コーナー）
                            if len(pn) >= 4:
                                pass4_list.append(int(pn[-1]))
                            elif len(pn) >= 2:
                                pass4_list.append(int(pn[-1]))
                            found_pass = True
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
            result['last_pop'] = pop_list[0]
        if odds_list:
            result['last_odds'] = odds_list[0]
        if race_dates:
            diff = (datetime.now() - race_dates[0]).days
            result['interval_days'] = max(diff, 1)
        else:
            result['interval_days'] = 30
        if pass_list:
            ap = sum(pass_list) / len(pass_list)
            result['avg_pass_pos'] = ap
            if ap <= 2.0: result['running_style'] = 1
            elif ap <= 5.0: result['running_style'] = 2
            elif ap <= 10.0: result['running_style'] = 3
            else: result['running_style'] = 4
        if pass4_list:
            result['last_pass4'] = pass4_list[0]
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
        waku, umaban = 0, 0
        wt = row.select_one("td.Waku span")
        if wt:
            w = wt.get_text(strip=True)
            if w.isdigit(): waku = int(w)
        ut = row.select_one("td.Umaban")
        if ut:
            u = ut.get_text(strip=True)
            if u.isdigit(): umaban = int(u)
        if umaban == 0:
            for td in row.find_all("td"):
                cls = " ".join(td.get("class", []))
                if "Num" in cls or "num" in cls or "Umaban" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
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
        if umaban == 0:
            umaban = len(horses) + 1
        nt = row.select_one("span.HorseName a")
        if not nt: continue
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
                    sa = t; break
        sex = sa[0] if sa else '牡'
        age = int(sa[1:]) if sa and sa[1:].isdigit() else 3
        kinryo = 55.0
        for td in row.find_all("td"):
            try:
                v = float(td.get_text(strip=True))
                if 48.0 <= v <= 62.0: kinryo = v; break
            except: continue
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

# ===== Pace Advantage =====
def calc_pace_advantage(distance, surface, condition, num_horses, is_nar=False):
    scores = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    reasons = {1: "", 2: "", 3: "", 4: ""}
    if is_nar:
        scores[1] += 0.15; scores[2] += 0.2; scores[3] -= 0.05; scores[4] -= 0.2
        reasons[1] = "地方小回り逃げ有利"
        reasons[2] = "地方先行圧倒的有利"
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
    if condition in ['重', '不良', '不']:
        scores[1] += 0.05; scores[2] += 0.08; scores[4] -= 0.1
        reasons[2] += " 重馬場前残り" if reasons[2] else "重馬場前残り"
    elif condition in ['稍', '稍重']:
        scores[2] += 0.04; scores[4] -= 0.05
        if not reasons[2]: reasons[2] = "稍重で前残り"
    if num_horses <= 10:
        scores[1] += 0.1; scores[2] += 0.05; scores[4] -= 0.1
        if not reasons[1]: reasons[1] = "少頭数逃げ有利"
    elif num_horses >= 16:
        scores[3] += 0.08; scores[4] += 0.05; scores[1] -= 0.05
        if not reasons[3]: reasons[3] = "多頭数で届く"
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
    if days <= 0: return "不明"
    if days <= 6: return "連闘"
    weeks = days // 7
    if weeks <= 8: return f"中{weeks}週"
    months = days // 30
    return f"{months}ヶ月"

def interval_cls(days):
    if days <= 0: return 'sr'
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
    html += '<div class="sgrid">'
    html += f'<div class="sitem"><div class="slbl">前走</div><div class="sval {finish_cls(h["前走着順"])}">{h["前走着順"]}着</div></div>'
    html += f'<div class="sitem"><div class="slbl">脚質</div><div class="sval"><span class="stag {style_css}">{style_name}</span>{pm_html}</div></div>'
    html += f'<div class="sitem"><div class="slbl">間隔</div><div class="sval {interval_cls(h.get("前走間隔",30))}">{interval_text(h.get("前走間隔",30))}</div></div>'
    html += f'<div class="sitem"><div class="slbl">体重</div><div class="sval">{weight_html(h.get("場体重増減",0))}</div></div>'
    html += f'<div class="sitem"><div class="slbl">上がり</div><div class="sval">{h.get("上がり3F",35.5):.1f}</div></div>'
    html += '</div>'
    html += '<div class="tagrow">'
    if father: html += f'<span class="tag tag-sire">父: {father}</span>'
    if fr > 0: html += f'<span class="tag">複勝率 {int(fr*100)}%</span>'
    bt_str = h.get('タイム表示', '')
    bt_date = h.get('タイム日付', '')
    bt_dist = h.get('タイム距離', 0)
    if bt_str:
        date_short = bt_date.replace('/', '.') if bt_date else ''
        date_display = f' ({date_short})' if date_short else ''
        html += f'<span class="tag" style="border:1px solid rgba(240,192,64,0.3);color:#f0c040 !important">⏱ {bt_dist}m {bt_str}{date_display}</span>'
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
    gap12 = t1['スコア'] - t2['スコア']
    conf_stars = '★★★★' if gap12 > 0.03 else ('★★★' if gap12 > 0.015 else '★★')
    html = '<div class="buy-card buy-honmei"><div class="buy-header">'
    html += '<span class="buy-type bt-hon">&#128293; 本命</span>'
    html += f'<span class="buy-conf">信頼度 {conf_stars}</span></div>'
    html += f'<div class="buy-row"><span class="buy-lbl">ワイド</span><span class="buy-horses">{n1} ― {n3}</span></div>'
    html += f'<div class="buy-row"><span class="buy-lbl">馬連</span><span class="buy-horses">{n1} ― {n2}</span></div>'
    s1_style = STYLE_NAMES.get(int(t1.get('脚質', 0)), '不明')
    s1_match = rank_map.get(int(t1.get('脚質', 0)), ('△','fair',''))[0]
    html += f'<div class="buy-note">TOP1 {t1["馬名"]}は{s1_style}({s1_match})で展開向き。前走{int(t1["前走着順"])}着の安定感。</div>'
    html += '</div>'
    himo = []
    if t4 is not None: himo.append(horse_num(t4))
    if t5 is not None: himo.append(horse_num(t5))
    if t6 is not None: himo.append(horse_num(t6))
    html += '<div class="buy-card buy-hiroku"><div class="buy-header">'
    html += '<span class="buy-type bt-hir">&#9889; 手広く</span>'
    html += '<span class="buy-conf">信頼度 ★★★</span></div>'
    himo_plus = [horse_num(t2), horse_num(t3)] + himo
    html += f'<div class="buy-row"><span class="buy-lbl">3連複</span><span class="buy-horses">{horse_num(t1)} ― {horse_num(t2)},{horse_num(t3)} ― {",".join(himo_plus)}</span></div>'
    pt = len(himo) * 2 + 1 if himo else 3
    html += f'<div class="buy-note">軸{t1["馬名"]}固定。2着候補にTOP2-3、3着候補にTOP2-6。計{pt}点（TOP1-2-3含む）</div></div>'
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
        bt_date = h.get('タイム日付', '')
        bt_date_short = bt_date.replace('/', '.') if bt_date else ''
        bt_display = bt if bt else '-'
        if bt and bt_date_short:
            bt_display = f'{bt}<br><span style="font-size:0.7em;color:#6a6a80">{bt_date_short}</span>'
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

# Model version badge
badge_css = f'badge-{model_version}'
auc_text = f' AUC {model_auc:.4f}' if model_auc > 0 else ''
leak_text = ' LEAK-FREE' if model_leak_free else ''
st.markdown(f'<div style="text-align:center;margin-top:-12px;margin-bottom:12px"><span class="model-badge {badge_css}">MODEL {model_version.upper()}{auc_text}{leak_text}</span></div>', unsafe_allow_html=True)

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
                    horse['母の父'] = stats.get('mother_father', '')
                    horse['血統スコア'] = calc_sire_score(stats.get('father',''), race_info['surface'], race_info['distance'])
                    horse['持ちタイム'] = stats.get('best_time', 0.0)
                    horse['タイム表示'] = stats.get('best_time_str', '')
                    horse['タイム日付'] = stats.get('best_time_date', '')
                    horse['タイム距離'] = stats.get('best_time_dist', 0)
                    # v3用追加データ
                    horse['通過順平均'] = stats.get('avg_pass_pos', 8.0)
                    horse['通過順4'] = stats.get('last_pass4', 8)
                    horse['前走オッズ'] = stats.get('last_odds', 15.0)
                    horse['前走人気'] = stats.get('last_pop', 8)
                    horse['所属地'] = stats.get('trainer_loc', '')
                except Exception:
                    horse.update({'前走着順':5,'距離適性':0.5,'馬場適性':0.5,'人気傾向':0.5,
                                  'コース適性':0.5,'前走間隔':30,'脚質':0,'上がり3F':35.5,
                                  '複勝率':0.0,'父':'','母の父':'','血統スコア':0.5,'持ちタイム':0.0,
                                  'タイム表示':'','タイム日付':'','タイム距離':0,
                                  '通過順平均':8.0,'通過順4':8,'前走オッズ':15.0,'前走人気':8,'所属地':''})
            else:
                horse.update({'前走着順':5,'距離適性':0.5,'馬場適性':0.5,'人気傾向':0.5,
                              'コース適性':0.5,'前走間隔':30,'脚質':0,'上がり3F':35.5,
                              '複勝率':0.0,'父':'','母の父':'','血統スコア':0.5,'持ちタイム':0.0,
                              'タイム表示':'','タイム日付':'','タイム距離':0,
                              '通過順平均':8.0,'通過順4':8,'前走オッズ':15.0,'前走人気':8,'所属地':''})
            progress_bar.progress((i + 1) / num_horses)
            if i < num_horses - 1:
                time.sleep(0.5)
        progress_bar.empty()
    # Score calculation
    df = pd.DataFrame(horses)

    # === 共通特徴量（v2/v3共通）===
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
    import datetime as dt_module
    now = dt_module.datetime.now()
    df['月'] = now.month
    m = now.month
    df['季節'] = 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3))
    df['枠馬場'] = df['枠位置'] * 10 + df['馬場状態_enc']
    df['馬齢グループ'] = df['馬齢'].clip(2, 7)

    # === v3専用特徴量 ===
    if model_version == 'v3':
        # 父馬_enc: sire_mapを使ってエンコード
        df['父馬_enc'] = df['父'].apply(lambda x: sire_map.get(x, 50) if sire_map else 50)

        # 母父_enc: bms_mapを使ってエンコード
        df['母父_enc'] = df['母の父'].apply(lambda x: bms_map.get(x, 50) if bms_map else 50)

        # 所属_enc: 美浦=0, 栗東=1, その他=3
        def encode_location(loc):
            if '美浦' in str(loc) or '美' == str(loc): return 0
            if '栗東' in str(loc) or '栗' == str(loc): return 1
            return 3
        df['所属_enc'] = df['所属地'].apply(encode_location)

        # 前走人気
        df['前走人気'] = df['前走人気'].fillna(8)

        # 前走オッズlog
        df['前走オッズlog'] = np.log1p(df['前走オッズ'].clip(1, 999).fillna(15.0))

        # 前走上がり3F
        df['前走上がり'] = df['上がり3F'].fillna(35.5)

        # 前走通過順1 / 前走通過順4
        df['前走通過順1'] = df['通過順平均'].fillna(8.0)
        df['前走通過順4'] = df['通過順4'].fillna(8)

    # === v5専用特徴量 ===
    if model_version == 'v5':
        n_top = 80
        # Sire/BMS encoding
        df['sire_enc'] = df['父'].apply(lambda x: sire_map.get(x, n_top) if sire_map else n_top)
        df['bms_enc'] = df['母の父'].apply(lambda x: bms_map.get(x, n_top) if bms_map else n_top)

        # Location encoding
        def enc_loc(loc):
            s = str(loc)
            if '美浦' in s or '美' == s: return 0
            if '栗東' in s or '栗' == s: return 1
            return 3
        df['location_enc'] = df['所属地'].apply(enc_loc)

        # Base mappings to English names
        df['horse_weight'] = df['馬体重']
        df['weight_diff'] = 0
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
        month_now = datetime.now().month
        df['month_val'] = month_now
        df['season'] = 0 if month_now in [3,4,5] else (1 if month_now in [6,7,8] else (2 if month_now in [9,10,11] else 3))
        df['bracket_cond'] = df['bracket_pos'] * 10 + df['馬場状態_enc']
        df['age_group'] = df['馬齢'].clip(2, 7)

        # Prev race data
        df['prev_pop'] = df['前走人気'].fillna(8)
        df['prev_odds_log'] = np.log1p(df['前走オッズ'].clip(1, 999).fillna(15.0))
        df['prev_last3f'] = df['上がり3F'].fillna(35.5)
        df['prev_pass1'] = df['通過順平均'].fillna(8.0)
        df['prev_pass4'] = df['通過順4'].fillna(8)
        df['prev_margin'] = 0
        df['prev_prize'] = 0

        # Aggregated (use available data, default for missing)
        df['avg_finish_3r'] = df['前走着順'].fillna(5)
        df['avg_finish_5r'] = df['前走着順'].fillna(5)
        df['avg_last3f_3r'] = df['上がり3F'].fillna(35.5)
        df['best_finish_5r'] = df['前走着順'].fillna(5)
        df['finish_trend'] = 0
        df['dist_change'] = 0
        df['dist_change_abs'] = 0
        df['rest_days'] = df.get('interval_days', pd.Series([30]*len(df))).fillna(30)
        if 'interval_days' in df.columns:
            df['rest_days'] = df['interval_days']
        df['rest_category'] = pd.cut(df['rest_days'], bins=[-1,6,14,35,63,180,9999], labels=[0,1,2,3,4,5]).astype(float).fillna(2)

        # Historical rates (use defaults - netkeibaから精密計算は困難)
        df['same_dist_rate'] = 0.3
        df['same_course_rate'] = 0.3
        df['same_surface_rate'] = 0.3

        # Horse cumulative stats
        df['horse_win_rate'] = 0.1
        df['horse_top3_rate'] = 0.3
        df['horse_race_count'] = 5

        # Jockey/Trainer
        df['jockey_course_wr'] = df['騎手勝率']
        df['jockey_dist_wr'] = df['騎手勝率']
        df['jockey_top3'] = df['騎手勝率'] * 3
        df['trainer_wr'] = 0.08
        df['trainer_top3'] = 0.25

        # Interactions
        df['weight_dist'] = df['馬体重'] * df['距離(m)'] / 10000.0
        df['age_season'] = df['馬齢'] * 10 + df['season']
        df['carry_per_weight'] = df['斤量'] / df['馬体重'].clip(1) * 100
        df['horse_num_ratio'] = df['馬番'] / df['頭数'].clip(1)
        df['weight_diff_abs'] = 0

    # === 特徴量のデフォルト値を保証 ===
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[FEATURES].values
    proba = model.predict_proba(X)
    ai_scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, :3].sum(axis=1)
    pop_scores = df['人気傾向'].values
    apt_scores = (df['距離適性'].values + df['馬場適性'].values) / 2.0
    # Pace scores
    pace_scores = []
    for _, h in df.iterrows():
        rs = int(h.get('脚質', 0))
        if rs == 0:
            pace_scores.append(0.5)
        else:
            base = pace_scores_map.get(rs, 0.5)
            pace_scores.append(max(0.0, min(1.0, base)))
    pace_scores = np.array(pace_scores)
    agari_scores = np.clip(1.0 - (df['上がり3F'].values - 33.0) / 5.0, 0.0, 1.0)
    course_scores = df['コース適性'].values
    other_scores = (df['血統スコア'].values + df['複勝率'].values) / 2.0
    jockey_scores = np.clip(df['騎手勝率'].values / 0.18, 0.0, 1.0)
    # Best time score
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
        final_scores = (
            ai_scores * 0.27 + pop_scores * 0.13 + pace_scores * 0.18
            + agari_scores * 0.08 + jockey_scores * 0.10 + apt_scores * 0.08
            + time_scores * 0.08 + course_scores * 0.04 + other_scores * 0.04
        )
    else:
        if model_version == 'v3':
            # v3: AIモデルの精度が高いので比重を上げる
            final_scores = (
                ai_scores * 0.55 + pop_scores * 0.10 + apt_scores * 0.08
                + pace_scores * 0.08 + agari_scores * 0.08 + course_scores * 0.04
                + other_scores * 0.04 + time_scores * 0.03
            )
        elif model_version == 'v5':
            # v5: 最高精度モデル - AI比重を最大化
            final_scores = (
                ai_scores * 0.60 + pop_scores * 0.08 + apt_scores * 0.07
                + pace_scores * 0.07 + agari_scores * 0.07 + course_scores * 0.04
                + other_scores * 0.04 + time_scores * 0.03
            )
        else:
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
