"""
共通予測コアモジュール
app.py / daily_predict.py / race_auto_notify.py が同一の予測ロジックを使用する。

このモジュールが予測結果の唯一の真実 (Single Source of Truth) となる。
"""
import warnings
# DataFrame fragmentation警告を抑制（175個のdf代入を全てconcatに書換えるのは工数大）
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

import pandas as pd
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except Exception:
    pass
import numpy as np
import pickle
import gzip
import json
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
from datetime import datetime
from itertools import combinations

# === v15新特徴量モジュール (Single Source of Truth for feature names) ===
_V15_NEW_FEATURES_AVAILABLE = False
try:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _BASE_DIR not in sys.path:
        sys.path.insert(0, _BASE_DIR)
    from train.features_v15_new import (
        get_v15_new_features,
        compute_jockey_horse_features,
        compute_transport_distance,
        compute_course_renovation,
        compute_gaisha_features,
        compute_all_v15_new_features,
    )
    _V15_NEW_FEATURES_AVAILABLE = True
except Exception as _e:
    def get_v15_new_features():
        return ['jockey_horse_rides', 'jockey_horse_wr', 'jockey_horse_top3r',
                'jockey_change', 'jockey_change_to_top',
                'transport_distance_km', 'is_long_transport',
                'course_renovated', 'post_renovation_flag',
                'gaisha_rank']

# === FT-Transformer (optional, requires torch) ===
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass

if _TORCH_AVAILABLE:
    class _NumericalEmbedding(nn.Module):
        """各数値特徴量を個別の線形層でd_token次元のembeddingに変換"""
        def __init__(self, n_features, d_token):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(n_features, d_token))
            self.biases = nn.Parameter(torch.zeros(n_features, d_token))

        def forward(self, x):
            return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)

    class _MultiHeadSelfAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            assert d_model % n_heads == 0
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            B, N, D = x.shape
            q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, N, D)
            return self.W_o(out)

    class _TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x, mask=None):
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.ff(self.norm2(x))
            return x

    class FTTransformer(nn.Module):
        """Feature Tokenizer + Transformer for tabular data."""
        def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3,
                     d_ff_mult=2, dropout=0.1):
            super().__init__()
            self.n_features = n_features
            self.d_token = d_token
            self.feature_embedding = _NumericalEmbedding(n_features, d_token)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
            d_ff = d_token * d_ff_mult
            self.layers = nn.ModuleList([
                _TransformerBlock(d_token, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Sequential(
                nn.Linear(d_token, d_token),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token, 1),
            )

        def forward(self, x):
            B = x.shape[0]
            feat_emb = self.feature_embedding(x)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, feat_emb], dim=1)
            for layer in self.layers:
                tokens = layer(tokens)
            cls_out = self.norm(tokens[:, 0])
            logit = self.head(cls_out).squeeze(-1)
            return logit

    class IntraRaceAttention(nn.Module):
        """同レース内の全馬を同時に見て相対評価."""
        def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                     d_ff_mult=2, dropout=0.1):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            self.layers = nn.ModuleList([
                _TransformerBlock(d_model, n_heads, d_model * d_ff_mult, dropout)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x, mask=None):
            # x: (batch_races, max_horses, n_features)
            h = self.proj(x)
            for layer in self.layers:
                h = layer(h, mask)
            h = self.norm(h)
            logits = self.head(h).squeeze(-1)  # (batch_races, max_horses)
            return logits

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
INVESTMENT_PER_RACE = 700

def _get_headers_with_cookie():
    """Cookie認証付きヘッダーを返す（.envから読み込み）"""
    headers = dict(HEADERS)
    try:
        cookie = os.environ.get('NETKEIBA_COOKIE', '')
        if not cookie:
            _env_path = os.path.join(BASE_DIR, '.env')
            if os.path.exists(_env_path):
                with open(_env_path, 'r') as f:
                    for line in f:
                        if line.startswith('NETKEIBA_COOKIE='):
                            cookie = line.split('=', 1)[1].strip().strip('"').strip("'")
                            break
        if cookie:
            headers['Cookie'] = cookie
    except Exception:
        pass
    return headers

# === マッピング定数 ===
COURSE_MAP = {
    '札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,'中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9,
}
SURFACE_MAP = {'芝':0,'ダ':1,'障':2}
COND_MAP = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}
SEX_MAP = {'牡':0,'牝':1,'セ':2,'騸':2}

# ===== 収益パターンフィルタ =====
_PP_CACHE = None

def _load_pp():
    global _PP_CACHE
    if _PP_CACHE is not None:
        return _PP_CACHE
    pp_path = os.path.join(BASE_DIR, 'data', 'profitable_patterns.json')
    if not os.path.exists(pp_path):
        _PP_CACHE = []
        return []
    try:
        with open(pp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _PP_CACHE = data.get('profitable_patterns', [])
    except Exception:
        _PP_CACHE = []
    return _PP_CACHE


def match_profitable_patterns(cond_key, venue, surface, distance, top1_odds=0):
    """レース条件に合致する収益パターンをマッチし、最高★数を返す。
    Returns: (max_stars, matched_list[:5])
    """
    patterns = _load_pp()
    if not patterns:
        return 0, []
    # オッズ帯
    if top1_odds <= 0: ob = 'all'
    elif top1_odds < 3: ob = '1-3'
    elif top1_odds < 10: ob = '3-10'
    elif top1_odds < 30: ob = '10-30'
    else: ob = '30-100'
    # 距離帯
    if distance <= 1400: db = 'sprint'
    elif distance <= 1600: db = 'mile'
    elif distance <= 2200: db = 'middle'
    else: db = 'long'
    sj = '芝' if surface in ('芝', 0, '0') else 'ダート'

    matched = []
    for p in patterns:
        if p.get('type') != 'trio':
            continue
        if p.get('condition') != 'all' and p.get('condition') != cond_key:
            continue
        if p.get('venue') != 'all' and p.get('venue') != venue:
            continue
        if p.get('surface') != 'all' and p.get('surface') != sj:
            continue
        if p.get('distance') != 'all' and p.get('distance') != db:
            continue
        if p.get('odds_band') != 'all' and p.get('odds_band') != ob:
            continue
        matched.append(p)
    if not matched:
        return 0, []
    matched.sort(key=lambda x: x.get('roi', 0), reverse=True)
    return max(p.get('stars', 0) for p in matched), matched[:5]


CONDITION_PROFILES = {
    'A': {'bet_type':'trio','label':'条件A','desc':'8-14頭/1600m+/良~稍','investment':700,'roi':205.3,'hit_rate':44.5,'recommended':True},
    'B': {'bet_type':'trio','label':'条件B','desc':'8-14頭/1600m+/重~不良','investment':700,'roi':236.9,'hit_rate':45.2,'recommended':True},
    'C': {'bet_type':'trio','label':'条件C','desc':'15頭+/1600m+/良~稍','investment':700,'roi':285.6,'hit_rate':33.7,'recommended':True},
    'D': {'bet_type':'trio','label':'条件D','desc':'1200-1400m','investment':700,'roi':136.0,'hit_rate':27.0,'recommended':True},
    'E': {'bet_type':'umaren','label':'条件E','desc':'7頭以下','investment':700,'roi':118.0,'hit_rate':53.4,'recommended':True},
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

# 追い切りランク → 調教タイム推定マッピング
RANK_TO_WOOD_4F = {'A': 51.5, 'B': 53.0, 'C': 54.5, 'D': 55.5}
RANK_TO_SAKARO_4F = {'A': 53.5, 'B': 56.0, 'C': 58.0, 'D': 59.5}
RANK_TO_SAKARO_3F = {'A': 37.5, 'B': 39.0, 'C': 40.5, 'D': 41.5}


# === ルックアップテーブル ===
_FEATURE_LOOKUPS = None


def load_feature_lookups():
    """事前計算済み特徴量ルックアップテーブルをロード"""
    global _FEATURE_LOOKUPS
    if _FEATURE_LOOKUPS is not None:
        return _FEATURE_LOOKUPS
    lookup_path = os.path.join(BASE_DIR, "data", "feature_lookups.pkl")
    if os.path.exists(lookup_path):
        try:
            with open(lookup_path, 'rb') as f:
                _FEATURE_LOOKUPS = pickle.load(f)
            return _FEATURE_LOOKUPS
        except Exception as e:
            print(f"[WARN] ルックアップテーブルロード失敗: {e}")
    _FEATURE_LOOKUPS = {}
    return _FEATURE_LOOKUPS


# === ユーティリティ ===

_jockey_wr = None


def _load_jockey_wr():
    global _jockey_wr
    if _jockey_wr is not None:
        return _jockey_wr
    fpath = os.path.join(BASE_DIR, "jockey_wr.json")
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            _jockey_wr = json.load(f)
    except Exception:
        _jockey_wr = {}
    return _jockey_wr


def find_jockey_wr(name):
    jwr = _load_jockey_wr()
    if name in MODERN_JOCKEY_WR:
        return MODERN_JOCKEY_WR[name]
    if name in jwr:
        return jwr[name]
    for k, v in MODERN_JOCKEY_WR.items():
        if k in name or name in k:
            return v
    for k, v in jwr.items():
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


# === 条件分類 ===

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

    profile = dict(CONDITION_PROFILES[cond_key])
    if cond_key == 'D' and dist <= 1000:
        profile['recommended'] = False
        profile['desc'] = '1000m以下（非推奨：ROI 85%）'
    return cond_key, profile


# === 買い目生成 ===

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


# === モデルロード ===

def _load_pkl(fpath):
    """pkl or pkl.gz をロードする"""
    candidates = [fpath] if fpath.endswith('.gz') else [fpath + '.gz', fpath]
    for p in candidates:
        if os.path.exists(p):
            try:
                opener = gzip.open if p.endswith('.gz') else open
                with opener(p, 'rb') as f:
                    return pickle.load(f), p
            except Exception as e:
                print(f"[WARN] {os.path.basename(p)} ロード失敗: {e}")
    return None, None


def load_models():
    """Pattern B優先、Pattern Aフォールバック"""
    result = {'model': None, 'features': None, 'sire_map': {}, 'bms_map': {},
              'version': 'v9', 'n_top_encode': 80, 'is_live': False}

    candidates = [
        (os.path.join(BASE_DIR, 'keiba_model_v15_central_live.pkl.gz'), True, 'v15 Pattern B (当日情報込み, 150特徴量)'),
        (os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz'), False, 'v15 Pattern A (リークフリー, 145特徴量)'),
        (os.path.join(BASE_DIR, 'keiba_model_v135_central_live.pkl.gz'), True, 'v13.5b Pattern B (当日情報込み)'),
        (os.path.join(BASE_DIR, 'keiba_model_v135_central.pkl.gz'), False, 'v13.5b Pattern A (リークフリー)'),
        (os.path.join(BASE_DIR, 'keiba_model_v134_central_live.pkl.gz'), True, 'v13.4 Pattern B (当日情報込み)'),
        (os.path.join(BASE_DIR, 'keiba_model_v134_central.pkl.gz'), False, 'v13.4 Pattern A (リークフリー)'),
        (os.path.join(BASE_DIR, 'keiba_model_v12_central_live.pkl.gz'), True, 'V12 Pattern B (当日情報込み)'),
        (os.path.join(BASE_DIR, 'keiba_model_v12_central.pkl.gz'), False, 'V12 Pattern A (リークフリー)'),
        (os.path.join(BASE_DIR, 'keiba_model_v9_central_live.pkl'), True, 'V9 Pattern B (当日情報込み)'),
        (os.path.join(BASE_DIR, 'keiba_model_v9_central.pkl'), False, 'V9 Pattern A (リークフリー)'),
        (os.path.join(BASE_DIR, 'keiba_model_v8.pkl'), False, 'V8フォールバック'),
    ]

    for fpath, is_live, label in candidates:
        data, loaded_path = _load_pkl(fpath)
        if data and isinstance(data, dict) and 'model' in data:
            result['model'] = data['model']
            result['features'] = data.get('features')
            result['sire_map'] = data.get('sire_map', {})
            result['bms_map'] = data.get('bms_map', {})
            result['version'] = data.get('version', 'v9')
            result['n_top_encode'] = data.get('n_top_encode', 80)
            result['is_live'] = is_live
            # XGB/CatBoost ensemble parts
            result['ensemble_weights'] = data.get('ensemble_weights', {})
            result['xgb_model'] = data.get('xgb_model')
            result['cb_model'] = data.get('cb_model')
            # FT-Transformer parts
            result['ft_model_state'] = data.get('ft_model_state')
            result['ft_model_config'] = data.get('ft_model_config')
            result['ft_scaler_mean'] = data.get('ft_scaler_mean')
            result['ft_scaler_scale'] = data.get('ft_scaler_scale')
            # IntraRace Attention parts
            result['ir_model_state'] = data.get('ir_model_state')
            result['ir_model_config'] = data.get('ir_model_config')
            result['ir_scaler_mean'] = data.get('ir_scaler_mean')
            result['ir_scaler_scale'] = data.get('ir_scaler_scale')
            print(f"[MODEL] {label} ロード完了 ({os.path.basename(loaded_path)})")
            return result

    print("[ERROR] モデルファイルが見つかりません")
    return result


# === 追い切りランク取得 ===

def fetch_oikiri_ranks(race_id):
    """netkeibaの追い切りページから各馬の調教ランク(A/B/C/D)を取得"""
    ranks = {}
    try:
        url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        wrapper = soup.find("div", class_="OikiriAllWrapper")
        if not wrapper:
            return ranks
        for row in wrapper.find_all("tr"):
            td_umaban = row.select_one("td.Umaban")
            if not td_umaban:
                continue
            try:
                umaban = int(td_umaban.get_text(strip=True))
            except (ValueError, TypeError):
                continue
            rank = ""
            for td in row.find_all("td"):
                for cls in td.get("class", []):
                    if cls.startswith("Rank_"):
                        rank = cls.replace("Rank_", "")
                        break
                if rank:
                    break
            if not rank:
                continue
            if rank in ('A', 'B', 'C', 'D'):
                ranks[umaban] = rank
            elif any(x in rank for x in ['好調教', '抜群', '絶好']):
                ranks[umaban] = 'A'
            elif any(x in rank for x in ['上々', '乗込入念', '良化']):
                ranks[umaban] = 'B'
            elif any(x in rank for x in ['まずまず', '平凡', '先着平凡', '乗込']):
                ranks[umaban] = 'C'
            else:
                ranks[umaban] = 'C'
    except Exception:
        pass
    return ranks


# === 出馬表スクレイピング ===

def parse_shutuba(race_id):
    """出馬表を解析。馬名・馬番・斤量・馬齢・オッズ等を取得"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")

    race_name = "レース"
    tag = soup.find("div", class_="RaceName")
    if not tag:
        tag = soup.find(class_="RaceName")
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
    if '障' in d01t:
        surface = '障'
    elif '芝' in d01t:
        surface = '芝'
    elif 'ダ' in d01t:
        surface = 'ダ'
    else:
        surface = 'ダ'
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

    start_time = ''
    tm = re.search(r'(\d{1,2}:\d{2})', d01t)
    if tm:
        start_time = tm.group(1)

    # 天候
    weather = ''
    wm = re.search(r'天候:(\S+)', d01t)
    if wm:
        weather = wm.group(1)
    elif '晴' in d01t:
        weather = '晴'
    elif '曇' in d01t:
        weather = '曇'
    elif '雨' in d01t:
        weather = '雨'
    elif '雪' in d01t:
        weather = '雪'

    race_info = dict(distance=distance, surface=surface, condition=condition,
                     course=course_name, race_num=race_num, start_time=start_time,
                     weather=weather)

    rows = soup.select("tr.HorseList")
    horses, horse_ids = [], []

    for row in rows:
        rc = row.get("class", [])
        if "Cancel" in rc:
            continue
        waku, umaban = 0, 0
        wt = row.select_one("td.Waku span")
        if not wt:
            for td in row.find_all("td"):
                cls = " ".join(td.get("class", []))
                if cls.startswith("Waku"):
                    w = td.get_text(strip=True)
                    if w.isdigit() and 1 <= int(w) <= 8:
                        waku = int(w)
                        break
        else:
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

        odds_val = 0.0
        odds_td = row.select_one("td.Popular")
        if odds_td:
            odds_text = odds_td.get_text(strip=True)
            try:
                odds_val = float(odds_text.replace(',', ''))
                if not (1.0 <= odds_val <= 9999.9):
                    odds_val = 0.0
            except (ValueError, TypeError):
                odds_val = 0.0

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
            'horse_id_val': int(horse_id) if horse_id and horse_id.isdigit() else 0,
            '単勝オッズ': odds_val,
        })
        horse_ids.append(horse_id)

    return race_name, horses, horse_ids, race_info


# === 馬成績取得 ===

def get_horse_stats(horse_id, target_distance, target_surface, target_course=""):
    """netkeibaから馬の過去成績を取得"""
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
        # UI用追加フィールド
        'best_time': 0.0, 'best_time_str': '', 'best_time_date': '', 'best_time_dist': 0,
        'margin_text': '', 'weight_diff': 0, 'prev_jockey': '',
        # Weight history for weight_ma5/weight_trend/weight_peak_diff
        'weight_history': [],
        # Lap data for prev_race_first3f/last3f/pace_diff
        'prev_race_first3f': 0, 'prev_race_last3f': 0, 'prev_race_pace_diff': 0,
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

        # Fallback: pedigree page
        if not result['father'] or not result['mother_father']:
            try:
                url_ped = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
                resp_ped = requests.get(url_ped, headers=HEADERS, timeout=10)
                resp_ped.encoding = "EUC-JP"
                soup_ped = BeautifulSoup(resp_ped.text, "html.parser")
                ped_table = soup_ped.find("table", class_="blood_table")
                if ped_table:
                    tds = ped_table.find_all("td")
                    if tds and not result['father']:
                        a = tds[0].find("a")
                        if a:
                            result['father'] = a.get_text(strip=True)
                    if not result['mother_father']:
                        large_rs_count = 0
                        dam_idx = None
                        for idx, td in enumerate(tds):
                            rs = td.get("rowspan", "1")
                            if rs.isdigit() and int(rs) >= 8:
                                large_rs_count += 1
                                if large_rs_count == 3:
                                    dam_idx = idx
                                    break
                        if dam_idx is not None and dam_idx + 1 < len(tds):
                            a = tds[dam_idx + 1].find("a")
                            if a:
                                result['mother_father'] = a.get_text(strip=True)
            except Exception:
                pass

        # Fallback: trainer location from main profile page
        if not result['trainer_loc']:
            try:
                url_prof = f"https://db.netkeiba.com/horse/{horse_id}/"
                resp_prof = requests.get(url_prof, headers=HEADERS, timeout=10)
                resp_prof.encoding = "EUC-JP"
                soup_prof = BeautifulSoup(resp_prof.text, "html.parser")
                prof3 = soup_prof.find("table", class_="db_prof_table")
                if prof3:
                    for tr in prof3.find_all("tr"):
                        th = tr.find("th")
                        td = tr.find("td")
                        if th and td and '調教師' in th.get_text():
                            td_text = td.get_text(strip=True)
                            if '美浦' in td_text or '(美)' in td_text:
                                result['trainer_loc'] = '美浦'
                            elif '栗東' in td_text or '(栗)' in td_text:
                                result['trainer_loc'] = '栗東'
                            break
            except Exception:
                pass

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
        jockey_results = {}  # {jockey_name: [finish1, finish2, ...]}
        pass4_list = []
        weight_list = []
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

            # 騎手名を集計（騎手×馬相性計算用）
            if len(tds) > 12:
                _jname_row = tds[12].get_text(strip=True)
                if _jname_row:
                    jockey_results.setdefault(_jname_row, []).append(finish)

            if len(finish_list) == 1:
                result['last_finish'] = finish
                if len(tds) > 18:
                    result['margin_text'] = tds[18].get_text(strip=True)
                if len(tds) > 12:
                    result['prev_jockey'] = tds[12].get_text(strip=True)

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

            # Weight extraction (format: "480(+4)" or "480")
            # db.netkeiba result table has 33 cols; weight is typically at td[28]
            try:
                for _wt_idx in range(20, min(len(tds), 33)):
                    _wt_text = tds[_wt_idx].get_text(strip=True)
                    _wt_match = re.match(r'(\d{3,})\(', _wt_text)
                    if not _wt_match:
                        _wt_match = re.match(r'^(\d{3})$', _wt_text)
                    if _wt_match:
                        _wt_val = int(_wt_match.group(1))
                        if 350 <= _wt_val <= 600:
                            weight_list.append(_wt_val)
                            break
            except Exception:
                pass

            dc = tds[14].get_text(strip=True)
            ddm = re.match(r'([芝ダ障])(\d+)', dc)
            if ddm:
                sc, dv = ddm.group(1), int(ddm.group(2))
                if target_distance > 0 and abs(dv - target_distance) <= 200:
                    dist_results.append(finish)
                    # Best time
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
                                for td_idx2 in range(min(5, len(tds))):
                                    td_text2 = tds[td_idx2].get_text(strip=True)
                                    dtm = re.search(r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', td_text2)
                                    if dtm:
                                        y2 = int(dtm.group(1))
                                        if 2000 <= y2 <= 2030:
                                            best_time_date = dtm.group(0)
                                            break
                            break
                sn = '芝' if sc == '芝' else 'ダ'
                if sn == target_surface:
                    surf_results.append(finish)

            if target_course and len(tds) > 1:
                if target_course in tds[1].get_text(strip=True):
                    course_results.append(finish)

            for tdi, td in enumerate(tds):
                txt = td.get_text(strip=True)
                cleaned = txt.replace(' ', '').replace('-', '-').replace('\uff0d', '-').replace('－', '-')
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
        result['best_time'] = best_time_sec if best_time_sec < 999.0 else 0.0
        result['best_time_str'] = best_time_str
        result['best_time_date'] = best_time_date
        result['best_time_dist'] = best_time_dist

        # Weight history (for weight_ma5/weight_trend/weight_peak_diff)
        result['weight_history'] = weight_list[:5]  # most recent 5

        # v15: 騎手×馬相性データ
        result['jockey_results'] = jockey_results

        # Lap data: get previous race's first3f/last3f from db.netkeiba race page
        try:
            first_row = rows[0] if rows else None
            if first_row:
                _race_link = first_row.find("a", href=re.compile(r'/race/\d+'))
                if _race_link:
                    _prev_rid_match = re.search(r'/race/(\d+)', _race_link['href'])
                    if _prev_rid_match:
                        _prev_rid = _prev_rid_match.group(1)
                        _race_url = f"https://db.netkeiba.com/race/{_prev_rid}/"
                        _race_resp = requests.get(_race_url, headers=HEADERS, timeout=8)
                        _race_resp.encoding = "EUC-JP"
                        _race_soup = BeautifulSoup(_race_resp.text, "html.parser")
                        # Find ラップ row: <th>ラップ</th><td>12.2 - 11.1 - ...</td>
                        _rap_nums = []
                        for _th in _race_soup.find_all("th"):
                            if 'ラップ' in _th.get_text():
                                _td = _th.find_next_sibling("td")
                                if _td:
                                    _rap_nums = re.findall(r'(\d{2}\.\d)', _td.get_text())
                                break
                        if len(_rap_nums) >= 4:
                            _rap_floats = [float(x) for x in _rap_nums]
                            _first3f = sum(_rap_floats[:3])
                            _last3f = sum(_rap_floats[-3:])
                            if 25.0 < _first3f < 50.0 and 25.0 < _last3f < 50.0:
                                result['prev_race_first3f'] = round(_first3f, 1)
                                result['prev_race_last3f'] = round(_last3f, 1)
                                result['prev_race_pace_diff'] = round(_last3f - _first3f, 1)
        except Exception:
            pass

    except Exception:
        pass

    return result


# === オッズ取得 ===

def fetch_realtime_odds(race_id):
    """単勝リアル���イムオッズ＋人気順位を取得

    Returns:
        dict: {馬番: オッズ} (後方互換)
        ※ pop_rank_dict, full結果は fetch_realtime_odds_full() で取得可能
    """
    result = _fetch_odds_api(race_id)
    return {u: v['odds'] for u, v in result.items()}


def fetch_realtime_odds_full(race_id):
    """単勝オッズ＋人気順位を取得

    Returns:
        dict: {馬番: {'odds': float, 'pop_rank': int}}
    """
    return _fetch_odds_api(race_id)


def _fetch_odds_api(race_id):
    """netkeiba APIから単勝オッズ＋人気順位を取得（Cookie認証対応）"""
    result = {}
    try:
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1"
        resp = requests.get(url, headers=_get_headers_with_cookie(), timeout=10)
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
                                pop_rank = int(vals[2]) if len(vals) >= 3 and str(vals[2]).isdigit() else 0
                                if 1.0 <= odds_val <= 9999.9:
                                    result[umaban] = {'odds': odds_val, 'pop_rank': pop_rank}
                            except (ValueError, TypeError, IndexError):
                                pass
    except Exception:
        pass
    return result


# === オッズ基準値キャッシュ（AM8:00予測時のオッズを基準として保存） ===
# 設計（2026-04-12 再実装）:
#   - daily_predict.py がAM8:00時点のオッズを data/odds_base_YYYYMMDD.csv に保存
#   - app.py の「予想する」押下時に当日CSVを読み、リアルタイムオッズと比較
#   - 1回の押下で odds_change_rate / pop_rank_change / odds_sharp_drop が埋まる
#   - 旧 odds_base_cache.json は load 時のフォールバックとしてのみ参照

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_ODDS_CACHE_PATH = os.path.join(_DATA_DIR, 'odds_base_cache.json')  # legacy fallback


def _odds_base_csv_path(date_str=None):
    """data/odds_base_YYYYMMDD.csv へのパス。date_str未指定なら今日。"""
    if not date_str:
        date_str = datetime.now().strftime('%Y%m%d')
    return os.path.join(_DATA_DIR, f'odds_base_{date_str}.csv')


def save_odds_base(race_id, odds_full, date_str=None):
    """基準オッズを当日CSVに保存（同一race_idは初回のみ書き込み、上書きしない）。"""
    if not odds_full:
        return
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        csv_path = _odds_base_csv_path(date_str)
        existing = set()
        rows = []
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    import csv as _csv
                    rd = _csv.DictReader(f)
                    for r in rd:
                        rows.append(r)
                        existing.add(str(r.get('race_id')))
            except Exception:
                pass
        rid = str(race_id)
        if rid in existing:
            return  # baseline既にあり、上書きしない
        ts = datetime.now().strftime('%Y-%m-%d %H:%M')
        for uma, info in odds_full.items():
            if isinstance(info, dict):
                odds_v = info.get('odds', 0)
                pop_v = info.get('pop_rank', 0)
            else:
                odds_v = float(info) if info else 0
                pop_v = 0
            rows.append({
                'race_id': rid, 'horse_num': int(uma),
                'odds': float(odds_v) if odds_v else 0.0,
                'pop_rank': int(pop_v) if pop_v else 0,
                'timestamp': ts,
            })
        import csv as _csv
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            wr = _csv.DictWriter(f, fieldnames=['race_id', 'horse_num', 'odds', 'pop_rank', 'timestamp'])
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    except Exception as e:
        print(f"[ODDS-BASE] save失敗: {e}")


def load_odds_base(race_id, date_str=None):
    """当日CSV(なければ前日CSV、最後にlegacy JSON)から基準オッズを読み込む。

    Returns:
        dict: {馬番(int): {'odds': float, 'pop_rank': int}} or {}
    """
    rid = str(race_id)
    # 1) 当日CSV → 2) race_idの開催日付プレフィックスに紐付くもの → 3) legacy JSON
    paths = []
    if date_str:
        paths.append(_odds_base_csv_path(date_str))
    paths.append(_odds_base_csv_path())  # 今日
    # race_idは年4桁始まりだが日付特定不可。代わりに data/odds_base_*.csv 全部走査
    try:
        import glob as _glob
        for p in sorted(_glob.glob(os.path.join(_DATA_DIR, 'odds_base_*.csv')), reverse=True):
            if p not in paths:
                paths.append(p)
    except Exception:
        pass
    for csv_path in paths:
        if not os.path.exists(csv_path):
            continue
        try:
            import csv as _csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                rd = _csv.DictReader(f)
                out = {}
                for r in rd:
                    if str(r.get('race_id')) != rid:
                        continue
                    out[int(r.get('horse_num') or 0)] = {
                        'odds': float(r.get('odds') or 0),
                        'pop_rank': int(r.get('pop_rank') or 0),
                    }
                if out:
                    return out
        except Exception:
            continue
    # legacy JSONフォールバック
    try:
        if os.path.exists(_ODDS_CACHE_PATH):
            with open(_ODDS_CACHE_PATH, 'r') as f:
                cache = json.load(f)
            entry = cache.get(rid, {})
            odds_data = entry.get('odds', {})
            return {int(k): v for k, v in odds_data.items()}
    except Exception:
        pass
    return {}


def compute_odds_change_features(df, race_id, odds_dict_full):
    """リアルタイムオッズとキャッシュ基準オッズからodds_change特徴量を計算

    Args:
        df: 馬DataFrame（horse_num列が必要）
        race_id: レースID
        odds_dict_full: fetch_realtime_odds_full()の結果

    Returns:
        df: odds_change_rate, pop_rank_change, odds_sharp_drop が追加されたdf
    """
    base = load_odds_base(race_id)
    if not base or not odds_dict_full:
        return df

    uma_col = 'horse_num' if 'horse_num' in df.columns else '馬番'
    change_rates = []
    pop_changes = []
    sharp_drops = []

    for _, row in df.iterrows():
        uma = int(row.get(uma_col, 0))
        b = base.get(uma, {})
        r = odds_dict_full.get(uma, {})
        base_odds = b.get('odds', 0) if isinstance(b, dict) else 0
        rt_odds = r.get('odds', 0) if isinstance(r, dict) else 0
        base_pop = b.get('pop_rank', 0) if isinstance(b, dict) else 0
        rt_pop = r.get('pop_rank', 0) if isinstance(r, dict) else 0

        if base_odds > 0 and rt_odds > 0:
            change_rates.append(np.clip((base_odds - rt_odds) / base_odds, -2.0, 2.0))
            sharp_drops.append(1 if rt_odds <= base_odds * 0.8 else 0)
        else:
            change_rates.append(0.0)
            sharp_drops.append(0)

        if base_pop > 0 and rt_pop > 0:
            pop_changes.append(int(base_pop - rt_pop))
        else:
            pop_changes.append(0)

    df['odds_change_rate'] = change_rates
    df['pop_rank_change'] = pop_changes
    df['odds_sharp_drop'] = sharp_drops
    return df


def is_race_started(race_id):
    """レースが発走済みかどうかを結果ページの存在で判定"""
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        s = BeautifulSoup(resp.text, "html.parser")
        return s.find("table", class_="RaceTable01") is not None
    except Exception:
        return False


def fetch_result_odds(race_id):
    """発走済みレースの結果ページから確定単勝オッズと人気順位を取得"""
    odds_dict = {}
    pop_dict = {}
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        result_table = soup.find("table", class_="RaceTable01")
        if not result_table:
            return odds_dict, pop_dict
        for row in result_table.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 11:
                continue
            umaban_text = tds[2].get_text(strip=True)
            if not umaban_text.isdigit():
                continue
            umaban = int(umaban_text)
            pop_text = tds[9].get_text(strip=True)
            if pop_text.isdigit():
                pop_dict[umaban] = int(pop_text)
            odds_text = tds[10].get_text(strip=True).replace(',', '')
            try:
                odds_val = float(odds_text)
                if 1.0 <= odds_val <= 9999.9:
                    odds_dict[umaban] = odds_val
            except (ValueError, TypeError):
                pass
    except Exception:
        pass
    return odds_dict, pop_dict


# === JRA馬場・天候取得 ===

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


# === 馬成績デフォルト値設定 ===

def set_horse_defaults(horse):
    """馬成績のデフォルト値を設定"""
    horse.update({
        '前走着順': 5, '距離適性': 0.5, '馬場適性': 0.5, '人気傾向': 0.5,
        'コース適性': 0.5, '前走間隔': 30, '脚質': 0, '上がり3F': 35.5,
        '複勝率': 0.0, '父': '', '母の父': '', '血統スコア': 0.5,
        '通過順平均': 8.0, '通過順4': 8, '前走オッズ': 15.0, '前走人気': 8, '所属地': '',
        'prev2_finish': 5, 'prev3_finish': 5, 'prev4_finish': 5, 'prev5_finish': 5,
        'avg_finish_3r': 5.0, 'avg_finish_5r': 5.0, 'best_finish_3r': 5, 'best_finish_5r': 5,
        'top3_count_3r': 0, 'top3_count_5r': 0, 'finish_trend': 0, 'prev2_last3f': 35.5,
        'weight_history': [],
        'prev_race_first3f_scraped': 0, 'prev_race_last3f_scraped': 0, 'prev_race_pace_diff_scraped': 0,
    })


def apply_horse_stats(horse, stats, race_info):
    """get_horse_statsの結果をhorse dictに適用"""
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
    horse['血統スコア'] = calc_sire_score(
        stats.get('father', ''), race_info['surface'], race_info['distance'])
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
    # Weight history for weight trend features
    horse['weight_history'] = stats.get('weight_history', [])
    # Lap features from previous race
    horse['prev_race_first3f_scraped'] = stats.get('prev_race_first3f', 0)
    horse['prev_race_last3f_scraped'] = stats.get('prev_race_last3f', 0)
    horse['prev_race_pace_diff_scraped'] = stats.get('prev_race_pace_diff', 0)
    # v15: 前走騎手（騎手変更検出用）+ 騎手×馬相性
    horse['prev_jockey'] = stats.get('prev_jockey', '')
    horse['jockey_results'] = stats.get('jockey_results', {})
    # UI用追加フィールド
    horse['持ちタイム'] = stats.get('best_time', 0.0)
    horse['タイム表示'] = stats.get('best_time_str', '')
    horse['タイム日付'] = stats.get('best_time_date', '')
    horse['タイム距離'] = stats.get('best_time_dist', 0)


# === 特徴量構築 ===

def build_features(horses, race_info, model_data, race_id=None, odds_dict=None,
                   jra_track_info=None, weather_info=None):
    """馬リストから特徴量DataFrameを構築（全スクリプト共通）"""
    df = pd.DataFrame(horses)
    num_horses = len(df)
    version = model_data.get('version', 'v9')
    is_live = model_data.get('is_live', False)
    n_top = model_data.get('n_top_encode', 80)
    use_sire_map = model_data.get('sire_map', {})
    use_bms_map = model_data.get('bms_map', {})

    # set_horse_defaults未呼出時のフォールバック
    _defaults = {
        '父': '', '母の父': '', '前走着順': 5, '所属地': '', '馬体重': 480,
        '場体重増減': 0, '騎手勝率': 0.0, '前走オッズ': 15.0, '前走人気': 8,
        '上がり3F': 35.5, '通過順平均': 8.0, '通過順4': 8, '前走間隔': 30,
        'prev2_finish': 5, 'prev3_finish': 5, 'prev4_finish': 5, 'prev5_finish': 5,
        'avg_finish_3r': 5.0, 'avg_finish_5r': 5.0, 'best_finish_3r': 5, 'best_finish_5r': 5,
        'top3_count_3r': 0, 'top3_count_5r': 0, 'finish_trend': 0, 'prev2_last3f': 35.5,
        'weight_history': None,
        'prev_race_first3f_scraped': 0, 'prev_race_last3f_scraped': 0,
        'prev_race_pace_diff_scraped': 0,
    }
    for col, default in _defaults.items():
        if col not in df.columns:
            df[col] = default

    df['頭数'] = num_horses
    df['斤量平均差'] = df['斤量'] - df['斤量'].mean()
    dist = race_info['distance']
    # Match training pd.cut bins=[0,1200,1400,1800,2200,9999] labels=[0,1,2,3,4]
    df['距離カテゴリ'] = 0 if dist <= 1200 else (1 if dist <= 1400 else (2 if dist <= 1800 else (3 if dist <= 2200 else 4)))
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
    # v5以降の全バージョンで英語名特徴量を生成（v15+でも自動対応）
    _ver_num = re.search(r'v(\d+)', version)
    if _ver_num and int(_ver_num.group(1)) >= 5:
        df['sire_enc'] = df['父'].apply(lambda x: use_sire_map.get(x, n_top) if use_sire_map else n_top)
        df['bms_enc'] = df['母の父'].apply(lambda x: use_bms_map.get(x, n_top) if use_bms_map else n_top)

        def enc_loc(loc):
            s = str(loc)
            if '美浦' in s or '美' == s:
                return 0
            if '栗東' in s or '栗' == s:
                return 1
            return 3
        df['location_enc'] = df.get('所属地', pd.Series([''] * len(df))).apply(enc_loc)

        df['horse_weight'] = df['馬体重']
        df['weight_diff'] = df['場体重増減'].fillna(0)
        df['weight_carry'] = df['斤量']
        df['num_horses_val'] = df['頭数']
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

        df['prev_pop'] = df.get('前走人気', pd.Series([8] * len(df))).fillna(8)
        df['prev_odds_log'] = np.log1p(df.get('前走オッズ', pd.Series([15.0] * len(df))).clip(1, 999).fillna(15.0))
        df['prev_last3f'] = df.get('上がり3F', pd.Series([35.5] * len(df))).fillna(35.5)
        df['prev_pass1'] = df.get('通過順平均', pd.Series([8.0] * len(df))).fillna(8.0)
        df['prev_pass4'] = df.get('通過順4', pd.Series([8] * len(df))).fillna(8)
        df['prev_margin'] = 0.0
        df['prev_prize'] = 0.0

        df['prev2_finish'] = df['prev2_finish'].fillna(5) if 'prev2_finish' in df.columns else 5
        df['prev3_finish'] = df['prev3_finish'].fillna(5) if 'prev3_finish' in df.columns else 5
        df['prev4_finish'] = df['prev4_finish'].fillna(5) if 'prev4_finish' in df.columns else 5
        df['prev5_finish'] = df['prev5_finish'].fillna(5) if 'prev5_finish' in df.columns else 5
        df['prev2_last3f'] = df['prev2_last3f'].fillna(35.5) if 'prev2_last3f' in df.columns else 35.5
        df['avg_finish_3r'] = df['avg_finish_3r'].fillna(5.0) if 'avg_finish_3r' in df.columns else 5.0
        df['avg_finish_5r'] = df['avg_finish_5r'].fillna(5.0) if 'avg_finish_5r' in df.columns else 5.0
        df['avg_last3f_3r'] = df.get('上がり3F', pd.Series([35.5] * len(df))).fillna(35.5)
        df['best_finish_3r'] = df['best_finish_3r'].fillna(5) if 'best_finish_3r' in df.columns else 5
        df['best_finish_5r'] = df['best_finish_5r'].fillna(5) if 'best_finish_5r' in df.columns else 5
        df['top3_count_3r'] = df['top3_count_3r'].fillna(0) if 'top3_count_3r' in df.columns else 0
        df['top3_count_5r'] = df['top3_count_5r'].fillna(0) if 'top3_count_5r' in df.columns else 0
        df['finish_trend'] = df['finish_trend'].fillna(0) if 'finish_trend' in df.columns else 0
        df['dist_change'] = 0.0
        df['dist_change_abs'] = 0.0
        df['rest_days'] = df.get('前走間隔', pd.Series([30] * len(df))).fillna(30)
        if '前走間隔' in df.columns:
            df['rest_days'] = df['前走間隔']
        df['rest_category'] = pd.cut(df['rest_days'], bins=[-1, 6, 14, 35, 63, 180, 9999], labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(2)

        # ルックアップテーブルから特徴量を取得
        lookups = load_feature_lookups()
        sire_surf_wr = lookups.get('sire_surface_wr', {})
        sire_dist_wr_map = lookups.get('sire_dist_wr', {})
        bms_surf_wr = lookups.get('bms_surface_wr', {})
        trainer_top3_map = lookups.get('trainer_top3', {})
        jockey_surf_wr = lookups.get('jockey_surface_wr', {})
        frame_cd_wr = lookups.get('frame_course_dist_wr', {})
        horse_stats_map = lookups.get('horse_stats', {})
        horse_name_map = lookups.get('horse_name_to_id', {})
        training_mean = lookups.get('training_mean', 52.0)

        cur_surface = int(df['芝ダート_enc'].iloc[0]) if len(df) > 0 else 0
        cur_dist_cat = int(df['距離カテゴリ'].iloc[0]) if len(df) > 0 else 2
        cur_course = int(df['競馬場コード_enc'].iloc[0]) if len(df) > 0 else 4

        # Sire/BMS/Trainer/Jockey lookups
        df['sire_surface_wr'] = df['父'].apply(
            lambda s: sire_surf_wr.get((s, cur_surface), 0.1))
        df['sire_dist_wr'] = df['父'].apply(
            lambda s: sire_dist_wr_map.get((s, cur_dist_cat), 0.1))
        df['bms_surface_wr'] = df['母の父'].apply(
            lambda s: bms_surf_wr.get((s, cur_surface), 0.1))

        if '調教師' in df.columns:
            df['trainer_top3_calc'] = df['調教師'].apply(
                lambda t: trainer_top3_map.get(t, 0.25))
        else:
            df['trainer_top3_calc'] = 0.25

        if '騎手名' in df.columns:
            df['jockey_surface_wr'] = df['騎手名'].apply(
                lambda j: jockey_surf_wr.get((j, cur_surface), 0.05))
        else:
            df['jockey_surface_wr'] = df['騎手勝率']

        df['frame_course_dist_wr'] = df['枠位置'].apply(
            lambda bp: frame_cd_wr.get((int(bp), cur_course, cur_dist_cat), 0.1))

        # Horse career stats from lookup (with horse_name fallback)
        horse_ids_col = df.get('horse_id_val', pd.Series([0] * len(df)))
        for idx_h in range(len(df)):
            hid = int(horse_ids_col.iloc[idx_h]) if horse_ids_col.iloc[idx_h] else 0
            hs = horse_stats_map.get(hid, {})
            if not hs and idx_h < len(horses):
                hname = horses[idx_h].get('馬名', '') or str(df.iloc[idx_h].get('馬名', ''))
                mapped_id = horse_name_map.get(hname, 0)
                if mapped_id:
                    hs = horse_stats_map.get(mapped_id, {})
            if hs:
                df.loc[df.index[idx_h], 'horse_career_races'] = hs.get('career_races', 5)
                df.loc[df.index[idx_h], 'horse_career_wr'] = hs.get('career_wr', 0.1)
                df.loc[df.index[idx_h], 'horse_career_top3r'] = hs.get('career_top3r', 0.3)
                dc_val = int(df.iloc[idx_h].get('dist_cat', 2))
                sv_val = int(df.iloc[idx_h].get('surface_enc', cur_surface) if 'surface_enc' in df.columns else cur_surface)
                df.loc[df.index[idx_h], 'horse_dist_top3r'] = hs.get('dist_top3', {}).get(dc_val, 0.3)
                df.loc[df.index[idx_h], 'horse_surface_top3r'] = hs.get('surf_top3', {}).get(sv_val, 0.3)
                # 調教データ補完
                lt4f = hs.get('last_training_4f', 0)
                if lt4f > 0:
                    existing_tt = df.iloc[idx_h].get('training_time_filled', 0)
                    if existing_tt == 0 or existing_tt == training_mean:
                        df.loc[df.index[idx_h], 'training_time_filled'] = lt4f
                        df.loc[df.index[idx_h], 'has_training'] = 1
            else:
                df.loc[df.index[idx_h], 'horse_career_races'] = 5
                df.loc[df.index[idx_h], 'horse_career_wr'] = 0.1
                df.loc[df.index[idx_h], 'horse_career_top3r'] = 0.3
                df.loc[df.index[idx_h], 'horse_dist_top3r'] = 0.3
                df.loc[df.index[idx_h], 'horse_surface_top3r'] = 0.3

        # dist_change
        for idx_h in range(len(df)):
            hid = int(horse_ids_col.iloc[idx_h]) if horse_ids_col.iloc[idx_h] else 0
            hs = horse_stats_map.get(hid, {})
            if not hs and idx_h < len(horses):
                hname = horses[idx_h].get('馬名', '')
                mapped_id = horse_name_map.get(hname, 0)
                if mapped_id:
                    hs = horse_stats_map.get(mapped_id, {})
            if hs:
                df.loc[df.index[idx_h], 'dist_change'] = dist - hs.get('last_distance', dist)
                df.loc[df.index[idx_h], 'prev_prize'] = hs.get('last_prize', 0)
        df['dist_change_abs'] = df['dist_change'].abs()

        df['prev_agari_relative'] = 0.0

        # Training features
        if 'training_time_filled' not in df.columns or (df.get('training_time_filled', pd.Series([0])) == 0).all():
            df['training_time_filled'] = training_mean
            df['has_training'] = 0
        df['training_time_filled'] = df['training_time_filled'].replace(0, training_mean)
        if 'has_training' not in df.columns:
            df['has_training'] = (df['training_time_filled'] != training_mean).astype(int)
        df['training_per_dist'] = training_mean / max(dist / 200, 1)

        # 調教関連 — scrape_training.py で実タイム or ランク推定を取得
        df['wood_best_4f_filled'] = df.get('wood_best_4f_filled', pd.Series([training_mean] * len(df)))
        df['has_wood_training'] = df.get('has_wood_training', pd.Series([0] * len(df)))
        df['wood_count_2w'] = df.get('wood_count_2w', pd.Series([0] * len(df)))
        df['sakaro_best_4f_filled'] = df.get('sakaro_best_4f_filled', pd.Series([training_mean + 1.0] * len(df)))
        df['sakaro_best_3f_filled'] = df.get('sakaro_best_3f_filled', pd.Series([training_mean - 13.0] * len(df)))
        df['has_sakaro_training'] = df.get('has_sakaro_training', pd.Series([0] * len(df)))
        df['total_training_count'] = df.get('total_training_count', pd.Series([0] * len(df)))

        # race_idがあれば scrape_training で調教データ取得
        if race_id and (df['wood_best_4f_filled'] == training_mean).all():
            try:
                from scrape_training import get_training_features, fetch_training_times, fetch_stable_comments
                _tf = get_training_features(race_id, len(df))
                for idx_h in range(len(df)):
                    _umaban = int(df.iloc[idx_h].get('馬番', df.iloc[idx_h].get('horse_num', 0)))
                    if _umaban in _tf:
                        for _k, _v in _tf[_umaban].items():
                            df.loc[df.index[idx_h], _k] = _v
            except Exception:
                # Fallback: oikiri ranks
                oikiri_ranks = fetch_oikiri_ranks(race_id)
                if oikiri_ranks:
                    for idx_h in range(len(df)):
                        _umaban = int(df.iloc[idx_h].get('馬番', df.iloc[idx_h].get('horse_num', 0)))
                        _rank = oikiri_ranks.get(_umaban, '')
                        if _rank in RANK_TO_WOOD_4F:
                            df.loc[df.index[idx_h], 'wood_best_4f_filled'] = RANK_TO_WOOD_4F[_rank]
                            df.loc[df.index[idx_h], 'has_wood_training'] = 1
                            df.loc[df.index[idx_h], 'wood_count_2w'] = 2 if _rank in ('A', 'B') else 1
                            df.loc[df.index[idx_h], 'sakaro_best_4f_filled'] = RANK_TO_SAKARO_4F[_rank]
                            df.loc[df.index[idx_h], 'sakaro_best_3f_filled'] = RANK_TO_SAKARO_3F[_rank]
                            df.loc[df.index[idx_h], 'has_sakaro_training'] = 1
                            df.loc[df.index[idx_h], 'total_training_count'] = 4 if _rank in ('A', 'B') else 2
                            df.loc[df.index[idx_h], 'training_time_filled'] = RANK_TO_WOOD_4F[_rank]
                            df.loc[df.index[idx_h], 'has_training'] = 1

        # ===== Stable comment score =====
        if 'stable_comment_score' not in df.columns:
            df['stable_comment_score'] = 0.0
        if race_id and (df['stable_comment_score'] == 0).all():
            try:
                from scrape_training import fetch_stable_comments
                _comments = fetch_stable_comments(race_id)
                if _comments:
                    for idx_h in range(len(df)):
                        _umaban = int(df.iloc[idx_h].get('馬番', df.iloc[idx_h].get('horse_num', 0)))
                        if _umaban in _comments:
                            df.loc[df.index[idx_h], 'stable_comment_score'] = float(_comments[_umaban].get('score', 0))
            except Exception:
                pass

        # ===== Weight trend features (weight_ma5, weight_trend, weight_peak_diff) =====
        try:
            for idx_h in range(len(df)):
                _wh = horses[idx_h].get('weight_history', []) if idx_h < len(horses) else []
                _current_w = float(df.iloc[idx_h].get('馬体重', 480))
                if _wh and len(_wh) >= 2:
                    _recent = _wh[:5]
                    df.loc[df.index[idx_h], 'weight_ma5'] = float(np.mean(_recent))
                    # Linear trend (slope): simple least squares
                    _n = len(_recent)
                    _x = np.arange(_n, dtype=float)
                    _y = np.array(_recent, dtype=float)
                    _slope = float(np.polyfit(_x, _y, 1)[0]) if _n >= 2 else 0.0
                    df.loc[df.index[idx_h], 'weight_trend'] = _slope
                    df.loc[df.index[idx_h], 'weight_peak_diff'] = _current_w - float(max(_recent))
                elif _current_w > 0:
                    df.loc[df.index[idx_h], 'weight_ma5'] = _current_w
                    df.loc[df.index[idx_h], 'weight_trend'] = 0.0
                    df.loc[df.index[idx_h], 'weight_peak_diff'] = 0.0
        except Exception:
            pass
        for _wc in ['weight_ma5', 'weight_trend', 'weight_peak_diff']:
            if _wc not in df.columns:
                df[_wc] = 480.0 if _wc == 'weight_ma5' else 0.0
            df[_wc] = pd.to_numeric(df[_wc], errors='coerce').fillna(480.0 if _wc == 'weight_ma5' else 0.0)

        # ===== Lap / Pace features (prev_race_first3f, prev_race_last3f, prev_race_pace_diff) =====
        # Use scraped data from get_horse_stats if available
        _lap_filled = False
        try:
            for idx_h in range(len(df)):
                if idx_h < len(horses):
                    _f3 = horses[idx_h].get('prev_race_first3f_scraped', 0)
                    _l3 = horses[idx_h].get('prev_race_last3f_scraped', 0)
                    _pd = horses[idx_h].get('prev_race_pace_diff_scraped', 0)
                    if _f3 > 0 and _l3 > 0:
                        df.loc[df.index[idx_h], 'prev_race_first3f'] = float(_f3)
                        df.loc[df.index[idx_h], 'prev_race_last3f'] = float(_l3)
                        df.loc[df.index[idx_h], 'prev_race_pace_diff'] = float(_pd)
                        # prev_agari_relative = horse's agari - race average agari
                        _horse_agari = float(df.iloc[idx_h].get('上がり3F', 35.5) if '上がり3F' in df.columns else 35.5)
                        df.loc[df.index[idx_h], 'prev_agari_relative'] = _horse_agari - float(_l3)
                        _lap_filled = True
        except Exception:
            pass
        if 'prev_race_first3f' not in df.columns or (df.get('prev_race_first3f', pd.Series([0])) == 0).all():
            df['prev_race_first3f'] = 35.0
            df['prev_race_last3f'] = 35.5
            df['prev_race_pace_diff'] = 0.5
            df['prev_agari_relative'] = 0.0

        # 脚質・ペース特徴量
        if 'running_style' not in df.columns or (df.get('running_style', pd.Series([0])) == 0).all():
            for _ih in range(len(df)):
                _pass_avg = float(df.iloc[_ih].get('通過順平均', 5.0) if '通過順平均' in df.columns else 5.0)
                _rs = 1 if _pass_avg <= 2 else (2 if _pass_avg <= 5 else (3 if _pass_avg <= 10 else 4))
                df.loc[df.index[_ih], 'running_style'] = float(_rs)
            _n_front = (df['running_style'] <= 2).sum()
            _front_ratio = _n_front / max(num_horses, 1)
            _pace = 0 if _front_ratio < 0.2 else (2 if _front_ratio > 0.4 else 1)
            df['predicted_pace'] = float(_pace)
            _adv_map = {(0,1):2,(0,2):1,(0,3):-1,(0,4):-2,(1,1):0,(1,2):0.5,(1,3):0.5,(1,4):0,(2,1):-2,(2,2):-1,(2,3):1,(2,4):2}
            df['pace_advantage'] = df['running_style'].apply(lambda s: _adv_map.get((_pace, int(s)), 0.0))
            df['style_vs_pace'] = df['running_style'] * 10 + df['predicted_pace']

        df['same_dist_rate'] = 0.3
        df['same_course_rate'] = 0.3
        df['same_surface_rate'] = 0.3
        df['horse_win_rate'] = df.get('horse_career_wr', pd.Series([0.1] * len(df)))
        df['horse_top3_rate'] = df.get('horse_career_top3r', pd.Series([0.3] * len(df)))
        df['horse_race_count'] = df.get('horse_career_races', pd.Series([5] * len(df)))
        df['jockey_course_wr'] = df['騎手勝率']
        df['jockey_dist_wr'] = df['騎手勝率']
        df['jockey_top3'] = df['騎手勝率'] * 3
        df['trainer_wr'] = 0.08
        df['trainer_top3'] = df['trainer_top3_calc']
        df['weight_dist'] = df['馬体重'] * df['距離(m)'] / 10000.0
        df['age_season'] = df['馬齢'] * 10 + df['season']
        df['carry_per_weight'] = df['斤量'] / df['馬体重'].clip(1) * 100
        df['horse_num_ratio'] = df['馬番'] / df['頭数'].clip(1)
        df['weight_diff_abs'] = 0.0
        df['surface_enc'] = df['芝ダート_enc']
        df['jockey_wr_calc'] = df['騎手勝率']
        df['jockey_course_wr_calc'] = df['騎手勝率']
        df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
        df['surface_dist_enc'] = df['芝ダート_enc'] * 10 + df['dist_cat']
        df['cond_surface'] = df['馬場状態_enc'] * 10 + df['芝ダート_enc']
        df['course_surface'] = df['競馬場コード_enc'] * 10 + df['芝ダート_enc']
        df['is_nar'] = 0

    # オッズ判定
    has_odds = (odds_dict and len(odds_dict) > 0) or \
               ('単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any())
    if has_odds and '単勝オッズ' in df.columns:
        df['odds_log'] = np.log1p(df['単勝オッズ'].clip(1, 999).replace(0, 15.0))
        # リアルタイムオッズがある場合、prev_odds_log も更新
        has_real_odds = df['単勝オッズ'] > 0
        if has_real_odds.any() and 'prev_odds_log' in df.columns:
            df.loc[has_real_odds, 'prev_odds_log'] = df.loc[has_real_odds, 'odds_log']
    else:
        df['odds_log'] = np.log1p(pd.Series([15.0] * len(df)))

    # Pattern B 当日特徴量
    if is_live:
        df['weight_change'] = df['場体重増減'].fillna(0)
        df['weight_change_abs'] = df['weight_change'].abs()
        weather_str = str(race_info.get('weather', '晴'))
        weather_map = {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3}
        df['weather_enc'] = weather_map.get(weather_str, 0)

        if '人気順位' in df.columns and (df['人気順位'] > 0).any():
            df['pop_rank'] = df['人気順位'].replace(0, 8)
        elif has_odds and '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any():
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
                df['moisture_rate'] = 0.0
        else:
            df['cushion_value'] = 0.0
            df['moisture_rate'] = 0.0

        if weather_info:
            df['temperature'] = weather_info.get('temperature', 0)
            df['humidity'] = weather_info.get('humidity', 0)
            df['wind_speed'] = weather_info.get('wind_speed', 0)
            df['precipitation'] = weather_info.get('precipitation', 0)
        else:
            df['temperature'] = 0.0
            df['humidity'] = 0.0
            df['wind_speed'] = 0.0
            df['precipitation'] = 0.0

    # ===== V12新特徴量 =====

    # 1-3. Speed index (index_max, index_run1, index_avg5)
    _si_found = 0
    if race_id:
        # Source 1: CSV (historical data)
        try:
            _si_csv = os.path.join(BASE_DIR, 'data', 'netkeiba_speed_index.csv')
            if os.path.exists(_si_csv):
                _si = pd.read_csv(_si_csv, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
                _si_race = _si[_si['race_id'] == str(race_id)]
                for idx_h in range(len(df)):
                    _uma = str(int(df.iloc[idx_h].get('馬番', df.iloc[idx_h].get('horse_num', 0))))
                    _match = _si_race[_si_race['umaban'] == _uma]
                    if len(_match) > 0:
                        _row = _match.iloc[0]
                        df.loc[df.index[idx_h], 'index_max_filled'] = pd.to_numeric(_row.get('index_max', 0), errors='coerce') or 0
                        df.loc[df.index[idx_h], 'index_avg5_filled'] = pd.to_numeric(_row.get('index_avg5', 0), errors='coerce') or 0
                        df.loc[df.index[idx_h], 'index_run1_filled'] = pd.to_numeric(_row.get('index_run1', 0), errors='coerce') or 0
                        _si_found += 1
        except Exception:
            pass
        # Source 2: Premium cache (weekly_premium_cache/{date}/premium_cache.json)
        if _si_found == 0:
            try:
                import json as _json_si
                import glob as _glob_si
                _cache_dirs = sorted(_glob_si.glob(os.path.join(BASE_DIR, 'data', 'weekly_premium_cache', '*', 'premium_cache.json')))
                for _cache_path in reversed(_cache_dirs):
                    with open(_cache_path, 'r', encoding='utf-8') as _f:
                        _cache = _json_si.load(_f)
                    _race_cache = _cache.get(str(race_id), {})
                    _si_data = _race_cache.get('speed_index', {})
                    if _si_data:
                        for idx_h in range(len(df)):
                            _uma = str(int(df.iloc[idx_h].get('馬番', df.iloc[idx_h].get('horse_num', 0))))
                            if _uma in _si_data:
                                _sd = _si_data[_uma]
                                df.loc[df.index[idx_h], 'index_max_filled'] = float(_sd.get('index_max', 0))
                                df.loc[df.index[idx_h], 'index_avg5_filled'] = float(_sd.get('index_avg5', 0))
                                df.loc[df.index[idx_h], 'index_run1_filled'] = float(_sd.get('index_run1', 0))
                                _si_found += 1
                        break
            except Exception:
                pass
    # Speed index: mean fill with training data average (not 0)
    _si_defaults = {'index_max_filled': 1045.0, 'index_run1_filled': 1040.0, 'index_avg5_filled': 1035.0}
    for _col, _default in _si_defaults.items():
        if _col not in df.columns:
            df[_col] = _default
        df[_col] = pd.to_numeric(df[_col], errors='coerce').fillna(_default)
        # Replace 0 with default (0 means no match, not actual 0)
        df.loc[df[_col] == 0, _col] = _default

    # 4-5. Training 1F and intensity
    # Source 1: premium cache or scrape_training (already fetched upstream)
    _int_map = {'一杯': 3, '強め': 2, '馬なり': 1}
    if race_id:
        try:
            from scrape_training import fetch_training_times
            _tt_data = fetch_training_times(race_id)
            if _tt_data:
                for idx_h in range(len(df)):
                    _uma = int(df.iloc[idx_h].get('馬番', df.iloc[idx_h].get('horse_num', 0)))
                    if _uma in _tt_data:
                        _td = _tt_data[_uma]
                        _1f = _td.get('time_1f', 0)
                        if _1f and _1f > 0:
                            df.loc[df.index[idx_h], 'time_1f_last_filled'] = float(_1f)
                        _inten = str(_td.get('intensity', '')).strip()
                        if _inten:
                            df.loc[df.index[idx_h], 'training_intensity_enc'] = _int_map.get(_inten, 0)
        except Exception:
            pass
    # Source 2: fallback to CSV if not filled
    if 'time_1f_last_filled' not in df.columns or (df.get('time_1f_last_filled', pd.Series([0])) == 0).all():
        try:
            _tt_csv = os.path.join(BASE_DIR, 'data', 'netkeiba_training_times.csv')
            if os.path.exists(_tt_csv) and race_id:
                _tt = pd.read_csv(_tt_csv, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
                _tt_race = _tt[_tt['race_id'] == str(race_id)]
                for idx_h in range(len(df)):
                    _uma = str(int(df.iloc[idx_h].get('馬番', df.iloc[idx_h].get('horse_num', 0))))
                    _match = _tt_race[_tt_race['umaban'] == _uma]
                    if len(_match) > 0:
                        _row = _match.iloc[0]
                        _1f = pd.to_numeric(_row.get('time_1f', 0), errors='coerce')
                        if _1f and _1f > 0:
                            df.loc[df.index[idx_h], 'time_1f_last_filled'] = _1f
                        _inten = str(_row.get('intensity', '')).strip()
                        df.loc[df.index[idx_h], 'training_intensity_enc'] = _int_map.get(_inten, 0)
        except Exception:
            pass
    for _col in ['time_1f_last_filled', 'training_intensity_enc']:
        if _col not in df.columns:
            df[_col] = 12.5 if 'time_1f' in _col else 0
        df[_col] = pd.to_numeric(df[_col], errors='coerce').fillna(12.5 if 'time_1f' in _col else 0)

    # 6. Sire shinba top3r (from static CSV)
    try:
        _ss_csv = os.path.join(BASE_DIR, 'data', 'sire_shinba_stats.csv')
        if os.path.exists(_ss_csv):
            _ss = pd.read_csv(_ss_csv, encoding='utf-8')
            _ss_map = dict(zip(_ss['sire'], _ss['top3_rate']))
            _ss_mean = _ss['top3_rate'].mean()
            for idx_h in range(len(df)):
                _father = horses[idx_h].get('父', '') if idx_h < len(horses) else ''
                df.loc[df.index[idx_h], 'sire_shinba_top3r'] = _ss_map.get(_father, _ss_mean)
    except Exception:
        pass
    if 'sire_shinba_top3r' not in df.columns:
        df['sire_shinba_top3r'] = 0.22

    # 7. PCI (Pace Change Index)
    if 'prev_race_first3f' in df.columns and 'prev_race_last3f' in df.columns:
        _f3f = df['prev_race_first3f'].replace(0, np.nan)
        _l3f = df['prev_race_last3f'].replace(0, np.nan)
        df['pci'] = (_l3f / _f3f).fillna(1.0)
    else:
        df['pci'] = 1.0

    # 新馬評価（表示用、モデル特徴量外）
    if race_info.get('race_name', '') and '新馬' in str(race_info.get('race_name', '')):
        shinba_scores = []
        for h in horses:
            cs = h.get('新馬スコア', None) or h.get('shinba_comment_score', None)
            shinba_scores.append(int(cs) if cs is not None else 0)
        df['shinba_comment_score'] = shinba_scores


    # ===== JRDB特徴量統合 (v13) =====
    try:
        from tools.jrdb_features import merge_jrdb_predict_features
        df = merge_jrdb_predict_features(df, race_id)
    except Exception as e:
        print(f"[JRDB] 特徴量取得スキップ: {e}")

    # ===== netkeibaオッズ変動特徴量（JRDB OZがない場合のフォールバック） =====
    _odds_feats_zero = (
        ('odds_change_rate' not in df.columns or (df.get('odds_change_rate', pd.Series([0])) == 0).all()) and
        ('pop_rank_change' not in df.columns or (df.get('pop_rank_change', pd.Series([0])) == 0).all())
    )
    if _odds_feats_zero and race_id and odds_dict:
        try:
            odds_full = fetch_realtime_odds_full(race_id)
            if odds_full:
                _base = load_odds_base(race_id)
                if not _base:
                    # baselineが無ければ今このタイミングを基準として保存（次以降のため）
                    save_odds_base(race_id, odds_full)
                    print(f"[ODDS] 基準オッズ未登録 → 現在値を基準として保存（差分は0）")
                else:
                    df = compute_odds_change_features(df, race_id, odds_full)
                    _n_change = (df.get('odds_change_rate', pd.Series([0])) != 0).sum()
                    print(f"[ODDS] 基準オッズ vs リアルタイム比較完了 ({_n_change}/{len(df)}馬で変動検出)")
        except Exception as e:
            print(f"[WARN] netkeibaオッズ変動計算失敗: {e}")

    # ===== v15新特徴量（騎手×馬相性・輸送距離・コース改修・外厩） =====
    _ver_num_v15 = re.search(r'v(\d+)', version)
    if _ver_num_v15 and int(_ver_num_v15.group(1)) >= 15:
        # 輸送距離（予測時に計算可能）
        if 'transport_distance_km' not in df.columns or (df.get('transport_distance_km', pd.Series([0])) == 0).all():
            try:
                _venue_coords = {
                    0:(43.06,141.35),1:(41.80,140.73),2:(37.75,140.47),3:(37.89,139.02),
                    4:(35.67,139.48),5:(35.76,140.00),6:(35.05,137.05),7:(34.93,135.72),
                    8:(34.78,135.36),9:(33.88,130.87),
                }
                _tc_coords = {0:(36.03,140.18), 1:(34.98,136.00)}
                import math as _math
                def _hav(lat1,lon1,lat2,lon2):
                    R=6371.0; dlat=_math.radians(lat2-lat1); dlon=_math.radians(lon2-lon1)
                    a=_math.sin(dlat/2)**2+_math.cos(_math.radians(lat1))*_math.cos(_math.radians(lat2))*_math.sin(dlon/2)**2
                    return R*2*_math.atan2(_math.sqrt(a),_math.sqrt(1-a))
                for idx_h in range(len(df)):
                    _loc = int(df.iloc[idx_h].get('location_enc', 0))
                    _crs = int(df.iloc[idx_h].get('course_enc', cur_course))
                    if _loc in _tc_coords and _crs in _venue_coords:
                        _d = _hav(_tc_coords[_loc][0],_tc_coords[_loc][1],_venue_coords[_crs][0],_venue_coords[_crs][1])
                        df.loc[df.index[idx_h], 'transport_distance_km'] = round(_d, 1)
                df['is_long_transport'] = (df.get('transport_distance_km', pd.Series([0]*len(df))) > 500).astype(int)
            except Exception:
                df['transport_distance_km'] = 0.0
                df['is_long_transport'] = 0

        # コース改修フラグ（予測時に計算可能）
        if 'course_renovated' not in df.columns:
            _reno_dates = [(6,2012,3),(7,2023,4)]  # 中京2012/3, 京都2023/4
            _now = datetime.now()
            _race_ym = _now.year * 12 + _now.month
            df['course_renovated'] = 0
            df['post_renovation_flag'] = 0
            for _rc, _ry, _rm in _reno_dates:
                _reno_ym = _ry * 12 + _rm
                if cur_course == _rc:
                    if _race_ym >= _reno_ym:
                        df['post_renovation_flag'] = 1
                        df['course_renovated'] = 1  # v16: 永久フラグ化

        # 騎手変更 + 騎手×馬相性（get_horse_statsで取得済みのjockey_resultsを使用）
        # リーディング上位騎手リスト（jockey_change_to_top判定用）
        _top_jockeys = {'C.ルメール','川田将雅','戸崎圭太','横山武史','R.ムーア','武豊',
                        '松山弘平','坂井瑠星','D.レーン','横山和生','岩田望来','吉田隼人'}
        if 'jockey_change' not in df.columns:
            df['jockey_change'] = 0
            df['jockey_change_to_top'] = 0
            df['jockey_horse_rides'] = 0
            df['jockey_horse_wr'] = 0.0
            df['jockey_horse_top3r'] = 0.0
            for idx_h in range(min(len(df), len(horses))):
                _prev_jockey = horses[idx_h].get('prev_jockey', '')
                _cur_jockey = horses[idx_h].get('騎手名', '')
                # 騎手変更判定
                if _prev_jockey and _cur_jockey and str(_prev_jockey) != str(_cur_jockey):
                    df.loc[df.index[idx_h], 'jockey_change'] = 1
                    if str(_cur_jockey) in _top_jockeys:
                        df.loc[df.index[idx_h], 'jockey_change_to_top'] = 1
                # 騎手×馬相性（過去成績から計算）
                _jr = horses[idx_h].get('jockey_results', {})
                if _jr and _cur_jockey:
                    _jr_finishes = _jr.get(str(_cur_jockey), [])
                    if _jr_finishes:
                        _n = len(_jr_finishes)
                        df.loc[df.index[idx_h], 'jockey_horse_rides'] = _n
                        df.loc[df.index[idx_h], 'jockey_horse_wr'] = sum(1 for f in _jr_finishes if f == 1) / _n
                        df.loc[df.index[idx_h], 'jockey_horse_top3r'] = sum(1 for f in _jr_finishes if f <= 3) / _n
        for _v15f in ['jockey_horse_rides', 'jockey_horse_wr', 'jockey_horse_top3r',
                       'jockey_change', 'jockey_change_to_top', 'transport_distance_km',
                       'is_long_transport', 'course_renovated', 'post_renovation_flag', 'gaisha_rank']:
            if _v15f not in df.columns:
                df[_v15f] = 0

        # PACI特徴量（JRDBマージで取得済み、なければ0）
        for _paci in ['paci_sogo_mark', 'paci_idm_mark', 'paci_jockey_mark', 'paci_train_mark',
                       'paci_manken_idx', 'paci_goal_rank', 'paci_dochu_rank', 'paci_goal_diff',
                       'paci_jockey_exp_wr', 'paci_jockey_exp_3rd', 'paci_ninki_idx']:
            if _paci not in df.columns:
                df[_paci] = 0

    # ===== V18 candidate features (Phase 11、 5/10) =====
    # 全 15 features、 Phase 11 = scaffold + default values 段階。
    # 5/12+ で JRDB data lookup + expanding window 集計 を本実装予定。
    # V15 model_data['features'] に含まれない → V15 推論影響 なし (use_features fallback 通過)。
    # V18 学習時にこれらを取り込む form。

    # A. 外厩 (training farm) features - 4 features
    # source: JRDB UKC (馬基本) / CHA (前走詳細) の外厩 column
    df['gaika_id_enc'] = 0           # 外厩 ID encoded (TODO: UKC.外厩 lookup)
    df['gaika_top3r_3r'] = 0.33      # 過去 3 R 外厩 top3 率 (expanding) baseline 33%
    df['gaika_winrate'] = 0.20       # 外厩別 通算 勝率 baseline 20%
    df['gaika_dist_winrate'] = 0.20  # 外厩 X 距離別 勝率 baseline 20%

    # B. 時系列オッズ features - 4 features
    # source: JRDB OT/OV/OW (時系列オッズ) または save_odds_base 蓄積データ
    df['odds_change_3h_v18'] = 0.0    # 3h 前 → 直前 odds 変化率
    df['odds_change_30m_v18'] = 0.0   # 30m 前 → 直前 odds 変化率
    df['popularity_shift_v18'] = 0    # 朝人気 - 直前人気 (整数差)
    df['odds_volatility_v18'] = 0.0   # 期間内 odds std / mean

    # C. 騎手マスタ拡張 features - 4 features
    # source: JRDB KKA (既統合 90.4%) を distance/track/class/trainer で再集計
    df['jockey_dist_winrate'] = 0.10   # 騎手 X 距離 勝率 baseline 10%
    df['jockey_track_winrate'] = 0.10  # 騎手 X 芝/ダート 勝率
    df['jockey_class_winrate'] = 0.10  # 騎手 X クラス 勝率
    df['jockey_x_trainer_wr'] = 0.15   # 騎手 X 調教師 連携 勝率 baseline 15%

    # D. 返し馬 / パドック features - 3 features
    # source: JRDB CYB (調教コメント) / TYB (直前) - 既存 jrdb_paddock_idx を超えた評価
    df['return_horse_score'] = 0.0   # 返し馬 評価 (-3 〜 +3)
    df['paddock_eval_v18'] = 0.0     # パドック評価 v18 (jrdb_paddock_idx と独立)
    df['saddle_room_score'] = 0.0    # 装鞍所 状態 (-3 〜 +3)

    # ===== V18 candidate features END =====

    # 必要な特徴量の確保
    use_features = model_data.get('features')
    if use_features:
        for f in use_features:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    return df


# === 予測実行 ===

def calc_pace_advantage(distance, surface, condition, num_horses):
    """展開有利度を計算"""
    scores = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    if surface == 'ダ':
        scores[1] += 0.1; scores[2] += 0.15; scores[3] -= 0.05; scores[4] -= 0.15
    else:
        if distance <= 1400:
            scores[1] += 0.15; scores[2] += 0.1; scores[3] -= 0.05; scores[4] -= 0.2
        elif distance >= 2400:
            scores[3] += 0.1; scores[4] += 0.05; scores[1] -= 0.1
    if condition in ('重', '不良', '不'):
        scores[1] += 0.05; scores[2] += 0.08; scores[4] -= 0.1
    elif condition in ('稍', '稍重'):
        scores[2] += 0.04; scores[4] -= 0.05
    if num_horses <= 10:
        scores[1] += 0.1; scores[2] += 0.05; scores[4] -= 0.1
    elif num_horses >= 16:
        scores[3] += 0.08; scores[4] += 0.05; scores[1] -= 0.05
    return scores


def predict_race(df, model_data, odds_available=False, race_info=None):
    """予測を実行してスコア・順位を付与（全スクリプト共通）"""
    use_features = model_data.get('features')
    if not use_features:
        print("[ERROR] モデルに特徴量リストがありません")
        return df
    use_model = model_data['model']
    X_full = df[use_features].values
    n = len(df)

    # LGB/XGBは学習時の特徴量数に合わせる（124列）
    # pkl.gzのfeaturesリスト(129)にはPattern B TYB特徴量(5)が含まれるが
    # LGB/XGBは先頭124列で学習されている
    n_lgb_features = use_model.num_feature() if hasattr(use_model, 'num_feature') else X_full.shape[1]
    X = X_full[:, :n_lgb_features]

    if hasattr(use_model, 'predict_proba'):
        proba = use_model.predict_proba(X)
        ai_scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, :3].sum(axis=1)
    else:
        ai_scores = use_model.predict(X)

    # XGB/CatBoost/FT ensemble (if available in model data)
    ens_w = model_data.get('ensemble_weights', {})
    xgb_m = model_data.get('xgb_model')
    cb_m = model_data.get('cb_model')
    ft_state = model_data.get('ft_model_state')
    ft_config = model_data.get('ft_model_config')
    ft_scaler_mean = model_data.get('ft_scaler_mean')
    ft_scaler_scale = model_data.get('ft_scaler_scale')

    has_ensemble = (xgb_m is not None or cb_m is not None
                    or (ft_state is not None and _TORCH_AVAILABLE))
    if has_ensemble:
        w_lgb = ens_w.get('lgb', 0.5)
        w_xgb = ens_w.get('xgb', 0.3)
        w_cb = ens_w.get('cb', 0.0)
        w_ft = ens_w.get('ft', 0.0)

        # FT-Transformer inference
        ft_pred = None
        if ft_state is not None and ft_scaler_mean is not None and _TORCH_AVAILABLE:
            try:
                # FT model was trained on Pattern A features (124).
                # For live models with extra features, use only the first N features
                # matching the scaler dimension.
                n_ft_features = len(ft_scaler_mean)
                X_ft = X[:, :n_ft_features].astype(np.float32)

                # Apply StandardScaler
                mean = np.array(ft_scaler_mean, dtype=np.float32)
                scale = np.array(ft_scaler_scale, dtype=np.float32)
                scale[scale == 0] = 1.0  # avoid division by zero
                X_ft_scaled = (X_ft - mean) / scale

                # Reconstruct model using actual weight dimensions, not stored config
                actual_n_features = n_ft_features
                cfg = dict(ft_config) if ft_config else {}
                cfg['n_features'] = actual_n_features
                ft_model = FTTransformer(
                    n_features=cfg['n_features'],
                    d_token=cfg.get('d_token', 64),
                    n_heads=cfg.get('n_heads', 4),
                    n_layers=cfg.get('n_layers', 3),
                    dropout=cfg.get('dropout', 0.1),
                )
                ft_model.load_state_dict(ft_state)
                ft_model.eval()

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ft_model.to(device)
                X_tensor = torch.tensor(X_ft_scaled, dtype=torch.float32).to(device)

                with torch.no_grad():
                    logits = ft_model(X_tensor)
                    ft_pred = torch.sigmoid(logits).cpu().numpy()
                print(f"[FT] FT-Transformer推論完了 (n_features={actual_n_features})")
            except Exception as e:
                print(f"[WARN] FT-Transformer推論失敗 (LGB+XGBにフォールバック): {e}")
                ft_pred = None

        # If FT unavailable, redistribute its weight
        if ft_pred is None:
            w_ft = 0.0

        # IntraRace Attention inference
        ir_pred = None
        w_ir = ens_w.get('ir', 0.0)
        ir_state = model_data.get('ir_model_state')
        ir_config = model_data.get('ir_model_config')
        ir_scaler_mean = model_data.get('ir_scaler_mean')
        ir_scaler_scale = model_data.get('ir_scaler_scale')

        if ir_state is not None and ir_scaler_mean is not None and _TORCH_AVAILABLE and w_ir > 0:
            try:
                n_ir_features = len(ir_scaler_mean)
                X_ir = X[:, :n_ir_features].astype(np.float32)

                # Scale
                ir_mean = np.array(ir_scaler_mean, dtype=np.float32)
                ir_scale = np.array(ir_scaler_scale, dtype=np.float32)
                ir_scale[ir_scale == 0] = 1.0
                X_ir_scaled = (X_ir - ir_mean) / ir_scale

                # Pad to race batch: (1, max_horses, n_features)
                MAX_HORSES = 28
                n_horses = len(X_ir_scaled)
                x_padded = np.zeros((1, MAX_HORSES, n_ir_features), dtype=np.float32)
                mask_padded = np.zeros((1, MAX_HORSES), dtype=np.float32)
                n_fill = min(n_horses, MAX_HORSES)
                x_padded[0, :n_fill] = X_ir_scaled[:n_fill]
                mask_padded[0, :n_fill] = 1.0

                cfg_ir = dict(ir_config) if ir_config else {}
                cfg_ir['n_features'] = n_ir_features
                ir_model = IntraRaceAttention(
                    n_features=cfg_ir['n_features'],
                    d_model=cfg_ir.get('d_model', 64),
                    n_heads=cfg_ir.get('n_heads', 4),
                    n_layers=cfg_ir.get('n_layers', 2),
                    dropout=cfg_ir.get('dropout', 0.1),
                )
                ir_model.load_state_dict(ir_state)
                ir_model.eval()

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ir_model.to(device)
                x_t = torch.tensor(x_padded, dtype=torch.float32).to(device)
                m_t = torch.tensor(mask_padded, dtype=torch.float32).to(device)

                with torch.no_grad():
                    logits = ir_model(x_t, m_t)
                    probs = torch.sigmoid(logits).cpu().numpy()
                ir_pred = probs[0, :n_fill]
                print(f"[IR] IntraRace Attention推論完了 (n_horses={n_fill})")
            except Exception as e:
                print(f"[WARN] IntraRace Attention推論失敗: {e}")
                ir_pred = None
                w_ir = 0.0
        else:
            w_ir = 0.0

        total_w = w_lgb + w_xgb + w_cb + w_ft + w_ir
        if total_w <= 0:
            total_w = 1.0
        combined = ai_scores * (w_lgb / total_w)

        if xgb_m is not None:
            try:
                import xgboost as xgb_lib
                xgb_pred = xgb_m.predict(xgb_lib.DMatrix(X))
                combined += xgb_pred * (w_xgb / total_w)
            except Exception:
                combined += ai_scores * (w_xgb / total_w)
        else:
            combined += ai_scores * (w_xgb / total_w)

        if cb_m is not None:
            try:
                cb_pred = cb_m.predict_proba(X)[:, 1]
                combined += cb_pred * (w_cb / total_w)
            except Exception:
                combined += ai_scores * (w_cb / total_w)
        elif w_cb > 0:
            combined += ai_scores * (w_cb / total_w)

        if ft_pred is not None:
            combined += ft_pred * (w_ft / total_w)

        if ir_pred is not None:
            combined += ir_pred * (w_ir / total_w)

        ai_scores = combined

    # 補助スコア
    pop_scores = df['人気傾向'].values if '人気傾向' in df.columns else np.full(n, 0.5)
    apt_scores = np.full(n, 0.5)
    if '距離適性' in df.columns and '馬場適性' in df.columns:
        apt_scores = (df['距離適性'].values + df['馬場適性'].values) / 2.0

    # 脚質スコア
    if race_info:
        pace_scores_map = calc_pace_advantage(
            race_info.get('distance', 1600),
            race_info.get('surface', '芝'),
            race_info.get('condition', '良'),
            n
        )
    else:
        pace_scores_map = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    if '脚質' in df.columns:
        pace_scores = np.array([
            np.clip(pace_scores_map.get(int(v), 0.5), 0.0, 1.0)
            if v != 0 else 0.5
            for v in df['脚質'].values
        ])
    else:
        pace_scores = np.full(n, 0.5)

    # 上がり3Fスコア
    if '上がり3F' in df.columns:
        agari_scores = np.clip(1.0 - (df['上がり3F'].values - 33.0) / 5.0, 0.0, 1.0)
    else:
        agari_scores = np.full(n, 0.5)

    # コース適性スコア
    course_scores = df['コース適性'].values if 'コース適性' in df.columns else np.full(n, 0.5)

    # その他スコア（血統+複勝率）
    blood = df['血統スコア'].values if '血統スコア' in df.columns else np.full(n, 0.5)
    fukusho = df['複勝率'].values if '複勝率' in df.columns else np.full(n, 0.0)
    other_scores = (blood + fukusho) / 2.0

    # ===== FINAL SCORE（v8/v9 統一ウェイト） =====
    if odds_available and '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any():
        odds_vals = df['単勝オッズ'].replace(0, 15.0)
        odds_scores = np.clip(1.0 - np.log1p(odds_vals) / np.log1p(100.0), 0.0, 1.0)
        final_scores = (
            ai_scores * 0.65 + odds_scores * 0.08 + apt_scores * 0.06
            + pace_scores * 0.06 + agari_scores * 0.05 + course_scores * 0.04
            + other_scores * 0.03 + pop_scores * 0.03
        )
    else:
        final_scores = (
            ai_scores * 0.70 + pop_scores * 0.06 + apt_scores * 0.06
            + pace_scores * 0.06 + agari_scores * 0.05 + course_scores * 0.04
            + other_scores * 0.03
        )

    df['スコア'] = final_scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    return df


def calc_horse_ev(df_sorted, odds_dict=None):
    """各馬のEV（期待値）を計算。EV = 複勝圏確率 × 想定三連複配当倍率。
    odds_dict: {馬番: 単勝オッズ} (任意)
    Returns: {馬番: ev_value}
    """
    ev_map = {}
    if len(df_sorted) < 3:
        return ev_map
    scores = df_sorted['スコア'].values
    score_sum = scores.sum()
    if score_sum <= 0:
        return ev_map
    # 各馬の複勝圏確率をスコア比率から推定（×3で3着以内確率、上限0.85）
    probs = scores / score_sum
    top3_probs = np.clip(probs * 3.0, 0.0, 0.85)

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        umaban = int(row['馬番'])
        p = top3_probs[idx]
        odds = row.get('単勝オッズ', 0)
        if odds_dict:
            odds = odds_dict.get(umaban, odds)
        if odds <= 0:
            odds = 10.0  # デフォルト
        # 三連複の想定配当倍率 ≈ 単勝オッズの2〜3倍（経験則）
        # EV = 複勝圏確率 × (単勝オッズ × 2.5) / 投資倍率
        # 簡易EV: 確率が高く、オッズも高い馬が高EV
        trio_multiplier = odds * 2.0  # 三連複は単勝の約2倍のリターン
        ev = p * trio_multiplier
        ev_map[umaban] = round(ev, 2)
    return ev_map


def calc_race_confidence(df_sorted, feat_summary=None, cond_profile=None):
    """レースの信頼度スコアを0-100で算出。
    - TOP3スコア差（大きいほど信頼度高い）
    - 特徴量取得率
    - 条件ROI実績
    Returns: (confidence_score, reasons_list)
    """
    reasons = []
    score = 50  # ベース

    if len(df_sorted) < 3:
        return 20, ['出走頭数不足']

    scores = df_sorted['スコア'].values
    top1, top2, top3 = scores[0], scores[1], scores[2]

    # 1. TOP1-TOP2のスコア差（0〜+25点）
    gap_12 = top1 - top2
    gap_bonus = min(gap_12 / 0.05 * 25, 25)  # 0.05差で+25点
    score += gap_bonus
    if gap_12 >= 0.03:
        reasons.append(f'軸馬明確 (差{gap_12:.3f})')
    elif gap_12 < 0.01:
        reasons.append(f'混戦 (差{gap_12:.3f})')
        score -= 10

    # 2. TOP1-TOP3のスコア差（0〜+15点）
    gap_13 = top1 - top3
    if gap_13 >= 0.05:
        score += 15
        reasons.append('上位3頭に明確差')
    elif gap_13 < 0.02:
        score -= 5

    # 3. 特徴量取得率（0〜+15点）
    if feat_summary:
        total = feat_summary.get('total', 1)
        acquired = feat_summary.get('acquired', 0)
        feat_rate = acquired / max(total, 1)
        feat_bonus = feat_rate * 15
        score += feat_bonus
        if feat_rate < 0.7:
            reasons.append(f'特徴量不足 ({acquired}/{total})')
            score -= 10
        elif feat_rate >= 0.9:
            reasons.append(f'特徴量充実 ({acquired}/{total})')

    # 4. 条件ROI実績（0〜+15点）
    if cond_profile:
        roi = cond_profile.get('roi', 100)
        if roi >= 200:
            score += 15
            reasons.append(f'高ROI条件 ({roi:.0f}%)')
        elif roi >= 150:
            score += 10
        elif roi >= 100:
            score += 5
        else:
            score -= 5
            reasons.append(f'低ROI条件 ({roi:.0f}%)')

        if not cond_profile.get('recommended', True):
            score -= 20
            reasons.append('購入非推奨条件')

    return max(0, min(100, int(score))), reasons


def calc_variable_investment(confidence, cond_key, base_investment=700):
    """信頼度ベースで投資額を可変にする。
    - 信頼度高い → 最大1,000円
    - 信頼度低い → 最小400円
    - 条件D/Eは控えめに
    Returns: investment_amount (int, 100円単位)
    """
    if confidence >= 80:
        amount = 1000
    elif confidence >= 65:
        amount = 800
    elif confidence >= 50:
        amount = 700
    elif confidence >= 35:
        amount = 500
    else:
        amount = 400

    # 条件D/Eは保守的ROIが100%以下なので控えめ
    if cond_key in ('D', 'E'):
        amount = min(amount, 700)

    # 100円単位に丸め
    return int(round(amount / 100) * 100)


def calc_kelly_investment(df_sorted, odds_dict=None, cond_key='A',
                          base_bankroll=30000, fraction=0.5,
                          min_bet=100, max_bet=2000):
    """Kelly基準で投資額を計算。

    ハーフKelly（fraction=0.5）でリスク抑制。
    三連複7点フォーメーションの場合、TOP1-3の複勝圏確率を使う。

    Args:
        df_sorted: スコア降順ソート済みDF
        odds_dict: {馬番: 単勝オッズ}
        cond_key: 条件(A-X)
        base_bankroll: 仮想バンクロール（Kelly計算用）
        fraction: Kelly fraction（0.5=ハーフKelly推奨）
        min_bet: 最低ベット額
        max_bet: 最大ベット額

    Returns:
        (amount, kelly_info): (投資額(100円単位), {'kelly_f': f値, 'ev': EV, 'edge': エッジ})
    """
    if len(df_sorted) < 3 or not odds_dict:
        return 700, {'kelly_f': 0, 'ev': 1.0, 'edge': 0}

    scores = df_sorted['スコア'].values
    score_sum = scores.sum()
    if score_sum <= 0:
        return 700, {'kelly_f': 0, 'ev': 1.0, 'edge': 0}

    # TOP1の複勝圏（3着以内）確率を推定
    top1_prob = min(scores[0] / score_sum * 3.0, 0.85)

    # TOP1-3の三連複的中確率（TOP1軸-TOP2,3-TOP2~6フォーメーション）
    # 簡易推定: TOP1が3着以内 × TOP2が3着以内 × TOP3~6のいずれかが3着以内
    top2_prob = min(scores[1] / score_sum * 3.0, 0.80)
    remaining_prob = min(sum(scores[2:6]) / score_sum * 3.0, 0.95)
    trio_hit_prob = top1_prob * top2_prob * remaining_prob

    # 三連複の想定配当を推定（条件別）
    cond_avg_payout = {
        'A': 4000, 'B': 5000, 'C': 8000, 'D': 5000,
        'E': 2500, 'X': 10000,
    }
    avg_payout = cond_avg_payout.get(cond_key, 5000)

    # 単勝オッズから補正（オッズが高い＝配当も高い）
    top1_uma = int(df_sorted.iloc[0]['馬番'])
    top1_odds = odds_dict.get(top1_uma, 10.0)
    payout_est = avg_payout * max(top1_odds / 5.0, 0.5)  # 5倍馬を基準に補正

    # Kelly基準: f* = (p * b - q) / b
    # p = 的中確率, b = 配当倍率-1, q = 1-p
    if cond_key == 'E':
        # 馬連2点
        b = payout_est / 350 - 1  # 350円/点 × 2点
    else:
        # 三連複7点
        b = payout_est / 100 - 1  # 100円/点 × 7点

    p = trio_hit_prob
    q = 1 - p

    if b <= 0:
        kelly_f = 0
    else:
        kelly_f = (p * b - q) / b

    # ハーフKelly
    kelly_f_adj = kelly_f * fraction

    # EV = 期待値
    ev = p * (b + 1) if b > 0 else 0

    # 投資額計算
    if kelly_f_adj <= 0:
        amount = min_bet
        edge = 0
    else:
        raw_amount = base_bankroll * kelly_f_adj
        amount = max(min_bet, min(max_bet, raw_amount))
        edge = kelly_f

    # 100円単位に丸め
    amount = int(round(amount / 100) * 100)
    amount = max(min_bet, min(max_bet, amount))

    # 条件D/Eは控えめ
    if cond_key in ('D', 'E'):
        amount = min(amount, 1000)

    kelly_info = {
        'kelly_f': round(kelly_f, 4),
        'kelly_adj': round(kelly_f_adj, 4),
        'ev': round(ev, 2),
        'edge': round(kelly_f * 100, 1),  # エッジ%
        'trio_hit_prob': round(trio_hit_prob, 4),
        'payout_est': int(payout_est),
    }
    return amount, kelly_info


def scan_value_horses(df_sorted, odds_dict=None, threshold=2.0):
    """穴馬シグナルスキャン — AIスコアvsオッズの乖離検出。

    AI確率が高いのにオッズが高い馬 = 市場が過小評価している馬。
    乖離度 = AI推定確率 / 市場確率(1/オッズ)

    Args:
        df_sorted: スコア降順ソート済みDF
        odds_dict: {馬番: 単勝オッズ}
        threshold: 乖離度閾値（2.0以上で穴馬判定）

    Returns:
        list of {'umaban': int, 'horse_name': str, 'score': float, 'odds': float,
                 'divergence': float, 'reasons': list}
    """
    signals = []
    if len(df_sorted) < 3 or not odds_dict:
        return signals

    scores = df_sorted['スコア'].values
    score_sum = scores.sum()
    if score_sum <= 0:
        return signals

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        umaban = int(row['馬番'])
        horse_name = str(row.get('馬名', f'{umaban}番'))
        ai_score = scores[idx]
        ai_prob = ai_score / score_sum * 3.0  # 複勝圏確率推定

        odds = odds_dict.get(umaban, 0)
        if odds <= 1.0:
            continue

        market_prob = 1.0 / odds  # 市場の勝率推定
        market_top3_prob = market_prob * 2.5  # 複勝圏は約2.5倍（経験則）

        if market_top3_prob <= 0:
            continue

        divergence = ai_prob / market_top3_prob

        if divergence >= threshold:
            reasons = []
            # 過小評価の理由候補
            if 'prev_finish' in row and row['prev_finish'] >= 8:
                reasons.append('前走大敗→巻き返し')
            if 'rest_days' in row and row['rest_days'] >= 60:
                reasons.append('休み明け→フレッシュ')
            if 'jrdb_prev_interference' in row and row.get('jrdb_prev_interference', 0) > 0:
                reasons.append('前走不利あり→実力以上')
            if 'finish_trend' in row and row.get('finish_trend', 0) < -2:
                reasons.append('着順上昇傾向')
            if 'training_intensity_enc' in row and row.get('training_intensity_enc', 0) >= 2:
                reasons.append('調教強め以上')
            if not reasons:
                reasons.append('AI高評価vs市場低評価')

            signals.append({
                'umaban': umaban,
                'horse_name': horse_name,
                'ai_score': round(ai_score, 4),
                'ai_prob': round(ai_prob, 3),
                'odds': odds,
                'market_prob': round(market_top3_prob, 3),
                'divergence': round(divergence, 2),
                'reasons': reasons,
            })

    return sorted(signals, key=lambda x: -x['divergence'])
