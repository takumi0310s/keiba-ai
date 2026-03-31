#!/usr/bin/env python
"""KEIBA AI NAR (地方競馬) 専用モデル学習 + 条件別バックテスト
- netkeibaから地方レースをスクレイピングしてデータ収集
- 騎手勝率・厩舎成績・馬の過去成績を取得
- 地方専用LightGBM + XGBoostアンサンブル
- 条件A-E,X別にROI算出、80%以上のみ買い推奨
"""
import sys

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import re
import time
import random
import warnings
warnings.filterwarnings('ignore')

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')
NAR_DATA_PATH = os.path.join(OUTPUT_DIR, 'data', 'chihou_races_2020_2025.csv')
NAR_CACHE_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_scraped_cache.json')
NAR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'keiba_model_v9_nar.pkl')
BACKTEST_RESULT_PATH = os.path.join(OUTPUT_DIR, 'backtest_nar_condition.json')

# NAR venue codes / maps
COURSE_MAP_NAR = {
    '大井': 10, '川崎': 11, '船橋': 12, '浦和': 13, '園田': 14, '姫路': 15,
    '門別': 16, '盛岡': 17, '水沢': 18, '金沢': 19, '笠松': 20, '名古屋': 21,
    '高知': 22, '佐賀': 23, '帯広': 24,
}
SURFACE_MAP = {'芝': 0, 'ダ': 1, '障': 2}
COND_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
SEX_MAP = {'牡': 0, '牝': 1, 'セ': 2, '騸': 2}

# NAR専用特徴量 (地方で実際に取れるもの中心)
NAR_FEATURES = [
    'odds_log',              # オッズ(log)
    'num_horses',            # 出走頭数
    'distance',              # 距離
    'surface_enc',           # 芝ダ障
    'condition_enc',         # 馬場状態
    'course_enc',            # 競馬場
    'horse_weight',          # 馬体重
    'weight_carry',          # 斤量
    'age',                   # 年齢
    'sex_enc',               # 性別
    'horse_num',             # 馬番
    'bracket',               # 枠番
    'jockey_wr',             # 騎手勝率
    'jockey_place_rate',     # 騎手複勝率
    'trainer_wr',            # 厩舎勝率
    'prev_finish',           # 前走着順
    'prev2_finish',          # 2走前着順
    'prev3_finish',          # 3走前着順
    'avg_finish_3r',         # 直近3走平均着順
    'best_finish_3r',        # 直近3走最高着順
    'top3_count_3r',         # 直近3走複勝回数
    'finish_trend',          # 着順トレンド
    'prev_odds_log',         # 前走オッズ(log)
    'rest_days',             # 休養日数
    'rest_category',         # 休養カテゴリ
    'dist_cat',              # 距離カテゴリ
    'weight_cat',            # 体重カテゴリ
    'age_group',             # 年齢グループ
    'horse_num_ratio',       # 馬番/頭数
    'bracket_pos',           # 枠位置
    'carry_diff',            # 斤量差
    'dist_change',           # 距離変更
    'dist_change_abs',       # 距離変更(絶対値)
    'is_nar',                # NARフラグ
    'pop_rank',              # 人気順位
]


def collect_nar_race_ids(start_year=2020, end_year=2025, max_races=600):
    """Collect NAR race IDs from db.netkeiba.com for multiple years."""
    print(f"\n  Collecting NAR race IDs ({start_year}-{end_year})...")
    all_races = []
    seen = set()

    # Sample dates across years (every 3rd day to cover spread)
    for year in range(start_year, end_year + 1):
        # Sample ~20 dates per year spread across months
        for month in range(1, 13):
            for day_offset in [5, 15, 25]:
                d = datetime(year, month, min(day_offset, 28))
                date_str = d.strftime('%Y%m%d')
                try:
                    url = f"https://db.netkeiba.com/race/list/{date_str}/"
                    resp = requests.get(url, headers=HEADERS, timeout=10)
                    resp.encoding = "EUC-JP"
                    soup = BeautifulSoup(resp.text, "html.parser")
                    links = soup.find_all("a", href=re.compile(r'/race/\d{12}'))
                    for link in links:
                        href = link.get("href", "")
                        m = re.search(r'/race/(\d{12})/', href)
                        if m:
                            rid = m.group(1)
                            venue = int(rid[4:6])
                            if venue > 10 and rid not in seen:
                                seen.add(rid)
                                all_races.append({'race_id': rid, 'date': date_str, 'year': year})
                except Exception:
                    pass
                time.sleep(random.uniform(3.0, 5.0))

                if len(all_races) >= max_races * 3:
                    break
            if len(all_races) >= max_races * 3:
                break
        print(f"    {year}: {len(all_races)} NAR races found so far")
        if len(all_races) >= max_races * 3:
            break

    print(f"  Total: {len(all_races)} NAR races collected")

    # Sample evenly across years
    if len(all_races) > max_races:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_races), max_races, replace=False)
        all_races = [all_races[i] for i in sorted(indices)]

    return all_races


def scrape_horse_history(horse_id, max_races=5):
    """Scrape a horse's past race results from db.netkeiba.com."""
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="db_h_race_results") or soup.find("table", {"summary": re.compile("レース")})
        if not table:
            return []

        rows = table.find_all("tr")
        history = []
        for row in rows[1:max_races + 1]:  # Skip header
            tds = row.find_all("td")
            if len(tds) < 12:
                continue
            try:
                finish = int(tds[4].get_text(strip=True)) if tds[4].get_text(strip=True).isdigit() else 99
                odds_text = tds[9].get_text(strip=True) if len(tds) > 9 else '15.0'
                try:
                    odds = float(odds_text)
                except:
                    odds = 15.0
                date_text = tds[0].get_text(strip=True)
                history.append({
                    'finish': finish,
                    'odds': odds,
                    'date': date_text,
                })
            except Exception:
                continue
        return history
    except Exception:
        return []


def scrape_nar_race_full(race_id):
    """Scrape a NAR race with full horse data + payouts."""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else ""

        header_text = soup.get_text()[:3000]
        dm = re.search(r'([芝ダ障])(\d{3,4})m', header_text)
        distance = int(dm.group(2)) if dm else 1600
        surface = dm.group(1) if dm else 'ダ'
        cm = re.search(r'馬場:(良|稍重|稍|重|不良|不)', header_text)
        if not cm:
            cm = re.search(r'(良|稍重|重|不良)', header_text[:1000])
        condition = cm.group(1) if cm else '良'

        course_name = ''
        for cn in COURSE_MAP_NAR:
            if cn in title_text:
                course_name = cn
                break

        table = soup.find("table", class_="race_table_01")
        if not table:
            return None
        rows = table.find_all("tr")

        horses = []
        actual_finishes = {}
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 13:
                continue
            finish_text = tds[0].get_text(strip=True)
            if not finish_text.isdigit():
                continue
            finish = int(finish_text)
            umaban = int(tds[2].get_text(strip=True)) if tds[2].get_text(strip=True).isdigit() else 0
            if umaban == 0:
                continue
            actual_finishes[umaban] = finish

            horse_name = ''
            horse_id = ''
            horse_link = tds[3].find("a") if len(tds) > 3 else None
            if horse_link:
                horse_name = horse_link.get_text(strip=True)
                href = horse_link.get('href', '')
                hm = re.search(r'/horse/(\w+)', href)
                if hm:
                    horse_id = hm.group(1)

            jockey_name = ''
            jockey_link = tds[6].find("a") if len(tds) > 6 else None
            if jockey_link:
                jockey_name = jockey_link.get_text(strip=True)

            sex_age = tds[4].get_text(strip=True) if len(tds) > 4 else '牡3'
            sex = sex_age[0] if sex_age else '牡'
            age = int(sex_age[1:]) if len(sex_age) > 1 and sex_age[1:].isdigit() else 3

            kinryo = 55.0
            if len(tds) > 5:
                try:
                    kinryo = float(tds[5].get_text(strip=True))
                except:
                    pass

            hw = 480
            if len(tds) > 14:
                hw_text = tds[14].get_text(strip=True)
                hw_m = re.match(r'(\d{3,})', hw_text)
                if hw_m:
                    hw = int(hw_m.group(1))

            odds_val = 15.0
            if len(tds) > 9:
                try:
                    ov = float(tds[9].get_text(strip=True))
                    if 1.0 <= ov <= 999:
                        odds_val = ov
                except:
                    pass

            pop_rank = 8
            if len(tds) > 10:
                try:
                    pr = int(tds[10].get_text(strip=True))
                    if 1 <= pr <= 30:
                        pop_rank = pr
                except:
                    pass

            horses.append({
                'horse_name': horse_name,
                'horse_id': horse_id,
                'umaban': umaban,
                'horse_weight': hw,
                'weight_carry': kinryo,
                'age': age,
                'sex': sex,
                'distance': distance,
                'surface': surface,
                'condition': condition,
                'course': course_name,
                'course_enc': COURSE_MAP_NAR.get(course_name, 10),
                'surface_enc': SURFACE_MAP.get(surface, 1),
                'condition_enc': COND_MAP.get(condition, 0),
                'sex_enc': SEX_MAP.get(sex, 0),
                'horse_num': umaban,
                'num_horses': 14,  # update below
                'odds': odds_val,
                'pop_rank': pop_rank,
                'jockey_name': jockey_name,
                'finish': finish,
            })

        if len(horses) < 5:
            return None

        for h in horses:
            h['num_horses'] = len(horses)
            h['bracket'] = min(8, max(1, (h['umaban'] - 1) * 8 // max(1, len(horses)) + 1))

        # Scrape payouts
        payouts = scrape_payouts(soup)

        race_meta = {
            'race_id': race_id,
            'course': course_name,
            'distance': distance,
            'surface': surface,
            'condition': condition,
            'num_horses': len(horses),
        }

        return horses, actual_finishes, payouts, race_meta

    except Exception as e:
        return None


def scrape_payouts(soup):
    """Extract trio/umaren/wide payouts from result page soup.
    db.netkeiba.com pay_table_01 structure:
    - Table 0: 単勝, 複勝, 枠連, 馬連
    - Table 1: ワイド, 枠単, 馬単, 三連複, 三連単
    td[0]=組番号, td[1]=払戻金, td[2]=人気
    ワイドは<br>区切りで複数の払戻金が入る
    """
    payouts = {'trio': 0, 'umaren': 0, 'wide': []}
    try:
        payout_tables = soup.find_all("table", class_="pay_table_01")
        if not payout_tables:
            payout_tables = soup.find_all("table")

        for table in payout_tables:
            for row in table.find_all("tr"):
                th = row.find("th")
                if not th:
                    continue
                th_text = th.get_text(strip=True)
                tds = row.find_all("td")
                if len(tds) < 2:
                    continue

                # 三連複
                if ('三連複' in th_text or '3連複' in th_text) and payouts['trio'] == 0:
                    payout_text = tds[1].get_text(strip=True).replace(',', '')
                    pm = re.search(r'(\d+)', payout_text)
                    if pm:
                        payouts['trio'] = int(pm.group(1))

                # 馬連 (馬単を除外)
                elif '馬連' in th_text and '馬単' not in th_text and payouts['umaren'] == 0:
                    payout_text = tds[1].get_text(strip=True).replace(',', '')
                    pm = re.search(r'(\d+)', payout_text)
                    if pm:
                        payouts['umaren'] = int(pm.group(1))

                # ワイド - <br>区切りで複数の払戻金
                elif 'ワイド' in th_text and not payouts['wide']:
                    # <br>タグで分割して個別に取得
                    payout_td = tds[1]
                    # brタグを改行に変換
                    for br in payout_td.find_all("br"):
                        br.replace_with("\n")
                    lines = payout_td.get_text().split("\n")
                    for line in lines:
                        line = line.strip().replace(',', '')
                        pm = re.search(r'(\d+)', line)
                        if pm:
                            val = int(pm.group(1))
                            if 100 <= val <= 999999:  # 妥当な払戻金の範囲
                                payouts['wide'].append(val)
    except Exception:
        pass
    return payouts


def build_nar_features(horses_list, jockey_stats, trainer_stats):
    """Build feature matrix for NAR horses with actual statistics."""
    df = pd.DataFrame(horses_list)

    # Jockey stats
    df['jockey_wr'] = df['jockey_name'].map(
        lambda j: jockey_stats.get(j, {}).get('wr', 0.08)
    )
    df['jockey_place_rate'] = df['jockey_name'].map(
        lambda j: jockey_stats.get(j, {}).get('place_rate', 0.25)
    )

    # Trainer stats (not available from scraping, use default)
    df['trainer_wr'] = 0.10

    # Odds features
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999))
    df['prev_odds_log'] = df.get('prev_odds_log', np.log1p(15.0))

    # Derived features
    df['dist_cat'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                            labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0, 440, 480, 520, 9999],
                              labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df['age_group'] = df['age'].clip(2, 7)
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0, 3, 6, 8],
                               labels=[0, 1, 2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df['weight_carry'].mean()
    df['rest_category'] = df.get('rest_category', 2)
    df['is_nar'] = 1

    # Ensure all features exist
    for f in NAR_FEATURES:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    return df


def classify_condition(num_horses, distance, condition):
    """Classify race condition A-E,X."""
    heavy = any(c in str(condition) for c in ['重', '不'])
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


def calc_bets(ranking):
    """Calculate trio/umaren/wide bets from ranking."""
    if len(ranking) < 3:
        return [], [], []
    nums = ranking[:6] if len(ranking) >= 6 else ranking
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]

    # Trio 7-bet
    trio_bets = set()
    for s in second:
        for t in third:
            combo = tuple(sorted({n1, s, t}))
            if len(combo) == 3:
                trio_bets.add(combo)
    trio_bets = sorted(trio_bets)

    # Umaren 2-bet
    umaren_bets = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]

    # Wide 2-bet
    wide_bets = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]

    return trio_bets, wide_bets, umaren_bets


def check_hits(actual_finishes, trio_bets, wide_bets, umaren_bets):
    """Check which bets hit."""
    # Top 3 horses
    top3 = set()
    for uma, fin in actual_finishes.items():
        if fin <= 3:
            top3.add(uma)

    # Trio hit
    trio_hit = False
    for combo in trio_bets:
        if set(combo) == top3:
            trio_hit = True
            break

    # Wide hits
    wide_hits = []
    for bet in wide_bets:
        if set(bet).issubset(top3):
            wide_hits.append(bet)

    # Umaren hits
    top2 = set()
    for uma, fin in actual_finishes.items():
        if fin <= 2:
            top2.add(uma)
    umaren_hits = []
    for bet in umaren_bets:
        if set(bet) == top2:
            umaren_hits.append(bet)

    return trio_hit, wide_hits, umaren_hits


def scrape_and_collect(race_ids, cache_path=NAR_CACHE_PATH):
    """Scrape NAR races and collect horse data with history."""
    # Load cache
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"  Loaded cache: {len(cache)} races")
        except Exception:
            cache = {}

    all_rows = []          # For training data
    backtest_results = []  # For backtest
    jockey_stats_raw = {}  # Jockey name -> {wins, races}
    new_scraped = 0

    consecutive_errors = 0
    for ri, race_info in enumerate(race_ids):
        rid = race_info['race_id']

        # Check cache
        if rid in cache:
            race_data = cache[rid]
            consecutive_errors = 0
        else:
            try:
                result = scrape_nar_race_full(rid)
                if not result:
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print(f"    WARNING: 10 consecutive errors, stopping early")
                        break
                    time.sleep(random.uniform(3.0, 5.0))
                    continue
                horses, actual_finishes, payouts, race_meta = result
                race_data = {
                    'horses': horses,
                    'actual_finishes': actual_finishes,
                    'payouts': payouts,
                    'meta': race_meta,
                }
                cache[rid] = race_data
                new_scraped += 1
                consecutive_errors = 0
                time.sleep(random.uniform(3.0, 5.0))
            except Exception as e:
                print(f"    ERROR scraping {rid}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    print(f"    WARNING: 10 consecutive errors, stopping early")
                    break
                time.sleep(random.uniform(5.0, 8.0))
                continue

        horses = race_data['horses']
        actual = race_data['actual_finishes']
        payouts = race_data['payouts']
        meta = race_data['meta']

        # Collect jockey stats
        for h in horses:
            jn = h.get('jockey_name', '')
            if jn:
                if jn not in jockey_stats_raw:
                    jockey_stats_raw[jn] = {'wins': 0, 'races': 0, 'top3': 0}
                jockey_stats_raw[jn]['races'] += 1
                if h.get('finish', 99) == 1:
                    jockey_stats_raw[jn]['wins'] += 1
                if h.get('finish', 99) <= 3:
                    jockey_stats_raw[jn]['top3'] += 1

        # Add to training rows
        for h in horses:
            row = dict(h)
            row['race_id'] = rid
            # Default lag features
            row['prev_finish'] = 5
            row['prev2_finish'] = 5
            row['prev3_finish'] = 5
            row['avg_finish_3r'] = 5.0
            row['best_finish_3r'] = 5
            row['top3_count_3r'] = 0
            row['finish_trend'] = 0
            row['prev_odds_log'] = np.log1p(h.get('odds', 15.0))
            row['rest_days'] = 30
            row['rest_category'] = 2
            row['dist_change'] = 0
            row['dist_change_abs'] = 0
            row['target'] = 1 if h.get('finish', 99) <= 3 else 0
            all_rows.append(row)

        # Backtest entry
        bt_entry = {
            **meta,
            'actual_finishes': actual,
            'payouts': payouts,
            'horses': horses,
        }
        backtest_results.append(bt_entry)

        if (ri + 1) % 20 == 0:
            print(f"    {ri+1}/{len(race_ids)} processed ({len(backtest_results)} valid, {new_scraped} new)")

        # Save cache periodically
        if new_scraped > 0 and new_scraped % 50 == 0:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, default=str)

    # Final cache save
    if new_scraped > 0:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, default=str)
        print(f"  Cache saved: {len(cache)} total races ({new_scraped} new)")

    # Compute jockey stats
    jockey_stats = {}
    for jn, stats in jockey_stats_raw.items():
        n = stats['races']
        global_wr = 0.08
        alpha = 20
        wr = (stats['wins'] + alpha * global_wr) / (n + alpha)
        place_rate = (stats['top3'] + alpha * 0.25) / (n + alpha)
        jockey_stats[jn] = {'wr': wr, 'place_rate': place_rate, 'races': n}

    return all_rows, backtest_results, jockey_stats


def train_nar_model(all_rows, jockey_stats):
    """Train NAR-dedicated LightGBM + XGBoost model."""
    print("\n" + "=" * 60)
    print("  Training NAR-dedicated model")
    print("=" * 60)

    df = pd.DataFrame(all_rows)
    print(f"  Total rows: {len(df)}, races: {df['race_id'].nunique()}")

    # Build features
    df = build_nar_features(df.to_dict('records'), jockey_stats, {})
    df['target'] = [r['target'] for r in all_rows]

    # Ensure features
    for f in NAR_FEATURES:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[NAR_FEATURES].values
    y = df['target'].values

    print(f"  Features: {len(NAR_FEATURES)}")
    print(f"  Target rate: {y.mean():.3f}")

    # Train/valid split (80/20 random)
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Valid: {len(X_valid)}")

    # LightGBM
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 30,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=NAR_FEATURES)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=NAR_FEATURES, reference=dtrain)

    lgb_model = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)
    print(f"\n  NAR LightGBM AUC: {lgb_auc:.4f}")

    # XGBoost
    try:
        import xgboost as xgb_lib
        dtrain_xgb = xgb_lib.DMatrix(X_train, label=y_train)
        dvalid_xgb = xgb_lib.DMatrix(X_valid, label=y_valid)
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 30,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': 42,
            'tree_method': 'hist', 'verbosity': 0,
        }
        xgb_model = xgb_lib.train(
            xgb_params, dtrain_xgb, num_boost_round=500,
            evals=[(dvalid_xgb, 'valid')],
            early_stopping_rounds=30, verbose_eval=50,
        )
        xgb_pred = xgb_model.predict(dvalid_xgb)
        xgb_auc = roc_auc_score(y_valid, xgb_pred)
        print(f"  NAR XGBoost AUC: {xgb_auc:.4f}")

        # Ensemble
        total = lgb_auc + xgb_auc
        w_lgb = lgb_auc / total
        w_xgb = xgb_auc / total
        ensemble_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
        ensemble_auc = roc_auc_score(y_valid, ensemble_pred)
        print(f"  NAR Ensemble AUC: {ensemble_auc:.4f}")
    except ImportError:
        xgb_model = None
        xgb_auc = 0
        w_lgb = 1.0
        w_xgb = 0.0
        ensemble_auc = lgb_auc
        print("  XGBoost not available, using LightGBM only")

    # Feature importance
    importance = lgb_model.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({
        'feature': NAR_FEATURES,
        'importance': importance,
    }).sort_values('importance', ascending=False)
    print(f"\n  NAR Feature Importance TOP 15:")
    for _, row in fi_df.head(15).iterrows():
        bar = '#' * int(row['importance'] / fi_df['importance'].max() * 25)
        print(f"    {row['feature']:25s} {row['importance']:10.1f} {bar}")

    return lgb_model, xgb_model, lgb_auc, xgb_auc, ensemble_auc, {'lgb': w_lgb, 'xgb': w_xgb}, jockey_stats


def run_nar_condition_backtest(backtest_results, lgb_model, xgb_model, weights, jockey_stats):
    """Run condition-based backtest on NAR races."""
    print("\n" + "=" * 60)
    print("  NAR CONDITION-BASED BACKTEST")
    print("=" * 60)

    condition_results = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'X': []}

    for race in backtest_results:
        horses = race['horses']
        actual = race['actual_finishes']
        payouts = race['payouts']
        meta_nh = race.get('num_horses', len(horses))
        meta_dist = race.get('distance', 1600)
        meta_cond = race.get('condition', '良')

        cond_key = classify_condition(meta_nh, meta_dist, meta_cond)

        # Build features and predict
        df = build_nar_features(horses, jockey_stats, {})

        for f in NAR_FEATURES:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

        X = df[NAR_FEATURES].values
        scores = lgb_model.predict(X)

        if xgb_model:
            try:
                import xgboost as xgb_lib
                xgb_pred = xgb_model.predict(xgb_lib.DMatrix(X))
                scores = scores * weights['lgb'] + xgb_pred * weights['xgb']
            except Exception:
                pass

        df['score'] = scores
        df = df.sort_values('score', ascending=False)
        ranking = df['umaban'].astype(int).tolist()

        trio_bets, wide_bets, umaren_bets = calc_bets(ranking)
        trio_hit, wide_hits, umaren_hits = check_hits(actual, trio_bets, wide_bets, umaren_bets)

        result_entry = {
            'race_id': race.get('race_id', ''),
            'condition_key': cond_key,
            'num_horses': meta_nh,
            'distance': meta_dist,
            'condition': meta_cond,
            'trio_hit': trio_hit,
            'trio_payout': payouts.get('trio', 0) if trio_hit else 0,
            'wide_hits': len(wide_hits),
            'wide_payout': sum(payouts.get('wide', [])[:len(wide_hits)]) if wide_hits else 0,
            'umaren_hit': len(umaren_hits) > 0,
            'umaren_payout': payouts.get('umaren', 0) if umaren_hits else 0,
            'ranking_top3': ranking[:3],
        }
        condition_results[cond_key].append(result_entry)

    # Analyze per condition - test all bet types per condition
    print(f"\n  {'COND':<4} {'DESC':<28} {'N':>4} | {'BET':<8} {'HIT':>4} {'RATE':>7} {'INVEST':>8} {'PAYOUT':>8} {'ROI':>7}")
    print(f"  {'-' * 95}")

    best_per_condition = {}
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = condition_results.get(ckey, [])
        n = len(races)
        if n == 0:
            best_per_condition[ckey] = {'n': 0, 'roi': 0, 'bet_type': 'trio', 'recommended': False}
            continue

        desc_map = {
            'A': '8-14頭/1600m+/良~稍',
            'B': '8-14頭/1600m+/重~不良',
            'C': '15頭+/1600m+/良~稍',
            'D': '1400m以下',
            'E': '7頭以下',
            'X': '15頭+/重~不良',
        }

        # Test trio/umaren/wide for each condition
        results_by_bet = {}
        for bt, n_bets, hit_key, pay_key in [
            ('trio', 7, 'trio_hit', 'trio_payout'),
            ('umaren', 2, 'umaren_hit', 'umaren_payout'),
            ('wide', 2, 'wide_hits', 'wide_payout'),
        ]:
            if bt == 'wide':
                hits = sum(1 for r in races if r.get(hit_key, 0) > 0)
            elif bt == 'umaren':
                hits = sum(1 for r in races if r.get(hit_key, False))
            else:
                hits = sum(1 for r in races if r.get(hit_key, False))

            investment = n * n_bets * 100
            total_payout = sum(r.get(pay_key, 0) for r in races)
            roi = total_payout / investment * 100 if investment > 0 else 0
            hit_rate = hits / n * 100 if n > 0 else 0

            results_by_bet[bt] = {
                'hits': hits, 'hit_rate': hit_rate,
                'investment': investment, 'payout': total_payout, 'roi': roi,
            }

        # Find best bet type
        best_bt = max(results_by_bet, key=lambda b: results_by_bet[b]['roi'])
        best = results_by_bet[best_bt]
        recommended = best['roi'] >= 80

        best_per_condition[ckey] = {
            'n': n,
            'bet_type': best_bt,
            'hits': best['hits'],
            'hit_rate': best['hit_rate'],
            'investment': best['investment'],
            'payout': best['payout'],
            'roi': best['roi'],
            'recommended': recommended,
            'all_bets': results_by_bet,
        }

        # Print all bet types for this condition
        first = True
        for bt in ['trio', 'umaren', 'wide']:
            r = results_by_bet[bt]
            marker = ' <<<' if bt == best_bt else ''
            rec = ' REC' if bt == best_bt and recommended else ''
            if first:
                print(f"  {ckey:<4} {desc_map.get(ckey, '?'):<28} {n:>4} | {bt:<8} {r['hits']:>4} {r['hit_rate']:>6.1f}% {r['investment']:>7,} {r['payout']:>7,} {r['roi']:>6.1f}%{marker}{rec}")
                first = False
            else:
                print(f"  {'':4} {'':28} {'':4} | {bt:<8} {r['hits']:>4} {r['hit_rate']:>6.1f}% {r['investment']:>7,} {r['payout']:>7,} {r['roi']:>6.1f}%{marker}{rec}")

    # Summary
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY: NAR 条件別最適買い目")
    print(f"  {'=' * 60}")
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        info = best_per_condition[ckey]
        if info['n'] == 0:
            print(f"  {ckey}: データなし")
            continue
        rec_str = "★買い推奨" if info['recommended'] else "×非推奨"
        print(f"  {ckey}: {info['bet_type']} ROI {info['roi']:.1f}% ({info['n']}レース) → {rec_str}")

    return best_per_condition


def save_nar_model(lgb_model, xgb_model, auc, ensemble_auc, weights, features, jockey_stats, condition_results):
    """Save NAR-dedicated model."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    nar_pkl = {
        'model': lgb_model,
        'xgb_model': xgb_model,
        'features': features,
        'version': 'nar_v1',
        'auc': auc,
        'ensemble_auc': ensemble_auc,
        'ensemble_weights': weights,
        'model_type': 'nar_dedicated',
        'trained_at': now,
        'jockey_stats': jockey_stats,
        'condition_results': {k: {
            'bet_type': v['bet_type'],
            'roi': v['roi'],
            'recommended': v['recommended'],
            'n': v['n'],
            'hit_rate': v.get('hit_rate', 0),
        } for k, v in condition_results.items()},
    }

    with open(NAR_MODEL_PATH, 'wb') as f:
        pickle.dump(nar_pkl, f)
    print(f"\n  Saved NAR model: {NAR_MODEL_PATH}")

    # Save backtest results
    bt_save = {
        'generated_at': now,
        'condition_results': {k: {
            'bet_type': v['bet_type'],
            'roi': v['roi'],
            'hit_rate': v.get('hit_rate', 0),
            'recommended': v['recommended'],
            'n': v['n'],
        } for k, v in condition_results.items()},
    }
    with open(BACKTEST_RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(bt_save, f, ensure_ascii=False, indent=2)
    print(f"  Saved backtest results: {BACKTEST_RESULT_PATH}")


def main():
    print("=" * 60)
    print("  KEIBA AI NAR 専用モデル学習 + 条件別バックテスト")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Step 1: Collect race IDs
    race_ids = collect_nar_race_ids(start_year=2022, end_year=2025, max_races=500)
    if not race_ids:
        print("  ERROR: No NAR races found")
        return

    # Step 2: Scrape race data
    print(f"\n  Scraping {len(race_ids)} NAR races...")
    all_rows, backtest_results, jockey_stats = scrape_and_collect(race_ids)
    print(f"  Collected {len(all_rows)} horse rows from {len(backtest_results)} races")
    print(f"  Jockey stats: {len(jockey_stats)} jockeys")

    if len(all_rows) < 100:
        print("  ERROR: Not enough data for training")
        return

    # Save CSV
    csv_df = pd.DataFrame(all_rows)
    csv_df.to_csv(NAR_DATA_PATH, index=False, encoding='utf-8')
    print(f"  Saved CSV: {NAR_DATA_PATH} ({len(csv_df)} rows)")

    # Step 3: Train model
    lgb_model, xgb_model, lgb_auc, xgb_auc, ensemble_auc, weights, jockey_stats = \
        train_nar_model(all_rows, jockey_stats)

    # Step 4: Condition-based backtest (use last 100 races)
    test_races = backtest_results[-100:] if len(backtest_results) > 100 else backtest_results
    print(f"\n  Using {len(test_races)} races for backtest")
    condition_results = run_nar_condition_backtest(
        test_races, lgb_model, xgb_model, weights, jockey_stats
    )

    # Step 5: Save model
    save_nar_model(lgb_model, xgb_model, lgb_auc, ensemble_auc, weights,
                   NAR_FEATURES, jockey_stats, condition_results)

    print("\n  NAR training complete!")
    return condition_results


if __name__ == '__main__':
    main()
