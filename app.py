import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import sqlite3
import os
from datetime import datetime
from itertools import combinations

st.set_page_config(page_title="KEIBA AI - 中央競馬専用", page_icon="🏇", layout="wide")

# ===== SQLite DB =====
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba_predictions.db")

INVESTMENT_PER_RACE = 700

# 条件別買い目ロジック
# WF backtest 2020-2025 (20,655R) リークフリーPattern A (AUC 0.8017)
# 実配当ROI: JRA公式配当データ(27,541R)×WFバックテスト
# 推定ROI: trio=o1*o2*o3*20 (参考値、実ROIの約2倍に過大評価)
CONDITION_PROFILES = {
    'A': {
        'label': '条件A',
        'desc': '8-14頭 / 1600m+ / 良〜稍重',
        'bet_type': 'trio',
        'bet_label': '三連複7点',
        'bet_detail': 'TOP1軸-TOP2,3-TOP2~6',
        'investment': 700,
        'roi': 205.3,      # 実配当ROI (JRA公式)
        'roi_estimated': 439.6,  # 推定ROI (参考値)
        'hit_rate': 44.5,
        'recommended': True,
        'wf_n': 6438,
    },
    'B': {
        'label': '条件B',
        'desc': '8-14頭 / 1600m+ / 重〜不良',
        'bet_type': 'trio',
        'bet_label': '三連複7点',
        'bet_detail': 'TOP1軸-TOP2,3-TOP2~6',
        'investment': 700,
        'roi': 236.9,      # 実配当ROI
        'roi_estimated': 445.1,
        'hit_rate': 45.2,
        'recommended': True,
        'wf_n': 847,
    },
    'C': {
        'label': '条件C',
        'desc': '15頭+ / 1600m+ / 良〜稍重',
        'bet_type': 'trio',
        'bet_label': '三連複7点',
        'bet_detail': 'TOP1軸-TOP2,3-TOP2~6',
        'investment': 700,
        'roi': 285.6,      # 実配当ROI
        'roi_estimated': 538.8,
        'hit_rate': 33.7,
        'recommended': True,
        'wf_n': 4774,
    },
    'D': {
        'label': '条件D',
        'desc': '1200-1400m（スプリント）',
        'bet_type': 'trio',
        'bet_label': '三連複7点',
        'bet_detail': 'TOP1軸-TOP2,3-TOP2~6',
        'investment': 700,
        'roi': 136.0,      # 実配当ROI (1200-1400m)
        'roi_estimated': 236.0,
        'hit_rate': 27.0,
        'recommended': True,
        'wf_n': 7254,
    },
    'E': {
        'label': '条件E',
        'desc': '7頭以下（少頭数）',
        'bet_type': 'umaren',
        'bet_label': '馬連1軸2流し',
        'bet_detail': 'TOP1軸-TOP2,TOP3',
        'investment': 700,
        'roi': 118.0,      # 実配当ROI (umaren)
        'roi_estimated': 145.2,
        'hit_rate': 53.4,
        'recommended': True,
        'wf_n': 461,
    },
    'X': {
        'label': '条件外',
        'desc': '15頭+ / 重〜不良',
        'bet_type': 'trio',
        'bet_label': '三連複7点',
        'bet_detail': 'TOP1軸-TOP2,3-TOP2~6',
        'investment': 700,
        'roi': 330.5,      # 実配当ROI
        'roi_estimated': 544.2,
        'hit_rate': 35.5,
        'recommended': True,
        'wf_n': 805,
    },
}





def classify_race_condition(race_info, num_horses, is_nar=False):
    """レース条件を分類してプロファイルを返す（中央競馬専用）。
    Returns: (condition_key, profile_dict)
    1000m以下は条件Dだが購入非推奨(recommended=False)。
    """
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

    profile = dict(CONDITION_PROFILES[cond_key])  # コピーして変更
    # 1000m以下は非推奨
    if cond_key == 'D' and dist <= 1000:
        profile['recommended'] = False
        profile['desc'] = '1000m以下（非推奨：ROI 85%）'
    return cond_key, profile

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT, race_name TEXT, race_date TEXT,
        course TEXT, distance INTEGER, surface TEXT, condition TEXT,
        horse_name TEXT, horse_num INTEGER, ai_rank INTEGER,
        ai_score REAL, odds REAL, predicted_at TEXT,
        actual_finish INTEGER DEFAULT NULL,
        is_top3_pred INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS race_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT UNIQUE, race_name TEXT,
        predicted_at TEXT, result_updated_at TEXT DEFAULT NULL,
        num_horses INTEGER, top1_name TEXT, top1_score REAL,
        trio_bets TEXT DEFAULT NULL,
        hit_trio INTEGER DEFAULT NULL,
        hit_combo TEXT DEFAULT NULL,
        payout INTEGER DEFAULT 0
    )""")
    # 既存テーブルにカラムがない場合追加（マイグレーション）
    for col_sql in [
        "ALTER TABLE race_results ADD COLUMN trio_bets TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN hit_trio INTEGER DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN hit_combo TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN payout INTEGER DEFAULT 0",
        "ALTER TABLE race_results ADD COLUMN is_nar INTEGER DEFAULT 0",
        "ALTER TABLE race_results ADD COLUMN wide_bets TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN hit_wide INTEGER DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN wide_payout INTEGER DEFAULT 0",
        "ALTER TABLE race_results ADD COLUMN buy_recommended INTEGER DEFAULT 1",
        "ALTER TABLE race_results ADD COLUMN bet_condition TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN bet_type TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN umaren_bets TEXT DEFAULT NULL",
    ]:
        try:
            c.execute(col_sql)
        except: pass
    conn.commit()
    conn.close()

def generate_trio_bets(df_sorted):
    """AI順位TOP1軸 - TOP2,TOP3 - TOP2~TOP6 の三連複フォーメーション7点を生成"""
    if len(df_sorted) < 3:
        return []
    top6 = df_sorted.head(min(6, len(df_sorted)))
    nums = [int(top6.iloc[i]['馬番']) for i in range(len(top6))]
    n1 = nums[0]  # 軸
    second = nums[1:3]  # TOP2, TOP3
    third = nums[1:min(6, len(nums))]  # TOP2~TOP6
    bets = set()
    for s in second:
        for t in third:
            combo = tuple(sorted({n1, s, t}))
            if len(combo) == 3:
                bets.add(combo)
    return [list(b) for b in sorted(bets)]


def generate_wide_bets(df_sorted):
    """AI順位TOP1軸 - TOP2,TOP3 のワイド1軸2流し(2点)を生成"""
    if len(df_sorted) < 3:
        return []
    nums = [int(df_sorted.iloc[i]['馬番']) for i in range(min(3, len(df_sorted)))]
    bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
    if bets[0] == bets[1]:
        return [bets[0]]
    return bets


def generate_umaren_bets(df_sorted):
    """AI順位TOP1軸 - TOP2,TOP3 の馬連1軸2流し(2点)を生成"""
    if len(df_sorted) < 3:
        return []
    nums = [int(df_sorted.iloc[i]['馬番']) for i in range(min(3, len(df_sorted)))]
    bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
    if bets[0] == bets[1]:
        return [bets[0]]
    return bets


def generate_bets_for_condition(df_sorted, cond_key, profile):
    """条件に応じた買い目を生成"""
    bet_type = profile['bet_type']
    if bet_type == 'trio':
        return generate_trio_bets(df_sorted), 'trio'
    elif bet_type == 'wide':
        return generate_wide_bets(df_sorted), 'wide'
    elif bet_type == 'umaren':
        return generate_umaren_bets(df_sorted), 'umaren'
    return [], 'none'


def save_prediction(race_id, race_name, race_info, df_sorted, is_nar=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("DELETE FROM predictions WHERE race_id = ?", (race_id,))
    c.execute("DELETE FROM race_results WHERE race_id = ?", (race_id,))
    for _, row in df_sorted.iterrows():
        c.execute("""INSERT INTO predictions
            (race_id, race_name, race_date, course, distance, surface, condition,
             horse_name, horse_num, ai_rank, ai_score, odds, predicted_at, is_top3_pred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (race_id, race_name, now[:10], race_info.get('course',''),
             race_info.get('distance',0), race_info.get('surface',''),
             race_info.get('condition',''), row['馬名'], int(row['馬番']),
             int(row['AI順位']), float(row['スコア']),
             float(row.get('単勝オッズ', 0)), now,
             1 if int(row['AI順位']) <= 3 else 0))
    top1 = df_sorted.iloc[0]
    trio_bets = generate_trio_bets(df_sorted)
    wide_bets = generate_wide_bets(df_sorted)
    umaren_bets = generate_umaren_bets(df_sorted)
    cond_key, profile = classify_race_condition(race_info, len(df_sorted), is_nar=is_nar)
    buy_rec = profile['recommended']
    c.execute("""INSERT INTO race_results
        (race_id, race_name, predicted_at, num_horses, top1_name, top1_score,
         trio_bets, wide_bets, umaren_bets, is_nar, buy_recommended, bet_condition, bet_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (race_id, race_name, now, len(df_sorted), top1['馬名'], float(top1['スコア']),
         json.dumps(trio_bets), json.dumps(wide_bets), json.dumps(umaren_bets),
         1 if is_nar else 0, 1 if buy_rec else 0, cond_key, profile['bet_type']))
    conn.commit()
    conn.close()

def update_actual_results(race_id, results_dict, payouts=None):
    """results_dict: {馬番: 着順}, payouts: {'trio': 金額, 'umaren': 金額, 'wide': 金額}"""
    if payouts is None:
        payouts = {'trio': 0, 'umaren': 0, 'wide': 0}
    # 後方互換: 旧呼び出し(trio_payout=int)対応
    if isinstance(payouts, (int, float)):
        payouts = {'trio': int(payouts), 'umaren': 0, 'wide': 0}
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 各馬の着順を更新
    for horse_num, finish in results_dict.items():
        c.execute("UPDATE predictions SET actual_finish = ? WHERE race_id = ? AND horse_num = ?",
                  (finish, race_id, horse_num))
    # 上位馬番を取得（同着考慮）
    top2_nums = set(n for n, f in results_dict.items() if f <= 2)
    top3_nums = set(n for n, f in results_dict.items() if f <= 3)
    # DBからbet_typeと各種買い目を取得
    c.execute("SELECT trio_bets, wide_bets, umaren_bets, bet_type FROM race_results WHERE race_id = ?", (race_id,))
    row = c.fetchone()
    hit_trio = 0
    hit_combo = None
    payout = 0
    if row:
        trio_bets_raw, wide_bets_raw, umaren_bets_raw, bet_type = row
        bet_type = bet_type or 'trio'  # NULLの場合デフォルトtrio

        # bet_typeに応じた的中判定
        if bet_type == 'trio':
            bets = json.loads(trio_bets_raw) if trio_bets_raw else []
            target_nums = top3_nums
            target_payout = payouts.get('trio', 0)
            for bet in bets:
                if set(bet).issubset(target_nums):
                    hit_trio = 1
                    hit_combo = json.dumps(sorted(bet))
                    payout = target_payout
                    break
        elif bet_type == 'umaren':
            bets = json.loads(umaren_bets_raw) if umaren_bets_raw else []
            target_payout = payouts.get('umaren', 0)
            for bet in bets:
                if set(bet).issubset(top2_nums):
                    hit_trio = 1  # hit_trioカラムを的中フラグとして流用
                    hit_combo = json.dumps(sorted(bet))
                    payout = target_payout
                    break
        elif bet_type == 'wide':
            bets = json.loads(wide_bets_raw) if wide_bets_raw else []
            target_payout = payouts.get('wide', 0)
            for bet in bets:
                if set(bet).issubset(top3_nums):
                    hit_trio = 1
                    hit_combo = json.dumps(sorted(bet))
                    payout = target_payout
                    break

        c.execute("""UPDATE race_results SET result_updated_at=?, hit_trio=?,
                     hit_combo=?, payout=? WHERE race_id=?""",
                  (now, hit_trio, hit_combo, payout, race_id))
    else:
        c.execute("""INSERT OR IGNORE INTO race_results
            (race_id, race_name, predicted_at, num_horses, top1_name, top1_score,
             trio_bets, hit_trio, hit_combo, payout, result_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (race_id, '(結果のみ)', now, len(results_dict), '', 0.0,
             None, hit_trio, hit_combo, payout, now))
    conn.commit()
    conn.close()

def _calc_stats_from_rows(rows):
    """race_results行リストから集計値を計算"""
    total = len(rows)
    settled = [r for r in rows if r['hit_trio'] is not None]
    sr = len(settled)
    hit_count = sum(1 for r in settled if r['hit_trio'] == 1)
    total_payout = sum(r['payout'] or 0 for r in settled)
    investment = sr * INVESTMENT_PER_RACE
    return {
        'total_races': total, 'settled_races': sr,
        'hit_count': hit_count,
        'hit_rate': hit_count / sr if sr > 0 else 0.0,
        'total_investment': investment,
        'total_payout': total_payout,
        'profit': total_payout - investment,
        'roi': (total_payout / investment * 100) if investment > 0 else 0.0,
    }

def get_dashboard_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM race_results ORDER BY predicted_at DESC")
    all_rows = [dict(r) for r in c.fetchall()]
    conn.close()
    jra_rows = [r for r in all_rows if not r.get('is_nar', 0)]
    stats = {
        'all': _calc_stats_from_rows(all_rows),
        'jra': _calc_stats_from_rows(jra_rows),
        'nar': {'total_races': 0, 'total_hits': 0, 'total_payout': 0, 'hit_rate': 0, 'roi': 0},
        'recent': all_rows[:10],
    }
    return stats

def delete_race_records(race_ids):
    """指定されたrace_idリストのレコードを削除"""
    if not race_ids:
        return 0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    placeholders = ','.join('?' for _ in race_ids)
    c.execute(f"DELETE FROM predictions WHERE race_id IN ({placeholders})", race_ids)
    c.execute(f"DELETE FROM race_results WHERE race_id IN ({placeholders})", race_ids)
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted

def delete_all_race_records():
    """全レコードを削除"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    c.execute("DELETE FROM race_results")
    conn.commit()
    conn.close()

def get_all_race_records():
    """全レース結果を取得（削除UI用）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT race_id, race_name, predicted_at, hit_trio, payout, is_nar, bet_type, bet_condition FROM race_results ORDER BY predicted_at DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

JRA_VENUES = ['東京', '中山', '阪神', '京都', '小倉', '新潟', '福島', '札幌', '函館', '中京']

def get_track_record_all():
    """TRACK RECORD用: 全レース結果+開催情報を取得"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT r.race_id, r.race_name, r.predicted_at, r.num_horses,
               r.hit_trio, r.payout, r.bet_condition, r.bet_type,
               r.trio_bets, r.wide_bets, r.umaren_bets, r.hit_combo,
               r.buy_recommended,
               (SELECT course FROM predictions p WHERE p.race_id = r.race_id LIMIT 1) as course,
               (SELECT race_date FROM predictions p WHERE p.race_id = r.race_id LIMIT 1) as race_date,
               (SELECT distance FROM predictions p WHERE p.race_id = r.race_id LIMIT 1) as distance,
               (SELECT surface FROM predictions p WHERE p.race_id = r.race_id LIMIT 1) as surface,
               (SELECT condition FROM predictions p WHERE p.race_id = r.race_id LIMIT 1) as track_condition
        FROM race_results r
        WHERE r.is_nar = 0
        ORDER BY r.predicted_at DESC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_predictions_for_race(race_id):
    """特定レースの全馬予測データを取得"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""SELECT horse_name, horse_num, ai_rank, ai_score, odds, actual_finish
                 FROM predictions WHERE race_id = ? ORDER BY ai_rank ASC""", (race_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_weekly_analysis():
    """週次分析データを取得。直近4週分のデータをコース/距離/馬場/頭数別に分析"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT p.race_id, p.race_name, p.race_date, p.course, p.distance, p.surface, p.condition,
               p.horse_name, p.horse_num, p.ai_rank, p.ai_score, p.odds, p.actual_finish,
               r.hit_trio, r.payout, r.is_nar, r.num_horses
        FROM predictions p
        LEFT JOIN race_results r ON p.race_id = r.race_id
        WHERE r.hit_trio IS NOT NULL
        ORDER BY p.race_date DESC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    if not rows:
        return None
    from datetime import timedelta
    today = datetime.now().date()
    last_monday = today - timedelta(days=today.weekday())
    weeks = {}
    for i in range(4):
        week_start = last_monday - timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)
        week_key = week_start.strftime('%m/%d') + '-' + week_end.strftime('%m/%d')
        weeks[week_key] = {'start': week_start, 'end': week_end, 'races': set(), 'hits': 0, 'investment': 0, 'payout': 0}
    race_data = {}
    for r in rows:
        rid = r['race_id']
        if rid not in race_data:
            race_data[rid] = r
    for rid, r in race_data.items():
        rd = r.get('race_date', '')[:10]
        try:
            race_date = datetime.strptime(rd, '%Y-%m-%d').date()
        except:
            continue
        for wk, wd in weeks.items():
            if wd['start'] <= race_date <= wd['end']:
                wd['races'].add(rid)
                if r.get('hit_trio') == 1:
                    wd['hits'] += 1
                wd['investment'] += INVESTMENT_PER_RACE
                wd['payout'] += r.get('payout', 0) or 0
                break
    analysis = {'weeks': {}, 'by_course': {}, 'by_distance': {}, 'by_condition': {}, 'by_field_size': {}}
    for wk, wd in weeks.items():
        n = len(wd['races'])
        analysis['weeks'][wk] = {
            'races': n, 'hits': wd['hits'],
            'hit_rate': wd['hits'] / n if n > 0 else 0,
            'investment': wd['investment'], 'payout': wd['payout'],
            'roi': wd['payout'] / wd['investment'] * 100 if wd['investment'] > 0 else 0,
        }
    for rid, r in race_data.items():
        course = r.get('course', '不明') or '不明'
        if course not in analysis['by_course']:
            analysis['by_course'][course] = {'races': 0, 'hits': 0, 'investment': 0, 'payout': 0}
        d = analysis['by_course'][course]
        d['races'] += 1
        if r.get('hit_trio') == 1: d['hits'] += 1
        d['investment'] += INVESTMENT_PER_RACE
        d['payout'] += r.get('payout', 0) or 0
    for rid, r in race_data.items():
        dist = r.get('distance', 0)
        if dist <= 1200: cat = '短距離(~1200)'
        elif dist <= 1600: cat = 'マイル(1201-1600)'
        elif dist <= 2000: cat = '中距離(1601-2000)'
        elif dist <= 2400: cat = '中長距離(2001-2400)'
        else: cat = '長距離(2401~)'
        if cat not in analysis['by_distance']:
            analysis['by_distance'][cat] = {'races': 0, 'hits': 0, 'investment': 0, 'payout': 0}
        d = analysis['by_distance'][cat]
        d['races'] += 1
        if r.get('hit_trio') == 1: d['hits'] += 1
        d['investment'] += INVESTMENT_PER_RACE
        d['payout'] += r.get('payout', 0) or 0
    for rid, r in race_data.items():
        cond = r.get('condition', '不明') or '不明'
        if cond not in analysis['by_condition']:
            analysis['by_condition'][cond] = {'races': 0, 'hits': 0, 'investment': 0, 'payout': 0}
        d = analysis['by_condition'][cond]
        d['races'] += 1
        if r.get('hit_trio') == 1: d['hits'] += 1
        d['investment'] += INVESTMENT_PER_RACE
        d['payout'] += r.get('payout', 0) or 0
    for rid, r in race_data.items():
        nh = r.get('num_horses', 0) or 0
        if nh <= 8: cat = '少頭数(~8頭)'
        elif nh <= 12: cat = '中頭数(9-12頭)'
        elif nh <= 16: cat = '多頭数(13-16頭)'
        else: cat = '大頭数(17頭~)'
        if cat not in analysis['by_field_size']:
            analysis['by_field_size'][cat] = {'races': 0, 'hits': 0, 'investment': 0, 'payout': 0}
        d = analysis['by_field_size'][cat]
        d['races'] += 1
        if r.get('hit_trio') == 1: d['hits'] += 1
        d['investment'] += INVESTMENT_PER_RACE
        d['payout'] += r.get('payout', 0) or 0
    return analysis

def render_weekly_report(analysis):
    """週次分析レポートをHTML描画"""
    if not analysis:
        return '<div class="ev-card"><span class="ev-lbl">分析データがありません。レース結果を登録してください。</span></div>'
    html = '<div class="ev-card">'
    html += '<div style="font-family:Oswald;font-size:0.8em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:8px;">WEEKLY TREND</div>'
    for wk, wd in analysis['weeks'].items():
        if wd['races'] == 0:
            continue
        hr = wd['hit_rate'] * 100
        roi = wd['roi']
        profit = wd['payout'] - wd['investment']
        profit_color = '#2ecc40' if profit >= 0 else '#ff4060'
        roi_color = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 70 else '#ff4060')
        html += f'<div class="ev-row">'
        html += f'<span class="ev-lbl">{wk} ({wd["races"]}R)</span>'
        html += f'<span style="font-family:Oswald;font-size:0.85em;">的中{wd["hits"]}R ({hr:.0f}%)</span>'
        html += f'<span style="font-family:Oswald;color:{roi_color} !important;">ROI {roi:.0f}%</span>'
        html += f'<span style="font-family:Oswald;color:{profit_color} !important;">{profit:+,}円</span>'
        html += '</div>'
    def render_category(data, title):
        if not data:
            return ''
        h = f'<div style="border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;padding-top:10px;">'
        h += f'<div style="font-family:Oswald;font-size:0.75em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:6px;">{title}</div>'
        sorted_items = sorted(data.items(), key=lambda x: x[1]['races'], reverse=True)
        for name, d in sorted_items:
            if d['races'] == 0:
                continue
            roi = d['payout'] / d['investment'] * 100 if d['investment'] > 0 else 0
            profit = d['payout'] - d['investment']
            pc = '#2ecc40' if profit >= 0 else '#ff4060'
            rc = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 70 else '#ff4060')
            bar_w = min(d['races'] * 8, 100)
            bar_color = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 70 else '#ff4060')
            h += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;font-size:0.85em;">'
            h += f'<span style="min-width:100px;color:#b0b8c8 !important;">{name}</span>'
            h += f'<div style="flex:1;height:14px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">'
            h += f'<div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:3px;"></div></div>'
            h += f'<span style="font-family:Oswald;font-size:0.82em;min-width:40px;color:{rc} !important;">{roi:.0f}%</span>'
            h += f'<span style="font-family:Oswald;font-size:0.82em;min-width:55px;color:{pc} !important;">{profit:+,}</span>'
            h += '</div>'
        h += '</div>'
        return h
    html += render_category(analysis['by_course'], 'BY COURSE')
    html += render_category(analysis['by_distance'], 'BY DISTANCE')
    html += render_category(analysis['by_condition'], 'BY CONDITION')
    html += render_category(analysis['by_field_size'], 'BY FIELD SIZE')
    worst_cats = []
    for cat_name, cat_data in [('コース', analysis['by_course']), ('距離', analysis['by_distance']),
                                 ('馬場', analysis['by_condition']), ('頭数', analysis['by_field_size'])]:
        for name, d in cat_data.items():
            if d['races'] >= 2 and d['investment'] > 0:
                roi = d['payout'] / d['investment'] * 100
                if roi < 50:
                    worst_cats.append((f'{cat_name}:{name}', d['races'], roi))
    if worst_cats:
        worst_cats.sort(key=lambda x: x[2])
        html += '<div style="border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;padding-top:10px;">'
        html += '<div style="font-family:Oswald;font-size:0.75em;color:#ff4060 !important;letter-spacing:2px;margin-bottom:6px;">LOSS PATTERN</div>'
        for name, races, roi in worst_cats[:5]:
            html += f'<div style="font-size:0.82em;color:#ff4060 !important;margin-bottom:2px;">&#9888; {name} ({races}R / ROI {roi:.0f}%) — 見送り推奨</div>'
        html += '</div>'
    html += '</div>'
    return html

def load_backtest_report():
    """backtest_results.jsonからバックテスト結果を読み込む"""
    bt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json")
    if not os.path.exists(bt_path):
        return None
    try:
        with open(bt_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def load_backtest_5year():
    """backtest_results_5year.jsonから5年分バックテスト結果を読み込む"""
    bt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results_5year.json")
    if not os.path.exists(bt_path):
        return None
    try:
        with open(bt_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def _classify_bt_race(r):
    """バックテスト結果の1レースを条件分類"""
    n = r.get('num_horses', 0)
    dist = r.get('distance', 0)
    cond = str(r.get('condition', ''))
    good = any(c in cond for c in ['良', '稍'])
    heavy = any(c in cond for c in ['重', '不'])
    if n <= 7:
        return 'E'
    if dist <= 1400:
        return 'D'
    if 8 <= n <= 14 and dist >= 1600 and good:
        return 'A'
    if 8 <= n <= 14 and dist >= 1600 and heavy:
        return 'B'
    if n >= 15 and dist >= 1600 and good:
        return 'C'
    return 'X'


def _compute_condition_roi(results):
    """条件別ROIを集計"""
    from collections import defaultdict
    stats = defaultdict(lambda: {
        'count': 0, 'trio_hit': 0, 'trio_payout': 0,
        'wide_hit': 0, 'wide_payout': 0,
        'umaren_hit': 0, 'umaren_payout': 0,
    })
    for r in results:
        cat = _classify_bt_race(r)
        s = stats[cat]
        s['count'] += 1
        if r.get('v9_trio_hit'):
            s['trio_hit'] += 1
            s['trio_payout'] += r.get('payouts', {}).get('trio', 0) if isinstance(r.get('payouts'), dict) else 0
        if r.get('v9_wide_hits'):
            s['wide_hit'] += 1
            s['wide_payout'] += r.get('v9_wide_payout') or 0
        if r.get('v9_umaren_hits'):
            s['umaren_hit'] += 1
            s['umaren_payout'] += r.get('v9_umaren_payout') or 0
    return dict(stats)


def render_5year_report(bt5):
    """5年分バックテスト結果をHTML描画（条件別ROI＋年別ROIグラフ付き）"""
    if not bt5:
        return '<div class="ev-card"><span class="ev-lbl">5年バックテスト未実施</span></div>'

    generated = bt5.get('generated_at', '')
    summary = bt5.get('central_5year_summary', {})
    yearly = bt5.get('yearly_summaries', {})
    results = bt5.get('central_results_5year', [])

    html = '<div class="ev-card">'
    html += '<div style="font-family:Oswald;font-size:0.8em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:8px;">'
    html += f'5-YEAR BACKTEST (2020-2025) &mdash; {generated[:10] if generated else "N/A"}</div>'

    # Overall summary table
    if summary:
        n_races = summary.get('v8', {}).get('n_races', 0) or summary.get('v9', {}).get('n_races', 0)
        html += f'<div style="font-size:0.85em;color:#b0b8c8 !important;margin-bottom:8px;">Overall &mdash; {n_races} races</div>'

        html += '<table style="width:100%;border-collapse:collapse;font-size:0.82em;margin-bottom:10px;">'
        html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.1);">'
        html += '<th style="text-align:left;padding:4px 6px;color:#6a6a80 !important;font-family:Oswald;">BET TYPE</th>'
        for ver in ['V8', 'V9']:
            html += f'<th colspan="2" style="text-align:center;padding:4px 6px;font-family:Oswald;">{ver}</th>'
        html += '</tr>'
        html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
        html += '<th></th>'
        for _ in ['V8', 'V9']:
            html += '<th style="text-align:center;padding:2px 4px;color:#6a6a80 !important;font-size:0.9em;">HIT</th>'
            html += '<th style="text-align:center;padding:2px 4px;color:#6a6a80 !important;font-size:0.9em;">ROI</th>'
        html += '</tr>'

        for label, key in [('Trio 7-bet', 'trio'), ('Wide 1ax-2flow', 'wide'), ('Umaren 1ax-2flow', 'umaren')]:
            html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            html += f'<td style="padding:4px 6px;color:#b0b8c8 !important;">{label}</td>'
            for ver in ['v8', 'v9']:
                d = summary.get(ver, {})
                hit_rate = d.get(f'{key}_hit_rate', 0)
                roi = d.get(f'{key}_roi', 0)
                hit_color = '#2ecc40' if hit_rate >= 30 else ('#f0c040' if hit_rate >= 15 else '#ff4060')
                roi_color = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 70 else '#ff4060')
                html += f'<td style="text-align:center;padding:4px;font-family:Oswald;color:{hit_color} !important;">{hit_rate:.1f}%</td>'
                html += f'<td style="text-align:center;padding:4px;font-family:Oswald;color:{roi_color} !important;">{roi:.1f}%</td>'
            html += '</tr>'
        html += '</table>'

    # ===== 条件別ROI表 =====
    if results:
        cond_stats = _compute_condition_roi(results)
        html += '<div style="border-top:1px solid rgba(255,255,255,0.06);margin:12px 0;padding-top:10px;">'
        html += '<div style="font-family:Oswald;font-size:0.75em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:8px;">CONDITION-BASED ROI (V9)</div>'

        html += '<table style="width:100%;border-collapse:collapse;font-size:0.8em;margin-bottom:10px;">'
        html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.1);">'
        html += '<th style="text-align:left;padding:4px 6px;color:#6a6a80 !important;">COND</th>'
        html += '<th style="text-align:left;padding:4px 6px;color:#6a6a80 !important;">DESC</th>'
        html += '<th style="text-align:center;padding:4px;color:#6a6a80 !important;">RACES</th>'
        html += '<th style="text-align:center;padding:4px;color:#6a6a80 !important;">BET</th>'
        html += '<th style="text-align:center;padding:4px;color:#6a6a80 !important;">HIT%</th>'
        html += '<th style="text-align:center;padding:4px;color:#6a6a80 !important;">ROI</th>'
        html += '<th style="text-align:center;padding:4px;color:#6a6a80 !important;">REC</th>'
        html += '</tr>'

        cond_order = [
            ('A', '8-14頭/1600m+/良稍', 'trio'),
            ('B', '8-14頭/1600m+/重不', 'trio'),
            ('C', '15頭+/1600m+/良稍', 'trio'),
            ('D', '1400m以下', 'trio'),
            ('E', '7頭以下', 'umaren'),
            ('X', '15頭+/重不', 'trio'),
        ]
        for ck, desc, best_bet in cond_order:
            s = cond_stats.get(ck)
            if not s or s['count'] == 0:
                continue
            n = s['count']
            if best_bet == 'trio':
                hit = s['trio_hit']
                inv = n * 700
                pay = s['trio_payout']
                bet_lbl = 'Trio'
            elif best_bet == 'wide':
                hit = s['wide_hit']
                inv = n * 200
                pay = s['wide_payout']
                bet_lbl = 'Wide'
            elif best_bet == 'umaren':
                hit = s['umaren_hit']
                inv = n * 200
                pay = s['umaren_payout']
                bet_lbl = 'Umaren'
            else:
                hit = s['trio_hit']
                inv = n * 700
                pay = s['trio_payout']
                bet_lbl = '-'

            hit_rate = hit / n * 100 if n > 0 else 0
            roi = pay / inv * 100 if inv > 0 else 0
            is_rec = roi >= 80 and best_bet != 'none'

            hit_c = '#2ecc40' if hit_rate >= 30 else ('#f0c040' if hit_rate >= 15 else '#ff4060')
            roi_c = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 80 else '#ff4060')
            rec_icon = '<span style="color:#2ecc40 !important;">&#9745;</span>' if is_rec else '<span style="color:#ff4060 !important;">&#9746;</span>'

            html += f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            html += f'<td style="padding:4px 6px;font-family:Oswald;color:#f0c040 !important;">{ck}</td>'
            html += f'<td style="padding:4px 6px;color:#b0b8c8 !important;font-size:0.9em;">{desc}</td>'
            html += f'<td style="text-align:center;padding:4px;font-family:Oswald;">{n}</td>'
            html += f'<td style="text-align:center;padding:4px;font-family:Oswald;">{bet_lbl}</td>'
            html += f'<td style="text-align:center;padding:4px;font-family:Oswald;color:{hit_c} !important;">{hit_rate:.1f}%</td>'
            html += f'<td style="text-align:center;padding:4px;font-family:Oswald;color:{roi_c} !important;">{roi:.1f}%</td>'
            html += f'<td style="text-align:center;padding:4px;">{rec_icon}</td>'
            html += '</tr>'
        html += '</table>'
        html += '</div>'

    # ===== 年別ROIグラフ =====
    if yearly:
        html += '<div style="border-top:1px solid rgba(255,255,255,0.06);margin:12px 0;padding-top:10px;">'
        html += '<div style="font-family:Oswald;font-size:0.75em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:8px;">V9 YEARLY ROI CHART</div>'

        for year in sorted(yearly.keys(), key=lambda x: int(x)):
            ys = yearly[year]
            v9d = ys.get('v9', {})
            n = v9d.get('n_races', 0)
            trio_rate = v9d.get('trio_hit_rate', 0)
            trio_roi = v9d.get('trio_roi', 0)
            wide_rate = v9d.get('wide_hit_rate', 0)
            wide_roi = v9d.get('wide_roi', 0)

            # Trio ROI bar
            trio_bar_w = min(trio_roi / 1.5, 100)
            trio_c = '#2ecc40' if trio_roi >= 100 else ('#f0c040' if trio_roi >= 80 else '#ff4060')
            # Wide ROI bar
            wide_bar_w = min(wide_roi / 1.5, 100)
            wide_c = '#2ecc40' if wide_roi >= 100 else ('#f0c040' if wide_roi >= 80 else '#ff4060')

            html += f'<div style="margin-bottom:8px;">'
            html += f'<div style="font-family:Oswald;font-size:0.85em;color:#b0b8c8 !important;margin-bottom:3px;">{year} ({n}R)</div>'
            # Trio bar
            html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;font-size:0.82em;">'
            html += f'<span style="min-width:45px;color:#6a6a80 !important;">Trio</span>'
            html += f'<div style="flex:1;height:16px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;position:relative;">'
            html += f'<div style="width:{trio_bar_w}%;height:100%;background:{trio_c};border-radius:3px;"></div>'
            # 100% line
            html += f'<div style="position:absolute;left:66.7%;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.3);"></div>'
            html += f'</div>'
            html += f'<span style="font-family:Oswald;min-width:110px;color:{trio_c} !important;">ROI {trio_roi:.0f}% / HIT {trio_rate:.0f}%</span>'
            html += '</div>'
            # Wide bar
            html += f'<div style="display:flex;align-items:center;gap:6px;font-size:0.82em;">'
            html += f'<span style="min-width:45px;color:#6a6a80 !important;">Wide</span>'
            html += f'<div style="flex:1;height:16px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;position:relative;">'
            html += f'<div style="width:{wide_bar_w}%;height:100%;background:{wide_c};border-radius:3px;"></div>'
            html += f'<div style="position:absolute;left:66.7%;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.3);"></div>'
            html += f'</div>'
            html += f'<span style="font-family:Oswald;min-width:110px;color:{wide_c} !important;">ROI {wide_roi:.0f}% / HIT {wide_rate:.0f}%</span>'
            html += '</div>'
            html += '</div>'

        html += '</div>'

    html += '</div>'
    return html


def render_backtest_report(bt_data):
    """V8/V9バックテスト比較レポートをHTML描画"""
    if not bt_data:
        return '<div class="ev-card"><span class="ev-lbl">バックテスト未実施</span></div>'

    generated = bt_data.get('generated_at', '')

    html = '<div class="ev-card">'
    html += '<div style="font-family:Oswald;font-size:0.8em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:8px;">'
    html += f'LEAK-FREE BACKTEST &mdash; {generated[:10] if generated else "N/A"}</div>'

    # Render each section (central / nar)
    sections = []
    if bt_data.get('central_summary'):
        sections.append(('Central (JRA)', bt_data['central_summary'], bt_data.get('central_results', [])))
    if bt_data.get('nar_summary'):
        sections.append(('NAR', bt_data['nar_summary'], bt_data.get('nar_results', [])))

    if not sections:
        html += '<div style="color:#6a6a80 !important;">No data</div></div>'
        return html

    for section_label, summary, results_list in sections:
        n_races = summary.get('v8', {}).get('n_races', 0) or summary.get('v9', {}).get('n_races', 0)
        html += f'<div style="font-size:0.85em;color:#b0b8c8 !important;margin-bottom:8px;margin-top:10px;">{section_label} 2025/10-12 &mdash; {n_races} races</div>'

        # Comparison table
        html += '<table style="width:100%;border-collapse:collapse;font-size:0.82em;margin-bottom:10px;">'
        html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.1);">'
        html += '<th style="text-align:left;padding:4px 6px;color:#6a6a80 !important;font-family:Oswald;">BET TYPE</th>'
        for ver in ['V8', 'V9']:
            html += f'<th colspan="2" style="text-align:center;padding:4px 6px;font-family:Oswald;">{ver}</th>'
        html += '</tr>'
        html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
        html += '<th></th>'
        for _ in ['V8', 'V9']:
            html += '<th style="text-align:center;padding:2px 4px;color:#6a6a80 !important;font-size:0.9em;">HIT</th>'
            html += '<th style="text-align:center;padding:2px 4px;color:#6a6a80 !important;font-size:0.9em;">ROI</th>'
        html += '</tr>'

        for label, key in [('Trio 7-bet', 'trio'), ('Wide 1ax-2flow', 'wide'), ('Umaren 1ax-2flow', 'umaren')]:
            html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            html += f'<td style="padding:4px 6px;color:#b0b8c8 !important;">{label}</td>'
            for ver in ['v8', 'v9']:
                d = summary.get(ver, {})
                hit_rate = d.get(f'{key}_hit_rate', 0)
                roi = d.get(f'{key}_roi', 0)
                hit_color = '#2ecc40' if hit_rate >= 30 else ('#f0c040' if hit_rate >= 15 else '#ff4060')
                roi_color = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 70 else '#ff4060')
                html += f'<td style="text-align:center;padding:4px;font-family:Oswald;color:{hit_color} !important;">{hit_rate:.1f}%</td>'
                html += f'<td style="text-align:center;padding:4px;font-family:Oswald;color:{roi_color} !important;">{roi:.1f}%</td>'
            html += '</tr>'
        html += '</table>'

    # Winner badge (central data)
    central = bt_data.get('central_summary', {})
    if central:
        v8_trio = central.get('v8', {}).get('trio_hit_rate', 0)
        v9_trio = central.get('v9', {}).get('trio_hit_rate', 0)
        if v9_trio > v8_trio:
            diff = v9_trio - v8_trio
            html += f'<div style="background:rgba(46,204,64,0.1);border:1px solid rgba(46,204,64,0.3);border-radius:6px;padding:6px 10px;margin-bottom:10px;">'
            html += f'<span style="font-family:Oswald;color:#2ecc40 !important;">V9 WINS</span>'
            html += f'<span style="color:#b0b8c8 !important;font-size:0.85em;"> &mdash; Trio hit rate +{diff:.1f}pp</span></div>'
        elif v8_trio > v9_trio:
            diff = v8_trio - v9_trio
            html += f'<div style="background:rgba(40,152,216,0.1);border:1px solid rgba(40,152,216,0.3);border-radius:6px;padding:6px 10px;margin-bottom:10px;">'
            html += f'<span style="font-family:Oswald;color:#2898d8 !important;">V8 WINS</span>'
            html += f'<span style="color:#b0b8c8 !important;font-size:0.85em;"> &mdash; Trio hit rate +{diff:.1f}pp</span></div>'

    # Loss pattern analysis (use central results)
    central_results = bt_data.get('central_results', [])
    if central_results:
        html += '<div style="border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;padding-top:10px;">'
        html += '<div style="font-family:Oswald;font-size:0.75em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:6px;">LOSS PATTERN (V9 Trio)</div>'

        v9_wins = [r for r in central_results if r.get('v9_trio_hit')]
        v9_losses = [r for r in central_results if not r.get('v9_trio_hit')]

        categories = [
            ('Field Size', [
                ('5-9', lambda r: 5 <= r.get('num_horses', 0) <= 9),
                ('10-14', lambda r: 10 <= r.get('num_horses', 0) <= 14),
                ('15-18', lambda r: 15 <= r.get('num_horses', 0) <= 18),
            ]),
            ('Distance', [
                ('Sprint', lambda r: r.get('distance', 0) <= 1400),
                ('Mile', lambda r: 1401 <= r.get('distance', 0) <= 1800),
                ('Mid', lambda r: 1801 <= r.get('distance', 0) <= 2200),
                ('Long', lambda r: r.get('distance', 0) >= 2201),
            ]),
        ]

        for cat_name, items in categories:
            html += f'<div style="font-size:0.78em;color:#6a6a80 !important;margin:6px 0 3px;">{cat_name}</div>'
            for label, fn in items:
                w = sum(1 for r in v9_wins if fn(r))
                total = w + sum(1 for r in v9_losses if fn(r))
                if total == 0:
                    continue
                rate = w / total * 100
                bar_w = min(rate * 2, 100)
                color = '#2ecc40' if rate >= 30 else ('#f0c040' if rate >= 15 else '#ff4060')
                html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;font-size:0.88em;">'
                html += f'<span style="min-width:70px;color:#b0b8c8 !important;">{label}</span>'
                html += f'<div style="flex:1;height:12px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">'
                html += f'<div style="width:{bar_w}%;height:100%;background:{color};border-radius:3px;"></div></div>'
                html += f'<span style="font-family:Oswald;font-size:0.85em;min-width:80px;color:{color} !important;">{w}/{total} ({rate:.0f}%)</span>'
                html += '</div>'

        html += '</div>'

    html += '</div>'
    return html


init_db()

# ===== CSS =====
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Oswald:wght@400;600;700&family=Noto+Sans+JP:wght@300;500;700;900&display=swap');
[data-testid="stAppViewContainer"] { background: #10141c; }
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
    background: linear-gradient(135deg, #142a4a 0%, #101c30 60%, #160e2e 100%);
    border-radius: 16px; padding: 22px; margin: 16px 0;
    border: 1px solid rgba(255,255,255,0.10); position: relative; overflow: hidden;
}
.grade-badge {
    display: inline-block; padding: 3px 10px; border-radius: 6px; font-family: 'Oswald';
    font-size: 0.78em; font-weight: 700; letter-spacing: 1px; margin-right: 6px; vertical-align: middle;
}
.grade-g1 { background: linear-gradient(90deg, #c8a030, #f0d060); color: #000 !important; }
.grade-g2 { background: linear-gradient(90deg, #2070c0, #40a0f0); color: #fff !important; }
.grade-g3 { background: linear-gradient(90deg, #1a8a50, #2ecc71); color: #fff !important; }
.grade-op { background: rgba(168,85,247,0.25); color: #c090ff !important; border: 1px solid rgba(168,85,247,0.4); }
.grade-list { background: rgba(0,212,255,0.15); color: #00d4ff !important; border: 1px solid rgba(0,212,255,0.3); }
.race-num { font-family: 'Oswald'; font-size: 0.85em; color: #8890a0 !important; margin-right: 6px; }
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
    background: #161a24; border-radius: 14px; padding: 18px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.06); position: relative; overflow: hidden;
}
.hcard::before { content: ''; position: absolute; top: 0; left: 0; bottom: 0; width: 4px; }
.hcard-g { background: linear-gradient(135deg, #1e1a10 0%, #161a24 60%); border: 1px solid rgba(240,192,64,0.15); }
.hcard-g::before { background: linear-gradient(180deg, #f0c040, #c89020); }
.hcard-s { background: linear-gradient(135deg, #181a1e 0%, #161a24 60%); border: 1px solid rgba(176,184,200,0.12); }
.hcard-s::before { background: linear-gradient(180deg, #b0b8c8, #808898); }
.hcard-b { background: linear-gradient(135deg, #1c1610 0%, #161a24 60%); border: 1px solid rgba(200,120,64,0.12); }
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
    background: #161a24; border-radius: 14px; padding: 18px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.06); position: relative; overflow: hidden;
}
.buy-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.buy-honmei::before { background: linear-gradient(90deg, #f0c040, #00e87b); }
.buy-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.buy-type { font-family: 'Oswald'; font-size: 0.8em; letter-spacing: 2px; padding: 3px 12px; border-radius: 6px; }
.bt-hon { background: rgba(240,192,64,0.15); color: #f0c040 !important; }
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

/* Table */
.htable { width: 100%; border-collapse: separate; border-spacing: 0 5px; }
.htable th { font-size: 0.62em; color: #6a6a80 !important; letter-spacing: 1px; padding: 6px 3px; text-align: center; font-weight: 500; }
.htable td { background: #161a24; padding: 9px 5px; text-align: center; font-size: 0.78em; }
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
.badge-v8 { background: rgba(0,232,123,0.2); color: #00e87b !important; border: 1px solid rgba(0,232,123,0.5); font-weight:700; }
.badge-v9 { background: rgba(168,85,247,0.2); color: #a855f7 !important; border: 1px solid rgba(168,85,247,0.5); font-weight:700; }
.badge-central { background: rgba(0,212,255,0.15); color: #00d4ff !important; border: 1px solid rgba(0,212,255,0.4); font-weight:700; margin-left:4px; }
.badge-nar { background: rgba(230,126,34,0.15); color: #e67e22 !important; border: 1px solid rgba(230,126,34,0.4); font-weight:700; margin-left:4px; }
.badge-v1 { background: rgba(255,255,255,0.08); color: #6a6a80 !important; }

.disclaimer {
    margin: 20px 0; padding: 12px; border-radius: 10px;
    background: rgba(255,64,96,0.06); border: 1px solid rgba(255,64,96,0.15);
    font-size: 0.7em; color: #6a6a80 !important; text-align: center;
}

/* EV Table */
.ev-card {
    background: #161a24; border-radius: 14px; padding: 18px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}
.ev-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04); }
.ev-row:last-child { border-bottom: none; }
.ev-lbl { font-size: 0.85em; color: #b0b8c8 !important; }
.ev-val { font-family: 'Oswald', sans-serif; font-size: 1.1em; font-weight: 700; }
.ev-hot { color: #2ecc40 !important; }
.ev-warm { color: #f0c040 !important; }
.ev-cold { color: #6a6a80 !important; }

/* Dashboard */
.dash-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0; }
.dash-item {
    text-align: center; padding: 12px 6px; border-radius: 10px;
    background: rgba(255,255,255,0.05);
}
.dash-num { font-family: 'Oswald', sans-serif; font-size: 1.6em; font-weight: 700; }
.dash-lbl { font-size: 0.7em; color: #6a6a80 !important; margin-top: 2px; }

/* System Check */
.sys-ok { background: rgba(46,204,64,0.08); border: 1px solid rgba(46,204,64,0.25); border-radius: 10px;
    padding: 10px 16px; margin-bottom: 12px; text-align: center; }
.sys-warn { background: rgba(255,64,96,0.08); border: 1px solid rgba(255,64,96,0.25); border-radius: 10px;
    padding: 10px 16px; margin-bottom: 12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===== Model & Data =====
@st.cache_resource(ttl=3600)
def load_model():
    import os, gzip
    # v8 → v6 → v5 → v3 → v2 → v1 の順で検索
    for fname in [
        "keiba_model_v8.pkl.gz", "keiba_model_v8.pkl",
        "keiba_model_v6.pkl.gz", "keiba_model_v6_pkl.gz", "keiba_model_v6.pkl",
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

@st.cache_resource(ttl=3600)
def load_v9_models():
    """v9中央モデル（Pattern A + Pattern B）を読み込み"""
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    models = {'central': None, 'central_live': None}
    # Pattern A（リークフリー/バックテスト評価用）
    fpath = os.path.join(base, 'keiba_model_v9_central.pkl')
    if os.path.exists(fpath):
        try:
            with open(fpath, 'rb') as f:
                models['central'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Pattern Aモデル読み込みエラー: {e}")
    # Pattern B（当日情報込み/実運用予測用）
    fpath_live = os.path.join(base, 'keiba_model_v9_central_live.pkl')
    if os.path.exists(fpath_live):
        try:
            with open(fpath_live, 'rb') as f:
                models['central_live'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Pattern Bモデル読み込みエラー: {e}")
    return models

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

# v9中央/地方モデル
_v9_models = load_v9_models()

def get_model_for_race(is_nar=False, use_live=True):
    """中央競馬モデルを返す。
    use_live=True: Pattern B（当日情報込み/実運用）を優先
    use_live=False: Pattern A（リークフリー/評価用）
    """
    if use_live:
        live_data = _v9_models.get('central_live')
        if live_data and isinstance(live_data, dict) and 'model' in live_data:
            return live_data, 'central_live'
    v9_data = _v9_models.get('central')
    if v9_data and isinstance(v9_data, dict) and 'model' in v9_data:
        return v9_data, 'central'
    return _loaded if isinstance(_loaded, dict) else {'model': _loaded, 'features': None, 'version': 'v1'}, 'default'

jockey_wr = load_jockey_wr()

# ===== 騎手データ自動更新（7日経過で自動実行） =====
def auto_update_jockey_wr():
    """前回更新から7日以上経過していたら騎手データを自動更新"""
    try:
        from update_jockey_wr import needs_update, update_jockey_wr as do_update
        if needs_update(days=7):
            do_update()
            # 更新後に再読み込み
            return load_jockey_wr()
    except Exception:
        pass
    return None

if 'jockey_wr_checked' not in st.session_state:
    st.session_state['jockey_wr_checked'] = True
    updated = auto_update_jockey_wr()
    if updated:
        jockey_wr = updated

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
        # 過去5走lag特徴量
        'prev2_finish': 5, 'prev3_finish': 5, 'prev4_finish': 5, 'prev5_finish': 5,
        'avg_finish_3r': 5.0, 'avg_finish_5r': 5.0,
        'best_finish_3r': 5, 'best_finish_5r': 5,
        'top3_count_3r': 0, 'top3_count_5r': 0,
        'finish_trend': 0,
        'prev2_last3f': 35.5,
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
            # 過去5走lag特徴量
            fl = finish_list  # newest first
            if len(fl) >= 2: result['prev2_finish'] = fl[1]
            if len(fl) >= 3: result['prev3_finish'] = fl[2]
            if len(fl) >= 4: result['prev4_finish'] = fl[3]
            if len(fl) >= 5: result['prev5_finish'] = fl[4]
            # 3走平均・最高・3着以内回数
            fl3 = fl[:min(3, len(fl))]
            result['avg_finish_3r'] = sum(fl3) / len(fl3)
            result['best_finish_3r'] = min(fl3)
            result['top3_count_3r'] = sum(1 for f in fl3 if f <= 3)
            # 5走平均・最高・3着以内回数
            fl5 = fl[:min(5, len(fl))]
            result['avg_finish_5r'] = sum(fl5) / len(fl5)
            result['best_finish_5r'] = min(fl5)
            result['top3_count_5r'] = sum(1 for f in fl5 if f <= 3)
            # 着順トレンド（直近が良いほど正）
            if len(fl) >= 3:
                result['finish_trend'] = fl[2] - fl[0]  # prev3 - prev1: positive = improving
            elif len(fl) >= 2:
                result['finish_trend'] = fl[1] - fl[0]
        # 2走前の上がり3F
        if len(agari_list) >= 2:
            result['prev2_last3f'] = agari_list[1]
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

# ===== Fetch Realtime Odds =====
def fetch_realtime_odds(race_id, is_nar=False):
    """netkeibaから単勝リアルタイムオッズを取得。{馬番: オッズ} を返す"""
    odds_dict = {}
    try:
        # 方法1: オッズJSON APIエンドポイント
        if is_nar:
            url = f"https://nar.netkeiba.com/api/api_get_nar_odds.html?race_id={race_id}&type=1"
        else:
            url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        try:
            data = resp.json()
            if isinstance(data, dict) and 'data' in data and isinstance(data['data'], dict):
                odds_data = data['data'].get('odds', data['data'])
                if isinstance(odds_data, dict):
                    # 単勝オッズ: key '1' の中に馬番->オッズ
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
                                except (ValueError, TypeError):
                                    pass
                            elif isinstance(vals, (int, float, str)):
                                try:
                                    odds_val = float(str(vals).replace(',', ''))
                                    if 1.0 <= odds_val <= 9999.9:
                                        odds_dict[umaban] = odds_val
                                except (ValueError, TypeError):
                                    pass
        except (ValueError, KeyError):
            pass
        if odds_dict:
            return odds_dict
        # 方法2: APIレスポンスをHTML断片として解析（フォールバック）
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 2:
                continue
            umaban = None
            odds_val = None
            for td in tds:
                text = td.get_text(strip=True)
                if umaban is None and text.isdigit() and 1 <= int(text) <= 18:
                    umaban = int(text)
                elif umaban is not None and odds_val is None:
                    try:
                        v = float(text.replace(',', ''))
                        if 1.0 <= v <= 9999.9:
                            odds_val = v
                    except:
                        pass
            if umaban and odds_val:
                odds_dict[umaban] = odds_val
        if odds_dict:
            return odds_dict
        # 方法3: オッズページを直接スクレイピング
        if is_nar:
            url2 = f"https://nar.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"
        else:
            url2 = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"
        resp2 = requests.get(url2, headers=HEADERS, timeout=10)
        resp2.encoding = "EUC-JP"
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        for row in soup2.select("tr"):
            tds = row.find_all("td")
            umaban = None
            odds_val = None
            for td in tds:
                text = td.get_text(strip=True)
                cls = " ".join(td.get("class", []))
                if ("Num" in cls or "Umaban" in cls or "num" in cls) and text.isdigit():
                    umaban = int(text)
                if "Odds" in cls or "odds" in cls:
                    m = re.search(r'[\d,]+\.?\d*', text)
                    if m:
                        try:
                            v = float(m.group().replace(',', ''))
                            if 1.0 <= v <= 9999.9:
                                odds_val = v
                        except:
                            pass
            if umaban and odds_val:
                odds_dict[umaban] = odds_val
    except Exception:
        pass
    return odds_dict

# ===== Fetch Training (Oikiri) Data =====
def fetch_training_data(race_id, is_nar=False):
    """netkeibaの追い切りページから各馬の調教評価を取得。
    返り値: {馬番: {'rank': 'A'/'B'/'C'/'D', 'evaluation': '好調教', 'label': '◎良化'/'○平凡'/'△不安'}}
    """
    training_dict = {}
    try:
        if is_nar:
            url = f"https://nar.netkeiba.com/race/oikiri.html?race_id={race_id}"
        else:
            url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        wrapper = soup.find("div", class_="OikiriAllWrapper")
        if not wrapper:
            return training_dict
        rows = wrapper.select("tr")
        for row in rows:
            # 馬番
            td_umaban = row.select_one("td.Umaban")
            if not td_umaban:
                continue
            try:
                umaban = int(td_umaban.get_text(strip=True))
            except (ValueError, TypeError):
                continue
            # 評価テキスト（好調教、上々、まずまず、変わり身など）
            td_critic = row.select_one("td.Training_Critic")
            evaluation = td_critic.get_text(strip=True) if td_critic else ""
            # ランク（A/B/C/D）- クラスが Rank_A, Rank_B 等
            rank = ""
            for td in row.find_all("td"):
                cls_list = td.get("class", [])
                for cls in cls_list:
                    if cls.startswith("Rank_"):
                        rank = cls.replace("Rank_", "")
                        break
                if rank:
                    break
            # ラベル変換
            if rank == "A":
                label = "◎良化"
            elif rank == "B":
                label = "○平凡"
            else:
                label = "△不安" if rank in ("C", "D") else ""
            training_dict[umaban] = {
                'rank': rank,
                'evaluation': evaluation,
                'label': label,
            }
    except Exception:
        pass
    return training_dict

# ===== Fetch Lap Time Data =====
def fetch_lap_times(race_id, is_nar=False):
    """netkeibaの過去レース結果ページからラップタイムを取得。
    返り値: {'laps': [12.5, 11.2, ...], 'pace': 'H'/'M'/'S', 'first_half': 34.5, 'second_half': 35.0}
    or None if not available.
    """
    try:
        url = f"https://db.netkeiba.com/race/{race_id}/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # ラップタイムは race_lap_cell クラスか、テーブル内のラップ行から取得
        lap_data = {'laps': [], 'pace': '', 'first_half': 0, 'second_half': 0}

        # パターン1: race_lap_cell (新レイアウト)
        lap_cell = soup.find("td", class_="race_lap_cell")
        if not lap_cell:
            # パターン2: nowrap_data_intro 内のラップ
            lap_cell = soup.find("div", class_="race_lap_data")

        if lap_cell:
            lap_text = lap_cell.get_text(strip=True)
            # "12.5-11.2-12.0-..." のようなフォーマット
            laps = re.findall(r'(\d+\.?\d*)', lap_text)
            lap_data['laps'] = [float(l) for l in laps if 9.0 <= float(l) <= 20.0]

        # パターン3: スパンセルからの取得
        if not lap_data['laps']:
            for span in soup.find_all("span"):
                text = span.get_text(strip=True)
                if re.match(r'^\d+\.\d\s*-\s*\d+\.\d', text):
                    laps = re.findall(r'(\d+\.\d)', text)
                    lap_data['laps'] = [float(l) for l in laps if 9.0 <= float(l) <= 20.0]
                    break

        # パターン4: テーブル全体からラップ行を探す
        if not lap_data['laps']:
            all_text = soup.get_text()
            # "ラップ" の近くにある数値列を探す
            m = re.search(r'ラップ[^\d]*?([\d\.]+\s*-\s*[\d\.]+(?:\s*-\s*[\d\.]+)+)', all_text)
            if m:
                laps = re.findall(r'(\d+\.\d)', m.group(1))
                lap_data['laps'] = [float(l) for l in laps if 9.0 <= float(l) <= 20.0]

        if lap_data['laps'] and len(lap_data['laps']) >= 4:
            mid = len(lap_data['laps']) // 2
            lap_data['first_half'] = round(sum(lap_data['laps'][:mid]), 1)
            lap_data['second_half'] = round(sum(lap_data['laps'][mid:]), 1)
            diff = lap_data['first_half'] - lap_data['second_half']
            if diff < -1.0:
                lap_data['pace'] = 'S'  # スロー（前半遅い）
            elif diff > 1.0:
                lap_data['pace'] = 'H'  # ハイ（前半速い）
            else:
                lap_data['pace'] = 'M'  # ミドル

        return lap_data if lap_data['laps'] else None

    except Exception:
        return None


def fetch_horse_lap_history(horse_id, n_races=5):
    """馬の過去nレースのラップタイム傾向を取得。
    返り値: {'avg_first_half': float, 'avg_second_half': float, 'pace_pattern': str,
             'avg_lap_variance': float}
    """
    try:
        url = f"https://db.netkeiba.com/horse/{horse_id}/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # 過去レースIDを取得
        race_ids = []
        table = soup.find("table", class_="db_h_race_results")
        if not table:
            return None

        for row in table.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 5:
                continue
            link = tds[4].find("a") if len(tds) > 4 else None
            if link:
                href = link.get("href", "")
                m = re.search(r'/race/(\d{12})/', href)
                if m:
                    race_ids.append(m.group(1))
            if len(race_ids) >= n_races:
                break

        if not race_ids:
            return None

        # 各レースのラップを取得
        first_halves = []
        second_halves = []
        all_variances = []

        for rid in race_ids[:n_races]:
            lap_data = fetch_lap_times(rid)
            if lap_data and lap_data['first_half'] > 0:
                first_halves.append(lap_data['first_half'])
                second_halves.append(lap_data['second_half'])
                if lap_data['laps']:
                    all_variances.append(np.std(lap_data['laps']))
            time.sleep(0.5)

        if not first_halves:
            return None

        avg_fh = round(np.mean(first_halves), 1)
        avg_sh = round(np.mean(second_halves), 1)
        diff = avg_fh - avg_sh
        if diff < -1.0:
            pattern = 'S'
        elif diff > 1.0:
            pattern = 'H'
        else:
            pattern = 'M'

        return {
            'avg_first_half': avg_fh,
            'avg_second_half': avg_sh,
            'pace_pattern': pattern,
            'avg_lap_variance': round(np.mean(all_variances), 2) if all_variances else 0,
            'n_races': len(first_halves),
        }
    except Exception:
        return None


# ===== Fetch Realtime Track Condition =====
def fetch_track_condition(race_id, is_nar=False):
    """netkeibaから当日の馬場状態をリアルタイム取得。
    返り値: {'surface': '芝'/'ダ', 'condition': '良'/'稍重'/'重'/'不良', 'changed': bool}
    """
    result = {'condition': None, 'changed': False}
    try:
        if is_nar:
            url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
        else:
            url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        # RaceData01から馬場状態を取得
        d01 = soup.find("div", class_="RaceData01")
        if d01:
            text = d01.get_text(strip=True)
            cm = re.search(r'馬場:(\S+)', text)
            if not cm:
                cm = re.search(r'(良|稍重|稍|重|不良)', text)
            if cm:
                result['condition'] = cm.group(1)
        # 天候情報
        weather_tag = soup.find("span", class_="Weather")
        if weather_tag:
            weather_icon = weather_tag.find("img")
            if weather_icon:
                alt = weather_icon.get("alt", "")
                result['weather'] = alt
            else:
                result['weather'] = weather_tag.get_text(strip=True)
    except Exception:
        pass
    return result


def adjust_scores_for_track(df, condition, original_condition, surface, rank_map):
    """馬場悪化時にスコアを補正する。
    - 馬場悪化: 逃げ・先行有利度UP、追込有利度DOWN
    - ダート馬場悪化: パワー型・先行有利
    - 各馬の馬場適性を考慮
    """
    if not condition or condition == original_condition:
        return df, rank_map, False

    cond_level = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
    orig_lv = cond_level.get(original_condition, 0)
    new_lv = cond_level.get(condition, 0)

    if new_lv <= orig_lv:
        return df, rank_map, False  # 馬場改善 or 変化なし

    # 馬場悪化度合い
    delta = new_lv - orig_lv  # 1-3

    # 脚質別スコア補正
    style_adj = {
        1: 0.015 * delta,   # 逃げ: UP
        2: 0.020 * delta,   # 先行: UP
        3: -0.005 * delta,  # 差し: やや DOWN
        4: -0.020 * delta,  # 追込: DOWN
    }

    if 'スコア' in df.columns and '脚質' in df.columns:
        for style, adj in style_adj.items():
            mask = df['脚質'] == style
            df.loc[mask, 'スコア'] = df.loc[mask, 'スコア'] + adj

        # 馬場適性が低い馬はペナルティ
        if '馬場適性' in df.columns:
            # 馬場適性が0.3以下の馬は悪化でさらに不利
            low_apt = df['馬場適性'] < 0.3
            df.loc[low_apt, 'スコア'] = df.loc[low_apt, 'スコア'] - 0.01 * delta
            df.loc[low_apt, '馬場警告'] = True

        # スコア再順位付け
        df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
        df = df.sort_values('AI順位')

    # rank_mapも更新（逃げ先行の有利度を上げる）
    updated_rank_map = dict(rank_map)
    # 先行の評価を上げる
    for style in [1, 2]:
        if style in updated_rank_map:
            lbl, css, reason = updated_rank_map[style]
            if lbl in ['△', '×']:
                if delta >= 2:
                    updated_rank_map[style] = ('○', 'good', reason + " 馬場悪化で前有利")
                else:
                    updated_rank_map[style] = ('△', 'fair', reason + " 馬場悪化傾向")
            elif lbl == '○':
                updated_rank_map[style] = ('◎', 'best', reason + " 馬場悪化で前有利")
    # 追込の評価を下げる
    if 4 in updated_rank_map:
        lbl, css, reason = updated_rank_map[4]
        if lbl in ['◎', '○']:
            updated_rank_map[4] = ('△', 'fair', reason + " 馬場悪化で届かず")
        elif lbl == '△':
            updated_rank_map[4] = ('×', 'bad', reason + " 馬場悪化で届かず")

    return df, updated_rank_map, True

# ===== 展開予測モデル =====
def predict_race_pace(df, distance, surface, num_horses):
    """レース全体の展開を予測。各馬の脚質分布からペースを推定し、脚質別の有利不利を返す。
    Returns: (pace_label, pace_reason, pace_advantage_dict)
    pace: 'high'/'middle'/'slow'
    pace_advantage: {脚質コード: スコア補正値}
    """
    if '脚質' not in df.columns or len(df) == 0:
        return 'middle', 'データ不足', {1: 0, 2: 0, 3: 0, 4: 0}
    styles = df['脚質'].value_counts().to_dict()
    nige = styles.get(1, 0)
    senk = styles.get(2, 0)
    sasi = styles.get(3, 0)
    oiko = styles.get(4, 0)
    front_runners = nige + senk
    front_ratio = front_runners / max(num_horses, 1)
    # ペース判定
    pace = 'middle'
    reason = ''
    if nige >= 3 or (nige >= 2 and front_ratio >= 0.55):
        pace = 'high'
        reason = f'逃げ{nige}頭+先行{senk}頭で先行争い激化'
    elif nige >= 2 and distance <= 1400:
        pace = 'high'
        reason = f'短距離で逃げ{nige}頭 ハイペース濃厚'
    elif nige <= 1 and front_ratio <= 0.35:
        pace = 'slow'
        reason = f'逃げ{nige}頭のみ スローペース濃厚'
    elif nige == 0:
        pace = 'slow'
        reason = f'逃げ馬不在 超スローの可能性'
    else:
        pace = 'middle'
        reason = f'逃げ{nige} 先行{senk} 差し{sasi} 追込{oiko}'
    # 距離補正
    if distance >= 2400 and pace == 'middle' and nige <= 1:
        pace = 'slow'
        reason += ' 長距離でペース落ち'
    elif distance <= 1200 and pace == 'middle' and nige >= 2:
        pace = 'high'
        reason += ' スプリント戦で流れる'
    # 脚質別補正値
    if pace == 'high':
        adv = {1: -0.025, 2: -0.010, 3: 0.020, 4: 0.025}
    elif pace == 'slow':
        adv = {1: 0.025, 2: 0.020, 3: -0.010, 4: -0.025}
    else:
        adv = {1: 0.005, 2: 0.010, 3: 0.005, 4: -0.010}
    return pace, reason, adv

def render_pace_prediction(pace, reason, adv):
    """展開予測パネルのHTML生成"""
    pace_labels = {'high': 'HIGH PACE', 'middle': 'MIDDLE PACE', 'slow': 'SLOW PACE'}
    pace_colors = {'high': '#ff4060', 'middle': '#f0c040', 'slow': '#00d4ff'}
    pace_icons = {'high': '🔥', 'middle': '⚖️', 'slow': '🧊'}
    label = pace_labels.get(pace, 'MIDDLE')
    color = pace_colors.get(pace, '#f0c040')
    icon = pace_icons.get(pace, '')
    html = f'<div style="margin:10px 0;padding:14px;background:linear-gradient(135deg,rgba({",".join(str(int(color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.08),transparent);border:1px solid {color}30;border-radius:12px;">'
    html += f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
    html += f'<span style="font-family:Oswald;font-size:0.8em;letter-spacing:2px;color:#8890a0 !important;">PACE PREDICTION</span>'
    html += f'<span style="font-family:Oswald;font-weight:700;font-size:1.1em;color:{color} !important;letter-spacing:1px;">{icon} {label}</span></div>'
    html += f'<div style="font-size:0.8em;color:#a0a8b8 !important;margin-bottom:8px;">{reason}</div>'
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">'
    style_names = {1: '逃げ', 2: '先行', 3: '差し', 4: '追込'}
    for sid in [1, 2, 3, 4]:
        v = adv.get(sid, 0)
        if v > 0.015:
            vc, vl = '#00e87b', '◎有利'
        elif v > 0:
            vc, vl = '#f0c040', '○やや有利'
        elif v > -0.015:
            vc, vl = '#b0b8c8', '△普通'
        else:
            vc, vl = '#ff4060', '×不利'
        html += f'<div style="text-align:center;padding:6px 2px;background:rgba(255,255,255,0.03);border-radius:6px;">'
        html += f'<div style="font-size:0.75em;color:#8890a0 !important;">{style_names[sid]}</div>'
        html += f'<div style="font-weight:700;color:{vc} !important;font-size:0.85em;">{vl}</div></div>'
    html += '</div></div>'
    return html

# ===== 馬場バイアス自動検出 =====
def fetch_today_bias(race_id, course_name, is_nar=False):
    """当日の同競馬場の前レース結果から馬場バイアスを検出。
    race_id(12桁)の末尾2桁がレース番号。1R〜(現在R-1)の結果を取得して分析。
    Returns: bias_info dict
    """
    bias_info = {
        'front_bias': 0.0, 'inner_bias': 0.0,
        'front_desc': '', 'inner_desc': '',
        'analyzed_races': 0, 'details': [],
    }
    try:
        if len(race_id) < 4:
            return bias_info
        current_race_num = int(race_id[-2:])
        base_id = race_id[:-2]
        if current_race_num <= 1:
            return bias_info
        front_finishes = []  # (通過順位, 着順, 枠番) のリスト
        all_entries = []
        races_analyzed = 0
        # 最大5レース分（直近を優先）
        start_r = max(1, current_race_num - 5)
        for r_num in range(start_r, current_race_num):
            rid = f"{base_id}{r_num:02d}"
            try:
                if is_nar:
                    url = f"https://nar.netkeiba.com/race/result.html?race_id={rid}"
                else:
                    url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
                resp = requests.get(url, headers=HEADERS, timeout=8)
                resp.encoding = "EUC-JP"
                soup = BeautifulSoup(resp.text, "html.parser")
                rows = soup.select("tr.HorseList")
                if not rows:
                    continue
                races_analyzed += 1
                for row in rows:
                    tds = row.find_all("td")
                    if len(tds) < 5:
                        continue
                    # 着順
                    finish_td = row.select_one("td.Result_Num")
                    ft = finish_td.get_text(strip=True) if finish_td else tds[0].get_text(strip=True)
                    if not ft.isdigit():
                        continue
                    finish = int(ft)
                    # 枠番
                    waku = 0
                    waku_td = row.select_one("td.Waku span")
                    if waku_td:
                        wt = waku_td.get_text(strip=True)
                        if wt.isdigit():
                            waku = int(wt)
                    # 馬番
                    umaban = 0
                    for td in tds:
                        cls = " ".join(td.get("class", []))
                        if "Txt_C" in cls and "Horse" not in cls:
                            t = td.get_text(strip=True)
                            if t.isdigit() and 1 <= int(t) <= 18:
                                umaban = int(t)
                                break
                    # 通過順位
                    pass_pos = 0
                    for td in tds:
                        txt = td.get_text(strip=True).replace(' ','').replace('－','-').replace('-','-')
                        pn = re.findall(r'\d+', txt)
                        if len(pn) >= 2 and re.match(r'^\d{1,2}-\d{1,2}', txt):
                            pass_pos = int(pn[0])
                            break
                    num_h = len([r for r in rows if r.select_one("td.Result_Num") and r.select_one("td.Result_Num").get_text(strip=True).isdigit()])
                    all_entries.append({
                        'finish': finish, 'waku': waku, 'umaban': umaban,
                        'pass_pos': pass_pos, 'num_horses': num_h,
                    })
                time.sleep(0.3)
            except Exception:
                continue
        bias_info['analyzed_races'] = races_analyzed
        if not all_entries or races_analyzed == 0:
            return bias_info
        # === 前残りバイアス分析 ===
        front_horses = [e for e in all_entries if 1 <= e['pass_pos'] <= 3]
        back_horses = [e for e in all_entries if e['pass_pos'] >= 6]
        if front_horses:
            front_avg_finish = sum(e['finish'] for e in front_horses) / len(front_horses)
            front_top3_rate = sum(1 for e in front_horses if e['finish'] <= 3) / len(front_horses)
        else:
            front_avg_finish = 5.0
            front_top3_rate = 0.3
        if back_horses:
            back_avg_finish = sum(e['finish'] for e in back_horses) / len(back_horses)
            back_top3_rate = sum(1 for e in back_horses if e['finish'] <= 3) / len(back_horses)
        else:
            back_avg_finish = 5.0
            back_top3_rate = 0.3
        # 前残りスコア: -1(差し有利) 〜 +1(前残り)
        front_bias = 0.0
        if front_horses and back_horses:
            finish_diff = back_avg_finish - front_avg_finish
            rate_diff = front_top3_rate - back_top3_rate
            front_bias = np.clip(finish_diff * 0.1 + rate_diff * 0.5, -1.0, 1.0)
        bias_info['front_bias'] = front_bias
        if front_bias > 0.3:
            bias_info['front_desc'] = f'前残り傾向 (先行3着内率{front_top3_rate:.0%})'
        elif front_bias < -0.3:
            bias_info['front_desc'] = f'差し有利 (後方3着内率{back_top3_rate:.0%})'
        else:
            bias_info['front_desc'] = 'フラット'
        # === 枠バイアス分析 ===
        inner = [e for e in all_entries if 1 <= e['waku'] <= 3]
        outer = [e for e in all_entries if e['waku'] >= 6]
        if inner and outer:
            inner_avg = sum(e['finish'] for e in inner) / len(inner)
            outer_avg = sum(e['finish'] for e in outer) / len(outer)
            inner_top3 = sum(1 for e in inner if e['finish'] <= 3) / len(inner)
            outer_top3 = sum(1 for e in outer if e['finish'] <= 3) / len(outer)
            inner_bias = np.clip((outer_avg - inner_avg) * 0.1 + (inner_top3 - outer_top3) * 0.5, -1.0, 1.0)
        else:
            inner_bias = 0.0
            inner_top3 = 0.0
            outer_top3 = 0.0
        bias_info['inner_bias'] = inner_bias
        if inner_bias > 0.3:
            bias_info['inner_desc'] = f'内枠有利 (内3着内率{inner_top3:.0%})'
        elif inner_bias < -0.3:
            bias_info['inner_desc'] = f'外枠有利 (外3着内率{outer_top3:.0%})'
        else:
            bias_info['inner_desc'] = 'フラット'
    except Exception:
        pass
    return bias_info

def apply_bias_adjustment(df, bias_info, rank_map):
    """馬場バイアスに基づいてスコアを補正（最大±10%）。
    前残りバイアス: 逃げ・先行UP、差し・追込DOWN
    枠バイアス: 内枠UP or 外枠UP
    """
    if not bias_info or bias_info.get('analyzed_races', 0) == 0:
        return df, rank_map
    front_b = bias_info.get('front_bias', 0.0)
    inner_b = bias_info.get('inner_bias', 0.0)
    if '脚質' in df.columns and abs(front_b) > 0.15:
        # 前残りバイアス補正（最大±5%）
        style_adj = {
            1: front_b * 0.04,   # 逃げ
            2: front_b * 0.03,   # 先行
            3: -front_b * 0.02,  # 差し
            4: -front_b * 0.04,  # 追込
        }
        for style, adj in style_adj.items():
            mask = df['脚質'] == style
            df.loc[mask, 'スコア'] = df.loc[mask, 'スコア'] * (1.0 + adj)
    if '枠番' in df.columns and abs(inner_b) > 0.15:
        # 枠バイアス補正（最大±5%）
        max_waku = df['枠番'].max() if len(df) > 0 else 8
        mid = max_waku / 2.0
        for idx in df.index:
            waku = df.at[idx, '枠番']
            # 内枠有利(inner_b>0): 内枠ほどプラス
            pos = (mid - waku) / mid  # -1(外) ~ +1(内)
            adj = inner_b * pos * 0.04
            df.at[idx, 'スコア'] = df.at[idx, 'スコア'] * (1.0 + adj)
    # 再ランキング
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    # rank_map更新
    if abs(front_b) > 0.3:
        updated = dict(rank_map)
        if front_b > 0:
            for s in [1, 2]:
                if s in updated:
                    lbl, css, r = updated[s]
                    if lbl in ['△', '×']:
                        updated[s] = ('○', 'good', r + ' 当日前残り傾向')
            if 4 in updated:
                lbl, css, r = updated[4]
                if lbl in ['◎', '○']:
                    updated[4] = ('△', 'fair', r + ' 当日前残り傾向')
        else:
            for s in [3, 4]:
                if s in updated:
                    lbl, css, r = updated[s]
                    if lbl in ['△', '×']:
                        updated[s] = ('○', 'good', r + ' 当日差し有利')
        rank_map = updated
    return df, rank_map

def render_bias_panel(bias_info, course_name):
    """馬場バイアスパネルのHTML生成"""
    n = bias_info.get('analyzed_races', 0)
    if n == 0:
        return ''
    fb = bias_info.get('front_bias', 0)
    ib = bias_info.get('inner_bias', 0)
    fd = bias_info.get('front_desc', 'フラット')
    ind = bias_info.get('inner_desc', 'フラット')
    # 全体色
    if abs(fb) > 0.3 or abs(ib) > 0.3:
        border_c = '#e67e22'
        bg = 'rgba(230,126,34,0.06)'
    else:
        border_c = '#3498db'
        bg = 'rgba(52,152,219,0.06)'
    html = f'<div style="margin:10px 0;padding:14px;background:{bg};border:1px solid {border_c}30;border-radius:12px;">'
    html += f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
    html += f'<span style="font-family:Oswald;font-size:0.8em;letter-spacing:2px;color:#8890a0 !important;">TRACK BIAS ({course_name} 本日{n}R分析)</span></div>'
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
    # 前残りバイアス
    if fb > 0.3:
        fc, fi = '#ff4060', '⬆️'
    elif fb < -0.3:
        fc, fi = '#00d4ff', '⬇️'
    else:
        fc, fi = '#b0b8c8', '➡️'
    html += f'<div style="padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;text-align:center;">'
    html += f'<div style="font-size:0.7em;color:#8890a0 !important;">脚質傾向</div>'
    html += f'<div style="font-weight:700;color:{fc} !important;font-size:0.9em;">{fi} {fd}</div></div>'
    # 枠バイアス
    if ib > 0.3:
        ic, ii = '#2ecc71', '⬅️'
    elif ib < -0.3:
        ic, ii = '#e67e22', '➡️'
    else:
        ic, ii = '#b0b8c8', '↔️'
    html += f'<div style="padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;text-align:center;">'
    html += f'<div style="font-size:0.7em;color:#8890a0 !important;">枠傾向</div>'
    html += f'<div style="font-weight:700;color:{ic} !important;font-size:0.9em;">{ii} {ind}</div></div>'
    html += '</div></div>'
    return html

# ===== 起動時自動チェック =====
@st.cache_data(ttl=300)
def run_system_checks():
    """全システムの動作確認。返り値: [(name, ok, detail), ...]"""
    checks = []
    # 1. モデルファイル確認
    model_files = ["keiba_model_v8.pkl", "keiba_model_v8.pkl.gz"]
    found = any(os.path.exists(f) for f in model_files)
    v9c = os.path.exists("keiba_model_v9_central.pkl")
    v9_text = f' / v9: 中央' if v9c else ''
    checks.append(('モデルファイル', found, f'v8検出{v9_text}' if found else 'モデルファイルが見つかりません'))
    # 2. 特徴量チェック
    try:
        test_features = FEATURES if FEATURES else FEATURES_V1
        checks.append(('特徴量定義', len(test_features) > 0, f'{len(test_features)}個の特徴量'))
    except Exception as e:
        checks.append(('特徴量定義', False, str(e)[:50]))
    # 3. netkeiba接続確認
    try:
        r = requests.get("https://race.netkeiba.com/", headers=HEADERS, timeout=8)
        ok = r.status_code == 200
        checks.append(('netkeiba接続', ok, f'HTTP {r.status_code}'))
    except Exception as e:
        checks.append(('netkeiba接続', False, str(e)[:50]))
    # 4. オッズ取得テスト（軽量チェック - APIエンドポイントの応答確認）
    try:
        r = requests.get("https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=000000000000&type=1",
                        headers=HEADERS, timeout=8)
        checks.append(('オッズAPI', r.status_code in [200, 404], f'API応答あり (HTTP {r.status_code})'))
    except Exception as e:
        checks.append(('オッズAPI', False, str(e)[:50]))
    # 5. 調教データ取得テスト
    try:
        r = requests.get("https://race.netkeiba.com/race/oikiri.html?race_id=000000000000",
                        headers=HEADERS, timeout=8)
        checks.append(('調教データ', r.status_code in [200, 404], f'応答あり (HTTP {r.status_code})'))
    except Exception as e:
        checks.append(('調教データ', False, str(e)[:50]))
    # 6. DB整合性確認
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM race_results")
        cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions")
        pcnt = c.fetchone()[0]
        conn.close()
        checks.append(('DB整合性', True, f'race_results: {cnt}件, predictions: {pcnt}件'))
    except Exception as e:
        checks.append(('DB整合性', False, str(e)[:50]))
    # 7. 騎手データ確認
    try:
        jwr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jockey_wr.json")
        if os.path.exists(jwr_path):
            mtime = os.path.getmtime(jwr_path)
            days_ago = (datetime.now() - datetime.fromtimestamp(mtime)).days
            ok = days_ago <= 30
            checks.append(('騎手データ', ok, f'最終更新: {days_ago}日前' + ('' if ok else ' (30日超過)')))
        else:
            checks.append(('騎手データ', False, 'jockey_wr.json が見つかりません'))
    except Exception as e:
        checks.append(('騎手データ', False, str(e)[:50]))
    return checks

# ===== Fetch Race Results =====
def fetch_race_results(race_id, is_nar=False):
    """netkeibaのレース結果ページから着順と払戻金を取得。
    race.netkeiba.com（新形式）と db.netkeiba.com（旧形式）の両方に対応。
    返り値: (results_dict={馬番:着順}, payouts={'trio':金額, 'umaren':金額, 'wide':金額})"""
    results = {}
    trio_payout = 0
    umaren_payout = 0
    wide_payout = 0
    # 新形式（race.netkeiba.com）と旧形式（db.netkeiba.com）の両方を試す
    urls = []
    if is_nar:
        urls.append(f"https://nar.netkeiba.com/race/result.html?race_id={race_id}")
    else:
        urls.append(f"https://race.netkeiba.com/race/result.html?race_id={race_id}")
    urls.append(f"https://db.netkeiba.com/race/{race_id}/")  # 旧形式フォールバック
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = "EUC-JP"
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            is_old_format = "db.netkeiba.com" in url
            # ===== 着順取得 =====
            if is_old_format:
                # 旧形式: table.race_table_01 → td[0]=着順, td[1]=枠番, td[2]=馬番
                table = soup.find("table", class_="race_table_01")
                rows = table.find_all("tr") if table else []
            else:
                # 新形式: tr.HorseList → td.Result_Num, td.Txt_C
                rows = soup.select("tr.HorseList")
                if not rows:
                    rows = soup.select("table.RaceTable01 tr")
                if not rows:
                    table = soup.find("table", class_=re.compile(r"Result|Race"))
                    if table:
                        rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) < 3:
                    continue
                # 着順
                if is_old_format:
                    finish_text = tds[0].get_text(strip=True)
                else:
                    finish_td = row.select_one("td.Result_Num")
                    finish_text = finish_td.get_text(strip=True) if finish_td else tds[0].get_text(strip=True)
                if not finish_text.isdigit():
                    continue
                finish = int(finish_text)
                # 馬番
                umaban = None
                if is_old_format:
                    # 旧形式: td[2]が馬番
                    text = tds[2].get_text(strip=True)
                    if text.isdigit() and 1 <= int(text) <= 18:
                        umaban = int(text)
                else:
                    # 新形式: Txt_Cクラスのtd（Horse除外）
                    for td in tds:
                        cls = " ".join(td.get("class", []))
                        if "Txt_C" in cls and "Horse" not in cls:
                            text = td.get_text(strip=True)
                            if text.isdigit() and 1 <= int(text) <= 18:
                                umaban = int(text)
                                break
                # フォールバック: 3番目のtd
                if umaban is None and len(tds) >= 3:
                    text = tds[2].get_text(strip=True)
                    if text.isdigit() and 1 <= int(text) <= 18:
                        umaban = int(text)
                if umaban and 1 <= finish <= 30:
                    results[umaban] = finish
            # ===== 払戻金取得（三連複・馬連・ワイド） =====
            if is_old_format:
                payout_tables = soup.find_all("table", class_="pay_table_01")
            else:
                payout_tables = soup.select("table.Payout_Detail_Table")
            if not payout_tables:
                payout_tables = soup.find_all("table")
            def _extract_payout(r):
                tds_p = r.find_all("td")
                payout_td = r.select_one("td.Payout")
                if payout_td:
                    t = payout_td.get_text(strip=True)
                elif len(tds_p) >= 2:
                    t = tds_p[1].get_text(strip=True)
                else:
                    return 0
                t = t.replace(',', '').replace('円', '').replace('¥', '')
                pm = re.search(r'(\d{3,})', t)
                return int(pm.group(1)) if pm else 0
            for table in payout_tables:
                for row in table.find_all("tr"):
                    th = row.find("th")
                    if not th:
                        continue
                    th_text = th.get_text(strip=True)
                    if ('3連複' in th_text or '三連複' in th_text) and trio_payout == 0:
                        trio_payout = _extract_payout(row)
                    elif ('馬連' in th_text) and '馬単' not in th_text and umaren_payout == 0:
                        umaren_payout = _extract_payout(row)
                    elif ('ワイド' in th_text) and wide_payout == 0:
                        # ワイドは複数行あるが最初の1つ（最低配当）を取得
                        wide_payout = _extract_payout(row)
            if results:
                break  # 結果取得成功、次のURLを試す必要なし
        except Exception:
            continue
    payouts = {'trio': trio_payout, 'umaren': umaren_payout, 'wide': wide_payout}
    return results, payouts

# ===== 開催日レース一覧取得 =====
def fetch_race_list(date_str=None):
    """netkeibaから指定日の中央競馬全レースURLを取得。date_str: 'YYYYMMDD' or None(今日)
    返り値: ([{race_id, race_name, course, race_num, time}], error_msg or None)

    netkeibaのレース一覧はAJAX読み込みのため、race_list_sub.htmlを直接取得する。
    HTML構造が変わることがあるため、複数のパース戦略を試行する。
    """
    races = []
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    last_error = None
    try:
        # race_list_sub.html: レース一覧のAJAXサブページ（実データが含まれる）
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
        # リトライ付きHTTP取得（Streamlit Cloud等のネットワーク不安定対策）
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                if attempt == 2:
                    raise
                import time; time.sleep(1)
        if resp is None or resp.status_code != 200:
            last_error = f"HTTP {resp.status_code if resp else 'no response'}"
            return races
        resp.encoding = "UTF-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # --- 戦略1: dl.RaceList_DataList構造（旧レイアウト） ---
        dls = soup.find_all("dl", class_="RaceList_DataList")
        for dl in dls:
            course_name = ""
            header = dl.find("div", class_="RaceList_DataHeader_Top")
            if header:
                header_text = header.get_text(strip=True)
                for cn in COURSE_MAP:
                    if cn in header_text:
                        course_name = cn
                        break
            for li in dl.find_all("li"):
                link = li.find("a", href=re.compile(r'race_id=\d+'))
                if not link:
                    continue
                href = link.get("href", "")
                rid_m = re.search(r'race_id=(\d+)', href)
                if not rid_m:
                    continue
                rid = rid_m.group(1)
                num_div = link.find("div", class_=re.compile(r'Race_Num'))
                race_num = num_div.get_text(strip=True) if num_div else ""
                title_span = link.find("span", class_="ItemTitle")
                race_name = title_span.get_text(strip=True) if title_span else race_num
                time_span = link.find("span", class_="RaceList_Itemtime")
                race_time = time_span.get_text(strip=True) if time_span else ""
                races.append({
                    'race_id': rid,
                    'race_name': f"{race_num} {race_name}",
                    'race_num': race_num,
                    'course': course_name,
                    'time': race_time,
                })

        # --- 戦略2: 全リンクからrace_idを抽出（現行レイアウト対応） ---
        if not races:
            # コース名マッピング: race_idの場コード(5-6桁目)→コース名
            # race_id形式: YYYY JJ CC NNRR (年4桁, 場2桁, 回日2桁, レース番号2桁)
            COURSE_CODE_MAP = {
                '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
                '05': '東京', '06': '中山', '07': '中京', '08': '京都',
                '09': '阪神', '10': '小倉',
            }
            seen = set()
            for link in soup.find_all("a", href=re.compile(r'race_id=\d+')):
                href = link.get("href", "")
                # movie.htmlリンクはスキップ（result.htmlやshutuba.htmlのみ対象）
                if 'movie.html' in href:
                    continue
                rid_m = re.search(r'race_id=(\d+)', href)
                if not rid_m or rid_m.group(1) in seen:
                    continue
                rid = rid_m.group(1)
                seen.add(rid)
                text = link.get_text(strip=True)

                # レース番号を抽出
                num_m = re.search(r'(\d{1,2})R', text)
                race_num = num_m.group(0) if num_m else ''
                if not race_num and len(rid) >= 12:
                    # race_idの末尾2桁からレース番号を推定
                    try:
                        race_num = str(int(rid[-2:])) + 'R'
                    except ValueError:
                        pass

                # コース名をrace_idから推定 (5-6桁目が場コード)
                course_name = ''
                if len(rid) >= 10:
                    course_code = rid[4:6]
                    course_name = COURSE_CODE_MAP.get(course_code, '')

                # レース名: テキストから発走時刻・距離なども含む
                # 例: "1R3歳未勝利10:05 ダ1800m 16頭"
                race_name = text.strip() if text.strip() else race_num

                # 発走時刻を抽出
                time_m = re.search(r'(\d{1,2}:\d{2})', text)
                race_time = time_m.group(1) if time_m else ''

                races.append({
                    'race_id': rid,
                    'race_name': race_name if race_name else race_num,
                    'race_num': race_num,
                    'course': course_name,
                    'time': race_time,
                })

        # レース番号でソート
        if races:
            def sort_key(r):
                nm = re.search(r'(\d+)', r.get('race_num', ''))
                return (r.get('course', ''), int(nm.group(1)) if nm else 99)
            races.sort(key=sort_key)

    except Exception as e:
        last_error = str(e)
        import traceback
        traceback.print_exc()

    # --- フォールバック: db.netkeiba.comから取得 ---
    if not races:
        try:
            db_url = f"https://db.netkeiba.com/race/list/{date_str}/"
            resp2 = requests.get(db_url, headers=HEADERS, timeout=15)
            resp2.encoding = "EUC-JP"
            if resp2.status_code == 200 and len(resp2.content) > 500:
                soup2 = BeautifulSoup(resp2.text, "html.parser")
                seen = set()
                VENUE_NAMES = {
                    '01':'札幌','02':'函館','03':'福島','04':'新潟',
                    '05':'東京','06':'中山','07':'中京','08':'京都',
                    '09':'阪神','10':'小倉',
                }
                for link in soup2.find_all("a", href=re.compile(r'/race/\d{12}/')):
                    href = link.get("href", "")
                    rid_m = re.search(r'/race/(\d{12})/', href)
                    if not rid_m:
                        continue
                    rid = rid_m.group(1)
                    if rid in seen:
                        continue
                    venue_code = int(rid[4:6])
                    if venue_code > 10:
                        continue  # JRA以外をスキップ
                    seen.add(rid)
                    text = link.get_text(strip=True)
                    race_num_int = int(rid[-2:])
                    race_num = f"{race_num_int}R"
                    course_name = VENUE_NAMES.get(rid[4:6], '')
                    race_name = text if text else race_num
                    races.append({
                        'race_id': rid,
                        'race_name': f"{race_num} {race_name}",
                        'race_num': race_num,
                        'course': course_name,
                    })
                races.sort(key=lambda r: (r.get('course', ''), r.get('race_num', '')))
                if races:
                    last_error = None  # フォールバック成功
                    print(f"[INFO] db.netkeiba.comフォールバックで{len(races)}レース取得")
        except Exception as e2:
            print(f"[WARN] db.netkeiba.comフォールバックも失敗: {e2}")
            if last_error is None:
                last_error = str(e2)

    return races, last_error

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
    race_num = ""
    race_grade = ""
    tag = soup.find("div", class_="RaceName")
    if tag and tag.get_text(strip=True):
        race_name = tag.get_text(strip=True)
        # グレードアイコンを検出 (Icon_GradeType1=G1, 2=G2, 3=G3, 5=OP, 15=Listed, etc.)
        grade_span = tag.find("span", class_=re.compile(r'Icon_GradeType'))
        if grade_span:
            grade_cls = " ".join(grade_span.get("class", []))
            if "GradeType1" in grade_cls or "GradeType16" in grade_cls:
                race_grade = "G1"
            elif "GradeType2" in grade_cls or "GradeType17" in grade_cls:
                race_grade = "G2"
            elif "GradeType3" in grade_cls or "GradeType18" in grade_cls:
                race_grade = "G3"
            elif "GradeType5" in grade_cls:
                race_grade = "OP"
            elif "GradeType15" in grade_cls:
                race_grade = "L"
        if not race_grade:
            grade_img = tag.find("img")
            if grade_img:
                alt = grade_img.get("alt", "")
                for g in ["G1", "G2", "G3", "GI", "GII", "GIII"]:
                    if g in alt.upper():
                        race_grade = g[:2].replace("GI", "G1").replace("GI", "G2") if len(g) > 2 else g
                        break
        # グレード文字列をレース名から除去（重複防止）
        race_name = re.sub(r'\s*\(G[I123]+\)\s*', '', race_name).strip()
        race_name = re.sub(r'\s*G[I123]+\s*$', '', race_name).strip()
    else:
        tag = soup.find("title")
        if tag:
            m = re.search(r'(\S+\d+R)', tag.get_text(strip=True))
            if m: race_name = m.group(1)
    # レース番号 (例: "11R")
    num_tag = soup.find("span", class_="RaceNum")
    if num_tag:
        race_num = num_tag.get_text(strip=True)
    if not race_num:
        nm = re.search(r'(\d{1,2})R', race_name)
        if nm:
            race_num = nm.group(0)
    # テキストからグレード検出（フォールバック）
    if not race_grade:
        full_text = soup.get_text()
        gm = re.search(r'[\(（](G[I1][I1I]*|G[123])[\)）]', full_text[:3000])
        if gm:
            g = gm.group(1).replace("GI", "G1").replace("GII", "G2").replace("GIII", "G3")
            if g in ("G1", "G2", "G3"):
                race_grade = g
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
    race_info = dict(distance=distance, surface=surface, condition=condition, course=course_name,
                     grade=race_grade, race_num=race_num)
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
    c, _ = colors.get(rank, ('', 'default'))
    style_name = STYLE_NAMES.get(h.get('脚質', 0), '不明')
    style_css = STYLE_CSS.get(h.get('脚質', 0), 'st-senk')
    pm_html = get_pace_match_html(h.get('脚質', 0), rank_map)
    pct = int(h['スコア'] / max_score * 100) if max_score > 0 else 0
    sex_age = h.get('性別', '牡') + str(h.get('馬齢', 3))
    fr = h.get('複勝率', 0)
    father = h.get('父', '')
    card_cls = f'hcard-{c}' if c else ''
    html = f'<div class="hcard {card_cls}"><div class="hcard-top">'
    rank_cls = f'hrank-{c}' if c else ''
    html += f'<div class="hrank {rank_cls}">{rank}</div>'
    html += f'<div style="flex:1"><div class="hname">{h["馬名"]}</div>'
    html += f'<div class="hjockey">🏇 {h["騎手名"]} ｜ {sex_age} ｜ {h["斤量"]}kg</div></div>'
    score_cls = f'hscore-{c}' if c else ''
    html += f'<div class="hscore {score_cls}">{h["スコア"]:.3f}</div></div>'
    html += '<div class="sgrid">'
    html += f'<div class="sitem"><div class="slbl">前走</div><div class="sval {finish_cls(h["前走着順"])}">{h["前走着順"]}着</div></div>'
    html += f'<div class="sitem"><div class="slbl">脚質</div><div class="sval"><span class="stag {style_css}">{style_name}</span>{pm_html}</div></div>'
    html += f'<div class="sitem"><div class="slbl">間隔</div><div class="sval {interval_cls(h.get("前走間隔",30))}">{interval_text(h.get("前走間隔",30))}</div></div>'
    html += f'<div class="sitem"><div class="slbl">体重</div><div class="sval">{weight_html(h.get("場体重増減",0))}</div></div>'
    html += f'<div class="sitem"><div class="slbl">上がり</div><div class="sval">{h.get("上がり3F",35.5):.1f}</div></div>'
    odds = h.get('単勝オッズ', 0.0)
    if odds > 0:
        odds_color = '#ff4060' if odds <= 3.0 else ('#f0c040' if odds <= 10.0 else ('#b0b8c8' if odds <= 30.0 else '#6a6a80'))
        html += f'<div class="sitem"><div class="slbl">単勝</div><div class="sval" style="color:{odds_color} !important">{odds:.1f}</div></div>'
    html += '</div>'
    # 調教評価
    train_label = h.get('調教ラベル', '')
    train_eval = h.get('調教評価', '')
    train_rank = h.get('調教ランク', '')
    if train_label:
        if train_rank == 'A':
            train_color = '#4ade80'  # green
            train_border = 'rgba(74,222,128,0.4)'
        elif train_rank == 'B':
            train_color = '#f0c040'  # yellow
            train_border = 'rgba(240,192,64,0.3)'
        else:
            train_color = '#ff6080'  # red
            train_border = 'rgba(255,96,128,0.3)'
        train_text = train_label
        if train_eval:
            train_text += f' {train_eval}'
        html += f'<div style="margin:2px 0 4px 0;"><span style="font-size:0.82em;padding:2px 8px;border-radius:4px;border:1px solid {train_border};color:{train_color} !important;background:rgba(0,0,0,0.2)">🏋️ {train_text}</span></div>'
    # 展開適性バッジ
    pace_adv_data = st.session_state.get('pred_pace_adv', {})
    horse_style = int(h.get('脚質', 0))
    if horse_style > 0 and pace_adv_data:
        padv = pace_adv_data.get(horse_style, 0)
        pace_label = st.session_state.get('pred_pace', 'middle')
        pace_disp = {'high': 'H', 'middle': 'M', 'slow': 'S'}.get(pace_label, 'M')
        if padv > 0.015:
            p_color, p_border, p_text = '#00e87b', 'rgba(0,232,123,0.4)', f'展開◎ ({pace_disp}ペース向き)'
        elif padv > 0:
            p_color, p_border, p_text = '#f0c040', 'rgba(240,192,64,0.3)', f'展開○ ({pace_disp}ペース)'
        elif padv > -0.015:
            p_color, p_border, p_text = '#b0b8c8', 'rgba(176,184,200,0.3)', f'展開△ ({pace_disp}ペース)'
        else:
            p_color, p_border, p_text = '#ff4060', 'rgba(255,64,96,0.3)', f'展開× ({pace_disp}ペース不利)'
        html += f'<div style="margin:2px 0 4px 0;"><span style="font-size:0.82em;padding:2px 8px;border-radius:4px;border:1px solid {p_border};color:{p_color} !important;background:rgba(0,0,0,0.2)">🔄 {p_text}</span></div>'
    # 馬場適性警告
    if h.get('馬場警告'):
        html += '<div style="margin:2px 0 4px 0;"><span style="font-size:0.82em;padding:2px 8px;border-radius:4px;border:1px solid rgba(255,64,96,0.4);color:#ff4060 !important;background:rgba(60,0,0,0.3)">⚠️ 馬場適性注意 — 重馬場苦手</span></div>'
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
    bar_cls = f'sbar-{c}' if c else 'sbar-b'
    html += f'<div class="sbar-w"><div class="sbar {bar_cls}" style="width:{pct}%"></div></div></div>'
    return html

def calc_expected_values(df_sorted, realtime_odds):
    """各買い目の期待値を計算。AI順位上位の的中確率 × オッズで期待値を出す。"""
    ev_list = []
    if not realtime_odds:
        return ev_list
    n = len(df_sorted)
    scores = df_sorted['スコア'].values
    score_sum = scores.sum()
    if score_sum <= 0:
        return ev_list
    # 各馬の3着以内確率をスコアから推定
    probs = scores / score_sum
    top6 = df_sorted.head(6)
    # 単勝期待値
    for _, h in top6.iterrows():
        umaban = int(h['馬番'])
        odds = h.get('単勝オッズ', 0)
        if odds <= 0:
            continue
        rank = int(h['AI順位'])
        # 1着確率 = スコア比率 × 補正
        win_prob = probs[rank - 1] * 1.5  # 上位ほど1着確率は高め
        win_prob = min(win_prob, 0.6)
        ev = win_prob * odds
        ev_list.append({
            'type': '単勝', 'horses': f"{umaban} {h['馬名'][:5]}",
            'odds': odds, 'prob': win_prob, 'ev': ev,
            'umaban': [umaban]
        })
    # ワイド期待値（TOP3の組み合わせ）
    top3 = df_sorted.head(3)
    for i, j in combinations(range(len(top3)), 2):
        h1, h2 = top3.iloc[i], top3.iloc[j]
        r1, r2 = int(h1['AI順位']), int(h2['AI順位'])
        # 両方3着以内の確率
        p1 = min(probs[r1-1] * 3.0, 0.85)
        p2 = min(probs[r2-1] * 3.0, 0.85)
        wide_prob = p1 * p2 * 0.8  # 独立ではないので補正
        ev_list.append({
            'type': 'ワイド',
            'horses': f"{int(h1['馬番'])}-{int(h2['馬番'])}",
            'odds': 0, 'prob': wide_prob, 'ev': 0,
            'umaban': [int(h1['馬番']), int(h2['馬番'])]
        })
    # 三連複期待値（TOP1軸 ― TOP2,3 ― TOP2-6）
    top1 = df_sorted.iloc[0]
    for i in range(1, min(3, len(df_sorted))):
        for j in range(i+1, min(6, len(df_sorted))):
            h2 = df_sorted.iloc[i]
            h3 = df_sorted.iloc[j]
            r1 = int(top1['AI順位'])
            r2 = int(h2['AI順位'])
            r3 = int(h3['AI順位'])
            p1 = min(probs[r1-1] * 3.0, 0.85)
            p2 = min(probs[r2-1] * 3.0, 0.85)
            p3 = min(probs[r3-1] * 3.0, 0.85)
            trio_prob = p1 * p2 * p3 * 0.6
            ev_list.append({
                'type': '三連複',
                'horses': f"{int(top1['馬番'])}-{int(h2['馬番'])}-{int(h3['馬番'])}",
                'odds': 0, 'prob': trio_prob, 'ev': 0,
                'umaban': sorted([int(top1['馬番']), int(h2['馬番']), int(h3['馬番'])])
            })
    return ev_list

def fetch_pair_odds(race_id, bet_type='wide', is_nar=False):
    """ワイド(type=5)または馬連(type=4)のオッズを取得。{(n1,n2): odds} を返す"""
    pair_odds = {}
    type_code = '5' if bet_type == 'wide' else '4'
    try:
        if is_nar:
            url = f"https://nar.netkeiba.com/api/api_get_nar_odds.html?race_id={race_id}&type={type_code}"
        else:
            url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={type_code}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 2:
                continue
            combo_text = tds[0].get_text(strip=True)
            nums = re.findall(r'\d+', combo_text)
            if len(nums) == 2:
                key = tuple(sorted(int(n) for n in nums))
                for td in tds[1:]:
                    try:
                        txt = td.get_text(strip=True).replace(',', '')
                        # ワイドは「2.3 - 5.1」形式（下限-上限）の場合あり
                        if '-' in txt and bet_type == 'wide':
                            parts = txt.split('-')
                            v = float(parts[0].strip())
                        else:
                            v = float(txt)
                        if v >= 1.0:
                            pair_odds[key] = v
                            break
                    except:
                        continue
    except Exception:
        pass
    return pair_odds


def fetch_trio_odds(race_id, is_nar=False):
    """三連複オッズを取得。{(n1,n2,n3): odds} を返す"""
    trio_odds = {}
    try:
        if is_nar:
            url = f"https://nar.netkeiba.com/api/api_get_nar_odds.html?race_id={race_id}&type=6"
        else:
            url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=6"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 2:
                continue
            # 組み合わせとオッズを探す
            combo_text = tds[0].get_text(strip=True)
            nums = re.findall(r'\d+', combo_text)
            if len(nums) == 3:
                key = tuple(sorted(int(n) for n in nums))
                for td in tds[1:]:
                    try:
                        v = float(td.get_text(strip=True).replace(',', ''))
                        if v >= 1.0:
                            trio_odds[key] = v
                            break
                    except:
                        continue
    except Exception:
        pass
    return trio_odds

def render_ev_section(ev_list):
    """期待値セクションをHTML描画"""
    if not ev_list:
        return '<div class="ev-card"><div class="ev-row"><span class="ev-lbl">オッズ未取得のため期待値計算不可</span></div></div>'
    # 期待値でソート（EVが計算できたもの優先、高い順）
    ev_with_val = [e for e in ev_list if e['ev'] > 0]
    ev_with_val.sort(key=lambda x: x['ev'], reverse=True)
    if not ev_with_val:
        return '<div class="ev-card"><div class="ev-row"><span class="ev-lbl">期待値データなし</span></div></div>'
    html = '<div class="ev-card">'
    for e in ev_with_val[:10]:
        ev = e['ev']
        if ev >= 1.5:
            ev_cls = 'ev-hot'
            ev_icon = '&#128293;'
        elif ev >= 1.0:
            ev_cls = 'ev-warm'
            ev_icon = '&#9733;'
        else:
            ev_cls = 'ev-cold'
            ev_icon = ''
        prob_pct = e['prob'] * 100
        html += f'<div class="ev-row">'
        html += f'<span class="ev-lbl">{e["type"]} {e["horses"]}</span>'
        html += f'<span class="ev-lbl">({prob_pct:.0f}% x {e["odds"]:.1f})</span>'
        html += f'<span class="ev-val {ev_cls}">{ev_icon} EV {ev:.2f}</span>'
        html += '</div>'
    html += '</div>'
    # 期待値1.0以上のサマリー
    hot_bets = [e for e in ev_with_val if e['ev'] >= 1.0]
    if hot_bets:
        html += f'<div style="margin-top:8px;padding:10px;background:#1a3a1a;border:1px solid #2ecc40;border-radius:8px;color:#2ecc40 !important;text-align:center;font-weight:bold;">'
        html += f'&#128176; 期待値1.0超の買い目: {len(hot_bets)}点</div>'
    return html

def _render_stats_block(s, label=""):
    """統計ブロック1つ分のHTMLを生成"""
    sr = s['settled_races']
    html = ''
    if label:
        html += f'<div style="font-size:0.75em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:6px;">{label}</div>'
    html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px;">'
    html += f'<div class="dash-item"><div class="dash-num" style="font-size:1.3em">{s["total_races"]}</div><div class="dash-lbl">予想R</div></div>'
    if sr > 0:
        hc = '#2ecc40' if s['hit_rate'] >= 0.15 else ('#f0c040' if s['hit_rate'] >= 0.08 else '#b0b8c8')
        html += f'<div class="dash-item"><div class="dash-num" style="font-size:1.3em">{s["hit_count"]}<span style="font-size:0.45em;color:#6a6a80 !important">/{sr}</span></div><div class="dash-lbl">的中</div></div>'
        html += f'<div class="dash-item"><div class="dash-num" style="font-size:1.3em;color:{hc} !important">{s["hit_rate"]*100:.1f}%</div><div class="dash-lbl">的中率</div></div>'
    else:
        html += '<div class="dash-item"><div class="dash-num" style="font-size:1.3em">-</div><div class="dash-lbl">的中</div></div>'
        html += '<div class="dash-item"><div class="dash-num" style="font-size:1.3em">-</div><div class="dash-lbl">的中率</div></div>'
    inv, pay, profit, roi = s['total_investment'], s['total_payout'], s['profit'], s['roi']
    if sr > 0:
        pc = '#2ecc40' if profit >= 0 else '#ff4060'
        ps = '+' if profit >= 0 else ''
        rc = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 70 else '#ff4060')
        html += f'<div class="dash-item"><div class="dash-num" style="font-size:1em">&yen;{inv:,}</div><div class="dash-lbl">投資</div></div>'
        html += f'<div class="dash-item"><div class="dash-num" style="font-size:1em">&yen;{pay:,}</div><div class="dash-lbl">払戻</div></div>'
        html += f'<div class="dash-item"><div class="dash-num" style="font-size:1em;color:{pc} !important">&yen;{ps}{profit:,}</div><div class="dash-lbl">収支</div></div>'
    else:
        html += '<div class="dash-item"><div class="dash-num" style="font-size:1em">-</div><div class="dash-lbl">投資</div></div>'
        html += '<div class="dash-item"><div class="dash-num" style="font-size:1em">-</div><div class="dash-lbl">払戻</div></div>'
        html += '<div class="dash-item"><div class="dash-num" style="font-size:1em">-</div><div class="dash-lbl">収支</div></div>'
    html += '</div>'
    # 回収率バー
    if sr > 0:
        bw = min(roi, 200) / 2
        bc = '#2ecc40' if roi >= 100 else ('#f0c040' if roi >= 70 else '#ff4060')
        rc = bc
        html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
        html += f'<span style="font-size:0.7em;color:#6a6a80 !important;width:36px;">回収率</span>'
        html += f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:5px;height:20px;position:relative;overflow:hidden;">'
        html += f'<div style="width:{bw}%;height:100%;background:{bc};border-radius:5px;"></div>'
        html += f'<div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:rgba(255,255,255,0.2);"></div>'
        html += f'</div>'
        html += f'<span style="font-family:Oswald;font-weight:700;font-size:1em;color:{rc} !important;width:55px;text-align:right;">{roi:.1f}%</span>'
        html += '</div>'
    return html

def render_dashboard():
    """成績ダッシュボードをHTML描画（全体/中央/地方）"""
    stats = get_dashboard_stats()
    if stats['all']['total_races'] == 0:
        return ''
    html = '<div class="ev-card">'
    # 全体
    html += _render_stats_block(stats['all'], '&#127942; ALL')
    # 中央成績
    has_jra = stats['jra']['total_races'] > 0
    if has_jra:
        html += '<div style="border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;"></div>'
        html += _render_stats_block(stats['jra'], '&#127807; JRA CENTRAL')
    # 直近レース結果
    recent = stats['recent']
    if recent:
        html += '<div style="border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;"></div>'
        html += '<div style="font-size:0.75em;color:#6a6a80 !important;letter-spacing:2px;margin-bottom:6px;">RECENT RACES</div>'
        html += '<div style="font-size:0.8em;">'
        BET_SHORT = {'trio': '三', 'umaren': '連', 'wide': 'W'}
        for r in recent[:8]:
            name = r.get('race_name', '')
            date = r.get('predicted_at', '')
            hit_trio = r.get('hit_trio')
            payout = r.get('payout', 0) or 0
            db_bet_type = r.get('bet_type', 'trio') or 'trio'
            bet_short = BET_SHORT.get(db_bet_type, '三')
            date_short = date[:10] if date else ''
            tag = '<span style="font-size:0.7em;padding:1px 4px;border-radius:3px;background:#1a3a1a;color:#2ecc40 !important;margin-right:4px;">JRA</span>'
            bet_tag = f'<span style="font-size:0.65em;padding:1px 3px;border-radius:2px;background:#2a2a3a;color:#aab !important;margin-right:3px;">{bet_short}</span>'
            if hit_trio is not None:
                if hit_trio == 1:
                    icon = '&#9989;'
                    pt = f'<span style="color:#2ecc40 !important;font-family:Oswald;">+&yen;{payout - INVESTMENT_PER_RACE:,}</span>'
                else:
                    icon = '&#10060;'
                    pt = f'<span style="color:#ff4060 !important;font-family:Oswald;">-&yen;{INVESTMENT_PER_RACE}</span>'
                html += f'<div class="ev-row"><span class="ev-lbl">{tag}{bet_tag}{date_short} {name[:8]}</span><span>{pt}</span><span>{icon}</span></div>'
            else:
                html += f'<div class="ev-row"><span class="ev-lbl">{tag}{bet_tag}{date_short} {name[:8]}</span><span style="color:#6a6a80 !important;font-size:0.85em;">&#8987;</span></div>'
        html += '</div>'
    html += '</div>'
    return html

def render_track_record_race_list(races):
    """TRACK RECORD: レース一覧をst.expanderで表示（予測詳細付き）"""
    BET_LABELS = {'trio': '三連複', 'umaren': '馬連', 'wide': 'ワイド'}
    BET_KEYS = {'trio': 'trio_bets', 'wide': 'wide_bets', 'umaren': 'umaren_bets'}
    if not races:
        st.info("該当するレースがありません。")
        return
    for r in races:
        race_name = r.get('race_name', '不明')
        race_id = r.get('race_id', '')
        hit = r.get('hit_trio')
        payout = r.get('payout', 0) or 0
        bet_type = r.get('bet_type', 'trio') or 'trio'
        bet_cond = r.get('bet_condition', '?') or '?'
        distance = r.get('distance', 0) or 0
        surface = r.get('surface', '') or ''
        track_cond = r.get('track_condition', '') or ''
        num_horses = r.get('num_horses', 0) or 0
        if hit == 1:
            status_icon = "✅"
            status_text = f"+¥{payout - INVESTMENT_PER_RACE:,}"
        elif hit == 0:
            status_icon = "❌"
            status_text = f"-¥{INVESTMENT_PER_RACE}"
        else:
            status_icon = "⏳"
            status_text = "未確定"
        label = f"{status_icon} {race_name}　[{bet_cond}] {BET_LABELS.get(bet_type, bet_type)} {status_text}"
        with st.expander(label, expanded=False):
            # レース情報
            info_parts = []
            if surface:
                info_parts.append(surface)
            if distance:
                info_parts.append(f"{distance}m")
            if track_cond:
                info_parts.append(track_cond)
            if num_horses:
                info_parts.append(f"{num_horses}頭")
            cond_profile = CONDITION_PROFILES.get(bet_cond, {})
            cond_desc = cond_profile.get('desc', '')
            st.markdown(f"**レース情報**: {' / '.join(info_parts)}")
            st.markdown(f"**条件 {bet_cond}**: {cond_desc}")
            st.markdown(f"**推奨買い目**: {BET_LABELS.get(bet_type, bet_type)}")
            # 買い目
            bets_raw = r.get(BET_KEYS.get(bet_type, 'trio_bets'), '[]')
            try:
                bets = json.loads(bets_raw) if bets_raw else []
            except Exception:
                bets = []
            if bets:
                inv = len(bets) * 100
                # フォーメーション構造表示
                if len(bets) >= 3 and len(bets[0]) == 3:
                    all_nums = set()
                    for b in bets:
                        all_nums.update(b)
                    all_nums = sorted(all_nums)
                    # 軸: 全買い目に含まれる馬番
                    axis = sorted([n for n in all_nums if all(n in b for b in bets)])
                    if not axis:
                        axis = sorted(set(bets[0]) & set(bets[1])) if len(bets) > 1 else [bets[0][0]]
                    # 軸候補から馬名を取得
                    horses_pred = get_predictions_for_race(race_id)
                    name_map = {h['horse_num']: h['horse_name'][:5] for h in horses_pred} if horses_pred else {}
                    col2 = sorted(set(n for b in bets for n in b if n not in axis and any(n in b2 for b2 in bets if set(b2) != set(b))))
                    col3 = sorted(all_nums - set(axis))
                    axis_txt = ', '.join(f'{n}番 {name_map.get(n, "")}' for n in axis)
                    col3_txt = ', '.join(str(n) for n in col3)
                    struct_html = f'<div style="font-size:0.82em;color:#8890a0 !important;margin:2px 0 6px;padding:0 4px;">'
                    struct_html += f'1列目(軸): <span style="color:#f0c040 !important;font-weight:bold;">{axis_txt}</span>'
                    struct_html += f' / 相手: <span style="font-family:Oswald;">{col3_txt}</span></div>'
                    st.markdown(f"**買い目** ({len(bets)}点 × ¥100 = ¥{inv:,})")
                    st.markdown(struct_html, unsafe_allow_html=True)
                elif len(bets[0]) == 2:
                    # 馬連/ワイド: 軸-相手（金額表示付き）
                    axis_num = bets[0][0] if len(bets) > 1 and bets[0][0] == bets[1][0] else min(bets[0])
                    horses_pred = get_predictions_for_race(race_id)
                    name_map = {h['horse_num']: h['horse_name'][:5] for h in horses_pred} if horses_pred else {}
                    bet_label = BET_LABELS.get(bet_type, bet_type)
                    umaren_amts = [400, 300]  # TOP2=400, TOP3=300
                    bet_details = []
                    for bi, b in enumerate(bets):
                        amt = umaren_amts[bi] if bi < len(umaren_amts) else 100
                        bet_details.append(f'{bet_label} {b[0]}-{b[1]}: {amt}円')
                    struct_html = f'<div style="font-size:0.82em;color:#8890a0 !important;margin:2px 0 6px;padding:0 4px;">'
                    struct_html += f'軸: <span style="color:#f0c040 !important;font-weight:bold;">{axis_num}番 {name_map.get(axis_num, "")}</span>'
                    struct_html += f' / ' + ' / '.join(f'<span style="font-family:Oswald;">{d}</span>' for d in bet_details)
                    struct_html += '</div>'
                    st.markdown(f"**買い目** (合計 ¥700)")
                    st.markdown(struct_html, unsafe_allow_html=True)
                else:
                    st.markdown(f"**買い目** ({len(bets)}点 × ¥100 = ¥{inv:,})")
                bet_strs = ['  '.join(str(n) for n in sorted(b)) for b in bets]
                bets_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 8px 0;">'
                for bs in bet_strs:
                    bets_html += f'<span style="background:#1a2a3a;padding:3px 8px;border-radius:4px;font-family:Oswald;font-size:0.85em;color:#b0d0f0 !important;">{bs}</span>'
                bets_html += '</div>'
                st.markdown(bets_html, unsafe_allow_html=True)
            # AI予測上位馬
            horses = get_predictions_for_race(race_id)
            if horses:
                st.markdown("**AI予測スコア**")
                top_horses = horses[:6]
                max_score = top_horses[0]['ai_score'] if top_horses else 1
                horse_html = '<div style="font-size:0.85em;">'
                for h in top_horses:
                    rank = h['ai_rank']
                    name = h['horse_name']
                    score = h['ai_score']
                    num = h['horse_num']
                    finish = h.get('actual_finish')
                    bar_w = score / max_score * 100 if max_score > 0 else 0
                    finish_str = ''
                    finish_color = ''
                    if finish:
                        if finish <= 3:
                            finish_str = f' → <span style="color:#2ecc40 !important;font-weight:bold;">{finish}着</span>'
                        else:
                            finish_str = f' → {finish}着'
                    rank_colors = {1: '#ffd700', 2: '#c0c0c0', 3: '#cd7f32'}
                    rank_c = rank_colors.get(rank, '#6a6a80')
                    horse_html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
                    horse_html += f'<span style="min-width:20px;color:{rank_c} !important;font-weight:bold;font-family:Oswald;">{rank}</span>'
                    horse_html += f'<span style="min-width:24px;color:#8a8aa0 !important;font-family:Oswald;">{num}番</span>'
                    horse_html += f'<span style="min-width:80px;">{name}</span>'
                    horse_html += f'<div style="flex:1;height:10px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">'
                    horse_html += f'<div style="width:{bar_w:.0f}%;height:100%;background:linear-gradient(90deg,#2898d8,#00e87b);border-radius:3px;"></div></div>'
                    horse_html += f'<span style="font-family:Oswald;font-size:0.85em;min-width:45px;color:#b0b8c8 !important;">{score:.3f}</span>'
                    horse_html += f'<span style="font-size:0.82em;">{finish_str}</span>'
                    horse_html += '</div>'
                horse_html += '</div>'
                st.markdown(horse_html, unsafe_allow_html=True)
            # 結果表示
            if hit == 1:
                hit_combo = r.get('hit_combo', '')
                combo_str = ''
                if hit_combo:
                    try:
                        combo = json.loads(hit_combo)
                        combo_str = f" ({', '.join(str(n) for n in combo)})"
                    except Exception:
                        pass
                st.success(f"的中！{combo_str} 配当: ¥{payout:,}（収支: +¥{payout - INVESTMENT_PER_RACE:,}）")
            elif hit == 0:
                st.error(f"不的中（投資: ¥{INVESTMENT_PER_RACE}）")

def render_buy_section(df, race_info, rank_map, cond_key=None, cond_profile=None, pair_odds=None):
    """条件別バックテスト結果に基づき、最適な1種類の買い目のみ表示。
    pair_odds: {(n1,n2): odds} ワイドまたは馬連のペアオッズ
    """
    if cond_profile is None:
        is_nar = False
        cond_key, cond_profile = classify_race_condition(race_info, len(df), is_nar=is_nar)

    bet_type = cond_profile['bet_type']
    if bet_type == 'none' or not cond_profile['recommended']:
        return ''

    sorted_df = df.sort_values('スコア', ascending=False).reset_index(drop=True)
    if len(sorted_df) < 3:
        return ''
    top = sorted_df.head(6)
    t1 = top.iloc[0]; t2 = top.iloc[1]; t3 = top.iloc[2]

    def hn(h):
        n = int(h['馬番'])
        return str(n) if n > 0 else h['馬名'][:3]

    roi = cond_profile['roi']
    hit_rate = cond_profile['hit_rate']
    roi_c = '#2ecc40' if roi >= 100 else '#f0c040'
    wf_n = cond_profile.get('wf_n', cond_profile.get('leakfree_n', 0))
    is_small_sample = wf_n > 0 and wf_n < 30

    # WFバックテスト検証バッジ
    if wf_n >= 30:
        lf_badge = f'<span style="font-size:0.72em;padding:2px 6px;background:#1a3a2a;color:#2ecc40 !important;border-radius:4px;margin-left:6px;">WF検証済 N={wf_n:,}</span>'
    elif wf_n > 0:
        lf_badge = f'<span style="font-size:0.72em;padding:2px 6px;background:#3a2a00;color:#f0c040 !important;border-radius:4px;margin-left:6px;">サンプル少 N={wf_n}</span>'
    else:
        lf_badge = ''

    html = '<div class="buy-card buy-honmei">'
    html += '<div class="buy-header">'

    if bet_type == 'trio':
        bets = generate_trio_bets(sorted_df)
        html += f'<span class="buy-type bt-hon">&#127942; 三連複 7点</span>'
        html += f'<span class="buy-conf" style="color:{roi_c} !important;">ROI {roi:.1f}% / HIT {hit_rate:.1f}%</span>{lf_badge}</div>'
        html += f'<div style="font-size:0.82em;color:#6a6a80 !important;margin:4px 0 8px;padding:0 12px;">{cond_profile["label"]} : {cond_profile["desc"]}</div>'
        html += f'<div style="padding:4px 12px;margin-bottom:4px;">'
        html += f'<div style="font-size:0.85em;color:#b0b8c8 !important;margin-bottom:6px;">1列目(軸): <span style="font-family:Oswald;color:#f0c040 !important;">{hn(t1)}</span> {t1["馬名"][:5]}</div>'
        col2 = sorted([int(t2['馬番']), int(t3['馬番'])])
        col2_txt = ', '.join(f'<span style="font-family:Oswald;">{n}</span>' for n in col2)
        html += f'<div style="font-size:0.85em;color:#b0b8c8 !important;margin-bottom:6px;">2列目: {col2_txt}</div>'
        himo = sorted([int(top.iloc[i]['馬番']) for i in range(1, min(6, len(top)))])
        himo_txt = ', '.join(str(n) for n in himo)
        html += f'<div style="font-size:0.85em;color:#b0b8c8 !important;margin-bottom:8px;">3列目: <span style="font-family:Oswald;">{himo_txt}</span></div>'
        html += '</div>'
        html += '<div style="padding:0 12px 8px;display:flex;flex-wrap:wrap;gap:6px;">'
        for b in bets:
            html += f'<span style="font-family:Oswald;font-size:0.85em;padding:3px 8px;background:rgba(255,255,255,0.06);border-radius:4px;color:#b0b8c8 !important;">{b[0]}-{b[1]}-{b[2]}</span>'
        html += '</div>'
        html += f'<div style="padding:4px 12px 12px;font-family:Oswald;font-size:0.85em;color:#6a6a80 !important;">{len(bets)}点 &times; 100円 = 700円</div>'

    elif bet_type in ('wide', 'umaren'):
        if bet_type == 'wide':
            bets = generate_wide_bets(sorted_df)
            type_label = 'ワイド 1軸2流し'
        else:
            bets = generate_umaren_bets(sorted_df)
            type_label = '馬連 1軸2流し'

        # 予測順位連動投資額振り分け: TOP2に400円、TOP3に300円
        n1 = int(t1['馬番']); n2 = int(t2['馬番']); n3 = int(t3['馬番'])
        key1 = tuple(sorted([n1, n2]))
        key2 = tuple(sorted([n1, n3]))
        amt1, amt2 = 400, 300  # TOP2=400円, TOP3=300円（上位予測の相手ほど高額）
        odds1 = (pair_odds or {}).get(key1, 0)
        odds2 = (pair_odds or {}).get(key2, 0)

        if odds1 > 0 and odds2 > 0:
            exp1 = int(odds1 * amt1)
            exp2 = int(odds2 * amt2)
            odds1_txt = f'<span style="font-family:Oswald;color:#00d4ff !important;">{odds1:.1f}倍</span>'
            odds2_txt = f'<span style="font-family:Oswald;color:#00d4ff !important;">{odds2:.1f}倍</span>'
            exp1_txt = f'<span style="font-family:Oswald;font-size:0.82em;color:#2ecc40 !important;">期待払戻&yen;{exp1:,}</span>'
            exp2_txt = f'<span style="font-family:Oswald;font-size:0.82em;color:#2ecc40 !important;">期待払戻&yen;{exp2:,}</span>'
        else:
            odds1_txt = '<span style="font-size:0.82em;color:#6a6a80 !important;">オッズ未取得</span>'
            odds2_txt = '<span style="font-size:0.82em;color:#6a6a80 !important;">オッズ未取得</span>'
            exp1_txt = ''
            exp2_txt = ''

        html += f'<span class="buy-type bt-hon">&#127942; {type_label}</span>'
        html += f'<span class="buy-conf" style="color:{roi_c} !important;">ROI {roi:.1f}% / HIT {hit_rate:.1f}%</span>{lf_badge}</div>'
        html += f'<div style="font-size:0.82em;color:#6a6a80 !important;margin:4px 0 8px;padding:0 12px;">{cond_profile["label"]} : {cond_profile["desc"]}</div>'
        html += '<div style="padding:4px 12px 8px;">'
        # Bet 1
        html += f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;padding:8px 10px;background:rgba(255,255,255,0.03);border-radius:8px;">'
        html += f'<span style="font-family:Oswald;font-size:1.1em;color:#f0c040 !important;min-width:20px;">{hn(t1)}</span>'
        html += f'<span style="color:#b0b8c8 !important;">{t1["馬名"][:5]}</span>'
        html += f'<span style="color:#6a6a80 !important;">―</span>'
        html += f'<span style="font-family:Oswald;font-size:1.1em;">{hn(t2)}</span>'
        html += f'<span style="color:#b0b8c8 !important;">{t2["馬名"][:5]}</span>'
        html += f'<span style="margin-left:auto;">{odds1_txt}</span>'
        html += f'<span style="font-family:Oswald;font-weight:700;color:#f0c040 !important;">{amt1}円</span>'
        html += f'</div>'
        if exp1_txt:
            html += f'<div style="text-align:right;margin:-4px 10px 6px 0;">{exp1_txt}</div>'
        # Bet 2
        html += f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;padding:8px 10px;background:rgba(255,255,255,0.03);border-radius:8px;">'
        html += f'<span style="font-family:Oswald;font-size:1.1em;color:#f0c040 !important;min-width:20px;">{hn(t1)}</span>'
        html += f'<span style="color:#b0b8c8 !important;">{t1["馬名"][:5]}</span>'
        html += f'<span style="color:#6a6a80 !important;">―</span>'
        html += f'<span style="font-family:Oswald;font-size:1.1em;">{hn(t3)}</span>'
        html += f'<span style="color:#b0b8c8 !important;">{t3["馬名"][:5]}</span>'
        html += f'<span style="margin-left:auto;">{odds2_txt}</span>'
        html += f'<span style="font-family:Oswald;font-weight:700;color:#f0c040 !important;">{amt2}円</span>'
        html += f'</div>'
        if exp2_txt:
            html += f'<div style="text-align:right;margin:-4px 10px 6px 0;">{exp2_txt}</div>'
        html += '</div>'
        html += f'<div style="padding:4px 12px 12px;font-family:Oswald;font-size:0.85em;color:#6a6a80 !important;">TOTAL: {amt1} + {amt2} = 700円</div>'

    # TOP1 note
    s1_style = STYLE_NAMES.get(int(t1.get('脚質', 0)), '不明')
    s1_match = rank_map.get(int(t1.get('脚質', 0)), ('△','fair',''))[0]
    t1_odds = t1.get('単勝オッズ', 0)
    odds_note = f' 単勝{t1_odds:.1f}倍' if t1_odds > 0 else ''
    html += f'<div class="buy-note">軸 {t1["馬名"]} ({s1_style}/{s1_match}) 前走{int(t1["前走着順"])}着{odds_note}</div>'
    html += '</div>'
    return html

def render_table(df, rank_map):
    sorted_df = df.sort_values('AI順位')
    html = '<table class="htable"><tr><th>#</th><th>馬名</th><th>騎手</th><th>脚質</th><th>前走</th><th>単勝</th><th>間隔</th><th>体重</th><th>⏱ Best</th><th>SCORE</th></tr>'
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
        odds = h.get('単勝オッズ', 0.0)
        if odds > 0:
            odds_color = '#ff4060' if odds <= 3.0 else ('#f0c040' if odds <= 10.0 else ('#b0b8c8' if odds <= 30.0 else '#6a6a80'))
            html += f'<td style="font-family:Oswald;color:{odds_color} !important">{odds:.1f}</td>'
        else:
            html += '<td>-</td>'
        html += f'<td class="{interval_cls(h.get("前走間隔",30))}">{interval_text(h.get("前走間隔",30))}</td>'
        html += f'<td>{weight_html(int(h.get("場体重増減",0)))}</td>'
        html += f'<td style="font-family:Oswald;font-size:0.82em">{bt_display}</td>'
        html += f'<td class="tscore" style="color:{rc}">{h["スコア"]:.3f}</td></tr>'
    html += '</table>'
    return html

# ===== MAIN =====
st.markdown('<div class="site-header"><div class="site-logo">KEIBA AI</div><div class="site-sub">PREDICTION SYSTEM</div></div>', unsafe_allow_html=True)

# Model version badge (default, updated after race type detection)
badge_css = f'badge-{model_version}'
auc_text = f' AUC {model_auc:.4f}' if model_auc > 0 else ''
leak_text = ' LEAK-FREE' if model_leak_free else ''
# v9モデル利用可能状況
v9_avail = _v9_models.get('central') is not None
model_badge_placeholder = st.empty()
if v9_avail:
    _v9c_auc = _v9_models['central'].get('auc', 0)
    _v9c_ver = _v9_models['central'].get('version', 'v9').upper()
    model_badge_placeholder.markdown(f'<div style="text-align:center;margin-top:-12px;margin-bottom:12px"><span class="model-badge badge-central">CENTRAL {_v9c_ver} AUC {_v9c_auc:.4f}</span> <span class="model-badge badge-nar">NAR専用 A,B,E推奨</span> <span class="model-badge badge-v9">AUTO SELECT</span> <span class="model-badge" style="background:linear-gradient(135deg,#1a3a2a,#0a2a1a);border:1px solid #2ecc40;color:#2ecc40 !important;">LEAK-FREE verified</span></div>', unsafe_allow_html=True)
else:
    model_badge_placeholder.markdown(f'<div style="text-align:center;margin-top:-12px;margin-bottom:12px"><span class="model-badge {badge_css}">MODEL {model_version.upper()}{auc_text}{leak_text}</span></div>', unsafe_allow_html=True)

# ===== 起動時自動チェック =====
sys_checks = run_system_checks()
all_ok = all(c[1] for c in sys_checks)
if all_ok:
    st.markdown('<div class="sys-ok"><span style="color:#2ecc40 !important;font-weight:bold;">&#9989; 全システム正常</span></div>', unsafe_allow_html=True)
else:
    warn_html = '<div class="sys-warn">'
    for name, ok, detail in sys_checks:
        if not ok:
            warn_html += f'<div style="color:#ff4060 !important;font-weight:bold;">&#9888;&#65039; {name}に問題あり — {detail}</div>'
    warn_html += '</div>'
    st.markdown(warn_html, unsafe_allow_html=True)

url_input = st.text_input("netkeibaの出馬表URLを入力（中央競馬専用）")

if st.button("🔍 予想する") and url_input:
    is_nar = False
    if "nar" in url_input:
        st.warning("地方競馬(NAR)のURLが入力されました。中央競馬専用モデルのため精度が低下する可能性があります。")
    url_input = url_input.replace("race.sp.netkeiba.com", "race.netkeiba.com")
    # v9モデル自動切替
    active_model_data, active_model_type = get_model_for_race(is_nar, use_live=True)
    active_model = active_model_data.get('model') if isinstance(active_model_data, dict) else None
    if active_model is None:
        active_model = model
        active_model_type = 'default'
    active_features = active_model_data.get('features', model_features) if isinstance(active_model_data, dict) else model_features
    active_version = active_model_data.get('version', model_version) if isinstance(active_model_data, dict) else model_version
    active_auc = active_model_data.get('auc', 0.0) if isinstance(active_model_data, dict) else 0.0
    active_sire_map = active_model_data.get('sire_map', sire_map) if isinstance(active_model_data, dict) else sire_map
    active_bms_map = active_model_data.get('bms_map', bms_map) if isinstance(active_model_data, dict) else bms_map
    is_live_model = active_model_type == 'central_live'
    pattern_a_auc = active_model_data.get('pattern_a_auc', active_auc) if isinstance(active_model_data, dict) else active_auc
    if is_live_model:
        race_badge = f'<span class="model-badge badge-central">LIVE {active_version.upper()} (Pattern B) 評価AUC {pattern_a_auc:.4f}</span>'
    else:
        race_badge = f'<span class="model-badge badge-central">CENTRAL {active_version.upper()} AUC {active_auc:.4f}</span>'
    model_badge_placeholder.markdown(f'<div style="text-align:center;margin-top:-12px;margin-bottom:12px">{race_badge}</div>', unsafe_allow_html=True)
    rid_match = re.search(r'race_id=(\d+)', url_input)
    if not rid_match:
        rid_match = re.search(r'/race/(\d{10,12})/?', url_input)
    if not rid_match:
        st.error("URLからrace_idを取得できませんでした")
        st.stop()
    race_id = rid_match.group(1)
    st.session_state['last_race_id'] = race_id
    st.session_state['last_is_nar'] = is_nar
    st.session_state['prediction_done'] = True
    with st.spinner("出馬表を取得中..."):
        race_name, horses, horse_ids, race_info = parse_shutuba(race_id, is_nar=is_nar)
    if not horses:
        st.error("馬データを取得できませんでした")
        st.stop()
    # 障害レース警告
    if race_info.get('surface') == '障':
        st.warning("障害レースはAIモデルの対象外です。予測精度は保証されません。")
    # Fetch realtime odds
    with st.spinner("オッズを取得中..."):
        realtime_odds = fetch_realtime_odds(race_id, is_nar=is_nar)
    # Fetch training (oikiri) data
    with st.spinner("調教データを取得中..."):
        training_data = fetch_training_data(race_id, is_nar=is_nar)
    for horse in horses:
        umaban = horse.get('馬番', 0)
        if umaban in realtime_odds:
            horse['単勝オッズ'] = realtime_odds[umaban]
        else:
            horse['単勝オッズ'] = 0.0  # 取得できなかった場合
        # 調教データ
        if umaban in training_data:
            horse['調教ランク'] = training_data[umaban]['rank']
            horse['調教評価'] = training_data[umaban]['evaluation']
            horse['調教ラベル'] = training_data[umaban]['label']
        else:
            horse['調教ランク'] = ''
            horse['調教評価'] = ''
            horse['調教ラベル'] = ''
    odds_available = len(realtime_odds) > 0
    # Fetch track bias (当日前レース結果分析)
    with st.spinner("馬場バイアスを分析中..."):
        bias_info = fetch_today_bias(race_id, race_info.get('course', ''), is_nar=is_nar)
    # Fetch JRA track condition & weather (Pattern B用)
    jra_track_info = {}
    weather_info = {}
    if is_live_model:
        with st.spinner("馬場情報・天候データを取得中..."):
            try:
                from scrape_jra_track import fetch_jra_track_info, get_moisture_rate
                jra_track_info = fetch_jra_track_info(race_info.get('course', ''))
            except Exception:
                jra_track_info = {}
            try:
                from scrape_weather import get_weather_features
                weather_info = get_weather_features(race_info.get('course', ''))
            except Exception:
                weather_info = {}
    # Race card
    surf_badge = 'badge-turf' if race_info['surface'] == '芝' else 'badge-dirt'
    surf_icon = '🟢 TURF' if race_info['surface'] == '芝' else '🟤 DIRT'
    num_horses = len(horses)
    rc_html = f'<div class="race-card"><span class="race-badge {surf_badge}">{surf_icon}</span>'
    # グレードバッジとレース番号
    grade = race_info.get('grade', '')
    race_num_str = race_info.get('race_num', '')
    grade_html = ''
    if grade:
        gcss = {'G1':'grade-g1','G2':'grade-g2','G3':'grade-g3','OP':'grade-op','L':'grade-list'}.get(grade, 'grade-op')
        grade_html = f'<span class="grade-badge {gcss}">{grade}</span>'
    num_html = f'<span class="race-num">{race_num_str}</span>' if race_num_str else ''
    rc_html += f'<div class="race-name">{num_html}{grade_html}{race_info["course"]} {race_name}</div>'
    rc_html += f'<div class="race-meta"><span>📏 {race_info["distance"]}m</span>'
    rc_html += f'<span>🏟️ {race_info["course"]}</span>'
    rc_html += f'<span>💧 {race_info["condition"]}</span>'
    weather_disp = race_info.get('weather', '')
    if weather_disp:
        weather_icon = {'晴':'☀️','曇':'☁️','雨':'🌧️','小雨':'🌦️','雪':'❄️'}.get(weather_disp, '🌤️')
        rc_html += f'<span>{weather_icon} {weather_disp}</span>'
    rc_html += f'<span>🐎 {num_horses}頭</span>'
    if odds_available:
        rc_html += f'<span>💰 オッズ取得済</span>'
    if is_live_model:
        rc_html += f'<span>🔴 Pattern B (当日情報込み)</span>'
    rc_html += '</div>'
    rank_map, pace_scores_map = calc_pace_advantage(
        race_info['distance'], race_info['surface'], race_info['condition'], num_horses, is_nar=is_nar
    )
    rc_html += render_pace_panel(rank_map)
    rc_html += '</div>'
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
                    # 過去5走lag特徴量
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
                    horse.update({'前走着順':5,'距離適性':0.5,'馬場適性':0.5,'人気傾向':0.5,
                                  'コース適性':0.5,'前走間隔':30,'脚質':0,'上がり3F':35.5,
                                  '複勝率':0.0,'父':'','母の父':'','血統スコア':0.5,'持ちタイム':0.0,
                                  'タイム表示':'','タイム日付':'','タイム距離':0,
                                  '通過順平均':8.0,'通過順4':8,'前走オッズ':15.0,'前走人気':8,'所属地':'',
                                  'prev2_finish':5,'prev3_finish':5,'prev4_finish':5,'prev5_finish':5,
                                  'avg_finish_3r':5.0,'avg_finish_5r':5.0,'best_finish_3r':5,'best_finish_5r':5,
                                  'top3_count_3r':0,'top3_count_5r':0,'finish_trend':0,'prev2_last3f':35.5})
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
    use_version = active_version
    if use_version == 'v3':
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
    if use_version in ('v5', 'v6', 'v8', 'v9'):
        n_top = active_model_data.get('n_top_encode', _loaded.get('n_top_encode', 80) if isinstance(_loaded, dict) else 80)
        # Sire/BMS encoding (v9モデル時はそのモデルのマップを使用)
        use_sire_map = active_sire_map if active_sire_map else sire_map
        use_bms_map = active_bms_map if active_bms_map else bms_map
        df['sire_enc'] = df['父'].apply(lambda x: use_sire_map.get(x, n_top) if use_sire_map else n_top)
        df['bms_enc'] = df['母の父'].apply(lambda x: use_bms_map.get(x, n_top) if use_bms_map else n_top)

        # Location encoding（地方対応）
        def enc_loc(loc):
            s = str(loc)
            if '美浦' in s or '美' == s: return 0
            if '栗東' in s or '栗' == s: return 1
            if is_nar: return 2  # 地方
            return 3
        df['location_enc'] = df['所属地'].apply(enc_loc)

        # Base mappings to English names
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

        # 過去5走lag特徴量（netkeibaから取得した実データ）
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

        # v8用追加特徴量
        df['surface_enc'] = df['芝ダート_enc']
        df['jockey_wr_calc'] = df['騎手勝率']
        df['jockey_course_wr_calc'] = df['騎手勝率']
        df['trainer_top3_calc'] = df['trainer_top3']
        df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
        df['surface_dist_enc'] = df['芝ダート_enc'] * 10 + df['dist_cat']
        df['cond_surface'] = df['馬場状態_enc'] * 10 + df['芝ダート_enc']
        df['course_surface'] = df['競馬場コード_enc'] * 10 + df['芝ダート_enc']
        df['is_nar'] = 1 if is_nar else 0

    # === リアルタイムオッズ特徴量 ===
    if odds_available and '単勝オッズ' in df.columns:
        df['odds_log'] = np.log1p(df['単勝オッズ'].clip(1, 999).replace(0, 15.0))
        # リアルタイムオッズがある場合、前走オッズの代わりに使う
        has_odds = df['単勝オッズ'] > 0
        if has_odds.any():
            if 'prev_odds_log' in df.columns:
                df.loc[has_odds, 'prev_odds_log'] = df.loc[has_odds, 'odds_log']
            if '前走オッズlog' in df.columns:
                df.loc[has_odds, '前走オッズlog'] = df.loc[has_odds, 'odds_log']
    else:
        df['odds_log'] = np.log1p(pd.Series([15.0] * len(df)))

    # === Pattern B 当日追加特徴量 ===
    if is_live_model:
        # 馬体重増減（当日 - 前走）
        df['weight_change'] = df['場体重増減'].fillna(0)
        df['weight_change_abs'] = df['weight_change'].abs()

        # 天候エンコード
        weather_str = str(race_info.get('weather', '晴'))
        weather_map = {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3}
        df['weather_enc'] = weather_map.get(weather_str, 0)

        # 人気順位（オッズから算出）
        if odds_available and '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any():
            df['pop_rank'] = df['単勝オッズ'].replace(0, 9999).rank(method='min')
        else:
            df['pop_rank'] = 8

        # 馬場指数（JRA公式）
        surface = race_info.get('surface', '芝')
        if jra_track_info:
            df['cushion_value'] = jra_track_info.get('cushion_value') or 0
            try:
                from scrape_jra_track import get_moisture_rate
                mr = get_moisture_rate(jra_track_info, surface)
                df['moisture_rate'] = mr if mr is not None else 0
            except Exception:
                df['moisture_rate'] = 0
        else:
            df['cushion_value'] = 0
            df['moisture_rate'] = 0

        # 天候データ（気象庁）
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

    # 使用する特徴量リスト（v9モデル時はそのモデルの特徴量を使用）
    use_features = active_features if active_features else FEATURES
    for f in use_features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    X = df[use_features].values
    use_model = active_model if (active_model is not None and active_model_type != 'default') else model
    if use_model is None:
        st.error("モデルが読み込めていません。keiba_model_v8.pkl が存在するか確認してください。")
        st.stop()
    if hasattr(use_model, 'predict_proba'):
        proba = use_model.predict_proba(X)
        ai_scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, :3].sum(axis=1)
    else:
        ai_scores = use_model.predict(X)
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

    # Odds scores（オッズが低いほど高スコア）
    if odds_available and '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any():
        odds_vals = df['単勝オッズ'].replace(0, df['単勝オッズ'][df['単勝オッズ'] > 0].median() if (df['単勝オッズ'] > 0).any() else 15.0)
        odds_scores = np.clip(1.0 - np.log1p(odds_vals) / np.log1p(100.0), 0.0, 1.0)
    else:
        odds_scores = pop_scores  # オッズ未取得時は人気傾向で代替

    # ===== FINAL SCORE =====
    if is_nar:
        if odds_available:
            final_scores = (
                ai_scores * 0.27 + odds_scores * 0.10 + pop_scores * 0.03 + pace_scores * 0.18
                + agari_scores * 0.08 + jockey_scores * 0.10 + apt_scores * 0.08
                + time_scores * 0.08 + course_scores * 0.04 + other_scores * 0.04
            )
        else:
            final_scores = (
                ai_scores * 0.27 + pop_scores * 0.13 + pace_scores * 0.18
                + agari_scores * 0.08 + jockey_scores * 0.10 + apt_scores * 0.08
                + time_scores * 0.08 + course_scores * 0.04 + other_scores * 0.04
            )
    else:
        if use_version == 'v3':
            final_scores = (
                ai_scores * 0.55 + pop_scores * 0.10 + apt_scores * 0.08
                + pace_scores * 0.08 + agari_scores * 0.08 + course_scores * 0.04
                + other_scores * 0.04 + time_scores * 0.03
            )
        elif use_version == 'v5':
            final_scores = (
                ai_scores * 0.60 + pop_scores * 0.08 + apt_scores * 0.07
                + pace_scores * 0.07 + agari_scores * 0.07 + course_scores * 0.04
                + other_scores * 0.04 + time_scores * 0.03
            )
        elif use_version == 'v6':
            final_scores = (
                ai_scores * 0.65 + pop_scores * 0.06 + apt_scores * 0.06
                + pace_scores * 0.06 + agari_scores * 0.06 + course_scores * 0.04
                + other_scores * 0.04 + time_scores * 0.03
            )
        elif use_version in ('v8', 'v9'):
            # v8/v9: 過去3走特徴量込み - リアルタイムオッズ反映
            if odds_available:
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
        else:
            final_scores = (
                ai_scores * 0.45 + pop_scores * 0.15 + apt_scores * 0.10
                + pace_scores * 0.10 + agari_scores * 0.10 + course_scores * 0.05
                + other_scores * 0.05
            )
    df['スコア'] = final_scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    # ===== 展開予測補正 =====
    pace_pred, pace_reason, pace_adv = predict_race_pace(
        df, race_info['distance'], race_info['surface'], num_horses
    )
    # 展開予測に基づくスコア補正
    if '脚質' in df.columns:
        for style, adj in pace_adv.items():
            mask = df['脚質'] == style
            df.loc[mask, 'スコア'] = df.loc[mask, 'スコア'] + adj
        df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
        df = df.sort_values('AI順位')
    # 展開予測結果を保存
    st.session_state['pred_pace'] = pace_pred
    st.session_state['pred_pace_reason'] = pace_reason
    st.session_state['pred_pace_adv'] = pace_adv
    # ===== 馬場バイアス補正 =====
    if bias_info.get('analyzed_races', 0) > 0:
        df, rank_map = apply_bias_adjustment(df, bias_info, rank_map)
    st.session_state['pred_bias_info'] = bias_info
    # 馬場警告フラグ初期化
    df['馬場警告'] = False
    # ===== 馬場変化リアルタイム補正 =====
    track_changed = False
    track_info = fetch_track_condition(race_id, is_nar=is_nar)
    original_condition = race_info.get('condition', '良')
    live_condition = track_info.get('condition')
    if live_condition and live_condition != original_condition:
        df, rank_map, track_changed = adjust_scores_for_track(
            df, live_condition, original_condition, race_info['surface'], rank_map
        )
        if track_changed:
            race_info['condition_original'] = original_condition
            race_info['condition'] = live_condition
            # race card HTMLを更新
            rc_html = rc_html.replace(
                f'💧 {original_condition}',
                f'💧 {live_condition} ⚠️馬場変化'
            )
    # Save prediction to SQLite
    save_prediction(race_id, race_name, race_info, df, is_nar=is_nar)
    # 予測結果をsession_stateに保存
    st.session_state['pred_df'] = df
    st.session_state['pred_race_name'] = race_name
    st.session_state['pred_race_info'] = race_info
    st.session_state['pred_rank_map'] = rank_map
    st.session_state['pred_odds_available'] = odds_available
    st.session_state['pred_realtime_odds'] = realtime_odds
    st.session_state['pred_rc_html'] = rc_html
    st.session_state['pred_track_changed'] = track_changed
    st.session_state['pred_model_type'] = active_model_type
    st.session_state['pred_model_version'] = active_version
    st.session_state['pred_model_auc'] = active_auc
    st.session_state['pred_is_nar'] = is_nar
    st.session_state['pred_is_live_model'] = is_live_model
    st.session_state['pred_pattern_a_auc'] = pattern_a_auc
    st.session_state['pred_jra_track'] = jra_track_info
    st.session_state['pred_weather'] = weather_info

# ===== 予測結果の表示（session_stateから） =====
if st.session_state.get('prediction_done') and 'pred_df' in st.session_state:
    df = st.session_state['pred_df']
    race_name = st.session_state['pred_race_name']
    race_info = st.session_state['pred_race_info']
    rank_map = st.session_state['pred_rank_map']
    odds_available = st.session_state['pred_odds_available']
    realtime_odds = st.session_state['pred_realtime_odds']
    rc_html = st.session_state['pred_rc_html']
    race_id = st.session_state.get('last_race_id', '')
    is_nar = st.session_state.get('last_is_nar', False)

    # モデルバッジ更新
    p_model_ver = st.session_state.get('pred_model_version', model_version)
    p_model_auc = st.session_state.get('pred_model_auc', model_auc)
    p_is_live = st.session_state.get('pred_is_live_model', False)
    p_pattern_a_auc = st.session_state.get('pred_pattern_a_auc', p_model_auc)
    if p_is_live:
        p_race_badge = f'<span class="model-badge badge-central">LIVE {p_model_ver.upper()} (Pattern B) 評価AUC {p_pattern_a_auc:.4f}</span>'
    else:
        p_race_badge = f'<span class="model-badge badge-central">CENTRAL {p_model_ver.upper()} AUC {p_model_auc:.4f}</span>'
    model_badge_placeholder.markdown(f'<div style="text-align:center;margin-top:-12px;margin-bottom:12px">{p_race_badge}</div>', unsafe_allow_html=True)
    st.markdown(rc_html, unsafe_allow_html=True)

    # === 当日リアルタイム情報パネル ===
    weather_str = race_info.get('weather', '')
    condition_str = race_info.get('condition', '良')
    rt_items = []
    if weather_str:
        rt_items.append(f"天候: {weather_str}")
    rt_items.append(f"馬場: {condition_str}")
    if odds_available:
        rt_items.append("オッズ: 取得済")

    # 馬体重急変警告（±5kg以上）
    weight_warnings = []
    if '場体重増減' in df.columns:
        for _, row in df.iterrows():
            w_diff = row.get('場体重増減', 0)
            if abs(w_diff) >= 5:
                sign = "+" if w_diff > 0 else ""
                severity = 'critical' if abs(w_diff) >= 10 else 'warning'
                weight_warnings.append((f"{row.get('馬名', '?')}({row.get('馬番', '?')}番): {sign}{w_diff:.0f}kg", severity))

    # オッズ急変警告（1番人気が10倍以上 or 上位人気に大穴）
    odds_warnings = []
    if odds_available and '単勝オッズ' in df.columns:
        odds_vals = df[df['単勝オッズ'] > 0]['単勝オッズ']
        if len(odds_vals) > 0:
            min_odds = odds_vals.min()
            if min_odds >= 10.0:
                odds_warnings.append(f"1番人気でも{min_odds:.1f}倍 - 混戦レース")

    if weight_warnings or odds_warnings:
        warn_html = '<div style="background:rgba(220,53,69,0.08);border:1px solid rgba(220,53,69,0.3);border-radius:8px;padding:10px;margin:8px 0">'
        warn_html += '<b style="color:#ff4d4d !important;">当日アラート</b><br>'
        for w_text, sev in weight_warnings:
            if sev == 'critical':
                warn_html += f'<span style="color:#ff3333 !important;font-weight:bold;">体重急変: {w_text}</span><br>'
            else:
                warn_html += f'<span style="color:#fd7e14 !important;font-weight:bold;">体重変動: {w_text}</span><br>'
        for w in odds_warnings:
            warn_html += f'<span style="color:#fd7e14 !important;">{w}</span><br>'
        warn_html += '</div>'
        st.markdown(warn_html, unsafe_allow_html=True)
    # 馬場・天候情報パネル（Pattern B）
    p_jra_track = st.session_state.get('pred_jra_track', {})
    p_weather = st.session_state.get('pred_weather', {})
    if p_jra_track or p_weather:
        info_parts = []
        if p_weather.get('temperature'):
            info_parts.append(f"気温 {p_weather['temperature']:.1f}℃")
        if p_weather.get('humidity') and p_weather['humidity'] != 60.0:
            info_parts.append(f"湿度 {p_weather['humidity']:.0f}%")
        if p_weather.get('wind_speed'):
            wd = p_weather.get('wind_direction', '')
            info_parts.append(f"風速 {p_weather['wind_speed']:.1f}m/s {wd}")
        if p_weather.get('precipitation', 0) > 0:
            info_parts.append(f"降水量 {p_weather['precipitation']:.1f}mm/h")
        if p_weather.get('weather_text'):
            info_parts.append(f"予報: {p_weather['weather_text']}")
        if p_jra_track.get('cushion_value'):
            info_parts.append(f"クッション値 {p_jra_track['cushion_value']:.1f}")
        if p_jra_track.get('moisture_turf_goal') or p_jra_track.get('moisture_dirt_goal'):
            surface = race_info.get('surface', '芝')
            if surface == '芝' and p_jra_track.get('moisture_turf_goal'):
                info_parts.append(f"含水率(芝) {p_jra_track['moisture_turf_goal']:.1f}%")
            elif p_jra_track.get('moisture_dirt_goal'):
                info_parts.append(f"含水率(ダ) {p_jra_track['moisture_dirt_goal']:.1f}%")

        if info_parts:
            env_html = '<div style="background:rgba(40,152,216,0.1);border:1px solid rgba(40,152,216,0.3);border-radius:8px;padding:10px;margin:8px 0">'
            env_html += '<b>🌤️ 馬場・天候データ</b> <span style="font-size:0.8em;color:#6a6a80">(Pattern B特徴量に反映済)</span><br>'
            env_html += '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:4px">'
            for part in info_parts:
                env_html += f'<span style="font-size:0.9em">{part}</span>'
            env_html += '</div></div>'
            st.markdown(env_html, unsafe_allow_html=True)

    # 展開予測パネル
    p_pace = st.session_state.get('pred_pace', 'middle')
    p_reason = st.session_state.get('pred_pace_reason', '')
    p_adv = st.session_state.get('pred_pace_adv', {1:0,2:0,3:0,4:0})
    st.markdown(render_pace_prediction(p_pace, p_reason, p_adv), unsafe_allow_html=True)
    # 馬場バイアスパネル（前レース結果がある場合）
    p_bias = st.session_state.get('pred_bias_info', {})
    if p_bias.get('analyzed_races', 0) > 0:
        st.markdown(render_bias_panel(p_bias, race_info.get('course', '')), unsafe_allow_html=True)
    # 馬場変化アラート
    if st.session_state.get('pred_track_changed'):
        orig_cond = race_info.get('condition_original', '良')
        new_cond = race_info.get('condition', '良')
        st.markdown(f'<div style="margin:8px 0;padding:12px;background:linear-gradient(90deg,#3a1a0a,#2a1a1a);border:1px solid #e67e22;border-radius:10px;text-align:center;color:#e67e22 !important;font-weight:bold;">⚠️ 馬場変化検知: {orig_cond} → {new_cond}　スコア自動補正済（前残り有利に調整）</div>', unsafe_allow_html=True)
    # Render TOP3
    st.markdown('<div class="sec-title">🏆 AI TOP 3<span class="sec-line"></span></div>', unsafe_allow_html=True)
    max_score = df['スコア'].max()
    for _, row in df.head(3).iterrows():
        st.markdown(render_horse_card(int(row['AI順位']), row, max_score, rank_map), unsafe_allow_html=True)
    # 条件別買い目自動切替（統合表示）
    is_nar_pred = st.session_state.get('pred_is_nar', False)
    cond_key, cond_profile = classify_race_condition(race_info, len(df), is_nar=is_nar_pred)
    is_recommended = cond_profile['recommended'] and cond_profile['roi'] >= 80

    # ワイド/馬連オッズ取得（オッズ連動投資額振り分け用）
    pair_odds = {}
    bet_type = cond_profile['bet_type']
    if is_recommended and bet_type in ('wide', 'umaren') and odds_available:
        pair_odds = fetch_pair_odds(race_id, bet_type=bet_type, is_nar=is_nar_pred)

    if is_recommended:
        st.markdown('<div class="sec-title">🎯 AI推奨 買い目<span class="sec-line"></span></div>', unsafe_allow_html=True)
        buy_html = render_buy_section(df, race_info, rank_map, cond_key=cond_key, cond_profile=cond_profile, pair_odds=pair_odds)
        if buy_html:
            st.markdown(buy_html, unsafe_allow_html=True)
    else:
        dist_val = race_info.get('distance', 0)
        if cond_key == 'D' and dist_val <= 1000:
            reason = f'1000m以下：非推奨（WFバックテスト ROI 85.0%, N=534）。購入非推奨。'
        else:
            reason = '5年バックテスト結果: この条件では的中率・ROIが低下。見送りまたは少額投資推奨。'
        st.markdown(f'''<div style="margin:8px 0;padding:14px;background:linear-gradient(135deg,#2a0a0a,#3a1a1a);border:2px solid #ff4060;border-radius:12px;">
<div style="font-family:Oswald;font-size:1.1em;color:#ff4060 !important;margin-bottom:8px;">&#9888;&#65039; NOT RECOMMENDED</div>
<div style="font-size:0.9em;color:#ff4060 !important;">{cond_profile["label"]}: {cond_profile["desc"]}</div>
<div style="font-size:0.82em;color:#6a6a80 !important;margin-top:8px;">{reason}</div>
</div>''', unsafe_allow_html=True)
    # Expected Value Section (with trio odds integration)
    if odds_available:
        st.markdown('<div class="sec-title">💰 期待値分析<span class="sec-line"></span></div>', unsafe_allow_html=True)
        ev_list = calc_expected_values(df, realtime_odds)
        # 三連複オッズを取得してEVを更新
        trio_odds = fetch_trio_odds(race_id, is_nar=is_nar)
        if trio_odds:
            for e in ev_list:
                if e['type'] == '三連複' and e['ev'] == 0:
                    key = tuple(sorted(e['umaban']))
                    if key in trio_odds:
                        e['odds'] = trio_odds[key]
                        e['ev'] = e['prob'] * trio_odds[key]
        ev_display = [e for e in ev_list if e['ev'] > 0]
        st.markdown(render_ev_section(ev_display), unsafe_allow_html=True)
        # レース全体のEVサマリー
        trio_evs = [e for e in ev_display if e['type'] == '三連複']
        if trio_evs:
            race_ev = sum(e['ev'] for e in trio_evs) / len(trio_evs) if trio_evs else 0
            max_ev = max(e['ev'] for e in trio_evs) if trio_evs else 0
            ev_above_1 = sum(1 for e in trio_evs if e['ev'] >= 1.0)
            ev_color = '#2ecc40' if race_ev >= 1.0 else ('#f0c040' if race_ev >= 0.7 else '#ff4060')
            st.markdown(f'''<div style="margin:8px 0;padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;display:flex;gap:20px;align-items:center;">
<span style="font-size:0.85em;color:#8890a0 !important;">Race EV:</span>
<span style="font-family:Oswald;font-size:1.1em;color:{ev_color} !important;font-weight:bold;">{race_ev:.2f}</span>
<span style="font-size:0.8em;color:#6a6a80 !important;">Max: {max_ev:.2f} | EV&gt;1.0: {ev_above_1}/{len(trio_evs)}点</span>
</div>''', unsafe_allow_html=True)
        # EV>=1.0の三連複買い目をハイライト
        hot_trio = [e for e in ev_display if e['type'] == '三連複' and e['ev'] >= 1.0]
        if hot_trio:
            hot_html = '<div style="margin:8px 0;padding:12px;background:linear-gradient(135deg,#1a3a1a,#0a2a0a);border:1px solid #2ecc40;border-radius:10px;">'
            hot_html += '<div style="color:#2ecc40 !important;font-weight:bold;margin-bottom:6px;">🎯 高期待値の三連複買い目</div>'
            for e in sorted(hot_trio, key=lambda x: x['ev'], reverse=True):
                ev_tag = '<span style="color:#ff4060 !important;font-weight:bold;">🔥 HOT</span>' if e['ev'] >= 1.5 else '<span style="color:#f0c040 !important;">★</span>'
                hot_html += f'<div style="padding:4px 0;font-size:0.9em;">{ev_tag} {e["horses"]} (EV {e["ev"]:.2f} / {e["odds"]:.1f}倍)</div>'
            hot_html += '</div>'
            st.markdown(hot_html, unsafe_allow_html=True)
    st.success(f"予測結果をDBに保存しました ({race_name})")
    # Chart - カラフルな横棒グラフ（HTML）
    st.markdown('<div class="sec-title">📊 全馬スコア<span class="sec-line"></span></div>', unsafe_allow_html=True)
    chart_html = '<div style="padding:12px 0;">'
    sorted_chart = df.sort_values('AI順位')
    chart_max = sorted_chart['スコア'].max()
    for _, row in sorted_chart.iterrows():
        r = int(row['AI順位'])
        pct = row['スコア'] / chart_max * 100 if chart_max > 0 else 0
        if r == 1:
            bar_bg = 'linear-gradient(90deg, #c89020, #f0c040)'
            lbl_c = '#f0c040'
        elif r == 2:
            bar_bg = 'linear-gradient(90deg, #808898, #b0b8c8)'
            lbl_c = '#b0b8c8'
        elif r == 3:
            bar_bg = 'linear-gradient(90deg, #905020, #c87840)'
            lbl_c = '#c87840'
        elif r <= 6:
            bar_bg = 'linear-gradient(90deg, #1a5080, #2898d8)'
            lbl_c = '#60b0e0'
        else:
            bar_bg = 'linear-gradient(90deg, #303848, #485068)'
            lbl_c = '#8890a0'
        chart_html += f'<div style="display:flex;align-items:center;margin-bottom:4px;gap:6px;">'
        chart_html += f'<span style="font-size:0.75em;width:60px;text-align:right;color:{lbl_c} !important;white-space:nowrap;overflow:hidden;">{row["馬名"][:5]}</span>'
        chart_html += f'<div style="flex:1;height:18px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden;">'
        chart_html += f'<div style="width:{pct:.0f}%;height:100%;background:{bar_bg};border-radius:4px;"></div></div>'
        chart_html += f'<span style="font-family:Oswald;font-size:0.78em;width:44px;color:{lbl_c} !important;">{row["スコア"]:.3f}</span>'
        chart_html += '</div>'
    chart_html += '</div>'
    st.markdown(chart_html, unsafe_allow_html=True)
    # Table
    st.markdown('<div class="sec-title">📋 ALL HORSES<span class="sec-line"></span></div>', unsafe_allow_html=True)
    st.markdown(render_table(df, rank_map), unsafe_allow_html=True)
    # Disclaimer
    st.markdown('<div class="disclaimer">&#9888;&#65039; 本予想はAIによる統計分析です。馬券の購入は自己責任でお願いします。</div>', unsafe_allow_html=True)

# ===== TRACK RECORD（全面改修） =====
st.markdown('<div class="sec-title">📈 TRACK RECORD<span class="sec-line"></span></div>', unsafe_allow_html=True)

_tr_all_data = get_track_record_all()
if not _tr_all_data:
    st.info("予測記録がありません。レースを予測すると記録が表示されます。")
else:
    # --- サマリー（月別フィルタ付き） ---
    _tr_all_months = sorted(set(
        r.get('race_date', r.get('predicted_at', ''))[:7]
        for r in _tr_all_data
        if r.get('race_date') or r.get('predicted_at')
    ), reverse=True)
    _tr_month_options = ['全期間'] + _tr_all_months
    _tr_sel_month = st.selectbox("📅 期間フィルタ", _tr_month_options, key="tr_month_filter")
    if _tr_sel_month == '全期間':
        _tr_filtered = _tr_all_data
    else:
        _tr_filtered = [r for r in _tr_all_data
                        if (r.get('race_date') or r.get('predicted_at', ''))[:7] == _tr_sel_month]
    _tr_stats = _calc_stats_from_rows(_tr_filtered)
    _tr_summary_html = '<div class="ev-card">'
    _tr_summary_html += _render_stats_block(_tr_stats, f'&#127942; {_tr_sel_month}')
    _tr_summary_html += '</div>'
    st.markdown(_tr_summary_html, unsafe_allow_html=True)

    # --- レースブラウザ（2タブ） ---
    _tr_tab_venue, _tr_tab_date = st.tabs(["🏟️ 場所から探す", "📅 日付から探す"])

    with _tr_tab_venue:
        # 競馬場ボタン（データあり=緑, なし=グレー）
        _tr_venues_with_data = {}
        for r in _tr_all_data:
            v = r.get('course', '')
            if v:
                _tr_venues_with_data[v] = _tr_venues_with_data.get(v, 0) + 1
        for row_start in [0, 5]:
            _tr_vcols = st.columns(5)
            for i in range(5):
                venue = JRA_VENUES[row_start + i]
                cnt = _tr_venues_with_data.get(venue, 0)
                with _tr_vcols[i]:
                    if cnt > 0:
                        if st.button(f"🟢 {venue} ({cnt})", key=f"tr_v_{venue}", use_container_width=True):
                            st.session_state['tr_selected_venue'] = venue
                            st.session_state.pop('tr_venue_date_idx', None)
                            st.rerun()
                    else:
                        st.button(f"⚫ {venue}", key=f"tr_v_{venue}", disabled=True, use_container_width=True)

        _tr_sv = st.session_state.get('tr_selected_venue')
        if _tr_sv and _tr_sv in _tr_venues_with_data:
            st.markdown(f"### {_tr_sv}")
            _tr_venue_races = [r for r in _tr_all_data if r.get('course') == _tr_sv]
            _tr_venue_dates = sorted(set(
                (r.get('race_date') or r.get('predicted_at', ''))[:10]
                for r in _tr_venue_races
                if r.get('race_date') or r.get('predicted_at')
            ), reverse=True)
            if _tr_venue_dates:
                _tr_vd_labels = []
                _tr_weekdays = ['月', '火', '水', '木', '金', '土', '日']
                for d in _tr_venue_dates:
                    try:
                        dt = datetime.strptime(d, '%Y-%m-%d')
                        wd = _tr_weekdays[dt.weekday()]
                        day_races = [r for r in _tr_venue_races if (r.get('race_date') or '')[:10] == d]
                        _tr_vd_labels.append(f"🟢 {d}({wd}) - {len(day_races)}R")
                    except Exception:
                        _tr_vd_labels.append(d)
                _tr_vd_idx = st.selectbox(
                    "予測済みの日付を選択", range(len(_tr_venue_dates)),
                    format_func=lambda i: _tr_vd_labels[i],
                    key="tr_venue_date_idx"
                )
                _tr_selected_vd = _tr_venue_dates[_tr_vd_idx]
                _tr_day_races = [r for r in _tr_venue_races
                                 if (r.get('race_date') or r.get('predicted_at', ''))[:10] == _tr_selected_vd]
                _tr_day_races.sort(key=lambda r: r.get('predicted_at', ''))
                render_track_record_race_list(_tr_day_races)

    with _tr_tab_date:
        # 日付一覧（予測済みのみ）
        _tr_all_dates = sorted(set(
            (r.get('race_date') or r.get('predicted_at', ''))[:10]
            for r in _tr_all_data
            if r.get('race_date') or r.get('predicted_at')
        ), reverse=True)
        if _tr_all_dates:
            _tr_d_labels = []
            _tr_weekdays2 = ['月', '火', '水', '木', '金', '土', '日']
            for d in _tr_all_dates:
                try:
                    dt = datetime.strptime(d, '%Y-%m-%d')
                    wd = _tr_weekdays2[dt.weekday()]
                    day_races = [r for r in _tr_all_data if (r.get('race_date') or '')[:10] == d]
                    venues = set(r.get('course', '') for r in day_races if r.get('course'))
                    venues_str = '/'.join(sorted(venues)) if venues else ''
                    _tr_d_labels.append(f"🟢 {d}({wd}) {venues_str} - {len(day_races)}R")
                except Exception:
                    _tr_d_labels.append(d)
            _tr_d_idx = st.selectbox(
                "予測済みの日付を選択", range(len(_tr_all_dates)),
                format_func=lambda i: _tr_d_labels[i],
                key="tr_date_select_idx"
            )
            _tr_selected_d = _tr_all_dates[_tr_d_idx]
            _tr_d_data = [r for r in _tr_all_data
                          if (r.get('race_date') or r.get('predicted_at', ''))[:10] == _tr_selected_d]
            _tr_d_venues = sorted(set(r.get('course', '') for r in _tr_d_data if r.get('course')))

            if len(_tr_d_venues) > 1:
                # 複数開催場：場所選択ボタン
                _tr_dv_cols = st.columns(len(_tr_d_venues))
                for i, venue in enumerate(_tr_d_venues):
                    v_cnt = len([r for r in _tr_d_data if r.get('course') == venue])
                    with _tr_dv_cols[i]:
                        if st.button(f"🟢 {venue} ({v_cnt}R)", key=f"tr_dv_{venue}", use_container_width=True):
                            st.session_state['tr_date_venue'] = venue
                            st.rerun()
                _tr_dv = st.session_state.get('tr_date_venue')
                if _tr_dv and _tr_dv in _tr_d_venues:
                    st.markdown(f"### {_tr_selected_d} {_tr_dv}")
                    _tr_dv_races = [r for r in _tr_d_data if r.get('course') == _tr_dv]
                    _tr_dv_races.sort(key=lambda r: r.get('predicted_at', ''))
                    render_track_record_race_list(_tr_dv_races)
                else:
                    # 場未選択 → 全場表示
                    for venue in _tr_d_venues:
                        st.markdown(f"### {venue}")
                        _tr_venue_d_races = [r for r in _tr_d_data if r.get('course') == venue]
                        _tr_venue_d_races.sort(key=lambda r: r.get('predicted_at', ''))
                        render_track_record_race_list(_tr_venue_d_races)
            elif len(_tr_d_venues) == 1:
                st.markdown(f"### {_tr_selected_d} {_tr_d_venues[0]}")
                _tr_d_data.sort(key=lambda r: r.get('predicted_at', ''))
                render_track_record_race_list(_tr_d_data)
            else:
                _tr_d_data.sort(key=lambda r: r.get('predicted_at', ''))
                render_track_record_race_list(_tr_d_data)

    # --- 管理（削除機能） ---
    with st.expander("🗑️ TRACK RECORD 管理（選択削除・全件削除）"):
        all_records = get_all_race_records()
        if not all_records:
            st.info("記録されたレースはありません。")
        else:
            st.markdown(f"**登録レース数: {len(all_records)}件**")
            col_del_all, col_spacer = st.columns([1, 2])
            with col_del_all:
                if st.button("⚠️ 全件削除", key="delete_all_btn"):
                    st.session_state['confirm_delete_all'] = True
            if st.session_state.get('confirm_delete_all'):
                st.warning("本当に全レコードを削除しますか？この操作は取り消せません。")
                col_yes, col_no, _ = st.columns([1, 1, 2])
                with col_yes:
                    if st.button("はい、全件削除", key="confirm_yes"):
                        delete_all_race_records()
                        st.session_state['confirm_delete_all'] = False
                        st.success("全レコードを削除しました。")
                        st.rerun()
                with col_no:
                    if st.button("キャンセル", key="confirm_no"):
                        st.session_state['confirm_delete_all'] = False
                        st.rerun()
            selected_ids = []
            for rec in all_records:
                rid = rec.get('race_id', '')
                name = rec.get('race_name', '')[:12]
                date = (rec.get('predicted_at', '') or '')[:10]
                hit = rec.get('hit_trio')
                payout = rec.get('payout', 0) or 0
                bt = rec.get('bet_type', 'trio') or 'trio'
                bt_label = {'trio': '三連複', 'umaren': '馬連', 'wide': 'ワイド'}.get(bt, '三連複')
                if hit is not None:
                    status = f"✅ +¥{payout - INVESTMENT_PER_RACE:,}" if hit == 1 else f"❌ -¥{INVESTMENT_PER_RACE}"
                else:
                    status = "⏳ 未確定"
                label = f"[JRA] {date} {name} ({bt_label}) {status}"
                if st.checkbox(label, key=f"del_{rid}"):
                    selected_ids.append(rid)
            if selected_ids:
                if st.button(f"🗑️ 選択した {len(selected_ids)} 件を削除", key="delete_selected_btn"):
                    delete_race_records(selected_ids)
                    st.success(f"{len(selected_ids)} 件のレコードを削除しました。")
                    st.rerun()

# ===== モデル情報・特徴量重要度 =====
with st.expander("🤖 モデル情報・特徴量重要度"):
    if _v9_models.get('central_live'):
        v9l = _v9_models['central_live']
        v9l_ver = v9l.get('version', 'v9_live').upper()
        v9l_auc = v9l.get('ensemble_auc', v9l.get('auc', 0))
        v9l_pa = v9l.get('pattern_a_auc', 0)
        st.markdown(f"**{v9l_ver} (実運用/Pattern B):** Ensemble AUC {v9l_auc:.4f} / 評価AUC(Pattern A) {v9l_pa:.4f}")
        live_feats = v9l.get('live_features', [])
        if live_feats:
            st.markdown(f"当日特徴量: `{'`, `'.join(live_feats)}`")
    if _v9_models.get('central'):
        v9c = _v9_models['central']
        v9c_ver = v9c.get('version', 'v9').upper()
        v9c_auc = v9c.get('auc', 0)
        v9c_ens = v9c.get('ensemble_auc', v9c_auc)
        st.markdown(f"**CENTRAL {v9c_ver} (評価用/Pattern A):** LGB AUC {v9c_auc:.4f} / Ensemble AUC {v9c_ens:.4f}")
    if not _v9_models.get('central') and not _v9_models.get('central_live'):
        st.markdown(f"**V8 (フォールバック):** AUC {model_auc:.4f}")
    # Feature importance (top 20)
    fi_model = None
    fi_features = None
    for src_name, src_data in [('v9 live (Pattern B)', _v9_models.get('central_live')), ('v9 central (Pattern A)', _v9_models.get('central')), ('v8', _loaded)]:
        if src_data and isinstance(src_data, dict) and 'model' in src_data:
            m = src_data['model']
            importances = None
            if hasattr(m, 'feature_importance'):
                # LightGBM Booster
                importances = m.feature_importance(importance_type='gain')
            elif hasattr(m, 'feature_importances_'):
                # sklearn-like models
                importances = m.feature_importances_
            if importances is not None:
                fi_model = m
                fi_features = src_data.get('features') or FEATURES
                st.markdown(f"**特徴量重要度 ({src_name}): TOP 20**")
                pairs = sorted(zip(fi_features, importances), key=lambda x: x[1], reverse=True)[:20]
                fi_html = '<div style="font-size:0.85em;">'
                max_imp = pairs[0][1] if pairs else 1
                for fname, imp in pairs:
                    bar_w = imp / max_imp * 100 if max_imp > 0 else 0
                    fi_html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">'
                    fi_html += f'<span style="min-width:120px;color:#b0b8c8 !important;font-size:0.82em;">{fname}</span>'
                    fi_html += f'<div style="flex:1;height:12px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">'
                    fi_html += f'<div style="width:{bar_w:.0f}%;height:100%;background:linear-gradient(90deg,#2898d8,#00e87b);border-radius:3px;"></div></div>'
                    fi_html += f'<span style="font-family:Oswald;font-size:0.75em;min-width:60px;color:#6a6a80 !important;">{imp:.0f}</span>'
                    fi_html += '</div>'
                fi_html += '</div>'
                st.markdown(fi_html, unsafe_allow_html=True)
                break
    if fi_model is None:
        st.info("特徴量重要度はLightGBMモデルの読み込み時に表示されます。")

# ===== 週次分析レポート =====
with st.expander("📊 週次分析レポート"):
    weekly = get_weekly_analysis()
    if weekly:
        st.markdown(render_weekly_report(weekly), unsafe_allow_html=True)
    else:
        st.info("分析データがありません。レース結果を登録すると、コース/距離/馬場/頭数別の的中率・回収率を自動分析します。")

# ===== V8 vs V9 バックテスト比較 =====
with st.expander("🧪 V8 vs V9 Backtest Report"):
    bt_data = load_backtest_report()
    if bt_data:
        st.markdown(render_backtest_report(bt_data), unsafe_allow_html=True)
    else:
        st.info("バックテスト未実施。`python backtest_v8_v9.py` を実行するとリークフリーの精度検証レポートが生成されます。")

    bt5 = load_backtest_5year()
    if bt5:
        st.markdown(render_5year_report(bt5), unsafe_allow_html=True)
    else:
        st.info("5年バックテスト未実施。`python backtest_v8_v9.py --5year` で2020-2025の拡張検証を実行できます。")

# ===== 一括予測用モデルスコアリング =====
def _batch_score_race(horses, race_info, is_nar):
    """バッチ予測用: 特徴量エンジニアリング+モデル予測+スコア計算。
    Returns: (df_sorted, cond_key, cond_profile, odds_available) or None on error."""
    try:
        b_md, b_mt = get_model_for_race(is_nar, use_live=True)
        b_model = b_md.get('model') if isinstance(b_md, dict) else model
        if b_model is None:
            b_model = model
            b_mt = 'default'
        b_feats = b_md.get('features', model_features) if isinstance(b_md, dict) else model_features
        b_ver = b_md.get('version', model_version) if isinstance(b_md, dict) else model_version
        b_smap = b_md.get('sire_map', sire_map) if isinstance(b_md, dict) else sire_map
        b_bmap = b_md.get('bms_map', bms_map) if isinstance(b_md, dict) else bms_map
        b_live = b_mt == 'central_live'
        n_top = b_md.get('n_top_encode', 80) if isinstance(b_md, dict) else 80

        df = pd.DataFrame(horses)
        if len(df) < 2:
            return None
        num_h = len(df)
        odds_avail = '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any()

        # === 共通特徴量 ===
        df['頭数'] = num_h
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
        now_dt = dt_module.datetime.now()
        df['月'] = now_dt.month
        m = now_dt.month
        df['季節'] = 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3))
        df['枠馬場'] = df['枠位置'] * 10 + df['馬場状態_enc']
        df['馬齢グループ'] = df['馬齢'].clip(2, 7)

        # === v5/v8/v9 特徴量 ===
        if b_ver in ('v5', 'v6', 'v8', 'v9'):
            use_smap = b_smap if b_smap else sire_map
            use_bmap = b_bmap if b_bmap else bms_map
            df['sire_enc'] = df['父'].apply(lambda x: use_smap.get(x, n_top) if use_smap else n_top)
            df['bms_enc'] = df['母の父'].apply(lambda x: use_bmap.get(x, n_top) if use_bmap else n_top)
            def enc_loc(loc):
                s = str(loc)
                if '美浦' in s or '美' == s: return 0
                if '栗東' in s or '栗' == s: return 1
                if is_nar: return 2
                return 3
            df['location_enc'] = df.get('所属地', pd.Series(['']*len(df))).apply(enc_loc)
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
            month_now = datetime.now().month
            df['month_val'] = month_now
            df['season'] = 0 if month_now in [3,4,5] else (1 if month_now in [6,7,8] else (2 if month_now in [9,10,11] else 3))
            df['bracket_cond'] = df['bracket_pos'] * 10 + df['馬場状態_enc']
            df['age_group'] = df['馬齢'].clip(2, 7)
            df['prev_pop'] = df.get('前走人気', pd.Series([8]*len(df))).fillna(8)
            df['prev_odds_log'] = np.log1p(df.get('前走オッズ', pd.Series([15.0]*len(df))).clip(1, 999).fillna(15.0))
            df['prev_last3f'] = df.get('上がり3F', pd.Series([35.5]*len(df))).fillna(35.5)
            df['prev_pass1'] = df.get('通過順平均', pd.Series([8.0]*len(df))).fillna(8.0)
            df['prev_pass4'] = df.get('通過順4', pd.Series([8]*len(df))).fillna(8)
            df['prev_margin'] = 0
            df['prev_prize'] = 0
            for col in ['prev2_finish','prev3_finish','prev4_finish','prev5_finish']:
                df[col] = df.get(col, pd.Series([5]*len(df))).fillna(5)
            df['prev2_last3f'] = df.get('prev2_last3f', pd.Series([35.5]*len(df))).fillna(35.5)
            for col in ['avg_finish_3r','avg_finish_5r']:
                df[col] = df.get(col, pd.Series([5.0]*len(df))).fillna(5.0)
            for col in ['best_finish_3r','best_finish_5r']:
                df[col] = df.get(col, pd.Series([5]*len(df))).fillna(5)
            for col in ['top3_count_3r','top3_count_5r']:
                df[col] = df.get(col, pd.Series([0]*len(df))).fillna(0)
            df['finish_trend'] = df.get('finish_trend', pd.Series([0]*len(df))).fillna(0)
            df['dist_change'] = 0
            df['dist_change_abs'] = 0
            df['rest_days'] = df.get('前走間隔', pd.Series([30]*len(df))).fillna(30)
            df['rest_category'] = pd.cut(df['rest_days'], bins=[-1,6,14,35,63,180,9999], labels=[0,1,2,3,4,5]).astype(float).fillna(2)
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
            df['is_nar'] = 1 if is_nar else 0

        # === オッズ特徴量 ===
        if odds_avail and '単勝オッズ' in df.columns:
            df['odds_log'] = np.log1p(df['単勝オッズ'].clip(1, 999).replace(0, 15.0))
            has_odds = df['単勝オッズ'] > 0
            if has_odds.any():
                if 'prev_odds_log' in df.columns:
                    df.loc[has_odds, 'prev_odds_log'] = df.loc[has_odds, 'odds_log']
        else:
            df['odds_log'] = np.log1p(pd.Series([15.0] * len(df)))

        # === Pattern B 当日特徴量 ===
        if b_live:
            df['weight_change'] = df['場体重増減'].fillna(0)
            df['weight_change_abs'] = df['weight_change'].abs()
            weather_str = str(race_info.get('weather', '晴'))
            weather_map = {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3}
            df['weather_enc'] = weather_map.get(weather_str, 0)
            if odds_avail and '単勝オッズ' in df.columns and (df['単勝オッズ'] > 0).any():
                df['pop_rank'] = df['単勝オッズ'].replace(0, 9999).rank(method='min')
            else:
                df['pop_rank'] = 8
            df['cushion_value'] = 0
            df['moisture_rate'] = 0
            df['temperature'] = 0
            df['humidity'] = 0
            df['wind_speed'] = 0
            df['precipitation'] = 0

        # === モデル予測 ===
        use_features = b_feats if b_feats else FEATURES
        for f in use_features:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
        X = df[use_features].values
        use_model = b_model if (b_model is not None and b_mt != 'default') else model
        if use_model is None:
            return None
        if hasattr(use_model, 'predict_proba'):
            proba = use_model.predict_proba(X)
            ai_scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, :3].sum(axis=1)
        else:
            ai_scores = use_model.predict(X)

        # === スコア計算 ===
        pop_scores = df['人気傾向'].values if '人気傾向' in df.columns else np.full(num_h, 0.5)
        apt_scores = ((df['距離適性'].values if '距離適性' in df.columns else np.full(num_h, 0.5)) +
                      (df['馬場適性'].values if '馬場適性' in df.columns else np.full(num_h, 0.5))) / 2.0
        agari_scores = np.clip(1.0 - (df['上がり3F'].values - 33.0) / 5.0, 0.0, 1.0) if '上がり3F' in df.columns else np.full(num_h, 0.5)
        course_scores = df['コース適性'].values if 'コース適性' in df.columns else np.full(num_h, 0.5)
        other_scores = ((df['血統スコア'].values if '血統スコア' in df.columns else np.full(num_h, 0.5)) +
                        (df['複勝率'].values if '複勝率' in df.columns else np.full(num_h, 0.0))) / 2.0
        if odds_avail:
            odds_vals = df['単勝オッズ'].replace(0, 15.0)
            odds_scores = np.clip(1.0 - np.log1p(odds_vals) / np.log1p(100.0), 0.0, 1.0)
            final_scores = (ai_scores * 0.65 + odds_scores * 0.08 + apt_scores * 0.06
                            + np.full(num_h, 0.5) * 0.06 + agari_scores * 0.05
                            + course_scores * 0.04 + other_scores * 0.03 + pop_scores * 0.03)
        else:
            final_scores = (ai_scores * 0.70 + pop_scores * 0.06 + apt_scores * 0.06
                            + np.full(num_h, 0.5) * 0.06 + agari_scores * 0.05
                            + course_scores * 0.04 + other_scores * 0.03)

        df['スコア'] = final_scores
        df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
        df = df.sort_values('AI順位')
        cond_key, cond_profile = classify_race_condition(race_info, num_h, is_nar=is_nar)
        return df, cond_key, cond_profile, odds_avail
    except Exception:
        return None


# ===== 複数レース一括予測 =====
with st.expander("🏇 複数レース一括予測（開催日全レース）"):
    batch_col1, batch_col2 = st.columns([2, 1])
    with batch_col1:
        batch_date = st.date_input("開催日を選択", value=datetime.now().date(), key="batch_date")
    with batch_col2:
        batch_type = "JRA（中央）"
    if st.button("📋 レース一覧を取得", key="fetch_batch"):
        date_str = batch_date.strftime('%Y%m%d')
        with st.spinner("レース一覧を取得中..."):
            race_list, fetch_err = fetch_race_list(date_str)
        if race_list:
            st.session_state['batch_races'] = race_list
            st.session_state['batch_is_nar'] = False
            st.session_state.pop('batch_results', None)
            st.success(f"{len(race_list)}レースを取得しました")
        else:
            msg = "レースが見つかりませんでした（開催日を確認してください）"
            if fetch_err:
                msg += f"\n\n詳細: {fetch_err}"
            st.warning(msg)
    # レース選択UI
    if 'batch_races' in st.session_state:
        batch_races = st.session_state['batch_races']
        # 開催場ごとにグルーピング
        _batch_venues = {}
        for r in batch_races:
            v = r.get('course', '不明')
            if v not in _batch_venues:
                _batch_venues[v] = []
            _batch_venues[v].append(r)
        st.markdown(f"**{len(batch_races)}レース検出**（{', '.join(_batch_venues.keys())}）")
        selected_batch = []
        for venue, races_in_venue in _batch_venues.items():
            st.markdown(f"**{venue}**")
            for r in races_in_venue:
                time_str = f" {r['time']}" if r.get('time') else ""
                race_num = r.get('race_num', '')
                label = f"{race_num} {r['race_name'][:20]}{time_str}"
                if st.checkbox(label, value=True, key=f"batch_{r['race_id']}"):
                    selected_batch.append(r)
        if selected_batch and st.button(f"🚀 {len(selected_batch)}レースを一括予測", key="run_batch"):
            st.session_state['batch_results'] = []
            batch_is_nar = st.session_state.get('batch_is_nar', False)
            progress = st.progress(0)
            status_text = st.empty()
            for idx, race in enumerate(selected_batch):
                rid = race['race_id']
                rn_short = race['race_name'][:12]
                course_name = race.get('course', '')
                status_text.markdown(f"⏳ {course_name} {rn_short} を予測中... ({idx+1}/{len(selected_batch)})")
                try:
                    rn, horses, hids, rinfo = parse_shutuba(rid, is_nar=batch_is_nar)
                    if not horses:
                        continue
                    # 障害レース自動除外（モデルは平地専用）
                    if rinfo.get('surface') == '障':
                        status_text.markdown(f"⏭️ {course_name} {rn_short} — 障害レース（スキップ）")
                        continue
                    ro = fetch_realtime_odds(rid, is_nar=batch_is_nar)
                    for h in horses:
                        ub = h.get('馬番', 0)
                        h['単勝オッズ'] = ro.get(ub, 0.0)
                    num_h = len(horses)
                    for hi, (h, hid) in enumerate(zip(horses, hids)):
                        if hid:
                            try:
                                stats = get_horse_stats(hid, rinfo['distance'], rinfo['surface'], rinfo['course'])
                                h['前走着順'] = stats.get('last_finish', 5)
                                h['距離適性'] = stats.get('dist_apt', 0.5)
                                h['馬場適性'] = stats.get('surf_apt', 0.5)
                                h['人気傾向'] = stats.get('pop_score', 0.5)
                                h['コース適性'] = stats.get('course_apt', 0.5)
                                h['前走間隔'] = stats.get('interval_days', 30)
                                h['脚質'] = stats.get('running_style', 0)
                                h['上がり3F'] = stats.get('avg_agari', 35.5)
                                h['複勝率'] = stats.get('fukusho_rate', 0.0)
                                h['父'] = stats.get('father', '')
                                h['母の父'] = stats.get('mother_father', '')
                                h['血統スコア'] = calc_sire_score(stats.get('father',''), rinfo['surface'], rinfo['distance'])
                                h['騎手勝率'] = h.get('騎手勝率', 0.05)
                                h['通過順平均'] = stats.get('avg_pass_pos', 8.0)
                                h['通過順4'] = stats.get('last_pass4', 8)
                                h['前走オッズ'] = stats.get('last_odds', 15.0)
                                h['前走人気'] = stats.get('last_pop', 8)
                                h['所属地'] = stats.get('trainer_loc', '')
                                h['prev2_finish'] = stats.get('prev2_finish', 5)
                                h['prev3_finish'] = stats.get('prev3_finish', 5)
                                h['prev4_finish'] = stats.get('prev4_finish', 5)
                                h['prev5_finish'] = stats.get('prev5_finish', 5)
                                h['avg_finish_3r'] = stats.get('avg_finish_3r', 5.0)
                                h['avg_finish_5r'] = stats.get('avg_finish_5r', 5.0)
                                h['best_finish_3r'] = stats.get('best_finish_3r', 5)
                                h['best_finish_5r'] = stats.get('best_finish_5r', 5)
                                h['top3_count_3r'] = stats.get('top3_count_3r', 0)
                                h['top3_count_5r'] = stats.get('top3_count_5r', 0)
                                h['finish_trend'] = stats.get('finish_trend', 0)
                                h['prev2_last3f'] = stats.get('prev2_last3f', 35.5)
                            except Exception:
                                pass
                        time.sleep(0.3)
                    # モデル予測実行
                    result = _batch_score_race(horses, rinfo, batch_is_nar)
                    if result is not None:
                        scored_df, cond_key, cond_profile, has_odds = result
                        save_prediction(rid, rn, rinfo, scored_df, is_nar=batch_is_nar)
                        top3 = scored_df.head(3)
                        top3_info = []
                        for _, row in top3.iterrows():
                            top3_info.append({
                                'num': int(row['馬番']),
                                'name': row['馬名'],
                                'score': float(row['スコア']),
                            })
                        # 買い目を取得
                        bets, bet_type_used = generate_bets_for_condition(scored_df, cond_key, cond_profile)
                        st.session_state['batch_results'].append({
                            'race_id': rid, 'race_name': rn,
                            'course': rinfo.get('course',''),
                            'distance': rinfo.get('distance', 0),
                            'surface': rinfo.get('surface',''),
                            'condition': rinfo.get('condition', ''),
                            'num_horses': num_h,
                            'race_num': race.get('race_num',''),
                            'has_odds': has_odds,
                            'cond_key': cond_key,
                            'bet_type': cond_profile['bet_type'],
                            'bets': bets,
                            'top3': top3_info,
                        })
                except Exception:
                    pass
                progress.progress((idx + 1) / len(selected_batch))
                time.sleep(0.5)
            progress.empty()
            status_text.empty()
            st.success(f"✅ {len(st.session_state.get('batch_results', []))}レースの予測を完了しました")
            st.rerun()
    # バッチ結果表示
    if st.session_state.get('batch_results'):
        results = st.session_state['batch_results']
        BET_LABELS_B = {'trio': '三連複', 'umaren': '馬連', 'wide': 'ワイド'}
        # サマリー
        n_total = len(results)
        cond_counts = {}
        for r in results:
            ck = r.get('cond_key', '?')
            cond_counts[ck] = cond_counts.get(ck, 0) + 1
        total_bets = sum(len(r.get('bets', [])) for r in results)
        total_inv = total_bets * 100
        cond_str = '　'.join(f"{k}:{v}" for k, v in sorted(cond_counts.items()))
        st.markdown(f"### 一括予測結果 ({n_total}R)")
        st.markdown(f"**条件分布**: {cond_str}　|　**総買い目**: {total_bets}点 (¥{total_inv:,})")
        # 開催場ごとに表示
        _br_venues = {}
        for r in results:
            v = r.get('course', '不明')
            if v not in _br_venues:
                _br_venues[v] = []
            _br_venues[v].append(r)
        for venue, venue_results in _br_venues.items():
            st.markdown(f"#### {venue}")
            for r in venue_results:
                rnum = r.get('race_num', '')
                rname = r.get('race_name', '')
                ck = r.get('cond_key', '?')
                bt = r.get('bet_type', 'trio')
                bets = r.get('bets', [])
                top3 = r.get('top3', [])
                n_h = r.get('num_horses', 0)
                surf = r.get('surface', '')
                dist = r.get('distance', 0)
                cond = r.get('condition', '')
                bt_label = BET_LABELS_B.get(bt, bt)
                inv = len(bets) * 100
                header = f"{rnum} {rname}　{surf}{dist}m {cond} {n_h}頭　[{ck}] {bt_label}{len(bets)}点(¥{inv})"
                with st.expander(header, expanded=False):
                    # AI予測TOP3
                    if top3:
                        top3_html = '<div style="display:flex;gap:12px;margin-bottom:8px;">'
                        rank_colors = {0: '#ffd700', 1: '#c0c0c0', 2: '#cd7f32'}
                        for i, h in enumerate(top3):
                            rc = rank_colors.get(i, '#6a6a80')
                            top3_html += f'<div style="text-align:center;padding:6px 12px;background:rgba(255,255,255,0.04);border-radius:8px;border-left:3px solid {rc};">'
                            top3_html += f'<div style="font-family:Oswald;font-size:1.1em;color:{rc} !important;">{i+1}位</div>'
                            top3_html += f'<div style="font-size:0.9em;">{h["num"]}番 {h["name"]}</div>'
                            top3_html += f'<div style="font-family:Oswald;font-size:0.8em;color:#b0b8c8 !important;">{h["score"]:.3f}</div>'
                            top3_html += '</div>'
                        top3_html += '</div>'
                        st.markdown(top3_html, unsafe_allow_html=True)
                    # 条件情報
                    cond_p = CONDITION_PROFILES.get(ck, {})
                    is_1000m = ck == 'D' and dist <= 1000
                    if is_1000m:
                        st.markdown(f"**条件{ck}**: 1000m以下（非推奨：ROI 85%）　/　**購入非推奨**")
                        st.warning("1000m以下：非推奨（WFバックテスト ROI 85.0%, N=534）")
                    else:
                        st.markdown(f"**条件{ck}**: {cond_p.get('desc', '')}　/　**推奨**: {bt_label}")
                    # 買い目
                    if bets:
                        # フォーメーション構造表示
                        if len(bets) >= 3 and len(bets[0]) == 3:
                            all_nums = sorted(set(n for b in bets for n in b))
                            axis = sorted([n for n in all_nums if all(n in b for b in bets)])
                            if not axis and top3:
                                axis = [top3[0]['num']]
                            col3 = sorted(set(all_nums) - set(axis))
                            axis_names = []
                            for n in axis:
                                name = next((h['name'] for h in top3 if h.get('num') == n), '')
                                axis_names.append(f'{n}番 {name[:5]}' if name else f'{n}番')
                            struct_html = f'<div style="font-size:0.8em;color:#8890a0 !important;margin:2px 0 4px;">'
                            struct_html += f'軸: <span style="color:#f0c040 !important;">{", ".join(axis_names)}</span>'
                            struct_html += f' / 相手: <span style="font-family:Oswald;">{", ".join(str(n) for n in col3)}</span></div>'
                            st.markdown(struct_html, unsafe_allow_html=True)
                        elif len(bets[0]) == 2 and top3:
                            axis_num = top3[0]['num']
                            bt_label = BET_LABELS_B.get(bt, bt)
                            umaren_amts = [400, 300]
                            bet_details = []
                            for bi, b in enumerate(bets):
                                amt = umaren_amts[bi] if bi < len(umaren_amts) else 100
                                bet_details.append(f'{bt_label} {b[0]}-{b[1]}: {amt}円')
                            struct_html = f'<div style="font-size:0.8em;color:#8890a0 !important;margin:2px 0 4px;">'
                            struct_html += f'軸: <span style="color:#f0c040 !important;">{axis_num}番 {top3[0]["name"][:5]}</span>'
                            struct_html += f' / ' + ' / '.join(f'<span style="font-family:Oswald;">{d}</span>' for d in bet_details)
                            struct_html += '</div>'
                            st.markdown(struct_html, unsafe_allow_html=True)
                        bets_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:4px 0;">'
                        for b in bets:
                            bets_html += f'<span style="background:#1a2a3a;padding:3px 8px;border-radius:4px;font-family:Oswald;font-size:0.85em;color:#b0d0f0 !important;">{"  ".join(str(n) for n in sorted(b))}</span>'
                        bets_html += '</div>'
                        st.markdown(bets_html, unsafe_allow_html=True)

# Results update section
with st.expander("📝 レース結果を登録（的中率集計用）"):
    result_url = st.text_input(
        "netkeibaの結果ページURLを貼り付け",
        placeholder="https://race.netkeiba.com/race/result.html?race_id=... または https://db.netkeiba.com/race/XXXX/",
        key="result_url"
    )
    if st.button("結果を取得・保存") and result_url:
        is_nar_result = False
        rid_match = re.search(r'race_id=(\d+)', result_url)
        if not rid_match:
            # db.netkeiba.com/race/XXXX/ 形式のURL対応
            rid_match = re.search(r'/race/(\d{10,12})/?', result_url)
        if not rid_match:
            st.error("URLからrace_idを取得できませんでした。対応URL形式:\n- `https://race.netkeiba.com/race/result.html?race_id=XXXX`\n- `https://db.netkeiba.com/race/XXXX/`")
        else:
            result_race_id = rid_match.group(1)
            st.info(f"race_id: {result_race_id} (JRA)")
            try:
                with st.spinner("レース結果・払戻金を取得中..."):
                    results_dict, result_payouts = fetch_race_results(result_race_id, is_nar=is_nar_result)
            except Exception as e:
                results_dict, result_payouts = {}, {'trio': 0, 'umaren': 0, 'wide': 0}
                st.error(f"結果取得中にエラー: {e}")
            if results_dict:
                st.markdown(f"**{len(results_dict)}頭の着順を取得しました**")
                sorted_results = sorted(results_dict.items(), key=lambda x: x[1])
                preview_df = pd.DataFrame([
                    {"馬番": k, "着順": v} for k, v in sorted_results
                ])
                st.dataframe(preview_df, hide_index=True)
                # DBからbet_typeと買い目を取得
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT trio_bets, wide_bets, umaren_bets, bet_type, bet_condition FROM race_results WHERE race_id = ?", (result_race_id,))
                row = c.fetchone()
                conn.close()
                top2_nums = set(n for n, f in results_dict.items() if f <= 2)
                top3_nums = set(n for n, f in results_dict.items() if f <= 3)
                st.markdown(f"**実際の3着以内:** {'-'.join(str(n) for n in sorted(top3_nums))}")
                if row:
                    trio_bets_raw, wide_bets_raw, umaren_bets_raw, db_bet_type, db_cond = row
                    db_bet_type = db_bet_type or 'trio'
                    # 買い目種別に応じた判定
                    BET_LABELS = {'trio': '三連複7点', 'umaren': '馬連1軸2流し', 'wide': 'ワイド1軸2流し'}
                    bet_label = BET_LABELS.get(db_bet_type, '三連複7点')
                    if db_bet_type == 'trio' and trio_bets_raw:
                        bets = json.loads(trio_bets_raw)
                        target_nums = top3_nums
                        relevant_payout = result_payouts.get('trio', 0)
                    elif db_bet_type == 'umaren' and umaren_bets_raw:
                        bets = json.loads(umaren_bets_raw)
                        target_nums = top2_nums
                        relevant_payout = result_payouts.get('umaren', 0)
                    elif db_bet_type == 'wide' and wide_bets_raw:
                        bets = json.loads(wide_bets_raw)
                        target_nums = top3_nums
                        relevant_payout = result_payouts.get('wide', 0)
                    else:
                        bets = []
                        target_nums = top3_nums
                        relevant_payout = 0
                    if relevant_payout > 0:
                        st.info(f"{bet_label} 払戻金: **{relevant_payout:,}円**")
                    else:
                        st.warning(f"{bet_label}の払戻金を取得できませんでした（払戻金0円として記録）")
                    if bets:
                        hit = any(set(b).issubset(target_nums) for b in bets)
                        cond_info = f" [条件{db_cond}]" if db_cond else ""
                        st.markdown(f"**AI {bet_label}{cond_info}:** {', '.join('-'.join(str(n) for n in b) for b in bets)}")
                        if hit:
                            profit = relevant_payout - INVESTMENT_PER_RACE
                            st.success(f"🎉 的中！ 払戻 {relevant_payout:,}円 - 投資 {INVESTMENT_PER_RACE}円 = **+{profit:,}円**")
                        else:
                            st.error(f"❌ ハズレ（投資 -{INVESTMENT_PER_RACE}円）")
                    else:
                        st.warning("買い目データがDBにありません。")
                else:
                    st.warning("このレースのAI予測データがDBにありません（先に予測を実行してください）。着順のみ記録します。")
                # DB更新
                try:
                    update_actual_results(result_race_id, results_dict, result_payouts)
                    payout_summary = '/'.join(f"{k}:{v:,}円" for k, v in result_payouts.items() if v > 0) or '払戻なし'
                    st.success(f"✅ 結果をDBに保存しました（{len(results_dict)}頭 / {payout_summary}）")
                except Exception as e:
                    st.error(f"DB保存エラー: {e}")
            else:
                st.error("着順データを取得できませんでした。考えられる原因:\n- レースがまだ確定していない\n- URLが正しくない\n- netkeiba接続エラー")

# ===== 実運用成績ダッシュボード =====
with st.expander("📈 実運用成績ダッシュボード"):
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions_log.csv")
    _mc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monte_carlo_results.json")
    _roi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "roi_comparison.csv")

    if os.path.exists(_log_path):
        _log_df = pd.read_csv(_log_path)
        _settled = _log_df[_log_df['result_status'] == 'settled']
        _pending = _log_df[_log_df['result_status'] == 'pending']

        st.markdown(f"**総予測数:** {len(_log_df)} | **確定:** {len(_settled)} | **未確定:** {len(_pending)}")

        if len(_settled) > 0:
            _total_invest = _settled['investment'].sum()
            _total_payout = _settled['payout'].sum()
            _total_hits = (_settled['hit'] == 1).sum()
            _total_roi = _total_payout / _total_invest * 100 if _total_invest > 0 else 0

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("的中率", f"{_total_hits/len(_settled)*100:.1f}%", f"{_total_hits}/{len(_settled)}")
            col_b.metric("投資額", f"{_total_invest:,}円")
            col_c.metric("回収額", f"{_total_payout:,}円")
            _profit = _total_payout - _total_invest
            col_d.metric("ROI", f"{_total_roi:.1f}%", f"{_profit:+,}円")

            # 条件別成績
            st.markdown("#### 条件別成績")
            _cond_data = []
            for _ck in sorted(_settled['cond_key'].unique()):
                _sub = _settled[_settled['cond_key'] == _ck]
                _n = len(_sub)
                _h = (_sub['hit'] == 1).sum()
                _inv = _sub['investment'].sum()
                _pay = _sub['payout'].sum()
                _r = _pay / _inv * 100 if _inv > 0 else 0
                _cond_data.append({'条件': _ck, 'N': _n, '的中': _h,
                                   '的中率': f"{_h/_n*100:.1f}%",
                                   '投資': f"{_inv:,}円", '回収': f"{_pay:,}円",
                                   'ROI': f"{_r:.1f}%"})
            st.dataframe(pd.DataFrame(_cond_data), use_container_width=True, hide_index=True)

            # 日別推移
            if 'predicted_at' in _settled.columns:
                _sc = _settled.copy()
                _sc['date'] = pd.to_datetime(_sc['predicted_at']).dt.date
                _daily = _sc.groupby('date').agg(
                    N=('hit', 'count'),
                    hits=('hit', 'sum'),
                    invest=('investment', 'sum'),
                    payout=('payout', 'sum'),
                ).reset_index()
                _daily['ROI'] = _daily['payout'] / _daily['invest'] * 100
                _daily['cumulative_profit'] = (_daily['payout'] - _daily['invest']).cumsum()

                st.markdown("#### 累積収支推移")
                st.line_chart(_daily.set_index('date')['cumulative_profit'])

        # 予測ログ一覧
        st.markdown("#### 予測ログ一覧")
        _display_cols = ['predicted_at', 'race_name', 'course', 'distance', 'cond_key',
                         'bet_type', 'top1_name', 'result_status', 'hit', 'payout']
        _avail_cols = [c for c in _display_cols if c in _log_df.columns]
        st.dataframe(_log_df[_avail_cols].tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("予測ログがありません。`python predict_and_log.py <URL>` で予測を記録してください。")

    # バックテストROI vs 実配当ROI比較
    _actual_roi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'actual_roi_results.json')
    if os.path.exists(_actual_roi_path):
        st.markdown("---")
        st.markdown("#### 実配当ROI (JRA公式配当データ × WFバックテスト)")
        with open(_actual_roi_path, 'r', encoding='utf-8') as _arf:
            _ar = json.load(_arf)
        st.caption(f"WF 2020-2025 / {_ar.get('matched_races', 0):,}レース / "
                   f"マッチ率 {_ar.get('match_rate', 0):.1f}% / AUC {_ar.get('avg_auc', 0):.4f}")
        _roi_rows = []
        for _ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            _ci = _ar.get('conditions', {}).get(_ck)
            if not _ci:
                continue
            _t = _ci['actual_roi']['trio']
            _u = _ci['actual_roi']['umaren']
            _w = _ci['actual_roi']['wide']
            _et = _ci['estimated_roi']['trio']
            _roi_rows.append({
                '条件': _ck,
                'N': _ci['n'],
                'trio的中': f"{_t['hit_rate']}%",
                'trio実ROI': f"{_t['roi']}%",
                'trio推定ROI': f"{_et['roi']}%",
                'umaren実ROI': f"{_u['roi']}%",
                'wide実ROI': f"{_w['roi']}%",
                '最適券種': _ci['best_bet'],
                '推奨': '○' if _ci['recommended'] else '×',
            })
        st.dataframe(pd.DataFrame(_roi_rows), use_container_width=True, hide_index=True)
    elif os.path.exists(_roi_path):
        st.markdown("---")
        st.markdown("#### バックテスト推定ROI vs 実配当ROI")
        _roi_df = pd.read_csv(_roi_path)
        st.dataframe(_roi_df, use_container_width=True, hide_index=True)

    # モンテカルロシミュレーション結果
    if os.path.exists(_mc_path):
        st.markdown("---")
        st.markdown("#### モンテカルロシミュレーション結果")
        with open(_mc_path, "r", encoding="utf-8") as _mf:
            _mc = json.load(_mf)

        for _fund_key, _sim in _mc.get('simulations', {}).items():
            _fund = _sim['initial_fund']
            st.markdown(f"**初期資金 {_fund:,}円** ({_sim['num_races']}レース×{_sim['num_trials']:,}回)")
            _mc_col1, _mc_col2, _mc_col3, _mc_col4 = st.columns(4)
            _mc_col1.metric("破産確率", f"{_sim['ruin_probability']:.2f}%")
            _mc_col2.metric("利益確率", f"{_sim['profit_probability']:.1f}%")
            _mc_col3.metric("期待最終資金", f"{_sim['avg_final_fund']:,}円")
            _mc_col4.metric("最大DD", f"{_sim['worst_max_drawdown']:.1f}%")
            st.caption(f"95%CI: {_sim['ci95_lower']:,}円 〜 {_sim['ci95_upper']:,}円")

    if not os.path.exists(_mc_path) and not os.path.exists(_roi_path):
        st.info("モンテカルロ結果: `python monte_carlo_sim.py` / ROI検証: `python verify_real_roi.py` を実行してください。")
