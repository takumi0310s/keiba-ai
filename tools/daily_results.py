"""
毎夕の結果照合スクリプト
当日の予測結果と実際のレース結果を照合し、的中判定・ROI計算を行う。

データソース:
  --source csv : daily_predict CSV (data/daily_predictions/) から読み込み（デフォルト）
  --source db  : Streamlit DB (keiba_predictions.db) から読み込み

結果出力:
  - data/daily_results/{date}.csv        — 日別結果
  - data/cumulative_results.csv          — 累積結果
  - data/track_record.csv               — Streamlit Cloud同期用（git管理）
  - data/actual_roi_results.json         — ROI統計（--recalc-roiで再計算）

Usage:
    python tools/daily_results.py                     # 今日の結果（CSV参照）
    python tools/daily_results.py --date 20260315     # 日付指定
    python tools/daily_results.py --source db         # Streamlit DB参照
    python tools/daily_results.py --backfill-all      # 全予測CSVをバックフィル
    python tools/daily_results.py --backfill 20260411,20260418
    python tools/daily_results.py --recalc-roi        # actual_roi_results.json再計算
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
import json
import sqlite3
import argparse
import time
import unicodedata
import glob
from datetime import datetime

try:
    _tools_dir = os.path.dirname(os.path.abspath(__file__))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from build_allscores_html import update_allscores_result, build_allscores_html
    _ALLSCORES_HTML_AVAILABLE = True
except ImportError:
    _ALLSCORES_HTML_AVAILABLE = False


def _norm(text):
    """NFKC正規化 — 全角数字・全角英字を半角に統一し、文字化け耐性を持たせる"""
    if text is None:
        return ''
    return unicodedata.normalize('NFKC', text)

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# === 定数 ===
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
INVESTMENT_PER_RACE = 700
STREAMLIT_DB_PATHS = [
    os.path.join(BASE_DIR, "keiba_predictions.db"),    # app.pyのメインDB
    os.path.join(BASE_DIR, "keiba_race_results.db"),   # 予備DB
]
TRACK_RECORD_CSV = os.path.join(BASE_DIR, "data", "track_record.csv")
CUMUL_CSV = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
ROI_JSON = os.path.join(BASE_DIR, "data", "actual_roi_results.json")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "daily_predictions")

# 障害レース/特殊距離の除外判定
EXCLUDED_DISTANCE_MIN = 1000  # 1000m以下は購入非推奨（予測CSVに含まれる場合もスキップ）


def fetch_race_result(race_id):
    """netkeibaからレース結果を取得"""
    result = {
        'finish_order': {},
        'payouts': {'trio': 0, 'umaren': 0, 'wide': 0, 'tansho': 0},
        'trio_nums': None,
        'umaren_nums': None,
    }
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        # netkeibaはページによってEUC-JP/UTF-8混在 — meta宣言を優先、だめなら apparent_encoding
        detected = None
        head = resp.content[:4096].decode('ascii', errors='ignore').lower()
        m_charset = re.search(r'charset=["\']?([\w\-]+)', head)
        if m_charset:
            detected = m_charset.group(1)
        resp.encoding = detected or resp.apparent_encoding or "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  [ERROR] 結果ページ取得失敗: {e}")
        return None

    result_table = soup.find("table", class_="RaceTable01")
    if not result_table:
        result_table = soup.find("table", class_="race_table_01")
    if not result_table:
        print(f"  [WARN] 結果テーブルが見つかりません（レース未確定の可能性）")
        return None

    top3_nums = []
    top2_nums = []
    for row in result_table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 4:
            continue
        finish_text = tds[0].get_text(strip=True)
        if not finish_text.isdigit():
            continue
        finish = int(finish_text)
        umaban = 0
        for td in tds:
            cls = " ".join(td.get("class", []))
            if "Txt_C" in cls:
                t = td.get_text(strip=True)
                if t.isdigit() and 1 <= int(t) <= 18:
                    umaban = int(t)
                    break
        if umaban == 0:
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "Umaban" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
        if umaban == 0 and len(tds) >= 3:
            t = tds[2].get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 18:
                umaban = int(t)
        if umaban > 0:
            result['finish_order'][umaban] = finish
            if finish <= 3:
                top3_nums.append((finish, umaban))
            if finish <= 2:
                top2_nums.append((finish, umaban))

    top3_nums.sort()
    top2_nums.sort()
    if len(top3_nums) >= 3:
        result['trio_nums'] = sorted([n for _, n in top3_nums[:3]])
    if len(top2_nums) >= 2:
        result['umaren_nums'] = sorted([n for _, n in top2_nums[:2]])

    # 払戻金テーブルの解析
    BET_TYPE_HEX = {
        'tansho':  'e58d98e58b9d',
        'umaren':  'e9a6ace980a3',
        'umatan':  'e9a6ace58d98',
        'wide':    'e383afe382a4e38389',
        'trio_g':  '33e980a3e8a487',
        'tierce_g':'33e980a3e58d98',
    }

    payout_tables = soup.find_all("table", class_="Payout_Detail_Table")
    if not payout_tables:
        payout_tables = soup.find_all("table", class_="pay_table_01")

    for pt in payout_tables:
        for row in pt.find_all("tr"):
            th = row.find("th")
            if not th:
                continue
            th_text = _norm(th.get_text(strip=True))
            th_hex = th_text.encode('utf-8').hex()

            tds = row.find_all("td")
            payout_vals = []
            for td in tds:
                td_text = _norm(td.get_text(strip=True))
                for m in re.finditer(r'([\d,]+)円', td_text):
                    payout_vals.append(int(m.group(1).replace(',', '')))

            if not payout_vals:
                continue
            payout_val = payout_vals[0]

            # NFKC正規化済みテキストで判定（三連複/3連複/３連複の全バリエーション対応）
            is_tansho = ('単勝' in th_text) and ('連' not in th_text)
            is_trio   = ('三連複' in th_text) or ('3連複' in th_text) or (BET_TYPE_HEX['trio_g'] in th_hex)
            is_tierce = ('三連単' in th_text) or ('3連単' in th_text)
            is_umaren = ('馬連' in th_text) and ('三' not in th_text) and ('単' not in th_text) and ('3' not in th_text)
            is_wide   = ('ワイド' in th_text) or (BET_TYPE_HEX['wide'] in th_hex)

            if is_tansho:
                result['payouts']['tansho'] = payout_val
            elif is_trio:
                result['payouts']['trio'] = payout_val
            elif is_umaren:
                result['payouts']['umaren'] = payout_val
            elif is_wide:
                if result['payouts']['wide'] == 0:
                    result['payouts']['wide'] = payout_val
            elif is_tierce:
                pass  # 三連単は使わない

    return result


def check_trio_hit(bets_str, actual_trio_nums):
    """三連複的中判定。各買い目 "1-2-3" 形式、セミコロン区切り。"""
    if not bets_str or not actual_trio_nums:
        return False, None
    actual_set = set(actual_trio_nums)
    bets = [b.strip() for b in str(bets_str).split(";") if b.strip()]
    for bet in bets:
        try:
            nums = [int(n) for n in bet.split("-")]
        except ValueError:
            continue
        if len(nums) == 3 and set(nums) == actual_set:
            return True, bet
    return False, None


def check_umaren_hit(bets_str, actual_umaren_nums):
    """馬連的中判定。各買い目 "1-2" 形式、セミコロン区切り。"""
    if not bets_str or not actual_umaren_nums:
        return False, None
    actual_set = set(actual_umaren_nums)
    bets = [b.strip() for b in str(bets_str).split(";") if b.strip()]
    for bet in bets:
        try:
            nums = [int(n) for n in bet.split("-")]
        except ValueError:
            continue
        if len(nums) == 2 and set(nums) == actual_set:
            return True, bet
    return False, None


# ===== データソース読み込み =====

def load_predictions_from_db(date_str):
    """Streamlit DB から予測データを読み込む"""
    race_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    for db_path in STREAMLIT_DB_PATHS:
        if not os.path.exists(db_path):
            continue
        result = _try_load_from_db(db_path, race_date)
        if result:
            db_name = os.path.basename(db_path)
            print(f"  [DB] {db_name} から {len(result)}レース分の予測を読み込み")
            return result

    return None


def _try_load_from_db(db_path, race_date):
    """指定DBから予測データを読み込む"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT rr.race_id, rr.race_name, rr.bet_condition, rr.bet_type,
                   rr.trio_bets, rr.umaren_bets, rr.num_horses, rr.buy_recommended
            FROM race_results rr
            WHERE rr.predicted_at LIKE ?
            ORDER BY rr.race_id
        """, (f"{race_date}%",))
        rr_rows = c.fetchall()

        if not rr_rows:
            conn.close()
            return None

        c.execute("""
            SELECT race_id, horse_num, ai_rank
            FROM predictions
            WHERE race_date = ? AND ai_rank <= 6
            ORDER BY race_id, ai_rank
        """, (race_date,))
        pred_rows = c.fetchall()
        conn.close()

        top_nums = {}
        for pr in pred_rows:
            rid = str(pr['race_id'])
            if rid not in top_nums:
                top_nums[rid] = {}
            top_nums[rid][pr['ai_rank']] = pr['horse_num']

        predictions = []
        for rr in rr_rows:
            rid = str(rr['race_id'])
            trio_bets_raw = rr['trio_bets']
            umaren_bets_raw = rr['umaren_bets']
            bet_type = rr['bet_type'] or 'trio'
            condition = rr['bet_condition'] or ''

            if bet_type == 'trio' and trio_bets_raw:
                bets_list = json.loads(trio_bets_raw)
                bets_str = '; '.join(
                    '-'.join(str(n) for n in sorted(b)) for b in bets_list
                )
            elif bet_type == 'umaren' and umaren_bets_raw:
                bets_list = json.loads(umaren_bets_raw)
                bets_str = '; '.join(
                    '-'.join(str(n) for n in sorted(b)) for b in bets_list
                )
            else:
                bets_str = ''

            race_name = rr['race_name'] or ''
            course = re.sub(r'\d+R$', '', race_name)
            race_num_m = re.search(r'(\d+)R$', race_name)
            race_num = int(race_num_m.group(1)) if race_num_m else 0

            tops = top_nums.get(rid, {})

            predictions.append({
                'race_id': rid,
                'course': course,
                'race_num': race_num,
                'race_name': race_name,
                'condition': condition,
                'trio_bets': bets_str,
                'bet_type': bet_type,
                'investment': INVESTMENT_PER_RACE,
                'top1_num': tops.get(1, 0),
                'top2_num': tops.get(2, 0),
                'top3_num': tops.get(3, 0),
                'distance': 0,
                'surface': '',
            })

        return predictions

    except Exception as e:
        print(f"  [WARN] DB読み込みエラー ({os.path.basename(db_path)}): {e}")
        return None


def _parse_should_bet(val) -> bool:
    """CSV 由来の should_bet を bool に頑健変換。 旧CSV(欠損/NaN)は買い扱い(True)。

    pandas read_csv は bool 列を文字列 'True'/'False' で読むことがあるため両対応。
    """
    if isinstance(val, bool):
        return val
    if val is None:
        return True
    try:
        # NaN 判定 (float('nan') != 自身)
        if isinstance(val, float) and val != val:
            return True
    except Exception:
        pass
    s = str(val).strip().lower()
    if s in ('false', '0', '0.0', 'no', 'n', ''):
        return False
    return True


def _compute_bet_buckets(settled):
    """settled 結果を 3区分に集計 (2026-05-29)。

    (A) 実投票     = should_bet=True  (戦略⑦通過、 実際に賭けるレース)
    (B) 見送りｼｬﾄﾞｰ = should_bet=False (フィルタ除外を「もし買っていたら」)
    (C) 全部買い   = 全 settled       (全平地を「全部買っていたら」)
    ★ should_bet で分割するだけ。 実投票額・既存集計は不変 (A+B=C) ★
    """
    def _agg(rows):
        n = len(rows)
        inv = sum(r.get('investment', 0) for r in rows)
        pay = sum(r.get('trio_payout', 0) + r.get('umaren_payout', 0) for r in rows)
        hits = sum(1 for r in rows if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1)
        return {
            'n': n, 'hits': hits, 'inv': inv, 'payout': pay,
            'profit': pay - inv,
            'hit_rate': (hits / n * 100) if n > 0 else 0.0,
            'roi': (pay / inv * 100) if inv > 0 else 0.0,
        }
    buy = [r for r in settled if r.get('should_bet', True)]
    skip = [r for r in settled if not r.get('should_bet', True)]
    return {'actual': _agg(buy), 'shadow': _agg(skip), 'all': _agg(settled)}


def _format_buckets_text(b):
    """3区分を Discord/console 用テキストに整形。"""
    def _line(label, s):
        sign = '+' if s['profit'] >= 0 else ''
        return (f"{label}: {s['hits']}/{s['n']}的中 ({s['hit_rate']:.0f}%) "
                f"ROI {s['roi']:.0f}% 収支{sign}{s['profit']:,.0f}円")
    return "\n".join([
        "📊 買っていたら的中率 (3区分)",
        _line("(A)実投票", b['actual']),
        _line("(B)見送りｼｬﾄﾞｰ", b['shadow']),
        _line("(C)全部買い", b['all']),
    ])


def _format_buckets_html(b):
    """3区分を全馬HTML 先頭の summary block (HTML) に整形。"""
    def _row(label, s):
        sign = '+' if s['profit'] >= 0 else ''
        col = '#90ee90' if s['profit'] >= 0 else '#ee9090'
        return (f"<tr><td style='text-align:left;padding:2px 10px'>{label}</td>"
                f"<td style='padding:2px 10px'>{s['hits']}/{s['n']}</td>"
                f"<td style='padding:2px 10px'>{s['hit_rate']:.0f}%</td>"
                f"<td style='padding:2px 10px'>{s['roi']:.0f}%</td>"
                f"<td style='padding:2px 10px;color:{col}'>{sign}{s['profit']:,.0f}円</td></tr>")
    return (
        "<div style='margin:8px 0 16px;font-size:12px;background:#1a1a1a;"
        "padding:8px 10px;border-left:3px solid #555'>"
        "<div style='color:#bbb;font-weight:bold;margin-bottom:4px'>"
        "買っていたら的中率 (3区分)</div>"
        "<table style='border-collapse:collapse'>"
        "<thead><tr style='color:#888'>"
        "<th style='text-align:left;padding:2px 10px'>区分</th>"
        "<th style='padding:2px 10px'>的中</th><th style='padding:2px 10px'>的中率</th>"
        "<th style='padding:2px 10px'>ROI</th><th style='padding:2px 10px'>収支</th></tr></thead>"
        "<tbody>"
        + _row("🟢(A) 実投票 (買いのみ)", b['actual'])
        + _row("⚪(B) 見送りｼｬﾄﾞｰ", b['shadow'])
        + _row("(C) 全部買い (全平地)", b['all'])
        + "</tbody></table>"
        "<div style='color:#666;font-size:10px;margin-top:3px'>"
        "(A)実投票=戦略⑦通過の実際の買い目 / (B)見送り=フィルタ除外を仮に買った場合 / "
        "(C)=全平地全部買い。 ※1000m以下・障害は集計対象外。 (A)vs(C)でフィルタ妥当性を検証</div></div>"
    )


def load_predictions_from_csv(date_str):
    """daily_predict CSVから予測データを読み込む"""
    pred_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.csv")
    if not os.path.exists(pred_path):
        return None

    df_pred = pd.read_csv(pred_path, encoding='utf-8-sig')
    predictions = []
    for _, row in df_pred.iterrows():
        distance = int(row.get('distance', 0) or 0)
        surface = str(row.get('surface', '') or '')
        # 1000m以下は除外（ROI 85%で購入非推奨）
        if 0 < distance <= EXCLUDED_DISTANCE_MIN:
            continue
        # 障害は surface に "障" が含まれる想定（予測CSVにはそもそも含まれないはず）
        if '障' in surface:
            continue
        predictions.append({
            'race_id': str(row['race_id']),
            'course': row.get('course', ''),
            'race_num': int(row.get('race_num', 0) or 0),
            'race_name': row.get('race_name', ''),
            'condition': row.get('condition', ''),
            'trio_bets': row.get('trio_bets', ''),
            'bet_type': row.get('bet_type', 'trio'),
            'investment': int(row.get('investment', INVESTMENT_PER_RACE) or INVESTMENT_PER_RACE),
            'top1_num': int(row.get('top1_num', 0) or 0),
            'top1_name': str(row.get('top1_name', '') or ''),
            'top1_score': float(row.get('top1_score', 0.0) or 0.0),
            'top2_num': int(row.get('top2_num', 0) or 0),
            'top2_name': str(row.get('top2_name', '') or ''),
            'top2_score': float(row.get('top2_score', 0.0) or 0.0),
            'top3_num': int(row.get('top3_num', 0) or 0),
            'top3_name': str(row.get('top3_name', '') or ''),
            'top3_score': float(row.get('top3_score', 0.0) or 0.0),
            'distance': distance,
            'surface': surface,
            # 買い/見送り判定 (戦略⑦、 2026-05-29)。 旧CSV(列なし)は買い扱い(True)。
            'should_bet': _parse_should_bet(row.get('should_bet', True)),
            'skip_reason': ('' if pd.isna(row.get('skip_reason'))
                            else str(row.get('skip_reason') or '')),
        })
    return predictions


# ===== 結果同期用CSV書き出し =====

def save_track_record_csv(results, date_str):
    """track_record.csv に結果を追記（Streamlit Cloud同期用）"""
    settled = [r for r in results if r.get('status') == 'settled']
    if not settled:
        return

    rows = []
    for r in settled:
        rows.append({
            'date': date_str,
            'race_id': r['race_id'],
            'course': r['course'],
            'race_num': r['race_num'],
            'race_name': r['race_name'],
            'condition': r['condition'],
            'bet_type': r.get('bet_type', 'trio'),
            'trio_bets': r.get('trio_bets_str', ''),
            'trio_result': r.get('trio_result', ''),
            'hit': 1 if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1 else 0,
            'payout': r.get('trio_payout', 0) + r.get('umaren_payout', 0),
            'investment': r['investment'],
            'profit': r['profit'],
        })

    df_new = pd.DataFrame(rows)

    if os.path.exists(TRACK_RECORD_CSV):
        df_existing = pd.read_csv(TRACK_RECORD_CSV, encoding='utf-8-sig')
        # 同日分は上書き（date + race_id）
        df_existing['date'] = df_existing['date'].astype(str)
        df_existing['race_id'] = df_existing['race_id'].astype(str)
        keep_mask = ~df_existing.set_index(['date','race_id']).index.isin(
            df_new.set_index(['date','race_id']).index
        )
        df_existing = df_existing[keep_mask]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(TRACK_RECORD_CSV, index=False, encoding='utf-8-sig')
    print(f"TRACK RECORD CSV更新: {TRACK_RECORD_CSV}")


def _upsert_cumulative(results, date_str):
    """cumulative_results.csv を (date, race_id) キーでupsert。

    既存の settled 行は上書き対象外（結果は変わらない）、
    pending 行は新規 settled で上書きされる。
    重複排除も同時に実施（これまでの重複データを片付ける）。
    """
    df_new = pd.DataFrame(results)
    df_new['date'] = date_str
    df_new['date'] = df_new['date'].astype(str)
    df_new['race_id'] = df_new['race_id'].astype(str)

    if os.path.exists(CUMUL_CSV):
        df_cumul = pd.read_csv(CUMUL_CSV, encoding='utf-8-sig')
        df_cumul['date'] = df_cumul['date'].astype(str)
        df_cumul['race_id'] = df_cumul['race_id'].astype(str)

        # 全体の重複排除（settled優先）
        df_cumul['_settled_rank'] = (df_cumul['status'] == 'settled').astype(int)
        df_cumul = (df_cumul.sort_values('_settled_rank', ascending=False)
                            .drop_duplicates(subset=['date','race_id'], keep='first')
                            .drop(columns=['_settled_rank']))

        # 新規分のキーを既存から除去
        new_keys = set(zip(df_new['date'], df_new['race_id']))
        mask = df_cumul.apply(lambda r: (r['date'], r['race_id']) not in new_keys, axis=1)
        df_cumul = df_cumul[mask]

        df_out = pd.concat([df_cumul, df_new], ignore_index=True)
    else:
        df_out = df_new

    # 最終デデュプ（念のため）
    df_out['_settled_rank'] = (df_out['status'] == 'settled').astype(int)
    df_out = (df_out.sort_values(['date','race_id','_settled_rank'], ascending=[True,True,False])
                    .drop_duplicates(subset=['date','race_id'], keep='first')
                    .drop(columns=['_settled_rank']))
    df_out = df_out.sort_values(['date','race_id']).reset_index(drop=True)
    df_out.to_csv(CUMUL_CSV, index=False, encoding='utf-8-sig')
    return df_out


def _load_settled_keys():
    """cumulative_results.csv から既に settled の (date, race_id) キーセットを返す。"""
    if not os.path.exists(CUMUL_CSV):
        return set()
    df = pd.read_csv(CUMUL_CSV, encoding='utf-8-sig')
    if 'status' not in df.columns:
        return set()
    s = df[df['status'] == 'settled'].copy()
    s['date'] = s['date'].astype(str)
    s['race_id'] = s['race_id'].astype(str)
    return set(zip(s['date'], s['race_id']))


# ===== メイン処理 =====

def run_daily_results(date_str, source='csv', skip_settled=False):
    """指定日の結果照合。skip_settledがTrueなら全races settledの場合スキップ。"""
    print(f"{'=' * 60}")
    print(f"KEIBA AI 結果照合 - {date_str}")
    print(f"{'=' * 60}")

    predictions = None
    pred_source = ''

    if source == 'csv':
        predictions = load_predictions_from_csv(date_str)
        if predictions:
            pred_source = 'daily_predict CSV'
        else:
            pred_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.csv")
            print(f"[WARN] 予測CSVが見つかりません: {pred_path}")
            # DBフォールバック
            predictions = load_predictions_from_db(date_str)
            if predictions:
                pred_source = 'Streamlit DB (fallback)'
            else:
                print(f"[ERROR] CSV/DB両方で予測データなし")
                return 0
    elif source == 'db':
        predictions = load_predictions_from_db(date_str)
        if predictions:
            pred_source = 'Streamlit DB'
        else:
            print(f"[ERROR] Streamlit DBに {date_str} の予測データがありません")
            for p in STREAMLIT_DB_PATHS:
                exists = "存在" if os.path.exists(p) else "なし"
                print(f"  {os.path.basename(p)}: {exists}")
            return 0

    print(f"\n予測ソース: {pred_source}")
    print(f"予測レース数: {len(predictions)}")

    # idempotency: 既にsettled済みレースをスキップ
    settled_keys = _load_settled_keys()
    to_process = [p for p in predictions
                  if (date_str, str(p['race_id'])) not in settled_keys]
    skipped = len(predictions) - len(to_process)
    if skipped:
        print(f"既にsettled済み: {skipped}R（スキップ）")

    if skip_settled and len(to_process) == 0:
        print(f"[SKIP] {date_str}: 全レース処理済み")
        return 0

    results = []
    full_payout_dump = []  # s2b paper用 全払戻ダンプ(I/Oのみ・本番集計に無影響)
    for idx, row in enumerate(to_process):
        race_id = row['race_id']
        course = row['course']
        race_num = row['race_num']
        race_name = row['race_name']
        condition = row['condition']
        bets_str = row['trio_bets']
        bet_type = row['bet_type']
        investment = row['investment']
        # 買い/見送り判定 (戦略⑦、 3区分集計用)。 旧予測(列なし)は買い扱い。
        should_bet = bool(row.get('should_bet', True))
        skip_reason = str(row.get('skip_reason', '') or '')

        print(f"\n[{idx+1}/{len(to_process)}] {course} {race_num}R {race_name} (ID={race_id})")

        race_result = fetch_race_result(race_id)
        time.sleep(1.2)

        if race_result is None:
            print(f"  結果未確定")
            results.append({
                'race_id': race_id, 'course': course, 'race_num': race_num,
                'race_name': race_name, 'condition': condition,
                'trio_hit': None, 'trio_payout': 0,
                'umaren_hit': None, 'umaren_payout': 0,
                'investment': investment, 'profit': -investment,
                'status': 'pending', 'bet_type': bet_type,
                'trio_bets_str': bets_str,
                'distance': row.get('distance', 0),
                'surface': row.get('surface', ''),
                'should_bet': should_bet, 'skip_reason': skip_reason,
            })
            continue

        finish_order = race_result['finish_order']
        payouts = race_result['payouts']
        trio_nums = race_result['trio_nums']
        umaren_nums = race_result['umaren_nums']

        # ===== s2b paper用 全払戻ダンプ (I/Oのみ・本番の集計/CSV/累積に一切影響しない) =====
        # 本番が既に netkeiba から取得した結果(着順+払戻)を保存するだけ。 s2bが新規アクセス無しで照合できる。
        # ★失敗しても本番結果集計は通常続行(末尾でtry/except)★
        try:
            full_payout_dump.append({
                'race_id': str(race_id), 'course': str(course), 'race_num': int(race_num),
                'finish_order': {int(k): int(v) for k, v in finish_order.items()},
                'payouts': {k: int(v) for k, v in payouts.items()},
                'trio_nums': [int(n) for n in trio_nums] if trio_nums else [],
                'umaren_nums': [int(n) for n in umaren_nums] if umaren_nums else [],
            })
        except Exception:
            pass

        trio_hit = False
        trio_payout = 0
        umaren_hit = False
        umaren_payout = 0
        hit_combo = None

        if bet_type == 'trio' and trio_nums:
            trio_hit, hit_combo = check_trio_hit(bets_str, trio_nums)
            if trio_hit:
                trio_payout = payouts.get('trio', 0)
        elif bet_type == 'umaren':
            # bets_str で直接判定（"1-2; 1-3" 等）
            umaren_hit, hit_combo = check_umaren_hit(bets_str, umaren_nums)
            # top1_num/top2_num/top3_num ベースでもフォールバック判定
            if not umaren_hit and umaren_nums:
                top1_num = row.get('top1_num', 0)
                top2_num = row.get('top2_num', 0)
                top3_num = row.get('top3_num', 0)
                actual_set = set(umaren_nums)
                for pair in [set([top1_num, top2_num]), set([top1_num, top3_num])]:
                    if len(pair) == 2 and pair == actual_set:
                        umaren_hit = True
                        hit_combo = '-'.join(str(n) for n in sorted(pair))
                        break
            if umaren_hit:
                umaren_payout = payouts.get('umaren', 0)

        actual_payout = trio_payout if trio_hit else (umaren_payout if umaren_hit else 0)
        profit = actual_payout - investment

        top1_finish = finish_order.get(row.get('top1_num', 0), '-')
        top2_finish = finish_order.get(row.get('top2_num', 0), '-')
        top3_finish = finish_order.get(row.get('top3_num', 0), '-')

        result_row = {
            'race_id': race_id, 'course': course, 'race_num': race_num,
            'race_name': race_name, 'condition': condition,
            'bet_type': bet_type,
            'trio_bets_str': bets_str,
            'trio_hit': 1 if trio_hit else 0,
            'trio_payout': trio_payout,
            'umaren_hit': 1 if umaren_hit else 0,
            'umaren_payout': umaren_payout,
            'actual_payout': actual_payout,
            'investment': investment,
            'profit': profit,
            'status': 'settled',
            'should_bet': should_bet,
            'skip_reason': skip_reason,
            'top1_num': row.get('top1_num', 0),
            'top1_name': row.get('top1_name', ''),
            'top1_score': row.get('top1_score', 0.0),
            'top2_num': row.get('top2_num', 0),
            'top2_name': row.get('top2_name', ''),
            'top2_score': row.get('top2_score', 0.0),
            'top3_num': row.get('top3_num', 0),
            'top3_name': row.get('top3_name', ''),
            'top3_score': row.get('top3_score', 0.0),
            'top1_finish': top1_finish,
            'top2_finish': top2_finish,
            'top3_finish': top3_finish,
            'trio_result': '-'.join(str(n) for n in trio_nums) if trio_nums else '',
            'distance': row.get('distance', 0),
            'surface': row.get('surface', ''),
        }
        results.append(result_row)

        # allscores HTML: 着順を JSON に追記
        if _ALLSCORES_HTML_AVAILABLE and finish_order:
            try:
                update_allscores_result(date_str, race_id, finish_order)
            except Exception as _ae:
                print(f"  [allscores] update failed: {_ae}")

        hit_mark = "HIT!" if (trio_hit or umaren_hit) else "miss"
        payout_disp = f"払戻 {actual_payout:,}円" if actual_payout > 0 else ""
        # HITなのにpayout=0はpayoutテーブル解析失敗の可能性 → 警告 + Discord通知
        if (trio_hit or umaren_hit) and actual_payout == 0:
            print(f"  [ALERT] HITだがpayout=0 race_id={race_id} ({hit_combo}). Payoutテーブル解析要調査")
            try:
                from notify import send_discord
                send_discord(
                    '🔴 CRITICAL payout=0 検知',
                    (f'race_id={race_id} ({course} {race_num}R)\n'
                     f'的中: {hit_combo}\n'
                     f'trio_payout={trio_payout} umaren_payout={umaren_payout}\n'
                     f'payout テーブル解析失敗の可能性。手動再取得を検討。'),
                    color='red', channel='updates',
                )
            except Exception:
                pass
        print(f"  結果: {hit_mark} {payout_disp}")
        print(f"  三連複結果: {result_row['trio_result']}")
        print(f"  AI TOP3 着順: {top1_finish}着/{top2_finish}着/{top3_finish}着")
        if trio_hit or umaren_hit:
            print(f"  的中組合せ: {hit_combo}")

    if not results:
        print(f"\n[INFO] {date_str}: 処理対象なし")
        return 0

    # 日別CSV
    out_dir = os.path.join(BASE_DIR, "data", "daily_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.csv")
    df_results = pd.DataFrame(results)
    # 日別CSVは同日分だけ保持
    df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n保存先: {out_path}")

    # s2b paper用 全払戻JSON (I/Oのみ・本番に無影響・netkeiba新規アクセスなし)
    try:
        if full_payout_dump:
            fp_dir = os.path.join(BASE_DIR, "data", "daily_results_full")
            os.makedirs(fp_dir, exist_ok=True)
            fp_path = os.path.join(fp_dir, f"{date_str}.json")
            existing = {}
            if os.path.exists(fp_path):
                try:
                    for r in json.load(open(fp_path, encoding='utf-8')):
                        existing[(r.get('race_id'))] = r
                except Exception:
                    existing = {}
            for r in full_payout_dump:
                existing[r['race_id']] = r  # 再照合時は最新で上書き
            with open(fp_path, 'w', encoding='utf-8') as f:
                json.dump(list(existing.values()), f, ensure_ascii=False)
            print(f"s2b払戻ダンプ: {fp_path} ({len(existing)}R)")
    except Exception as _fpe:
        print(f"[s2b払戻ダンプ] skip (本番集計は通常続行): {_fpe}")

    # 累積CSV（upsert + dedup）
    _upsert_cumulative(results, date_str)
    print(f"累積結果更新: {CUMUL_CSV}")

    # track_record.csv（Streamlit Cloud同期用）
    save_track_record_csv(results, date_str)

    # サマリー出力
    settled = [r for r in results if r.get('status') == 'settled']
    pending = [r for r in results if r.get('status') == 'pending']
    if settled:
        total_inv = sum(r['investment'] for r in settled)
        total_payout = sum(r.get('trio_payout', 0) + r.get('umaren_payout', 0) for r in settled)
        total_profit = total_payout - total_inv
        hit_count = sum(1 for r in settled if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1)
        roi = (total_payout / total_inv * 100) if total_inv > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"  日次サマリー: {date_str}")
        print(f"{'=' * 60}")
        print(f"  予測ソース: {pred_source}")
        print(f"  対象レース: {len(settled)}R (未確定: {len(pending)}R)")
        print(f"  的中: {hit_count}/{len(settled)} ({hit_count/len(settled)*100:.1f}%)")
        print(f"  投資: {total_inv:,}円")
        print(f"  払戻: {total_payout:,}円")
        profit_sign = '+' if total_profit >= 0 else ''
        print(f"  収支: {profit_sign}{total_profit:,}円")
        print(f"  ROI: {roi:.1f}%")

        cond_stats = {}
        for r in settled:
            c = r.get('condition', 'X')
            if c not in cond_stats:
                cond_stats[c] = {'count': 0, 'hit': 0, 'inv': 0, 'pay': 0}
            cond_stats[c]['count'] += 1
            if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1:
                cond_stats[c]['hit'] += 1
            cond_stats[c]['inv'] += r['investment']
            cond_stats[c]['pay'] += r.get('trio_payout', 0) + r.get('umaren_payout', 0)

        print(f"\n  条件別:")
        for c in sorted(cond_stats.keys()):
            s = cond_stats[c]
            c_roi = (s['pay'] / s['inv'] * 100) if s['inv'] > 0 else 0
            print(f"    {c}: {s['hit']}/{s['count']}的中 ROI {c_roi:.1f}%")

        # 3区分集計 (実投票 / 見送りシャドー / 全部買い) — 2026-05-29
        _buckets = _compute_bet_buckets(settled)
        print(f"\n  買っていたら的中率 (3区分):")
        for _k, _lab in (('actual', '(A)実投票  '), ('shadow', '(B)見送りｼｬﾄﾞｰ'), ('all', '(C)全部買い')):
            _s = _buckets[_k]
            _sg = '+' if _s['profit'] >= 0 else ''
            print(f"    {_lab}: {_s['hits']}/{_s['n']}的中 ({_s['hit_rate']:.0f}%) "
                  f"ROI {_s['roi']:.0f}% 収支{_sg}{_s['profit']:,.0f}円")
        print(f"  → (A)実投票 vs (C)全部買い でフィルタ妥当性を検証")
        print(f"{'=' * 60}")

    # 全馬スコア HTML を結果反映版で再生成 → Discord 添付
    if _ALLSCORES_HTML_AVAILABLE and settled:
        try:
            # 3区分サマリーを HTML 先頭に埋め込む (2026-05-29)
            _buckets_html = ""
            try:
                _buckets_html = _format_buckets_html(_compute_bet_buckets(settled))
            except Exception as _bh_err:
                print(f"[WARN] 3区分HTML生成skip: {_bh_err}")
            html_path = build_allscores_html(date_str, summary_block=_buckets_html)
            if html_path:
                n_hit = sum(1 for r in settled if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1)
                total_inv = sum(r['investment'] for r in settled)
                total_pay = sum(r.get('trio_payout', 0) + r.get('umaren_payout', 0) for r in settled)
                roi_val = (total_pay / total_inv * 100) if total_inv > 0 else 0
                profit_val = total_pay - total_inv
                p_sign = '+' if profit_val >= 0 else ''
                # 3区分テキストを caption にも付与
                _buckets_txt = ""
                try:
                    _buckets_txt = "\n\n" + _format_buckets_text(_compute_bet_buckets(settled))
                except Exception:
                    pass
                summary = (
                    f"**{date_str}** 結果反映 {len(settled)}R / 的中 {n_hit}R\n"
                    f"ROI {roi_val:.1f}% / 収支 {p_sign}{profit_val:,}円\n"
                    f"添付 HTML で全馬の予測順位 vs 実着順を確認できます"
                    f"{_buckets_txt}"
                )
                from notify import send_discord_file
                ok = send_discord_file(
                    f"全馬照合 {date_str} ({n_hit}/{len(settled)} 的中)",
                    summary,
                    filepath=html_path,
                    color="green" if n_hit > 0 else "gray",
                    channel="bets",
                )
                print(f"[daily_results] 全馬照合 HTML 送信: {'OK' if ok else 'FAILED'}")
        except Exception as _he:
            print(f"[WARN] 全馬照合 HTML 送信失敗: {_he}")

    return len(settled)


# ===== バックフィル =====

def run_backfill(dates, source='csv'):
    """指定日付リスト（YYYYMMDD）を順次処理。"""
    print(f"\n{'#' * 60}")
    print(f"# バックフィル開始: {len(dates)}日")
    print(f"# 対象: {', '.join(dates)}")
    print(f"{'#' * 60}\n")

    total_settled = 0
    for date_str in dates:
        print(f"\n>>> バックフィル: {date_str}")
        n = run_daily_results(date_str, source=source, skip_settled=True)
        total_settled += n
        time.sleep(2.0)  # IP制限対策

    print(f"\n{'#' * 60}")
    print(f"# バックフィル完了: {total_settled}R 処理")
    print(f"{'#' * 60}\n")
    return total_settled


def discover_backfill_dates():
    """daily_predictions/ 配下の全CSVから日付を抽出。_prereceは除外。"""
    files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.csv")))
    dates = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if re.fullmatch(r'\d{8}', name):
            dates.append(name)
    return sorted(set(dates))


# ===== ROI再計算 =====

def recalc_roi():
    """cumulative_results.csv から累計・条件別・直近30R ROI を再計算し、
    data/actual_roi_results.json を更新する。"""
    if not os.path.exists(CUMUL_CSV):
        print(f"[ERROR] {CUMUL_CSV} が存在しません")
        return None

    df = pd.read_csv(CUMUL_CSV, encoding='utf-8-sig')
    if 'status' not in df.columns:
        print(f"[ERROR] status列がありません")
        return None

    settled = df[df['status'] == 'settled'].copy()
    settled['date'] = settled['date'].astype(str)
    settled['race_id'] = settled['race_id'].astype(str)
    # settled もdedup
    settled = settled.drop_duplicates(subset=['date','race_id'], keep='first')

    def _hit_mask(d):
        tm = (d['trio_hit'].fillna(0).astype(float) > 0)
        um = pd.Series([False]*len(d))
        if 'umaren_hit' in d.columns:
            um = (d['umaren_hit'].fillna(0).astype(float) > 0)
        return tm | um

    def _payout_sum(d):
        p = d['trio_payout'].fillna(0).astype(float).sum()
        if 'umaren_payout' in d.columns:
            p += d['umaren_payout'].fillna(0).astype(float).sum()
        return p

    # 全体
    n_total = len(settled)
    inv_total = int(settled['investment'].fillna(0).sum())
    pay_total = int(_payout_sum(settled))
    hits_total = int(_hit_mask(settled).sum())
    roi_total = (pay_total / inv_total * 100) if inv_total > 0 else 0.0
    profit_total = pay_total - inv_total

    # 条件別
    conditions = {}
    for c in sorted(settled['condition'].dropna().unique()):
        sub = settled[settled['condition'] == c]
        inv = int(sub['investment'].fillna(0).sum())
        pay = int(_payout_sum(sub))
        hits = int(_hit_mask(sub).sum())
        conditions[str(c)] = {
            'n': len(sub),
            'hits': hits,
            'hit_rate': round(hits / len(sub) * 100, 2) if len(sub) > 0 else 0,
            'investment': inv,
            'payout': pay,
            'profit': pay - inv,
            'roi': round(pay / inv * 100, 2) if inv > 0 else 0,
        }

    # 直近30R（date, race_idで昇順の末尾30件）
    recent = settled.sort_values(['date','race_id']).tail(30)
    inv_30 = int(recent['investment'].fillna(0).sum())
    pay_30 = int(_payout_sum(recent))
    hits_30 = int(_hit_mask(recent).sum())
    roi_30 = (pay_30 / inv_30 * 100) if inv_30 > 0 else 0.0

    stats = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'data/cumulative_results.csv (settled rows)',
        'model': 'v13.5b (4-model grid ensemble)',
        'total': {
            'n': n_total,
            'hits': hits_total,
            'hit_rate': round(hits_total / n_total * 100, 2) if n_total > 0 else 0,
            'investment': inv_total,
            'payout': pay_total,
            'profit': profit_total,
            'roi': round(roi_total, 2),
        },
        'conditions': conditions,
        'recent_30': {
            'n': len(recent),
            'hits': hits_30,
            'hit_rate': round(hits_30 / len(recent) * 100, 2) if len(recent) > 0 else 0,
            'investment': inv_30,
            'payout': pay_30,
            'profit': pay_30 - inv_30,
            'roi': round(roi_30, 2),
        },
        'dates_covered': sorted(settled['date'].unique().tolist()),
    }

    with open(ROI_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] ROI再計算完了: {ROI_JSON}")
    print(f"  累計: {n_total}R, ROI {roi_total:.1f}%, "
          f"{'+' if profit_total >= 0 else ''}{profit_total:,}円")
    print(f"  直近30R: ROI {roi_30:.1f}%")
    print(f"  条件別:")
    for c in sorted(conditions.keys()):
        cs = conditions[c]
        print(f"    {c}: N={cs['n']}, hit={cs['hits']}/{cs['n']} ({cs['hit_rate']}%), ROI {cs['roi']}%")

    return stats


def _snapshot_pre():
    """バックフィル前のROIスナップショットを取得。"""
    if not os.path.exists(CUMUL_CSV):
        return None
    df = pd.read_csv(CUMUL_CSV, encoding='utf-8-sig')
    if 'status' not in df.columns:
        return None
    s = df[df['status'] == 'settled'].copy()
    inv = int(s['investment'].fillna(0).sum())
    pay = int(s['trio_payout'].fillna(0).sum())
    if 'umaren_payout' in s.columns:
        pay += int(s['umaren_payout'].fillna(0).sum())
    hit_mask = (s['trio_hit'].fillna(0).astype(float) > 0)
    if 'umaren_hit' in s.columns:
        hit_mask = hit_mask | (s['umaren_hit'].fillna(0).astype(float) > 0)
    cond_stats = {}
    for c in sorted(s['condition'].dropna().unique()):
        sub = s[s['condition'] == c]
        cinv = int(sub['investment'].fillna(0).sum())
        cpay = int(sub['trio_payout'].fillna(0).sum())
        if 'umaren_payout' in sub.columns:
            cpay += int(sub['umaren_payout'].fillna(0).sum())
        cond_stats[str(c)] = {
            'n': len(sub),
            'investment': cinv,
            'payout': cpay,
            'profit': cpay - cinv,
            'roi': round(cpay/cinv*100, 2) if cinv > 0 else 0,
        }
    return {
        'n': len(s),
        'investment': inv,
        'payout': pay,
        'profit': pay - inv,
        'roi': round(pay/inv*100, 2) if inv > 0 else 0,
        'hits': int(hit_mask.sum()),
        'conditions': cond_stats,
    }


def save_backfill_report(pre_snap, post_stats, processed_dates, today_str):
    """バックフィル前後の比較レポートをJSONで保存。"""
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backfill_dates': sorted(processed_dates),
        'before': pre_snap,
        'after': {
            'n': post_stats['total']['n'],
            'investment': post_stats['total']['investment'],
            'payout': post_stats['total']['payout'],
            'profit': post_stats['total']['profit'],
            'roi': post_stats['total']['roi'],
            'hits': post_stats['total']['hits'],
            'conditions': {c: {k: v for k, v in cs.items() if k in ('n','investment','payout','profit','roi')}
                           for c, cs in post_stats['conditions'].items()},
        },
        'delta': {
            'added_races': post_stats['total']['n'] - (pre_snap['n'] if pre_snap else 0),
            'profit_change': post_stats['total']['profit'] - (pre_snap['profit'] if pre_snap else 0),
            'roi_change': round(post_stats['total']['roi'] - (pre_snap['roi'] if pre_snap else 0), 2),
        },
    }
    path = os.path.join(BASE_DIR, "data", f"backfill_report_{today_str}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] バックフィルレポート保存: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEIBA AI 結果照合")
    parser.add_argument("--date", type=str, default=None,
                        help="照合日 YYYYMMDD (デフォルト: 今日)")
    parser.add_argument("--source", type=str, default="csv",
                        choices=["db", "csv"],
                        help="予測ソース: csv=daily_predict CSV (デフォルト), db=Streamlit DB")
    parser.add_argument("--backfill", type=str, default=None,
                        help="日付カンマ区切り指定でバックフィル: 20260411,20260412,...")
    parser.add_argument("--backfill-all", action="store_true",
                        help="daily_predictions/ 配下の全CSVをバックフィル")
    parser.add_argument("--recalc-roi", action="store_true",
                        help="cumulative_results.csv からactual_roi_results.json再計算のみ")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_results.py 開始")

    # --recalc-roi のみなら結果照合スキップ
    if args.recalc_roi and not (args.backfill or args.backfill_all):
        recalc_roi()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_results.py 終了")
        sys.exit(0)

    # バックフィル処理
    if args.backfill_all or args.backfill:
        today_str = datetime.now().strftime("%Y%m%d")
        pre_snap = _snapshot_pre()

        if args.backfill_all:
            dates = discover_backfill_dates()
        else:
            dates = [d.strip() for d in args.backfill.split(",") if d.strip()]

        # 日付検証
        valid_dates = []
        for d in dates:
            try:
                datetime.strptime(d, "%Y%m%d")
                valid_dates.append(d)
            except ValueError:
                print(f"[WARN] 日付形式不正、スキップ: {d}")

        run_backfill(valid_dates, source=args.source)

        # ROI再計算
        post_stats = recalc_roi()
        if post_stats:
            save_backfill_report(pre_snap, post_stats, valid_dates, today_str)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_results.py 終了")
        sys.exit(0)

    # 通常モード：指定日（または今日）の結果照合
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime("%Y%m%d")

    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"[ERROR] 日付形式が不正です: {date_str} (YYYYMMDD)")
        sys.exit(1)

    run_daily_results(date_str, source=args.source)

    # ROI監視チェック
    try:
        from tools.roi_monitor import run_monitor
        run_monitor(dry_run=False)
    except Exception:
        try:
            from roi_monitor import run_monitor
            run_monitor(dry_run=False)
        except Exception:
            pass

    # Discord通知（リッチ版）
    try:
        from notify import send_discord
        result_path = os.path.join(BASE_DIR, "data", "daily_results", f"{date_str}.csv")
        cumul_path = CUMUL_CSV

        if os.path.exists(result_path):
            rdf = pd.read_csv(result_path, encoding='utf-8-sig')
            settled = rdf[rdf['status'] == 'settled']
            n = len(settled)
            if n > 0:
                hits = ((settled['trio_hit'] == 1) | (settled.get('umaren_hit', 0) == 1)).sum()
                inv = settled['investment'].sum()
                pay = settled['trio_payout'].fillna(0).sum() + settled.get('umaren_payout', pd.Series([0]*n)).fillna(0).sum()
                profit = pay - inv
                roi = pay / inv * 100 if inv > 0 else 0
                sign = '+' if profit >= 0 else ''

                hit_rows = settled[(settled['trio_hit'] == 1) | (settled.get('umaren_hit', 0) == 1)]
                hit_lines = []
                for _, hr in hit_rows.iterrows():
                    p = int(hr.get('trio_payout', 0) or hr.get('umaren_payout', 0))
                    hit_lines.append(f"  {hr.get('race_name','')} **{p:,}円**")

                cumul_msg = ''
                if os.path.exists(cumul_path):
                    cdf = pd.read_csv(cumul_path, encoding='utf-8-sig')
                    cs = cdf[cdf['status'] == 'settled']
                    if len(cs) > 0:
                        c_inv = cs['investment'].sum()
                        c_pay = cs['trio_payout'].fillna(0).sum()
                        if 'umaren_payout' in cs.columns:
                            c_pay += cs['umaren_payout'].fillna(0).sum()
                        c_roi = c_pay / c_inv * 100 if c_inv > 0 else 0
                        c_profit = c_pay - c_inv
                        cumul_msg = f"\n累計: ROI {c_roi:.0f}% / {'+' if c_profit>=0 else ''}{c_profit:,.0f}円 ({len(cs)}R)"

                # 3区分集計 (実投票/見送りシャドー/全部買い) — 2026-05-29
                buckets_msg = ''
                try:
                    _recs = settled.to_dict('records')
                    for _r in _recs:
                        _r['should_bet'] = _parse_should_bet(_r.get('should_bet', True))
                    buckets_msg = "\n\n" + _format_buckets_text(_compute_bet_buckets(_recs))
                except Exception as _bk_err:
                    print(f"[WARN] 3区分集計skip: {_bk_err}")

                color = "green" if profit >= 0 else "red"
                msg = (f"**{date_str}** {hits}/{n}的中 ROI **{roi:.0f}%**\n"
                       f"収支: **{sign}{profit:,.0f}円** ({inv:,.0f}円→{pay:,.0f}円)\n"
                       + ("\n".join(hit_lines) + "\n" if hit_lines else "")
                       + cumul_msg
                       + buckets_msg)
                send_discord(f"結果 {date_str} ({hits}/{n}的中)", msg, color=color)
            else:
                send_discord("結果照合", f"{date_str} 確定レース0件", color="yellow")
    except Exception:
        pass

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_results.py 終了")
