#!/usr/bin/env python
"""netkeiba スーパープレミアム 調教タイムスクレイパー

Cookie認証でnetkeibaの調教タイムページから実タイムを取得する。
Cookieは .env ファイルから読み込む。

取得データ:
- 最終追い切り日
- 調教コース（坂路/CW/ポリ/芝/ダート）
- 4Fタイム, 3Fタイム, 1Fタイム
- 調教強度（馬なり/強め/一杯）
- ラップタイム

Usage:
    from scrape_training import fetch_training_times
    data = fetch_training_times(race_id)
    # {馬番: {'course': '坂路', 'time_4f': 52.3, 'time_3f': 38.1, ...}}
"""
import requests
import re
import os
import time as time_module
from bs4 import BeautifulSoup
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# In-memory cache: {race_id: {umaban: data}}
_CACHE = {}


def _load_cookie():
    """Load netkeiba cookie from .env file."""
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return None
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('NETKEIBA_COOKIE='):
                val = line[len('NETKEIBA_COOKIE='):]
                val = val.strip('"').strip("'")
                return val if val and val != 'nkauth=XXXX; _nk_session=XXXX; ...' else None
    return None


def _make_session():
    """Create a requests session with premium cookie."""
    cookie_str = _load_cookie()
    if not cookie_str:
        return None
    session = requests.Session()
    session.headers.update(HEADERS)
    # Parse cookie string into dict
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            key, val = part.split('=', 1)
            session.cookies.set(key.strip(), val.strip())
    return session


def _parse_time_list(ul):
    """TrainingTimeDataList (ul) から4F/3F/1Fタイムを抽出する。

    Returns: (time_4f, time_3f, time_1f) — 取得できなかった場合は0.0
    """
    lis = ul.find_all("li")
    times = []
    for li in lis:
        text = li.get_text(strip=True)
        m = re.match(r'(\d+\.?\d*)\(', text)
        times.append(float(m.group(1)) if m else 0.0)

    t4f, t3f, t1f = 0.0, 0.0, 0.0
    if len(times) >= 5:
        if times[0] > 0:
            # Full CW: 6F-5F-4F-3F-1F
            if 35 < times[2] < 70: t4f = times[2]
            if 25 < times[3] < 50: t3f = times[3]
            if 10 < times[4] < 20: t1f = times[4]
        elif times[1] > 0:
            # 坂路 or 5F CW: -4F-3F-2F-1F (5th item = 1F)
            if 35 < times[1] < 70: t4f = times[1]
            if 25 < times[2] < 50: t3f = times[2]
            if 10 < times[4] < 20: t1f = times[4]
        elif times[2] > 0:
            # Short CW (4F only): --4F-3F-1F
            if 35 < times[2] < 70: t4f = times[2]
            if 25 < times[3] < 50: t3f = times[3]
            if 10 < times[4] < 20: t1f = times[4]
    elif len(times) >= 4:
        # 4-item variant: 4F-3F-2F-1F
        if 35 < times[0] < 70: t4f = times[0]
        if 25 < times[1] < 50: t3f = times[1]
        if 10 < times[3] < 20: t1f = times[3]
    return t4f, t3f, t1f


def _parse_time_text(time_text):
    """TrainingTimeData cell の生テキストから4F/3F/1Fを抽出する。

    Example: "81.3(16.9)64.4(15.5)48.9(13.0)35.9(12.4)12.4"
             → 4F=48.9, 3F=35.9, 1F=12.4
    """
    t4f, t3f, t1f = 0.0, 0.0, 0.0
    if not time_text or '----' in time_text:
        return t4f, t3f, t1f

    # Extract cumulative/split times: digits.digits patterns
    nums = re.findall(r'(\d{2}\.\d)', time_text)
    if not nums:
        return t4f, t3f, t1f

    vals = [float(x) for x in nums]
    # Strategy: find values in expected ranges
    candidates_4f = [v for v in vals if 35 < v < 70]
    candidates_3f = [v for v in vals if 25 < v < 50]
    candidates_1f = [v for v in vals if 10 < v < 20]

    if candidates_4f:
        t4f = candidates_4f[0]
    if candidates_3f:
        # Pick the 3F that's smaller than 4F
        for v in candidates_3f:
            if t4f == 0 or v < t4f:
                t3f = v
                break
    if candidates_1f:
        t1f = candidates_1f[-1]  # Last 1F value (closest to finish)

    return t4f, t3f, t1f


_INT_MAP = {'一杯': '一杯', '強め': '強め', '馬也': '馬なり', '馬ナリ': '馬なり',
            'G前': '強め', '直一杯': '一杯', '仕掛': '強め', '末強め': '強め'}


def _detect_course(text):
    """テキストからコース種別を判定する。Returns: (course, is_sakaro, is_wood)"""
    if not text:
        return '', False, False
    if '坂' in text:
        return '坂路', True, False
    if re.search(r'[ＣC][ＷW]|ウッド', text):
        return 'CW', False, True
    if re.search(r'[ＤD][ＰP]|ポリ', text):
        return 'ポリトラック', False, True
    if re.match(r'^[美栗]?[ＥＢＡＤＣEBADCニ]$', text.strip()):
        return 'CW', False, True
    if re.search(r'[美栗]坂', text):
        return '坂路', True, False
    if re.search(r'[美栗][ＷW]', text):
        return 'CW', False, True
    if re.search(r'[美栗][ＥＢＡＤＣEBADCニ]', text):
        return 'CW', False, True
    if re.search(r'芝', text):
        return '芝', False, True
    if re.search(r'ダ[ートー]', text) or text.strip() == 'ダ':
        return 'ダート', False, True
    return '', False, False


def _extract_rank(row):
    """行からRank (A/B/C/D) を抽出する。"""
    for td in row.find_all("td"):
        for cls in td.get("class", []):
            if cls.startswith("Rank_"):
                rank_text = cls.replace("Rank_", "")
                if rank_text in ('A', 'B', 'C', 'D'):
                    return rank_text
    return ""


def fetch_training_times(race_id, is_nar=False):
    """netkeibaの追い切りページから各馬の調教タイムを取得。

    HTML構造パターン:
      Pattern A: 1行13cells — 馬情報+調教詳細が同一行
      Pattern B: 2行ペア — 5cells(馬情報) + 9cells(調教詳細)
      混在あり (同一レースで一部A、一部Bのケースも対応)

    Returns:
        dict: {馬番: {
            'course': str,       # '坂路', 'CW', 'ポリ', '芝', 'ダート', etc.
            'time_4f': float,    # 4Fタイム (秒)
            'time_3f': float,    # 3Fタイム (秒)
            'time_1f': float,    # ラスト1Fタイム (秒)
            'intensity': str,    # '馬なり', '強め', '一杯', etc.
            'rank': str,         # 'A'/'B'/'C'/'D'
            'evaluation': str,   # '好調教', 'まずまず', etc.
            'review': str,       # 記者評コメント
            'date': str,         # '3/19' or '2026/03/19'
            'is_sakaro': bool,   # True if 坂路
            'is_wood': bool,     # True if CW/ウッド
            'laps': list,        # [lap1, lap2, lap3, lap4]
        }}
        Returns empty dict if cookie not set or fetch fails.
    """
    # Check cache
    if race_id in _CACHE:
        return _CACHE[race_id]

    session = _make_session()

    result = {}
    try:
        if is_nar:
            url = f"https://nar.netkeiba.com/race/oikiri.html?race_id={race_id}"
        else:
            url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"

        if session:
            resp = session.get(url, timeout=10)
        else:
            resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        wrapper = soup.find("div", class_="OikiriAllWrapper")
        if not wrapper:
            _CACHE[race_id] = result
            return result

        table = wrapper.find("table")
        if not table:
            _CACHE[race_id] = result
            return result

        # Collect all TrainingTimeDataList upfront
        time_lists = soup.find_all("ul", class_="TrainingTimeDataList")

        # ===== Single-pass parsing: iterate all rows, build horse entries =====
        # Track horse encounter order for TrainingTimeDataList alignment
        horse_order = []  # list of umaban in HTML encounter order
        rows = table.find_all("tr")
        pending_umaban = None  # For Pattern B: waiting for detail row

        for row in rows:
            td_umaban = row.select_one("td.Umaban")
            cells = row.find_all("td")

            if td_umaban:
                try:
                    umaban = int(td_umaban.get_text(strip=True))
                except (ValueError, TypeError):
                    continue

                entry = {
                    'course': '', 'time_4f': 0.0, 'time_3f': 0.0, 'time_1f': 0.0,
                    'intensity': '', 'rank': '', 'evaluation': '', 'review': '',
                    'date': '', 'is_sakaro': False, 'is_wood': False, 'laps': [],
                }

                # Extract review comment (TrainingReview_Cell)
                td_review = row.select_one("td.TrainingReview_Cell")
                if td_review:
                    entry['review'] = td_review.get_text(strip=True)

                if len(cells) >= 10:
                    # ===== Pattern A: single row (13 cells) =====
                    entry['rank'] = _extract_rank(row)

                    # Intensity
                    td_load = row.select_one("td.TrainingLoad")
                    if td_load:
                        load_text = td_load.get_text(strip=True)
                        for key, val in _INT_MAP.items():
                            if key in load_text:
                                entry['intensity'] = val
                                break

                    # Evaluation
                    td_critic = row.select_one("td.Training_Critic")
                    if td_critic:
                        entry['evaluation'] = td_critic.get_text(strip=True)

                    # Course
                    for td in cells:
                        ct = td.get_text(strip=True)
                        course, is_s, is_w = _detect_course(ct)
                        if course and len(ct) <= 12:
                            entry['course'] = course
                            entry['is_sakaro'] = is_s
                            entry['is_wood'] = is_w
                            break

                    # Date
                    td_day = row.select_one("td.Training_Day")
                    if td_day:
                        entry['date'] = td_day.get_text(strip=True)

                    # Times from TrainingTimeData cell text
                    td_time = row.select_one("td.TrainingTimeData")
                    if td_time:
                        t4, t3, t1 = _parse_time_text(td_time.get_text(strip=True))
                        if t4 > 0: entry['time_4f'] = t4
                        if t3 > 0: entry['time_3f'] = t3
                        if t1 > 0: entry['time_1f'] = t1

                    result[umaban] = entry
                    horse_order.append(umaban)
                    pending_umaban = None
                else:
                    # ===== Pattern B horse row (5 cells): wait for detail row =====
                    result[umaban] = entry
                    pending_umaban = umaban

            elif pending_umaban is not None and pending_umaban in result:
                # ===== Pattern B detail row (9 cells) =====
                entry = result[pending_umaban]

                entry['rank'] = _extract_rank(row)

                # Course (cell[1])
                course_text = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                row_text = row.get_text(strip=True)
                course, is_s, is_w = _detect_course(course_text)
                if not course:
                    course, is_s, is_w = _detect_course(row_text)
                if course:
                    entry['course'] = course
                    entry['is_sakaro'] = is_s
                    entry['is_wood'] = is_w

                # Intensity
                td_load = row.select_one("td.TrainingLoad")
                if td_load:
                    load_text = td_load.get_text(strip=True)
                    for key, val in _INT_MAP.items():
                        if key in load_text:
                            entry['intensity'] = val
                            break

                # Evaluation
                td_critic = row.select_one("td.Training_Critic")
                if td_critic:
                    entry['evaluation'] = td_critic.get_text(strip=True)

                # Date
                td_day = row.select_one("td.Training_Day")
                if td_day:
                    entry['date'] = td_day.get_text(strip=True)

                # Times from TrainingTimeData cell text
                td_time = row.select_one("td.TrainingTimeData")
                if td_time:
                    t4, t3, t1 = _parse_time_text(td_time.get_text(strip=True))
                    if t4 > 0: entry['time_4f'] = t4
                    if t3 > 0: entry['time_3f'] = t3
                    if t1 > 0: entry['time_1f'] = t1

                horse_order.append(pending_umaban)
                pending_umaban = None

        # Handle last pending horse (Pattern B without detail row)
        if pending_umaban is not None and pending_umaban not in horse_order:
            horse_order.append(pending_umaban)

        # ===== TrainingTimeDataList: fill times for horses still missing 4F =====
        for ti, uma in enumerate(horse_order):
            if ti >= len(time_lists):
                break
            if uma not in result or result[uma]['time_4f'] > 0:
                continue
            t4, t3, t1 = _parse_time_list(time_lists[ti])
            if t4 > 0: result[uma]['time_4f'] = t4
            if t3 > 0 and result[uma]['time_3f'] == 0: result[uma]['time_3f'] = t3
            if t1 > 0 and result[uma]['time_1f'] == 0: result[uma]['time_1f'] = t1

        # ===== Rank-based fallback for horses with rank but no 4F =====
        _RANK_4F = {'A': 51.5, 'B': 53.0, 'C': 54.5, 'D': 55.5}
        _RANK_3F = {'A': 37.5, 'B': 39.0, 'C': 40.5, 'D': 41.5}
        for uma in result:
            if result[uma]['time_4f'] == 0 and result[uma].get('rank', '') in _RANK_4F:
                rank = result[uma]['rank']
                result[uma]['time_4f'] = _RANK_4F[rank]
                if result[uma]['time_3f'] == 0:
                    result[uma]['time_3f'] = _RANK_3F[rank]

        _CACHE[race_id] = result

    except Exception:
        pass

    return result




def check_premium_access():
    """Check if premium cookie is valid."""
    session = _make_session()
    if session is None:
        return False, "Cookie未設定: .envファイルにNETKEIBA_COOKIEを設定してください"

    try:
        # Try accessing a premium page
        url = "https://db.netkeiba.com?pid=horse_training&id=2023102916"
        resp = session.get(url, timeout=10)
        resp.encoding = "EUC-JP"

        # If we get redirected to premium registration, cookie is invalid
        if 'プレミアムサービス案内' in resp.text:
            return False, "Cookie無効: ログインセッション切れ。ブラウザで再ログインしCookie更新してください"

        # Check for training time data
        if re.search(r'\d{2}\.\d', resp.text):
            return True, "Premium認証OK: 調教タイムデータ取得可能"
        else:
            return True, "Premium認証OK: ページアクセス可（データ有無は未確認）"

    except Exception as e:
        return False, f"接続エラー: {e}"


def get_training_features(race_id, num_horses, is_nar=False):
    """fetch_training_timesの結果をモデル特徴量に変換する。

    Returns:
        dict: {umaban: {
            'wood_best_4f_filled': float,
            'sakaro_best_4f_filled': float,
            'sakaro_best_3f_filled': float,
            'has_wood_training': int,
            'has_sakaro_training': int,
            'total_training_count': int,
            'training_time_filled': float,
            'has_training': int,
            'wood_count_2w': int,
        }}
    """
    training_data = fetch_training_times(race_id, is_nar=is_nar)

    # Rank-based fallback values
    RANK_TO_WOOD_4F = {'A': 51.5, 'B': 53.0, 'C': 54.5, 'D': 55.5}
    RANK_TO_SAKARO_4F = {'A': 53.5, 'B': 56.0, 'C': 58.0, 'D': 59.5}
    RANK_TO_SAKARO_3F = {'A': 37.5, 'B': 39.0, 'C': 40.5, 'D': 41.5}

    features = {}
    for umaban, data in training_data.items():
        feat = {}
        rank = data.get('rank', '')

        if data['time_4f'] > 0:
            # Real time available from premium
            feat['training_time_filled'] = data['time_4f']
            feat['has_training'] = 1

            if data['is_sakaro']:
                feat['sakaro_best_4f_filled'] = data['time_4f']
                feat['has_sakaro_training'] = 1
                if data['time_3f'] > 0:
                    feat['sakaro_best_3f_filled'] = data['time_3f']
                # 坂路の馬もwood側にはデフォルト平均をセット（学習データと同じ扱い）
                feat['wood_best_4f_filled'] = 52.0
                feat['has_wood_training'] = 0
                feat['wood_count_2w'] = 0
            elif data['is_wood']:
                feat['wood_best_4f_filled'] = data['time_4f']
                feat['has_wood_training'] = 1
                feat['wood_count_2w'] = 2
                # CW馬もsakaro側にはデフォルト平均をセット
                feat['sakaro_best_4f_filled'] = 53.0
                feat['sakaro_best_3f_filled'] = 39.0
                feat['has_sakaro_training'] = 0
            else:
                feat['wood_best_4f_filled'] = data['time_4f']
                feat['has_wood_training'] = 1
                feat['wood_count_2w'] = 2
                feat['sakaro_best_4f_filled'] = 53.0
                feat['sakaro_best_3f_filled'] = 39.0
                feat['has_sakaro_training'] = 0
            feat['total_training_count'] = 3
        elif rank in RANK_TO_WOOD_4F:
            # Fallback: rank-based estimation
            feat['wood_best_4f_filled'] = RANK_TO_WOOD_4F[rank]
            feat['has_wood_training'] = 1
            feat['wood_count_2w'] = 2 if rank in ('A', 'B') else 1
            feat['sakaro_best_4f_filled'] = RANK_TO_SAKARO_4F[rank]
            feat['sakaro_best_3f_filled'] = RANK_TO_SAKARO_3F[rank]
            feat['has_sakaro_training'] = 1
            feat['total_training_count'] = 4 if rank in ('A', 'B') else 2
            feat['training_time_filled'] = RANK_TO_WOOD_4F[rank]
            feat['has_training'] = 1

        if feat:
            features[umaban] = feat

    return features


# ============================================================
# 厩舎コメント取得・スコア化
# ============================================================

# ポジティブ/ネガティブキーワード → スコア
_POSITIVE_KEYWORDS = {
    3: ['抜群', '絶好調', '文句なし', '破格', '圧巻', '最高', '凄い動き'],
    2: ['好調', '上昇', '仕上がり良', '動き良', '好内容', '力強い', '好時計', '良化', '好気配',
        '成長', '充実', '上積み', '好仕上', '態勢整', '万全', '上向き', '具合は良',
        '楽しみ', '期待', '魅力', '自信', 'いい動き', 'いい状態', '良い状態',
        '手応え十分', '申し分', '素晴らし', '状態良'],
    1: ['順調', 'まずまず', '変わりなく', '無難', '及第点', '堅実', '安定', '問題な',
        'しっかり', 'いつも通り', '落ち着い', '悪くな', '前走と同じ', '変わらず'],
}
_NEGATIVE_KEYWORDS = {
    -1: ['平凡', '物足りない', '平行線', 'ひと息', '微妙', '地味', '強調材料に欠け',
         '前走ほどでは', 'もう一つ', 'どうか', '未知数'],
    -2: ['不安', '下降', '太め', 'イマイチ', '重め', '気になる', '反応鈍い', '落ち',
         '心配', '課題', '苦しい', '力んで', 'ピリッとしない'],
    -3: ['休み明け', '状態悪', '仕上がり途上', '不振', '深刻', '故障', '痛め'],
}


def _score_comment(text):
    """コメントテキストをスコア化。"""
    if not text:
        return 0
    score = 0
    for s, keywords in _POSITIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                score = max(score, s)
    for s, keywords in _NEGATIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                score = min(score, s)
    return score


_COMMENT_CACHE = {}


def fetch_stable_comments(race_id, is_nar=False):
    """netkeibaの厩舎コメントページから各馬のコメントとスコアを取得。

    Returns:
        dict: {馬番: {'comment': str, 'score': int (-3 to +3)}}
    """
    if race_id in _COMMENT_CACHE:
        return _COMMENT_CACHE[race_id]

    result = {}
    try:
        session = _make_session()
        if is_nar:
            url = f"https://nar.netkeiba.com/race/comment.html?race_id={race_id}"
        else:
            url = f"https://race.netkeiba.com/race/comment.html?race_id={race_id}"

        if session:
            resp = session.get(url, timeout=10)
        else:
            resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_=re.compile(r"Stable_Comment|Comment_Table"))
        if not table:
            _COMMENT_CACHE[race_id] = result
            return result

        rows = table.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 4:
                continue
            # Column order: 枠, 馬番, 馬名, コメント, [評価]
            try:
                umaban = int(tds[1].get_text(strip=True))
            except (ValueError, TypeError, IndexError):
                continue
            comment = tds[3].get_text(strip=True) if len(tds) > 3 else ""
            score = _score_comment(comment)
            result[umaban] = {'comment': comment, 'score': score}

        _COMMENT_CACHE[race_id] = result
    except Exception:
        pass
    return result


# ============================================================
# Cookie状態チェック（キャッシュ付き）
# ============================================================
_COOKIE_STATUS = None  # (bool, str) or None


def get_cookie_status(force=False):
    """Cookie状態を取得（キャッシュ付き）。"""
    global _COOKIE_STATUS
    if _COOKIE_STATUS is not None and not force:
        return _COOKIE_STATUS
    _COOKIE_STATUS = check_premium_access()
    return _COOKIE_STATUS


def is_cookie_valid():
    """Cookie有効かどうか。"""
    ok, _ = get_cookie_status()
    return ok


def cookie_warning_html():
    """Cookie無効時の警告HTML（app.py表示用）。空文字ならOK。"""
    ok, msg = get_cookie_status()
    if ok:
        return ''
    if _load_cookie() is None:
        return ''  # Cookie未設定 → 警告不要（無料モード）
    # Cookie設定済みだが無効
    return (
        '<div style="margin:8px 0;padding:10px 16px;background:#3a1a0a;'
        'border:1px solid #e67e22;border-radius:8px;color:#e67e22;font-size:0.85em;">'
        '&#9888; netkeiba Cookie期限切れ: 調教実タイムが取得できません。'
        'ブラウザで再ログインし .env の NETKEIBA_COOKIE を更新してください。'
        '（ランク推定でフォールバック中）</div>'
    )


if __name__ == '__main__':
    import sys, io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    ok, msg = check_premium_access()
    print(f"Premium: {msg}")

    # Test training times
    test_id = '202606020501'
    data = fetch_training_times(test_id)
    realtime = sum(1 for d in data.values() if d.get('time_4f', 0) > 0)
    print(f"\nTraining: {len(data)} horses ({realtime} realtime)")
    for umaban, d in sorted(data.items())[:5]:
        print(f"  馬番{umaban}: rank={d['rank']} 4F={d['time_4f']} course={d['course']}")

    # Test stable comments
    comments = fetch_stable_comments(test_id)
    print(f"\nComments: {len(comments)} horses")
    for umaban, c in sorted(comments.items())[:5]:
        print(f"  馬番{umaban}: score={c['score']:+d} comment={c['comment'][:40]}")
