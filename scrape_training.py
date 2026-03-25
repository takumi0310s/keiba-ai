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


def fetch_training_times(race_id, is_nar=False):
    """netkeibaの追い切りページから各馬の調教タイムを取得。

    Returns:
        dict: {馬番: {
            'course': str,       # '坂路', 'CW', 'ポリ', '芝', 'ダート', etc.
            'time_4f': float,    # 4Fタイム (秒)
            'time_3f': float,    # 3Fタイム (秒)
            'time_1f': float,    # ラスト1Fタイム (秒)
            'intensity': str,    # '馬なり', '強め', '一杯', etc.
            'rank': str,         # 'A'/'B'/'C'/'D'
            'evaluation': str,   # '好調教', 'まずまず', etc.
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

    # Try premium session first, fall back to plain requests
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

        # Check if premium content is available
        # Premium content: TrainingTimeDataList, detailed time rows
        wrapper = soup.find("div", class_="OikiriAllWrapper")
        if not wrapper:
            _CACHE[race_id] = result
            return result

        # Parse the standard oikiri table first (always available)
        table = wrapper.find("table")
        if not table:
            _CACHE[race_id] = result
            return result

        rows = table.find_all("tr")
        for row in rows:
            td_umaban = row.select_one("td.Umaban")
            if not td_umaban:
                continue
            try:
                umaban = int(td_umaban.get_text(strip=True))
            except (ValueError, TypeError):
                continue

            # Get evaluation rank (free)
            rank = ""
            evaluation = ""
            td_critic = row.select_one("td.Training_Critic")
            if td_critic:
                evaluation = td_critic.get_text(strip=True)
            for td in row.find_all("td"):
                for cls in td.get("class", []):
                    if cls.startswith("Rank_"):
                        rank_text = cls.replace("Rank_", "")
                        if rank_text in ('A', 'B', 'C', 'D'):
                            rank = rank_text
                        elif any(x in rank_text for x in ['好調教', '抜群', '絶好']):
                            rank = 'A'
                        elif any(x in rank_text for x in ['上々', '乗込入念', '良化']):
                            rank = 'B'
                        elif any(x in rank_text for x in ['まずまず', '平凡', '乗込']):
                            rank = 'C'
                        else:
                            rank = 'C'
                        break
                if rank:
                    break

            result[umaban] = {
                'course': '', 'time_4f': 0.0, 'time_3f': 0.0, 'time_1f': 0.0,
                'intensity': '', 'rank': rank, 'evaluation': evaluation,
                'date': '', 'is_sakaro': False, 'is_wood': False, 'laps': [],
            }

        # Parse premium content from OikiriTable rows + TrainingTimeDataList
        _parse_premium_oikiri(soup, result)

        _CACHE[race_id] = result

    except Exception:
        pass

    return result


def _parse_premium_oikiri(soup, result):
    """Parse premium training data from the oikiri page.

    Structure (verified 2026-03):
    - table.OikiriTable contains tr.OikiriDataHead{N} per horse
    - Each tr has: 枠, 馬番, 印, 馬名, 日付, コース, 馬場, 乗り役, タイム(6F~1F)
    - Also: ul.TrainingTimeDataList per horse with 5 items: [6F?, 5F?, 4F, 3F, 1F]
      each formatted as "52.3(14.4)" = cumulative_time(lap_time)
    """
    # Find the premium table
    table = soup.find("table", class_=re.compile(r"OikiriTable"))
    if not table:
        return

    # Get all TrainingTimeDataList (one per horse, in order)
    time_lists = soup.find_all("ul", class_="TrainingTimeDataList")

    rows = table.find_all("tr", class_=re.compile(r"OikiriDataHead"))
    for idx, row in enumerate(rows):
        # Extract umaban
        td_umaban = row.select_one("td.Umaban")
        if not td_umaban:
            continue
        try:
            umaban = int(td_umaban.get_text(strip=True))
        except (ValueError, TypeError):
            continue
        if umaban not in result:
            continue

        full_text = row.get_text(strip=True)

        # Extract course from row text: 美Ｗ=美浦ウッド, 美坂=美浦坂路, 栗Ｗ=栗東ウッド, 栗坂=栗東坂路
        if re.search(r'[美栗]坂', full_text):
            result[umaban]['course'] = '坂路'
            result[umaban]['is_sakaro'] = True
        elif re.search(r'[美栗][ＷW]', full_text):
            result[umaban]['course'] = 'CW'
            result[umaban]['is_wood'] = True
        elif re.search(r'ポリ', full_text):
            result[umaban]['course'] = 'ポリトラック'
            result[umaban]['is_wood'] = True

        # Extract intensity
        intensity = _extract_intensity(full_text)
        if intensity:
            result[umaban]['intensity'] = intensity

        # Extract date
        date = _extract_date(full_text)
        if date:
            result[umaban]['date'] = date

        # Extract times from TrainingTimeDataList
        if idx < len(time_lists):
            ul = time_lists[idx]
            lis = ul.find_all("li")
            # 5 items: index 0=6F(?), 1=5F(?), 2=4F, 3=3F, 4=1F
            # Format: "52.3(14.4)" or "-" (no data)
            times = []
            for li in lis:
                text = li.get_text(strip=True)
                m = re.match(r'(\d+\.?\d*)\(', text)
                if m:
                    times.append(float(m.group(1)))
                else:
                    times.append(0.0)

            # Assign: positions depend on course
            # CW (ウッド): 6F, 5F, 4F, 3F, 1F (all 5 values)
            # 坂路: -, 4F, 3F, 2F, 1F (first is "-")
            if len(times) >= 5:
                if times[0] > 0:
                    # CW: times[2]=4F, times[3]=3F, times[4]=1F
                    if 35 < times[2] < 70:
                        result[umaban]['time_4f'] = times[2]
                    if 25 < times[3] < 50:
                        result[umaban]['time_3f'] = times[3]
                    if 10 < times[4] < 20:
                        result[umaban]['time_1f'] = times[4]
                else:
                    # 坂路: times[1]=4F, times[2]=3F, times[4]=1F
                    if 35 < times[1] < 70:
                        result[umaban]['time_4f'] = times[1]
                    if 25 < times[2] < 50:
                        result[umaban]['time_3f'] = times[2]
                    if 10 < times[4] < 20:
                        result[umaban]['time_1f'] = times[4]


def _extract_times(text, entry):
    """Extract 4F/3F/1F times from text."""
    # Pattern: "52.3-38.1-12.5" (4F-3F-1F)
    m = re.search(r'(\d{2}\.\d)\s*[-ー]\s*(\d{2}\.\d)\s*[-ー]\s*(\d{2}\.\d)', text)
    if m:
        t4 = float(m.group(1))
        t3 = float(m.group(2))
        t1 = float(m.group(3))
        if 35 < t4 < 75 and t3 < t4 and t1 < t3:
            entry['time_4f'] = t4
            entry['time_3f'] = t3
            entry['time_1f'] = t1
            return

    # Pattern: "4F 52.3" or "4F52.3"
    m4 = re.search(r'4F\s*(\d{2}\.\d)', text)
    if m4:
        t4 = float(m4.group(1))
        if 35 < t4 < 75:
            entry['time_4f'] = t4

    m3 = re.search(r'3F\s*(\d{2}\.\d)', text)
    if m3:
        t3 = float(m3.group(1))
        if 30 < t3 < 55:
            entry['time_3f'] = t3

    m1 = re.search(r'1F\s*(\d{2}\.\d)', text)
    if m1:
        t1 = float(m1.group(1))
        if 10 < t1 < 20:
            entry['time_1f'] = t1

    # Pattern: sequence of decimals that look like lap times
    laps = re.findall(r'(\d{2}\.\d)', text)
    if len(laps) >= 4 and not entry.get('laps'):
        lap_vals = [float(x) for x in laps[:4]]
        if all(10 < x < 20 for x in lap_vals):
            entry['laps'] = lap_vals


def _extract_course(text):
    """Extract training course from text."""
    for pattern, name in [
        (r'坂路', '坂路'),
        (r'[CＣ][WＷ]', 'CW'),
        (r'ウッド', 'CW'),
        (r'ポリ', 'ポリトラック'),
        (r'芝', '芝'),
        (r'ダート|ダ', 'ダート'),
    ]:
        if re.search(pattern, text):
            return name
    return ''


def _extract_intensity(text):
    """Extract training intensity from text."""
    for pattern, name in [
        (r'一杯', '一杯'),
        (r'強め', '強め'),
        (r'馬なり|馬ナリ', '馬なり'),
        (r'仕掛け', '仕掛け'),
        (r'末強め', '末強め'),
    ]:
        if re.search(pattern, text):
            return name
    return ''


def _extract_date(text):
    """Extract training date from text."""
    m = re.search(r'(\d{1,2})/(\d{1,2})', text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return ''


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
