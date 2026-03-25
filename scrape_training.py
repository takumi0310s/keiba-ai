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

        # Now try to parse premium content: detailed training time data
        # Premium content is in TrainingTimeDataList or in extended table rows
        _parse_premium_training(soup, result)

        _CACHE[race_id] = result

    except Exception:
        pass

    return result


def _parse_premium_training(soup, result):
    """Parse premium training time data from the oikiri page.

    Tries multiple HTML patterns since netkeiba's structure varies.
    """
    # Pattern 1: TrainingTimeDataList (ul > li structure)
    for ul in soup.find_all("ul", class_=re.compile(r"TrainingTimeDataList")):
        _parse_training_list(ul, result)

    # Pattern 2: Extended table with training details
    # Sometimes training times appear in additional table rows/cells
    for table in soup.find_all("table"):
        _parse_training_table(table, result)

    # Pattern 3: div-based training data (newer layout)
    for div in soup.find_all("div", class_=re.compile(r"OikiriDataHead|TrainingDetail")):
        _parse_training_div(div, result)


def _parse_training_list(ul, result):
    """Parse ul.TrainingTimeDataList structure."""
    lis = ul.find_all("li", recursive=False)
    current_umaban = 0

    for li in lis:
        # Check if this li has a horse number
        umaban_elem = li.find(class_=re.compile(r"Umaban|Horse_Num"))
        if umaban_elem:
            try:
                current_umaban = int(umaban_elem.get_text(strip=True))
            except (ValueError, TypeError):
                pass

        if current_umaban == 0 or current_umaban not in result:
            continue

        text = li.get_text(strip=True)

        # Extract training course
        course = _extract_course(text)
        if course:
            result[current_umaban]['course'] = course
            result[current_umaban]['is_sakaro'] = '坂路' in course
            result[current_umaban]['is_wood'] = any(x in course for x in ['CW', 'ウッド', 'Ｗ'])

        # Extract times (patterns: 52.3-38.1-12.5 or 4F 52.3)
        _extract_times(text, result[current_umaban])

        # Extract intensity
        intensity = _extract_intensity(text)
        if intensity:
            result[current_umaban]['intensity'] = intensity

        # Extract date
        date = _extract_date(text)
        if date:
            result[current_umaban]['date'] = date


def _parse_training_table(table, result):
    """Parse extended training table for time data."""
    rows = table.find_all("tr")
    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 3:
            continue

        # Try to find umaban
        umaban = 0
        for td in tds:
            cls = " ".join(td.get("class", []))
            if "Umaban" in cls:
                try:
                    umaban = int(td.get_text(strip=True))
                except (ValueError, TypeError):
                    pass
                break

        if umaban == 0 or umaban not in result:
            continue

        # Look for time data in remaining cells
        full_text = row.get_text(strip=True)
        _extract_times(full_text, result[umaban])

        course = _extract_course(full_text)
        if course:
            result[umaban]['course'] = course
            result[umaban]['is_sakaro'] = '坂路' in course
            result[umaban]['is_wood'] = any(x in course for x in ['CW', 'ウッド', 'Ｗ'])

        intensity = _extract_intensity(full_text)
        if intensity:
            result[umaban]['intensity'] = intensity


def _parse_training_div(div, result):
    """Parse div-based training detail sections."""
    text = div.get_text(strip=True)
    # Try to associate with a horse number
    umaban_match = re.search(r'(\d{1,2})番', text)
    if not umaban_match:
        return
    umaban = int(umaban_match.group(1))
    if umaban not in result:
        return
    _extract_times(text, result[umaban])


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

            if data['is_wood'] or data['is_sakaro']:
                if data['is_wood']:
                    feat['wood_best_4f_filled'] = data['time_4f']
                    feat['has_wood_training'] = 1
                    feat['wood_count_2w'] = 2
                if data['is_sakaro']:
                    feat['sakaro_best_4f_filled'] = data['time_4f']
                    feat['has_sakaro_training'] = 1
                    if data['time_3f'] > 0:
                        feat['sakaro_best_3f_filled'] = data['time_3f']
            else:
                # Unknown course → treat as wood
                feat['wood_best_4f_filled'] = data['time_4f']
                feat['has_wood_training'] = 1
                feat['wood_count_2w'] = 2
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


if __name__ == '__main__':
    # Test
    import sys, io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    ok, msg = check_premium_access()
    print(f"Premium: {msg}")

    if ok:
        data = fetch_training_times('202609010901')
        print(f"\nTraining data for 202609010901: {len(data)} horses")
        for umaban, d in sorted(data.items()):
            print(f"  馬番{umaban}: rank={d['rank']} course={d['course']} "
                  f"4F={d['time_4f']} 3F={d['time_3f']} 1F={d['time_1f']} "
                  f"intensity={d['intensity']}")
    else:
        print("Testing with rank-only mode (no premium)...")
        data = fetch_training_times('202609010901')
        print(f"Rank data: {len(data)} horses")
        features = get_training_features('202609010901', 10)
        print(f"Features: {len(features)} horses")
        for umaban, f in sorted(features.items()):
            print(f"  馬番{umaban}: {f}")
