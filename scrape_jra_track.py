"""JRA公式サイトから馬場情報を取得

取得データ:
- クッション値（芝レースの場合）
- 含水率（芝ゴール前/4コーナー、ダートゴール前/4コーナー）
- 馬場状態・天候

JRA馬場情報ページ: https://www.jra.go.jp/keiba/baba/
データはJavaScript経由で3つのAPIから動的ロードされる:
  1. _data_cushion.html  - クッション値（HTML形式）
  2. _data_moist.html    - 含水率（HTML形式）
  3. /JRADB/accessJ.html - 馬場状態・天候（JSON API, POST）
"""
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

BASE_URL = "https://www.jra.go.jp"

# 競馬場名 → JRA API内のrc ID (sort値)
# A/B/Cは開催順で毎週変わるため、JRADB APIのjyonameで動的マッピング
COURSE_TO_JRA = {
    '札幌': 'sapporo', '函館': 'hakodate', '福島': 'fukushima',
    '新潟': 'niigata', '東京': 'tokyo', '中山': 'nakayama',
    '中京': 'chukyo', '京都': 'kyoto', '阪神': 'hanshin', '小倉': 'kokura',
}


def _get_venue_sort_map():
    """JRADB APIから今日の開催場 → sort(A/B/C)マッピングを取得"""
    try:
        r = requests.post(
            f"{BASE_URL}/JRADB/accessJ.html",
            data={'CNAME': 'pw01iwtS3/CD'},
            headers=HEADERS,
            timeout=10,
        )
        r.encoding = 'shift_jis'
        data = r.json()
        mapping = {}  # {'中山': {'sort': 'A', 'weather': '晴', 'ba_s': '良', 'ba_d': '良'}, ...}
        for info in data.get('kaisai_info', []):
            name = info.get('jyoname', '')
            if name:
                mapping[name] = {
                    'sort': info.get('sort', ''),
                    'weather': info.get('weather', ''),
                    'ba_s': info.get('ba_s', ''),
                    'ba_d': info.get('ba_d', ''),
                }
        return mapping
    except Exception:
        return {}


def _fetch_cushion_data():
    """クッション値データを取得 (HTML形式)

    Returns:
        dict: {'rcA': [{'time': '...', 'value': 10.0}, ...], 'rcB': [...], ...}
    """
    result = {}
    try:
        r = requests.get(f"{BASE_URL}/keiba/baba/_data_cushion.html", headers=HEADERS, timeout=10)
        r.encoding = 'shift_jis'
        soup = BeautifulSoup(r.text, 'html.parser')

        for rc_div in soup.find_all('div', id=lambda x: x and x.startswith('rc')):
            rc_id = rc_div.get('id', '')
            venue_name = rc_div.get('title', '')
            entries = []
            for unit in rc_div.find_all('div', class_='unit'):
                time_tag = unit.find('div', class_='time')
                val_tag = unit.find('div', class_='cushion')
                if time_tag and val_tag:
                    try:
                        entries.append({
                            'time': time_tag.get_text(strip=True),
                            'value': float(val_tag.get_text(strip=True)),
                        })
                    except (ValueError, TypeError):
                        pass
            if entries:
                result[rc_id] = {'venue': venue_name, 'data': entries}
    except Exception:
        pass
    return result


def _fetch_moist_data():
    """含水率データを取得 (HTML形式)

    Returns:
        dict: {'rcA': {'venue': '中山', 'data': [{'time': ..., 'turf_goal': ..., ...}]}, ...}
    """
    result = {}
    try:
        r = requests.get(f"{BASE_URL}/keiba/baba/_data_moist.html", headers=HEADERS, timeout=10)
        r.encoding = 'shift_jis'
        soup = BeautifulSoup(r.text, 'html.parser')

        for rc_div in soup.find_all('div', id=lambda x: x and x.startswith('rc')):
            rc_id = rc_div.get('id', '')
            venue_name = rc_div.get('title', '')
            entries = []
            for unit in rc_div.find_all('div', class_='unit'):
                time_tag = unit.find('div', class_='time')
                time_str = time_tag.get_text(strip=True) if time_tag else ''

                entry = {'time': time_str}
                turf_div = unit.find('div', class_='turf')
                if turf_div:
                    mg = turf_div.find('span', class_='mg')
                    m4c = turf_div.find('span', class_='m4c')
                    if mg:
                        try:
                            entry['turf_goal'] = float(mg.get_text(strip=True))
                        except (ValueError, TypeError):
                            pass
                    if m4c:
                        try:
                            entry['turf_4c'] = float(m4c.get_text(strip=True))
                        except (ValueError, TypeError):
                            pass

                dirt_div = unit.find('div', class_='dirt')
                if dirt_div:
                    mg = dirt_div.find('span', class_='mg')
                    m4c = dirt_div.find('span', class_='m4c')
                    if mg:
                        try:
                            entry['dirt_goal'] = float(mg.get_text(strip=True))
                        except (ValueError, TypeError):
                            pass
                    if m4c:
                        try:
                            entry['dirt_4c'] = float(m4c.get_text(strip=True))
                        except (ValueError, TypeError):
                            pass

                if len(entry) > 1:  # time + at least one value
                    entries.append(entry)

            if entries:
                result[rc_id] = {'venue': venue_name, 'data': entries}
    except Exception:
        pass
    return result


def fetch_jra_track_info(course_name):
    """JRA公式から馬場情報を取得

    3つのAPIを叩いてクッション値・含水率・馬場状態を取得する。
    - _data_cushion.html: クッション値（最新測定値）
    - _data_moist.html: 含水率（芝/ダート × ゴール前/4コーナー）
    - /JRADB/accessJ.html: 馬場状態・天候（リアルタイム）

    Args:
        course_name: 競馬場名（例: '東京', '阪神'）

    Returns:
        dict: {
            'cushion_value': float or None,
            'moisture_turf_goal': float or None,
            'moisture_turf_4c': float or None,
            'moisture_dirt_goal': float or None,
            'moisture_dirt_4c': float or None,
            'condition_turf': str or None,
            'condition_dirt': str or None,
            'weather': str or None,
            'source': 'jra_api',
        }
    """
    result = {
        'cushion_value': None,
        'moisture_turf_goal': None, 'moisture_turf_4c': None,
        'moisture_dirt_goal': None, 'moisture_dirt_4c': None,
        'condition_turf': None, 'condition_dirt': None,
        'weather': None,
        'source': 'jra_api',
    }

    if course_name not in COURSE_TO_JRA:
        return result

    try:
        # 1. JRADB APIで開催場→sort(A/B/C)マッピングと馬場状態を取得
        venue_map = _get_venue_sort_map()
        venue_info = venue_map.get(course_name)
        if not venue_info:
            return result

        sort_letter = venue_info['sort']  # 'A', 'B', or 'C'
        rc_id = f'rc{sort_letter}'

        # 馬場状態・天候
        result['condition_turf'] = venue_info.get('ba_s') or None
        result['condition_dirt'] = venue_info.get('ba_d') or None
        result['weather'] = venue_info.get('weather') or None

        # 2. クッション値（最新値を取得）
        cushion_data = _fetch_cushion_data()
        if rc_id in cushion_data:
            entries = cushion_data[rc_id]['data']
            if entries:
                v = entries[0]['value']  # 最新（先頭）
                if 5.0 <= v <= 15.0:
                    result['cushion_value'] = v

        # 3. 含水率（最新値を取得）
        moist_data = _fetch_moist_data()
        if rc_id in moist_data:
            entries = moist_data[rc_id]['data']
            if entries:
                latest = entries[0]  # 最新（先頭）
                result['moisture_turf_goal'] = latest.get('turf_goal')
                result['moisture_turf_4c'] = latest.get('turf_4c')
                result['moisture_dirt_goal'] = latest.get('dirt_goal')
                result['moisture_dirt_4c'] = latest.get('dirt_4c')

    except Exception as e:
        print(f"  [JRA Track] {course_name}: {e}")

    return result


def get_moisture_rate(track_info, surface):
    """含水率を取得（芝/ダート切替）"""
    if surface == '芝':
        vals = [track_info.get('moisture_turf_goal'), track_info.get('moisture_turf_4c')]
    else:
        vals = [track_info.get('moisture_dirt_goal'), track_info.get('moisture_dirt_4c')]
    vals = [v for v in vals if v is not None]
    if vals:
        return sum(vals) / len(vals)
    return None


if __name__ == '__main__':
    import json
    for course in ['中山', '阪神', '中京']:
        print(f"\n{'='*40}")
        print(f"  {course}")
        print(f"{'='*40}")
        info = fetch_jra_track_info(course)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        mr_turf = get_moisture_rate(info, '芝')
        mr_dirt = get_moisture_rate(info, 'ダ')
        print(f"  含水率平均: 芝={mr_turf}, ダート={mr_dirt}")
