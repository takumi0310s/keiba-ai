"""JRA公式サイトから馬場情報を取得

取得データ:
- クッション値（芝レースの場合）
- 含水率（芝ゴール前/4コーナー、ダートゴール前/4コーナー）
- 馬場状態

JRA馬場情報ページ: https://www.jra.go.jp/keiba/baba/
"""
import requests
from bs4 import BeautifulSoup
import re

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 競馬場名 → JRA公式URLパス
COURSE_TO_JRA = {
    '札幌': 'sapporo', '函館': 'hakodate', '福島': 'fukushima',
    '新潟': 'niigata', '東京': 'tokyo', '中山': 'nakayama',
    '中京': 'chukyo', '京都': 'kyoto', '阪神': 'hanshin', '小倉': 'kokura',
}


def fetch_jra_track_info(course_name):
    """JRA公式から馬場情報を取得

    Args:
        course_name: 競馬場名（例: '東京', '阪神'）

    Returns:
        dict: {
            'cushion_value': float or None,  # クッション値（芝のみ）
            'moisture_turf_goal': float or None,  # 含水率 芝ゴール前
            'moisture_turf_4c': float or None,    # 含水率 芝4コーナー
            'moisture_dirt_goal': float or None,   # 含水率 ダートゴール前
            'moisture_dirt_4c': float or None,     # 含水率 ダート4コーナー
            'condition_turf': str or None,  # 芝馬場状態
            'condition_dirt': str or None,  # ダート馬場状態
            'source': 'jra',
        }
    """
    result = {
        'cushion_value': None,
        'moisture_turf_goal': None, 'moisture_turf_4c': None,
        'moisture_dirt_goal': None, 'moisture_dirt_4c': None,
        'condition_turf': None, 'condition_dirt': None,
        'source': 'jra',
    }

    jra_code = COURSE_TO_JRA.get(course_name)
    if not jra_code:
        return result

    try:
        # JRA馬場情報トップページ（含水率テーブルあり）
        url = "https://www.jra.go.jp/keiba/baba/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # テーブルから含水率・クッション値を抽出
        _parse_baba_tables(soup, result)

        # 開催場のリンクを探す（概要ページ）
        target_link = None
        for a in soup.find_all('a', href=True):
            href = a['href']
            if jra_code in href and 'baba' in href:
                target_link = href
                break

        if target_link:
            if target_link.startswith('/'):
                target_link = f"https://www.jra.go.jp{target_link}"
            resp2 = requests.get(target_link, headers=HEADERS, timeout=10)
            resp2.encoding = 'utf-8'
            soup2 = BeautifulSoup(resp2.text, 'html.parser')
            _parse_baba_page(soup2, course_name, result)
        else:
            _parse_baba_page(soup, course_name, result)

    except Exception as e:
        print(f"  [JRA Track] {course_name}: {e}")

    return result


def _parse_baba_tables(soup, result):
    """JRA馬場トップページのテーブルからデータ抽出
    Table 0: クッション値説明
    Table 1: 含水率（ゴール前/4コーナー × 芝/ダート）
    Table 2: 含水率→馬場状態 対応表
    """
    tables = soup.find_all('table')

    # Table 1: 含水率テーブル（3行: ヘッダー, 芝, ダート）
    if len(tables) > 1:
        rows = tables[1].find_all('tr')
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(['th', 'td'])]
            if len(cells) >= 3:
                nums = []
                for c in cells[1:]:
                    m = re.search(r'([\d.]+)', c)
                    if m:
                        nums.append(float(m.group(1)))

                row_text = cells[0]
                if any(k in row_text for k in ['芝', 'turf']):
                    if len(nums) >= 1:
                        result['moisture_turf_goal'] = nums[0]
                    if len(nums) >= 2:
                        result['moisture_turf_4c'] = nums[1]
                elif any(k in row_text for k in ['ダ', 'dirt']):
                    if len(nums) >= 1:
                        result['moisture_dirt_goal'] = nums[0]
                    if len(nums) >= 2:
                        result['moisture_dirt_4c'] = nums[1]

    # クッション値を探す（ページ内テキストから）
    text = soup.get_text()
    # "クッション値 9.5" のようなパターン
    for m in re.finditer(r'([\d.]+)', text):
        v = float(m.group(1))
        if 5.0 <= v <= 15.0:
            # クッション値の文脈かチェック
            start = max(0, m.start() - 50)
            context = text[start:m.start()]
            if 'クッション' in context or 'cushion' in context.lower():
                result['cushion_value'] = v
                break


def _parse_baba_page(soup, course_name, result):
    """馬場情報ページからデータを抽出"""
    text = soup.get_text()

    # クッション値: "クッション値 : 9.5" のようなパターン
    cushion_match = re.search(r'クッション値\s*[:：]?\s*([\d.]+)', text)
    if cushion_match:
        try:
            result['cushion_value'] = float(cushion_match.group(1))
        except ValueError:
            pass

    # 含水率: "含水率" の後に数値が続くパターン
    # 芝ゴール前、芝4コーナー、ダートゴール前、ダート4コーナー
    moisture_patterns = [
        (r'芝.*?ゴール前\s*[:：]?\s*([\d.]+)\s*%', 'moisture_turf_goal'),
        (r'芝.*?4コーナー\s*[:：]?\s*([\d.]+)\s*%', 'moisture_turf_4c'),
        (r'ダート.*?ゴール前\s*[:：]?\s*([\d.]+)\s*%', 'moisture_dirt_goal'),
        (r'ダート.*?4コーナー\s*[:：]?\s*([\d.]+)\s*%', 'moisture_dirt_4c'),
    ]
    for pattern, key in moisture_patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass

    # 含水率テーブルからの抽出（テーブル形式の場合）
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(['th', 'td'])]
            cell_text = ' '.join(cells)

            # 含水率の数値を探す
            if '含水率' in cell_text or 'moisture' in cell_text.lower():
                nums = re.findall(r'([\d.]+)\s*%?', cell_text)
                if len(nums) >= 2:
                    if '芝' in cell_text:
                        result['moisture_turf_goal'] = float(nums[0])
                        if len(nums) >= 2:
                            result['moisture_turf_4c'] = float(nums[1])
                    elif 'ダート' in cell_text:
                        result['moisture_dirt_goal'] = float(nums[0])
                        if len(nums) >= 2:
                            result['moisture_dirt_4c'] = float(nums[1])

            # クッション値
            if 'クッション' in cell_text:
                nums = re.findall(r'([\d.]+)', cell_text)
                for n in nums:
                    v = float(n)
                    if 5.0 <= v <= 15.0:  # クッション値の妥当範囲
                        result['cushion_value'] = v
                        break

    # 馬場状態
    for pattern, key in [
        (r'芝\s*[:：]?\s*(良|稍重|稍|重|不良)', 'condition_turf'),
        (r'ダート\s*[:：]?\s*(良|稍重|稍|重|不良)', 'condition_dirt'),
    ]:
        m = re.search(pattern, text)
        if m:
            result[key] = m.group(1)


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
    for course in ['東京', '中山', '阪神']:
        print(f"\n{'='*40}")
        print(f"  {course}")
        print(f"{'='*40}")
        info = fetch_jra_track_info(course)
        print(json.dumps(info, ensure_ascii=False, indent=2))
