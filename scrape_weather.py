"""気象庁APIから競馬場周辺の天候データを取得

気象庁API:
- 予報: https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json
- アメダス: https://www.jma.go.jp/bosai/amedas/

取得データ:
- 気温, 湿度, 風速, 降水量
- 天候（晴/曇/雨/雪）
"""
import requests
import json
import re
from datetime import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 競馬場 → 気象庁エリアコード（都道府県単位）
COURSE_TO_AREA = {
    '札幌': '016000',   # 石狩地方
    '函館': '017000',   # 渡島地方
    '福島': '070000',   # 福島県
    '新潟': '150000',   # 新潟県
    '東京': '130000',   # 東京都
    '中山': '120000',   # 千葉県
    '中京': '230000',   # 愛知県
    '京都': '260000',   # 京都府
    '阪神': '280000',   # 兵庫県
    '小倉': '400000',   # 福岡県
}

# 競馬場 → 最寄りアメダス観測所コード
COURSE_TO_AMEDAS = {
    '札幌': '14163',    # 札幌
    '函館': '23232',    # 函館
    '福島': '36127',    # 福島
    '新潟': '54232',    # 新潟
    '東京': '44132',    # 府中
    '中山': '45147',    # 船橋
    '中京': '51106',    # 名古屋
    '京都': '61286',    # 京都
    '阪神': '63518',    # 宝塚（なければ神戸63437）
    '小倉': '82182',    # 北九州（小倉）
}

# 天気コード → テキスト（気象庁予報用）
WEATHER_CODES = {
    100: '晴', 101: '晴時々曇', 102: '晴一時雨', 103: '晴時々雨',
    104: '晴一時雪', 110: '晴後曇', 111: '晴後曇一時雨',
    200: '曇', 201: '曇時々晴', 202: '曇一時雨', 203: '曇時々雨',
    204: '曇一時雪', 205: '曇時々雪', 210: '曇後晴', 211: '曇後雨',
    300: '雨', 301: '雨時々晴', 302: '雨時々曇', 303: '雨時々雪',
    311: '雨後晴', 313: '雨後曇',
    400: '雪', 401: '雪時々晴', 402: '雪時々曇', 403: '雪時々雨',
}


def fetch_jma_forecast(course_name):
    """気象庁予報APIから天候データを取得

    Args:
        course_name: 競馬場名（例: '東京', '阪神'）

    Returns:
        dict: {
            'temperature': float or None,     # 気温 (℃)
            'humidity': float or None,         # 湿度 (%)
            'wind_speed': float or None,       # 風速 (m/s)
            'precipitation': float or None,    # 降水量 (mm)
            'weather_text': str or None,       # 天気テキスト
            'weather_code': int or None,       # 天気コード
            'wind_direction': str or None,     # 風向
            'source': 'jma_forecast',
        }
    """
    result = {
        'temperature': None, 'humidity': None,
        'wind_speed': None, 'precipitation': None,
        'weather_text': None, 'weather_code': None,
        'wind_direction': None,
        'source': 'jma_forecast',
    }

    area_code = COURSE_TO_AREA.get(course_name)
    if not area_code:
        return result

    try:
        url = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data or len(data) == 0:
            return result

        # 短期予報（最初の要素）
        forecast = data[0]
        time_series = forecast.get('timeSeries', [])

        now = datetime.now()
        now_hour = now.hour

        # timeSeries[0]: 天気・風
        if len(time_series) > 0:
            ts0 = time_series[0]
            time_defines = ts0.get('timeDefines', [])
            areas = ts0.get('areas', [])

            if areas:
                area = areas[0]  # 最初のエリア（通常は該当地域）
                weathers = area.get('weathers', [])
                winds = area.get('winds', [])
                weather_codes = area.get('weatherCodes', [])

                # 現在時刻に最も近い予報を選択
                best_idx = _find_closest_time_index(time_defines, now)

                if best_idx < len(weathers):
                    result['weather_text'] = weathers[best_idx]

                if best_idx < len(weather_codes):
                    try:
                        code = int(weather_codes[best_idx])
                        result['weather_code'] = code
                    except (ValueError, TypeError):
                        pass

                if best_idx < len(winds):
                    wind_text = winds[best_idx]
                    # 風速パース: "北の風 やや強く" → 風向と強さ
                    result['wind_direction'] = _parse_wind_direction(wind_text)
                    result['wind_speed'] = _parse_wind_speed(wind_text)

        # timeSeries[1]: 降水確率（6時間ごと）
        if len(time_series) > 1:
            ts1 = time_series[1]
            areas1 = ts1.get('areas', [])
            if areas1:
                pops = areas1[0].get('pops', [])
                # 降水確率から概算降水量を推定
                if pops:
                    best_idx1 = min(len(pops) - 1, now_hour // 6)
                    try:
                        pop = int(pops[best_idx1])
                        # 降水確率 → 概算降水量（簡易推定）
                        if pop >= 80:
                            result['precipitation'] = 5.0
                        elif pop >= 60:
                            result['precipitation'] = 2.0
                        elif pop >= 40:
                            result['precipitation'] = 0.5
                        else:
                            result['precipitation'] = 0.0
                    except (ValueError, TypeError):
                        pass

        # timeSeries[2]: 気温
        if len(time_series) > 2:
            ts2 = time_series[2]
            areas2 = ts2.get('areas', [])
            if areas2:
                temps = areas2[0].get('temps', [])
                if temps:
                    try:
                        # temps通常は[最低, 最高]のペア
                        temp_vals = [float(t) for t in temps if t]
                        if temp_vals:
                            # 現在時刻に応じて最低/最高の中間を使用
                            if len(temp_vals) >= 2:
                                t_min, t_max = min(temp_vals), max(temp_vals)
                                # 時刻に応じた補間
                                if now_hour < 6:
                                    result['temperature'] = t_min
                                elif now_hour < 14:
                                    ratio = (now_hour - 6) / 8.0
                                    result['temperature'] = t_min + (t_max - t_min) * ratio
                                else:
                                    ratio = (now_hour - 14) / 10.0
                                    result['temperature'] = t_max - (t_max - t_min) * ratio * 0.5
                            else:
                                result['temperature'] = temp_vals[0]
                    except (ValueError, TypeError):
                        pass

        # アメダスから湿度・風速の実測値を取得
        _fetch_amedas_data(course_name, result)

    except Exception as e:
        print(f"  [JMA Forecast] {course_name}: {e}")

    return result


def _fetch_amedas_data(course_name, result):
    """アメダス観測データから実測値を取得"""
    station = COURSE_TO_AMEDAS.get(course_name)
    if not station:
        return

    try:
        # 最新観測時刻を取得
        time_url = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
        time_resp = requests.get(time_url, headers=HEADERS, timeout=5)
        latest_time = time_resp.text.strip()
        # "2026-03-09T21:00:00+09:00" → "20260309_21"
        dt = datetime.fromisoformat(latest_time)
        time_key = dt.strftime("%Y%m%d_%H")

        # アメダスデータ取得
        amedas_url = f"https://www.jma.go.jp/bosai/amedas/data/point/{station}/{time_key}.json"
        resp = requests.get(amedas_url, headers=HEADERS, timeout=5)
        if resp.status_code != 200:
            return
        data = resp.json()

        if not data:
            return

        # 最新の観測時刻のデータを取得
        latest_key = sorted(data.keys())[-1] if data else None
        if not latest_key:
            return

        obs = data[latest_key]

        # 気温
        if 'temp' in obs and obs['temp'] and len(obs['temp']) > 0:
            result['temperature'] = obs['temp'][0]

        # 湿度
        if 'humidity' in obs and obs['humidity'] and len(obs['humidity']) > 0:
            result['humidity'] = obs['humidity'][0]

        # 風速
        if 'wind' in obs and obs['wind'] and len(obs['wind']) > 0:
            result['wind_speed'] = obs['wind'][0]

        # 降水量（10分降水量 → 時間換算）
        if 'precipitation10m' in obs and obs['precipitation10m'] and len(obs['precipitation10m']) > 0:
            p10 = obs['precipitation10m'][0]
            if p10 is not None:
                result['precipitation'] = p10 * 6  # 10分→1時間換算

        # 風向
        if 'windDirection' in obs and obs['windDirection'] and len(obs['windDirection']) > 0:
            wd = obs['windDirection'][0]
            directions = ['北', '北北東', '北東', '東北東', '東', '東南東', '南東', '南南東',
                          '南', '南南西', '南西', '西南西', '西', '西北西', '北西', '北北西']
            if isinstance(wd, (int, float)) and 1 <= wd <= 16:
                result['wind_direction'] = directions[int(wd) - 1]

    except Exception as e:
        # アメダス取得失敗は無視（予報データで代替）
        pass


def _find_closest_time_index(time_defines, now):
    """現在時刻に最も近い予報時刻のインデックスを返す"""
    if not time_defines:
        return 0
    best_idx = 0
    best_diff = float('inf')
    for i, td in enumerate(time_defines):
        try:
            t = datetime.fromisoformat(td.replace('Z', '+00:00'))
            diff = abs((now - t.replace(tzinfo=None)).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        except Exception:
            pass
    return best_idx


def _parse_wind_direction(wind_text):
    """風テキストから風向を抽出"""
    if not wind_text:
        return None
    directions = ['北北西', '北北東', '北西', '北東', '南南西', '南南東',
                  '南西', '南東', '西北西', '西南西', '東北東', '東南東',
                  '北', '南', '東', '西']
    for d in directions:
        if d in wind_text:
            return d
    return None


def _parse_wind_speed(wind_text):
    """風テキストから風速を推定"""
    if not wind_text:
        return None
    if '非常に強' in wind_text:
        return 15.0
    elif '強く' in wind_text or '強い' in wind_text:
        return 10.0
    elif 'やや強' in wind_text:
        return 7.0
    elif '弱く' in wind_text or '弱い' in wind_text:
        return 2.0
    return 4.0  # デフォルト（平均的な風）


def get_weather_features(course_name):
    """競馬場の天候特徴量を取得（統合関数）

    Returns:
        dict: {
            'temperature': float,      # 気温 (℃) デフォルト15.0
            'humidity': float,          # 湿度 (%) デフォルト60.0
            'wind_speed': float,        # 風速 (m/s) デフォルト3.0
            'precipitation': float,     # 降水量 (mm/h) デフォルト0.0
            'weather_text': str,        # 天気テキスト
            'wind_direction': str,      # 風向
        }
    """
    data = fetch_jma_forecast(course_name)

    return {
        'temperature': data.get('temperature') or 15.0,
        'humidity': data.get('humidity') or 60.0,
        'wind_speed': data.get('wind_speed') or 3.0,
        'precipitation': data.get('precipitation') or 0.0,
        'weather_text': data.get('weather_text') or '',
        'wind_direction': data.get('wind_direction') or '',
    }


if __name__ == '__main__':
    for course in ['東京', '中山', '阪神', '京都', '小倉']:
        print(f"\n{'='*50}")
        print(f"  {course}")
        print(f"{'='*50}")
        features = get_weather_features(course)
        for k, v in features.items():
            print(f"  {k}: {v}")
