"""天気予報 API fetcher (Session #40 E3、 試作).

5/9 等の前日 (5/8 evening) 〜 当日朝に、 開催場の 24h 前 天気予報を取得し、
馬場予測精度向上 (馬場発表前段階) に利用する。

データ source 候補:
1. 気象庁 API (公式、 無料、 https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json)
   既存 scrape_weather.py の主軸
2. OpenWeatherMap (商用、 無料枠 60 req/min、 https://openweathermap.org)
3. tenki.jp (Web scraping、 BAN リスク有)

本 script は 気象庁 API のみ使用 (公式無料、 信頼性高)。

開催場 → 気象庁 area_code mapping:
  中山 (千葉) → 120010
  東京 (東京) → 130010
  京都 (京都) → 260010
  阪神 (兵庫) → 280010
  中京 (愛知) → 230010
  小倉 (福岡) → 400010
  福島 (福島) → 070010
  新潟 (新潟) → 150010
  札幌 (北海道) → 016010
  函館 (北海道) → 017010

usage:
  python tools/weather_fetcher.py --course 東京 --date 20260509
  python tools/weather_fetcher.py --course 京都 --hours 24

V15 production 完全不変 (read-only)。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import requests

BASE = Path(r"C:/Users/takum/keiba-ai")

COURSE_TO_AREA = {
    # 都道府県 area_code (JMA forecast の root level、 末尾 "000")
    "中山": "120000", "千葉": "120000",
    "東京": "130000",
    "京都": "260000",
    "阪神": "280000", "兵庫": "280000",
    "中京": "230000", "愛知": "230000",
    "小倉": "400000", "福岡": "400000",
    "福島": "070000",
    "新潟": "150000",
    "札幌": "016000", "北海道": "016000",
    "函館": "017000",
}


def fetch_jma(area_code: str) -> dict | None:
    """気象庁 公式 forecast API."""
    url = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[weather] JMA fetch error: {e}", file=sys.stderr)
        return None


def parse_jma_for_date(jma_data: dict, target_date: str) -> dict:
    """JMA forecast から target_date (YYYYMMDD) の予報を抽出."""
    out = {
        "target_date": target_date,
        "weather": None,
        "wind": None,
        "wave": None,
        "min_temp": None,
        "max_temp": None,
        "precip_prob": None,
        "raw_text": None,
    }
    if not jma_data: return out
    try:
        # JMA forecast は時系列 list。 [0] が "今日明日明後日" の概況
        # weatherCodes / weathers / winds / waves
        first = jma_data[0]  # 主要発表
        time_series = first.get("timeSeries", [])

        target_dt = datetime.datetime.strptime(target_date, "%Y%m%d").date()

        for ts in time_series:
            time_defs = ts.get("timeDefines", [])
            areas = ts.get("areas", [])
            if not areas: continue
            for area in areas:
                # 時刻 list と data list を zip して target_date を find
                for i, time_str in enumerate(time_defs):
                    try:
                        dt = datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00")).date()
                    except Exception:
                        continue
                    if dt == target_dt:
                        if "weathers" in area and i < len(area["weathers"]):
                            out["weather"] = area["weathers"][i]
                        if "winds" in area and i < len(area["winds"]):
                            out["wind"] = area["winds"][i]
                        if "waves" in area and i < len(area["waves"]):
                            out["wave"] = area["waves"][i]
                        if "pops" in area and i < len(area["pops"]):
                            try: out["precip_prob"] = int(area["pops"][i])
                            except Exception: pass
                        if "tempsMin" in area and i < len(area["tempsMin"]):
                            try: out["min_temp"] = float(area["tempsMin"][i]) if area["tempsMin"][i] else None
                            except Exception: pass
                        if "tempsMax" in area and i < len(area["tempsMax"]):
                            try: out["max_temp"] = float(area["tempsMax"][i]) if area["tempsMax"][i] else None
                            except Exception: pass
        out["raw_text"] = first.get("publishingOffice", "")
        return out
    except Exception as e:
        print(f"[weather] parse error: {e}", file=sys.stderr)
        return out


def predict_track_condition(weather: str | None, precip_prob: int | None) -> str:
    """雑な馬場予測 (天気 + 降水確率 から 良/稍重/重/不良 推定)."""
    if not weather: return "unknown"
    w = weather.lower()
    if "雪" in w or precip_prob and precip_prob >= 80:
        return "不良"
    if "雨" in w or (precip_prob and precip_prob >= 60):
        return "重"
    if "曇" in w and precip_prob and precip_prob >= 30:
        return "稍重"
    return "良"


def main():
    p = argparse.ArgumentParser(description="天気予報 API fetcher (Session #40 E3)")
    p.add_argument("--course", required=True)
    p.add_argument("--date", default=None, help="YYYYMMDD (省略: 明日)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    code = COURSE_TO_AREA.get(args.course)
    if not code:
        print(f"[weather] unknown course: {args.course}", file=sys.stderr)
        sys.exit(1)

    target = args.date or (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y%m%d")
    print(f"[weather] course={args.course} (area={code}), target={target}")

    data = fetch_jma(code)
    if not data:
        print("[weather] API fetch fail", file=sys.stderr)
        sys.exit(2)

    parsed = parse_jma_for_date(data, target)
    parsed["course"] = args.course
    parsed["area_code"] = code
    parsed["predicted_track"] = predict_track_condition(parsed.get("weather"), parsed.get("precip_prob"))

    print(f"\n=== 予報 ({args.course}, {target}) ===")
    print(f"  weather: {parsed['weather']}")
    print(f"  wind:    {parsed['wind']}")
    print(f"  wave:    {parsed['wave']}")
    print(f"  precip:  {parsed['precip_prob']}%")
    print(f"  temp:    {parsed['min_temp']} - {parsed['max_temp']} °C")
    print(f"  推定 track condition: {parsed['predicted_track']}")
    print(f"  source: 気象庁 API ({parsed.get('raw_text', '')})")

    if args.out:
        out_path = BASE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
