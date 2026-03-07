"""
騎手データ自動更新スクリプト
netkeibaのレース出馬表から騎手IDを収集し、各騎手プロフィールから勝率を取得。
"""
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOCKEY_WR_PATH = os.path.join(SCRIPT_DIR, "jockey_wr.json")
UPDATE_LOG_PATH = os.path.join(SCRIPT_DIR, "jockey_wr_updated.txt")


def collect_jockey_ids_from_races():
    """直近のレースから騎手名とIDを収集する"""
    jockey_ids = {}
    now = datetime.now()
    # 直近の開催日のレースIDパターンを生成
    # JRA: 2025年なら 2025XXYYZZ (XX=場所, YY=開催, ZZ=日+R)
    # 最近のレースから取得する方が確実
    sample_race_ids = []
    # 主要場のレースIDを試す (東京、中山、阪神、京都、中京、小倉)
    venues = ['05', '06', '08', '07', '09', '10']  # 東京、中山、阪神、京都、小倉、中京
    year = now.year
    for venue in venues:
        for kai in ['01', '02', '03', '04', '05']:
            for day_race in ['0811', '0812', '0111', '0112']:
                sample_race_ids.append(f"{year}{venue}{kai}{day_race}")
    # 最大20レースから騎手を収集
    tried = 0
    for race_id in sample_race_ids:
        if tried >= 20 or len(jockey_ids) >= 100:
            break
        try:
            url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = "EUC-JP"
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=lambda h: h and "/jockey/result/recent/" in str(h))
            if not links:
                continue
            tried += 1
            for a in links:
                href = a.get("href", "")
                name = a.get_text(strip=True)
                m = re.search(r"/jockey/result/recent/(\d+)/", href)
                if m and name and len(name) > 1:
                    jockey_ids[name] = m.group(1)
            time.sleep(0.3)
        except Exception:
            continue
    return jockey_ids


def fetch_jockey_win_rate(jockey_id):
    """個別騎手のプロフィールページから勝率を取得"""
    try:
        url = f"https://db.netkeiba.com/jockey/{jockey_id}/"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        now = datetime.now()
        current_year = str(now.year)
        last_year = str(now.year - 1)
        # ResultsByYears テーブル（中央成績 = 最初のテーブル）
        for table in soup.find_all("table", class_="ResultsByYears"):
            rows = table.find_all("tr")
            year_data = {}
            total_wr = None
            for row in rows[1:]:
                tds = row.find_all(["th", "td"])
                vals = [td.get_text(strip=True) for td in tds]
                if not vals or len(vals) < 5:
                    continue
                row_key = vals[0]
                # 勝率(xx.x％)を探す
                wr_from_text = None
                for v in vals:
                    m = re.match(r'^(\d+\.?\d*)[％%]$', v)
                    if m:
                        wr_from_text = float(m.group(1)) / 100.0
                        break
                # 数値を抽出（カンマ除去）
                nums = []
                for v in vals[1:]:
                    clean = v.replace(",", "")
                    if clean.isdigit():
                        nums.append(int(clean))
                # 年度行（最低騎乗数10以上で信頼性確保）
                if re.match(r'^\d{4}$', row_key):
                    ride_count = sum(nums[:4]) if len(nums) >= 4 else 0
                    if wr_from_text is not None and wr_from_text > 0 and ride_count >= 10:
                        year_data[row_key] = wr_from_text
                    elif len(nums) >= 4 and ride_count >= 10:
                        wins = nums[0]
                        year_data[row_key] = wins / ride_count
                # 累計行
                elif any(c in row_key for c in ["計", "累"]):
                    if wr_from_text is not None and wr_from_text > 0:
                        total_wr = wr_from_text
                    elif len(nums) >= 4 and sum(nums[:4]) > 0:
                        total_wr = nums[0] / sum(nums[:4])
            # 優先: 昨年(十分データ) > 今年 > 累計
            # 年初は今年のサンプルが少ないので昨年を優先
            if last_year in year_data and year_data[last_year] > 0:
                return year_data[last_year]
            if current_year in year_data and year_data[current_year] > 0:
                return year_data[current_year]
            if total_wr is not None and total_wr > 0:
                return total_wr
            break
    except Exception:
        pass
    return None


def update_jockey_wr():
    """jockey_wr.jsonを更新する"""
    existing = {}
    if os.path.exists(JOCKEY_WR_PATH):
        try:
            with open(JOCKEY_WR_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    print(f"Existing entries: {len(existing)}")

    # レースから騎手IDを収集
    print("Collecting jockey IDs from recent races...")
    jockey_ids = collect_jockey_ids_from_races()
    print(f"  Found {len(jockey_ids)} jockeys")

    if not jockey_ids:
        print("No jockeys found. Skipping update.")
        # 更新日時だけ記録（リトライ防止）
        now = datetime.now()
        with open(UPDATE_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(now.strftime("%Y-%m-%d %H:%M:%S"))
        return len(existing)

    merged = existing.copy()
    updated = 0
    added = 0

    for name, jid in jockey_ids.items():
        wr = fetch_jockey_win_rate(jid)
        if wr is not None and wr > 0:
            for key in set([name, name[:3]]):
                old = merged.get(key)
                if old is not None:
                    if abs(old - wr) > 0.001:
                        updated += 1
                else:
                    added += 1
                merged[key] = round(wr, 6)
            print(f"  {name}: {wr:.3f}")
        time.sleep(0.3)

    # 保存
    with open(JOCKEY_WR_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=None)

    now = datetime.now()
    with open(UPDATE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(now.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"Updated: {updated}, Added: {added}, Total: {len(merged)}")
    return len(merged)


def get_last_update_time():
    """最終更新日時を取得"""
    if os.path.exists(UPDATE_LOG_PATH):
        try:
            with open(UPDATE_LOG_PATH, "r", encoding="utf-8") as f:
                return datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return None


def needs_update(days=7):
    """更新が必要かどうか"""
    last = get_last_update_time()
    if last is None:
        return True
    return (datetime.now() - last).days >= days


if __name__ == "__main__":
    if needs_update(days=7):
        print("Jockey data needs update.")
        update_jockey_wr()
    else:
        last = get_last_update_time()
        print(f"Data is up to date (last updated: {last})")
