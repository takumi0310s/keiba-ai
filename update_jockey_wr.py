"""
騎手・調教師データ自動更新スクリプト
netkeibaから最新の騎手勝率・調教師成績を取得してjockey_wr.jsonを更新する。
"""
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOCKEY_WR_PATH = os.path.join(SCRIPT_DIR, "jockey_wr.json")
UPDATE_LOG_PATH = os.path.join(SCRIPT_DIR, "jockey_wr_updated.txt")


def fetch_jockey_rankings(year=None):
    """netkeibaの騎手リーディングページから騎手勝率を取得"""
    if year is None:
        year = datetime.now().year
    jockey_data = {}
    # JRA騎手リーディング (複数ページ)
    for page in range(1, 4):
        try:
            url = f"https://db.netkeiba.com//?pid=jockey_leading&year={year}&page={page}"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = "EUC-JP"
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", class_="nk_tb_common")
            if not table:
                tables = soup.find_all("table")
                table = next((t for t in tables if t.find("th", string=re.compile("勝率"))), None)
            if not table:
                break
            rows = table.find_all("tr")
            found = 0
            for row in rows[1:]:  # skip header
                tds = row.find_all("td")
                if len(tds) < 6:
                    continue
                # 騎手名
                name_td = row.find("a", href=re.compile(r"/jockey/"))
                if not name_td:
                    continue
                name = name_td.get_text(strip=True)
                # 勝率を探す: 1着数/出走数
                try:
                    # 各列を検査して勝率を計算
                    texts = [td.get_text(strip=True) for td in tds]
                    # 典型的なカラム: 順位, 騎手名, 1着, 2着, 3着, 着外, 出走, 勝率, 連対率, 複勝率
                    wins = 0
                    total = 0
                    win_rate = 0.0
                    for i, t in enumerate(texts):
                        # 勝率が直接書いてある場合 (xx.x% or 0.xxx)
                        m = re.match(r'^(\d+\.\d+)%?$', t)
                        if m and 0 < float(m.group(1)) < 100 and i >= 5:
                            # これが勝率の可能性
                            val = float(m.group(1))
                            if val > 1:
                                win_rate = val / 100.0
                            else:
                                win_rate = val
                            break
                    if win_rate == 0:
                        # 1着数と出走数から計算
                        nums = []
                        for t in texts:
                            if t.isdigit():
                                nums.append(int(t))
                        if len(nums) >= 5:
                            # nums[0]=順位, nums[1]=1着, ..., nums[4]=出走
                            wins = nums[1] if len(nums) > 1 else 0
                            total = nums[-1] if nums[-1] > 10 else (nums[4] if len(nums) > 4 else 0)
                            if total > 0:
                                win_rate = wins / total
                    if name and (win_rate > 0 or total > 0):
                        # 名前を3文字に短縮（既存jsonの形式に合わせる）
                        short_name = name[:3] if len(name) > 3 else name
                        jockey_data[short_name] = round(win_rate, 6)
                        jockey_data[name] = round(win_rate, 6)
                        found += 1
                except (ValueError, IndexError):
                    continue
            if found == 0:
                break
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    return jockey_data


def fetch_trainer_rankings(year=None):
    """netkeibaの調教師リーディングから調教師勝率を取得"""
    if year is None:
        year = datetime.now().year
    trainer_data = {}
    for page in range(1, 3):
        try:
            url = f"https://db.netkeiba.com//?pid=trainer_leading&year={year}&page={page}"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = "EUC-JP"
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", class_="nk_tb_common")
            if not table:
                tables = soup.find_all("table")
                table = next((t for t in tables if t.find("th", string=re.compile("勝率"))), None)
            if not table:
                break
            rows = table.find_all("tr")
            found = 0
            for row in rows[1:]:
                tds = row.find_all("td")
                if len(tds) < 6:
                    continue
                name_td = row.find("a", href=re.compile(r"/trainer/"))
                if not name_td:
                    continue
                name = name_td.get_text(strip=True)
                try:
                    texts = [td.get_text(strip=True) for td in tds]
                    win_rate = 0.0
                    for i, t in enumerate(texts):
                        m = re.match(r'^(\d+\.\d+)%?$', t)
                        if m and 0 < float(m.group(1)) < 100 and i >= 5:
                            val = float(m.group(1))
                            win_rate = val / 100.0 if val > 1 else val
                            break
                    if win_rate == 0:
                        nums = [int(t) for t in texts if t.isdigit()]
                        if len(nums) >= 5:
                            wins = nums[1] if len(nums) > 1 else 0
                            total = nums[-1] if nums[-1] > 10 else (nums[4] if len(nums) > 4 else 0)
                            if total > 0:
                                win_rate = wins / total
                    if name and (win_rate > 0):
                        trainer_data[name] = round(win_rate, 6)
                        found += 1
                except (ValueError, IndexError):
                    continue
            if found == 0:
                break
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching trainer page {page}: {e}")
            break
    return trainer_data


def update_jockey_wr():
    """jockey_wr.jsonを更新する"""
    # 既存データ読み込み
    existing = {}
    if os.path.exists(JOCKEY_WR_PATH):
        try:
            with open(JOCKEY_WR_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    print(f"Existing entries: {len(existing)}")

    # 今年と昨年のデータを取得
    now = datetime.now()
    current_year = now.year

    print(f"Fetching {current_year} jockey rankings...")
    new_jockeys = fetch_jockey_rankings(current_year)
    print(f"  Got {len(new_jockeys)} jockeys")

    if now.month <= 3:
        print(f"Fetching {current_year - 1} jockey rankings (early year supplement)...")
        prev_jockeys = fetch_jockey_rankings(current_year - 1)
        print(f"  Got {len(prev_jockeys)} jockeys from last year")
        # 昨年のデータは今年のデータで上書きされない場合のみ使用
        for k, v in prev_jockeys.items():
            if k not in new_jockeys:
                new_jockeys[k] = v

    print(f"Fetching {current_year} trainer rankings...")
    new_trainers = fetch_trainer_rankings(current_year)
    print(f"  Got {len(new_trainers)} trainers")

    # マージ（新しいデータで上書き、既存データは保持）
    merged = existing.copy()
    updated = 0
    added = 0
    for k, v in new_jockeys.items():
        if k in merged:
            if abs(merged[k] - v) > 0.001:
                updated += 1
        else:
            added += 1
        merged[k] = v

    # 調教師データもマージ（プレフィックス付き）
    for k, v in new_trainers.items():
        key = f"T_{k}"
        merged[key] = v

    # 保存
    with open(JOCKEY_WR_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=None)

    # 更新日時を記録
    with open(UPDATE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(now.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"Updated: {updated}, Added: {added}, Total: {len(merged)}")
    print(f"Saved to {JOCKEY_WR_PATH}")
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
    """更新が必要かどうか（前回更新から指定日数経過）"""
    last = get_last_update_time()
    if last is None:
        return True
    return (datetime.now() - last).days >= days


if __name__ == "__main__":
    if needs_update(days=7):
        print("Jockey/Trainer data needs update.")
        update_jockey_wr()
    else:
        last = get_last_update_time()
        print(f"Data is up to date (last updated: {last})")
