# -*- coding: utf-8 -*-
"""
Phase C-3: netkeiba 結果ページ流用バックフィル（ユーザー再承認済 2026-08-08）。
daily_results.py の結果ページ解析ロジックを流用し、per-horse の
  着順 / 確定単勝オッズ / 人気 / 複勝払戻 + レース配当
を全20日(ツール窓)分取得する。

- 取得先: race.netkeiba.com（Super Premium 正規利用・結果ページは公開）
- 全 HTTP 間 2秒スリープ厳守 / 再開可能
- 保存: research/ruiji/raw_results/<date>.json（★data/ には一切書かない★）
"""
import sys, os, re, json, time, io
sys.stdout.reconfigure(encoding="utf-8")
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "raw_results"); os.makedirs(OUT, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ruiji-research/1.0"}
INTERVAL = 2.1
DATES = ["20260606","20260607","20260613","20260614","20260620","20260621","20260627","20260628",
         "20260704","20260705","20260711","20260712","20260718","20260719","20260725","20260726",
         "20260801","20260802","20260808","20260809"]

_last = [0.0]
def wait():
    dt = time.time() - _last[0]
    if dt < INTERVAL: time.sleep(INTERVAL - dt)
    _last[0] = time.time()

def _norm(s): return __import__("unicodedata").normalize("NFKC", s or "")

def get(url):
    wait()
    r = requests.get(url, headers=HEADERS, timeout=20)
    head = r.content[:4096].decode("ascii", "ignore").lower()
    m = re.search(r'charset=["\']?([\w\-]+)', head)
    r.encoding = (m.group(1) if m else None) or r.apparent_encoding or "EUC-JP"
    return r.text

def race_ids_for(date):
    """kaisai の全 JRA race_id(12桁) を取得。"""
    html = get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}")
    ids = set(re.findall(r"race_id=(\d{12})", html))
    # 当該日(先頭8桁は場コードでなく年+場、date一致は下位で判別不可)→ 12桁をそのまま採用
    return sorted(ids)

def parse_result(race_id):
    html = get(f"https://race.netkeiba.com/race/result.html?race_id={race_id}")
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", class_="RaceTable01") or soup.find("table", class_="race_table_01")
    if not tbl:
        return None
    res = {"race_id": race_id, "finish_order": {}, "odds": {}, "popularity": {}, "agari": {},
           "payouts": {"trio": 0, "umaren": 0, "wide": 0, "tansho": 0, "fukusho": {}, "tierce": 0}}
    for row in tbl.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 11:
            continue
        fin = tds[0].get_text(strip=True)
        if not fin.isdigit():
            continue
        # 馬番 = 'Num Txt_C', 人気 = 'Odds Txt_C', オッズ = 'Odds Txt_R', 後3F = オッズ列の直後(Time)
        def cell(cls_all):
            return next((td for td in tds if set(cls_all).issubset(set(td.get("class", [])))), None)
        c_uma = cell(["Num", "Txt_C"]); c_pop = cell(["Odds", "Txt_C"]); c_odds = cell(["Odds", "Txt_R"])
        if not c_uma:
            continue
        ut = c_uma.get_text(strip=True)
        if not ut.isdigit():
            continue
        ub = int(ut)
        res["finish_order"][ub] = int(fin)
        if c_pop and c_pop.get_text(strip=True).isdigit():
            res["popularity"][ub] = int(c_pop.get_text(strip=True))
        if c_odds:
            ot = c_odds.get_text(strip=True)
            try: res["odds"][ub] = float(ot)
            except ValueError: pass
        # 後3F(上がり) = 通過順(PassageRate)列の直前セル(堅牢)
        c_pass = cell(["PassageRate"])
        if c_pass is not None:
            try:
                pi = tds.index(c_pass)
                at = tds[pi - 1].get_text(strip=True)
                av = float(at)
                if 28.0 <= av <= 50.0:   # 上がり3Fの妥当域
                    res["agari"][ub] = av
            except (IndexError, ValueError): pass
    if not res["finish_order"]:
        return None
    # 払戻
    for pt in (soup.find_all("table", class_="Payout_Detail_Table") or soup.find_all("table", class_="pay_table_01")):
        for row in pt.find_all("tr"):
            th = row.find("th")
            if not th: continue
            t = _norm(th.get_text(strip=True))
            tds = row.find_all("td")
            vals = [int(m.group(1).replace(",", "")) for td in tds
                    for m in re.finditer(r"([\d,]+)円", _norm(td.get_text(strip=True)))]
            if not vals: continue
            def result_umaban():
                rt = next((td for td in tds if "Result" in " ".join(td.get("class", []))), None)
                return [int(x) for x in re.findall(r"\d+", rt.get_text("\n", strip=True))] if rt else []
            if ("単勝" in t) and ("連" not in t):
                res["payouts"]["tansho"] = vals[0]
            elif ("複勝" in t) and ("連" not in t):
                for nb, pv in zip(result_umaban(), vals):
                    if 1 <= nb <= 18: res["payouts"]["fukusho"][nb] = pv
            elif ("三連複" in t) or ("3連複" in t):
                res["payouts"]["trio"] = vals[0]
            elif ("三連単" in t) or ("3連単" in t):
                res["payouts"]["tierce"] = vals[0]
            elif ("馬連" in t) and ("三" not in t) and ("単" not in t) and ("3" not in t):
                res["payouts"]["umaren"] = vals[0]
            elif "ワイド" in t and res["payouts"]["wide"] == 0:
                res["payouts"]["wide"] = vals[0]
    return res

def main():
    tot_r = tot_ok = 0
    for i, date in enumerate(DATES, 1):
        outp = os.path.join(OUT, f"{date}.json")
        if os.path.exists(outp) and os.path.getsize(outp) > 0:
            print(f"[{i}/{len(DATES)}] {date} 既存スキップ"); continue
        try:
            rids = race_ids_for(date)
        except Exception as e:
            print(f"[{i}/{len(DATES)}] {date} race_list 失敗: {e}"); continue
        results = []
        ok = 0
        for rid in rids:
            try:
                r = parse_result(rid)
            except Exception as e:
                r = None
            if r:
                results.append(r); ok += 1
        io.open(outp, "w", encoding="utf-8").write(json.dumps(results, ensure_ascii=False))
        tot_r += len(rids); tot_ok += ok
        print(f"[{i}/{len(DATES)}] {date} race_id={len(rids)} 結果取得={ok}", flush=True)
    print(f"完了: 総race_id={tot_r} 結果取得={tot_ok}")

if __name__ == "__main__":
    main()
