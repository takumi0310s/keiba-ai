# -*- coding: utf-8 -*-
"""土曜朝の当日TYB/KAB 配信時刻probe (2026-08-15 一回きり・朝パス時刻確定用)。
05:00-08:00 に10分毎で member/data/{Tyb,Kab}/<TYPE><today>.lzh の存在をHEAD確認し、
初出時刻を logs/publish_probe_<date>.log に記録。"""
import os, sys, time, requests
from datetime import datetime
from requests.auth import HTTPBasicAuth
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tools"))
from download_jrdb import load_credentials, JRDB_BASE
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    i, p = load_credentials(); auth = HTTPBasicAuth(i, p)
    today = datetime.now().strftime("%y%m%d")
    log = open(os.path.join(BASE, "logs", f"publish_probe_20{today}.log"), "a", encoding="utf-8")
    targets = {"TYB": f"{JRDB_BASE}/data/Tyb/TYB{today}.lzh",
               "KAB": f"{JRDB_BASE}/data/Kab/KAB{today}.lzh"}
    found = {}
    deadline = datetime.now().replace(hour=8, minute=5)
    while datetime.now() < deadline and len(found) < len(targets):
        for k, url in targets.items():
            if k in found:
                continue
            try:
                r = requests.head(url, auth=auth, timeout=10)
                ts = datetime.now().strftime("%H:%M:%S")
                if r.status_code == 200:
                    found[k] = ts
                    log.write(f"{ts} {k} AVAILABLE (200)\n"); log.flush()
                else:
                    log.write(f"{ts} {k} {r.status_code}\n"); log.flush()
            except Exception as e:
                log.write(f"{datetime.now().strftime('%H:%M:%S')} {k} ERR {e}\n"); log.flush()
        if len(found) < len(targets):
            time.sleep(600)
    log.write(f"RESULT: {found}\n")
    log.close()
    print(f"probe done: {found}")

if __name__ == "__main__":
    main()
