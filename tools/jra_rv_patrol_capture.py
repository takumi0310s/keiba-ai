#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JRA レーシングビュアー パトロール ビデオ access (ゴール後 40 分公開、 不利検知 source).

user 加入: JRA レーシングビュアー (¥550/月)。
パトロール ビデオ = 競走中の馬群俯瞰 + 不利 (interference) 検知 用。
V22 RL の「不利」 features の source として critical。

【公開 timing】
- レース後 **約 40 分** (ゴール 3 分後 の通常 race 映像より 遅い)
- archive で 過去分も 視聴可

【V15 投資保護】 frame 抽出 read-only、 V15 model 不変

Usage:
    python tools/jra_rv_patrol_capture.py --probe --race-id 202603010112
    python tools/jra_rv_patrol_capture.py --race-id 202603010112 --duration 60 --fps 3

Note:
    .env に既存 JRA_RV_LOGIN_ID / JRA_RV_PASSWORD 利用 (jra_racing_viewer_capture.py と共通)
"""
import argparse
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')
PATROL_URL_PATTERN = 'https://prc.jp/race/{race_id}/patrol'  # 推定


def cmd_probe(args):
    """パトロール page DOM 構造調査."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[ERROR] playwright not installed')
        return 1

    rid = args.race_id
    url = PATROL_URL_PATTERN.format(race_id=rid)
    print(f'[INFO] target: {url}')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = ctx.new_page()

        net_log = []
        page.on('request', lambda req: net_log.append({
            'method': req.method, 'url': req.url
        }) if any(k in req.url for k in ('.m3u8', '.mp4', '.mpd', 'patrol',
                                            'video', 'stream')) else None)

        try:
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
        except Exception as e:
            print(f'[WARN] goto error: {e}')

        page.wait_for_timeout(3000)
        print(f'[INFO] page title: {page.title()}')
        try:
            results = page.evaluate("""() => ({
                video: document.querySelectorAll('video').length,
                iframe: document.querySelectorAll('iframe').length,
                login_form: document.querySelectorAll('form[action*=login]').length,
            })""")
            print(f'[DOM] video: {results["video"]}, iframe: {results["iframe"]}, '
                  f'login_form: {results["login_form"]}')
        except Exception as e:
            print(f'[WARN] {e}')

        print(f'\n[NET] streaming candidates ({len(net_log)}):')
        for n in net_log[:10]:
            print(f'  {n["method"]} {n["url"][:160]}')

        print('\n=== 次 step ===')
        print('  1. JRA_RV_LOGIN_ID / PASSWORD で login')
        print('  2. URL pattern 確認 (推定: prc.jp/race/{race_id}/patrol)')
        print('  3. 実 url で paddock_video_capture.py の iframe screenshot pattern を適用')

        browser.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description='JRA RV パトロール capture (skeleton)')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--race-id', dest='race_id', required=False)
    ap.add_argument('--duration', type=int, default=60)
    ap.add_argument('--fps', type=int, default=3)
    args = ap.parse_args()

    if args.probe and args.race_id:
        return cmd_probe(args)
    print('[INFO] skeleton、 require:')
    print('  1. JRA_RV_LOGIN_ID / PASSWORD in .env')
    print('  2. python tools/jra_rv_patrol_capture.py --probe --race-id XXX')
    print('  3. URL pattern 確定 後 capture')
    return 0


if __name__ == '__main__':
    sys.exit(main())
