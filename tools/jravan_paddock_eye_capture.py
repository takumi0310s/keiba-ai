#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JRA-VAN パドックアイ Web 自動 access wrapper (発走 20 分前 提供).

user 加入: JRA-VAN DataLab (¥2,090) + JRA レーシングビュアー (¥550) 既加入。
パドックアイ = JRA-VAN ID login 経由で 馬別分割動画 + AI 姿勢推定 / 歩様解析。

【公開 timing】
- 発走 **20 分前** に提供開始 (paddock_movie の 締切 15 分前 より さらに早い)
- LIVE 予測 (R-5 分前) に十分 間に合う

【V15 投資保護】 動画 frame 抽出のみ、 V15 model 不変

【規約】 私的利用範囲、 ストリーミング cache + frame 抽出、 配布 NG

Usage:
    # login + DOM 構造調査 (probe)
    python tools/jravan_paddock_eye_capture.py --probe --race-id 202603010112

    # 実 capture (login flow 確立 後)
    python tools/jravan_paddock_eye_capture.py --race-id 202603010112 --horse-num 7

Note:
    .env に追加要:
      JRAVAN_ID=your_jra_van_id
      JRAVAN_PASSWORD=your_jra_van_password
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

JRAVAN_PADDOCK_EYE_URL = 'https://jra-van.jp/paddock-ai/'


def load_credentials():
    if not os.path.exists(ENV_PATH):
        return None, None
    with open(ENV_PATH, 'r', encoding='utf-8') as f:
        env = {}
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env.get('JRAVAN_ID'), env.get('JRAVAN_PASSWORD')


def cmd_probe(args):
    """JRA-VAN パドックアイ DOM 構造調査."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[ERROR] playwright not installed')
        return 1

    print(f'[INFO] JRA-VAN パドックアイ probe')
    print(f'[INFO] target: {JRAVAN_PADDOCK_EYE_URL}')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = ctx.new_page()

        net_log = []
        page.on('request', lambda req: net_log.append({
            'method': req.method, 'url': req.url
        }) if any(k in req.url for k in ('.m3u8', '.mp4', '.mpd', 'movie',
                                            'paddock-ai', 'video', 'stream')) else None)

        try:
            page.goto(JRAVAN_PADDOCK_EYE_URL, wait_until='domcontentloaded', timeout=30000)
        except Exception as e:
            print(f'[WARN] goto error: {e}')

        page.wait_for_timeout(3000)

        print(f'\n[INFO] page title: {page.title()}')
        # video / iframe / login form 探索
        try:
            results = page.evaluate("""() => ({
                video: document.querySelectorAll('video').length,
                iframe: document.querySelectorAll('iframe').length,
                login_form: document.querySelectorAll('form[action*=login]').length,
                login_link: document.querySelectorAll('a[href*=login]').length,
            })""")
            print(f'[DOM] video: {results["video"]}, iframe: {results["iframe"]}, '
                  f'login_form: {results["login_form"]}, login_link: {results["login_link"]}')
        except Exception as e:
            print(f'[WARN] DOM probe error: {e}')

        print(f'\n[NET] candidates ({len(net_log)}):')
        for n in net_log[:10]:
            print(f'  {n["method"]} {n["url"][:160]}')

        print('\n=== 次 step ===')
        print('  1. .env に JRAVAN_ID / JRAVAN_PASSWORD 追加')
        print('  2. login DOM 要素 (id/name) を 上記 probe で 特定')
        print('  3. login + cookie 保存 → 動画 page で frame 抽出 / m3u8 download')
        print('  4. race_id 別 url pattern を 試行 (例: /paddock-ai/?race_id=XXX)')

        browser.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description='JRA-VAN パドックアイ capture (skeleton)')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--race-id', dest='race_id', default=None)
    ap.add_argument('--horse-num', dest='horse_num', type=int, default=None)
    args = ap.parse_args()

    if args.probe:
        return cmd_probe(args)
    print('[INFO] full capture is skeleton、 require manual setup:')
    print('  1. python tools/jravan_paddock_eye_capture.py --probe')
    print('  2. DOM 構造 + login flow 確認')
    print('  3. wrapper 拡張 後 capture')
    return 0


if __name__ == '__main__':
    sys.exit(main())
