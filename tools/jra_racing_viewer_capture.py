#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JRA レーシングビュアー Web 自動 access wrapper (公式動画 frame capture).

user 加入済 JRAレーシングビュアー (¥550/月) の Web サイト (https://prc.jp/) を
Playwright で 自動 access し、 動画 frame を 抽出。

【公開 timing (official)】
- レース映像: ゴール **3 分後** 公開
- パドック映像: 締め切り **15 分前** 公開 ← LIVE 予測 間に合う!
- 調教映像: 木曜 14:00+ archive
- マルチカメラビュー: あり (詳細 form 解析 用)

【V15 投資保護】 動画 frame 抽出のみ、 V15 model 不変

【規約】 私的利用範囲、 ストリーミング再生 + 一時 cache、 配布 NG

Usage:
    # 動作確認 (login + 任意 race ページ)
    python tools/jra_racing_viewer_capture.py --probe --race-id 202603010112

    # 調教動画 frame 抽出
    python tools/jra_racing_viewer_capture.py --kind oikiri --race-id 202603010112 --horse-id 2022106229

    # レース動画 frame 抽出
    python tools/jra_racing_viewer_capture.py --kind race --race-id 202603010112

Note:
    JRA Racing Viewer login が必要。 加入済 ID/PW を .env に保存推奨:
        JRA_RV_LOGIN_ID=...
        JRA_RV_PASSWORD=...
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
COOKIE_PATH = os.path.join(BASE_DIR, 'data', 'jra_rv_cookies.json')
ENV_PATH = os.path.join(BASE_DIR, '.env')

JRA_RV_TOP = 'https://prc.jp/'
JRA_RV_LOGIN = 'https://prc.jp/login.html'  # 推定、 実 URL は確認要


def load_credentials():
    if not os.path.exists(ENV_PATH):
        return None, None
    with open(ENV_PATH, 'r', encoding='utf-8') as f:
        env = {}
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env.get('JRA_RV_LOGIN_ID'), env.get('JRA_RV_PASSWORD')


def cmd_probe(args):
    """Login 動作確認 + 動画 page DOM 構造 調査."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[ERROR] playwright not installed')
        return 1

    print(f'[INFO] JRA レーシングビュアー Web 動作確認')
    print(f'[INFO] target: {JRA_RV_TOP}')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # automated probe
        ctx = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0',
        )
        page = ctx.new_page()

        net_log = []
        page.on('request', lambda req: net_log.append({
            'method': req.method, 'url': req.url
        }) if any(k in req.url for k in ('.m3u8', '.mp4', '.mpd', 'stream')) else None)

        try:
            page.goto(JRA_RV_TOP, wait_until='domcontentloaded', timeout=30000)
        except Exception as e:
            print(f'[WARN] goto error: {e}')

        page.wait_for_timeout(5000)

        print(f'[INFO] page title: {page.title()}')
        # video element 探索
        try:
            video_count = page.evaluate("document.querySelectorAll('video').length")
            print(f'[INFO] video elements: {video_count}')
        except Exception:
            pass

        print(f'\n[NET] streaming candidates ({len(net_log)}):')
        for n in net_log[:10]:
            print(f'  {n["method"]} {n["url"][:160]}')

        print('\n=== 次 step ===')
        print('  1. .env に JRA_RV_LOGIN_ID / JRA_RV_PASSWORD 保存')
        print('  2. login form 要素 (id/name) を 上記 DOM probe で 確定')
        print('  3. login + cookie 保存 → 動画 page で frame 抽出')
        print('  4. m3u8/mp4 URL を DevTools で 拾って Playwright で auto navigate')

        browser.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description='JRA Racing Viewer Web capture (skeleton)')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--kind', choices=['paddock', 'oikiri', 'race', 'main'], default='oikiri')
    ap.add_argument('--race-id', dest='race_id', required=False)
    ap.add_argument('--horse-id', dest='horse_id', default='0')
    ap.add_argument('--fps', type=int, default=2)
    ap.add_argument('--duration', type=int, default=30)
    args = ap.parse_args()

    if args.probe:
        return cmd_probe(args)
    print('[INFO] full capture is skeleton、 require manual setup:')
    print('  1. python tools/jra_racing_viewer_capture.py --probe')
    print('  2. 確認 DOM 構造 + m3u8 URL pattern')
    print('  3. wrapper の login/navigate logic を 実装')
    print('  4. paddock_video_capture.py の iframe screenshot pattern 再利用可')
    return 0


if __name__ == '__main__':
    sys.exit(main())
