"""paddock LIVE capture skeleton (user 設定 必須).

★ user setup 後 動作 ★:
1. レーシングビュワー login + cookie 保存 (Playwright persistent context)
2. 当日 race ごと URL 取得
3. ffmpeg で paddock 動画 capture
4. tools/paddock_yolo_inference.py で frame 抽出 + features 計算

設定 必要 (user 1 回):
- pip install playwright
- playwright install chromium
- 初回 login (cookie 保存):
  python tools/paddock_live_capture.py --setup-login

★ 規約 注意 ★:
- DRM 確認 (PrintScreen + OBS test) 必須
- DRM なら frame 抽出 違法
- 配布 / 公開 NG
- 私的 AI 学習 用 のみ (Article 30)

★ V15 投資保護 完全 ★:
- V15 .pkl.gz / predict_core / app.py 完全不変
- capture / analyze は別 pipeline
- 失敗時 V15 fallback

usage:
    python tools/paddock_live_capture.py --setup-login         # 初回 login
    python tools/paddock_live_capture.py --date 20260516       # 当日 capture
    python tools/paddock_live_capture.py --race-id 202506050811 # 1 race のみ
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
COOKIE_DIR = BASE / 'data' / 'racing_viewer_cookies'
PADDOCK_DIR = BASE / 'data' / 'paddock_live'

RACING_VIEWER_BASE = 'https://members.jra.go.jp/'  # 参考 URL (実 URL は確認必要)


def setup_login():
    """初回 user login + cookie 保存 (1 回 のみ)."""
    print('=== Setup レーシングビュワー login (1 回 のみ) ===')
    print('1. ブラウザ window が開きます')
    print('2. レーシングビュワー へ login')
    print('3. paddock 動画 1 つ 再生 (cookie 完全保存)')
    print('4. このスクリプト の window を 閉じる')
    print('')

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[ERROR] playwright not installed')
        print('install: pip install playwright && playwright install chromium')
        sys.exit(1)

    COOKIE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # persistent context = cookie 自動保存
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(COOKIE_DIR),
            headless=False,
        )
        page = browser.new_page()
        page.goto(RACING_VIEWER_BASE)
        print('login 後 paddock 動画 1 つ 再生 → window 閉じる')
        print('cookie 自動保存')
        # Wait for user to close browser
        try:
            browser.wait_for_event('close', timeout=600000)  # 10 min
        except Exception:
            pass
        finally:
            try:
                browser.close()
            except Exception:
                pass

    print(f'cookie 保存: {COOKIE_DIR}')


def get_today_races(date_str: str):
    """当日 race list 取得 (netkeiba 既存 scraper 利用)."""
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(BASE / 'tools'))
    from race_auto_notify import get_todays_races
    return get_todays_races(date_str)


def capture_race(race_id: str, browser_context):
    """1 race の paddock 動画 capture."""
    page = browser_context.new_page()
    # URL pattern 要 reverse-engineer (user setup 後 確定)
    paddock_url = f'{RACING_VIEWER_BASE}paddock/{race_id}'
    print(f'  capturing {race_id} from {paddock_url} ...')

    try:
        page.goto(paddock_url, timeout=30000)
        page.wait_for_selector('video', timeout=30000)
        video_element = page.query_selector('video')
        if not video_element:
            print(f'  [WARN] {race_id}: video element 不在')
            return False

        # 動画 src 取得 (DRM ない なら 直接 URL)
        video_src = page.eval_on_selector('video', 'el => el.src or el.currentSrc')
        if not video_src or 'blob:' in str(video_src):
            print(f'  [WARN] {race_id}: blob URL (MSE/EME)、 直接 capture 不可')
            print(f'  → screen capture (OBS / ffmpeg gdigrab) 必要')
            return False

        # ffmpeg capture
        out_path = PADDOCK_DIR / f'{race_id}.mp4'
        PADDOCK_DIR.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            'ffmpeg', '-y', '-i', video_src,
            '-t', '600',  # 10 min max
            '-c', 'copy',
            str(out_path)
        ], capture_output=True, text=True, timeout=700)

        if result.returncode == 0:
            print(f'  ✓ saved: {out_path}')
            return True
        else:
            print(f'  [ERROR] ffmpeg: {result.stderr[:200]}')
            return False
    except Exception as e:
        print(f'  [ERROR] {race_id}: {e}')
        return False
    finally:
        page.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup-login', action='store_true', help='初回 login + cookie 保存')
    ap.add_argument('--date', default=None, help='当日 capture (YYYYMMDD)')
    ap.add_argument('--race-id', default=None, help='1 race のみ')
    args = ap.parse_args()

    if args.setup_login:
        setup_login()
        return

    if not COOKIE_DIR.exists():
        print(f'[ERROR] cookie 不在: {COOKIE_DIR}')
        print('実行: python tools/paddock_live_capture.py --setup-login')
        sys.exit(1)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[ERROR] playwright not installed')
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(COOKIE_DIR),
            headless=True,
        )

        if args.race_id:
            capture_race(args.race_id, browser)
        elif args.date:
            races = get_today_races(args.date)
            print(f'capturing {len(races)} races for {args.date}')
            for r in races:
                capture_race(r['race_id'], browser)
        else:
            print('--date or --race-id 必要')

        browser.close()


if __name__ == '__main__':
    main()
