#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plan A: netkeiba パドック動画 frame capture (個人視聴 + 私的複製範囲).

netkeiba 規約 第 14 条 (私的利用範囲外 NG) を遵守:
- 動画 file 自体は保存しない (原本 download なし)
- ストリーミング再生 + JS canvas drawImage で frame のみ抽出
- 抽出 frame は data/paddock_frames/{horse_id}/ に保存
- 用途: AI features 抽出 (gait / posture / weight perception) 私的学習のみ
- 配布 NG (.gitignore で commit 防止)

Usage:
    python tools/paddock_video_capture.py 2022106229
    python tools/paddock_video_capture.py 2022106229 --fps 5 --duration 60
    python tools/paddock_video_capture.py 2022106229 --headless false   # debug
    python tools/paddock_video_capture.py --probe 2022106229            # DOM 構造のみ調査

Output:
    data/paddock_frames/{horse_id}/frame_NNNN.jpg
    data/paddock_frames/{horse_id}/manifest.json
"""
import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOKIE_PATH = os.path.join(BASE_DIR, 'data', 'cookies.json')
OUT_BASE = os.path.join(BASE_DIR, 'data', 'paddock_frames')

PADDOCK_INDEX_URL = 'https://db.netkeiba.com/horse/paddock_movie.html?id={horse_id}'
PADDOCK_VIEWER_URL = 'https://race.netkeiba.com/race/paddock_movie.html?race_id={race_id}&id={horse_id}'

CAPTURE_JS = """
() => {
    const v = document.querySelector('video');
    if (!v) return {error: 'no_video_element'};
    if (!v.videoWidth) return {error: 'video_not_loaded', readyState: v.readyState};
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext('2d');
    try {
        ctx.drawImage(v, 0, 0);
        return {
            ok: true,
            data: c.toDataURL('image/jpeg', 0.85),
            currentTime: v.currentTime,
            duration: v.duration,
            paused: v.paused,
            w: v.videoWidth,
            h: v.videoHeight,
        };
    } catch (e) {
        return {error: 'canvas_taint', message: String(e)};
    }
}
"""

PROBE_JS = """
() => {
    const v = document.querySelector('video');
    const sources = Array.from(document.querySelectorAll('source')).map(s => ({src: s.src, type: s.type}));
    const iframes = Array.from(document.querySelectorAll('iframe')).map(f => ({src: f.src, w: f.width, h: f.height}));
    const result = {
        url: location.href,
        videoCount: document.querySelectorAll('video').length,
        iframeCount: iframes.length,
        iframes: iframes,
        sources: sources,
        premiumGate: !!document.querySelector('.Premium_Regist_Box02, .Premium_Regist_Box02.PaddockMovie'),
        loggedIn: !document.querySelector('.disp_none.header_stage_area.logout_show'),
    };
    if (v) {
        result.video = {
            src: v.src || v.currentSrc,
            duration: v.duration,
            videoWidth: v.videoWidth,
            videoHeight: v.videoHeight,
            readyState: v.readyState,
            paused: v.paused,
            poster: v.poster,
            outerHTML: v.outerHTML.slice(0, 600),
        };
    }
    return result;
}
"""


def load_cookies():
    if not os.path.exists(COOKIE_PATH):
        print(f'[ERROR] cookies.json not found: {COOKIE_PATH}')
        return None
    with open(COOKIE_PATH, 'r', encoding='utf-8') as f:
        cookies = json.load(f)
    out = []
    for c in cookies:
        if 'sameSite' not in c:
            c = {**c, 'sameSite': 'Lax'}
        if c.get('sameSite') not in ('Strict', 'Lax', 'None'):
            c['sameSite'] = 'Lax'
        out.append(c)
    return out


def save_frame(out_dir, idx, data_url):
    if not data_url.startswith('data:image/jpeg;base64,'):
        return None
    b64 = data_url.split(',', 1)[1]
    raw = base64.b64decode(b64)
    fp = os.path.join(out_dir, f'frame_{idx:04d}.jpg')
    with open(fp, 'wb') as f:
        f.write(raw)
    return len(raw)


def capture(horse_id, race_id=None, fps=3, duration=30, headless=True, probe_only=False):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[ERROR] playwright not installed: pip install playwright && playwright install chromium')
        return 1

    cookies = load_cookies()
    if cookies is None:
        return 1

    if race_id:
        out_dir = os.path.join(OUT_BASE, f'{race_id}_{horse_id}')
        url = PADDOCK_VIEWER_URL.format(race_id=race_id, horse_id=horse_id)
    else:
        out_dir = os.path.join(OUT_BASE, str(horse_id))
        url = PADDOCK_INDEX_URL.format(horse_id=horse_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[INFO] target: {url}')
    print(f'[INFO] out_dir: {out_dir}')
    print(f'[INFO] mode: {"PROBE" if probe_only else f"CAPTURE fps={fps} dur={duration}s"}')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        )
        ctx.add_cookies(cookies)
        page = ctx.new_page()

        net_log = []
        def on_request(req):
            u = req.url
            if any(k in u for k in ('.m3u8', '.mpd', '.ts', '.mp4', 'video', 'movie', 'stream')):
                net_log.append({'method': req.method, 'url': u, 'rt': req.resource_type})
        page.on('request', on_request)

        try:
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
        except Exception as e:
            print(f'[WARN] goto error: {e}')

        page.wait_for_timeout(3000)

        probe = page.evaluate(PROBE_JS)
        print(f'[PROBE] videoCount={probe.get("videoCount")}, iframes={probe.get("iframeCount")}, '
              f'premium_gate={probe.get("premiumGate")}, sources={len(probe.get("sources", []))}')
        if probe.get('premiumGate'):
            print('[AUTH] WARN: Premium_Regist_Box02 detected -> cookie expired / not logged in. '
                  'Run: python tools/refresh_cookie.py')
        if probe.get('iframeCount'):
            for f in probe.get('iframes', [])[:5]:
                print(f'[IFRAME] {f.get("src", "")[:140]}')
        if probe.get('video'):
            v = probe['video']
            print(f'[PROBE] src={v.get("src")[:120] if v.get("src") else "<empty>"}')
            print(f'[PROBE] duration={v.get("duration")}, {v.get("videoWidth")}x{v.get("videoHeight")}, readyState={v.get("readyState")}')

        manifest = {
            'horse_id': str(horse_id),
            'url': url,
            'captured_at': datetime.now().isoformat(),
            'probe': probe,
            'network': net_log[:30],
            'frames': [],
        }

        if probe_only:
            with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            print(f'[OK] probe saved: {out_dir}/manifest.json')
            print(f'[NET] m3u8/mp4 candidates: {len(net_log)}')
            for n in net_log[:5]:
                print(f'      {n["method"]} {n["url"][:140]}')
            browser.close()
            return 0

        try:
            page.evaluate("() => { const v = document.querySelector('video'); if (v) { v.muted = true; v.play().catch(() => {}); } }")
        except Exception:
            pass

        # Cross-origin iframe (admint.biz video player) detection
        iframe_locator = None
        for sel in ['iframe[src*="admint"]', 'iframe[src*="tv-player"]', 'iframe[src*="race-player"]']:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    iframe_locator = loc
                    print(f'[INFO] iframe video detected: {sel}')
                    break
            except Exception:
                pass

        # Click iframe to start video (autoplay often blocked)
        if iframe_locator is not None:
            try:
                iframe_locator.click(timeout=3000)
                page.wait_for_timeout(800)
            except Exception as e:
                print(f'[WARN] iframe click failed: {e}')

        page.wait_for_timeout(1500)

        # Capture mode: canvas (same-origin) or iframe screenshot (cross-origin)
        use_iframe_screenshot = iframe_locator is not None
        canvas_taint_detected = False

        interval_ms = max(1, int(1000 / fps))
        end = time.time() + duration
        idx = 0
        bytes_total = 0
        errs = 0
        last_t = -1.0
        print(f'[INFO] capture mode: {"iframe_screenshot" if use_iframe_screenshot else "canvas_drawImage"}')

        while time.time() < end:
            ct = None

            # Path A: canvas drawImage (only if video element in main DOM, not iframe)
            if not use_iframe_screenshot:
                try:
                    r = page.evaluate(CAPTURE_JS)
                except Exception as e:
                    errs += 1
                    print(f'[WARN] eval err #{errs}: {e}')
                    if errs > 5:
                        break
                    page.wait_for_timeout(interval_ms)
                    continue

                if r.get('error'):
                    if r.get('error') == 'canvas_taint' and not canvas_taint_detected:
                        canvas_taint_detected = True
                        print(f'[INFO] canvas_taint detected -> falling back to iframe screenshot')
                        use_iframe_screenshot = True
                        # find iframe again on the fly
                        for sel in ['iframe[src*="admint"]', 'iframe[src*="tv-player"]']:
                            loc = page.locator(sel).first
                            if loc.count() > 0:
                                iframe_locator = loc
                                break
                        continue

                    errs += 1
                    if errs <= 3:
                        print(f'[WARN] {r}')
                    if errs > 10:
                        print('[STOP] too many errors')
                        break
                    page.wait_for_timeout(interval_ms)
                    continue

                ct = r.get('currentTime', 0.0)
                if abs(ct - last_t) < 0.05:
                    page.wait_for_timeout(interval_ms)
                    continue
                last_t = ct
                sz = save_frame(out_dir, idx, r['data'])
                if sz:
                    bytes_total += sz
                    manifest['frames'].append({'idx': idx, 't': ct, 'bytes': sz})
                    idx += 1

            # Path B: iframe element screenshot (cross-origin safe)
            else:
                try:
                    fp = os.path.join(out_dir, f'frame_{idx:04d}.jpg')
                    iframe_locator.screenshot(path=fp, type='jpeg', quality=85, timeout=5000)
                    sz = os.path.getsize(fp)
                    bytes_total += sz
                    manifest['frames'].append({'idx': idx, 't': time.time(), 'bytes': sz})
                    idx += 1
                except Exception as e:
                    errs += 1
                    if errs <= 3:
                        print(f'[WARN] screenshot err #{errs}: {e}')
                    if errs > 10:
                        print('[STOP] too many screenshot errors')
                        break

            page.wait_for_timeout(interval_ms)

        manifest['summary'] = {
            'frames_saved': idx,
            'bytes_total': bytes_total,
            'errors': errs,
        }
        with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f'[OK] frames={idx}, size={bytes_total / 1024:.1f} KB, errs={errs}')
        print(f'[OK] saved: {out_dir}/')

        browser.close()
        return 0


def main():
    ap = argparse.ArgumentParser(description='netkeiba paddock video frame capture (Plan A)')
    ap.add_argument('horse_id', help='netkeiba horse_id (e.g. 2022106229)')
    ap.add_argument('--race-id', dest='race_id', default=None,
                    help='race_id (例 202603010112) 指定で viewer URL 直接、 未指定で index URL')
    ap.add_argument('--fps', type=int, default=3, help='frames per second (default: 3)')
    ap.add_argument('--duration', type=int, default=30, help='capture duration sec (default: 30)')
    ap.add_argument('--headless', default='true', help='true / false')
    ap.add_argument('--probe', action='store_true', help='DOM 構造調査のみ (frame 抽出 skip)')
    args = ap.parse_args()

    headless = args.headless.lower() not in ('false', '0', 'no')
    rc = capture(args.horse_id, race_id=args.race_id, fps=args.fps, duration=args.duration,
                 headless=headless, probe_only=args.probe)
    sys.exit(rc)


if __name__ == '__main__':
    main()
