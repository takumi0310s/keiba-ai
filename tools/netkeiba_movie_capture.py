#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1.2 + W1.3: netkeiba 動画 frame capture 統一 wrapper (paddock / oikiri / race).

paddock_video_capture.py の汎用化版。 3 種類の動画を共通 capture pipeline で処理:

- paddock  : パドック (db: paddock_movie.html, race: paddock_movie.html)
- oikiri   : 調教 (race: oikiri_movie.html)
- race     : レース映像 / 返し馬 含む (race: movie.html)

【規約】netkeiba 第 14 条 私的利用範囲、 frame のみ抽出 (動画 file 保存しない)、 配布 NG。

Usage:
    python tools/netkeiba_movie_capture.py --kind paddock 2022106229 --race-id 202603010112
    python tools/netkeiba_movie_capture.py --kind oikiri 2022106229 --race-id 202603010112
    python tools/netkeiba_movie_capture.py --kind race --race-id 202603010112
    python tools/netkeiba_movie_capture.py --kind paddock 2022106229 --race-id 202603010112 --probe
    python tools/netkeiba_movie_capture.py --kind oikiri --fps 3 --duration 60 ...

Output:
    data/{kind}_frames/{race_id}_{horse_id}/frame_NNNN.jpg
    data/{kind}_frames/{race_id}_{horse_id}/manifest.json
"""
import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime

# Windows cp932 console で UTF-8 出力可能にする
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOKIE_PATH = os.path.join(BASE_DIR, 'data', 'cookies.json')

# kind → URL pattern + output dir
KIND_CONFIG = {
    'paddock': {
        'viewer_url': 'https://race.netkeiba.com/race/paddock_movie.html?race_id={race_id}&id={horse_id}',
        'index_url': 'https://db.netkeiba.com/horse/paddock_movie.html?id={horse_id}',
        'out_subdir': 'paddock_frames',
        'needs_horse_id': True,
    },
    'oikiri': {
        'viewer_url': 'https://race.netkeiba.com/race/oikiri_movie.html?race_id={race_id}&id={horse_id}',
        'index_url': None,
        'out_subdir': 'training_video_frames',
        'needs_horse_id': True,
    },
    'race': {
        # レース映像 (本馬場入場 + 返し馬 + レース)、 horse_id 任意 (馬別ハイライト)
        'viewer_url': 'https://race.netkeiba.com/race/movie.html?race_id={race_id}',
        'index_url': None,
        'out_subdir': 'race_video_frames',
        'needs_horse_id': False,
    },
}

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
        premiumGate: !!document.querySelector('.Premium_Regist_Box02'),
    };
    if (v) {
        result.video = {
            src: v.src || v.currentSrc,
            duration: v.duration,
            videoWidth: v.videoWidth,
            videoHeight: v.videoHeight,
            readyState: v.readyState,
            paused: v.paused,
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


def capture(kind, horse_id=None, race_id=None, fps=3, duration=30,
            headless=True, probe_only=False):
    if kind not in KIND_CONFIG:
        print(f'[ERROR] unknown kind: {kind}')
        return 1
    cfg = KIND_CONFIG[kind]

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[ERROR] playwright not installed: pip install playwright && playwright install chromium')
        return 1

    cookies = load_cookies()
    if cookies is None:
        return 1

    if not race_id:
        print('[ERROR] --race-id required for all kinds')
        return 1
    if cfg['needs_horse_id'] and not horse_id:
        print(f'[ERROR] horse_id required for kind={kind}')
        return 1

    horse_id = horse_id or '0'
    url = cfg['viewer_url'].format(race_id=race_id, horse_id=horse_id)
    out_dir = os.path.join(BASE_DIR, 'data', cfg['out_subdir'], f'{race_id}_{horse_id}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[INFO] kind={kind}, target: {url}')
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
            if any(k in u for k in ('.m3u8', '.mpd', '.ts', '.mp4', 'race-player', 'admint')):
                net_log.append({'method': req.method, 'url': u, 'rt': req.resource_type})
        page.on('request', on_request)

        try:
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
        except Exception as e:
            print(f'[WARN] goto error: {e}')

        page.wait_for_timeout(3000)

        probe = page.evaluate(PROBE_JS)
        print(f'[PROBE] videoCount={probe.get("videoCount")}, iframes={probe.get("iframeCount")}, '
              f'premium_gate={probe.get("premiumGate")}')
        if probe.get('premiumGate'):
            print('[AUTH] WARN: Premium_Regist_Box02 detected -> cookie expired. '
                  'Run: python tools/refresh_cookie.py')

        manifest = {
            'kind': kind,
            'race_id': str(race_id),
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
            print(f'[NET] candidates: {len(net_log)}')
            for n in net_log[:5]:
                print(f'      {n["method"]} {n["url"][:140]}')
            browser.close()
            return 0

        # Detect cross-origin iframe video player
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

        try:
            page.evaluate("() => { const v = document.querySelector('video'); if (v) { v.muted = true; v.play().catch(() => {}); } }")
        except Exception:
            pass

        if iframe_locator is not None:
            try:
                iframe_locator.click(timeout=3000)
                page.wait_for_timeout(800)
            except Exception as e:
                print(f'[WARN] iframe click failed: {e}')

        page.wait_for_timeout(1500)

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
            if not use_iframe_screenshot:
                try:
                    r = page.evaluate(CAPTURE_JS)
                except Exception as e:
                    errs += 1
                    if errs > 5:
                        break
                    page.wait_for_timeout(interval_ms)
                    continue
                if r.get('error'):
                    if r.get('error') == 'canvas_taint' and not canvas_taint_detected:
                        canvas_taint_detected = True
                        print('[INFO] canvas_taint -> fallback iframe screenshot')
                        use_iframe_screenshot = True
                        for sel in ['iframe[src*="admint"]', 'iframe[src*="tv-player"]']:
                            loc = page.locator(sel).first
                            if loc.count() > 0:
                                iframe_locator = loc
                                break
                        continue
                    errs += 1
                    if errs > 10:
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
                    if errs > 10:
                        break

            page.wait_for_timeout(interval_ms)

        manifest['summary'] = {
            'frames_saved': idx,
            'bytes_total': bytes_total,
            'errors': errs,
        }
        with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f'[OK] frames={idx}, size={bytes_total/1024:.1f} KB, errs={errs}')
        print(f'[OK] saved: {out_dir}/')
        browser.close()
        return 0


def main():
    ap = argparse.ArgumentParser(description='netkeiba 動画 frame capture 統一 wrapper')
    ap.add_argument('horse_id', nargs='?', default=None,
                    help='netkeiba horse_id (paddock/oikiri 必須、 race 任意)')
    ap.add_argument('--kind', required=True, choices=list(KIND_CONFIG.keys()),
                    help='動画 種類: paddock / oikiri / race')
    ap.add_argument('--race-id', dest='race_id', required=True,
                    help='race_id (例 202603010112)')
    ap.add_argument('--fps', type=int, default=3, help='frames per second (default: 3)')
    ap.add_argument('--duration', type=int, default=30, help='capture duration sec')
    ap.add_argument('--headless', default='true')
    ap.add_argument('--probe', action='store_true', help='DOM 調査のみ')
    args = ap.parse_args()

    headless = args.headless.lower() not in ('false', '0', 'no')
    rc = capture(args.kind, horse_id=args.horse_id, race_id=args.race_id,
                 fps=args.fps, duration=args.duration,
                 headless=headless, probe_only=args.probe)
    sys.exit(rc)


if __name__ == '__main__':
    main()
