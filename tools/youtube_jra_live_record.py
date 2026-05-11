#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1.1: YouTube JRA 公式チャンネル 中央競馬全レース中継 LIVE 録画 (yt-dlp wrapper).

2026/3/14 から JRA 公式 YouTube channel で開催日 9:00-17:00 LIVE 配信開始 (無料、 規約 clean)。
本 script は yt-dlp で LIVE stream を自動 record する schtask 用 wrapper。

【規約】
- JRA 公式 YouTube 配信は無料公開、 個人視聴 OK
- 私的複製 (著作 30 条) 範囲内、 配布 NG
- .gitignore で data/youtube_jra_live/ commit 防止

【使い方】
    python tools/youtube_jra_live_record.py                      # 次の LIVE を自動 wait + record
    python tools/youtube_jra_live_record.py --probe              # LIVE id / 開始時刻のみ表示
    python tools/youtube_jra_live_record.py --quality 720        # 720p 制限
    python tools/youtube_jra_live_record.py --max-duration 28800 # 8 時間 (default)
    python tools/youtube_jra_live_record.py --video-id ID        # 特定 video 直接

【schtask 登録 (推奨)】
    schtasks /create /tn "Keiba-YouTubeLiveRecord-Sat" /sc WEEKLY /d SAT /st 08:55 ^
      /tr "python C:\\Users\\takum\\keiba-ai\\tools\\youtube_jra_live_record.py --quality 720"

【出力】
    data/youtube_jra_live/{date}_{video_id}.{ext}
    data/youtube_jra_live/{date}_{video_id}.info.json (metadata)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

# Windows cp932 console で UTF-8 出力可能にする
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'youtube_jra_live')

# JRA 公式 YouTube
JRA_CHANNEL_ID = 'UCj6AKkCWS6FJqf0o5wP45eQ'
JRA_CHANNEL_LIVE = f'https://www.youtube.com/channel/{JRA_CHANNEL_ID}/live'
JRA_CHANNEL_STREAMS = f'https://www.youtube.com/channel/{JRA_CHANNEL_ID}/streams'


def yt_dlp_cmd(*args):
    return [sys.executable, '-m', 'yt_dlp', *args]


def probe_live():
    """次の LIVE 配信 id / 開始時刻を取得 (録画はしない)."""
    cmd = yt_dlp_cmd(
        '--flat-playlist',
        '--playlist-end', '5',
        '--print', '%(id)s|%(title)s|%(release_timestamp)s|%(live_status)s',
        JRA_CHANNEL_STREAMS,
    )
    print(f'[INFO] probing: {JRA_CHANNEL_STREAMS}')
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           encoding='utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        print('[ERROR] yt-dlp probe timeout')
        return None

    results = []
    for line in r.stdout.splitlines():
        if not line.strip() or '|' not in line:
            continue
        parts = line.split('|')
        if len(parts) < 4:
            continue
        vid, title, rel, status = parts[0], parts[1], parts[2], parts[3]
        results.append({'id': vid, 'title': title, 'release': rel, 'status': status})

    return results


def record_live(video_id=None, quality='720', max_duration=28800,
                wait_for_live=True, dry_run=False):
    """LIVE stream を record. video_id=None なら channel/live を target.

    max_duration: 録画最大秒数 (default 8 時間 = 28800)
    quality: 360 / 480 / 720 / 1080 (default 720、 容量とのバランス)
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')

    if video_id:
        target = f'https://www.youtube.com/watch?v={video_id}'
    else:
        target = JRA_CHANNEL_LIVE

    # 出力 path
    output_template = os.path.join(OUT_DIR, f'{today}_%(id)s.%(ext)s')

    # yt-dlp args
    args = [
        '-f', f'bestvideo[height<={quality}][protocol*=m3u8]+bestaudio/best[height<={quality}]',
        '--merge-output-format', 'mp4',
        '--live-from-start',          # LIVE 開始からの全 chunk
        '--hls-use-mpegts',           # 途切れ耐性
        '--no-part',
        '--write-info-json',
        '--no-write-comments',
        '--no-write-thumbnail',
        '--retries', '10',
        '--fragment-retries', '20',
        '-o', output_template,
    ]

    if wait_for_live:
        args += ['--wait-for-video', '600']  # 最大 10 分待つ
    if max_duration:
        args += ['--download-sections', f'*0-{max_duration}']

    args.append(target)
    cmd = yt_dlp_cmd(*args)
    print('[INFO] cmd:', ' '.join(['yt-dlp'] + args))

    if dry_run:
        print('[DRY-RUN] yt-dlp not actually executed')
        return 0

    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print('[INTERRUPT] yt-dlp aborted')
        return 130

    print(f'[INFO] yt-dlp exited rc={rc}')
    return rc


def main():
    ap = argparse.ArgumentParser(description='YouTube JRA 公式 LIVE 録画')
    ap.add_argument('--probe', action='store_true', help='LIVE id / 時刻のみ表示')
    ap.add_argument('--quality', default='720', help='画質 (360/480/720/1080)')
    ap.add_argument('--max-duration', dest='max_duration', type=int, default=28800,
                    help='最大録画秒数 (default 28800 = 8h)')
    ap.add_argument('--video-id', dest='video_id', default=None,
                    help='特定 video_id を直接録画 (test 用)')
    ap.add_argument('--no-wait', dest='no_wait', action='store_true',
                    help='LIVE 開始待ちしない (現在配信中のみ)')
    ap.add_argument('--dry-run', dest='dry_run', action='store_true',
                    help='実行せず cmd 表示のみ')
    args = ap.parse_args()

    if args.probe:
        results = probe_live()
        if not results:
            print('[ERROR] no streams found')
            return 1
        print(f'[OK] {len(results)} streams (newest first):')
        for r in results[:10]:
            rel_ts = r.get('release', 'N/A')
            if rel_ts and rel_ts != 'N/A' and rel_ts != 'NA':
                try:
                    rel_dt = datetime.fromtimestamp(int(rel_ts)).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    rel_dt = rel_ts
            else:
                rel_dt = rel_ts
            print(f"  {r['id']}  status={r['status']:<10}  release={rel_dt}  {r['title'][:60]}")
        return 0

    rc = record_live(
        video_id=args.video_id,
        quality=args.quality,
        max_duration=args.max_duration,
        wait_for_live=not args.no_wait,
        dry_run=args.dry_run,
    )
    return rc


if __name__ == '__main__':
    sys.exit(main())
