#!/usr/bin/env python
"""netkeiba マスター DOM probe — Phase 18 A.

1 R に対し 4 系統 (AI 展開 / 波乱度 / 個別ラップ / トラックバイアス) の生 HTML を
保存し、 ユーザーが手動で BeautifulSoup REPL や ブラウザ DevTools で selector
真値化を行うための harness。

★ 重要 ★ 大量 fetch には使わない。 1 R で kill switch + 3 sec rate limit を
踏襲する。 出力は data/v18/dom_probe/{race_id}/ に gzip で保存。

Usage:
    python tools/netkeiba_master_dom_probe.py --race 202608030611
    python tools/netkeiba_master_dom_probe.py --race 202608030611 --umaban 5

出力:
    data/v18/dom_probe/{race_id}/
        ai_tenkai.html.gz
        ai_haran.html.gz
        lap.html.gz
        track_bias.html.gz
        probe_summary.json

ToS: 既存 netkeiba_master_scraper.py と同じ Cookie / rate limit / kill switch
を使用。 1 R fetch なので 4 request × 3 sec = 12 sec で完了。
"""
from __future__ import annotations
import os
import sys
import json
import gzip
import argparse
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'tools'))

from netkeiba_master_scraper import (  # noqa: E402
    URL_AI_TENKAI, URL_AI_HARAN, URL_LAP, URL_TRACK_BIAS,
    _make_session, _fetch, is_disabled,
)

PROBE_DIR = BASE_DIR / 'data' / 'v18' / 'dom_probe'


def _save_gz(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write(content)


def _quick_summary(html: str | None) -> dict:
    if not html:
        return {'ok': False, 'reason': 'empty', 'len': 0}
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return {'ok': True, 'len': len(html), 'note': 'bs4 unavailable'}
    soup = BeautifulSoup(html, 'html.parser')
    title = (soup.title.get_text(strip=True) if soup.title else '')[:120]
    table_count = len(soup.find_all('table'))
    div_classes = sorted({
        c for div in soup.find_all('div', class_=True)
        for c in (div.get('class') or [])
    })[:50]
    return {
        'ok': True,
        'len': len(html),
        'title': title,
        'tables': table_count,
        'sample_div_classes': div_classes,
    }


def probe_race(race_id: str, umaban: int = 1) -> dict:
    if is_disabled():
        return {'race_id': race_id, 'error': 'kill_switch_active'}

    session = _make_session()
    if session is None:
        return {'race_id': race_id, 'error': 'cookie_missing'}

    kaisai_id = race_id[:10] if len(race_id) >= 12 else race_id
    out_dir = PROBE_DIR / race_id
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ('ai_tenkai', URL_AI_TENKAI.format(race_id=race_id)),
        ('ai_haran', URL_AI_HARAN.format(race_id=race_id)),
        ('lap', URL_LAP.format(race_id=race_id)),
        ('track_bias', URL_TRACK_BIAS.format(kaisai_id=kaisai_id)),
    ]

    summary: dict = {
        'race_id': race_id,
        'umaban': umaban,
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
        'kaisai_id': kaisai_id,
        'pages': {},
    }

    for name, url in targets:
        html = _fetch(url, session)
        path = out_dir / f'{name}.html.gz'
        if html:
            _save_gz(path, html)
        summary['pages'][name] = {
            'url': url,
            'saved_path': str(path.relative_to(BASE_DIR)) if html else None,
            **_quick_summary(html),
        }

    summary_path = out_dir / 'probe_summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return summary


def _cli():
    p = argparse.ArgumentParser(description='netkeiba マスター DOM probe (Phase 18 A)')
    p.add_argument('--race', required=True, help='race_id (12 桁)')
    p.add_argument('--umaban', type=int, default=1)
    args = p.parse_args()
    s = probe_race(args.race, args.umaban)
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    _cli()
