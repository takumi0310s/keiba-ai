#!/usr/bin/env python
"""Phase 12 PoC: TFJV 2026 parsed CSV → 1 ヶ月 backfill JSON.

直近 1 ヶ月 (4/10-5/10) の RA/SE/HR records を per-race JSON に集約。
JV-Link 32-bit COM 環境 (5/24+) 切替時にも 同 schema 流用可。

入力:
  data/tfjv/RA_2026.csv   (RA records)
  data/tfjv/SE_2026.csv   (SE records、 馬毎)
  data/tfjv/HR_2026.csv   (HR records、 払戻)

出力:
  data/jvlink/2026/04/<race_id>.json
  data/jvlink/2026/05/<race_id>.json

V15 投資保護: read-only parse + 出力 dir 別 (data/jvlink/)。
predict_core.py / V15 model 完全不変。
"""
from __future__ import annotations
import csv
import json
import os
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

BASE = Path(__file__).resolve().parents[1]
TFJV_DIR = BASE / 'data' / 'tfjv'
JVLINK_DIR = BASE / 'data' / 'jvlink'
PERIOD_FROM = date(2026, 4, 10)
PERIOD_TO = date(2026, 5, 10)


def _build_race_id(course_code: str, year: str, kai: str, nichi: str, race_num: str) -> str:
    """JV race_id 構築 (12 digit、 既存 keiba-ai schema 互換)."""
    yy = year[2:] if len(year) == 4 else year
    return f"{year}{course_code}{kai}{nichi}{race_num}"  # 12 chars: YYYYCCKKNNRR


def _record_date(year: str, month_day: str) -> date | None:
    if not year or not month_day or len(month_day) != 4:
        return None
    try:
        return date(int(year), int(month_day[:2]), int(month_day[2:]))
    except Exception:
        return None


def _within_period(d: date | None) -> bool:
    return d is not None and PERIOD_FROM <= d <= PERIOD_TO


def load_ra_records() -> Dict[str, dict]:
    """RA records を race_id 別 dict 化、 1 ヶ月 filter."""
    out: Dict[str, dict] = {}
    fpath = TFJV_DIR / 'RA_2026.csv'
    with open(fpath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = _record_date(row.get('year', ''), row.get('month_day', ''))
            if not _within_period(d):
                continue
            rid = _build_race_id(row['course_code'], row['year'], row['kai'],
                                 row['nichi'], row['race_num'])
            out[rid] = {
                'race_id': rid,
                'date': d.isoformat(),
                'course_code': row['course_code'],
                'kai': row['kai'],
                'nichi': row['nichi'],
                'race_num': row['race_num'],
                'race_name': row.get('race_name', '').strip().replace('\x00', ''),
                'youbi_code': row.get('youbi_code', ''),
            }
    return out


def load_se_records(race_ids: set) -> Dict[str, List[dict]]:
    """SE records を race_id 別 dict (馬 list)."""
    out: Dict[str, List[dict]] = defaultdict(list)
    fpath = TFJV_DIR / 'SE_2026.csv'
    with open(fpath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = _build_race_id(row['course_code'], row['year'], row['kai'],
                                 row['nichi'], row['race_num'])
            if rid not in race_ids:
                continue
            out[rid].append({
                'umaban': row.get('umaban', ''),
                'wakuban': row.get('wakuban', ''),
                'horse_id': row.get('horse_id', '').strip(),
                'horse_name': row.get('horse_name', '').strip().replace('\x00', ''),
            })
    return dict(out)


def load_hr_records(race_ids: set) -> Dict[str, dict]:
    """HR records を race_id 別 dict (1 R 1 HR)."""
    out: Dict[str, dict] = {}
    fpath = TFJV_DIR / 'HR_2026.csv'
    with open(fpath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = _build_race_id(row['course_code'], row['year'], row['kai'],
                                 row['nichi'], row['race_num'])
            if rid not in race_ids:
                continue
            out[rid] = {
                'data_kbn': row.get('data_kbn', ''),
                'raw_payouts': row.get('raw_payouts', '')[:200],  # truncate (sample)
            }
    return out


def write_per_race_json(ra: Dict[str, dict], se: Dict[str, List[dict]],
                        hr: Dict[str, dict]) -> int:
    """per-race JSON を data/jvlink/<year>/<month>/<race_id>.json に書出."""
    n = 0
    for rid, ra_rec in ra.items():
        d = date.fromisoformat(ra_rec['date'])
        out_dir = JVLINK_DIR / f"{d.year:04d}" / f"{d.month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{rid}.json"
        rec = {
            'race_id': rid,
            'date': ra_rec['date'],
            'ra': ra_rec,
            'se': se.get(rid, []),
            'hr': hr.get(rid, {}),
            'source': 'TFJV_BINARY_2026',
            'phase': 'phase12_poc',
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        n += 1
    return n


def write_summary_index(ra: Dict[str, dict]) -> Path:
    """1 ヶ月 backfill summary index (race_id list) を生成."""
    out_path = JVLINK_DIR / 'phase12_poc_index.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'phase': 'phase12_poc',
        'period_from': PERIOD_FROM.isoformat(),
        'period_to': PERIOD_TO.isoformat(),
        'race_count': len(ra),
        'race_ids': sorted(ra.keys()),
        'generated_at': datetime.now().isoformat(),
        'source': 'TFJV_BINARY_2026 (RA/SE/HR records)',
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    print(f"[backfill_poc] Phase 12 PoC: {PERIOD_FROM} - {PERIOD_TO}")
    print(f"[backfill_poc] TFJV CSV dir: {TFJV_DIR}")
    print(f"[backfill_poc] output dir:   {JVLINK_DIR}")

    ra = load_ra_records()
    print(f"[backfill_poc] RA records (1 ヶ月 filter): {len(ra)}")

    if not ra:
        print("[backfill_poc] WARN: 1 ヶ月 内 RA records 0、 TFJV 2026 data が period 外の可能性")
        return

    se = load_se_records(set(ra.keys()))
    se_total = sum(len(v) for v in se.values())
    print(f"[backfill_poc] SE records: {se_total} (per-race avg {se_total / max(1, len(se)):.1f})")

    hr = load_hr_records(set(ra.keys()))
    print(f"[backfill_poc] HR records: {len(hr)}")

    n = write_per_race_json(ra, se, hr)
    idx_path = write_summary_index(ra)
    print(f"[backfill_poc] wrote per-race JSON: {n}")
    print(f"[backfill_poc] wrote summary index: {idx_path}")
    print("[backfill_poc] OK")


if __name__ == '__main__':
    main()
