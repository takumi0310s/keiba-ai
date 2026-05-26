"""Analyze RA record byte offsets to find 分割タイム (lap time) fields.

Usage (run after jv_ra_lap_probe.ps1):
    python train/analyze_ra_offsets.py
    python train/analyze_ra_offsets.py --probe data/v20/ra_probe_20150101.txt
"""
from __future__ import annotations
import sys, re, json, os, argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def load_ra_records(probe_file: Path) -> list[str]:
    recs = []
    with open(probe_file, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('#') or not line:
                continue
            if line[:2] == 'RA':
                recs.append(line)
    return recs


def scan_for_lap_pattern(rec: str) -> dict:
    """Scan for patterns consistent with 分割タイム.

    Lap times are typically 3-digit values (tenths of seconds):
    - 1ハロン (200m) lap ≈ 11.0-14.0s → 110-140 in raw
    - Realistic range: 100-150 (10.0s-15.0s per furlong)
    Expect 6-18 consecutive 3-digit values in valid range.
    """
    results = {}
    # Scan every byte offset for a run of consecutive valid 3-digit lap values
    for start in range(0, min(len(rec) - 54, 800)):
        valid_run = 0
        for i in range(18):  # max 18 furlongs
            pos = start + i * 3
            if pos + 3 > len(rec):
                break
            chunk = rec[pos:pos+3]
            if not chunk.isdigit():
                break
            val = int(chunk)
            if 100 <= val <= 200:  # 10.0s-20.0s per furlong
                valid_run += 1
            else:
                break
        if valid_run >= 4:  # found at least 4 consecutive valid laps
            vals = [int(rec[start + i*3:start + i*3 + 3]) for i in range(valid_run)]
            results[start] = {'run': valid_run, 'vals': vals,
                              'secs': [v/10 for v in vals]}
    return results


def analyze(probe_file: Path) -> dict:
    recs = load_ra_records(probe_file)
    print(f"Loaded {len(recs)} RA records from {probe_file}")
    if not recs:
        print("[ERROR] No RA records found")
        return {}

    # Check record lengths
    lengths = [len(r) for r in recs]
    print(f"Record lengths: min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)/len(lengths):.0f}")

    # Print header fields (known positions)
    r0 = recs[0]
    print(f"\nFirst RA record (len={len(r0)}):")
    print(f"  [0:2]  RecordSpec = {repr(r0[0:2])}")
    print(f"  [2:3]  DataKubun  = {repr(r0[2:3])}")
    print(f"  [3:17] MakeDate?  = {repr(r0[3:17])}")
    print(f"  [3:11] DatePart?  = {repr(r0[3:11])}")
    print(f"  [11:19]???        = {repr(r0[11:19])}")
    print(f"  [19:21]CourseCd?  = {repr(r0[19:21])}")
    print(f"  [17:19]RaceNum?   = {repr(r0[17:19])}")

    # Find lap patterns across all records
    offset_votes: dict[int, int] = {}
    for rec in recs:
        patterns = scan_for_lap_pattern(rec)
        for offset in patterns:
            offset_votes[offset] = offset_votes.get(offset, 0) + 1

    # Sort by vote count
    ranked = sorted(offset_votes.items(), key=lambda x: -x[1])
    print(f"\nTop candidate offsets for 分割タイム (found in N/{len(recs)} records):")
    for offset, votes in ranked[:15]:
        pct = votes / len(recs) * 100
        # Show example values from first record
        example_rec = next((r for r in recs if offset in scan_for_lap_pattern(r)), None)
        if example_rec:
            pattern = scan_for_lap_pattern(example_rec)[offset]
            example = ' '.join(f"{v/10:.1f}" for v in pattern['vals'][:8])
        else:
            example = ''
        print(f"  offset={offset:4d}: {votes:3d}/{len(recs)} ({pct:.0f}%)  "
              f"run={offset_votes.get(offset, '?')}  example=[{example}]")

    # Also find 分割数 (lap count) field candidates
    # The count field should be 1 byte with value 6-18, immediately before laps
    if ranked:
        best_offset = ranked[0][0]
        print(f"\nBest candidate: offset={best_offset}")
        print(f"  Bytes before: {repr(recs[0][best_offset-5:best_offset])}")
        lap_vals = []
        for i in range(8):
            chunk = recs[0][best_offset+i*3:best_offset+i*3+3]
            if chunk.isdigit():
                lap_vals.append(f"{int(chunk)/10:.1f}")
        print(f"  Lap times:    {' '.join(lap_vals)}")

    out = {
        'n_records': len(recs),
        'lengths': {'min': min(lengths), 'max': max(lengths)},
        'top_offsets': [{'offset': o, 'votes': v, 'pct': v/len(recs)*100}
                        for o, v in ranked[:10]],
    }
    out_file = BASE / 'data' / 'v20' / 'ra_offset_analysis.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {out_file}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', default=None, help='Path to ra_probe_*.txt file')
    args = ap.parse_args()

    if args.probe:
        probe_file = Path(args.probe)
    else:
        # Auto-find latest probe file
        v20_dir = BASE / 'data' / 'v20'
        probes = sorted(v20_dir.glob('ra_probe_*.txt'))
        if not probes:
            print(f"[ERROR] No ra_probe_*.txt files found in {v20_dir}")
            print("  Run: C:\\Windows\\SysWOW64\\WindowsPowerShell\\v1.0\\powershell.exe "
                  "-File tools\\jv_ra_lap_probe.ps1")
            sys.exit(1)
        probe_file = probes[-1]
        print(f"Using: {probe_file}")

    analyze(probe_file)


if __name__ == '__main__':
    main()
