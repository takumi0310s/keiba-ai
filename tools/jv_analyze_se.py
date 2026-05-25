"""SE raw record byte structure analyzer.
Parse se_raw.txt (cp932-encoded) to map field positions.
"""
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
raw_file = BASE / "data/jvlink/se_raw.txt"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# SE record fields to discover from JV-Data spec
# We'll print byte ranges and their decoded values to map the full layout.

records_bytes = []
with open(raw_file, "rb") as f:
    for line in f:
        b = line.rstrip(b"\r\n")
        if len(b) >= 2 and b[:2] == b"SE":
            records_bytes.append(b)
        if len(records_bytes) >= 3:
            break

if not records_bytes:
    print("No SE records found")
    sys.exit(1)

print(f"Found {len(records_bytes)} sample records")

for idx, raw in enumerate(records_bytes[:2]):
    print(f"\n{'='*70}")
    print(f"Record {idx+1}: byte_len={len(raw)}")
    print()

    def s(start, n, enc="ascii"):
        chunk = raw[start:start+n]
        try:
            v = chunk.decode(enc, errors="replace").strip()
        except Exception:
            v = repr(chunk)
        return v

    # Known fields from JV-Data spec SE record:
    print("=== Known header fields ===")
    print(f"  [  0- 1] record_type   : '{s(0,2)}'")
    print(f"  [  2- 2] data_kbn      : '{s(2,1)}'")
    print(f"  [  3-10] make_date     : '{s(3,8)}'")
    print(f"  [ 11-12] year_holding  : '{s(11,2)}'")  # holding year YY
    print(f"  [ 13-16] month_day     : '{s(13,4)}'")  # MMDD
    print(f"  [ 17-18] course_code   : '{s(17,2)}'")
    print(f"  [ 19-20] kai           : '{s(19,2)}'")
    print(f"  [ 21-22] nichi         : '{s(21,2)}'")
    print(f"  [ 23-24] race_num      : '{s(23,2)}'")
    print(f"  [ 25-25] wakuban       : '{s(25,1)}'")
    print(f"  [ 26-27] umaban        : '{s(26,2)}'")
    print(f"  [ 28-37] horse_id      : '{s(28,10)}'")
    print(f"  [ 38-73] horse_name    : '{s(38,36,'cp932')}'")  # 36 SJIS bytes

    # After horse_name (byte 74+):
    print("\n=== Scanning bytes 74-200 for result fields ===")
    for offset in range(74, min(200, len(raw)-10), 1):
        chunk2 = raw[offset:offset+6]
        try:
            v2 = chunk2.decode("ascii", errors="?")
            if all(c.isdigit() or c in " 0" for c in v2) and v2.strip():
                print(f"  [{offset:>3}-{offset+5:>3}] '{v2}' (potential numeric field)")
        except Exception:
            pass

    # Print bytes 70-200 in groups
    print("\n=== Bytes 70-220 (hex + ascii) ===")
    for i in range(70, min(220, len(raw)), 10):
        chunk = raw[i:i+10]
        hex_s = chunk.hex()
        try:
            asc = chunk.decode("cp932", errors=".").replace("\n",".")
        except Exception:
            asc = "?"
        print(f"  {i:>3}: {hex_s}  |{asc}|")

    # Scan for known patterns: finish_time (4-digit seconds*10), agari3f (3-digit)
    # Race finish time for typical horse: 70-170 seconds * 10 = 700-1700
    print("\n=== Scanning for finish_time pattern (700-1700 as 4-char) ===")
    for offset in range(100, min(300, len(raw)-4)):
        chunk = raw[offset:offset+4]
        try:
            v = int(chunk.decode("ascii").strip())
            if 700 <= v <= 1700:
                print(f"  byte {offset}: '{chunk.decode('ascii')}' = {v/10:.1f}s (finish time candidate)")
        except Exception:
            pass

    print("\n=== Scanning for agari3f pattern (300-400 as 3-char) ===")
    for offset in range(100, min(300, len(raw)-3)):
        chunk = raw[offset:offset+3]
        try:
            v = int(chunk.decode("ascii").strip())
            if 300 <= v <= 400:
                print(f"  byte {offset}: '{chunk.decode('ascii')}' = {v/10:.1f}s (agari3f candidate)")
        except Exception:
            pass

# Also: show full record as hex groups for manual inspection
print("\n=== Full byte layout (first record) ===")
raw = records_bytes[0]
for i in range(0, len(raw), 20):
    chunk = raw[i:i+20]
    hex_s = " ".join(f"{b:02x}" for b in chunk)
    try:
        asc = chunk.decode("cp932", errors=".").replace("\n", ".")
    except Exception:
        asc = "?"
    print(f"  {i:>3}: {hex_s}  |{asc}|")
