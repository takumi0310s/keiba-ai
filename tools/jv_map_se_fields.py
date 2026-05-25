"""SE record フィールド精密マップ: 全数値フィールドを offset ごとに表示。"""
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
raw_file = BASE / "data/jvlink/se_raw.txt"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

records = []
with open(raw_file, "rb") as f:
    for line in f:
        b = line.rstrip(b"\r\n")
        if len(b) >= 2 and b[:2] == b"SE" and len(b) > 100:
            records.append(b)
        if len(records) >= 5:
            break

print(f"Total sample records: {len(records)}")
print(f"Record byte length: {len(records[0])}")

# ASCII フィールドのみスキャン (SJIS 2-byte は skip)
def ascii_fields(raw):
    """全数値 ASCII フィールドを抽出 (SJIS 2-byte の影響を除いた位置)。"""
    results = []
    i = 0
    while i < len(raw):
        b = raw[i]
        # SJIS 2-byte lead check
        if 0x81 <= b <= 0x9F or 0xE0 <= b <= 0xFC:
            i += 2  # skip SJIS 2-byte
            continue
        if 0x30 <= b <= 0x39:  # ASCII digit
            # 連続 digit を収集
            j = i
            while j < len(raw) and 0x30 <= raw[j] <= 0x39:
                j += 1
            field_len = j - i
            if 2 <= field_len <= 6:  # 2-6 char数値フィールド
                val = raw[i:j].decode("ascii")
                results.append((i, field_len, val))
            i = j
        elif b == 0x20:  # space
            i += 1
        else:
            i += 1
    return results

print("\n=== 全レコードの数値フィールド (offset, len, value) ===")
# 複数レコードで同じ offset に来るフィールドを特定
from collections import defaultdict
field_map = defaultdict(list)

for r in records:
    for offset, flen, val in ascii_fields(r):
        field_map[offset].append((flen, val))

# 全レコード共通の offset を表示
print(f"\n{'Offset':>6}  {'Len':>3}  {'Values (all records)':40}  Comment")
print("-" * 80)

# finish_time 候補: 700-1700 (4char), agari3f 候補: 300-400 (3char)
for offset in sorted(field_map.keys()):
    entries = field_map[offset]
    if len(entries) < 2:
        continue
    lens = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    # filter: all same length
    if len(set(lens)) > 1:
        continue
    flen = lens[0]
    # 数値が有意か確認
    try:
        nums = [int(v) for v in vals]
    except ValueError:
        continue

    comments = []
    # finish_time: 4char, 700-1700 (70-170 秒 ×10)
    if flen == 4 and all(700 <= n <= 1700 for n in nums):
        comments.append("** finish_time 候補 **")
    # agari3f: 3char, 290-400 (29-40 秒 ×10)
    if flen == 3 and all(290 <= n <= 400 for n in nums):
        comments.append("** agari3f 候補 **")
    # corner: 2char, 1-18
    if flen == 2 and all(1 <= n <= 18 for n in nums):
        comments.append("corner/umaban?")
    # carrying weight: 3char, 480-600 (48-60 kg ×10)
    if flen == 3 and all(480 <= n <= 600 for n in nums):
        comments.append("斤量 候補?")

    if comments or flen >= 3:
        print(f"  {offset:>6}  {flen:>3}  {str(vals):40}  {' '.join(comments)}")

print("\n=== Record 1 詳細 (bytes 280-340) ===")
r = records[0]
for i in range(280, min(340, len(r))):
    b = r[i]
    if 0x20 <= b < 0x7F:
        c = chr(b)
    else:
        c = f"[{b:02x}]"
    print(f"  byte {i:>3}: {b:02x} = {c}")
