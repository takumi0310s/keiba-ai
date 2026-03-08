"""
KDSCOPE NAR Full Data Extractor
Extracts and merges:
  - NS.DAT: per-horse race entries/results
  - NR.DAT: race-level info (distance, class)
  - O1/*/: Win (tansho) odds by race/horse
Output: data/chihou_full_enriched.csv
"""
import csv
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

KDSCOPE_DATA = "C:/KDSCOPE/Data"
OUTPUT_CSV = "data/chihou_full_enriched.csv"

SEX_MAP = {"1": "牡", "2": "牝", "3": "セン"}

# NAR course code → venue name mapping
COURSE_MAP = {
    "42": "大井", "43": "船橋", "44": "浦和", "45": "川崎",
    "46": "高知", "47": "佐賀", "48": "金沢", "50": "門別",
    "51": "盛岡", "54": "園田", "55": "笠松",
}


def parse_nr_records(path):
    """Parse NR.DAT for race-level info."""
    races = {}
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if len(line) < 30 or not line.startswith("NR"):
                continue
            race_date = line[11:19]
            course = line[19:21]
            race_no = line[21:23]
            class_code = line[23]
            distance = line[24:28]
            # Starters info at [48-50] (3 digits)
            num_starters_str = line[48:50].strip()
            num_starters = int(num_starters_str) if num_starters_str.isdigit() else 0

            key = (race_date, course, race_no)
            races[key] = {
                "class_code": class_code,
                "distance": int(distance) if distance.strip().isdigit() else 0,
                "num_starters": num_starters,
            }
    return races


def parse_ns_records(path, race_info):
    """Parse NS.DAT for per-horse race entries/results."""
    records = []
    seen = set()

    with open(path, "rb") as f:
        raw = f.read()

    for line_bytes in raw.split(b"\r\n"):
        if len(line_bytes) < 135 or not line_bytes.startswith(b"NS"):
            continue

        try:
            race_date = line_bytes[11:19].decode("ascii")
            if not race_date.isdigit():
                continue

            course = line_bytes[19:21].decode("ascii")
            race_no = line_bytes[21:23].decode("ascii")
            umaban = line_bytes[23:25].decode("ascii")
            horse_id = line_bytes[25:35].decode("ascii")

            dedup_key = (race_date, course, race_no, umaban, horse_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            horse_name = line_bytes[35:71].decode("cp932", errors="replace")
            horse_name = horse_name.replace("\u3000", "").strip()

            sex_code = chr(line_bytes[71]) if line_bytes[71] in (0x31, 0x32, 0x33) else ""
            sex = SEX_MAP.get(sex_code, "")

            age_str = line_bytes[75:77].decode("ascii", errors="replace").strip()
            age = int(age_str) if age_str.isdigit() else 0

            jockey_code = line_bytes[77:82].decode("ascii", errors="replace").strip()
            jockey_name = line_bytes[82:90].decode("cp932", errors="replace")
            jockey_name = jockey_name.replace("\u3000", "").strip()

            trainer_name = line_bytes[103:111].decode("cp932", errors="replace")
            trainer_name = trainer_name.replace("\u3000", "").strip()

            weight_str = line_bytes[119:122].decode("ascii", errors="replace").strip()
            weight = int(weight_str) if weight_str.isdigit() else 0

            pos_str = line_bytes[126:129].decode("ascii", errors="replace").strip()
            finish_pos = int(pos_str) if pos_str.isdigit() else 0

            time_min_str = line_bytes[129:132].decode("ascii", errors="replace").strip()
            time_sec_str = line_bytes[132:135].decode("ascii", errors="replace").strip()
            time_min = int(time_min_str) if time_min_str.isdigit() else 0
            time_sec_tenth = int(time_sec_str) if time_sec_str.isdigit() else 0
            time_seconds = time_min * 60 + time_sec_tenth / 10.0 if (time_min_str.isdigit() and time_sec_str.isdigit()) else 0.0
            if time_seconds > 0:
                m = int(time_seconds) // 60
                s = time_seconds - m * 60
                time_str = f"{m}:{s:04.1f}"
            else:
                time_str = ""

            race_key = (race_date, course, race_no)
            ri = race_info.get(race_key, {})
            distance = ri.get("distance", 0)
            class_code = ri.get("class_code", "")
            num_starters = ri.get("num_starters", 0)

            rd = race_date
            rec = {
                "race_date": f"{rd[:4]}-{rd[4:6]}-{rd[6:8]}",
                "course_code": course,
                "course_name": COURSE_MAP.get(course, ""),
                "race_no": int(race_no),
                "class_code": class_code,
                "distance": distance,
                "num_starters": num_starters,
                "umaban": int(umaban),
                "horse_id": horse_id,
                "horse_name": horse_name,
                "sex": sex,
                "age": age,
                "jockey_code": jockey_code,
                "jockey_name": jockey_name,
                "trainer_name": trainer_name,
                "weight": weight,
                "finish_pos": finish_pos,
                "finish_time": time_str,
                "finish_time_sec": round(time_seconds, 1) if time_seconds > 0 else "",
                "tansho_odds": "",  # Will be filled from O1
            }
            records.append(rec)

        except Exception:
            continue

    return records


def parse_o1_odds(o1_dir):
    """Parse all O1 odds files. Returns dict: (date, course, race_no, umaban) -> odds."""
    odds_data = {}

    if not os.path.exists(o1_dir):
        print(f"  O1 directory not found: {o1_dir}")
        return odds_data

    total_files = 0
    total_records = 0

    for year in sorted(os.listdir(o1_dir)):
        year_dir = os.path.join(o1_dir, year)
        if not os.path.isdir(year_dir):
            continue

        for fname in sorted(os.listdir(year_dir)):
            fpath = os.path.join(year_dir, fname)
            if not fname.endswith('.DAT'):
                continue
            total_files += 1

            with open(fpath, 'rb') as f:
                data = f.read()

            for rec_bytes in data.split(b'\r\n'):
                if len(rec_bytes) < 50 or not rec_bytes.startswith(b'O1'):
                    continue

                try:
                    s = rec_bytes.decode('ascii', errors='replace')
                    date2 = s[11:19]
                    course = s[19:21]

                    # Only process NAR courses
                    if course not in COURSE_MAP:
                        continue

                    race_no = s[25:27]  # Race number

                    # Parse starters
                    num_starters_str = s[35:37]
                    num_starters = int(num_starters_str) if num_starters_str.isdigit() else 0
                    if num_starters == 0:
                        num_starters_str2 = s[37:39]
                        num_starters = int(num_starters_str2) if num_starters_str2.isdigit() else 0

                    # Parse per-horse odds starting at position 43
                    # Format: horse_no(2) + odds(4) + rank(2) = 8 bytes per horse
                    pos = 43
                    while pos + 6 <= len(s):
                        horse_no_str = s[pos:pos+2]
                        odds_str = s[pos+2:pos+6]

                        if not horse_no_str.isdigit() or not odds_str.isdigit():
                            break

                        horse_no = int(horse_no_str)
                        odds_val = int(odds_str) / 10.0

                        if odds_val > 0:
                            key = (date2, course, race_no, f"{horse_no:02d}")
                            odds_data[key] = odds_val
                            total_records += 1

                        pos += 8  # 2(horse) + 4(odds) + 2(rank)

                        if horse_no >= 18:  # Max horses in a race
                            break

                except Exception:
                    continue

    print(f"  O1 files processed: {total_files}")
    print(f"  O1 odds records: {total_records}")
    return odds_data


def main():
    nr_path = os.path.join(KDSCOPE_DATA, "NR", "NR.DAT")
    ns_path = os.path.join(KDSCOPE_DATA, "NS", "NS.DAT")
    o1_dir = os.path.join(KDSCOPE_DATA, "O1")

    for path, name in [(nr_path, "NR.DAT"), (ns_path, "NS.DAT")]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)

    print("1. Parsing NR.DAT (race info)...")
    race_info = parse_nr_records(nr_path)
    print(f"  {len(race_info)} races found")

    print("2. Parsing NS.DAT (horse entries)...")
    records = parse_ns_records(ns_path, race_info)
    print(f"  {len(records)} entries found")

    print("3. Parsing O1 (win odds)...")
    odds_data = parse_o1_odds(o1_dir)

    # Merge odds into records
    print("4. Merging odds...")
    matched = 0
    for rec in records:
        rd = rec["race_date"].replace("-", "")
        key = (rd, rec["course_code"], f"{rec['race_no']:02d}", f"{rec['umaban']:02d}")
        if key in odds_data:
            rec["tansho_odds"] = odds_data[key]
            matched += 1

    print(f"  Odds matched: {matched}/{len(records)} ({100*matched/len(records):.1f}%)")

    # Stats
    dates = set(r["race_date"] for r in records)
    courses = set(r["course_code"] for r in records)
    unique_races = len(set((r["race_date"], r["course_code"], r["race_no"]) for r in records))
    with_odds = sum(1 for r in records if r["tansho_odds"] != "")
    with_dist = sum(1 for r in records if r["distance"] > 0)
    with_wt = sum(1 for r in records if r["weight"] > 0)
    with_pos = sum(1 for r in records if r["finish_pos"] > 0)

    print(f"\n=== Summary ===")
    print(f"  Date range: {min(dates)} ~ {max(dates)}")
    print(f"  Courses: {sorted(courses)} ({', '.join(COURSE_MAP.get(c,'?') for c in sorted(courses))})")
    print(f"  Unique races: {unique_races}")
    print(f"  Total entries: {len(records)}")
    print(f"  Avg entries/race: {len(records)/unique_races:.1f}")
    print(f"  Distance fill: {with_dist}/{len(records)} ({100*with_dist/len(records):.1f}%)")
    print(f"  Weight fill: {with_wt}/{len(records)} ({100*with_wt/len(records):.1f}%)")
    print(f"  Finish pos fill: {with_pos}/{len(records)} ({100*with_pos/len(records):.1f}%)")
    print(f"  Odds fill: {with_odds}/{len(records)} ({100*with_odds/len(records):.1f}%)")

    # Write CSV
    os.makedirs("data", exist_ok=True)
    columns = [
        "race_date", "course_code", "course_name", "race_no", "class_code",
        "distance", "num_starters", "umaban", "horse_id", "horse_name",
        "sex", "age", "jockey_code", "jockey_name", "trainer_name",
        "weight", "finish_pos", "finish_time", "finish_time_sec", "tansho_odds",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        records.sort(key=lambda r: (r["race_date"], r["course_code"], r["race_no"], r["umaban"]))
        writer.writerows(records)

    print(f"\nSaved to {OUTPUT_CSV} ({len(records)} rows)")


if __name__ == "__main__":
    main()
