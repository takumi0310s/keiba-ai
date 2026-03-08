"""
KDSCOPE NV-Data → CSV converter
地方競馬(NAR) データを C:/KDSCOPE/Data/ から読み込み、CSV出力する

NS.DAT: per-horse race entries/results (bytes 11-18 = actual race date)
NR.DAT: race-level info with distance (bytes 11-18 = actual race date)
"""
import csv
import os
import sys

KDSCOPE_DATA = "C:/KDSCOPE/Data"
OUTPUT_CSV = "data/chihou_races_full.csv"

SEX_MAP = {"1": "牡", "2": "牝", "3": "セン"}


def parse_nr_records(path):
    """Parse NR.DAT for race-level info (distance, class, etc.)
    Key = (race_date, course, race_no) using date2 (bytes 11-18) as race date.
    """
    races = {}
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if len(line) < 30 or not line.startswith("NR"):
                continue
            race_date = line[11:19]  # date2 = actual race date
            course = line[19:21]
            race_no = line[21:23]
            class_code = line[23]       # 1-7, likely race grade
            distance = line[24:28]      # 4-digit distance in meters

            key = (race_date, course, race_no)
            races[key] = {
                "class_code": class_code,
                "distance": int(distance) if distance.strip().isdigit() else 0,
            }
    return races


def parse_ns_records(path, race_info):
    """Parse NS.DAT for per-horse race entries/results.
    Uses date2 (bytes 11-18) as the actual race date.
    Deduplicates by (race_date, course, race_no, umaban, horse_id).
    """
    records = []
    seen = set()

    with open(path, "rb") as f:
        raw = f.read()

    for line_bytes in raw.split(b"\r\n"):
        if len(line_bytes) < 135 or not line_bytes.startswith(b"NS"):
            continue

        try:
            race_date = line_bytes[11:19].decode("ascii")  # date2 = actual race date
            if not race_date.isdigit():
                continue

            course = line_bytes[19:21].decode("ascii")
            race_no = line_bytes[21:23].decode("ascii")
            umaban = line_bytes[23:25].decode("ascii")
            horse_id = line_bytes[25:35].decode("ascii")

            # Deduplicate (keep first occurrence)
            dedup_key = (race_date, course, race_no, umaban, horse_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Horse name: bytes 35-70 (36 bytes, cp932)
            horse_name = line_bytes[35:71].decode("cp932", errors="replace")
            horse_name = horse_name.replace("\u3000", "").strip()

            # Sex: byte 71
            sex_code = chr(line_bytes[71]) if line_bytes[71] in (0x31, 0x32, 0x33) else ""
            sex = SEX_MAP.get(sex_code, "")

            # Age: bytes 75-76
            age_str = line_bytes[75:77].decode("ascii", errors="replace").strip()
            age = int(age_str) if age_str.isdigit() else 0

            # Jockey code: bytes 77-81
            jockey_code = line_bytes[77:82].decode("ascii", errors="replace").strip()

            # Jockey name: bytes 82-89 (8 bytes, cp932, 4 kanji)
            jockey_name = line_bytes[82:90].decode("cp932", errors="replace")
            jockey_name = jockey_name.replace("\u3000", "").strip()

            # Trainer name: bytes 103-110 (8 bytes, cp932, 4 kanji)
            trainer_name = line_bytes[103:111].decode("cp932", errors="replace")
            trainer_name = trainer_name.replace("\u3000", "").strip()

            # Horse weight: bytes 119-121 (3 ASCII digits)
            weight_str = line_bytes[119:122].decode("ascii", errors="replace").strip()
            weight = int(weight_str) if weight_str.isdigit() else 0

            # Finish position: bytes 126-128 (3 digits)
            pos_str = line_bytes[126:129].decode("ascii", errors="replace").strip()
            finish_pos = int(pos_str) if pos_str.isdigit() else 0

            # Finish time: bytes 129-131 (minutes), 132-134 (seconds+tenths)
            time_min_str = line_bytes[129:132].decode("ascii", errors="replace").strip()
            time_sec_str = line_bytes[132:135].decode("ascii", errors="replace").strip()
            time_min = int(time_min_str) if time_min_str.isdigit() else 0
            time_sec_tenth = int(time_sec_str) if time_sec_str.isdigit() else 0
            # Convert: minutes * 60 + seconds_with_tenths / 10
            time_seconds = time_min * 60 + time_sec_tenth / 10.0 if (time_min_str.isdigit() and time_sec_str.isdigit()) else 0.0
            if time_seconds > 0:
                m = int(time_seconds) // 60
                s = time_seconds - m * 60
                time_str = f"{m}:{s:04.1f}"
            else:
                time_str = ""

            # Get distance from NR race info
            race_key = (race_date, course, race_no)
            ri = race_info.get(race_key, {})
            distance = ri.get("distance", 0)
            class_code = ri.get("class_code", "")

            rd = race_date
            rec = {
                "race_date": f"{rd[:4]}-{rd[4:6]}-{rd[6:8]}",
                "course_code": course,
                "race_no": int(race_no),
                "class_code": class_code,
                "distance": distance,
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
            }
            records.append(rec)

        except Exception:
            continue  # Skip malformed records

    return records


def main():
    nr_path = os.path.join(KDSCOPE_DATA, "NR", "NR.DAT")
    ns_path = os.path.join(KDSCOPE_DATA, "NS", "NS.DAT")

    if not os.path.exists(nr_path):
        print(f"ERROR: {nr_path} not found")
        sys.exit(1)
    if not os.path.exists(ns_path):
        print(f"ERROR: {ns_path} not found")
        sys.exit(1)

    print("Parsing NR.DAT (race info)...")
    race_info = parse_nr_records(nr_path)
    print(f"  {len(race_info)} races found")

    print("Parsing NS.DAT (horse entries)...")
    records = parse_ns_records(ns_path, race_info)
    print(f"  {len(records)} entries found")

    # Stats
    dates = set(r["race_date"] for r in records)
    courses = set(r["course_code"] for r in records)
    unique_races = len(set((r["race_date"], r["course_code"], r["race_no"]) for r in records))
    print(f"\n  Date range: {min(dates)} ~ {max(dates)}")
    print(f"  Courses: {sorted(courses)}")
    print(f"  Unique races: {unique_races}")
    print(f"  Avg entries/race: {len(records)/unique_races:.1f}")

    # Distance match rate
    with_dist = sum(1 for r in records if r["distance"] > 0)
    print(f"  Distance match rate: {with_dist}/{len(records)} ({100*with_dist/len(records):.1f}%)")

    # Weight fill rate
    with_wt = sum(1 for r in records if r["weight"] > 0)
    print(f"  Weight fill rate: {with_wt}/{len(records)} ({100*with_wt/len(records):.1f}%)")

    # Finish position fill rate
    with_pos = sum(1 for r in records if r["finish_pos"] > 0)
    print(f"  Finish position fill rate: {with_pos}/{len(records)} ({100*with_pos/len(records):.1f}%)")

    # Write CSV
    os.makedirs("data", exist_ok=True)
    columns = [
        "race_date", "course_code", "race_no", "class_code", "distance",
        "umaban", "horse_id", "horse_name", "sex", "age",
        "jockey_code", "jockey_name", "trainer_name",
        "weight", "finish_pos", "finish_time", "finish_time_sec",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        records.sort(key=lambda r: (r["race_date"], r["course_code"], r["race_no"], r["umaban"]))
        writer.writerows(records)

    print(f"\nSaved to {OUTPUT_CSV} ({len(records)} rows)")


if __name__ == "__main__":
    main()
