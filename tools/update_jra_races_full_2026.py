"""jra_races_full.csv 2026年4-5月分追加 (Phase 2.5 A/B/C unblock).

問題:
  jra_races_full.csv 38日 stale (3/27停止)、scrape_data_analysis.py /
  scrape_comments_bulk.py が race_id 列挙不能 (0 races to scrape)。

解決:
  daily_predictions/2026*.csv + daily_results/2026*.csv から race_id を抽出、
  netkeiba 12桁形式 → Target 10桁形式に変換、jra_races_full.csv に append。

Usage:
  python tools/update_jra_races_full_2026.py [--dry-run]
"""
import sys, os, argparse, glob
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)

JRA_CSV = 'data/jra_races_full.csv'
JRA_BAK = f'data/jra_races_full.csv.bak_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# 場名 → basho_code
COURSE_TO_BB = {'札幌':'01','函館':'02','福島':'03','新潟':'04',
                 '東京':'05','中山':'06','中京':'07','京都':'08',
                 '阪神':'09','小倉':'10'}

# nichi 数値 → Target 16進1桁
NICHI_TO_TGT = {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
                  10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F'}


def netkeiba_to_target_8(nk_rid):
    """netkeiba 12桁 → Target 10桁 (race level, suffix 01)."""
    if len(nk_rid) != 12:
        return None
    yyyy, bb, kk, nn, rr = nk_rid[0:4], nk_rid[4:6], nk_rid[6:8], nk_rid[8:10], nk_rid[10:12]
    yy = yyyy[2:]
    try:
        k = str(int(kk))  # 1-6 expected
        nn_int = int(nn)
        n_str = NICHI_TO_TGT.get(nn_int, str(nn_int))
    except Exception:
        return None
    return f'{bb}{yy}{k}{n_str}{rr}'  # 8 chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print("=" * 60)
    print(f"jra_races_full 2026年4-5月分追加 ({datetime.now()})")
    print("=" * 60)

    # === 既存 jra_races_full 読み込み ===
    print("\n=== Step 1: 既存 jra_races_full 読み込み ===")
    df_existing = pd.read_csv(JRA_CSV, dtype=str, low_memory=False, encoding='utf-8-sig')
    print(f"  rows: {len(df_existing):,}")
    print(f"  cols: {len(df_existing.columns)}")
    existing_rids = set(df_existing['race_id'].astype(str).str[:8].dropna().unique())
    print(f"  unique race-level ids (first 8 chars): {len(existing_rids)}")
    # 2026 既存
    df_2026 = df_existing[df_existing['year']=='26']
    print(f"  2026 rows: {len(df_2026)}")

    # === Step 2: daily_predictions / daily_results から race_id 収集 ===
    print("\n=== Step 2: 補充元 daily_predictions / daily_results 抽出 ===")
    new_races = {}  # netkeiba_rid → metadata

    pred_files = sorted(glob.glob('data/daily_predictions/2026*.csv'))
    res_files = sorted(glob.glob('data/daily_results/2026*.csv'))

    for fp in pred_files:
        # date from filename
        bn = os.path.basename(fp)
        if 'prerace' in bn or 'bak' in bn: continue
        try:
            ds = bn[:8]  # YYYYMMDD
            yyyy, mm, dd = ds[0:4], ds[4:6], ds[6:8]
            df = pd.read_csv(fp, dtype=str, low_memory=False)
            for _, r in df.iterrows():
                rid = str(r.get('race_id', '')).strip()
                if len(rid) != 12: continue
                if rid in new_races: continue
                course = str(r.get('course', '')).strip()
                race_num = str(r.get('race_num', '')).strip()
                race_name = str(r.get('race_name', '')).strip()
                num_horses = str(r.get('num_horses', '')).strip()
                distance = str(r.get('distance', '')).strip()
                surface = str(r.get('surface', '')).strip()
                condition = str(r.get('track_condition', '')).strip()
                bb = COURSE_TO_BB.get(course)
                if not bb: continue
                target_8 = netkeiba_to_target_8(rid)
                if not target_8: continue
                # year (2-digit), month, day from date_str
                yy2 = yyyy[2:]
                # kai, nichi from rid
                kk_int = int(rid[6:8])
                nn_int = int(rid[8:10])
                new_races[rid] = {
                    'year': yy2, 'month': mm.lstrip('0') or '0', 'day': dd.lstrip('0') or '0',
                    'kai': str(kk_int), 'course': course, 'nichi': str(nn_int),
                    'race_num': str(int(race_num)).zfill(2),
                    'race_name': race_name, 'class_code': '',
                    'surface': surface, 'distance': distance, 'condition': condition,
                    'num_horses': num_horses,
                    'race_id_target_8': target_8,
                    'date': ds,
                }
        except Exception as e:
            print(f"  WARN: {fp}: {e}")

    print(f"  daily_predictions から: {len(new_races)} races")

    # daily_results 補完
    n_before = len(new_races)
    for fp in res_files:
        bn = os.path.basename(fp)
        if not bn.startswith('2026') or '_payouts' in bn: continue
        try:
            ds = bn[:8]
            yyyy, mm, dd = ds[0:4], ds[4:6], ds[6:8]
            df = pd.read_csv(fp, dtype=str, encoding='utf-8-sig', low_memory=False)
            df = df.drop_duplicates('race_id', keep='last')
            for _, r in df.iterrows():
                rid = str(r.get('race_id','')).strip()
                if len(rid) != 12: continue
                if rid in new_races: continue
                course = str(r.get('course','')).strip()
                race_num = str(r.get('race_num','')).strip()
                race_name = str(r.get('race_name','')).strip()
                bb = COURSE_TO_BB.get(course)
                if not bb: continue
                target_8 = netkeiba_to_target_8(rid)
                if not target_8: continue
                yy2 = yyyy[2:]
                kk_int = int(rid[6:8])
                nn_int = int(rid[8:10])
                new_races[rid] = {
                    'year': yy2, 'month': mm.lstrip('0') or '0', 'day': dd.lstrip('0') or '0',
                    'kai': str(kk_int), 'course': course, 'nichi': str(nn_int),
                    'race_num': str(int(race_num)).zfill(2),
                    'race_name': race_name, 'class_code': '',
                    'surface': '', 'distance': '', 'condition': '',
                    'num_horses': '', 'race_id_target_8': target_8, 'date': ds,
                }
        except Exception as e:
            print(f"  WARN: {fp}: {e}")
    n_added_res = len(new_races) - n_before
    print(f"  daily_results 追加: +{n_added_res} races")
    print(f"  合計 unique races: {len(new_races)}")

    # === Step 3: 既存と重複除外 ===
    print("\n=== Step 3: 既存 jra_races_full との重複除外 ===")
    truly_new = {}
    for rid, meta in new_races.items():
        if meta['race_id_target_8'] not in existing_rids:
            truly_new[rid] = meta
    print(f"  既存に無い races: {len(truly_new)}")

    # === Step 4: race_name から class_code 推定 ===
    print("\n=== Step 4: class_code 推定 ===")
    # 11/12R OP/L → '70' or '67' 系、1勝 → 9, 2勝 → 30, 3勝 → 43, 未勝利 → 12, 新馬 → 15
    def infer_class_code(race_num, name):
        if not isinstance(name, str): return ''
        if '1勝' in name: return '9'
        if '2勝' in name: return '30'
        if '3勝' in name: return '43'
        if '未勝利' in name: return '12'
        if '新馬' in name: return '15'
        # 11R/12R 特別
        if race_num in ('11','12'): return '67'
        return ''

    for rid, m in truly_new.items():
        m['class_code'] = infer_class_code(m['race_num'], m['race_name'])

    # 11/12R 数 確認
    n_special = sum(1 for m in truly_new.values() if m['race_num'] in ('11','12'))
    print(f"  11R/12R: {n_special}")

    # === Step 5: jra_races_full に append (1行/race) ===
    print("\n=== Step 5: 新規行構築 ===")
    cols = list(df_existing.columns)
    rows = []
    for rid, m in truly_new.items():
        row = {c: '' for c in cols}
        row['year'] = m['year']
        row['month'] = m['month']
        row['day'] = m['day']
        row['kai'] = m['kai']
        row['course'] = m['course']
        row['nichi'] = m['nichi']
        row['race_num'] = m['race_num']
        row['race_name'] = m['race_name']
        row['class_code'] = m['class_code']
        if 'surface' in cols: row['surface'] = m.get('surface','')
        if 'distance' in cols: row['distance'] = m.get('distance','')
        if 'condition' in cols: row['condition'] = m.get('condition','')
        if 'num_horses' in cols: row['num_horses'] = m.get('num_horses','')
        # race_id (target 10桁: prefix 8 + horse 2 桁、horse seq=01)
        row['race_id'] = m['race_id_target_8'] + '01'
        rows.append(row)

    df_new = pd.DataFrame(rows)
    print(f"  new rows: {len(df_new)}")

    if args.dry_run:
        print("\n[DRY-RUN] 書き込みスキップ")
        print(df_new[['year','month','day','course','race_num','race_name','class_code','race_id']].head(20).to_string())
        return

    # === Step 6: backup + 書き込み ===
    print(f"\n=== Step 6: backup + write ===")
    print(f"  backup: {JRA_BAK}")
    import shutil
    shutil.copy(JRA_CSV, JRA_BAK)

    # Append
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    print(f"  combined rows: {len(df_combined):,}")

    # 書き込み (encoding utf-8-sig keep)
    df_combined.to_csv(JRA_CSV, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved {JRA_CSV}")

    # === Verify ===
    print("\n=== Verify ===")
    df_check = pd.read_csv(JRA_CSV, dtype=str, low_memory=False, usecols=['year','month','race_id'])
    print(f"  reload rows: {len(df_check):,}")
    df_2026 = df_check[df_check['year']=='26']
    print(f"  2026 rows: {len(df_2026)}")
    # Verify 5/3 races
    from_5_3 = sum(1 for rid in truly_new if rid.startswith('20260503'))
    from_5_2 = sum(1 for rid in truly_new if rid.startswith('20260502'))
    print(f"  5/2 races added: {from_5_2}")
    print(f"  5/3 races added: {from_5_3}")


if __name__ == '__main__':
    main()
