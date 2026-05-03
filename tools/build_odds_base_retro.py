"""odds_base retro 構築 — 5/2, 5/3 用に基準オッズ + 確定オッズの 2系統で生成.

Strategy:
1. Primary: jrdb_kyi.csv の 基準オッズ (前日朝オッズ) — phase 2 BT で 96% カバー実績
2. Fallback: netkeiba fetch_realtime_odds_full (post-race confirmed)
3. Output schema (既存 odds_base_DATE.csv 互換):
   race_id, horse_num, odds, pop_rank, timestamp

Usage:
  python tools/build_odds_base_retro.py --date 20260502
  python tools/build_odds_base_retro.py --date 20260503
"""
import sys, os, io, argparse, json
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

# Course code mapping for race_id construction
COURSE_TO_BASHO = {'札幌':'01','函館':'02','福島':'03','新潟':'04',
                    '東京':'05','中山':'06','中京':'07','京都':'08',
                    '阪神':'09','小倉':'10'}


def build_from_jrdb_kyi(date_str):
    """JRDB KYI 基準オッズ + 基準人気から odds_base 構築."""
    # KYI race_id format: 場(2)+年(2)+回(1)+日(1)+R(2) = 8 chars (Japanese cols)
    # Need to map to 12-digit netkeiba race_id (year YYYY + basho + kai + nichi + race)
    kyi = pd.read_csv('data/jrdb_kyi.csv', dtype=str, low_memory=False)
    kyi.columns = [c.replace('場コード','basho').replace('年','y2').replace('回','kai')
                    .replace('日','nichi').replace('R','rnum').replace('馬番','umaban_str')
                    .replace('血統登録番号','blood_num').replace('馬名','name')
                    for c in kyi.columns]

    # Parse date_str → year suffix + month/day for filtering
    yyyy = date_str[:4]
    yy2 = yyyy[2:]
    mm = date_str[4:6]
    dd = date_str[6:8]

    # We can't directly filter by date in KYI (no date col).
    # But for 5/2 and 5/3, we know race_ids from daily_predictions.
    pred_path = f'data/daily_predictions/{date_str}.csv'
    if not os.path.exists(pred_path):
        print(f"WARN: {pred_path} not found")
        return None
    df_pred = pd.read_csv(pred_path, dtype={'race_id':str})
    target_rids = set(df_pred['race_id'].astype(str).unique())
    print(f"  target race_ids: {len(target_rids)}")

    # Build kyi race_id (12-digit) and filter
    kyi['kyi_rid'] = (yyyy +
                       kyi['basho'].astype(str) +
                       kyi['kai'].astype(str).str.zfill(2).str.replace('00','0').apply(lambda x: x.zfill(2)) +
                       kyi['nichi'].astype(str).str.zfill(2).apply(lambda x: x.zfill(2)) +
                       kyi['rnum'].astype(str).str.zfill(2))
    # Re-build with proper int conversion
    def safe_pad(v, w):
        try:
            return f"{int(v):0{w}d}"
        except:
            return '00'
    kyi['kyi_rid'] = kyi.apply(lambda r: f"{yyyy}{r['basho']}{safe_pad(r['kai'],2)}{safe_pad(r['nichi'],2)}{safe_pad(r['rnum'],2)}", axis=1)
    # Filter
    kyi_target = kyi[kyi['kyi_rid'].isin(target_rids)].copy()
    print(f"  kyi rows matched: {len(kyi_target)} for {len(target_rids)} race_ids")

    if len(kyi_target) == 0:
        return None

    rows = []
    for _, r in kyi_target.iterrows():
        try:
            base_odds = float(r.get('基準オッズ', '0').strip()) if pd.notna(r.get('基準オッズ')) else 0
        except:
            try:
                base_odds = pd.to_numeric(r['基準オッズ'], errors='coerce')
            except:
                base_odds = 0
        try:
            pop_rank = int(r.get('基準人気順位', '0'))
        except:
            try:
                pop_rank = pd.to_numeric(r.get('基準人気順位'), errors='coerce') or 0
                pop_rank = int(pop_rank) if not pd.isna(pop_rank) else 0
            except:
                pop_rank = 0
        rows.append({
            'race_id': r['kyi_rid'],
            'horse_num': int(r['umaban_str']) if r['umaban_str'].isdigit() else 0,
            'odds': base_odds,
            'pop_rank': pop_rank,
            'timestamp': f"{yyyy}-{mm}-{dd} 06:00 (jrdb基準)",
        })

    df_odds = pd.DataFrame(rows)
    # Drop empty/zero odds
    df_odds = df_odds[df_odds['horse_num'] > 0]
    return df_odds


def build_from_netkeiba(date_str, missing_rids):
    """missing_rids について netkeiba 確定オッズ取得 (post-race)."""
    from predict_core import fetch_realtime_odds_full
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:8]
    rows = []
    for rid in missing_rids:
        try:
            odds_dict = fetch_realtime_odds_full(rid) or {}
        except Exception as e:
            print(f"  WARN: {rid} fetch failed: {e}")
            continue
        for uma, info in odds_dict.items():
            o = info if isinstance(info, (int, float)) else (info.get('odds', 0) if isinstance(info, dict) else 0)
            p = info.get('pop_rank', 0) if isinstance(info, dict) else 0
            rows.append({
                'race_id': str(rid),
                'horse_num': int(uma),
                'odds': float(o),
                'pop_rank': int(p),
                'timestamp': f"{yyyy}-{mm}-{dd} netkeiba_confirmed",
            })
    return pd.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    args = ap.parse_args()
    DATE = args.date

    print(f"=== odds_base retro for {DATE} ===")

    out_path = f'data/odds_base_{DATE}.csv'
    if os.path.exists(out_path):
        print(f"  {out_path} 既存 (skip overwrite)")
        return

    print("\nStep 1: jrdb_kyi 基準オッズから構築...")
    df_jrdb = build_from_jrdb_kyi(DATE)
    if df_jrdb is None:
        print("  jrdb 構築失敗")
        df_jrdb = pd.DataFrame()
    else:
        print(f"  jrdb: {len(df_jrdb)} rows")
        # Check coverage
        n_race_jrdb = df_jrdb['race_id'].nunique()
        print(f"  unique races: {n_race_jrdb}")

    # Determine missing race_ids (where jrdb 不足)
    pred_path = f'data/daily_predictions/{DATE}.csv'
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path, dtype={'race_id':str})
        target_rids = set(df_pred['race_id'].astype(str).unique())
        jrdb_rids = set(df_jrdb['race_id'].unique()) if len(df_jrdb) else set()
        missing = target_rids - jrdb_rids
        print(f"\nmissing rids (need netkeiba fetch): {len(missing)}")

        if missing:
            print("\nStep 2: netkeiba 確定オッズ補完...")
            df_nk = build_from_netkeiba(DATE, list(missing)[:50])  # limit 50 で速度確保
            if df_nk is not None and len(df_nk):
                print(f"  netkeiba: {len(df_nk)} rows")
                df_combined = pd.concat([df_jrdb, df_nk], ignore_index=True)
            else:
                df_combined = df_jrdb
        else:
            df_combined = df_jrdb
    else:
        df_combined = df_jrdb

    if df_combined is None or len(df_combined) == 0:
        print("\n❌ 構築失敗 (no rows)")
        return

    # Drop dups, save
    df_combined = df_combined.drop_duplicates(subset=['race_id', 'horse_num'], keep='first')
    df_combined['odds'] = pd.to_numeric(df_combined['odds'], errors='coerce').fillna(0)
    df_combined = df_combined[df_combined['odds'] > 0].copy()

    df_combined.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] Saved {out_path} ({len(df_combined)} rows, {df_combined['race_id'].nunique()} races)")

    # Summary
    print("\nSummary:")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  Unique race_ids: {df_combined['race_id'].nunique()}")
    print(f"  Mean odds: {df_combined['odds'].mean():.2f}")
    print(f"  Median odds: {df_combined['odds'].median():.2f}")


if __name__ == '__main__':
    main()
