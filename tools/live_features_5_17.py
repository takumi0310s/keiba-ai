#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5/17 race 用 動的 features 計算 tool (5/17 朝 LIVE 予測 用).

daily_predictions/{date}.csv (or 5/17 shutuba から) 入力:
- 各 race の各 horse_id を取得
- jra_races_full.csv (5/11 までの 過去 data) を lookup
- 各馬の hot_streak / class_down / pace_career / disadv 等 を 動的計算
- 5/17 race の features dict 出力

【V15 投資保護】 read-only、 V15 model 不変
【LEAK-free】 当該 race を含まない 過去 races のみ参照

Usage:
    # 5/17 shutuba から features 生成
    python tools/live_features_5_17.py 20260517

    # 任意 race_id list で 単発
    python tools/live_features_5_17.py --race-ids 202605020611,202605020612

【出力】 data/live_features/{date}.csv
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_history():
    """jra_races_full.csv を 過去 history として load."""
    import pandas as pd
    path = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(path, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'jockey_id', 'trainer_id',
                               'finish', 'year', 'month', 'day',
                               'distance', 'surface', 'class_code',
                               'pass1', 'pass2', 'pass3', 'pass4',
                               'agari_3f', 'num_horses'])
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = pd.to_numeric(df['horse_id'], errors='coerce').astype('Int64').astype(str)
    df['horse_id'] = df['horse_id'].str.replace('<NA>', '', regex=False)
    df['jockey_id'] = df['jockey_id'].astype(str)
    df['trainer_id'] = df['trainer_id'].astype(str)
    # date column 生成
    df['race_date'] = pd.to_datetime(
        (2000 + df['year']).astype(str) + '-' +
        df['month'].astype(str).str.zfill(2) + '-' +
        df['day'].astype(str).str.zfill(2),
        errors='coerce'
    )
    df = df.dropna(subset=['race_date'])
    return df


def compute_horse_features(horse_id, history_df, ref_date=None):
    """1 馬の features 動的計算 (LEAK-free、 ref_date 以前のみ)."""
    h = history_df[history_df['horse_id'] == str(horse_id)]
    if ref_date is not None:
        h = h[h['race_date'] < ref_date]
    h = h.sort_values('race_date')

    n = len(h)
    if n == 0:
        return {}

    last5 = h.tail(5)
    return {
        'horse_recent5_top3': last5['top3'].mean() if len(last5) > 0 else None,
        'horse_recent5_win': last5['win'].mean() if len(last5) > 0 else None,
        'horse_career_top3': h['top3'].mean(),
        'horse_career_n_races': n,
        'horse_last_finish': int(h.iloc[-1]['finish']) if n > 0 else None,
        'horse_last_class': int(h.iloc[-1]['class_code']) if n > 0 and not pd.isna(h.iloc[-1]['class_code']) else None,
        'horse_last_distance': int(h.iloc[-1]['distance']) if n > 0 and not pd.isna(h.iloc[-1]['distance']) else None,
        'horse_last_surface': h.iloc[-1]['surface'] if n > 0 else None,
        'horse_last_jockey': h.iloc[-1]['jockey_id'] if n > 0 else None,
        'horse_last_trainer': h.iloc[-1]['trainer_id'] if n > 0 else None,
        'horse_last_pass1': h.iloc[-1]['pass1'] if n > 0 and not pd.isna(h.iloc[-1]['pass1']) else None,
        'horse_last_pass4': h.iloc[-1]['pass4'] if n > 0 and not pd.isna(h.iloc[-1]['pass4']) else None,
        'horse_last_agari': h.iloc[-1]['agari_3f'] if n > 0 and not pd.isna(h.iloc[-1]['agari_3f']) else None,
    }


def compute_jockey_features(jockey_id, history_df, ref_date=None, window=30):
    h = history_df[history_df['jockey_id'] == str(jockey_id)]
    if ref_date is not None:
        h = h[h['race_date'] < ref_date]
    h = h.sort_values('race_date')
    recent = h.tail(window)
    if len(recent) == 0:
        return {}
    return {
        f'jockey_recent{window}_top3': recent['top3'].mean(),
        f'jockey_recent{window}_win': recent['win'].mean(),
        f'jockey_recent{window}_n': len(recent),
    }


def compute_trainer_features(trainer_id, history_df, ref_date=None, window=30):
    h = history_df[history_df['trainer_id'] == str(trainer_id)]
    if ref_date is not None:
        h = h[h['race_date'] < ref_date]
    h = h.sort_values('race_date')
    recent = h.tail(window)
    if len(recent) == 0:
        return {}
    return {
        f'trainer_recent{window}_top3': recent['top3'].mean(),
        f'trainer_recent{window}_win': recent['win'].mean(),
    }


def compute_jt_combo(jockey_id, trainer_id, history_df, ref_date=None):
    h = history_df[
        (history_df['jockey_id'] == str(jockey_id)) &
        (history_df['trainer_id'] == str(trainer_id))
    ]
    if ref_date is not None:
        h = h[h['race_date'] < ref_date]
    if len(h) == 0:
        return None
    return h['top3'].mean()


def jackpot_check(horse_feats, jockey_feats, current_class, current_jockey):
    """4-way Jackpot pattern check."""
    if not horse_feats or not jockey_feats:
        return False
    horse_recent5 = horse_feats.get('horse_recent5_top3', 0)
    jockey_recent30 = jockey_feats.get('jockey_recent30_top3', 0)
    if horse_recent5 is None or jockey_recent30 is None:
        return False
    # class_down: 当該 < 前走
    last_class = horse_feats.get('horse_last_class')
    if last_class is None or current_class is None:
        return False
    class_down = current_class < last_class
    # jockey_change: 同騎手か
    last_jockey = horse_feats.get('horse_last_jockey')
    jockey_same = (last_jockey == str(current_jockey))
    return (class_down and horse_recent5 >= 0.6 and
            jockey_recent30 >= 0.30 and jockey_same)


def main():
    ap = argparse.ArgumentParser(description='5/17 race 動的 features 計算')
    ap.add_argument('date', nargs='?', default=None,
                    help='YYYYMMDD (daily_predictions/{date}.csv 使用)')
    ap.add_argument('--race-ids', dest='race_ids', default=None,
                    help='comma-separated race_id list (test 用)')
    args = ap.parse_args()

    import pandas as pd

    print('[INFO] loading history (jra_races_full.csv)...')
    history = load_history()
    print(f'  history shape: {history.shape}')
    print(f'  history latest date: {history["race_date"].max()}')

    if args.date is None and args.race_ids is None:
        print('[ERROR] --date or --race-ids required')
        return 1

    # Determine target races
    if args.date:
        daily_path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{args.date}.csv')
        if not os.path.exists(daily_path):
            print(f'[ERROR] daily_predictions/{args.date}.csv not found')
            return 1
        daily = pd.read_csv(daily_path, encoding='utf-8-sig')
        race_ids = daily['race_id'].astype(str).unique().tolist()
        # ref_date = 当該 date (LEAK 防止、 当日 race を 計算時に 当日 race は除外)
        ref_date = pd.to_datetime(args.date)
        print(f'[INFO] target date: {args.date}, races: {len(race_ids)}')
    else:
        race_ids = args.race_ids.split(',')
        ref_date = pd.Timestamp.now()
        print(f'[INFO] target race_ids: {len(race_ids)}')

    # cookies for netkeiba shutuba
    cookies_path = os.path.join(BASE_DIR, 'data', 'cookies.json')
    if os.path.exists(cookies_path):
        cookies_list = json.load(open(cookies_path, 'r', encoding='utf-8'))
        cookies = {c['name']: c['value'] for c in cookies_list}
    else:
        cookies = {}

    # Process each race
    output_rows = []
    n_jackpot = 0
    for race_id in race_ids[:50]:  # safety limit
        # 当該 race の shutuba 取得 (netkeiba)
        import requests, re
        try:
            url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
            r = requests.get(url, cookies=cookies, timeout=15,
                              headers={'User-Agent': 'Mozilla/5.0'})
            r.encoding = 'euc-jp'
            html = r.text
        except Exception as e:
            print(f'[WARN] {race_id}: fetch error {e}')
            continue

        # 馬番 + horse_id + jockey_id 抽出 (簡易)
        # netkeiba HTML から複数の (umaban, horse_id, jockey_id) を取得
        horses_found = []
        pattern = re.compile(
            r'<td[^>]+class="[^"]*Umaban[^"]*"[^>]*>\s*(\d+)\s*</td>'
            r'[\s\S]{0,3000}?/horse/(\d{10})/?[\s\S]{0,2000}?/jockey/result/recent/(\d+)/?',
        )
        for m in pattern.finditer(html):
            umaban = int(m.group(1))
            h_id = m.group(2)
            j_id = m.group(3)
            if not any(h['umaban'] == umaban for h in horses_found):
                horses_found.append({'umaban': umaban, 'horse_id': h_id, 'jockey_id': j_id})

        # current race info (class_code) - 推定 (netkeiba race info から)
        # 簡易: 取れなければ 0
        cls_match = re.search(r'クラス(\d+)', html)
        current_class = int(cls_match.group(1)) if cls_match else 0

        if not horses_found:
            continue

        for h in horses_found:
            hf = compute_horse_features(h['horse_id'], history, ref_date=ref_date)
            jf = compute_jockey_features(h['jockey_id'], history, ref_date=ref_date)
            tf = compute_trainer_features(hf.get('horse_last_trainer', ''),
                                            history, ref_date=ref_date)
            jt = compute_jt_combo(h['jockey_id'], hf.get('horse_last_trainer', ''),
                                    history, ref_date=ref_date)
            is_jpot = jackpot_check(hf, jf, current_class, h['jockey_id'])
            if is_jpot:
                n_jackpot += 1

            row = {
                'race_id': race_id,
                'umaban': h['umaban'],
                'horse_id': h['horse_id'],
                'jockey_id': h['jockey_id'],
                'is_jackpot': int(is_jpot),
                **hf, **jf, **tf,
                'jockey_trainer_combo_top3': jt,
            }
            output_rows.append(row)

    # Output
    out_dir = os.path.join(BASE_DIR, 'data', 'live_features')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.date or "manual"}.csv')

    if output_rows:
        all_keys = sorted({k for r in output_rows for k in r.keys()})
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in output_rows:
                w.writerow({k: r.get(k) for k in all_keys})
        print(f'\n[OK] {len(output_rows)} horses processed')
        print(f'[OK] Jackpot 該当: {n_jackpot}')
        print(f'[OK] saved: {out_path}')
    else:
        print('[WARN] no horses processed')

    return 0


if __name__ == '__main__':
    import pandas as pd
    sys.exit(main())
