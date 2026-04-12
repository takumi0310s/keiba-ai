"""
毎朝の自動予測スクリプト
当日のJRA全レースを取得し、AI予測→買い目生成→CSV保存する。

Usage:
    python tools/daily_predict.py                  # 今日の予測
    python tools/daily_predict.py --date 20260315  # 日付指定
"""
import pandas as pd
import numpy as np
import re
import time
import os
import sys
import argparse
import requests
from datetime import datetime
from bs4 import BeautifulSoup

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

# === predict_core から全共通関数をインポート ===
from jrdb_features import merge_jrdb_predict_features
from predict_core import (
    HEADERS, INVESTMENT_PER_RACE, COURSE_MAP, SURFACE_MAP, COND_MAP, SEX_MAP,
    CONDITION_PROFILES, MODERN_JOCKEY_WR, SIRE_APT,
    RANK_TO_WOOD_4F, RANK_TO_SAKARO_4F, RANK_TO_SAKARO_3F,
    load_feature_lookups, find_jockey_wr, calc_sire_score,
    classify_race_condition, generate_trio_bets, generate_wide_bets, generate_umaren_bets,
    load_models, fetch_oikiri_ranks,
    parse_shutuba, get_horse_stats, build_features, predict_race,
    calc_pace_advantage, fetch_realtime_odds, fetch_realtime_odds_full,
    save_odds_base, is_race_started, fetch_result_odds,
    fetch_jra_and_weather, set_horse_defaults, apply_horse_stats,
)


# ===== レース一覧取得 =====

def fetch_race_list(date_str):
    """netkeibaからその日のレース一覧を取得
    date_str: YYYYMMDD形式

    Returns: list of dict {'race_id': str, 'course': str, 'race_num': int}
    """
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[ERROR] レース一覧取得失敗: {e}")
        return []

    races = []
    for dl in soup.find_all("dl", class_="RaceList_DataList"):
        dt = dl.find("dt")
        course_name = ""
        if dt:
            dt_text = dt.get_text(strip=True)
            for cn in COURSE_MAP:
                if cn in dt_text:
                    course_name = cn
                    break

        for a in dl.find_all("a", href=True):
            href = a.get("href", "")
            m = re.search(r'race_id=(\d{12})', href)
            if not m:
                m = re.search(r'/race/(\d{12})/', href)
            if m:
                race_id = m.group(1)
                race_num = 0
                nm = re.search(r'(\d{1,2})R', a.get_text(strip=True))
                if nm:
                    race_num = int(nm.group(1))
                else:
                    try:
                        race_num = int(race_id[-2:])
                    except ValueError:
                        pass

                if not any(r['race_id'] == race_id for r in races):
                    races.append({
                        'race_id': race_id,
                        'course': course_name,
                        'race_num': race_num,
                    })

    races.sort(key=lambda x: (x['course'], x['race_num']))
    return races


# ===== メイン処理 =====

def run_daily_predict(date_str):
    """指定日のJRA全レースを予測"""
    print(f"=" * 60)
    print(f"KEIBA AI 日次予測 - {date_str}")
    print(f"=" * 60)

    # モデルロード
    model_data = load_models()
    if model_data['model'] is None:
        print("[ERROR] モデルが見つかりません。終了します。")
        return

    # レース一覧取得
    print(f"\n[STEP 1] レース一覧取得中...")
    races = fetch_race_list(date_str)
    if not races:
        print(f"[INFO] {date_str} のレースが見つかりません（非開催日の可能性）")
        return

    print(f"  -> {len(races)}レース検出")
    for r in races:
        print(f"     {r['course']} {r['race_num']}R (race_id={r['race_id']})")

    # 各レースを予測
    results = []
    jra_weather_cache = {}

    for idx, race in enumerate(races):
        race_id = race['race_id']
        print(f"\n[STEP 2-{idx+1}/{len(races)}] {race['course']} {race['race_num']}R (ID={race_id})")

        try:
            # 出馬表取得
            race_name, horses, horse_ids, race_info = parse_shutuba(race_id)
            if not horses:
                print(f"  [WARN] 馬データなし、スキップ")
                continue

            # 障害レース自動除外
            if race_info.get('surface') == '障':
                print(f"  [SKIP] 障害レース（モデル非対応）")
                continue

            num_horses = len(horses)
            print(f"  レース名: {race_name} / {race_info['surface']}{race_info['distance']}m / {race_info['condition']} / {num_horses}頭")
            time.sleep(0.5)

            # オッズ取得
            race_started = is_race_started(race_id)
            pop_dict = {}
            if race_started:
                odds_dict, pop_dict = fetch_result_odds(race_id)
                if odds_dict:
                    print(f"  オッズ: 結果ページから{len(odds_dict)}頭分取得")
                else:
                    print(f"  オッズ: 結果ページから取得失敗")
            else:
                odds_full = fetch_realtime_odds_full(race_id)
                odds_dict = {u: v['odds'] for u, v in odds_full.items()}
                if odds_full:
                    save_odds_base(race_id, odds_full, date_str=date_str)
                print(f"  オッズ取得: {len(odds_dict)}頭分" if odds_dict else "  オッズ: 未取得（レース前オッズ未発表の可能性）")
            time.sleep(0.3)

            odds_available = len(odds_dict) > 0
            if odds_available:
                for horse in horses:
                    umaban = horse.get('馬番', 0)
                    if umaban in odds_dict:
                        horse['単勝オッズ'] = odds_dict[umaban]
                    if umaban in pop_dict:
                        horse['人気順位'] = pop_dict[umaban]
            if not odds_available:
                shutuba_odds = any(h.get('単勝オッズ', 0) > 0 for h in horses)
                if shutuba_odds:
                    odds_available = True
                    print(f"  オッズ: 出馬表ページから取得済み")

            # JRA馬場・天候（コースごとにキャッシュ）
            course_name = race_info.get('course', '')
            jra_info, weather_info = {}, {}
            if model_data['is_live'] and course_name:
                if course_name not in jra_weather_cache:
                    jra_info, weather_info = fetch_jra_and_weather(course_name)
                    jra_weather_cache[course_name] = (jra_info, weather_info)
                else:
                    jra_info, weather_info = jra_weather_cache[course_name]

            # 各馬の過去成績取得
            print(f"  各馬成績取得中...", end="", flush=True)
            for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
                if hid:
                    try:
                        stats = get_horse_stats(hid, race_info['distance'], race_info['surface'], course_name)
                        apply_horse_stats(horse, stats, race_info)
                    except Exception:
                        set_horse_defaults(horse)
                else:
                    set_horse_defaults(horse)
                if i < num_horses - 1:
                    time.sleep(0.5)
            print(f" 完了")

            # 特徴量構築 & 予測
            df = build_features(horses, race_info, model_data, race_id=race_id,
                                odds_dict=odds_dict, jra_track_info=jra_info, weather_info=weather_info)

            # JRDB特徴量マージ（KYI前日データ + TYB直前データ）
            try:
                df = merge_jrdb_predict_features(df, race_id)
            except Exception as e:
                print(f"    [JRDB] feature merge skipped: {e}")

            df = predict_race(df, model_data, odds_available, race_info=race_info)

            # 条件分類
            cond_key, cond_profile = classify_race_condition(race_info, num_horses)

            # 買い目生成
            sorted_df = df.sort_values('スコア', ascending=False).reset_index(drop=True)
            bet_type = cond_profile['bet_type']
            if bet_type == 'umaren':
                bets = generate_umaren_bets(sorted_df)
                bet_label = '馬連'
            elif bet_type == 'wide':
                bets = generate_wide_bets(sorted_df)
                bet_label = 'ワイド'
            else:
                bets = generate_trio_bets(sorted_df)
                bet_label = '三連複'

            top1 = sorted_df.iloc[0] if len(sorted_df) > 0 else None
            top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else None
            top3 = sorted_df.iloc[2] if len(sorted_df) > 2 else None

            bets_str = "; ".join(["-".join(str(n) for n in b) for b in bets])
            race_num_int = 0
            nm = re.search(r'(\d+)', str(race_info.get('race_num', '')))
            if nm:
                race_num_int = int(nm.group(1))

            row = {
                'race_id': race_id,
                'course': course_name,
                'race_num': race_num_int,
                'race_name': race_name,
                'condition': cond_key,
                'num_horses': num_horses,
                'distance': race_info['distance'],
                'surface': race_info['surface'],
                'track_condition': race_info['condition'],
                'top1_num': int(top1['馬番']) if top1 is not None else 0,
                'top1_name': top1['馬名'] if top1 is not None else '',
                'top1_score': float(top1['スコア']) if top1 is not None else 0,
                'top2_num': int(top2['馬番']) if top2 is not None else 0,
                'top2_name': top2['馬名'] if top2 is not None else '',
                'top3_num': int(top3['馬番']) if top3 is not None else 0,
                'top3_name': top3['馬名'] if top3 is not None else '',
                'trio_bets': bets_str,
                'bet_type': cond_profile['bet_type'],
                'investment': cond_profile['investment'],
            }
            results.append(row)

            # コンソール出力
            print(f"  条件: {cond_key} ({cond_profile['desc']})")
            if not cond_profile.get('recommended', True):
                print(f"  [WARNING] 購入非推奨（1000m以下：ROI 85%）")
            print(f"  TOP3: {top1['馬名']}({int(top1['馬番'])}) / {top2['馬名']}({int(top2['馬番'])}) / {top3['馬名']}({int(top3['馬番'])})")
            if bet_type == 'umaren' and len(bets) == 2:
                amts = [400, 300]
                bet_detail = ' / '.join(f'{"-".join(str(n) for n in b)}: {amts[i]}円' for i, b in enumerate(bets))
                print(f"  {bet_label} 2点: {bet_detail} (計700円)")
            else:
                print(f"  {bet_label} {len(bets)}点: {bets_str}")

            # Discord通知はrace_auto_notify.pyに一本化（重複防止）

        except Exception as e:
            print(f"  [ERROR] 予測失敗: {e}")
            import traceback
            traceback.print_exc()
            continue

    # CSV保存
    if results:
        out_dir = os.path.join(BASE_DIR, "data", "daily_predictions")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.csv")
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n{'=' * 60}")
        print(f"予測完了: {len(results)}レース")
        print(f"保存先: {out_path}")
        print(f"総投資額: {sum(r['investment'] for r in results):,}円")
        print(f"{'=' * 60}")
    else:
        print(f"\n[INFO] 予測結果なし")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEIBA AI 日次予測")
    parser.add_argument("--date", type=str, default=None,
                        help="予測日 YYYYMMDD (デフォルト: 今日)")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime("%Y%m%d")

    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"[ERROR] 日付形式が不正です: {date_str} (YYYYMMDD)")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_predict.py 開始")
    run_daily_predict(date_str)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_predict.py 終了")

    # Discord通知
    try:
        from notify import send_discord
        csv_path = os.path.join(BASE_DIR, "data", "daily_predictions", f"{date_str}.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            pdf = pd.read_csv(csv_path, encoding='utf-8-sig')
            n = len(pdf)
            total_inv = pdf['investment'].sum()
            cond_counts = pdf['condition'].value_counts().to_dict()
            cond_str = " / ".join(f"{k}:{v}件" for k, v in sorted(cond_counts.items()))

            top3 = pdf.nlargest(3, 'top1_score')
            top3_lines = []
            for _, r in top3.iterrows():
                if r.get('bet_type') == 'umaren':
                    top3_lines.append(
                        f"**{r['race_name']}** [{r['condition']}] "
                        f"馬連 軸:{int(r['top1_num'])}→{int(r['top2_num'])},{int(r['top3_num'])}")
                else:
                    top3_lines.append(
                        f"**{r['race_name']}** [{r['condition']}] "
                        f"三連複 軸:{int(r['top1_num'])} 2列:{int(r['top2_num'])},{int(r['top3_num'])}")

            msg = (f"**{date_str}** {n}レース予測完了\n"
                   f"条件: {cond_str}\n"
                   f"投資額: {total_inv:,}円\n\n"
                   + "\n".join(top3_lines))
            send_discord(f"予測完了 ({n}R)", msg, color="green", channel="updates")
        else:
            send_discord("予測完了", f"{date_str} 予測完了（結果0件）", color="yellow", channel="updates")
    except Exception:
        pass
