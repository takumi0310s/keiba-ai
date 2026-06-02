"""
毎朝の自動予測スクリプト
当日のJRA全レースを取得し、AI予測→買い目生成→CSV保存する。

Usage:
    python tools/daily_predict.py                  # 今日の予測
    python tools/daily_predict.py --date 20260315  # 日付指定
    python tools/daily_predict.py --resume         # 既存CSVから未予測レースだけ再実行
"""
# === Windows コンソール CLOSE イベント対策 (Intel Fortran/LightGBM/OpenMP) ===
# この設定は import 前に済ませる必要がある。forrtl error (200) 回避用。
import os
if os.name == "nt":
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import numpy as np
import re
import time
import sys
import argparse
import signal
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

try:
    from paper_shadow_v15_full import run_paper_shadow_comparison, log_paper_shadow
    _PAPER_SHADOW_AVAILABLE = True
except ImportError:
    _PAPER_SHADOW_AVAILABLE = False

try:
    from build_allscores_txt import save_race_allscores, build_allscores_txt
    from build_allscores_html import build_allscores_html
    _ALLSCORES_AVAILABLE = True
except ImportError:
    _ALLSCORES_AVAILABLE = False

# V16 ability model (lazy load)
_v16_model_cache = None

def _load_v16_model_daily():
    global _v16_model_cache
    if _v16_model_cache is not None:
        return _v16_model_cache
    import gzip, pickle
    mp = os.path.join(BASE_DIR, 'models', 'v16_ability_candidate.pkl.gz')
    if not os.path.exists(mp):
        return None
    try:
        with gzip.open(mp, 'rb') as f:
            _v16_model_cache = pickle.load(f)
        print(f"  [V16] model loaded ({len(_v16_model_cache.get('features',[]))} feats)")
    except Exception as e:
        print(f"  [V16] load failed: {e}")
        _v16_model_cache = {}
    return _v16_model_cache

def _get_v16_scores_daily(df, v16_data):
    """全馬の V16 スコアを {馬番str: score} で返す。"""
    if not v16_data or 'features' not in v16_data:
        return {}
    try:
        import xgboost as xgb
        feats = v16_data['features']
        df_in = df.reindex(columns=feats).fillna(0.0)
        X = df_in.values.astype('float32')
        p_lgb = v16_data['model'].predict(X)
        w = v16_data.get('ensemble_weights', {'lgb': 0.5, 'xgb': 0.5})
        if 'xgb_model' in v16_data:
            p_xgb = v16_data['xgb_model'].predict(xgb.DMatrix(X, feature_names=feats))
            scores = w.get('lgb', 0.5) * p_lgb + w.get('xgb', 0.5) * p_xgb
        else:
            scores = p_lgb
        return {str(int(df.iloc[i].get('馬番', i))): float(scores[i])
                for i in range(len(scores))}
    except Exception as e:
        print(f"  [V16] score error: {e}")
        return {}


# ===== 逐次CSV書き込み =====

PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "daily_predictions")

# CSV列順（新規作成時のヘッダ + append時の整合性確保用）
_CSV_COLUMNS = [
    'race_id', 'course', 'race_num', 'race_name', 'condition', 'num_horses',
    'distance', 'surface', 'track_condition',
    'top1_num', 'top1_name', 'top1_score',
    'top2_num', 'top2_name', 'top3_num', 'top3_name',
    'trio_bets', 'bet_type', 'investment',
    # 2026-05-29: 買い/見送り判定 (戦略⑦、 結果照合の3区分集計用)。
    # ★ investment は不変 (加算方式)。 should_bet=False は記録のみ・賭けない ★
    'should_bet', 'skip_reason',
]


def _csv_path_for(date_str):
    return os.path.join(PREDICTIONS_DIR, f"{date_str}.csv")


def _load_existing_race_ids(date_str):
    """既存CSVから race_id セットを返す。ファイル無ければ空set。"""
    path = _csv_path_for(date_str)
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', dtype={'race_id': str})
        return set(df['race_id'].astype(str).tolist())
    except Exception as e:
        print(f"[WARN] 既存CSV読み込み失敗 ({path}): {e}")
        return set()


def _append_prediction_to_csv(row, date_str):
    """1レース分の予測結果をCSVに即時append。クラッシュ耐性のためflush付。"""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    path = _csv_path_for(date_str)
    new_file = not os.path.exists(path)
    df_row = pd.DataFrame([row], columns=_CSV_COLUMNS)
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        df_row.to_csv(f, index=False, header=new_file)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _v2_log_phase1_safe(row, race_info, date_str, odds_dict=None, sorted_df=None):
    """race_notify_log v2 phase 1 (morning_predict) safe wrapper。

    ★ 既存 daily_predict logic 完全不変、 log 出力 file IO のみ ★

    log fail / import fail で例外を投げない (daily_predict 続行)。
    """
    try:
        # tools/ を sys.path に追加して import 可能化
        _tools_dir = os.path.dirname(os.path.abspath(__file__))
        if _tools_dir not in sys.path:
            sys.path.insert(0, _tools_dir)
        from race_notify_log_v2 import log_phase1 as _v2_log_phase1

        race_id = row.get('race_id')

        race_meta = {
            'race_name': row.get('race_name', ''),
            'course': row.get('course', ''),
            'race_num': row.get('race_num', 0),
            'distance': row.get('distance', 0),
            'surface': row.get('surface', ''),
            'track_condition': row.get('track_condition', ''),
            'num_horses': row.get('num_horses', 0),
            'condition': row.get('condition', ''),
            'bet_type': row.get('bet_type', ''),
            'investment': row.get('investment', 0),
        }

        # top1-5 ranking
        ranking_top5 = []
        try:
            if sorted_df is not None and len(sorted_df) > 0:
                for i in range(min(5, len(sorted_df))):
                    r = sorted_df.iloc[i]
                    ranking_top5.append({
                        'rank': i + 1,
                        'umaban': int(r.get('馬番', 0)),
                        'name': str(r.get('馬名', '')),
                        'score': float(r.get('スコア', 0)),
                    })
        except Exception:
            # fallback to top1-3 from row
            for k in ('top1', 'top2', 'top3'):
                num = row.get(f'{k}_num', 0)
                name = row.get(f'{k}_name', '')
                if num:
                    ranking_top5.append({'rank': int(k[-1]), 'umaban': int(num), 'name': str(name)})

        formation_planned = row.get('trio_bets', '')

        # morning_odds (sorted_df から取得試行)
        morning_odds = odds_dict or {}
        if not morning_odds and sorted_df is not None:
            try:
                for _, r in sorted_df.iterrows():
                    odds_val = r.get('odds', r.get('オッズ', 0))
                    uma = int(r.get('馬番', 0))
                    if uma and odds_val:
                        morning_odds[uma] = float(odds_val)
            except Exception:
                pass

        _v2_log_phase1(
            race_id=race_id,
            race_meta=race_meta,
            ranking_top5=ranking_top5,
            formation_planned=formation_planned,
            morning_odds=morning_odds,
            date_str=date_str,
        )
    except Exception as _e:
        print(f"[race_notify_log_v2 phase1 wrapper fail] {_e}", file=sys.stderr)


# ===== シグナルハンドラ（Ctrl+C / CLOSE 時の緊急停止） =====

_SHUTDOWN_STATE = {'requested': False, 'date': None}


def _install_signal_handlers(date_str):
    """SIGINT / SIGBREAK を捕捉して graceful shutdown する。"""
    _SHUTDOWN_STATE['date'] = date_str

    def _graceful(signum, frame):
        _SHUTDOWN_STATE['requested'] = True
        name = {signal.SIGINT: 'SIGINT'}.get(signum, f'signal={signum}')
        if hasattr(signal, 'SIGBREAK') and signum == signal.SIGBREAK:
            name = 'SIGBREAK'
        print(f"\n[SHUTDOWN] {name} 受信。既書込CSVを温存して終了します。")
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _graceful)
    except (ValueError, OSError):
        pass
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, _graceful)
        except (ValueError, OSError):
            pass


# ===== リトライユーティリティ =====

MAX_RETRIES = 3
RETRY_DELAY = 5


def retry_call(fn, label, *args, **kwargs):
    """指定関数を最大MAX_RETRIES回リトライする。
    成功時は戻り値を返し、全滅時は最終例外を投げる。
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                print(f"    [RETRY {attempt}/{MAX_RETRIES}] {label}: {e} → {RETRY_DELAY}s待機")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [FAIL] {label}: {MAX_RETRIES}回全失敗: {e}")
    if last_exc:
        raise last_exc


# ===== Cookie検証・自動更新 =====

def ensure_cookie_valid():
    """Cookie有効性をチェック、期限切れなら自動更新を試みる。
    Returns: (ok: bool, message: str)
    """
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
        from refresh_cookie import auto_refresh_if_expired
        refreshed, msg = auto_refresh_if_expired(headless=True)
        if refreshed:
            print(f"[COOKIE] {msg}")
            return True, msg
        if 'Cookie有効' in msg:
            return True, msg
        print(f"[COOKIE WARN] {msg}")
        return False, msg
    except Exception as e:
        print(f"[COOKIE ERR] 検証失敗: {e}")
        return False, str(e)


# ===== レース一覧取得 =====

def fetch_race_list(date_str):
    """netkeibaからその日のレース一覧を取得（リトライ付き）
    date_str: YYYYMMDD形式

    Returns: list of dict {'race_id': str, 'course': str, 'race_num': int}
    """
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"

    def _fetch():
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        return BeautifulSoup(resp.text, "html.parser")

    try:
        soup = retry_call(_fetch, f"race_list {date_str}")
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

def run_daily_predict(date_str, dry_run=False, resume=False):
    """指定日のJRA全レースを予測。

    Args:
        dry_run: CSV保存・Discord通知をスキップ
        resume: 既存CSVに記録済みのrace_idはスキップ（途中再開モード）
    """
    print(f"=" * 60)
    print(f"KEIBA AI 日次予測 - {date_str} (resume={resume})")
    print(f"=" * 60)

    # resume時: 既存CSVから済みrace_id収集
    done_race_ids = set()
    if resume and not dry_run:
        done_race_ids = _load_existing_race_ids(date_str)
        if done_race_ids:
            print(f"[RESUME] 既存CSVから {len(done_race_ids)} レース処理済を検出、スキップします")

    # Cookie検証＋自動更新
    print(f"\n[STEP 0] Cookie有効性チェック...")
    ensure_cookie_valid()

    # モデルロード
    model_data = load_models()
    if model_data['model'] is None:
        print("[ERROR] モデルが見つかりません。終了します。")
        return

    # レース一覧取得（開催日クロスチェック + リトライ + フェイルラウド警告）
    # ★ 2026-05-31 障害修正: netkeiba 0件を即「非開催日」と誤判定するフェイルサイレントを解消。
    #   JRDB(出走表)が当日にあれば「開催日」と判定し、netkeiba 0件=取得失敗としてリトライ＋警告。★
    print(f"\n[STEP 1] レース一覧取得中...")
    from race_day_check import fetch_race_list_robust

    def _notify_fetch_failed(message):
        from notify import send_discord
        send_discord(
            f"🚨 開催日なのに netkeiba一覧0件 ({date_str})",
            message, color="red", channel="updates",
        )

    rd = fetch_race_list_robust(
        lambda: fetch_race_list(date_str),
        date_str, BASE_DIR,
        log=print, notify_fn=_notify_fetch_failed,
    )
    races = rd['races']
    if not races:
        if rd['fetch_failed']:
            print(f"[ERROR] {date_str} は開催日(JRDB {rd['n_races_jrdb']}R)だが "
                  f"netkeiba一覧取得に失敗。予測未実行で終了（Discord警告済）。")
        else:
            print(f"[INFO] {date_str} のレースが見つかりません（非開催日）")
        return

    print(f"  -> {len(races)}レース検出")
    for r in races:
        print(f"     {r['course']} {r['race_num']}R (race_id={r['race_id']})")

    # 各レースを予測
    results = []
    failed_races = []  # 予測できなかったレース
    skipped_jump = []  # 障害除外レース
    jra_weather_cache = {}

    for idx, race in enumerate(races):
        race_id = race['race_id']
        if _SHUTDOWN_STATE['requested']:
            print(f"[SHUTDOWN] ループ中断 ({idx}/{len(races)} 時点)")
            break
        if resume and str(race_id) in done_race_ids:
            print(f"\n[STEP 2-{idx+1}/{len(races)}] {race['course']} {race['race_num']}R (ID={race_id}) -> [SKIP] resume済")
            continue
        print(f"\n[STEP 2-{idx+1}/{len(races)}] {race['course']} {race['race_num']}R (ID={race_id})")

        try:
            # 出馬表取得（リトライ付き）
            race_name, horses, horse_ids, race_info = retry_call(
                parse_shutuba, f"parse_shutuba {race_id}", race_id
            )
            if not horses:
                print(f"  [WARN] 馬データなし、スキップ")
                failed_races.append({'race_id': race_id, 'course': race['course'],
                                     'race_num': race['race_num'], 'reason': '馬データなし'})
                continue

            # 障害レース自動除外
            if race_info.get('surface') == '障':
                print(f"  [SKIP] 障害レース（モデル非対応）")
                skipped_jump.append(race_id)
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

            # paper shadow (V22 candidate comparison)
            if _PAPER_SHADOW_AVAILABLE:
                try:
                    v15_preds = df[['馬番', 'スコア']].rename(
                        columns={'馬番': 'horse_num', 'スコア': 'score'}
                    ).to_dict('records')
                    shadow_result = run_paper_shadow_comparison(race_id, df, v15_preds, race_info=race_info)
                    log_paper_shadow(shadow_result)
                except Exception:
                    pass

            # 条件分類
            cond_key, cond_profile = classify_race_condition(race_info, num_horses)

            # ===== 買い/見送り判定 (戦略⑦、 2026-05-29) =====
            # ★ 従来の投票判断ロジック不変。 investment も不変 (加算方式) ★
            # should_bet=True → 🟢買い (投票対象)、 False → ⚪見送り (記録のみ・賭けない)。
            # 結果照合の3区分 (実投票/見送りシャドー/全部買い) 集計に使用。
            _should_bet, _skip_reason = True, None
            try:
                from strategy_filters import evaluate_bet_decision as _eval_bet
                try:
                    from tools.strategy_rollback import get_effective_flags as _get_eff
                    _c4_eff, _ = _get_eff()
                except Exception:
                    _c4_eff = True
                _should_bet, _skip_reason = _eval_bet(
                    race_name, course_name, race_info.get('distance', 0), cond_key,
                    c4_effective=_c4_eff)
            except Exception as _bd_err:
                print(f"  [WARN] 買い/見送り判定 skip (買い扱い): {_bd_err}")
            print(f"  投票判定: {'🟢買い' if _should_bet else '⚪見送り (' + str(_skip_reason) + ')'}")
            # ===== 買い/見送り判定 ここまで =====

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
                'should_bet': bool(_should_bet),
                'skip_reason': _skip_reason or '',
            }
            results.append(row)

            # 逐次CSV書き込み（途中クラッシュ耐性）
            if not dry_run:
                try:
                    _append_prediction_to_csv(row, date_str)
                except Exception as _e:
                    print(f"  [WARN] 逐次CSV書込失敗: {_e}")

            # race_notify_log v2 phase 1 (★ 既存 logic 完全不変、 log 出力のみ ★)
            try:
                _v2_log_phase1_safe(row, race_info, date_str, odds_dict=None,
                                     sorted_df=sorted_df)
            except Exception:
                pass

            # 全馬スコア保存 (V15 全馬 + V16 全馬)
            if _ALLSCORES_AVAILABLE:
                try:
                    _v16_data = _load_v16_model_daily()
                    _v16_sc = _get_v16_scores_daily(df, _v16_data) if _v16_data else {}
                    _v15_sc = {str(int(r['馬番'])): float(r['スコア'])
                               for _, r in df.iterrows()
                               if pd.notna(r.get('馬番')) and pd.notna(r.get('スコア'))}
                    _horses_list = [{'馬番': int(r['馬番']), '馬名': str(r.get('馬名',''))}
                                    for _, r in df.iterrows() if pd.notna(r.get('馬番'))]
                    save_race_allscores(
                        date_str=date_str,
                        race_id=race_id,
                        race_name=race_name,
                        course=course_name,
                        race_num=race_num_int,
                        distance=race_info.get('distance', 0),
                        surface=race_info.get('surface', ''),
                        condition=race_info.get('condition', ''),
                        cond_key=cond_key,
                        num_horses=num_horses,
                        horses=_horses_list,
                        v15_scores=_v15_sc,
                        v16_scores=_v16_sc,
                        should_bet=bool(_should_bet),
                        skip_reason=_skip_reason or '',
                    )
                except Exception as _e:
                    print(f"  [allscores] save failed: {_e}")

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
            failed_races.append({'race_id': race_id, 'course': race['course'],
                                 'race_num': race['race_num'], 'reason': str(e)[:120]})
            continue

    # CSV最終化（逐次appendで既に書き込み済。ここでは全体dedup + カラム順正規化）
    if results:
        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"[DRY-RUN] 予測完了: {len(results)}レース (CSV保存スキップ)")
            print(f"総投資額: {sum(r['investment'] for r in results):,}円")
            print(f"{'=' * 60}")
        else:
            out_path = _csv_path_for(date_str)
            try:
                df_all = pd.read_csv(out_path, encoding='utf-8-sig', dtype={'race_id': str})
                df_all = df_all.drop_duplicates(subset='race_id', keep='last')
                df_all = df_all.reindex(columns=_CSV_COLUMNS)
                df_all.to_csv(out_path, index=False, encoding='utf-8-sig')
            except Exception as _e:
                print(f"[WARN] CSV最終化失敗（逐次appendは残存）: {_e}")
            # サマリ用: 最終CSVから総投資額再集計
            try:
                df_final = pd.read_csv(out_path, encoding='utf-8-sig')
                n_final = len(df_final)
                inv_final = int(df_final['investment'].sum())
            except Exception:
                n_final = len(results)
                inv_final = sum(r['investment'] for r in results)
            print(f"\n{'=' * 60}")
            print(f"予測完了: 今回{len(results)}レース / CSV累計{n_final}レース")
            print(f"保存先: {out_path}")
            print(f"総投資額: {inv_final:,}円")
            print(f"{'=' * 60}")
    else:
        print(f"\n[INFO] 予測結果なし")

    # 全レース取得完了の検証（障害レース除外後、取得漏れをチェック）
    total = len(races)
    jumps = len(skipped_jump)
    ok = len(results)
    miss = len(failed_races)
    expected = total - jumps  # 障害以外の予測対象数

    print(f"\n[STEP 3] 取得完了検証")
    print(f"  全レース: {total} / 障害除外: {jumps} / 予測対象: {expected}")
    print(f"  成功: {ok} / 失敗: {miss}")

    if miss > 0 or ok < expected:
        # 取り漏れあり — Discord警告通知
        lines = [f"**{date_str}** 取得漏れ {miss}件 / {expected}レース中"]
        for f in failed_races[:10]:
            lines.append(f"- {f['course']} {f['race_num']}R ({f['race_id']}): {f['reason']}")
        if len(failed_races) > 10:
            lines.append(f"...他 {len(failed_races)-10} 件")
        msg = "\n".join(lines)
        print(f"\n[WARNING] {msg}")
        if not dry_run:
            try:
                from notify import send_discord
                send_discord(f"⚠️ 予測取り漏れ ({miss}件)", msg, color="red", channel="updates")
            except Exception as e:
                print(f"[WARN] Discord通知失敗: {e}")
    else:
        print(f"  [OK] 全予測対象レースの取得完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEIBA AI 日次予測")
    parser.add_argument("--date", type=str, default=None,
                        help="予測日 YYYYMMDD (デフォルト: 今日)")
    parser.add_argument("--dry-run", action="store_true",
                        help="予測は実行するがCSV保存・Discord通知をスキップ")
    parser.add_argument("--resume", action="store_true",
                        help="既存CSVから未予測レースだけ再実行（途中再開モード）")
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

    # シグナルハンドラ設定（SIGINT/SIGBREAK でCSV温存終了）
    _install_signal_handlers(date_str)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_predict.py 開始 (dry_run={args.dry_run}, resume={args.resume})")
    run_daily_predict(date_str, dry_run=args.dry_run, resume=args.resume)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_predict.py 終了")

    if args.dry_run:
        sys.exit(0)

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

    # 整形済み買い目通知（Discord #買い目）
    try:
        from notify_bets_formatted import notify_formatted
        sent = notify_formatted(date_str, mode='morning', channel='bets')
        print(f"[daily_predict] 整形済み買い目通知送信: {sent} messages")
    except Exception as e:
        print(f"[WARN] 整形済み買い目通知失敗: {e}")

    # 全馬スコア HTML 生成 → Discord 添付 (bets チャンネル)
    try:
        from notify import send_discord_file, send_discord
        html_path = build_allscores_html(date_str) if _ALLSCORES_AVAILABLE else None
        if html_path and os.path.exists(html_path):
            csv_path = os.path.join(BASE_DIR, "data", "daily_predictions", f"{date_str}.csv")
            n_races = 0
            if os.path.exists(csv_path):
                import pandas as _pd2
                n_races = len(_pd2.read_csv(csv_path, encoding='utf-8-sig'))
            summary = (
                f"**{date_str}** {n_races}R 全馬スコア (朝時点)\n"
                f"V15=本番スコア順 / V16=参考(投票しない)\n"
                f"添付 HTML で全レース全馬を確認。結果反映後に再送します。"
            )
            ok = send_discord_file(
                f"全馬スコア {date_str} ({n_races}R)",
                summary,
                filepath=html_path,
                color="blue",
                channel="bets",
            )
            print(f"[daily_predict] 全馬スコア HTML 送信: {'OK' if ok else 'FAILED'}")
        else:
            print("[daily_predict] allscores HTML 生成スキップ (データなし)")
    except Exception as e:
        print(f"[WARN] 全馬スコア HTML 送信失敗: {e}")

    # paper shadow stats 通知 (updates チャンネル)
    try:
        from paper_shadow_v15_full import get_paper_shadow_stats
        from notify import send_discord
        stats = get_paper_shadow_stats(days=1)
        if stats:
            lines = [f"**{date_str} paper shadow**"]
            for mkey, s in sorted(stats.items()):
                n = s['n_total']
                agree = s['n_agree']
                rate = agree / n * 100 if n > 0 else 0.0
                lines.append(f"  {mkey}: {agree}/{n}R 一致 ({rate:.0f}%)")
            send_discord("paper shadow 本日集計", "\n".join(lines), color="blue", channel="updates")
            print(f"[daily_predict] paper shadow stats 通知送信")
    except Exception as e:
        print(f"[WARN] paper shadow stats 通知失敗: {e}")
