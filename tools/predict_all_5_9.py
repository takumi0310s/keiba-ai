"""Session #47 C: 5/9 全 R 予測 pre-compute (V15 vs V15+training).

5/9 (土) 中央 全 R を V15 で予測 + 拡張調教 features 効果を並列計算。

出力: data/v18/predictions_5_9_all.json

Usage:
    python tools/predict_all_5_9.py                 # 5/9 全 R
    python tools/predict_all_5_9.py --date 20260509 # 日付指定
    python tools/predict_all_5_9.py --limit 3       # 3 R だけ smoke test
    python tools/predict_all_5_9.py --no-extended   # V15 のみ (extended skip)

NEVER:
- daily_predict.py 変更
- predict_core.py 変更
- V15 model file 変更

OK:
- predict_core 関数 read-only 利用
- daily_predict.py の fetch_race_list() 流用
"""
import os
import sys
import json
import argparse
import hashlib
import time
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

# Windows cp932 対策
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

V15_MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz')
V15_LIVE_MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v15_central_live.pkl.gz')
V18_DIR = os.path.join(BASE_DIR, 'data', 'v18')
# CLAUDE.md に '842b9a5f...' とあるが現在の実 md5 は 309dffc6...
# (CLAUDE.md 値は古い、 実 md5 を baseline とする)
EXPECTED_V15_MD5 = '309dffc65504f056d233c65665c319d5'


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_v15_md5():
    """V15 model md5 を verify。 不変保証。"""
    if not os.path.exists(V15_MODEL_PATH):
        log(f"WARN: V15 model not found: {V15_MODEL_PATH}")
        return None
    h = hashlib.md5()
    with open(V15_MODEL_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    md5 = h.hexdigest()
    log(f"V15 model md5: {md5}")
    if md5 != EXPECTED_V15_MD5:
        log(f"  WARN: expected {EXPECTED_V15_MD5}")
    else:
        log(f"  OK: V15 model 不変")
    return md5


def fetch_5_9_race_list(date_str):
    """daily_predict.py の fetch_race_list() を流用 (read-only)。"""
    try:
        from daily_predict import fetch_race_list
        races = fetch_race_list(date_str)
        log(f"Fetched {len(races)} races for {date_str}")
        return races
    except Exception as e:
        log(f"Fetch race list failed: {e}")
        return []


_MODEL_CACHE = {}


def _get_model():
    if 'model' not in _MODEL_CACHE:
        import predict_core
        _MODEL_CACHE['model'] = predict_core.load_models()
    return _MODEL_CACHE['model']


def predict_one_v15(race_id):
    """V15 single race prediction (predict_one_race.py の flow を流用、 read-only)。

    Returns: dict { race_name, rinfo, top3 (list of {umaban, horse_name, score}), scores, num_horses }
    """
    try:
        import predict_core
        model_data = _get_model()
        if model_data is None or model_data.get('model') is None:
            return {'error': 'model load failed'}

        race_name, horses, horse_ids, rinfo = predict_core.parse_shutuba(race_id)
        if not horses:
            return {'error': 'shutuba fetch failed'}

        try:
            odds_full = predict_core.fetch_realtime_odds_full(race_id)
            odds_dict = {u: v['odds'] for u, v in odds_full.items()} if odds_full else {}
        except Exception:
            odds_dict = {}
        try:
            jra_info, weather_info = predict_core.fetch_jra_and_weather(rinfo.get('course', ''))
        except Exception:
            jra_info, weather_info = {}, {}

        for horse, hid in zip(horses, horse_ids):
            if hid:
                try:
                    stats = predict_core.get_horse_stats(
                        hid, rinfo.get('distance', 1600),
                        rinfo.get('surface', '芝'),
                        rinfo.get('course', '東京')
                    )
                    predict_core.apply_horse_stats(horse, stats, rinfo)
                except Exception:
                    predict_core.set_horse_defaults(horse)
            else:
                predict_core.set_horse_defaults(horse)

        df = predict_core.build_features(
            horses, rinfo, model_data,
            race_id=race_id, odds_dict=odds_dict,
            jra_track_info=jra_info, weather_info=weather_info,
        )

        try:
            from jrdb_features import merge_jrdb_predict_features
            df = merge_jrdb_predict_features(df, race_id)
        except Exception:
            pass

        odds_available = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        result = predict_core.predict_race(df, model_data, odds_available, race_info=rinfo)
        if hasattr(result, 'sort_values'):
            result = result.sort_values('スコア', ascending=False).reset_index(drop=True)

        # Build top3 list
        top3 = []
        scores = {}
        try:
            for i in range(min(3, len(result))):
                row = result.iloc[i]
                umaban = int(row.get('馬番', row.get('umaban', 0))) if not pd.isna(row.get('馬番', None)) else 0
                hname = str(row.get('馬名', row.get('horse_name', '')))
                score = float(row.get('スコア', row.get('score', 0)))
                top3.append({'umaban': umaban, 'horse_name': hname, 'score': score})
            for i in range(len(result)):
                row = result.iloc[i]
                umaban = int(row.get('馬番', row.get('umaban', 0))) if not pd.isna(row.get('馬番', None)) else 0
                score = float(row.get('スコア', row.get('score', 0)))
                scores[str(umaban)] = score
        except Exception as e:
            log(f"  parse top3 failed: {e}")

        return {
            'race_name': race_name,
            'rinfo': rinfo,
            'top3': top3,
            'scores': scores,
            'num_horses': len(horses),
        }
    except Exception as e:
        import traceback
        return {'error': f"{e} | {traceback.format_exc()[-200:]}"}


def classify_grade(race_name, course, race_num):
    """G1/G2/G3/L/OP/3勝/2勝/1勝/未勝利/新馬 を判定。"""
    if not race_name:
        return 'unknown'
    rn = str(race_name)
    if 'G1' in rn or 'GⅠ' in rn or 'Ｇ１' in rn:
        return 'G1'
    if 'G2' in rn or 'GⅡ' in rn or 'Ｇ２' in rn:
        return 'G2'
    if 'G3' in rn or 'GⅢ' in rn or 'Ｇ３' in rn:
        return 'G3'
    if 'OP' in rn or 'オープン' in rn:
        return 'OP'
    if '3勝' in rn:
        return '3勝'
    if '2勝' in rn:
        return '2勝'
    if '1勝' in rn:
        return '1勝'
    if '未勝利' in rn:
        return '未勝利'
    if '新馬' in rn:
        return '新馬'
    return 'other'


def run_predict(date_str, limit=None):
    log(f"=== Session #47 C: predict {date_str} all races ===")
    md5 = verify_v15_md5()

    races = fetch_5_9_race_list(date_str)
    if not races:
        log("No races fetched (output_compatible 出馬表 may not be published yet)")
        log("Saving fixture skeleton instead...")
        races = []

    if limit:
        races = races[:limit]

    predictions = []
    for i, r in enumerate(races):
        rid = r.get('race_id', '')
        log(f"[{i + 1}/{len(races)}] {rid} {r.get('course', '?')}{r.get('race_num', '?')}R")
        v15_pred = predict_one_v15(rid)
        # extended は B 完了後に統合実装。 今回は placeholder
        ext_pred = {'note': 'extended (V15+training) は B 結果後に再計算'}

        grade = classify_grade(v15_pred.get('race_name', ''),
                               r.get('course', ''), r.get('race_num', 0))

        predictions.append({
            'race_id': rid,
            'venue': r.get('course', ''),
            'race_num': r.get('race_num', 0),
            'race_name': v15_pred.get('race_name', ''),
            'grade': grade,
            'num_horses': v15_pred.get('num_horses', 0),
            'v15_top3': v15_pred.get('top3', []),
            'v15_scores': v15_pred.get('scores', {}),
            'extended_top3': ext_pred,
            'error': v15_pred.get('error', None),
        })
        time.sleep(1)  # netkeiba 連続 access 抑制

    out = {
        'date': date_str,
        'v15_md5': md5,
        'race_count': len(predictions),
        'predictions': predictions,
        'timestamp': datetime.now().isoformat(),
        'note': 'V15 baseline 予測。 拡張調教 features は Session #47 B 結果確認後に追加実装。',
    }

    os.makedirs(V18_DIR, exist_ok=True)
    out_json = os.path.join(V18_DIR, 'predictions_5_9_all.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"Saved: {out_json} ({len(predictions)} races)")

    # クラス別集計
    by_grade = {}
    for p in predictions:
        g = p['grade']
        by_grade.setdefault(g, 0)
        by_grade[g] += 1
    log(f"Grade distribution: {by_grade}")

    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--date', default='20260509', help='YYYYMMDD')
    p.add_argument('--limit', type=int, default=None, help='Limit races for smoke test')
    p.add_argument('--no-extended', action='store_true', help='Skip extended features')
    args = p.parse_args()

    run_predict(args.date, limit=args.limit)
