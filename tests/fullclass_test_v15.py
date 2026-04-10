#!/usr/bin/env python
"""v15 全クラス×全条件 内部ダンプテスト

各クラスの実レースでbuild_features()の内部変数を完全ダンプし、
特徴量値・マージ結果・v15新特徴量・★パターンを検証する。
"""
import sys, os, json, time, traceback
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
os.chdir(BASE_DIR)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from predict_core import (
    load_models, parse_shutuba, build_features, predict_race,
    classify_race_condition, generate_trio_bets, generate_umaren_bets,
    fetch_realtime_odds_full, fetch_jra_and_weather,
    get_horse_stats, apply_horse_stats, set_horse_defaults,
    match_profitable_patterns, CONDITION_PROFILES,
    load_feature_lookups, compute_odds_change_features,
)

# Load test races
with open(os.path.join(BASE_DIR, 'data', '_test_races_by_class.json'), 'r', encoding='utf-8') as f:
    TEST_RACES = json.load(f)

RESULTS = {
    'test_date': time.strftime('%Y-%m-%d %H:%M'),
    'model_version': '',
    'classes': {},
}


def test_one_race(cls_name, race_info_dict, model_data):
    """1レースの完全内部ダンプテスト"""
    rid = race_info_dict['race_id']
    result = {
        'race_id': rid,
        'race_name': race_info_dict.get('race_name', ''),
        'status': 'RUNNING',
        'errors': [],
        'warnings': [],
    }

    try:
        # === 1. 出馬表取得 ===
        race_name, horses, horse_ids, rinfo = parse_shutuba(rid)
        result['race_name'] = race_name
        result['num_horses'] = len(horses)
        result['distance'] = rinfo.get('distance', 0)
        result['surface'] = rinfo.get('surface', '')
        result['condition'] = rinfo.get('condition', '')

        if rinfo.get('surface') == '障':
            result['status'] = 'SKIP_OBSTACLE'
            return result

        # === 2. 馬情報取得 ===
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                try:
                    stats = get_horse_stats(hid, rinfo['distance'], rinfo['surface'], rinfo.get('course', ''))
                    apply_horse_stats(horse, stats, rinfo)
                except Exception:
                    set_horse_defaults(horse)
            else:
                set_horse_defaults(horse)

        # === 3. オッズ取得 ===
        odds_full = {}
        try:
            odds_full = fetch_realtime_odds_full(rid)
        except Exception:
            pass
        odds_dict = {u: v['odds'] for u, v in odds_full.items()} if odds_full else {}
        result['odds_available'] = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        result['odds_sample'] = {str(k): v for k, v in list(odds_dict.items())[:3]}

        # === 4. JRA馬場・天候 ===
        jra_info, weather_info = {}, {}
        try:
            jra_info, weather_info = fetch_jra_and_weather(rinfo.get('course', ''))
        except Exception:
            pass
        result['jra_track'] = {k: v for k, v in jra_info.items() if v} if jra_info else {}
        result['weather'] = {k: v for k, v in weather_info.items() if v} if weather_info else {}

        # === 5. build_features (CORE) ===
        df = build_features(horses, rinfo, model_data, race_id=rid,
                            odds_dict=odds_dict, jra_track_info=jra_info, weather_info=weather_info)

        features = model_data.get('features', [])
        result['df_shape'] = list(df.shape)
        result['model_features_count'] = len(features)

        # === 6. 全特徴量値ダンプ ===
        feat_dump = {}
        zero_feats = []
        missing_feats = []

        for feat in features:
            if feat in df.columns:
                vals = pd.to_numeric(df[feat], errors='coerce').fillna(0).values
                is_all_zero = (np.abs(vals) < 1e-9).all()
                feat_dump[feat] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'min': round(float(np.min(vals)), 4),
                    'max': round(float(np.max(vals)), 4),
                    'zero_count': int((np.abs(vals) < 1e-9).sum()),
                    'all_zero': bool(is_all_zero),
                }
                if is_all_zero:
                    zero_feats.append(feat)
            else:
                missing_feats.append(feat)
                feat_dump[feat] = {'MISSING': True}

        result['features'] = feat_dump
        result['zero_features'] = zero_feats
        result['zero_count'] = len(zero_feats)
        result['missing_features'] = missing_feats

        # === 7. ゼロ特徴量の原因分類 ===
        LEGITIMATE_ZEROS = {
            'season', 'is_nar', 'odds_change_rate', 'pop_rank_change', 'odds_sharp_drop',
            'course_renovated', 'post_renovation_flag', 'gaisha_rank',
            'jockey_horse_rides', 'jockey_horse_wr', 'jockey_horse_top3r',
            'jockey_change', 'jockey_change_to_top',
            'paci_sogo_mark', 'paci_idm_mark', 'paci_jockey_mark', 'paci_train_mark',
            'shinba_comment_score', 'dist_change', 'carry_diff',
        }
        suspicious_zeros = [f for f in zero_feats if f not in LEGITIMATE_ZEROS]
        result['suspicious_zeros'] = suspicious_zeros
        if suspicious_zeros:
            result['warnings'].append(f'怪しいゼロ特徴量: {suspicious_zeros}')

        # === 8. JRDB特徴量検証 ===
        jrdb_feats = [f for f in features if f.startswith('jrdb_')]
        jrdb_nonzero = [f for f in jrdb_feats if f not in zero_feats]
        result['jrdb_total'] = len(jrdb_feats)
        result['jrdb_nonzero'] = len(jrdb_nonzero)
        result['jrdb_coverage'] = f'{len(jrdb_nonzero)}/{len(jrdb_feats)}'

        # === 9. 調教データ検証 ===
        train_feats = ['training_time_filled', 'has_training', 'wood_best_4f_filled',
                       'has_wood_training', 'time_1f_last_filled', 'training_intensity_enc']
        train_info = {}
        for tf in train_feats:
            if tf in df.columns:
                v = df[tf].values
                train_info[tf] = {
                    'mean': round(float(np.mean(v)), 2),
                    'nonzero': int((np.abs(v) > 1e-9).sum()),
                    'total': len(v),
                }
        result['training_data'] = train_info

        # === 10. Speed Index検証 ===
        si_feats = ['index_max_filled', 'index_avg5_filled', 'index_run1_filled']
        si_info = {}
        for sf in si_feats:
            if sf in df.columns:
                v = df[sf].values
                si_info[sf] = {
                    'mean': round(float(np.mean(v)), 1),
                    'all_default': bool((v == 1045.0).all() or (v == 1040.0).all() or (v == 1035.0).all()),
                }
        result['speed_index'] = si_info

        # === 11. v15新特徴量検証 ===
        v15_feats = ['transport_distance_km', 'is_long_transport', 'course_renovated',
                     'post_renovation_flag', 'jockey_change', 'jockey_horse_rides']
        v15_info = {}
        for vf in v15_feats:
            if vf in df.columns:
                v = df[vf].values
                v15_info[vf] = {
                    'mean': round(float(np.mean(v)), 2),
                    'min': round(float(np.min(v)), 1),
                    'max': round(float(np.max(v)), 1),
                }
        result['v15_features'] = v15_info

        # 輸送距離の妥当性チェック
        if 'transport_distance_km' in df.columns:
            td = df['transport_distance_km'].values
            loc = df['location_enc'].values if 'location_enc' in df.columns else [0]*len(df)
            result['transport_check'] = {
                'miho_horses': int((np.array(loc) == 0).sum()),
                'ritto_horses': int((np.array(loc) == 1).sum()),
                'avg_km': round(float(np.mean(td)), 1),
                'range': f'{float(np.min(td)):.0f}-{float(np.max(td)):.0f}km',
            }

        # === 12. 予測実行 ===
        odds_available = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        df = predict_race(df, model_data, odds_available, race_info=rinfo)
        df_sorted = df.sort_values('スコア', ascending=False).reset_index(drop=True)

        scores = df_sorted['スコア'].values
        result['score_range'] = [round(float(scores.min()), 4), round(float(scores.max()), 4)]
        result['score_in_01'] = bool(scores.min() >= 0 and scores.max() <= 1)

        top3 = []
        for i in range(min(3, len(df_sorted))):
            row = df_sorted.iloc[i]
            top3.append({
                'rank': i+1,
                'horse_num': int(row['馬番']),
                'horse_name': str(row.get('馬名', '?')),
                'score': round(float(row['スコア']), 4),
            })
        result['top3'] = top3

        # === 13. 条件判定 ===
        cond_key, cond_profile = classify_race_condition(rinfo, len(df_sorted))
        result['condition_key'] = cond_key
        result['condition_label'] = cond_profile['label']
        result['bet_type'] = cond_profile['bet_type']
        result['recommended'] = cond_profile['recommended']

        # === 14. 買い目生成 ===
        if cond_profile['bet_type'] == 'umaren':
            bets = generate_umaren_bets(df_sorted)
        else:
            bets = generate_trio_bets(df_sorted)
        result['bets'] = [list(b) for b in bets]
        result['bet_count'] = len(bets)

        # === 15. ★パターンマッチ ===
        top1_odds = min((v for v in odds_dict.values() if v > 0), default=0) if odds_dict else 0
        pp_stars, pp_matched = match_profitable_patterns(
            cond_key, rinfo.get('course', ''), rinfo.get('surface', '芝'),
            rinfo.get('distance', 1600), top1_odds)
        result['pp_stars'] = pp_stars
        result['pp_matched_count'] = len(pp_matched)
        if pp_matched:
            result['pp_best'] = {
                'roi': pp_matched[0].get('roi', 0),
                'hit_rate': pp_matched[0].get('hit_rate', 0),
                'n': pp_matched[0].get('n', 0),
                'venue': pp_matched[0].get('venue', ''),
                'odds_band': pp_matched[0].get('odds_band', ''),
            }

        result['status'] = 'PASS' if not suspicious_zeros else 'WARN'

    except Exception as e:
        result['status'] = 'FAIL'
        result['errors'].append(f'{type(e).__name__}: {str(e)}')
        traceback.print_exc()

    return result


def main():
    print('=' * 70)
    print('  v15 全クラス×全条件 内部ダンプテスト')
    print('=' * 70)

    # モデルロード
    model_data = load_models()
    version = model_data.get('version', '?')
    RESULTS['model_version'] = version
    print(f'\nModel: {version}')
    print(f'Features: {len(model_data.get("features", []))}')
    print(f'LGB: {model_data["model"] is not None}')
    print(f'XGB: {model_data.get("xgb_model") is not None}')

    total_pass = 0
    total_warn = 0
    total_fail = 0

    for cls_name, race_info in TEST_RACES.items():
        print(f'\n{"="*60}')
        print(f'  [{cls_name}] {race_info["race_id"]} {race_info.get("race_name","")}')
        print(f'{"="*60}')

        result = test_one_race(cls_name, race_info, model_data)
        RESULTS['classes'][cls_name] = result

        status = result['status']
        if status == 'PASS':
            total_pass += 1
            icon = 'PASS'
        elif status == 'WARN':
            total_warn += 1
            icon = 'WARN'
        elif status.startswith('SKIP'):
            icon = 'SKIP'
        else:
            total_fail += 1
            icon = 'FAIL'

        print(f'\n  [{icon}] {cls_name}')
        print(f'    Horses: {result.get("num_horses", "?")}')
        print(f'    Score: {result.get("score_range", "?")}')
        print(f'    Condition: {result.get("condition_key", "?")} ({result.get("condition_label", "")})')
        print(f'    Bet: {result.get("bet_type", "?")} {result.get("bet_count", 0)}点')
        print(f'    Zero feats: {result.get("zero_count", "?")} (suspicious: {len(result.get("suspicious_zeros", []))})')
        print(f'    JRDB: {result.get("jrdb_coverage", "?")}')
        print(f'    PP Stars: {"★" * result.get("pp_stars", 0)} ({result.get("pp_stars", 0)})')
        print(f'    Transport: {result.get("transport_check", {}).get("range", "?")}')

        if result.get('suspicious_zeros'):
            print(f'    ⚠ Suspicious zeros: {result["suspicious_zeros"]}')
        if result.get('errors'):
            print(f'    ✗ Errors: {result["errors"]}')
        if result.get('top3'):
            for t in result['top3']:
                print(f'    TOP{t["rank"]}: {t["horse_num"]:2d} {t["horse_name"][:8]} score={t["score"]:.4f}')

        time.sleep(2)  # Avoid rate limiting

    # Summary
    RESULTS['summary'] = {
        'total': len(TEST_RACES),
        'pass': total_pass,
        'warn': total_warn,
        'fail': total_fail,
        'verdict': 'APPROVED' if total_fail == 0 else 'REJECTED',
    }

    print(f'\n{"="*70}')
    print(f'  SUMMARY: {total_pass} PASS / {total_warn} WARN / {total_fail} FAIL')
    print(f'  VERDICT: {RESULTS["summary"]["verdict"]}')
    print(f'{"="*70}')

    # Save
    out_path = os.path.join(BASE_DIR, 'data', 'fullclass_test_v15.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
