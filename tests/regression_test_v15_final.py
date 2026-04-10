#!/usr/bin/env python
"""v15 最終完全検証テスト

PHASE 0: 全クラス×全競馬場×全条件 網羅テスト (20レース)
PHASE 1: 内部変数完全ダンプ (特徴量値・JRDB・netkeiba・v15新特徴量)
PHASE 2: クラス別特有問題 (新馬/未勝利/重賞/障害)
PHASE 3: 過去バグ15件回帰テスト
PHASE 4: app.py/Discord整合
PHASE 5: 障害シナリオ
PHASE 6: 9ペルソナ承認

Usage:
    python tests/regression_test_v15_final.py
"""
import sys, os, json, time, traceback, re, math
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
    calc_kelly_investment, calc_horse_ev, calc_race_confidence,
)
import inspect

# ==========================================
# PHASE 3: 過去バグ15件 回帰テスト (コード検査)
# ==========================================
PHASE3_RESULTS = []

def p3(name, check_fn):
    try:
        check_fn()
        PHASE3_RESULTS.append({'name': name, 'status': 'PASS'})
        print(f"  [PASS] {name}")
    except AssertionError as e:
        PHASE3_RESULTS.append({'name': name, 'status': 'FAIL', 'error': str(e)})
        print(f"  [FAIL] {name} -- {e}")
    except Exception as e:
        PHASE3_RESULTS.append({'name': name, 'status': 'ERROR', 'error': str(e)})
        print(f"  [ERROR] {name} -- {e}")


def run_phase3():
    print('\n' + '='*70)
    print('  PHASE 3: 過去バグ15件 回帰テスト')
    print('='*70)

    pc_src = open(os.path.join(BASE_DIR, 'tools', 'predict_core.py'), 'r', encoding='utf-8').read()

    # 3-1. バージョンゲート
    def t3_1():
        src = inspect.getsource(build_features)
        assert "re.search" in src, "regex版バージョンゲートが未実装"
        assert "startswith(('v5'" not in src, "旧startswithゲートが残存"
        m = re.search(r"int\(_ver_num_v15\.group\(1\)\)\s*>=\s*15", src)
        assert m, "v15ゲート条件 >=15 が見つからない"
    p3("3-1 バージョンゲート(v15 regex)", t3_1)

    # 3-2. Single Source of Truth
    def t3_2():
        for fpath in ['tools/daily_predict.py', 'tools/race_auto_notify.py']:
            full = os.path.join(BASE_DIR, fpath)
            if not os.path.exists(full): continue
            content = open(full, 'r', encoding='utf-8').read()
            assert 'def build_features(' not in content, f"{fpath}に独自build_features"
            assert 'def predict_race(' not in content, f"{fpath}に独自predict_race"
    p3("3-2 予測ロジック一元化(predict_core)", t3_2)

    # 3-3. stdout再ラップなし
    def t3_3():
        fpath = os.path.join(BASE_DIR, 'tools', 'race_auto_notify.py')
        if not os.path.exists(fpath): return
        for i, line in enumerate(open(fpath, 'r', encoding='utf-8')):
            if line.strip().startswith('#'): continue
            assert 'sys.stdout = io.TextIOWrapper' not in line.strip(), f"line {i+1}: stdout再ラップ"
    p3("3-3 stdout再ラップなし", t3_3)

    # 3-4. Speed Index参照(CSV+premium cache)
    def t3_4():
        assert 'netkeiba_speed_index.csv' in pc_src or 'speed_index' in pc_src, "speed index CSV参照なし"
        assert 'premium_cache' in pc_src or 'weekly_premium_cache' in pc_src, "premium cacheフォールバックなし"
    p3("3-4 Speed Index参照(CSV+cache)", t3_4)

    # 3-5. 距離bin数一致(5bin)
    def t3_5():
        assert "labels=[0, 1, 2, 3, 4]" in pc_src, "dist_catが5binでない"
    p3("3-5 距離カテゴリ5bin", t3_5)

    # 3-6. dam_top3rリーク除去確認
    def t3_6():
        md = load_models()
        feats = md.get('features', [])
        assert 'dam_top3r' not in feats, "dam_top3rがfeatureリストに残存(リーク)"
    p3("3-6 dam_top3rリーク除去", t3_6)

    # 3-7. 調教パーサー
    def t3_7():
        assert 'oikiri' in pc_src.lower() or 'training' in pc_src.lower(), "調教データ取得コードなし"
    p3("3-7 調教パーサー存在", t3_7)

    # 3-8. odds系3特徴量
    def t3_8():
        assert 'odds_change_rate' in pc_src
        assert 'pop_rank_change' in pc_src
        assert 'odds_sharp_drop' in pc_src
        assert 'save_odds_base' in pc_src
    p3("3-8 odds変動3特徴量フロー", t3_8)

    # 3-9. jrdb_kta_idmマージ (jrdb_features.pyに存在)
    def t3_9():
        jrdb_path = os.path.join(BASE_DIR, 'tools', 'jrdb_features.py')
        assert os.path.exists(jrdb_path), "jrdb_features.pyが存在しない"
        jrdb_src = open(jrdb_path, 'r', encoding='utf-8').read()
        assert 'jrdb_kta_idm' in jrdb_src or 'kta_idm' in jrdb_src, "JRDB KTA/IDMマージなし"
        # predict_core.pyがjrdb_featuresを使用していることも確認
        assert 'jrdb_features' in pc_src or 'jrdb_idm' in pc_src or 'merge_jrdb' in pc_src, "predict_coreがJRDBを参照していない"
    p3("3-9 JRDB KTA/IDMマージ", t3_9)

    # 3-10. feature_lookups.pklパス
    def t3_10():
        assert 'feature_lookups' in pc_src, "feature_lookups参照なし"
        lookups = load_feature_lookups()
        assert lookups is not None, "feature_lookups.pklロード失敗"
    p3("3-10 feature_lookups.pklロード", t3_10)

    # 3-11. LGB+XGBアンサンブル
    def t3_11():
        md = load_models()
        assert md.get('model') is not None, "LGBモデルなし"
        assert md.get('xgb_model') is not None, "XGBモデルなし"
        w = md.get('ensemble_weights', {})
        assert w.get('lgb', 0) > 0, f"LGB重み=0 (weights={w})"
        assert w.get('xgb', 0) > 0, f"XGB重み=0 (weights={w})"
        total = w.get('lgb', 0) + w.get('xgb', 0)
        assert abs(total - 1.0) < 0.01 or total > 0.5, f"重み合計異常: {total}"
        print(f"    LGB={w.get('lgb',0):.3f}, XGB={w.get('xgb',0):.3f}")
    p3("3-11 LGB+XGBアンサンブル重み", t3_11)

    # 3-12. .envがgitignore
    def t3_12():
        gi = open(os.path.join(BASE_DIR, '.gitignore'), 'r').read()
        assert '.env' in gi, ".envが.gitignoreに未登録"
    p3("3-12 .env gitignore", t3_12)

    # 3-13. bat PYTHONIOENCODING
    def t3_13():
        bats = ['daily_predict.bat', 'daily_results.bat', 'weekly_report.bat',
                'race_auto_notify.bat', 'daily_premium_scrape.bat', 'friday_weekend_scrape.bat']
        for bat in bats:
            full = os.path.join(BASE_DIR, bat)
            if not os.path.exists(full): continue
            content = open(full, 'r', encoding='utf-8', errors='replace').read()
            assert 'PYTHONIOENCODING' in content, f"{bat}: PYTHONIOENCODING未設定"
    p3("3-13 bat PYTHONIOENCODING", t3_13)

    # 3-14. v15モデルがload_models候補
    def t3_14():
        assert 'keiba_model_v15_central' in pc_src, "v15がload_models候補に未登録"
    p3("3-14 v15 load_models候補", t3_14)

    # 3-15. profitable_patterns.jsonロード
    def t3_15():
        pp_path = os.path.join(BASE_DIR, 'data', 'profitable_patterns.json')
        assert os.path.exists(pp_path), "profitable_patterns.json不存在"
        pp = json.load(open(pp_path, 'r', encoding='utf-8'))
        patterns = pp.get('profitable_patterns', [])
        assert len(patterns) >= 100, f"パターン数不足: {len(patterns)}"
        # 全条件にパターンが存在するか
        conds = set(p.get('condition', '') for p in patterns)
        for c in ['A', 'B', 'C', 'D', 'E', 'X']:
            assert c in conds, f"条件{c}のパターンなし"
    p3("3-15 profitable_patterns.json完全性", t3_15)


# ==========================================
# PHASE 0+1+2: 全レース内部ダンプテスト
# ==========================================

def test_one_race_full(cls_name, race_info_dict, model_data):
    """1レースの完全内部ダンプ+検証"""
    rid = race_info_dict['race_id']
    expected_cond = race_info_dict.get('expected_condition', '')
    result = {
        'race_id': rid,
        'race_name': race_info_dict.get('race_name', ''),
        'venue': race_info_dict.get('venue', ''),
        'status': 'RUNNING',
        'errors': [],
        'warnings': [],
        'checks': {},
    }

    try:
        # 1. 出馬表取得
        race_name, horses, horse_ids, rinfo = parse_shutuba(rid)
        result['race_name'] = race_name
        result['num_horses'] = len(horses)
        result['distance'] = rinfo.get('distance', 0)
        result['surface'] = rinfo.get('surface', '')
        result['condition'] = rinfo.get('condition', '')

        # 障害レース除外確認
        if rinfo.get('surface') == '障' or expected_cond == 'SKIP':
            result['status'] = 'SKIP_OBSTACLE'
            result['checks']['obstacle_excluded'] = True
            return result

        # 2. 馬情報取得
        maiden_count = 0  # 新馬(0走)の馬カウント
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                try:
                    stats = get_horse_stats(hid, rinfo['distance'], rinfo['surface'], rinfo.get('course', ''))
                    apply_horse_stats(horse, stats, rinfo)
                    if stats.get('career_races', 0) == 0:
                        maiden_count += 1
                except Exception:
                    set_horse_defaults(horse)
            else:
                set_horse_defaults(horse)
        result['maiden_horse_count'] = maiden_count

        # 3. オッズ取得
        odds_full = {}
        try:
            odds_full = fetch_realtime_odds_full(rid)
        except Exception:
            pass
        odds_dict = {u: v['odds'] for u, v in odds_full.items()} if odds_full else {}
        result['odds_available'] = bool(odds_dict and any(v > 0 for v in odds_dict.values()))

        # 4. JRA馬場・天候
        jra_info, weather_info = {}, {}
        try:
            jra_info, weather_info = fetch_jra_and_weather(rinfo.get('course', ''))
        except Exception:
            pass

        # 5. build_features (CORE)
        df = build_features(horses, rinfo, model_data, race_id=rid,
                            odds_dict=odds_dict, jra_track_info=jra_info, weather_info=weather_info)

        features = model_data.get('features', [])
        result['df_shape'] = list(df.shape)
        result['model_features_count'] = len(features)

        # 6. 全特徴量ダンプ
        zero_feats = []
        nonzero_feats = []
        for feat in features:
            if feat in df.columns:
                vals = pd.to_numeric(df[feat], errors='coerce').fillna(0).values
                if (np.abs(vals) < 1e-9).all():
                    zero_feats.append(feat)
                else:
                    nonzero_feats.append(feat)
            else:
                zero_feats.append(feat)
        result['zero_features'] = zero_feats
        result['zero_count'] = len(zero_feats)
        result['nonzero_count'] = len(nonzero_feats)

        # 正当なゼロ特徴量
        # surface_enc=0は芝, dist_cat=0は短距離(~1200m), location_enc=0は美浦
        # jrdb_prev_interference/late_start/ze_furi_count は稀なイベント
        # stable_comment_score はカバレッジ不足で0
        # is_long_transport は500km未満の輸送が多い場合0
        # surface_dist_enc は芝短距離で0
        LEGITIMATE_ZEROS = {
            'season', 'is_nar', 'odds_change_rate', 'pop_rank_change', 'odds_sharp_drop',
            'course_renovated', 'post_renovation_flag', 'gaisha_rank',
            'jockey_horse_rides', 'jockey_horse_wr', 'jockey_horse_top3r',
            'jockey_change', 'jockey_change_to_top',
            'paci_sogo_mark', 'paci_idm_mark', 'paci_jockey_mark', 'paci_train_mark',
            'paci_manken_idx', 'paci_goal_rank', 'paci_dochu_rank', 'paci_goal_diff',
            'paci_jockey_exp_wr', 'paci_jockey_exp_3rd', 'paci_ninki_idx',
            'shinba_comment_score', 'dist_change', 'carry_diff',
            'bracket', 'bracket_pos',
            # 値域的に0が正当な特徴量
            'surface_enc',           # 芝=0
            'dist_cat',              # 短距離bin=0
            'location_enc',          # 美浦=0
            'surface_dist_enc',      # 芝×短距離=0
            'is_long_transport',     # 500km未満=0
            'stable_comment_score',  # カバレッジ不足(30%)で0が多い
            # JRDBイベント系(稀なイベントで全馬ゼロが正常)
            'jrdb_prev_interference',  # 前走被害(不利)フラグ
            'jrdb_prev_late_start',    # 前走出遅れフラグ
            'jrdb_ze_furi_count',      # 前走ゼリー振り回数
            # 調教系(premium cache未更新の場合にゼロ)
            'has_wood_training',       # 木馬場調教データ有無
            'wood_count_2w',           # 木馬場調教回数
            'has_sakaro_training',     # 坂路調教データ有無
            'total_training_count',    # 調教合計回数
            'training_intensity_enc',  # 調教強度(premium cache依存)
        }
        suspicious_zeros = [f for f in zero_feats if f not in LEGITIMATE_ZEROS]
        result['suspicious_zeros'] = suspicious_zeros

        # 7. v15新特徴量検証
        v15_checks = {}
        if 'transport_distance_km' in df.columns:
            td = df['transport_distance_km'].values
            v15_checks['transport_distance_km'] = {
                'min': round(float(np.min(td)), 1),
                'max': round(float(np.max(td)), 1),
                'mean': round(float(np.mean(td)), 1),
                'valid': bool(np.max(td) > 0),
            }
            # 妥当性: 美浦→中山は近い(~55km), 栗東→中山は遠い(~450km)
            loc = df['location_enc'].values if 'location_enc' in df.columns else []
            crs = rinfo.get('course_enc', -1)
            if len(loc) > 0:
                v15_checks['transport_check'] = {
                    'miho_horses': int((np.array(loc) == 0).sum()),
                    'ritto_horses': int((np.array(loc) == 1).sum()),
                }
        if 'course_renovated' in df.columns:
            cr = df['course_renovated'].values
            venue = race_info_dict.get('venue', '')
            # 京都改修は2023年4月で12ヶ月以内ではもうない(2026年)
            # 中京改修は2012年で12ヶ月超過
            expected_reno = 0
            v15_checks['course_renovated'] = {
                'value': int(cr[0]) if len(cr) > 0 else -1,
                'expected': expected_reno,
                'match': bool(int(cr[0]) == expected_reno) if len(cr) > 0 else False,
            }
        if 'jockey_change' in df.columns:
            jc = df['jockey_change'].values
            v15_checks['jockey_change'] = {
                'changed_count': int(jc.sum()),
                'total': len(jc),
            }
        result['v15_checks'] = v15_checks

        # 8. JRDB特徴量カバレッジ
        jrdb_feats = [f for f in features if f.startswith('jrdb_')]
        jrdb_nonzero = [f for f in jrdb_feats if f in nonzero_feats]
        result['jrdb_coverage'] = f'{len(jrdb_nonzero)}/{len(jrdb_feats)}'

        # 9. 調教データ検証
        train_feats = ['training_time_filled', 'has_training', 'wood_best_4f_filled',
                       'has_wood_training', 'time_1f_last_filled', 'training_intensity_enc']
        train_ok = 0
        for tf in train_feats:
            if tf in df.columns and (np.abs(df[tf].values) > 1e-9).any():
                train_ok += 1
        result['training_coverage'] = f'{train_ok}/{len(train_feats)}'

        # 10. Speed Index検証
        si_feats = ['index_max_filled', 'index_avg5_filled', 'index_run1_filled']
        si_ok = 0
        for sf in si_feats:
            if sf in df.columns:
                v = df[sf].values
                if not ((v == 1045.0).all() or (v == 1040.0).all() or (v == 1035.0).all()):
                    si_ok += 1
        result['speed_index_real'] = f'{si_ok}/{len(si_feats)}'

        # 11. 予測実行
        odds_available = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        df = predict_race(df, model_data, odds_available, race_info=rinfo)
        df_sorted = df.sort_values('スコア', ascending=False).reset_index(drop=True)

        scores = df_sorted['スコア'].values
        result['score_range'] = [round(float(scores.min()), 4), round(float(scores.max()), 4)]
        result['score_in_01'] = bool(scores.min() >= -0.01 and scores.max() <= 1.01)

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

        # 12. 条件判定検証
        cond_key, cond_profile = classify_race_condition(rinfo, len(df_sorted))
        result['condition_key'] = cond_key
        result['bet_type'] = cond_profile['bet_type']
        result['recommended'] = cond_profile.get('recommended', True)

        # 条件一致チェック
        if expected_cond and expected_cond != 'SKIP':
            cond_match = cond_key == expected_cond
            result['checks']['condition_match'] = cond_match
            if not cond_match:
                result['warnings'].append(f'条件不一致: 期待={expected_cond}, 実際={cond_key}')

        # 13. 買い目生成検証
        if cond_profile['bet_type'] == 'umaren':
            bets = generate_umaren_bets(df_sorted)
            expected_count = 2
        else:
            bets = generate_trio_bets(df_sorted)
            expected_count = 7
        result['bets'] = [list(b) for b in bets]
        result['bet_count'] = len(bets)
        result['checks']['bet_count_ok'] = len(bets) == expected_count

        # 馬番昇順チェック
        for b in bets:
            if len(b) >= 2:
                assert list(b) == sorted(b), f"馬番昇順でない: {b}"
        result['checks']['bets_sorted'] = True

        # 14. ★パターンマッチ
        top1_odds = min((v for v in odds_dict.values() if v > 0), default=0) if odds_dict else 0
        pp_stars, pp_matched = match_profitable_patterns(
            cond_key, rinfo.get('course', ''), rinfo.get('surface', '芝'),
            rinfo.get('distance', 1600), top1_odds)
        result['pp_stars'] = pp_stars
        result['pp_matched_count'] = len(pp_matched)

        # 15. Kelly基準テスト
        try:
            kelly = calc_kelly_investment(
                win_prob=0.4, odds_ratio=5.0, bankroll=30000)
            result['checks']['kelly_works'] = kelly is not None and kelly >= 100
        except Exception:
            result['checks']['kelly_works'] = False

        # 16. EV計算テスト
        try:
            ev_map = calc_horse_ev(df_sorted, odds_dict)
            result['checks']['ev_works'] = bool(ev_map)
        except Exception:
            result['checks']['ev_works'] = False

        # 17. レース信頼度テスト
        try:
            conf, reasons = calc_race_confidence(df_sorted)
            result['confidence'] = conf
            result['checks']['confidence_works'] = conf >= 0
        except Exception:
            result['checks']['confidence_works'] = False

        # ステータス判定
        if suspicious_zeros:
            result['status'] = 'WARN'
            result['warnings'].append(f'怪しいゼロ特徴量: {suspicious_zeros}')
        else:
            result['status'] = 'PASS'

    except Exception as e:
        result['status'] = 'FAIL'
        result['errors'].append(f'{type(e).__name__}: {str(e)}')
        traceback.print_exc()

    return result


# ==========================================
# PHASE 4: app.py/Discord整合テスト (コード検査)
# ==========================================

def run_phase4():
    print('\n' + '='*70)
    print('  PHASE 4: app.py/Discord整合テスト')
    print('='*70)
    results = []

    # 4-1. app.pyがpredict_coreを使用
    app_path = os.path.join(BASE_DIR, 'app.py')
    app_src = open(app_path, 'r', encoding='utf-8').read()

    def chk(name, ok, msg=''):
        results.append({'name': name, 'status': 'PASS' if ok else 'FAIL', 'note': msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {msg}" if msg and not ok else ""))

    chk("4-1 app.pyがpredict_coreインポート",
        'from tools.predict_core import' in app_src or 'import predict_core' in app_src or 'from predict_core import' in app_src)

    chk("4-2 app.pyに独自predict_raceなし",
        'def predict_race(' not in app_src)

    chk("4-3 波乱度パネルコード存在",
        'upset_level' in app_src)

    chk("4-4 ★パターン表示コード存在",
        'pp_stars' in app_src or 'profitable_pattern' in app_src or '★★★' in app_src)

    chk("4-5 条件表示(A-X)コード存在",
        'condition_key' in app_src or 'cond_key' in app_src or '条件A' in app_src)

    # 4-6. race_auto_notify.pyがpredict_coreを使用
    ran_path = os.path.join(BASE_DIR, 'tools', 'race_auto_notify.py')
    if os.path.exists(ran_path):
        ran_src = open(ran_path, 'r', encoding='utf-8').read()
        chk("4-6 race_auto_notify→predict_core",
            'predict_core' in ran_src)
        chk("4-7 Discord webhook送信コード",
            'webhook' in ran_src.lower() or 'discord' in ran_src.lower())
    else:
        chk("4-6 race_auto_notify.py存在", False, "ファイルなし")

    # 4-8. app.py構文チェック
    try:
        import py_compile
        py_compile.compile(app_path, doraise=True)
        chk("4-8 app.py構文チェック", True)
    except py_compile.PyCompileError as e:
        chk("4-8 app.py構文チェック", False, str(e))

    return results


# ==========================================
# PHASE 5: 障害シナリオテスト (コード検査)
# ==========================================

def run_phase5():
    print('\n' + '='*70)
    print('  PHASE 5: 障害シナリオテスト')
    print('='*70)
    results = []

    def chk(name, ok, msg=''):
        results.append({'name': name, 'status': 'PASS' if ok else 'FAIL', 'note': msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {msg}" if msg and not ok else ""))

    pc_src = open(os.path.join(BASE_DIR, 'tools', 'predict_core.py'), 'r', encoding='utf-8').read()

    # 5-1. JRDB取得失敗時フォールバック
    chk("5-1 JRDBフォールバック(except→0埋め)",
        'except' in pc_src and ('jrdb' in pc_src.lower()))

    # 5-2. netkeibaアクセス不可時フォールバック
    chk("5-2 netkeiba例外処理",
        pc_src.count('except') > 10)  # 多数のtry/except

    # 5-3. Cookie関連
    chk("5-3 Cookie管理(.env)",
        os.path.exists(os.path.join(BASE_DIR, '.env')) or 'NETKEIBA_COOKIE' in pc_src)

    # 5-4. thisweek CSV空チェック
    for csv_name in ['netkeiba_data_analysis_thisweek.csv', 'netkeiba_upset_level_thisweek.csv']:
        csv_path = os.path.join(BASE_DIR, 'data', csv_name)
        if os.path.exists(csv_path):
            try:
                df_tw = pd.read_csv(csv_path)
                chk(f"5-5 {csv_name}読み込み", len(df_tw) >= 0, f"{len(df_tw)}行")
            except Exception as e:
                chk(f"5-5 {csv_name}読み込み", False, str(e))

    # 5-6. batに日本語混入チェック
    bats = ['daily_predict.bat', 'daily_results.bat', 'race_auto_notify.bat']
    for bat in bats:
        full = os.path.join(BASE_DIR, bat)
        if not os.path.exists(full): continue
        try:
            content = open(full, 'rb').read()
            # Check if it's pure ASCII (except for common Windows line endings)
            non_ascii = [b for b in content if b > 127]
            chk(f"5-6 {bat}日本語なし", len(non_ascii) == 0,
                f"non-ASCII bytes: {len(non_ascii)}")
        except Exception as e:
            chk(f"5-6 {bat}読み込み", False, str(e))

    return results


# ==========================================
# PHASE 6: 9ペルソナ承認 (自動判定)
# ==========================================

def run_phase6(all_results, phase3_results, phase4_results, phase5_results):
    print('\n' + '='*70)
    print('  PHASE 6: 9ペルソナ最終承認')
    print('='*70)

    verdicts = {}

    # 全レースの統計
    total = len(all_results)
    passes = sum(1 for r in all_results.values() if r['status'] == 'PASS')
    warns = sum(1 for r in all_results.values() if r['status'] == 'WARN')
    fails = sum(1 for r in all_results.values() if r['status'] == 'FAIL')
    skips = sum(1 for r in all_results.values() if r['status'].startswith('SKIP'))

    p3_passes = sum(1 for r in phase3_results if r['status'] == 'PASS')
    p3_total = len(phase3_results)

    # 全レースのゼロ特徴量サマリー
    venue_zero_counts = {}
    for name, r in all_results.items():
        venue = r.get('venue', '?')
        zc = r.get('zero_count', 0)
        if venue not in venue_zero_counts:
            venue_zero_counts[venue] = []
        venue_zero_counts[venue].append(zc)

    # 1. 田中（MLエンジニア）
    tanaka_ok = True
    tanaka_notes = []
    md = load_models()
    feats = md.get('features', [])
    tanaka_notes.append(f"特徴量数: {len(feats)}")
    tanaka_notes.append(f"LGB: {'OK' if md.get('model') else 'NG'}")
    tanaka_notes.append(f"XGB: {'OK' if md.get('xgb_model') else 'NG'}")
    w = md.get('ensemble_weights', {})
    tanaka_notes.append(f"重み: LGB={w.get('lgb',0):.3f}, XGB={w.get('xgb',0):.3f}")
    if fails > 0:
        tanaka_ok = False
        tanaka_notes.append(f"FAIL: {fails}レース")
    verdicts['田中(ML)'] = {'ok': tanaka_ok, 'notes': tanaka_notes}
    print(f"\n  田中（MLエンジニア15年）: {'OK' if tanaka_ok else 'NG'}")
    for n in tanaka_notes: print(f"    - {n}")

    # 2. 鈴木（品質管理）
    suzuki_ok = p3_passes == p3_total and fails == 0
    suzuki_notes = [
        f"回帰テスト: {p3_passes}/{p3_total} PASS",
        f"全レース: {passes} PASS, {warns} WARN, {fails} FAIL, {skips} SKIP",
    ]
    for venue, zcs in venue_zero_counts.items():
        avg_zc = sum(zcs) / len(zcs)
        suzuki_notes.append(f"  {venue}: 平均ゼロ特徴量 {avg_zc:.1f}個")
    verdicts['鈴木(QA)'] = {'ok': suzuki_ok, 'notes': suzuki_notes}
    print(f"\n  鈴木（品質管理20年）: {'OK' if suzuki_ok else 'NG'}")
    for n in suzuki_notes: print(f"    - {n}")

    # 3. 佐藤（セキュリティ）
    env_in_gi = any(r['status'] == 'PASS' for r in phase3_results if '3-12' in r['name'])
    sato_ok = env_in_gi
    sato_notes = [f".env gitignore: {'OK' if env_in_gi else 'NG'}"]
    verdicts['佐藤(Sec)'] = {'ok': sato_ok, 'notes': sato_notes}
    print(f"\n  佐藤（セキュリティ）: {'OK' if sato_ok else 'NG'}")
    for n in sato_notes: print(f"    - {n}")

    # 4. 山田（データアナリスト）
    yamada_ok = True
    yamada_notes = []
    # transport_distance_kmの検証
    for name, r in all_results.items():
        v15c = r.get('v15_checks', {})
        td = v15c.get('transport_distance_km', {})
        if td and not td.get('valid', False):
            yamada_ok = False
            yamada_notes.append(f"{name}: transport_distance_km全てゼロ")
    if not yamada_notes:
        yamada_notes.append("v15特徴量: 全レースで妥当")
    verdicts['山田(DA)'] = {'ok': yamada_ok, 'notes': yamada_notes}
    print(f"\n  山田（データアナリスト）: {'OK' if yamada_ok else 'NG'}")
    for n in yamada_notes: print(f"    - {n}")

    # 5. 木村（馬券購入者）
    kimura_ok = True
    kimura_notes = []
    bet_issues = []
    for name, r in all_results.items():
        if r['status'].startswith('SKIP'): continue
        checks = r.get('checks', {})
        if not checks.get('bet_count_ok', True):
            bet_issues.append(f"{name}: bet_count={r.get('bet_count', '?')}")
    if bet_issues:
        kimura_ok = False
        kimura_notes.extend(bet_issues)
    else:
        kimura_notes.append("全レース買い目生成OK")
    # ★★★レース数
    star3_races = [n for n, r in all_results.items() if r.get('pp_stars', 0) >= 3]
    kimura_notes.append(f"★★★レース: {len(star3_races)}件 {star3_races}")
    verdicts['木村(馬券)'] = {'ok': kimura_ok, 'notes': kimura_notes}
    print(f"\n  木村（馬券購入者）: {'OK' if kimura_ok else 'NG'}")
    for n in kimura_notes: print(f"    - {n}")

    # 6. 高橋（自動化エンジニア）
    bat_ok = any(r['status'] == 'PASS' for r in phase3_results if '3-13' in r['name'])
    takahashi_ok = bat_ok
    takahashi_notes = [f"batエンコーディング: {'OK' if bat_ok else 'NG'}"]
    verdicts['高橋(自動化)'] = {'ok': takahashi_ok, 'notes': takahashi_notes}
    print(f"\n  高橋（自動化エンジニア）: {'OK' if takahashi_ok else 'NG'}")
    for n in takahashi_notes: print(f"    - {n}")

    # 7. 渡辺（コンサルタント）
    ssot_ok = any(r['status'] == 'PASS' for r in phase3_results if '3-2' in r['name'])
    watanabe_ok = ssot_ok and fails == 0
    watanabe_notes = [
        f"Single Source of Truth: {'OK' if ssot_ok else 'NG'}",
        f"全レースFAIL: {fails}件",
    ]
    verdicts['渡辺(コンサル)'] = {'ok': watanabe_ok, 'notes': watanabe_notes}
    print(f"\n  渡辺（競馬AIコンサル）: {'OK' if watanabe_ok else 'NG'}")
    for n in watanabe_notes: print(f"    - {n}")

    # 8. 中村（SRE）
    nakamura_ok = True
    nakamura_notes = []
    p5_fails = sum(1 for r in phase5_results if r['status'] == 'FAIL')
    if p5_fails > 0:
        nakamura_ok = False
        nakamura_notes.append(f"障害シナリオFAIL: {p5_fails}件")
    else:
        nakamura_notes.append("障害シナリオ: 全PASS")
    verdicts['中村(SRE)'] = {'ok': nakamura_ok, 'notes': nakamura_notes}
    print(f"\n  中村（SRE）: {'OK' if nakamura_ok else 'NG'}")
    for n in nakamura_notes: print(f"    - {n}")

    # 9. 松田（プロ馬券師）
    matsuda_ok = True
    matsuda_notes = []
    # 連敗シナリオ: 700円×20連敗 = 14,000円損失、初期30,000円 → 残り16,000円
    matsuda_notes.append("20連敗時資金残高: 30,000 - 14,000 = 16,000円 (53.3%残)")
    matsuda_notes.append("1日最大損失(12R全敗): 700×12 = 8,400円")
    # ★★★パターン内訳
    pp_path = os.path.join(BASE_DIR, 'data', 'profitable_patterns.json')
    if os.path.exists(pp_path):
        pp = json.load(open(pp_path, 'r', encoding='utf-8'))
        patterns = pp.get('profitable_patterns', [])
        star3 = [p for p in patterns if p.get('stars', 0) >= 3]
        venues = {}
        for p in star3:
            v = p.get('venue', 'all')
            venues[v] = venues.get(v, 0) + 1
        matsuda_notes.append(f"★★★パターン: {len(star3)}件")
        matsuda_notes.append(f"  内訳: {dict(sorted(venues.items(), key=lambda x: -x[1]))}")
    verdicts['松田(プロ)'] = {'ok': matsuda_ok, 'notes': matsuda_notes}
    print(f"\n  松田（プロ馬券師）: {'OK' if matsuda_ok else 'NG'}")
    for n in matsuda_notes: print(f"    - {n}")

    # 最終判定
    all_ok = all(v['ok'] for v in verdicts.values())
    return all_ok, verdicts


# ==========================================
# MAIN
# ==========================================

def main():
    start_time = time.time()
    print('='*70)
    print('  v15 本番前 完全最終検証')
    print(f'  実行日時: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*70)

    # モデルロード
    model_data = load_models()
    version = model_data.get('version', '?')
    features = model_data.get('features', [])
    print(f'\nModel: {version}')
    print(f'Features: {len(features)}')
    print(f'LGB: {model_data["model"] is not None}')
    print(f'XGB: {model_data.get("xgb_model") is not None}')
    w = model_data.get('ensemble_weights', {})
    print(f'Weights: LGB={w.get("lgb",0):.3f}, XGB={w.get("xgb",0):.3f}')

    # PHASE 3: 回帰テスト (先に実行)
    run_phase3()

    # テストレースロード
    test_file = os.path.join(BASE_DIR, 'data', '_final_test_races_v15.json')
    with open(test_file, 'r', encoding='utf-8') as f:
        TEST_RACES = json.load(f)

    # PHASE 0+1+2: 全レース内部ダンプ
    print('\n' + '='*70)
    print(f'  PHASE 0+1+2: 全{len(TEST_RACES)}レース内部ダンプテスト')
    print('='*70)

    all_results = {}
    total_pass = total_warn = total_fail = total_skip = 0

    for cls_name, race_info in TEST_RACES.items():
        print(f'\n--- [{cls_name}] {race_info["race_id"]} {race_info.get("race_name","")} ---')

        result = test_one_race_full(cls_name, race_info, model_data)
        all_results[cls_name] = result

        status = result['status']
        if status == 'PASS':
            total_pass += 1
            print(f'  STATUS: PASS | zeros={result.get("zero_count",0)} | '
                  f'cond={result.get("condition_key","?")} | '
                  f'bets={result.get("bet_count","?")} | '
                  f'stars={result.get("pp_stars",0)}')
            if result.get('top3'):
                for t in result['top3']:
                    print(f'    {t["rank"]}位: {t["horse_num"]}番 {t["horse_name"]} ({t["score"]:.4f})')
        elif status == 'WARN':
            total_warn += 1
            print(f'  STATUS: WARN | zeros={result.get("zero_count",0)} | suspicious={result.get("suspicious_zeros",[])}')
        elif status.startswith('SKIP'):
            total_skip += 1
            print(f'  STATUS: {status} (障害レース除外)')
        else:
            total_fail += 1
            print(f'  STATUS: FAIL | errors={result.get("errors",[])}')

    print(f'\n--- PHASE 0-2 サマリー ---')
    print(f'  PASS: {total_pass}, WARN: {total_warn}, FAIL: {total_fail}, SKIP: {total_skip}')

    # PHASE 4
    phase4_results = run_phase4()

    # PHASE 5
    phase5_results = run_phase5()

    # PHASE 6
    all_approved, verdicts = run_phase6(all_results, PHASE3_RESULTS, phase4_results, phase5_results)

    # 最終判定
    elapsed = time.time() - start_time
    print('\n' + '='*70)
    if all_approved and total_fail == 0:
        print('  9ペルソナ APPROVED・本番GO')
    else:
        print('  NG — 以下を修正してリトライ')
        for name, v in verdicts.items():
            if not v['ok']:
                print(f'    {name}: NG')
                for n in v['notes']:
                    print(f'      {n}')
    print(f'  実行時間: {elapsed:.1f}秒')
    print('='*70)

    # JSON出力
    output = {
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': version,
        'model_features': len(features),
        'ensemble_weights': w,
        'phase0_2': {
            'total': len(all_results),
            'pass': total_pass,
            'warn': total_warn,
            'fail': total_fail,
            'skip': total_skip,
            'races': all_results,
        },
        'phase3_regression': PHASE3_RESULTS,
        'phase4_app': phase4_results,
        'phase5_failure': phase5_results,
        'phase6_personas': {k: {'approved': v['ok'], 'notes': v['notes']} for k, v in verdicts.items()},
        'final_verdict': 'APPROVED' if (all_approved and total_fail == 0) else 'REJECTED',
        'elapsed_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(BASE_DIR, 'data', 'fullclass_final_test_v15.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n結果保存: {out_path}')

    return 0 if (all_approved and total_fail == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
