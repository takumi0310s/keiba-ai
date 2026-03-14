"""3/15 1レース分の全83特徴量検証"""
import sys, os, time, json, pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from tools.daily_predict import (
    parse_shutuba, build_features, load_models, load_feature_lookups,
    get_horse_stats, _set_defaults, predict_race, classify_race_condition,
    generate_trio_bets, generate_umaren_bets, fetch_race_list, calc_sire_score
)

def main():
    print("=" * 80)
    print("  3/15 全特徴量検証 (1レース)")
    print("=" * 80)

    # 1. レースリスト取得
    races = fetch_race_list('20260315')
    print(f"\n3/15 レースリスト: {len(races)}レース")

    # 中山1R or 阪神1R or 最初のレースを使用
    target = None
    for r in races:
        cname = r.get('course', '')
        rnum = r.get('race_num', 0)
        if rnum == 1:
            target = r
            break
    if not target:
        target = races[0]

    race_id = target['race_id']
    print(f"対象: {target['course']} {target['race_num']}R (race_id={race_id})")

    # 2. モデルロード
    model_data = load_models()
    model_features = model_data['features']
    print(f"モデル: Pattern {'B' if model_data['is_live'] else 'A'} ({len(model_features)}特徴量)")

    # 3. 出馬表取得
    race_name, horses, horse_ids, race_info = parse_shutuba(race_id)
    num_horses = len(horses)
    print(f"出馬表: {num_horses}頭, {race_info['surface']}{race_info['distance']}m, "
          f"馬場={race_info['condition']}, 競馬場={race_info['course']}")

    if num_horses == 0:
        print("[ERROR] 出馬表取得失敗")
        return

    # 4. 各馬の成績取得 (最初の3頭のみ詳細、残りはデフォルト)
    print(f"\n前走データ取得中...", end="", flush=True)
    for i, (h, hid) in enumerate(zip(horses, horse_ids)):
        if hid:
            try:
                stats = get_horse_stats(hid, race_info['distance'], race_info['surface'],
                                        race_info.get('course', ''))
                h['前走着順'] = stats.get('last_finish', 5)
                h['距離適性'] = stats.get('dist_apt', 0.5)
                h['馬場適性'] = stats.get('surf_apt', 0.5)
                h['人気傾向'] = stats.get('pop_score', 0.5)
                h['コース適性'] = stats.get('course_apt', 0.5)
                h['前走間隔'] = stats.get('interval_days', 30)
                h['脚質'] = stats.get('running_style', 0)
                h['上がり3F'] = stats.get('avg_agari', 35.5)
                h['複勝率'] = stats.get('fukusho_rate', 0.0)
                h['父'] = stats.get('father', '')
                h['母の父'] = stats.get('mother_father', '')
                h['血統スコア'] = calc_sire_score(stats.get('father', ''), race_info['surface'], race_info['distance'])
                h['通過順平均'] = stats.get('avg_pass_pos', 8.0)
                h['通過順4'] = stats.get('last_pass4', 8)
                h['前走オッズ'] = stats.get('last_odds', 15.0)
                h['前走人気'] = stats.get('last_pop', 8)
                h['所属地'] = stats.get('trainer_loc', '')
                h['prev2_finish'] = stats.get('prev2_finish', 5)
                h['prev3_finish'] = stats.get('prev3_finish', 5)
                h['prev4_finish'] = stats.get('prev4_finish', 5)
                h['prev5_finish'] = stats.get('prev5_finish', 5)
                h['avg_finish_3r'] = stats.get('avg_finish_3r', 5.0)
                h['avg_finish_5r'] = stats.get('avg_finish_5r', 5.0)
                h['best_finish_3r'] = stats.get('best_finish_3r', 5)
                h['best_finish_5r'] = stats.get('best_finish_5r', 5)
                h['top3_count_3r'] = stats.get('top3_count_3r', 0)
                h['top3_count_5r'] = stats.get('top3_count_5r', 0)
                h['finish_trend'] = stats.get('finish_trend', 0)
                h['prev2_last3f'] = stats.get('prev2_last3f', 35.5)
                print(".", end="", flush=True)
            except Exception as e:
                _set_defaults(h)
                print("x", end="", flush=True)
        else:
            _set_defaults(h)
            print("-", end="", flush=True)
        time.sleep(0.5)
    print(" done")

    # 5. 特徴量構築
    df = build_features(horses, race_info, model_data, odds_dict={},
                        jra_track_info={}, weather_info={})
    print(f"\nbuild_features: {len(df)}行 x {len(df.columns)}列")

    # 6. 予測実行
    df = predict_race(df, model_data, odds_available=False)

    # TOP1予測馬を取得
    # スコア列名を検出
    score_col = 'score' if 'score' in df.columns else 'スコア'
    sorted_df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    top1_idx = sorted_df.index[0]
    top1_row = sorted_df.iloc[0]

    print(f"\n{'=' * 80}")
    print(f"  TOP1予測馬: {top1_row.get('horse_name', top1_row.get('馬名', '?'))} "
          f"({int(top1_row.get('馬番', top1_row.get('horse_num', 0)))}番)")
    print(f"  スコア: {top1_row[score_col]:.6f}")
    print(f"{'=' * 80}")

    # 7. 全83特徴量の値を表示
    print(f"\n  全{len(model_features)}特徴量の値:")
    print(f"  {'#':>3} {'特徴量':30s} {'値':>15s} {'状態':>10s}")
    print(f"  {'-'*65}")

    issues = []
    warnings = []
    ok_count = 0

    # 既知のデフォルト値セット
    KNOWN_DEFAULTS = {
        'prev_race_first3f': (0, 'ペースデータ未実装'),
        'prev_race_last3f': (0, 'ペースデータ未実装'),
        'prev_race_pace_diff': (0, 'ペースデータ未実装'),
        'prev_agari_relative': (0, '未実装'),
        'has_wood_training': (0, '調教データ未取得'),
        'wood_count_2w': (0, '調教データ未取得'),
        'has_sakaro_training': (0, '調教データ未取得'),
        'total_training_count': (0, '調教データ未取得'),
    }

    EXPECTED_ZERO_OK = {
        'is_nar',           # JRAなので0が正しい
        'finish_trend',     # 着順変化がない場合0
        'weight_change',    # 前日発表前なので0
        'weight_change_abs',
        'precipitation',    # 降水なし=0は正常
        'condition_enc',    # 良=0は正常
        'cond_surface',     # 良×芝=0は正常
    }

    for i, feat in enumerate(model_features):
        val = top1_row.get(feat, 'MISSING')
        status = "OK"

        if val == 'MISSING' or (isinstance(val, float) and np.isnan(val)):
            status = "FAIL"
            issues.append((feat, val, "欠損"))
        elif feat in KNOWN_DEFAULTS:
            expected_val, reason = KNOWN_DEFAULTS[feat]
            if val == expected_val:
                status = f"WARN({reason[:8]})"
                warnings.append((feat, val, reason))
            else:
                status = "OK"
                ok_count += 1
        elif feat in EXPECTED_ZERO_OK:
            status = "OK"
            ok_count += 1
        elif val == 0 and feat not in EXPECTED_ZERO_OK:
            # 0が疑わしい特徴量
            suspicious = [
                'horse_weight', 'weight_carry', 'age', 'distance', 'course_enc',
                'num_horses', 'horse_num', 'bracket', 'sire_enc', 'bms_enc',
                'prev_finish', 'rest_days', 'horse_career_races',
            ]
            if feat in suspicious:
                status = "FAIL"
                issues.append((feat, val, "ゼロ(異常)"))
            else:
                status = "OK"
                ok_count += 1
        else:
            ok_count += 1

        val_str = f"{val:.6f}" if isinstance(val, float) else str(val)
        print(f"  {i+1:3d} {feat:30s} {val_str:>15s} {status:>10s}")

    # 8. サマリー
    print(f"\n{'=' * 80}")
    print(f"  検証結果サマリー")
    print(f"{'=' * 80}")
    print(f"  OK:       {ok_count}/{len(model_features)}")
    print(f"  ISSUES:   {len(issues)}")
    print(f"  WARNINGS: {len(warnings)} (既知の制約)")

    if issues:
        print(f"\n  --- ISSUES (要修正) ---")
        for feat, val, reason in issues:
            print(f"  [FAIL] {feat} = {val}: {reason}")

    if warnings:
        print(f"\n  --- WARNINGS (既知の制約) ---")
        for feat, val, reason in warnings:
            print(f"  [WARN] {feat} = {val}: {reason}")

    # 条件判定・買い目
    cond_key, cond_profile = classify_race_condition(race_info, num_horses)
    print(f"\n  条件判定: {cond_key} ({cond_profile['desc']})")
    print(f"  推奨券種: {cond_profile['bet_type']}")

    bet_type = cond_profile['bet_type']
    if bet_type == 'umaren':
        bets = generate_umaren_bets(sorted_df)
        print(f"  馬連 {len(bets)}点:")
        amts = [400, 300]
        for bi, b in enumerate(bets):
            print(f"    {b[0]}-{b[1]}: {amts[bi]}円")
    else:
        bets = generate_trio_bets(sorted_df)
        print(f"  三連複 {len(bets)}点:")
        for b in bets:
            print(f"    {b[0]}-{b[1]}-{b[2]}")

    # TOP3予測馬
    print(f"\n  TOP3予測:")
    for rank in range(min(3, len(sorted_df))):
        r = sorted_df.iloc[rank]
        name = r.get('horse_name', r.get('馬名', '?'))
        num = int(r.get('馬番', r.get('horse_num', 0)))
        score = r[score_col]
        print(f"    {rank+1}位: {num}番 {name} (score={score:.6f})")

    # 最終判定
    if len(issues) == 0:
        print(f"\n  >>> 明日の予測準備OK <<<")
        verdict = "READY"
    else:
        print(f"\n  >>> 要修正: {len(issues)}件の問題 <<<")
        verdict = "NEEDS_FIX"

    # JSON保存
    result = {
        'date': '20260315',
        'race_id': race_id,
        'verdict': verdict,
        'features_total': len(model_features),
        'features_ok': ok_count,
        'issues': len(issues),
        'warnings': len(warnings),
        'issue_details': [{'feature': f, 'value': str(v), 'reason': r} for f, v, r in issues],
        'warning_details': [{'feature': f, 'value': str(v), 'reason': r} for f, v, r in warnings],
    }
    out_path = os.path.join(BASE_DIR, 'data', 'pipeline_audit.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {out_path}")

if __name__ == '__main__':
    main()
