"""predict_core共通モジュールのテスト"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from predict_core import (
    load_models, build_features, predict_race,
    classify_race_condition, generate_trio_bets, generate_umaren_bets,
    find_jockey_wr, calc_sire_score, set_horse_defaults, apply_horse_stats,
    COURSE_MAP, SURFACE_MAP, COND_MAP, SEX_MAP,
)


def make_test_horses(n=10):
    """テスト用の馬データを生成"""
    horses = []
    for i in range(n):
        horse = {
            '馬名': f'テスト馬{i+1}',
            '馬体重': 480 + i * 4,
            '場体重増減': i - 5,
            '斤量': 55.0 + (i % 3) * 0.5,
            '馬齢': 3 + (i % 4),
            '距離(m)': 1600,
            '競馬場コード_enc': COURSE_MAP.get('中山', 5),
            '芝ダート_enc': SURFACE_MAP.get('芝', 0),
            '馬場状態_enc': COND_MAP.get('良', 0),
            '性別_enc': i % 2,
            '騎手勝率': 0.08 + i * 0.01,
            '騎手名': f'騎手{i}',
            '枠番': i % 8 + 1,
            '馬番': i + 1,
            '調教師': f'調教師{i}',
            '性別': '牡' if i % 2 == 0 else '牝',
            'horse_id_val': 2020100000 + i,
            '単勝オッズ': 5.0 + i * 3,
        }
        set_horse_defaults(horse)
        horse['前走着順'] = 3 + i % 5
        horse['父'] = 'ディープインパクト' if i % 3 == 0 else ''
        horse['母の父'] = ''
        horse['血統スコア'] = calc_sire_score(horse['父'], '芝', 1600)
        horses.append(horse)
    return horses


def test_basic_prediction():
    """基本的な予測パイプラインのテスト"""
    print("=== Test: Basic Prediction Pipeline ===")
    model_data = load_models()
    assert model_data['model'] is not None, "Model not loaded"
    print(f"  Model: version={model_data['version']}, is_live={model_data['is_live']}")

    horses = make_test_horses(10)
    race_info = {'distance': 1600, 'surface': '芝', 'condition': '良', 'course': '中山', 'weather': '晴'}
    odds_dict = {i + 1: 5.0 + i * 3 for i in range(10)}

    df = build_features(horses, race_info, model_data, odds_dict=odds_dict)
    assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
    print(f"  build_features: {len(df)} rows, {len(df.columns)} columns")

    df = predict_race(df, model_data, odds_available=True, race_info=race_info)
    assert 'スコア' in df.columns, "Score column missing"
    assert 'AI順位' in df.columns, "AI rank column missing"

    for _, row in df.iterrows():
        print(f"  #{int(row['AI順位'])}: 馬番{int(row['馬番'])} {row['馬名']} score={row['スコア']:.4f}")

    print("  PASSED\n")
    return df


def test_condition_classification():
    """条件分類のテスト"""
    print("=== Test: Condition Classification ===")
    tests = [
        ({'distance': 1600, 'condition': '良'}, 12, 'A'),
        ({'distance': 1600, 'condition': '重'}, 12, 'B'),
        ({'distance': 2000, 'condition': '良'}, 16, 'C'),
        ({'distance': 1200, 'condition': '良'}, 12, 'D'),
        ({'distance': 1600, 'condition': '良'}, 6, 'E'),
        ({'distance': 2000, 'condition': '不良'}, 16, 'X'),
    ]
    for race_info, nh, expected in tests:
        key, profile = classify_race_condition(race_info, nh)
        assert key == expected, f"Expected {expected}, got {key}"
        print(f"  {key}: {profile['desc']} (bet={profile['bet_type']}) OK")
    print("  PASSED\n")


def test_bet_generation():
    """買い目生成のテスト"""
    print("=== Test: Bet Generation ===")
    model_data = load_models()
    horses = make_test_horses(10)
    race_info = {'distance': 1600, 'surface': '芝', 'condition': '良', 'course': '中山', 'weather': '晴'}

    df = build_features(horses, race_info, model_data)
    df = predict_race(df, model_data, odds_available=False, race_info=race_info)
    sorted_df = df.sort_values('スコア', ascending=False).reset_index(drop=True)

    trio_bets = generate_trio_bets(sorted_df)
    assert len(trio_bets) == 7, f"Expected 7 trio bets, got {len(trio_bets)}"
    print(f"  Trio: {len(trio_bets)} bets = {trio_bets}")

    umaren_bets = generate_umaren_bets(sorted_df)
    assert len(umaren_bets) == 2, f"Expected 2 umaren bets, got {len(umaren_bets)}"
    print(f"  Umaren: {len(umaren_bets)} bets = {umaren_bets}")
    print("  PASSED\n")


def test_prediction_consistency():
    """同一データで2回予測して結果が一致することを確認"""
    print("=== Test: Prediction Consistency ===")
    model_data = load_models()
    horses = make_test_horses(12)
    race_info = {'distance': 2000, 'surface': '芝', 'condition': '稍重', 'course': '東京', 'weather': '曇'}
    odds_dict = {i + 1: 3.0 + i * 2.5 for i in range(12)}

    # Run 1
    df1 = build_features(horses, race_info, model_data, odds_dict=odds_dict)
    df1 = predict_race(df1, model_data, odds_available=True, race_info=race_info)
    scores1 = df1.sort_values('馬番')['スコア'].values

    # Run 2
    df2 = build_features(horses, race_info, model_data, odds_dict=odds_dict)
    df2 = predict_race(df2, model_data, odds_available=True, race_info=race_info)
    scores2 = df2.sort_values('馬番')['スコア'].values

    import numpy as np
    assert np.allclose(scores1, scores2), f"Scores differ!\n  Run1: {scores1}\n  Run2: {scores2}"
    print(f"  Two runs produce identical scores: {np.allclose(scores1, scores2)}")

    # Verify bet generation is also identical
    sorted1 = df1.sort_values('スコア', ascending=False).reset_index(drop=True)
    sorted2 = df2.sort_values('スコア', ascending=False).reset_index(drop=True)
    bets1 = generate_trio_bets(sorted1)
    bets2 = generate_trio_bets(sorted2)
    assert bets1 == bets2, f"Bets differ!\n  Run1: {bets1}\n  Run2: {bets2}"
    print(f"  Bets identical: {bets1}")
    print("  PASSED\n")


def test_daily_predict_uses_core():
    """daily_predict.pyがpredict_coreを使っていることを確認"""
    print("=== Test: daily_predict uses predict_core ===")
    from daily_predict import (
        load_models as dp_load_models,
        build_features as dp_build_features,
        predict_race as dp_predict_race,
        classify_race_condition as dp_classify,
        generate_trio_bets as dp_trio,
    )
    # These should be the EXACT SAME functions
    assert dp_load_models is load_models, "load_models is not from predict_core"
    assert dp_build_features is build_features, "build_features is not from predict_core"
    assert dp_predict_race is predict_race, "predict_race is not from predict_core"
    assert dp_classify is classify_race_condition, "classify is not from predict_core"
    assert dp_trio is generate_trio_bets, "trio is not from predict_core"
    print("  All functions are identical references to predict_core")
    print("  PASSED\n")


if __name__ == "__main__":
    test_condition_classification()
    test_basic_prediction()
    test_bet_generation()
    test_prediction_consistency()
    test_daily_predict_uses_core()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
