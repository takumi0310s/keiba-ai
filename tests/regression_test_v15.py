"""v15回帰テスト — 過去バグ11件 + v15新機能テスト

Usage:
    python -m pytest tests/regression_test_v15.py -v
    python tests/regression_test_v15.py

テスト項目:
    1-1. バージョンゲート: v15+で基本特徴量ブロック有効
    1-2. 予測ロジック一元化: predict_coreが唯一の真実
    1-3. 特徴量数一致: LGB(145) <= features(150)
    1-4. 距離カテゴリ5bin
    1-5. stdout再ラップなし
    1-6. .envがgitignore
    1-7. bat PYTHONIOENCODING
    1-8. v15モデルがload_models候補に存在
    1-9. v15新特徴量のバージョンゲート(>=15)
    1-10. odds変動特徴量フロー
    1-11. LGB+XGBアンサンブル
    2-1. 輸送距離計算コード
    2-2. コース改修フラグコード
    2-3. Kelly基準パラメータ
    2-4. drift_monitor存在
"""
import sys
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


# ===== PHASE 1: 過去バグ回帰テスト =====

def test_1_1_version_gate():
    """v15+でバージョンゲートが通ること"""
    from tools.predict_core import build_features
    import inspect
    src = inspect.getsource(build_features)
    assert "re.search" in src, "regex版バージョンゲートが未実装"
    assert "startswith(('v5'" not in src, "旧startswithゲートが残存"


def test_1_2_single_source_of_truth():
    """全スクリプトがpredict_coreを使用"""
    for fpath in ['tools/daily_predict.py', 'tools/race_auto_notify.py']:
        full = os.path.join(BASE_DIR, fpath)
        if not os.path.exists(full):
            continue
        with open(full, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'def build_features(' not in content, f"{fpath}に独自build_features"
        assert 'def predict_race(' not in content, f"{fpath}に独自predict_race"


def test_1_3_feature_count():
    """v15: LGB(145) <= features(150)"""
    from tools.predict_core import load_models
    md = load_models()
    feats = md.get('features', [])
    lgb = md.get('model')
    assert len(feats) >= 145, f"features数不足: {len(feats)}"
    if lgb and hasattr(lgb, 'num_feature'):
        n = lgb.num_feature()
        assert n <= len(feats), f"LGB({n}) > features({len(feats)})"


def test_1_4_dist_cat_5bins():
    """距離カテゴリ5bin(0-4)"""
    src_path = os.path.join(BASE_DIR, 'tools', 'predict_core.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "labels=[0, 1, 2, 3, 4]" in content, "dist_catが5binでない"


def test_1_5_no_stdout_rewrap():
    """race_auto_notify.pyにsys.stdout再ラップなし"""
    fpath = os.path.join(BASE_DIR, 'tools', 'race_auto_notify.py')
    if not os.path.exists(fpath):
        return
    with open(fpath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('#'):
                continue
            assert 'sys.stdout = io.TextIOWrapper' not in line.strip(), \
                f"line {i+1}: stdout再ラップ"


def test_1_6_env_gitignore():
    """.envが.gitignoreに含まれる"""
    gi = os.path.join(BASE_DIR, '.gitignore')
    with open(gi, 'r') as f:
        assert '.env' in f.read()


def test_1_7_bat_encoding():
    """全batにPYTHONIOENCODING設定"""
    bats = ['daily_predict.bat', 'daily_results.bat', 'weekly_report.bat',
            'race_auto_notify.bat', 'daily_premium_scrape.bat', 'friday_weekend_scrape.bat']
    for bat in bats:
        full = os.path.join(BASE_DIR, bat)
        if not os.path.exists(full):
            continue
        with open(full, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        assert 'PYTHONIOENCODING' in content, f"{bat}: PYTHONIOENCODING未設定"


def test_1_8_v15_in_load_models():
    """v15モデルがload_models候補に存在"""
    src_path = os.path.join(BASE_DIR, 'tools', 'predict_core.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'keiba_model_v15_central' in content, "v15がload_models候補に未登録"


def test_1_9_v15_feature_gate():
    """v15新特徴量のバージョンゲート(>=15)が存在"""
    src_path = os.path.join(BASE_DIR, 'tools', 'predict_core.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "int(_ver_num_v15.group(1)) >= 15" in content, "v15特徴量ゲートなし"


def test_1_10_odds_change_flow():
    """odds_change_rate計算関数が存在"""
    from tools.predict_core import compute_odds_change_features
    import inspect
    src = inspect.getsource(compute_odds_change_features)
    assert 'odds_change_rate' in src
    assert 'pop_rank_change' in src
    assert 'odds_sharp_drop' in src


def test_1_11_lgb_xgb_ensemble():
    """LGB+XGBアンサンブルがロードされる"""
    from tools.predict_core import load_models
    md = load_models()
    assert md.get('model') is not None, "LGBモデルなし"
    assert md.get('xgb_model') is not None, "XGBモデルなし"
    w = md.get('ensemble_weights', {})
    assert w.get('lgb', 0) > 0, "LGB重み=0"
    assert w.get('xgb', 0) > 0, "XGB重み=0"


# ===== PHASE 2: v15新機能テスト =====

def test_2_1_transport_distance():
    """輸送距離計算コードが存在"""
    src_path = os.path.join(BASE_DIR, 'tools', 'predict_core.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'transport_distance_km' in content
    assert 'is_long_transport' in content
    assert '_hav' in content or 'haversine' in content.lower()


def test_2_2_course_renovation():
    """コース改修フラグコードが存在"""
    src_path = os.path.join(BASE_DIR, 'tools', 'predict_core.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'course_renovated' in content
    assert 'post_renovation_flag' in content


def test_2_3_kelly_params():
    """Kelly基準: min_bet=100, max_bet=2000"""
    from tools.predict_core import calc_kelly_investment
    import inspect
    src = inspect.getsource(calc_kelly_investment)
    assert 'min_bet=100' in src
    assert 'max_bet=2000' in src


def test_2_4_drift_monitor():
    """drift_monitor.pyが存在"""
    assert os.path.exists(os.path.join(BASE_DIR, 'tools', 'drift_monitor.py'))


def test_2_5_profitable_patterns_data():
    """profitable_patterns.jsonが存在"""
    pp = os.path.join(BASE_DIR, 'data', 'profitable_patterns.json')
    assert os.path.exists(pp), "profitable_patterns.json不存在"
    assert os.path.getsize(pp) > 1000, "profitable_patterns.jsonが小さすぎる"


def test_2_6_upset_panel_code():
    """波乱度パネルコードが存在"""
    app_path = os.path.join(BASE_DIR, 'app.py')
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'upset_level' in content
    assert 'Lv4' in content or 'lv >= 4' in content.lower() or '_u_lv >= 4' in content


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {t.__name__} -- {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {t.__name__} -- {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
