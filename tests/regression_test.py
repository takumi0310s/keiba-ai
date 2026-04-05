"""回帰テスト — 過去発見バグの再発防止

Usage:
    python -m pytest tests/regression_test.py -v
    python tests/regression_test.py  # pytest不要な直接実行も可

テスト項目:
    1. バージョンゲート: v5〜v99で基本特徴量ブロックが有効
    2. 予測ロジック一元化: 全スクリプトがpredict_core.pyを使用
    3. 特徴量数一致: モデルのfeatures数とLGB/XGB/FT/IRの入力次元
    4. 距離カテゴリbin数: 学習時5bin(0-4)と予測時一致
    5. stdout再ラップなし: race_auto_notify.pyでクラッシュ防止
    6. .envがgitignore対象
"""
import sys
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def test_version_gate_future_proof():
    """v15, v20, v99でもバージョンゲートが通ること"""
    from tools.predict_core import build_features
    import inspect
    src = inspect.getsource(build_features)
    # Must NOT use startswith for version check
    assert "startswith(('v5'" not in src, "startswith版のバージョンゲートが残っている"
    # Must use regex-based version extraction
    assert "re.search" in src, "regex版のバージョンゲートが実装されていない"


def test_predict_core_is_single_source():
    """全スクリプトがpredict_coreのbuild_features/predict_raceを使用"""
    files_to_check = [
        'tools/daily_predict.py',
        'tools/race_auto_notify.py',
    ]
    for fpath in files_to_check:
        full = os.path.join(BASE_DIR, fpath)
        if not os.path.exists(full):
            continue
        with open(full, 'r', encoding='utf-8') as f:
            content = f.read()
        # Should import from predict_core, not define own
        assert 'def build_features(' not in content, f"{fpath} has its own build_features"
        assert 'def predict_race(' not in content, f"{fpath} has its own predict_race"


def test_app_uses_core_build_features():
    """app.pyがpredict_coreのbuild_featuresを使用"""
    app_path = os.path.join(BASE_DIR, 'app.py')
    if not os.path.exists(app_path):
        return  # Streamlit Cloud等で不在の場合スキップ
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert '_core_build_features' in content or 'build_features as' in content, \
        "app.pyがpredict_coreのbuild_featuresをインポートしていない"


def test_model_feature_count_consistency():
    """モデルのfeatures数とLGB/XGBの入力次元が整合"""
    from tools.predict_core import load_models
    md = load_models()
    feats = md.get('features', [])
    lgb = md.get('model')
    xgb_m = md.get('xgb_model')

    assert len(feats) > 0, "features list is empty"

    if lgb and hasattr(lgb, 'num_feature'):
        n_lgb = lgb.num_feature()
        assert n_lgb <= len(feats), f"LGB({n_lgb}) > features({len(feats)})"

    if xgb_m and hasattr(xgb_m, 'num_features'):
        n_xgb = xgb_m.num_features()
        assert n_xgb <= len(feats), f"XGB({n_xgb}) > features({len(feats)})"


def test_dist_cat_5bins():
    """距離カテゴリが5bin(0-4)で定義されていること"""
    src_path = os.path.join(BASE_DIR, 'tools', 'predict_core.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Must have 5 labels [0,1,2,3,4]
    assert "labels=[0, 1, 2, 3, 4]" in content or "labels=[0,1,2,3,4]" in content, \
        "dist_catが5binでない"


def test_no_stdout_rewrap_in_notify():
    """race_auto_notify.pyにsys.stdout再ラップがないこと"""
    fpath = os.path.join(BASE_DIR, 'tools', 'race_auto_notify.py')
    if not os.path.exists(fpath):
        return
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Must NOT have direct sys.stdout = io.TextIOWrapper
    lines = content.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        assert 'sys.stdout = io.TextIOWrapper' not in stripped, \
            f"line {i+1}: sys.stdout再ラップが残っている（クラッシュ原因）"


def test_env_in_gitignore():
    """.envが.gitignoreに含まれていること"""
    gi_path = os.path.join(BASE_DIR, '.gitignore')
    assert os.path.exists(gi_path), ".gitignore not found"
    with open(gi_path, 'r') as f:
        content = f.read()
    assert '.env' in content, ".envが.gitignoreにない"


def test_bat_has_encoding_vars():
    """race_auto_notify.batにPYTHONIOENCODINGが設定されていること"""
    bat_path = os.path.join(BASE_DIR, 'race_auto_notify.bat')
    if not os.path.exists(bat_path):
        return
    with open(bat_path, 'r') as f:
        content = f.read()
    assert 'PYTHONIOENCODING' in content, "PYTHONIOENCODINGが未設定"
    assert 'PYTHONUNBUFFERED' in content, "PYTHONUNBUFFEREDが未設定"


def test_4models_present():
    """4モデル(LGB/XGB/FT/IR)が全てロードされること"""
    from tools.predict_core import load_models
    md = load_models()
    assert md.get('model') is not None, "LGBモデルがない"
    assert md.get('xgb_model') is not None, "XGBモデルがない"
    assert md.get('ft_model_state') is not None, "FTモデルがない"
    assert md.get('ir_model_state') is not None, "IRモデルがない"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {t.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {t.__name__} — {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
