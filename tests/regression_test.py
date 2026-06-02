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
    with open(gi_path, 'r', encoding='utf-8') as f:
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


COURSE_NAMES = ['中山', '阪神', '福島', '東京', '京都', '中京', '小倉', '新潟', '函館', '札幌']


def _make_dummy_horse(i):
    """predict_core.build_featuresに渡せる最小限のダミー馬データ"""
    return {
        '馬番': i + 1,
        '馬名': f'テスト{i}',
        '馬齢': 4,
        '性別': '牡',
        '性別_enc': 0,
        '性齢': '牡4',
        '斤量': 56.0,
        '騎手名': '佐藤',
        '騎手': '佐藤',
        '騎手勝率': 0.1,
        '調教師': '美浦厩舎',
        '父': '',
        '母の父': '',
        '所属地': '美浦',
        '単勝オッズ': 10.0,
        '人気': i + 1,
        '枠番': (i // 2) + 1,
        '馬体重': 480,
        '場体重増減': 0,
        '前走着順': 5,
        '前走オッズ': 15.0,
        '前走人気': 8,
        '上がり3F': 35.5,
        '通過順平均': 8.0,
        '通過順4': 8,
        '前走間隔': 30,
        'horse_id_val': 0,
        '競馬場コード_enc': 4,
        '芝ダート_enc': 0,
        '馬場状態_enc': 0,
        '距離(m)': 1600,
    }


def test_all_courses_feature_build():
    """全10競馬場でbuild_featuresがエラーなく動作すること"""
    from tools.predict_core import load_models, build_features
    md = load_models()
    assert md.get('model') is not None, "モデルロード失敗"

    horses = [_make_dummy_horse(i) for i in range(10)]

    for course in COURSE_NAMES:
        race_info = {
            'course': course, 'distance': 1600, 'surface': '芝',
            'condition': '良', 'race_num': 11, 'num_horses': 10,
        }
        try:
            df = build_features(horses, race_info, md)
        except Exception as e:
            raise AssertionError(f"{course}でbuild_features失敗: {e}")
        assert df is not None and len(df) == 10, f"{course}: 行数不一致"


def test_app_load_model_latest_version_registered():
    """app.pyが現行最新モデル(v15想定)を検出できること"""
    import importlib.util, glob
    app_path = os.path.join(BASE_DIR, 'app.py')
    if not os.path.exists(app_path):
        return

    # _model_version_keyロジックの単体検証（Streamlit非依存）
    import re as _re
    def _key(fname):
        m = _re.search(r'keiba_model_v(\d+)([a-z]?)_central(?:_live)?', fname)
        if not m:
            return -1
        digits = m.group(1)
        suffix = m.group(2)
        if len(digits) == 2:
            digits = digits + '0'
        return int(digits) * 10 + (ord(suffix) - ord('a') + 1 if suffix else 0)

    # app.py内の_model_version_keyが存在すること
    with open(app_path, 'r', encoding='utf-8') as f:
        app_src = f.read()
    assert '_model_version_key' in app_src, "_model_version_keyが未定義"
    assert '_discover_latest_model' in app_src, "_discover_latest_modelが未定義"

    # 実在するモデルファイルの中で最大バージョンが選ばれること
    cands = glob.glob(os.path.join(BASE_DIR, 'keiba_model_v*_central*.pkl.gz'))
    assert cands, "モデルファイルが1つもない"
    cands.sort(key=lambda p: _key(os.path.basename(p)), reverse=True)
    top = os.path.basename(cands[0])
    # v15以上のバージョン番号が存在するはず（v15登録済みの回帰防止）
    assert _key(top) >= 1500, f"最新モデルが v15 未満: {top}"


def test_predict_core_column_count_matches_model():
    """build_featuresの生成列数 >= モデルが期待する特徴量数"""
    from tools.predict_core import load_models, build_features
    md = load_models()
    feats = md.get('features') or []
    if not feats:
        return  # features listが無いモデルはスキップ

    horses = [_make_dummy_horse(i) for i in range(8)]
    race_info = {'course': '中山', 'distance': 1600, 'surface': '芝',
                 'condition': '良', 'race_num': 11, 'num_horses': 8}
    df = build_features(horses, race_info, md)

    missing = [c for c in feats if c not in df.columns]
    assert not missing, f"build_featuresに不足列 {len(missing)}個: {missing[:5]}"


def test_4models_present():
    """主要モデル(LGB/XGB)がロードされること。FT/IRはv15以降で任意。"""
    from tools.predict_core import load_models
    md = load_models()
    assert md.get('model') is not None, "LGBモデルがない"
    assert md.get('xgb_model') is not None, "XGBモデルがない"
    # FT/IRはv15以降のshipでNoneのことがあるためwarnのみ
    if md.get('ft_model_state') is None:
        print("[warn] ft_model_state is None (v15以降では許容)")
    if md.get('ir_model_state') is None:
        print("[warn] ir_model_state is None (v15以降では許容)")


def test_app_auto_detects_latest_model():
    """app.py が _discover_latest_model() によるglob自動検出を使っていること。
    新モデル投入時の app.py 書き換え漏れで旧モデルがロードされる過去バグの再発防止。"""
    app_path = os.path.join(BASE_DIR, 'app.py')
    if not os.path.exists(app_path):
        return
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert '_discover_latest_model' in content, \
        "app.py に _discover_latest_model() が無い — ハードコードの可能性"
    assert 'keiba_model_v*_central' in content, \
        "app.py にglobパターン 'keiba_model_v*_central' が無い"


def test_features_v15_new_imported_by_predict_core():
    """tools/predict_core.py が train/features_v15_new.py の関数を import しているか。
    新特徴量モジュールが predict_core から呼ばれず、列が生成されなかった過去バグの再発防止。"""
    pc_path = os.path.join(BASE_DIR, 'tools', 'predict_core.py')
    with open(pc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'features_v15_new' in content, \
        "predict_core.py が features_v15_new を import していない"
    assert 'get_v15_new_features' in content, \
        "get_v15_new_features が predict_core.py に存在しない"


def test_scraper_guard_present():
    """netkeiba系スクレイパーに scraper_guard が入っていること。"""
    scrapers = [
        'tools/bulk_scrape_upset.py',
        'tools/bulk_scrape_history.py',
        'tools/scrape_master_index.py',
        'tools/scrape_speed_index.py',
        'tools/scrape_race_review.py',
        'tools/scrape_master_course.py',
    ]
    for rel in scrapers:
        p = os.path.join(BASE_DIR, rel)
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            src = f.read()
        assert 'scraper_guard' in src, f"{rel} に scraper_guard が未挿入"


def test_odds_base_perday_csv():
    """予測時の odds_base がper-day CSV化されていること。"""
    from tools.predict_core import _odds_base_csv_path
    p = _odds_base_csv_path('20260412')
    assert 'odds_base_20260412.csv' in p, f"unexpected path: {p}"


# 注: 戦略⑦ 案C フィルタは週末リファクタで race_auto_notify.py の inline から
# strategy_filters.evaluate_bet_decision() へ抽出された (behavior 不変・2026-05-30 ライブ稼働済)。
# 以下のテストは「実装場所の文字列 grep」ではなく、移動先関数の「挙動」を直接検証する
# (位置非依存・より強い回帰防止)。順序: distance≤1000→06特別→京都→C4→E→B→X。


def test_strategy_7_planC_kyoto_filter_present():
    """戦略⑦ 案 C: 京都 除外 filter が evaluate_bet_decision に実装されていること (P0-2 5/17 適用)."""
    from tools.strategy_filters import evaluate_bet_decision
    # 京都 + 非重賞 → 見送り、 理由=strategy_7_kyoto_p0_2_5_17
    should_bet, reason = evaluate_bet_decision(
        race_name='3歳未勝利', course='京都', distance=1800, cond_key='C')
    assert should_bet is False, "京都 非重賞が見送りにならない (案 C filter 欠落)"
    assert reason == 'strategy_7_kyoto_p0_2_5_17', f"京都 skip 理由が想定外: {reason}"


def test_strategy_7_planC_condition_X_filter_present():
    """戦略⑦ 案 C: 条件 X 除外 filter が evaluate_bet_decision に実装されていること (P0-2 5/17 適用)."""
    from tools.strategy_filters import evaluate_bet_decision
    # 条件X + 非重賞 (非京都) → 見送り、 理由=strategy_7_cond_X_p0_2_5_17
    should_bet, reason = evaluate_bet_decision(
        race_name='4歳以上1勝クラス', course='東京', distance=1800, cond_key='X')
    assert should_bet is False, "条件X 非重賞が見送りにならない (案 C filter 欠落)"
    assert reason == 'strategy_7_cond_X_p0_2_5_17', f"条件X skip 理由が想定外: {reason}"


def test_strategy_7_planC_excludes_kyoto_normal_race():
    """戦略⑦ 案 C: 京都 + 非重賞 で skip (should_bet=False) されること."""
    from tools.strategy_filters import evaluate_bet_decision
    should_bet, _ = evaluate_bet_decision(
        race_name='4歳以上2勝クラス', course='京都', distance=2000, cond_key='C')
    assert should_bet is False, "京都 平場レースが skip されていない"


def test_strategy_7_planC_excludes_condition_X():
    """戦略⑦ 案 C: 条件 X + 非重賞 で skip (should_bet=False) されること."""
    from tools.strategy_filters import evaluate_bet_decision
    should_bet, _ = evaluate_bet_decision(
        race_name='4歳以上1勝クラス', course='中京', distance=1600, cond_key='X')
    assert should_bet is False, "条件X 平場レースが skip されていない"


def test_strategy_7_planC_preserves_other_courses():
    """戦略⑦ 案 C: 中京/東京/阪神/新潟/福島 の非重賞・通常条件は skip されないこと.
    docs/P0_2_EXTENSION_DESIGN_2026_05_16.md §1.2: 中京 ROI 107.05% (positive) は除外しない.
    """
    from tools.strategy_filters import evaluate_bet_decision
    for course in ['中京', '東京', '阪神', '新潟', '福島']:
        # cond_key='C'・1900m (≤1000/特別/京都/C4(condA)/E/B/X いずれにも該当しない) → 買い
        should_bet, reason = evaluate_bet_decision(
            race_name='3歳1勝クラス', course=course, distance=1900, cond_key='C')
        assert should_bet is True, \
            f"{course} の通常レースが誤って skip された (reason={reason}、 案 C は 京都 のみ除外)"


def test_strategy_7_planC_graded_race_protected():
    """戦略⑦ 案 C: G1/G2/G3 (重賞) は 京都/条件 X でも skip されないこと.
    5/17 ヴィクトリアマイル (G1) は案 C 影響 0、 京都/条件X に重賞があっても G1 は通す.
    """
    from tools.strategy_filters import evaluate_bet_decision
    # 京都 + G1 → 買い (Graded 保護)
    should_bet_kyoto, _ = evaluate_bet_decision(
        race_name='天皇賞(G1)', course='京都', distance=2000, cond_key='C')
    assert should_bet_kyoto is True, "京都 filter で Graded race を保護していない"
    # 条件X + G1 → 買い (Graded 保護)
    should_bet_x, _ = evaluate_bet_decision(
        race_name='ヴィクトリアマイル(G1)', course='東京', distance=1600, cond_key='X')
    assert should_bet_x is True, "条件X filter で Graded race を保護していない"


def test_payout_integrity_suite():
    """tests/test_payout_integrity.py の4テストが全PASSすること (4/23 payout=0バグ再発防止)"""
    import subprocess
    fp = os.path.join(BASE_DIR, 'tests', 'test_payout_integrity.py')
    assert os.path.exists(fp), 'test_payout_integrity.py 不在'
    r = subprocess.run([sys.executable, fp], capture_output=True, text=True, timeout=60)
    out = r.stdout + r.stderr
    assert r.returncode == 0, f'payout integrity FAIL:\n{out[-500:]}'
    assert '0 failed' in out or '4 passed' in out, f'payout integrity 不完全:\n{out[-300:]}'


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
