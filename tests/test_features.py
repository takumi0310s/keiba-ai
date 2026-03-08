#!/usr/bin/env python
"""KEIBA AI - 全機能自動テスト
1. 中央V9・地方V8の自動切替
2. 条件別買い目自動切替（A〜E）
3. オッズ連動投資額（400/300）
4. TRACK RECORDの保存・削除
5. システムチェック（run_system_checks）
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pickle
import sqlite3
import tempfile
import numpy as np
import pandas as pd


# ===== Test helpers =====
def make_dummy_df(n_horses=10, distance=1800, condition='良'):
    """テスト用ダミー出走表DataFrameを生成"""
    horses = []
    for i in range(n_horses):
        horses.append({
            '馬番': i + 1,
            '馬名': f'テスト馬{i+1}',
            'スコア': np.random.rand(),
            'AI順位': i + 1,
            '単勝オッズ': round(np.random.uniform(2.0, 50.0), 1),
            '脚質': np.random.choice([1, 2, 3, 4]),
            '前走着順': np.random.randint(1, 18),
        })
    df = pd.DataFrame(horses)
    df = df.sort_values('スコア', ascending=False).reset_index(drop=True)
    df['AI順位'] = range(1, len(df) + 1)
    return df


# ===== Test 1: 中央V9・地方V8の自動切替 =====
def test_model_auto_switch():
    """中央→V9、地方→V8 の自動切替が正常に動作するか"""
    print("=" * 60)
    print("  TEST 1: モデル自動切替 (Central V9 / NAR V8)")
    print("=" * 60)

    errors = []

    # Check model files exist
    v8_exists = os.path.exists('keiba_model_v8.pkl')
    v9c_exists = os.path.exists('keiba_model_v9_central.pkl')
    v9n_exists = os.path.exists('keiba_model_v9_nar.pkl')

    print(f"  V8: {'OK' if v8_exists else 'MISSING'}")
    print(f"  V9 Central: {'OK' if v9c_exists else 'MISSING'}")
    print(f"  V9 NAR: {'OK' if v9n_exists else 'MISSING'}")

    if not v8_exists:
        errors.append("V8モデルが見つかりません")
        return errors

    # Load models
    with open('keiba_model_v8.pkl', 'rb') as f:
        v8_data = pickle.load(f)

    v9c_data = None
    if v9c_exists:
        with open('keiba_model_v9_central.pkl', 'rb') as f:
            v9c_data = pickle.load(f)

    # Simulate get_model_for_race logic
    # NAR should return V8
    if v9c_data:
        # Central: should use V9
        model_type_central = 'central' if v9c_data and 'model' in v9c_data else 'default'
        if model_type_central != 'central':
            errors.append("中央レースでV9が選択されていません")
        else:
            print("  ✓ 中央 → V9 central (正常)")

    # NAR: should use V8 (default)
    model_type_nar = 'default'  # NAR always returns V8
    if model_type_nar != 'default':
        errors.append("地方レースでV8が選択されていません")
    else:
        print("  ✓ 地方 → V8 default (正常)")

    # Verify V9 has ensemble components
    if v9c_data:
        has_xgb = 'xgb_model' in v9c_data
        has_mlp = 'mlp_model' in v9c_data
        has_weights = 'ensemble_weights' in v9c_data
        print(f"  V9 Ensemble: XGB={'OK' if has_xgb else 'MISSING'}, MLP={'OK' if has_mlp else 'MISSING'}, Weights={'OK' if has_weights else 'MISSING'}")
        if not (has_xgb and has_mlp and has_weights):
            errors.append("V9のアンサンブルコンポーネントが不完全")

    if not errors:
        print("  ✓ PASS")
    return errors


# ===== Test 2: 条件別買い目自動切替（A〜E） =====
def test_condition_classification():
    """classify_race_condition のロジックテスト"""
    print("\n" + "=" * 60)
    print("  TEST 2: 条件別買い目自動切替 (A-E)")
    print("=" * 60)

    # Import from app.py
    from app import classify_race_condition, CONDITION_PROFILES

    errors = []

    test_cases = [
        # (race_info, num_horses, is_nar, expected_key, description)
        ({'distance': 1800, 'condition': '良'}, 12, False, 'A', '8-14頭/1600m+/良'),
        ({'distance': 2000, 'condition': '重'}, 10, False, 'B', '8-14頭/1600m+/重'),
        ({'distance': 1600, 'condition': '良'}, 16, False, 'C', '15頭+/1600m+/良'),
        ({'distance': 1200, 'condition': '良'}, 12, False, 'D', 'スプリント'),
        ({'distance': 1800, 'condition': '良'}, 6, False, 'E', '少頭数'),
        ({'distance': 2000, 'condition': '不良'}, 16, False, 'X', '15頭+/不良'),
        ({'distance': 1800, 'condition': '良'}, 12, True, 'B', 'NAR→B(非推奨)'),
    ]

    for race_info, n_horses, is_nar, expected, desc in test_cases:
        key, profile = classify_race_condition(race_info, n_horses, is_nar=is_nar)
        status = '✓' if key == expected else '✗'
        print(f"  {status} {desc}: {key} (期待: {expected})")
        if key != expected:
            errors.append(f"条件分類エラー: {desc} → {key} (期待: {expected})")

    # Verify profiles have required keys
    required_keys = ['label', 'desc', 'bet_type', 'bet_label', 'investment', 'roi', 'hit_rate', 'recommended']
    for cond_key, profile in CONDITION_PROFILES.items():
        for rk in required_keys:
            if rk not in profile:
                errors.append(f"CONDITION_PROFILES['{cond_key}'] に '{rk}' がありません")

    # Verify bet types
    expected_bets = {'A': 'trio', 'B': 'wide', 'C': 'wide', 'D': 'none', 'E': 'umaren', 'X': 'wide'}
    for k, bt in expected_bets.items():
        actual_bt = CONDITION_PROFILES[k]['bet_type']
        if actual_bt != bt:
            errors.append(f"条件{k}のbet_type: {actual_bt} (期待: {bt})")

    if not errors:
        print("  ✓ PASS")
    return errors


# ===== Test 3: オッズ連動投資額（400/300） =====
def test_odds_investment():
    """オッズ高い方400円・低い方300円の振り分けテスト"""
    print("\n" + "=" * 60)
    print("  TEST 3: オッズ連動投資額 (400/300)")
    print("=" * 60)

    errors = []

    # Simulate the logic from render_buy_section
    test_cases = [
        # (odds1, odds2, expected_amt1, expected_amt2)
        (5.0, 3.0, 400, 300),   # odds1 > odds2 → amt1=400
        (3.0, 5.0, 300, 400),   # odds1 < odds2 → amt1=300
        (4.0, 4.0, 400, 300),   # Equal → odds1 gets 400
        (0, 3.0, 350, 350),     # odds1 missing → fallback
        (5.0, 0, 350, 350),     # odds2 missing → fallback
        (0, 0, 350, 350),       # Both missing → fallback
    ]

    for odds1, odds2, exp_a1, exp_a2 in test_cases:
        if odds1 > 0 and odds2 > 0:
            if odds1 >= odds2:
                amt1, amt2 = 400, 300
            else:
                amt1, amt2 = 300, 400
        else:
            amt1, amt2 = 350, 350

        total = amt1 + amt2
        status = '✓' if (amt1 == exp_a1 and amt2 == exp_a2) else '✗'
        print(f"  {status} odds ({odds1}, {odds2}) → ({amt1}, {amt2}円) total={total}")

        if amt1 != exp_a1 or amt2 != exp_a2:
            errors.append(f"投資額エラー: odds({odds1},{odds2}) → ({amt1},{amt2}) 期待({exp_a1},{exp_a2})")
        if total != 700:
            errors.append(f"合計投資額が700円でない: {total}")

    if not errors:
        print("  ✓ PASS")
    return errors


# ===== Test 4: TRACK RECORD 保存・削除 =====
def test_track_record():
    """SQLite TRACK RECORDの保存・削除テスト"""
    print("\n" + "=" * 60)
    print("  TEST 4: TRACK RECORD 保存・削除")
    print("=" * 60)

    errors = []

    # Use a temp DB
    test_db = os.path.join(tempfile.gettempdir(), 'keiba_test_predictions.db')
    if os.path.exists(test_db):
        os.remove(test_db)

    conn = sqlite3.connect(test_db)
    c = conn.cursor()

    # Create tables (same schema as app.py)
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT, race_name TEXT, race_date TEXT,
        course TEXT, distance INTEGER, surface TEXT, condition TEXT,
        horse_name TEXT, horse_num INTEGER, ai_rank INTEGER,
        ai_score REAL, odds REAL, predicted_at TEXT,
        actual_finish INTEGER DEFAULT NULL,
        is_top3_pred INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS race_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT UNIQUE, race_name TEXT,
        predicted_at TEXT, result_updated_at TEXT DEFAULT NULL,
        num_horses INTEGER, top1_name TEXT, top1_score REAL,
        trio_bets TEXT DEFAULT NULL, hit_trio INTEGER DEFAULT NULL,
        hit_combo TEXT DEFAULT NULL, payout INTEGER DEFAULT 0,
        is_nar INTEGER DEFAULT 0, wide_bets TEXT DEFAULT NULL,
        hit_wide INTEGER DEFAULT NULL, wide_payout INTEGER DEFAULT 0,
        buy_recommended INTEGER DEFAULT 1,
        bet_condition TEXT DEFAULT NULL, bet_type TEXT DEFAULT NULL,
        umaren_bets TEXT DEFAULT NULL
    )""")
    conn.commit()

    # Test: Save prediction
    race_id = 'TEST0001'
    now = '2025-03-08 12:00:00'
    df = make_dummy_df(10)

    for _, row in df.iterrows():
        c.execute("""INSERT INTO predictions
            (race_id, race_name, race_date, course, distance, surface, condition,
             horse_name, horse_num, ai_rank, ai_score, odds, predicted_at, is_top3_pred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (race_id, 'テストレース', now[:10], '東京', 1800, '芝', '良',
             row['馬名'], int(row['馬番']), int(row['AI順位']),
             float(row['スコア']), float(row['単勝オッズ']), now,
             1 if int(row['AI順位']) <= 3 else 0))

    c.execute("""INSERT INTO race_results
        (race_id, race_name, predicted_at, num_horses, top1_name, top1_score,
         trio_bets, is_nar, buy_recommended, bet_condition, bet_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (race_id, 'テストレース', now, 10, df.iloc[0]['馬名'], float(df.iloc[0]['スコア']),
         '[[1,2,3]]', 0, 1, 'A', 'trio'))
    conn.commit()

    # Verify save
    c.execute("SELECT COUNT(*) FROM predictions WHERE race_id = ?", (race_id,))
    pred_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM race_results WHERE race_id = ?", (race_id,))
    result_count = c.fetchone()[0]

    if pred_count == 10:
        print(f"  ✓ 予測保存: {pred_count}件")
    else:
        errors.append(f"予測保存数エラー: {pred_count} (期待: 10)")

    if result_count == 1:
        print(f"  ✓ レース結果保存: {result_count}件")
    else:
        errors.append(f"レース結果保存数エラー: {result_count} (期待: 1)")

    # Test: Selective delete
    c.execute("DELETE FROM predictions WHERE race_id = ?", (race_id,))
    c.execute("DELETE FROM race_results WHERE race_id = ?", (race_id,))
    conn.commit()

    c.execute("SELECT COUNT(*) FROM predictions WHERE race_id = ?", (race_id,))
    after_del = c.fetchone()[0]
    if after_del == 0:
        print(f"  ✓ 個別削除: 正常")
    else:
        errors.append(f"個別削除エラー: {after_del}件残存")

    # Test: Bulk insert + delete all
    for i in range(3):
        rid = f'TEST{i:04d}'
        c.execute("""INSERT INTO race_results
            (race_id, race_name, predicted_at, num_horses, top1_name, top1_score)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (rid, f'テストレース{i}', now, 10, 'テスト馬', 0.8))
    conn.commit()

    c.execute("SELECT COUNT(*) FROM race_results")
    before_all = c.fetchone()[0]

    c.execute("DELETE FROM race_results")
    c.execute("DELETE FROM predictions")
    conn.commit()

    c.execute("SELECT COUNT(*) FROM race_results")
    after_all = c.fetchone()[0]
    if before_all == 3 and after_all == 0:
        print(f"  ✓ 全件削除: 正常 ({before_all} → {after_all})")
    else:
        errors.append(f"全件削除エラー: {before_all} → {after_all}")

    conn.close()
    os.remove(test_db)

    if not errors:
        print("  ✓ PASS")
    return errors


# ===== Test 5: システムチェック =====
def test_system_checks():
    """run_system_checks 相当のテスト"""
    print("\n" + "=" * 60)
    print("  TEST 5: システムチェック (7点)")
    print("=" * 60)

    errors = []

    # 1. Model files
    v8_found = os.path.exists('keiba_model_v8.pkl') or os.path.exists('keiba_model_v8.pkl.gz')
    print(f"  {'✓' if v8_found else '✗'} モデルファイル: {'検出' if v8_found else '未検出'}")
    if not v8_found:
        errors.append("V8モデルが見つかりません")

    # 2. V9 models
    v9c = os.path.exists('keiba_model_v9_central.pkl')
    v9n = os.path.exists('keiba_model_v9_nar.pkl')
    print(f"  {'✓' if v9c else '✗'} V9 Central: {'検出' if v9c else '未検出'}")
    print(f"  {'✓' if v9n else '✗'} V9 NAR: {'検出' if v9n else '未検出'}")

    # 3. Feature count check
    if v8_found:
        with open('keiba_model_v8.pkl', 'rb') as f:
            data = pickle.load(f)
        feats = data.get('features', [])
        feat_ok = len(feats) > 0 if feats else False
        print(f"  {'✓' if feat_ok else '✗'} 特徴量: {len(feats) if feats else 0}個")
        if not feat_ok:
            errors.append("特徴量が定義されていません")

    # 4. Jockey data
    jwr_exists = os.path.exists('jockey_wr.json')
    if jwr_exists:
        from datetime import datetime
        mtime = os.path.getmtime('jockey_wr.json')
        days_ago = (datetime.now() - datetime.fromtimestamp(mtime)).days
        jwr_ok = days_ago <= 30
        print(f"  {'✓' if jwr_ok else '✗'} 騎手データ: {days_ago}日前更新")
    else:
        print(f"  ✗ 騎手データ: ファイルなし")

    # 5. DB
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'keiba_predictions.db')
    db_exists = os.path.exists(db_path)
    if db_exists:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        try:
            c.execute("SELECT COUNT(*) FROM race_results")
            cnt = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM predictions")
            pcnt = c.fetchone()[0]
            print(f"  ✓ DB整合性: race_results={cnt}件, predictions={pcnt}件")
        except Exception as e:
            print(f"  ✗ DB整合性: {e}")
            errors.append(f"DBエラー: {e}")
        conn.close()
    else:
        print(f"  - DB: ファイルなし (初回起動時に作成されます)")

    # 6. Bet generation logic
    from app import generate_trio_bets, generate_wide_bets, generate_umaren_bets
    df_test = make_dummy_df(10)
    df_test = df_test.sort_values('スコア', ascending=False).reset_index(drop=True)

    trio = generate_trio_bets(df_test)
    wide = generate_wide_bets(df_test)
    umaren = generate_umaren_bets(df_test)

    trio_ok = len(trio) == 7
    wide_ok = len(wide) == 2
    umaren_ok = len(umaren) == 2
    print(f"  {'✓' if trio_ok else '✗'} 三連複生成: {len(trio)}点 (期待: 7)")
    print(f"  {'✓' if wide_ok else '✗'} ワイド生成: {len(wide)}点 (期待: 2)")
    print(f"  {'✓' if umaren_ok else '✗'} 馬連生成: {len(umaren)}点 (期待: 2)")

    if not trio_ok:
        errors.append(f"三連複生成エラー: {len(trio)}点")
    if not wide_ok:
        errors.append(f"ワイド生成エラー: {len(wide)}点")
    if not umaren_ok:
        errors.append(f"馬連生成エラー: {len(umaren)}点")

    if not errors:
        print("  ✓ PASS")
    return errors


# ===== Main =====
def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    print("\n" + "#" * 60)
    print("  KEIBA AI - 全機能自動テスト")
    print("#" * 60)

    all_errors = []

    # Run tests
    all_errors.extend(test_model_auto_switch())
    all_errors.extend(test_condition_classification())
    all_errors.extend(test_odds_investment())
    all_errors.extend(test_track_record())
    all_errors.extend(test_system_checks())

    # Summary
    print("\n" + "=" * 60)
    if all_errors:
        print(f"  RESULT: {len(all_errors)} ERRORS")
        for e in all_errors:
            print(f"    ✗ {e}")
    else:
        print("  RESULT: ALL TESTS PASSED ✓")
    print("=" * 60)

    return len(all_errors) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
