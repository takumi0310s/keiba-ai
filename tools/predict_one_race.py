"""
1レース指定で予測を実行するラッパー v3 (4要素戻り値対応)
"""
import sys
import os
# tools/ 内なので、tools/ 自体を sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 親ディレクトリも追加 (BASE_DIR 用)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows cp932 対策
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import predict_core

def predict_one_race(race_id):
    """1レースだけ予測実行"""
    print(f'=== 1レース予測: {race_id} ===\n')
    
    # モデルロード
    model_data = predict_core.load_models()
    if model_data['model'] is None:
        print('[NG] モデルロード失敗')
        return None
    print(f'[OK] モデル: {model_data.get("version", "unknown")}')
    print(f'     特徴量数: {len(model_data.get("features", []))}\n')
    
    # 出馬表取得 (4要素戻り値!)
    print('1. 出馬表取得...')
    race_name, horses, horse_ids, rinfo = predict_core.parse_shutuba(race_id)
    if not horses:
        print('[NG] 出馬表取得失敗')
        return None
    print(f'   [OK] {race_name}: {len(horses)}頭')
    print(f'        {rinfo.get("course", "?")}{rinfo.get("race_num", "?")}R, {rinfo.get("distance", "?")}m {rinfo.get("surface", "?")}, {rinfo.get("condition", "?")}')
    print()
    
    # オッズ取得
    print('2. オッズ取得...')
    try:
        odds_full = predict_core.fetch_realtime_odds_full(race_id)
        odds_dict = {u: v['odds'] for u, v in odds_full.items()} if odds_full else {}
        if odds_full:
            predict_core.save_odds_base(race_id, odds_full)
            print(f'   [OK] オッズ取得: {len(odds_full)}頭')
        else:
            print('   [WARN] オッズ未確定')
            odds_dict = {}
    except Exception as e:
        print(f'   [WARN] オッズ取得失敗: {e}')
        odds_dict = {}
    print()
    
    # JRA + 天候
    print('3. JRA & 天候...')
    try:
        jra_info, weather_info = predict_core.fetch_jra_and_weather(rinfo.get('course', ''))
        print(f'   [OK]')
    except Exception:
        jra_info, weather_info = {}, {}
        print('   [WARN] スキップ')
    print()
    
    # 各馬成績
    print('4. 各馬成績取得...')
    for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
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
        if i < len(horses) - 1:
            time.sleep(0.3)
    print(f'   [OK] {len(horses)}頭分の成績取得完了\n')
    
    # 特徴量構築
    print('5. 特徴量構築...')
    try:
        df = predict_core.build_features(
            horses, rinfo, model_data,
            race_id=race_id, odds_dict=odds_dict,
            jra_track_info=jra_info, weather_info=weather_info
        )
        print(f'   [OK] {df.shape[0]}行 x {df.shape[1]}列\n')
    except Exception as e:
        print(f'   [NG] 特徴量構築失敗: {e}')
        import traceback
        traceback.print_exc()
        return None
    
    # JRDB マージ
    print('6. JRDB特徴量マージ...')
    try:
        from jrdb_features import merge_jrdb_predict_features
        df = merge_jrdb_predict_features(df, race_id)
        print('   [OK]\n')
    except Exception as e:
        print(f'   [WARN] スキップ: {e}\n')
    
    # 予測
    print('7. 予測実行...')
    try:
        odds_available = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        result = predict_core.predict_race(df, model_data, odds_available, race_info=rinfo)
        result = result.sort_values('スコア', ascending=False).reset_index(drop=True)
        print('   [OK] 予測完了\n')
    except Exception as e:
        print(f'   [NG] 予測失敗: {e}')
        import traceback
        traceback.print_exc()
        return None
    
    return result, race_name, rinfo

if __name__ == '__main__':
    rid = sys.argv[1] if len(sys.argv) > 1 else '202605020211'
    res = predict_one_race(rid)
    if res:
        result, race_name, rinfo = res
        print(f'=== {race_name} 予測結果 TOP8 ===')
        cols = ['馬番', '馬名', 'スコア']
        for c in cols:
            if c not in result.columns:
                cols.remove(c)
        if cols:
            print(result[cols].head(8).to_string(index=False))
        else:
            print(result.head(8).to_string(index=False))
