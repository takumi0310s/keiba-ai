"""
特徴量チェックツール — 過去レースURLで特徴量を取得し、ゼロ列・NaN列・取得数を表示。
引数なしで実行可能（デフォルトで直近の過去レースIDを使用）。

Usage:
    python tools/check_features.py
    python tools/check_features.py https://race.netkeiba.com/race/shutuba.html?race_id=202505030811
    python tools/check_features.py 202505030811
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.predict_core import (
    load_models, parse_shutuba, build_features, load_feature_lookups,
    fetch_realtime_odds, fetch_jra_and_weather, get_horse_stats,
    set_horse_defaults, apply_horse_stats,
)

# デフォルトの過去レースID（2025中山11R）
DEFAULT_RACE_ID = "202506010111"


def extract_race_id(arg):
    m = re.search(r'race_id=(\d+)', arg)
    if m:
        return m.group(1)
    if re.match(r'^\d{12}$', arg):
        return arg
    return None


def main():
    if len(sys.argv) > 1:
        race_id = extract_race_id(sys.argv[1])
        if not race_id:
            print(f"[ERROR] レースIDを抽出できません: {sys.argv[1]}")
            sys.exit(1)
    else:
        race_id = DEFAULT_RACE_ID

    print(f"=== 特徴量チェック ===")
    print(f"Race ID: {race_id}\n")

    # モデルロード
    print("[1/5] モデルロード...")
    model_data = load_models()
    if model_data['model'] is None:
        print("[ERROR] モデルが見つかりません")
        sys.exit(1)
    version = model_data.get('version', '?')
    is_live = model_data.get('is_live', False)
    print(f"  Version: {version}, Live: {is_live}")

    # feature_lookups
    print("[2/5] feature_lookups ロード...")
    lookups = load_feature_lookups()
    print(f"  Keys: {len(lookups) if lookups else 0}")

    # 出馬表取得
    print("[3/5] 出馬表取得...")
    try:
        race_info, horses = parse_shutuba(race_id)
    except Exception as e:
        print(f"[ERROR] 出馬表取得失敗: {e}")
        sys.exit(1)
    print(f"  レース: {race_info.get('race_name', '?')} / {race_info.get('course', '?')} "
          f"{race_info.get('distance', '?')}m / 頭数: {len(horses)}")

    # 馬情報補完
    print("[4/5] 馬情報取得（horse_stats）...")
    for h in horses:
        set_horse_defaults(h)
        try:
            stats = get_horse_stats(
                h.get('horse_id', ''),
                race_info.get('distance', 0),
                race_info.get('surface', '芝'),
                race_info.get('course', ''),
            )
            apply_horse_stats(h, stats, race_info)
        except Exception:
            pass

    # オッズ（過去レースなので取得できなくてもOK）
    odds_dict = {}
    try:
        odds_dict = fetch_realtime_odds(race_id) or {}
    except Exception:
        pass

    # JRA馬場・天候（過去なので取得できなくてもOK）
    jra_info, weather_info = None, None
    try:
        jra_info, weather_info = fetch_jra_and_weather(race_info.get('course', ''))
    except Exception:
        pass

    # 特徴量構築
    print("[5/5] build_features()...")
    try:
        df = build_features(
            horses, race_info, model_data,
            race_id=race_id,
            odds_dict=odds_dict,
            jra_track_info=jra_info,
            weather_info=weather_info,
        )
    except Exception as e:
        print(f"[ERROR] build_features 失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 結果表示
    expected = model_data.get('features', [])
    n_expected = len(expected) if expected else '?'
    n_actual = len(df.columns)

    print(f"\n{'='*60}")
    print(f"特徴量数: {n_actual} / 期待: {n_expected}")
    print(f"馬数: {len(df)}")

    # ゼロ列
    zero_cols = [c for c in df.columns if (df[c] == 0).all()]
    print(f"\n全ゼロ列: {len(zero_cols)}個")
    if zero_cols:
        for c in zero_cols:
            print(f"  - {c}")

    # NaN列
    nan_cols = [c for c in df.columns if df[c].isna().any()]
    print(f"\nNaN含む列: {len(nan_cols)}個")
    if nan_cols:
        for c in nan_cols:
            n_nan = df[c].isna().sum()
            print(f"  - {c} ({n_nan}/{len(df)} NaN)")

    # 期待特徴量で不足しているもの
    if expected:
        missing = [f for f in expected if f not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            print(f"\n不足特徴量: {len(missing)}個")
            for f in missing:
                print(f"  - {f}")
        if extra:
            print(f"\n余分な列: {len(extra)}個")
            for c in extra:
                print(f"  - {c}")

    # 全ゼロ + NaN 警告
    problem_count = len(zero_cols) + len(nan_cols)
    print(f"\n{'='*60}")
    if problem_count >= 5:
        print(f"⚠ 警告: 問題のある列が{problem_count}個あります。原因を調査してください。")
    elif problem_count > 0:
        print(f"△ {problem_count}個の列に注意（許容範囲内の可能性あり）")
    else:
        print("✓ 全特徴量が正常に取得されています")


if __name__ == "__main__":
    main()
