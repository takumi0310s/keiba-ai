"""
特徴量チェックツール — 予測パイプラインの総合診断。

機能:
  1. 過去/当日レースURLで特徴量取得 → ゼロ列・NaN列・取得数を表示
  2. モデル学習時特徴量と予測時特徴量の一致チェック
  3. data/フォルダの未使用jrdb_*.csv検出
  4. 4モデルアンサンブル（LGB+XGB+FT+IR）の状態確認

Usage:
    python tools/check_features.py
    python tools/check_features.py 202606030301
    python tools/check_features.py --audit
"""
import sys
import os
import re
import io
import glob

# Windows cp932対策
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from tools.predict_core import (
    load_models, parse_shutuba, build_features, load_feature_lookups,
    fetch_realtime_odds, fetch_jra_and_weather, get_horse_stats,
    set_horse_defaults, apply_horse_stats,
)

DEFAULT_RACE_ID = "202506010111"

# jrdb_features.pyで使用されるJRDB CSVファイル
USED_JRDB_FILES = {
    'jrdb_kyi.csv', 'jrdb_sed.csv', 'jrdb_tyb.csv', 'jrdb_cha.csv',
    'jrdb_jo.csv', 'jrdb_kta.csv', 'jrdb_ze.csv', 'jrdb_kka.csv',
    'jrdb_oz.csv', 'jrdb_paci.csv', 'jrdb_sr.csv', 'jrdb_skb.csv',
    'jrdb_joa.csv',
}


def extract_race_id(arg):
    m = re.search(r'race_id=(\d+)', arg)
    if m:
        return m.group(1)
    if re.match(r'^\d{12}$', arg):
        return arg
    return None


def audit_jrdb_files():
    """data/フォルダの未使用jrdb_*.csv検出"""
    print(f"\n{'='*60}")
    print("JRDB CSV ファイル監査")
    print(f"{'='*60}")
    data_dir = os.path.join(BASE_DIR, 'data')
    all_jrdb = sorted(glob.glob(os.path.join(data_dir, 'jrdb_*.csv')))

    used = []
    unused = []
    for path in all_jrdb:
        fname = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if fname in USED_JRDB_FILES:
            used.append((fname, size_mb))
        else:
            unused.append((fname, size_mb))

    print(f"\n使用中: {len(used)}ファイル")
    for fname, sz in used:
        print(f"  OK  {fname} ({sz:.1f}MB)")

    print(f"\n未使用: {len(unused)}ファイル")
    for fname, sz in unused:
        print(f"  --  {fname} ({sz:.1f}MB)")

    return unused


def audit_model_ensemble(model_data):
    """4モデルアンサンブル状態確認"""
    print(f"\n{'='*60}")
    print("モデルアンサンブル状態")
    print(f"{'='*60}")

    ew = model_data.get('ensemble_weights', {})
    print(f"\n  Ensemble weights: {ew}")
    print(f"  Ensemble type: {model_data.get('ensemble_type', '?')}")

    checks = {
        'LGB': model_data.get('model') is not None,
        'XGB': model_data.get('xgb_model') is not None,
        'FT': model_data.get('ft_model_state') is not None,
        'IR': model_data.get('ir_model_state') is not None,
    }
    for name, ok in checks.items():
        w = ew.get(name.lower(), 0)
        status = 'OK' if ok else 'MISSING'
        print(f"  {name}: {status} (weight={w})")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__} ({'CUDA' if torch.cuda.is_available() else 'CPU'})")
    except ImportError:
        print("  PyTorch: NOT INSTALLED (FT/IR will be skipped)")


def audit_feature_names(model_data):
    """学習時feature_namesと予測時特徴量の一致チェック"""
    print(f"\n{'='*60}")
    print("学習時 vs 予測時 特徴量チェック")
    print(f"{'='*60}")

    model_feats = model_data.get('features', [])
    lgb_model = model_data.get('model')

    # LGB feature names
    if hasattr(lgb_model, 'feature_name'):
        lgb_names = lgb_model.feature_name()
        n_lgb = lgb_model.num_feature()
        print(f"\n  Model features list: {len(model_feats)}")
        print(f"  LGB num_feature(): {n_lgb}")
        print(f"  LGB feature_name(): {lgb_names[:3]}... (Column_N形式 = 名前なし学習)")

        if n_lgb != len(model_feats):
            print(f"\n  [INFO] LGB({n_lgb}) != features list({len(model_feats)})")
            print(f"  → LGBには先頭{n_lgb}列のみ渡す（末尾{len(model_feats)-n_lgb}個はTYB直前データ）")
            extra_feats = model_feats[n_lgb:]
            print(f"  → LGB対象外: {extra_feats}")

    # XGB
    xgb_m = model_data.get('xgb_model')
    if xgb_m:
        n_xgb = xgb_m.num_features() if hasattr(xgb_m, 'num_features') else '?'
        print(f"  XGB num_features(): {n_xgb}")

    # FT scaler dimension
    ft_mean = model_data.get('ft_scaler_mean')
    if ft_mean is not None:
        print(f"  FT scaler dimension: {len(ft_mean)}")

    # IR scaler dimension
    ir_mean = model_data.get('ir_scaler_mean')
    if ir_mean is not None:
        print(f"  IR scaler dimension: {len(ir_mean)}")


def main():
    audit_mode = '--audit' in sys.argv

    # Race ID
    race_id = DEFAULT_RACE_ID
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            continue
        rid = extract_race_id(arg)
        if rid:
            race_id = rid

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
        race_name, horses, horse_ids, race_info = parse_shutuba(race_id)
    except Exception as e:
        print(f"[ERROR] 出馬表取得失敗: {e}")
        sys.exit(1)
    print(f"  レース: {race_name} / {race_info.get('course', '?')} "
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

    odds_dict = {}
    try:
        odds_dict = fetch_realtime_odds(race_id) or {}
    except Exception:
        pass

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

    # === 期待特徴量ベースの結果表示 ===
    expected = model_data.get('features', [])
    n_expected = len(expected) if expected else '?'

    print(f"\n{'='*60}")
    print(f"期待特徴量: {n_expected} / 生成列数: {len(df.columns)} / 馬数: {len(df)}")

    # 期待特徴量のステータス
    zero_expected = [f for f in expected if f in df.columns and (df[f] == 0).all()]
    nonzero_expected = [f for f in expected if f in df.columns and not (df[f] == 0).all()]
    missing_expected = [f for f in expected if f not in df.columns]

    print(f"\nOK (non-zero): {len(nonzero_expected)}/{n_expected}")
    print(f"ZERO (all zero): {len(zero_expected)}")
    print(f"MISSING: {len(missing_expected)}")

    # 正当なゼロを分類
    legit_zeros = {'dist_cat', 'season', 'location_enc', 'is_nar'}
    real_zeros = [f for f in zero_expected if f not in legit_zeros]
    legit_count = len([f for f in zero_expected if f in legit_zeros])

    if legit_count > 0:
        print(f"\n  正当なゼロ ({legit_count}): {[f for f in zero_expected if f in legit_zeros]}")

    if real_zeros:
        print(f"\n  要注意ゼロ ({len(real_zeros)}):")
        for f in real_zeros:
            print(f"    - {f}")

    if missing_expected:
        print(f"\n  不足特徴量 ({len(missing_expected)}):")
        for f in missing_expected:
            print(f"    - {f}")

    # NaN
    nan_cols = [c for c in df.columns if df[c].isna().any()]
    if nan_cols:
        print(f"\n  NaN列 ({len(nan_cols)}):")
        for c in nan_cols:
            print(f"    - {c} ({df[c].isna().sum()}/{len(df)})")

    # === 追加チェック ===
    # モデルアンサンブル状態
    audit_model_ensemble(model_data)

    # 学習時/予測時特徴量一致
    audit_feature_names(model_data)

    # JRDB CSV監査
    audit_jrdb_files()

    # === 総合判定 ===
    print(f"\n{'='*60}")
    print("総合判定")
    print(f"{'='*60}")
    issues = []
    if missing_expected:
        issues.append(f"不足特徴量 {len(missing_expected)}個")
    if len(real_zeros) >= 10:
        issues.append(f"要注意ゼロ {len(real_zeros)}個")
    if not model_data.get('ir_model_state'):
        issues.append("IR Attention モデルなし")
    if not model_data.get('ft_model_state'):
        issues.append("FT-Transformer モデルなし")

    if issues:
        print(f"\n  注意事項:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"\n  全チェック OK")

    rate = len(nonzero_expected) / len(expected) * 100 if expected else 0
    print(f"\n  特徴量取得率: {len(nonzero_expected)}/{n_expected} ({rate:.0f}%)")


if __name__ == "__main__":
    main()
