"""Phase 19: V18 真値版 学習 script (user CLI 5/16 実行 base)

【用途】
V15 base 150 features + Phase 11/12/13 真値化 features を統合した V18 candidate model 学習。

【現状 (5/10 22:00)】
- Phase 11 (5/12 真値化予定): 15 候補 features = scaffold defaults (gaika_*, odds_change_*_v18, jockey_*_winrate, return_horse_score 等)
- Phase 12 (5/12-5/13): JRA-VAN DataLab 17 候補 features = skeleton
- Phase 13 (5/13): netkeiba 25 候補 features = PoC 段階

→ 5/12-5/13 で 真値化完了後、 本 script を実行で V18 真値版 model 完成。
   現状 (5/10) で実行すると、 候補 features は default 値 (constant) のため LGB importance 0 になる
   (Phase 15 教訓: 57 features constant 問題)。 本 script は honest mode で 候補 features の
   constant 検出 + 警告を report。

【usage】
    python tools/train_v18_truevalue.py                  # Full WF train
    python tools/train_v18_truevalue.py --max-epochs 10  # 学習量制限
    python tools/train_v18_truevalue.py --check-only     # constant feature 検出のみ
    python tools/train_v18_truevalue.py --gpu            # GPU 利用 (RTX 4070 Ti SUPER)

【V15 投資保護】
V15 model file (keiba_model_v15_central_live.pkl.gz) 完全不変。
V18 model は新規 path (models/v18/) に保存、 V15 上書き禁止。
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import gzip
import pickle
from pathlib import Path
from typing import Optional

BASE = Path(r"C:/Users/takum/keiba-ai")
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models" / "v18"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "tools"))
sys.path.insert(0, str(BASE / "train"))


# Phase 11/12/13 候補 features (predict_core_v18.py / Phase 12 skeleton / Phase 13 PoC で定義)
PHASE_11_V18_FEATURES = [
    'gaika_id_enc', 'gaika_top3r_3r', 'gaika_winrate', 'gaika_dist_winrate',
    'odds_change_3h_v18', 'odds_change_30m_v18', 'popularity_shift_v18', 'odds_volatility_v18',
    'jockey_dist_winrate', 'jockey_track_winrate', 'jockey_class_winrate', 'jockey_x_trainer_wr',
    'return_horse_score', 'paddock_eval_v18', 'saddle_room_score',
]


def detect_constant_features(df, feature_list, tol: float = 1e-9) -> dict:
    """各 feature が constant (variance ≈ 0) かを検出.

    Phase 15 教訓: scaffold default のみだと variance 0 → LGB importance 0 → 学習効果なし。
    """
    import numpy as np
    constant_feats = []
    near_constant_feats = []
    real_signal_feats = []
    for f in feature_list:
        if f not in df.columns:
            continue
        col = df[f].astype(float, errors='ignore')
        try:
            std = float(col.std())
            unique = col.nunique()
        except Exception:
            continue
        if std < tol or unique <= 1:
            constant_feats.append(f)
        elif unique <= 5:
            near_constant_feats.append(f)
        else:
            real_signal_feats.append(f)
    return {
        'constant': constant_feats,
        'near_constant': near_constant_feats,
        'real_signal': real_signal_feats,
        'total': len(feature_list),
    }


def load_v15_base_features() -> tuple[list, list]:
    """V15 base 150 features list 取得 (model file から読込)"""
    pkl = BASE / "keiba_model_v15_central_live.pkl.gz"
    with gzip.open(pkl, 'rb') as f:
        m = pickle.load(f)
    v15_features = list(m['features'])
    return v15_features, list(m.get('sire_map', {}).keys())


def build_v18_dataframe() -> 'pd.DataFrame':
    """V15 cached training data を base に V18 candidate columns を追加.

    Phase 19 段階: V18 candidates は default scaffold 値 (Phase 11 通り)。
    5/12+ で各 feature 真値化計算 logic を以下に追加:
      - gaika_*: UKC/CHA から外厩 lookup
      - odds_change_*_v18: save_odds_base から時系列差分
      - jockey_*_winrate: KKA を distance/track/class で再集計
      - return/paddock/saddle: CYB/TYB 詳細
    """
    import pandas as pd
    cache = DATA_DIR / "_v15_train_df_cache.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"V15 cache 不在: {cache}\n→ train/train_v15_master.py で生成する必要あり")
    print(f"[load] {cache} ({cache.stat().st_size / 1e6:.0f} MB)")
    with open(cache, 'rb') as f:
        cached = pickle.load(f)
    # cache 構造: {'df': DataFrame, 'sire_map': dict, 'bms_map': dict, 'v15_features': list}
    if isinstance(cached, dict) and 'df' in cached:
        df = cached['df']
    else:
        df = cached
    print(f"[load] V15 cached df: {df.shape}")

    # Phase 11 V18 candidate columns 追加 (scaffold defaults)
    defaults = {
        'gaika_id_enc': 0,
        'gaika_top3r_3r': 0.33,
        'gaika_winrate': 0.20,
        'gaika_dist_winrate': 0.20,
        'odds_change_3h_v18': 0.0,
        'odds_change_30m_v18': 0.0,
        'popularity_shift_v18': 0,
        'odds_volatility_v18': 0.0,
        'jockey_dist_winrate': 0.10,
        'jockey_track_winrate': 0.10,
        'jockey_class_winrate': 0.10,
        'jockey_x_trainer_wr': 0.15,
        'return_horse_score': 0.0,
        'paddock_eval_v18': 0.0,
        'saddle_room_score': 0.0,
    }
    new_cols = {col: [d] * len(df) for col, d in defaults.items() if col not in df.columns}
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(f"[V18 candidate] {len(new_cols)}/15 columns added (scaffold defaults)")
    return df


def split_by_year(df, train_years, val_years, test_years):
    """時系列分割"""
    if 'year' in df.columns:
        train = df[df['year'].isin(train_years)]
        val = df[df['year'].isin(val_years)]
        test = df[df['year'].isin(test_years)]
        return train, val, test
    # Fallback: race_id 4 桁目
    if 'race_id' in df.columns:
        df['_year'] = df['race_id'].astype(str).str[:4].astype(int)
        train = df[df['_year'].isin(train_years)]
        val = df[df['_year'].isin(val_years)]
        test = df[df['_year'].isin(test_years)]
        return train, val, test
    raise ValueError("year/race_id column 不在")


def train_lgb_xgb_wf(df, v15_features, v18_features, args):
    """LGB + XGB WF training"""
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    all_features = v15_features + v18_features
    print(f"[train] V15 features: {len(v15_features)} + V18 candidates: {len(v18_features)} = {len(all_features)}")

    # 時系列 5 fold WF
    folds = [
        (range(2015, 2021), [2021]),
        (range(2015, 2022), [2022]),
        (range(2015, 2023), [2023]),
        (range(2015, 2024), [2024]),
        (range(2015, 2025), [2025]),
    ]

    # label 検出
    label_col = None
    for c in ('label_top3', 'is_top3', 'top3', 'label'):
        if c in df.columns:
            label_col = c
            break
    if not label_col:
        raise ValueError("label column 不在 (label_top3 / is_top3 / top3 / label)")
    print(f"[label] using {label_col}")

    if 'year' not in df.columns:
        df = df.copy()
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)

    fold_aucs_lgb = []
    fold_aucs_xgb = []
    for fi, (train_years, val_years) in enumerate(folds, 1):
        train_df = df[df['year'].isin(train_years)]
        val_df = df[df['year'].isin(val_years)]
        if len(train_df) == 0 or len(val_df) == 0:
            print(f"[fold {fi}] skip (no data: train={len(train_df)} val={len(val_df)})")
            continue

        X_train = train_df[all_features].fillna(0)
        y_train = train_df[label_col]
        X_val = val_df[all_features].fillna(0)
        y_val = val_df[label_col]

        # LGB
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'num_leaves': 63,
            'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'verbose': -1, 'seed': 42,
        }
        if args.gpu:
            lgb_params['device'] = 'gpu'
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=args.max_rounds,
                              valid_sets=[lgb_val], callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_pred = lgb_model.predict(X_val)
        lgb_auc = roc_auc_score(y_val, lgb_pred)
        fold_aucs_lgb.append(lgb_auc)

        # XGB
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 50,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': 42,
        }
        if args.gpu:
            xgb_params['tree_method'] = 'hist'
            xgb_params['device'] = 'cuda'
        else:
            xgb_params['tree_method'] = 'hist'
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=args.max_rounds,
                              evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        xgb_pred = xgb_model.predict(dval)
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        fold_aucs_xgb.append(xgb_auc)

        print(f"[fold {fi}] year={list(val_years)} LGB AUC={lgb_auc:.4f} / XGB AUC={xgb_auc:.4f} "
              f"(train={len(train_df):,}, val={len(val_df):,})")

        # Feature importance (V18 candidates のみ抽出)
        if fi == len(folds):  # 最終 fold のみ報告
            imp_df = pd.DataFrame({
                'feature': all_features,
                'lgb_imp': lgb_model.feature_importance(importance_type='gain'),
            }).sort_values('lgb_imp', ascending=False)
            v18_imp = imp_df[imp_df['feature'].isin(v18_features)]
            print(f"\n[V18 candidate feature importance (final fold LGB gain)]")
            print(v18_imp.to_string(index=False))
            v18_imp.to_csv(MODELS_DIR / "v18_feature_importance.csv", index=False)

    return {
        'lgb_aucs': fold_aucs_lgb,
        'xgb_aucs': fold_aucs_xgb,
        'lgb_mean': sum(fold_aucs_lgb) / max(len(fold_aucs_lgb), 1),
        'xgb_mean': sum(fold_aucs_xgb) / max(len(fold_aucs_xgb), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check-only', action='store_true', help='constant feature 検出のみ')
    p.add_argument('--gpu', action='store_true', help='GPU 利用')
    p.add_argument('--max-rounds', type=int, default=1000)
    p.add_argument('--save-model', action='store_true', help='V15 baseline 超えたら保存')
    args = p.parse_args()

    print("=" * 70)
    print("Phase 19: V18 真値版 学習 (Phase 1-18 累計、 V15 投資保護)")
    print("=" * 70)
    t0 = time.time()

    # V15 baseline
    v15_features, _ = load_v15_base_features()
    print(f"\nV15 baseline: {len(v15_features)} features (AUC 0.8939)")

    # Build df
    df = build_v18_dataframe()

    # Constant detection
    print("\n=== Phase 11/12/13 候補 features constant 検出 ===")
    detect = detect_constant_features(df, PHASE_11_V18_FEATURES)
    print(f"  constant (variance ~0): {len(detect['constant'])} -> 学習効果なし")
    if detect['constant']:
        for f in detect['constant'][:10]:
            print(f"    - {f}")
    print(f"  near_constant (uniq≤5): {len(detect['near_constant'])}")
    print(f"  real signal (uniq>5): {len(detect['real_signal'])}")

    if len(detect['constant']) >= 10:
        print("\n⚠ ★ Phase 15 教訓 警告 ★")
        print("  V18 候補 features の大半が constant (scaffold default のみ)。")
        print("  本 train で V15 比 AUC 改善は期待できません。")
        print("  → 5/12-5/13 で 各 feature の 真値化計算 logic を実装してから再実行を推奨。")

    if args.check_only:
        print("\n[check-only mode] 終了")
        return

    # Train
    print("\n=== WF training (5 fold) ===")
    result = train_lgb_xgb_wf(df, v15_features, PHASE_11_V18_FEATURES, args)

    # Report
    print("\n" + "=" * 70)
    print("Phase 19 学習 結果")
    print("=" * 70)
    print(f"  V15 baseline AUC: 0.8939")
    print(f"  V18 cand LGB mean: {result['lgb_mean']:.4f}")
    print(f"  V18 cand XGB mean: {result['xgb_mean']:.4f}")
    delta_lgb = result['lgb_mean'] - 0.8939
    print(f"  Δ LGB: {delta_lgb:+.4f}")
    print(f"  経過時間: {(time.time() - t0)/60:.1f} 分")

    # save report
    out = {
        'lgb_aucs': result['lgb_aucs'], 'xgb_aucs': result['xgb_aucs'],
        'lgb_mean': result['lgb_mean'], 'xgb_mean': result['xgb_mean'],
        'v15_baseline': 0.8939, 'delta_lgb': delta_lgb,
        'constant_features': detect['constant'],
        'real_signal_features': detect['real_signal'],
        'elapsed_min': (time.time() - t0) / 60,
    }
    report_path = MODELS_DIR / "v18_training_report.json"
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nreport saved: {report_path}")

    # 採用判定
    if delta_lgb > 0 and args.save_model:
        print("\n★ V18 candidate AUC > V15 baseline、 model 保存可能 ★")
        # NOTE: 実 model 保存は train_v15_master.save_production_model() pattern を流用
        print("  → train/train_v15_master.save_production_model() pattern で保存実装可")
    else:
        print(f"\n(AUC 改善なし or --save-model 未指定、 model 保存 skip)")


if __name__ == '__main__':
    main()
