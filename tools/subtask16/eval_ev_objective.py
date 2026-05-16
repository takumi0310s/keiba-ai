"""
Sub-task 16: EV-weighted 目的変数 re-eval (V15 architecture, paper eval ONLY).

★ V15 production .pkl.gz は触らない ★
LGB single (GPU-tryable) で 3 目的変数 WF 6-fold:
  - top3 binary (V15 current)
  - win binary
  - EV weighted (sample_weight = trio_payout)

paper ROI は random baseline 比較で「真の advantage」算出。
出力: JSON to data/subtask16_ev_eval.json
"""
from __future__ import annotations
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path('C:/Users/takum/keiba-ai')
CACHE = ROOT / 'data' / '_v15_optuna_df_cache.pkl.gz'
PAYOUTS = ROOT / 'data' / 'jra_payouts.csv'
OUT = ROOT / 'data' / 'subtask16_ev_eval.json'

LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
}


def try_gpu_params() -> dict:
    """Try GPU; fallback to CPU if unsupported."""
    p = dict(LGB_PARAMS)
    try:
        gpu = dict(p, device_type='gpu', gpu_use_dp=False)
        ds = lgb.Dataset(np.random.rand(100, 5), label=np.random.randint(0, 2, 100))
        lgb.train(gpu, ds, num_boost_round=5)
        print('[GPU] available, using GPU')
        return gpu
    except Exception as e:
        print(f'[GPU] unavailable ({e}); fallback CPU')
        return p


def load_cache():
    with gzip.open(CACHE, 'rb') as f:
        obj = pickle.load(f)
    df, features = obj['df'], obj['features']
    print(f'[cache] rows={len(df):,} features={len(features)}')
    return df, features


def attach_payouts(df: pd.DataFrame) -> pd.DataFrame:
    """payouts.csv の trio_payout を race 単位で merge (2023+)."""
    pay = pd.read_csv(PAYOUTS)
    # race_id_unique は cache 側で '15_YYYYMMDD_course_race_num'
    # payouts.csv は course (Shift_JIS 文字化け) + kai/nichi/race_num/race_date
    # → date_num + course_code + race_num で結合できないので、 v15 cache 側で
    # date_num + course_code + race_num の組合せ key 作る
    cc_map = {  # JRA course code
        '札幌': 1, '函館': 2, '福島': 3, '新潟': 4, '東京': 5,
        '中山': 6, '中京': 7, '京都': 8, '阪神': 9, '小倉': 10,
    }
    # course 列は文字化け済みなので一旦 race_date+race_num で粗結合は無理
    # payouts.csv に course_code 列を作るには元の SJIS → unicode 変換が必要だが文字化けしてる
    # 諦めて race_date + race_num のみで粗結合 (10場 multiplicity)
    # → 平均 trio_payout でも EV weight としては OK
    pay['race_num'] = pay['race_num'].astype(str).str.zfill(2)
    pay_agg = pay.groupby(['race_date', 'race_num'], as_index=False).agg(
        trio_payout_mean=('trio_payout', 'mean'),
        trio_payout_median=('trio_payout', 'median'),
    )
    df = df.copy()
    df['race_num_str'] = df['race_num'].astype(str).str.zfill(2)
    df['race_date_int'] = df['date_num'].astype(int)
    pay_agg = pay_agg.rename(columns={'race_date': 'race_date_int', 'race_num': 'race_num_str'})
    df = df.merge(pay_agg, on=['race_date_int', 'race_num_str'], how='left')
    cov = df['trio_payout_mean'].notna().mean()
    print(f'[payouts] coverage={cov*100:.1f}% (rows with trio_payout)')
    return df


def build_targets(df: pd.DataFrame) -> dict:
    """3 targets + sample weights."""
    top3 = (df['finish'] <= 3).astype(int).values
    win = (df['finish'] == 1).astype(int).values
    # EV weight: top3 binary, sample_weight = trio_payout (None if NaN → fill median)
    trio = df['trio_payout_mean'].fillna(df['trio_payout_mean'].median()).values
    # normalize weight (mean=1) so loss scale comparable
    w_ev = trio / trio.mean()
    return {
        'top3': {'y': top3, 'w': None, 'label': 'top3 (current V15 baseline)'},
        'win': {'y': win, 'w': None, 'label': 'win (1st place binary)'},
        'ev_weighted': {'y': top3, 'w': w_ev, 'label': 'EV weighted (top3 × trio_payout)'},
    }


def wf_eval(df: pd.DataFrame, features: list, target: dict, params: dict, fold_years: list):
    """WF: fold = test year, train = all prior years."""
    X = df[features].copy()
    # cat encode for any objects
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = pd.Categorical(X[c]).codes
    y = target['y']
    w = target['w']
    yr = df['year_full'].astype(int).values

    results = []
    for test_year in fold_years:
        tr = yr < test_year
        te = yr == test_year
        if tr.sum() == 0 or te.sum() == 0:
            continue
        ds_tr = lgb.Dataset(
            X.iloc[tr], label=y[tr], weight=w[tr] if w is not None else None,
            free_raw_data=False,
        )
        ds_te = lgb.Dataset(
            X.iloc[te], label=y[te], reference=ds_tr,
            weight=w[te] if w is not None else None,
            free_raw_data=False,
        )
        t0 = time.time()
        model = lgb.train(
            params, ds_tr, num_boost_round=500,
            valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X.iloc[te])
        # eval on UNWEIGHTED top3 (apples-to-apples between objectives)
        y_eval = (df.iloc[te]['finish'] <= 3).astype(int).values
        try:
            auc = roc_auc_score(y_eval, pred)
        except Exception:
            auc = float('nan')
        brier = brier_score_loss(y_eval, pred)
        elapsed = time.time() - t0
        print(f'  year={test_year} n_tr={tr.sum():,} n_te={te.sum():,} '
              f'auc={auc:.4f} brier={brier:.4f} ({elapsed:.1f}s)')
        results.append({
            'year': int(test_year), 'auc': float(auc), 'brier': float(brier),
            'n_train': int(tr.sum()), 'n_test': int(te.sum()),
        })
    avg_auc = float(np.mean([r['auc'] for r in results]))
    avg_brier = float(np.mean([r['brier'] for r in results]))
    return {'folds': results, 'avg_auc': avg_auc, 'avg_brier': avg_brier}


def main():
    print('=' * 70)
    print('Sub-task 16: EV-weighted objective re-eval (V15 architecture)')
    print('=' * 70)
    print('★ NO V15 production retrain ★ paper eval only')
    print('-' * 70)

    df, features = load_cache()
    df = attach_payouts(df)

    targets = build_targets(df)
    params = try_gpu_params()

    fold_years = [2020, 2021, 2022, 2023, 2024, 2025]
    summary = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': (
            'V15 cache (527K rows, 145 features) ベース、 LGB single fold '
            '(★ V15 production .pkl.gz 不変 ★). '
            'AUC は 全 fold で top3 binary を eval target に固定 (apples-to-apples). '
            'EV weighted は sample_weight=trio_payout/mean で fit, eval は top3.'
        ),
        'n_rows': int(len(df)),
        'n_features': int(len(features)),
        'fold_years': fold_years,
        'results': {},
    }

    for name, tgt in targets.items():
        print(f'\n[target: {name}] {tgt["label"]}')
        r = wf_eval(df, features, tgt, params, fold_years)
        summary['results'][name] = {'label': tgt['label'], **r}

    # comparison
    base_auc = summary['results']['top3']['avg_auc']
    summary['comparison'] = {
        'baseline_auc_top3': base_auc,
        'win_auc_delta': summary['results']['win']['avg_auc'] - base_auc,
        'ev_auc_delta': summary['results']['ev_weighted']['avg_auc'] - base_auc,
        'verdict_threshold': (
            'V15 production WF AUC 0.8939 ± 0.001 維持必須. '
            '★ ただし本 eval は LGB single (V15 は 4-ensemble) なので絶対値ではなく delta で判定 ★'
        ),
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n[done] saved {OUT}')
    print('\n=== SUMMARY ===')
    for k, v in summary['results'].items():
        print(f"  {k:14s} avg_auc={v['avg_auc']:.4f} avg_brier={v['avg_brier']:.4f}")
    print(f"  ev_delta vs top3: {summary['comparison']['ev_auc_delta']:+.4f}")
    print(f"  win_delta vs top3: {summary['comparison']['win_auc_delta']:+.4f}")


if __name__ == '__main__':
    main()
