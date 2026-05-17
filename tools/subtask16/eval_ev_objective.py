"""
Sub-task 16: EV-weighted 目的変数 re-eval (★ 軽量化 final、 5/17 ★).

★ V15 production .pkl.gz は触らない、 paper eval ONLY ★

LGB single (GPU-tryable, ★ FT/IR 抜き ★) で 3 目的変数 WF 6-fold:
  - top3_binary   (V15 current baseline)
  - win_binary    (1st place only)
  - ev_weighted   (sample_weight = trio_payout)

各 target で:
  - AUC は top3 binary fixed (apples-to-apples)
  - Brier
  - paper ROI 2 種類:
      (a) inflated estimator (o1*o2*o3*20), Phase 2 と同 source、 396.2% claim 再現
      (b) jra_payouts.csv 実配当 merge (date+race_num、 course は SJIS mojibake で
          失った → 10x multiplicity を mean で吸収。 absolute は粗いが delta 比較は OK)

LEAK 監査:
  - features から odds_* / pop_rank* を抜いた "leak-free" 版でも同じ eval を実行
  - delta = full - leak_free を出力

Hard limit: 60 min wall。 fold-level checkpoint → timeout でも partial 保存。
Output: data/subtask16_ev_eval_final.json
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
OUT = ROOT / 'data' / 'subtask16_ev_eval_final.json'
CKPT = ROOT / 'data' / 'subtask16_ev_eval_ckpt.json'

TIME_BUDGET_SEC = 60 * 60   # ★ 60 min hard limit ★
T0 = time.time()


def remaining() -> float:
    return TIME_BUDGET_SEC - (time.time() - T0)


def out_of_time() -> bool:
    return remaining() < 60   # leave 60s slack for final save


# leak features (Pattern A の LEAK + odds/popularity 系)
LEAK_FEATURES = {
    'prev_odds_log',       # confirmed odds = leak
    'oz_tansho_base_log',  # 当日オッズ系
    'oz_fukusho_base_log',
    'oz_base_pop_rank',
    'odds_change_rate',
    'pop_rank_change',
    'odds_sharp_drop',
    'paci_ninki_idx',      # 人気 index
}

# light LGB params (★ 軽量化 ★、 num_boost_round も 短縮)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.07,        # 少し高め (round 数減らすため)
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
NUM_BOOST_ROUND = 250
EARLY_STOP = 20


def try_gpu_params() -> dict:
    p = dict(LGB_PARAMS)
    try:
        gpu = dict(p, device_type='gpu', gpu_use_dp=False)
        ds = lgb.Dataset(np.random.rand(100, 5), label=np.random.randint(0, 2, 100))
        lgb.train(gpu, ds, num_boost_round=5)
        print('[GPU] available')
        return gpu
    except Exception as e:
        print(f'[GPU] unavailable ({type(e).__name__}); CPU fallback')
        return p


def load_cache():
    with gzip.open(CACHE, 'rb') as f:
        obj = pickle.load(f)
    df, features = obj['df'], obj['features']
    print(f'[cache] rows={len(df):,} features={len(features)}')
    return df, features


def attach_payouts(df: pd.DataFrame) -> pd.DataFrame:
    """payouts.csv の trio_payout を race 単位で merge.

    course は SJIS mojibake で複合 key 作れない → date + race_num の 10x mean で
    吸収。 absolute payout は 粗いが、 target 間の relative ROI 比較には十分。
    """
    pay = pd.read_csv(PAYOUTS)
    pay['race_num'] = pay['race_num'].astype(int)
    pay_agg = pay.groupby(['race_date', 'race_num'], as_index=False).agg(
        trio_payout_mean=('trio_payout', 'mean'),
    )
    df = df.copy()
    df['race_num_int'] = df['race_num'].astype(int)
    df['race_date_int'] = df['date_num'].astype(int)
    df = df.merge(
        pay_agg.rename(columns={'race_date': 'race_date_int', 'race_num': 'race_num_int'}),
        on=['race_date_int', 'race_num_int'], how='left',
    )
    cov = df['trio_payout_mean'].notna().mean()
    print(f'[payouts] coverage={cov*100:.1f}%')
    return df


def build_targets(df: pd.DataFrame) -> dict:
    top3 = (df['finish'] <= 3).astype(int).values
    win = (df['finish'] == 1).astype(int).values
    trio = df['trio_payout_mean'].fillna(df['trio_payout_mean'].median()).values
    w_ev = trio / max(trio.mean(), 1e-6)
    return {
        'top3': {'y': top3, 'w': None, 'label': 'top3 (V15 current baseline)'},
        'win': {'y': win, 'w': None, 'label': 'win (1st binary)'},
        'ev_weighted': {'y': top3, 'w': w_ev, 'label': 'EV weighted (top3 × trio_payout)'},
    }


def _paper_roi(df_te: pd.DataFrame, pred: np.ndarray, mode: str) -> dict:
    """Paper ROI: race_key = race_id_unique + course (mojibake でも race 単位識別 OK).

    買い目 = pred 上位 3 頭、 全 3 頭 == actual top3 で hit (順不同).
    mode='inflated': o1*o2*o3*20 (Phase 2 と同 estimator)
    mode='actual':   trio_payout_mean (jra_payouts merge、 ~10x multiplicity 注意)
    """
    df_eval = df_te.copy()
    df_eval['_pred'] = pred
    df_eval['_race_key'] = (df_eval['race_id_unique'].astype(str) + '|'
                            + df_eval['course'].astype(str))
    inv_per_race = 700
    inv_total = pay_total = hits = n = 0
    for _, g in df_eval.groupby('_race_key', sort=False):
        if len(g) < 4:
            continue
        actual_top3 = set(g.loc[g['finish'] <= 3, 'umaban'].astype(int).tolist())
        if len(actual_top3) < 3:
            continue
        pred_top3 = set(g.nlargest(3, '_pred')['umaban'].astype(int).tolist())
        hit = (actual_top3 == pred_top3)
        pay = 0.0
        if hit:
            if mode == 'inflated':
                if 'oz_tansho_base_log' in g.columns:
                    odds_map = {int(r['umaban']): float(np.exp(r['oz_tansho_base_log']))
                                for _, r in g.iterrows()}
                else:
                    odds_map = {int(r['umaban']): 10.0 for _, r in g.iterrows()}
                o = sorted([odds_map.get(u, 10.0) for u in actual_top3])
                pay = max(100.0, o[0] * o[1] * o[2] * 20.0)
            else:  # actual
                v = g['trio_payout_mean'].iloc[0]
                pay = float(v) if not pd.isna(v) else 0.0
        inv_total += inv_per_race
        pay_total += pay
        hits += int(hit)
        n += 1
    if n == 0:
        return {'n': 0, 'roi': 0.0, 'hit_rate': 0.0}
    return {
        'n': n,
        'roi': pay_total / inv_total * 100.0 if inv_total else 0.0,
        'hit_rate': hits / n * 100.0,
    }


def paper_roi_inflated(df_te, pred):
    return _paper_roi(df_te, pred, 'inflated')


def paper_roi_actual(df_te, pred):
    return _paper_roi(df_te, pred, 'actual')


def wf_eval(df: pd.DataFrame, features: list, target: dict, params: dict,
            fold_years: list, *, leak_free: bool = False, tag: str = '') -> dict:
    feat_use = [f for f in features if (not leak_free) or (f not in LEAK_FEATURES)]
    X = df[feat_use].copy()
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = pd.Categorical(X[c]).codes
    y = target['y']
    w = target['w']
    yr = df['year_full'].astype(int).values

    results = []
    for test_year in fold_years:
        if out_of_time():
            print(f'  [{tag}] year={test_year} SKIPPED (timeout, remaining={remaining():.0f}s)')
            break
        tr = (yr < test_year) & (yr >= 2015)
        te = yr == test_year
        if tr.sum() == 0 or te.sum() == 0:
            continue
        ds_tr = lgb.Dataset(
            X.iloc[tr], label=y[tr], weight=w[tr] if w is not None else None,
            free_raw_data=False,
        )
        ds_te = lgb.Dataset(
            X.iloc[te], label=y[te], reference=ds_tr,
            weight=w[te] if w is not None else None, free_raw_data=False,
        )
        t0 = time.time()
        model = lgb.train(
            params, ds_tr, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X.iloc[te])
        df_te = df.iloc[te].copy()
        y_eval = (df_te['finish'] <= 3).astype(int).values
        try:
            auc = roc_auc_score(y_eval, pred)
        except Exception:
            auc = float('nan')
        brier = brier_score_loss(y_eval, pred)
        inflated = paper_roi_inflated(df_te, pred)
        actual = paper_roi_actual(df_te, pred)
        elapsed = time.time() - t0
        msg = (f'  [{tag}] year={test_year} n_tr={tr.sum():,} n_te={te.sum():,} '
               f'auc={auc:.4f} brier={brier:.4f} '
               f'roi_inf={inflated["roi"]:.1f}% roi_act={actual["roi"]:.1f}% '
               f'hit={actual["hit_rate"]:.1f}% ({elapsed:.1f}s)')
        print(msg)
        results.append({
            'year': int(test_year), 'auc': float(auc), 'brier': float(brier),
            'n_train': int(tr.sum()), 'n_test': int(te.sum()),
            'roi_inflated': inflated['roi'], 'roi_actual': actual['roi'],
            'hit_rate': actual['hit_rate'], 'n_races_eval': inflated['n'],
            'elapsed_sec': elapsed,
        })
        # checkpoint
        try:
            with open(CKPT, 'w', encoding='utf-8') as f:
                json.dump({'tag': tag, 'last_year': int(test_year),
                           'remaining_sec': remaining()}, f)
        except Exception:
            pass
    if not results:
        return {'folds': [], 'avg_auc': float('nan'), 'avg_brier': float('nan'),
                'avg_roi_inflated': 0.0, 'avg_roi_actual': 0.0, 'avg_hit_rate': 0.0,
                'n_features': len(feat_use)}
    return {
        'folds': results,
        'avg_auc': float(np.mean([r['auc'] for r in results])),
        'avg_brier': float(np.mean([r['brier'] for r in results])),
        'avg_roi_inflated': float(np.mean([r['roi_inflated'] for r in results])),
        'avg_roi_actual': float(np.mean([r['roi_actual'] for r in results])),
        'avg_hit_rate': float(np.mean([r['hit_rate'] for r in results])),
        'n_features': len(feat_use),
    }


def main():
    print('=' * 72)
    print('Sub-task 16: EV-weighted objective re-eval (★ 軽量化 final ★)')
    print('=' * 72)
    print(f'★ NO V15 retrain ★ paper eval only. budget={TIME_BUDGET_SEC}s')
    print('-' * 72)

    df, features = load_cache()
    df = attach_payouts(df)

    targets = build_targets(df)
    params = try_gpu_params()

    # ★ 軽量化: 3 fold だけ (2023/2024/2025、 直近 + 安定) ★
    # 6 fold は元 script 設計だが 60 min budget では timeout 確実
    fold_years = [2023, 2024, 2025]
    summary = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'budget_sec': TIME_BUDGET_SEC,
        'note': (
            'V15 cache (527K rows, 145 features) ベース、 LGB single (★ FT/IR 抜き ★). '
            'AUC は 全 fold で top3 binary 固定 eval (apples-to-apples). '
            'fold = 2023/2024/2025 直近 3 年 (60 min budget 内収束のため). '
            'LEAK 監査: full vs leak_free (odds/pop 8 features 抜き) を併走.'
        ),
        'n_rows': int(len(df)),
        'n_features_full': int(len(features)),
        'leak_features_removed': sorted(LEAK_FEATURES & set(features)),
        'fold_years': fold_years,
        'lgb_params': {**LGB_PARAMS, 'num_boost_round': NUM_BOOST_ROUND, 'early_stop': EARLY_STOP},
        'results': {},
    }

    # === full features ===
    for name, tgt in targets.items():
        if out_of_time():
            print(f'[skip {name}] timeout')
            break
        print(f'\n[FULL  / target: {name}] {tgt["label"]}')
        r = wf_eval(df, features, tgt, params, fold_years,
                    leak_free=False, tag=f'full-{name}')
        summary['results'][f'full_{name}'] = {'label': tgt['label'], **r}
        # incremental save
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # === leak-free features ===
    for name, tgt in targets.items():
        if out_of_time():
            print(f'[skip leak-free {name}] timeout')
            break
        print(f'\n[LEAKFREE / target: {name}] {tgt["label"]}')
        r = wf_eval(df, features, tgt, params, fold_years,
                    leak_free=True, tag=f'leakfree-{name}')
        summary['results'][f'leakfree_{name}'] = {'label': tgt['label'] + ' (leak-free)', **r}
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # comparison
    def safe_get(k, fld):
        return summary['results'].get(k, {}).get(fld, float('nan'))

    base = safe_get('full_top3', 'avg_auc')
    summary['comparison'] = {
        'baseline_full_top3_auc': base,
        'full_win_auc_delta':       safe_get('full_win', 'avg_auc') - base,
        'full_ev_auc_delta':        safe_get('full_ev_weighted', 'avg_auc') - base,
        'full_top3_roi_inflated':   safe_get('full_top3', 'avg_roi_inflated'),
        'full_ev_roi_inflated':     safe_get('full_ev_weighted', 'avg_roi_inflated'),
        'full_top3_roi_actual':     safe_get('full_top3', 'avg_roi_actual'),
        'full_ev_roi_actual':       safe_get('full_ev_weighted', 'avg_roi_actual'),
        # leak audit
        'leak_audit_top3_auc_drop':  base - safe_get('leakfree_top3', 'avg_auc'),
        'leak_audit_ev_auc_drop':    safe_get('full_ev_weighted', 'avg_auc') - safe_get('leakfree_ev_weighted', 'avg_auc'),
        'leak_audit_top3_roi_inflated_drop':
            safe_get('full_top3', 'avg_roi_inflated') - safe_get('leakfree_top3', 'avg_roi_inflated'),
        'leak_audit_ev_roi_inflated_drop':
            safe_get('full_ev_weighted', 'avg_roi_inflated') - safe_get('leakfree_ev_weighted', 'avg_roi_inflated'),
        'verdict_note': (
            'V15 production 4-ens WF AUC 0.8939 は本 eval の LGB single (3-fold直近) と '
            '直接比較 不可。 ★ delta で判定 ★. paper ROI inflated は Phase 2 (3 月 396.2% claim) '
            'と同 estimator → 真贋: o1*o2*o3*20 は 実配当の 約 2x で systematically inflated '
            '(CLAUDE.md Phase 2-3 検証済). 実配当 (roi_actual) を判定基準とする.'
        ),
    }
    summary['phase2_396pct_verdict'] = {
        'claim': '396.2% (3 月 phase 2 ev_weighted overall_estimated_roi)',
        'source': 'tools/validation_2_target_variable.py + backtest_central_leakfree.estimate_payouts (o1*o2*o3*20)',
        'verdict': 'INFLATED ESTIMATOR (NOT honest real ROI)',
        'evidence': [
            'Phase 2 で place_current も 同 estimator で 368.8% → ev_weighted の advantage は +27.4pt のみ',
            'CLAUDE.md Phase 2b/3 検証: o1*o2*o3*20 は 実配当 の 約 2x (jra_payouts 比 ~16x 推定 → 実 ~2x)',
            'Phase 3 実配当 検証で trio 真 ROI = 225.8% (place_current 同程度) → 396.2% は 50-60% 過大',
            '本 eval roi_inflated vs roi_actual を見て 同様の inflation 比 を確認',
        ],
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n[done] {OUT}')
    print('\n=== SUMMARY ===')
    for k, v in summary['results'].items():
        auc = v.get('avg_auc', float('nan'))
        roi_i = v.get('avg_roi_inflated', 0.0)
        roi_a = v.get('avg_roi_actual', 0.0)
        hit = v.get('avg_hit_rate', 0.0)
        print(f'  {k:25s} auc={auc:.4f} roi_inf={roi_i:6.1f}% '
              f'roi_act={roi_a:6.1f}% hit={hit:5.1f}%')
    print(f"\n  base auc (full_top3) = {summary['comparison']['baseline_full_top3_auc']:.4f}")
    print(f"  ev delta (full)      = {summary['comparison']['full_ev_auc_delta']:+.4f}")
    print(f"  win delta (full)     = {summary['comparison']['full_win_auc_delta']:+.4f}")
    print(f"  leak top3 drop       = {summary['comparison']['leak_audit_top3_auc_drop']:+.4f}")
    print(f"  leak ev drop         = {summary['comparison']['leak_audit_ev_auc_drop']:+.4f}")
    print(f"\n  Phase 2 396.2% verdict: {summary['phase2_396pct_verdict']['verdict']}")
    print(f'  elapsed: {time.time() - T0:.1f}s / budget {TIME_BUDGET_SEC}s')


if __name__ == '__main__':
    main()
