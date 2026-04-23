"""v16 dry-run — race_id_unique 修正の動作確認 (15分以内)

Usage:
    python tools/v16_dry_run.py

実行内容:
    1. _v15_train_df_cache.pkl 読み込み
    2. build_race_id() 呼び出し → race_id_unique 生成確認
    3. 2025年データのみで LightGBM を 1 fold 学習 (~5min)
    4. AUC 計測
    5. CatBoost 簡易学習 (race_id_unique 使用) でエラー無し確認 (~5min)

採用判定: 動くか動かないか (AUC値は不問)
出力: data/v16_dry_run_result.json
"""
from __future__ import annotations
import os, sys, json, time, pickle, traceback
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'train'))
sys.path.insert(0, os.path.join(BASE, 'tools'))

CACHE = os.path.join(BASE, 'data', '_v15_train_df_cache.pkl')
OUT = os.path.join(BASE, 'data', 'v16_dry_run_result.json')


def main():
    t0 = time.time()
    result = {
        'started_at': datetime.now().isoformat(),
        'steps': [],
        'errors': [],
    }

    print('[1/5] cache load...')
    if not os.path.exists(CACHE):
        result['errors'].append(f'cache not found: {CACHE}')
        _save(result, t0)
        return 1
    with open(CACHE, 'rb') as f:
        d = pickle.load(f)
    df = d['df']
    feats = d.get('v15_features', [])
    print(f'  shape={df.shape}  features={len(feats)}')
    result['steps'].append({'step': 'cache_load', 'shape': list(df.shape),
                            'features': len(feats)})

    print('[2/5] build_race_id...')
    try:
        from train_v135b_intra_ensemble import build_race_id
        df = build_race_id(df)
        n_races = df['race_id_unique'].nunique()
        print(f'  race_id_unique: {n_races} races')
        result['steps'].append({'step': 'build_race_id', 'n_races': n_races,
                                'ok': True})
    except Exception as e:
        tb = traceback.format_exc()
        result['errors'].append(f'build_race_id: {e}\n{tb[-500:]}')
        _save(result, t0)
        return 1

    print('[3/5] LightGBM 単一fold (2025年Test)...')
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        if 'year_full' in df.columns:
            year_col = 'year_full'
            train_df = df[(df[year_col] >= 2022) & (df[year_col] <= 2024)]
            test_df = df[df[year_col] == 2025]
        elif 'year' in df.columns:
            year_col = 'year'
            train_df = df[(df[year_col] >= 22) & (df[year_col] <= 24)]
            test_df = df[df[year_col] == 25]
        else:
            raise RuntimeError('year col not found')
        print(f'  train={len(train_df)} test={len(test_df)}')

        # target
        if 'target' in df.columns:
            y_train = train_df['target']
            y_test = test_df['target']
        else:
            y_train = (train_df['finish'] <= 3).astype(int)
            y_test = (test_df['finish'] <= 3).astype(int)

        feats_use = [f for f in feats if f in df.columns][:120]
        X_train = train_df[feats_use].fillna(0).astype(float)
        X_test = test_df[feats_use].fillna(0).astype(float)

        params = {'objective': 'binary', 'metric': 'auc', 'verbose': -1,
                  'num_leaves': 31, 'learning_rate': 0.1, 'seed': 42}
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=100,
                        valid_sets=[dtest],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        pred = bst.predict(X_test)
        auc = float(roc_auc_score(y_test, pred))
        print(f'  LGB AUC: {auc:.4f}')
        result['steps'].append({'step': 'lgb_train', 'auc': auc, 'ok': True,
                                'n_train': len(train_df), 'n_test': len(test_df),
                                'n_features': len(feats_use)})
    except Exception as e:
        tb = traceback.format_exc()
        result['errors'].append(f'lgb_train: {e}\n{tb[-500:]}')

    print('[4/5] CatBoost race_id_unique 確認 (簡易)...')
    try:
        from catboost import CatBoostClassifier
        # race_id_unique を group_id 相当として使えるか
        cb = CatBoostClassifier(iterations=50, depth=4, learning_rate=0.1,
                                verbose=0, allow_writing_files=False)
        # 軽量サンプル
        sample_train = train_df.sample(min(20000, len(train_df)), random_state=0)
        sample_test = test_df.sample(min(5000, len(test_df)), random_state=0)
        Xs = sample_train[feats_use].fillna(0).astype(float).values
        ys = (sample_train['finish'] <= 3).astype(int).values
        Xt = sample_test[feats_use].fillna(0).astype(float).values
        yt = (sample_test['finish'] <= 3).astype(int).values
        cb.fit(Xs, ys, eval_set=(Xt, yt), early_stopping_rounds=10)
        pred_cb = cb.predict_proba(Xt)[:, 1]
        auc_cb = float(roc_auc_score(yt, pred_cb))
        print(f'  CatBoost AUC: {auc_cb:.4f}')
        result['steps'].append({'step': 'catboost_train', 'auc': auc_cb,
                                'ok': True})

        # race_id_unique キーアクセス確認 (KeyError 出ないか)
        if 'race_id_unique' in test_df.columns:
            _ = test_df['race_id_unique'].iloc[0]
            result['steps'].append({'step': 'race_id_unique_access', 'ok': True})
    except Exception as e:
        tb = traceback.format_exc()
        result['errors'].append(f'catboost_train: {e}\n{tb[-500:]}')

    print('[5/5] 結果保存...')
    _save(result, t0)

    n_err = len(result['errors'])
    elapsed = (time.time() - t0) / 60
    print(f'\n=== Done elapsed={elapsed:.1f}min errors={n_err} ===')
    if result['errors']:
        for e in result['errors']:
            print('  ERR:', e[:200])
    return 0 if n_err == 0 else 2


def _save(result, t0):
    result['elapsed_sec'] = round(time.time() - t0, 1)
    result['ended_at'] = datetime.now().isoformat()
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    sys.exit(main())
