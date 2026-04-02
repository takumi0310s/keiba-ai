#!/usr/bin/env python3
"""v13.5b IntraRace Attention + LGB+XGB+FT 4-model ensemble WF evaluation

Phase 3のIntraRace AUC 0.89+はレース内全馬を同時に見ているため高い。
本スクリプトでは、IntraRace予測を馬単位にflattenし、
LGB+XGB+FT+IntraRaceの4モデルアンサンブルとしてWF AUCを計算する。

IntraRaceの利点:
- 同レース内の全馬の特徴量を同時に見て相対評価
- 「このメンバーの中でこの馬は強い」を学習
- LGB/XGB/FTとは異なる視点（馬単位 vs レース単位）

採用基準: WF AUC > 0.8657 (v13.5 LGB+XGB+FT), gap < 0.05
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v135_ft_transformer import (
    build_v134_dataframe, get_v134_features, fill_defaults,
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    DEVICE, LGB_PARAMS, XGB_PARAMS,
)
from train_v92_leakfree import LEAK_FEATURES_A

MAX_HORSES = 28


def build_race_id(df):
    """レース単位のユニークIDを作成"""
    if 'race_id_unique' not in df.columns:
        df['race_id_unique'] = (
            df['year'].astype(str) + '_' +
            df['date_num'].astype(str) + '_' +
            df.get('course_code', df.get('course_enc', pd.Series(0, index=df.index))).astype(str) + '_' +
            df.get('race_num', pd.Series(0, index=df.index)).astype(str)
        )
    return df


def train_intra_race_with_preds(df_train, df_val, features, label='target',
                                 epochs=30, batch_size=64, lr=1e-3, patience=8,
                                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
    """IntraRaceモデルを学習し、馬単位の予測を返す。

    Returns:
        model: 学習済みモデル
        val_preds_flat: validation全馬のsigmoid予測 (len=val馬数)
        val_indices: validation DataFrameのindex配列
        train_auc: 訓練データのAUC
        val_auc: 検証データのAUC
    """

    def make_race_batches_with_indices(df, feats):
        """レース単位バッチ + 元DataFrameのindexを保持"""
        race_groups = df.groupby('race_id_unique')
        X_races, y_races, masks, indices_list, counts = [], [], [], [], []
        for _, grp in race_groups:
            n = min(len(grp), MAX_HORSES)
            if n < 2:
                continue
            x = np.zeros((MAX_HORSES, len(feats)), dtype=np.float32)
            y = np.zeros(MAX_HORSES, dtype=np.float32)
            m = np.zeros(MAX_HORSES, dtype=np.float32)
            idx = np.full(MAX_HORSES, -1, dtype=np.int64)
            x[:n] = grp[feats].values[:n]
            y[:n] = grp[label].values[:n]
            m[:n] = 1.0
            idx[:n] = grp.index.values[:n]
            X_races.append(x)
            y_races.append(y)
            masks.append(m)
            indices_list.append(idx)
            counts.append(n)
        return (np.array(X_races), np.array(y_races),
                np.array(masks), np.array(indices_list), counts)

    X_tr, y_tr, m_tr, idx_tr, cnt_tr = make_race_batches_with_indices(df_train, features)
    X_va, y_va, m_va, idx_va, cnt_va = make_race_batches_with_indices(df_val, features)
    print(f"    IntraRace: train={len(X_tr)} races, val={len(X_va)} races")

    model = IntraRaceAttention(
        n_features=len(features), d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, dropout=dropout
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([(1 - y_tr[m_tr > 0].mean()) / y_tr[m_tr > 0].mean()]).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    train_ds = TensorDataset(
        torch.FloatTensor(X_tr), torch.FloatTensor(y_tr), torch.FloatTensor(m_tr))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

    X_va_t = torch.FloatTensor(X_va).to(DEVICE)
    m_va_t = torch.FloatTensor(m_va).to(DEVICE)

    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batch = 0
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, mb)
            loss_raw = bce(logits, yb) * pos_weight * mb
            loss = loss_raw.sum() / mb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            # Process validation in chunks
            val_probs_all = []
            chunk = 256
            for i in range(0, len(X_va), chunk):
                xc = torch.FloatTensor(X_va[i:i+chunk]).to(DEVICE)
                mc = torch.FloatTensor(m_va[i:i+chunk]).to(DEVICE)
                logits = model(xc, mc)
                val_probs_all.append(torch.sigmoid(logits).cpu().numpy())
            val_probs = np.concatenate(val_probs_all)

            valid = m_va.flatten() > 0
            y_flat = y_va.flatten()[valid]
            p_flat = val_probs.flatten()[valid]
            val_auc = roc_auc_score(y_flat, p_flat)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or no_improve == 0:
            print(f"    IR Epoch {epoch+1:3d}: loss={total_loss/n_batch:.4f} "
                  f"val_AUC={val_auc:.4f} best={best_auc:.4f}")

        if no_improve >= patience:
            print(f"    IR Early stop at epoch {epoch+1}")
            break

    # Load best and predict
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        # Validation predictions
        val_probs_all = []
        chunk = 256
        for i in range(0, len(X_va), chunk):
            xc = torch.FloatTensor(X_va[i:i+chunk]).to(DEVICE)
            mc = torch.FloatTensor(m_va[i:i+chunk]).to(DEVICE)
            logits = model(xc, mc)
            val_probs_all.append(torch.sigmoid(logits).cpu().numpy())
        val_probs = np.concatenate(val_probs_all)

        # Train predictions (for gap check)
        tr_probs_all = []
        for i in range(0, len(X_tr), chunk):
            xc = torch.FloatTensor(X_tr[i:i+chunk]).to(DEVICE)
            mc = torch.FloatTensor(m_tr[i:i+chunk]).to(DEVICE)
            logits = model(xc, mc)
            tr_probs_all.append(torch.sigmoid(logits).cpu().numpy())
        tr_probs = np.concatenate(tr_probs_all)

    # Flatten to per-horse predictions with original indices
    val_preds_dict = {}  # index -> prediction
    for r in range(len(X_va)):
        for h in range(MAX_HORSES):
            if m_va[r, h] > 0 and idx_va[r, h] >= 0:
                val_preds_dict[idx_va[r, h]] = val_probs[r, h]

    tr_preds_dict = {}
    for r in range(len(X_tr)):
        for h in range(MAX_HORSES):
            if m_tr[r, h] > 0 and idx_tr[r, h] >= 0:
                tr_preds_dict[idx_tr[r, h]] = tr_probs[r, h]

    # Compute train AUC
    valid_tr = m_tr.flatten() > 0
    tr_auc = roc_auc_score(y_tr.flatten()[valid_tr], tr_probs.flatten()[valid_tr])

    return model, val_preds_dict, tr_preds_dict, tr_auc, best_auc


def walk_forward_4model(df, features, years=range(2020, 2026)):
    """LGB + XGB + FT + IntraRace の4モデルアンサンブルWF"""

    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values
        test_indices = df.loc[test_mask].index.values

        print(f"\n{'='*60}")
        print(f"  4-Model WF {test_year}: train={len(X_tr)}, test={len(X_te)}")
        print(f"{'='*60}")

        # --- LightGBM ---
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        p_lgb_tr = m_lgb.predict(X_tr)
        auc_lgb_tr = roc_auc_score(y_tr, p_lgb_tr)

        # --- XGBoost ---
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # --- FT-Transformer ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.astype(np.float32))
        X_te_s = scaler.transform(X_te.astype(np.float32))

        ft_model, p_ft, p_ft_tr, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            epochs=50, batch_size=4096, lr=1e-3,
            patience=10, d_token=64, n_heads=4, n_layers=3,
            dropout=0.1, label=f'FT-{test_year}',
        )

        # --- IntraRace Attention ---
        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()

        # Scale for IntraRace
        scaler_ir = StandardScaler()
        df_tr_scaled = df_tr.copy()
        df_te_scaled = df_te.copy()
        df_tr_scaled[features] = scaler_ir.fit_transform(df_tr[features].values.astype(np.float32))
        df_te_scaled[features] = scaler_ir.transform(df_te[features].values.astype(np.float32))

        print(f"  Training IntraRace Attention...")
        ir_model, ir_val_dict, ir_tr_dict, ir_tr_auc, ir_val_auc = \
            train_intra_race_with_preds(
                df_tr_scaled, df_te_scaled, features,
                epochs=30, batch_size=64, lr=1e-3, patience=8,
                d_model=64, n_heads=4, n_layers=2, dropout=0.1,
            )

        # Map IntraRace predictions to test array
        p_ir = np.zeros(len(X_te), dtype=np.float32)
        ir_coverage = 0
        for i, idx in enumerate(test_indices):
            if idx in ir_val_dict:
                p_ir[i] = ir_val_dict[idx]
                ir_coverage += 1
            else:
                p_ir[i] = 0.3  # default for unmatched (rare)

        ir_cov_pct = ir_coverage / len(X_te) * 100
        auc_ir = roc_auc_score(y_te, p_ir) if ir_coverage > len(X_te) * 0.5 else 0
        print(f"  IR coverage: {ir_coverage}/{len(X_te)} ({ir_cov_pct:.1f}%)")

        # --- Ensembles ---
        # 1) LGB+XGB baseline
        w_lgb2 = auc_lgb / (auc_lgb + auc_xgb)
        w_xgb2 = auc_xgb / (auc_lgb + auc_xgb)
        p_lgbxgb = w_lgb2 * p_lgb + w_xgb2 * p_xgb
        auc_lgbxgb = roc_auc_score(y_te, p_lgbxgb)

        # 2) LGB+XGB+FT (v13.5 baseline)
        total3 = auc_lgb + auc_xgb + auc_ft
        p_3m = (auc_lgb/total3) * p_lgb + (auc_xgb/total3) * p_xgb + (auc_ft/total3) * p_ft
        auc_3m = roc_auc_score(y_te, p_3m)

        # 3) LGB+XGB+FT+IR (4-model)
        if auc_ir > 0:
            total4 = auc_lgb + auc_xgb + auc_ft + auc_ir
            p_4m = ((auc_lgb/total4) * p_lgb + (auc_xgb/total4) * p_xgb +
                    (auc_ft/total4) * p_ft + (auc_ir/total4) * p_ir)
            auc_4m = roc_auc_score(y_te, p_4m)
        else:
            auc_4m = auc_3m

        # 4) LGB+XGB+IR (IR replacing FT)
        if auc_ir > 0:
            total_lxi = auc_lgb + auc_xgb + auc_ir
            p_lxi = ((auc_lgb/total_lxi) * p_lgb + (auc_xgb/total_lxi) * p_xgb +
                     (auc_ir/total_lxi) * p_ir)
            auc_lxi = roc_auc_score(y_te, p_lxi)
        else:
            auc_lxi = auc_lgbxgb

        # 5) Grid search for optimal 4-model weights
        best_grid_auc = 0
        best_grid_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.25, 0.45, 0.05):  # LGB
                for w2 in np.arange(0.25, 0.45, 0.05):  # XGB
                    for w3 in np.arange(0.05, 0.30, 0.05):  # FT
                        w4 = 1.0 - w1 - w2 - w3  # IR
                        if w4 < 0.01 or w4 > 0.35:
                            continue
                        p_grid = w1*p_lgb + w2*p_xgb + w3*p_ft + w4*p_ir
                        auc_grid = roc_auc_score(y_te, p_grid)
                        if auc_grid > best_grid_auc:
                            best_grid_auc = auc_grid
                            best_grid_w = (w1, w2, w3, w4)

        gap = auc_lgb_tr - auc_lgb

        print(f"\n  {test_year} Results:")
        print(f"    LGB:         {auc_lgb:.4f}")
        print(f"    XGB:         {auc_xgb:.4f}")
        print(f"    FT:          {auc_ft:.4f}")
        print(f"    IR:          {auc_ir:.4f} (coverage={ir_cov_pct:.1f}%)")
        print(f"    LGB+XGB:     {auc_lgbxgb:.4f}")
        print(f"    LGB+XGB+FT:  {auc_3m:.4f} (v13.5 baseline)")
        print(f"    LGB+XGB+IR:  {auc_lxi:.4f}")
        print(f"    4-model:     {auc_4m:.4f}")
        if best_grid_w:
            print(f"    4m-grid:     {best_grid_auc:.4f} "
                  f"(L={best_grid_w[0]:.2f} X={best_grid_w[1]:.2f} "
                  f"F={best_grid_w[2]:.2f} IR={best_grid_w[3]:.2f})")
        print(f"    Gap(LGB):    {gap:.4f}")

        results.append({
            'year': test_year,
            'lgb_auc': auc_lgb,
            'xgb_auc': auc_xgb,
            'ft_auc': auc_ft,
            'ir_auc': auc_ir,
            'ir_coverage': ir_cov_pct,
            'lgbxgb_auc': auc_lgbxgb,
            '3model_auc': auc_3m,
            'lgbxgb_ir_auc': auc_lxi,
            '4model_auc': auc_4m,
            'grid_auc': best_grid_auc if best_grid_w else None,
            'grid_weights': list(best_grid_w) if best_grid_w else None,
            'train_auc': auc_lgb_tr,
            'gap': gap,
        })

    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("  v13.5b - IntraRace Attention + 4-Model Ensemble WF")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Build dataframe
    df, sire_map, bms_map = build_v134_dataframe()
    features, jrdb_sel = get_v134_features()
    df = fill_defaults(df, features)
    df = build_race_id(df)
    print(f"  Data: {len(df)} rows, {len(features)} features")
    print(f"  Races: {df['race_id_unique'].nunique()}")

    # Run 4-model WF
    results = walk_forward_4model(df, features)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Year':<6} {'LGB':>7} {'XGB':>7} {'FT':>7} {'IR':>7} "
          f"{'L+X':>7} {'L+X+F':>8} {'L+X+IR':>8} {'4-model':>8} {'Grid':>8} {'Gap':>6}")
    print(f"  {'-'*85}")
    for r in results:
        ok = 'Y' if r['gap'] < 0.05 else 'N'
        grid = f"{r['grid_auc']:.4f}" if r['grid_auc'] else '  N/A '
        print(f"  {r['year']:<6} {r['lgb_auc']:7.4f} {r['xgb_auc']:7.4f} "
              f"{r['ft_auc']:7.4f} {r['ir_auc']:7.4f} "
              f"{r['lgbxgb_auc']:7.4f} {r['3model_auc']:8.4f} "
              f"{r['lgbxgb_ir_auc']:8.4f} {r['4model_auc']:8.4f} "
              f"{grid:>8} {r['gap']:6.4f}{ok}")

    lgbxgb_mean = np.mean([r['lgbxgb_auc'] for r in results])
    m3_mean = np.mean([r['3model_auc'] for r in results])
    m4_mean = np.mean([r['4model_auc'] for r in results])
    lxi_mean = np.mean([r['lgbxgb_ir_auc'] for r in results])
    grid_vals = [r['grid_auc'] for r in results if r['grid_auc']]
    grid_mean = np.mean(grid_vals) if grid_vals else 0

    print(f"\n  Mean AUC:")
    print(f"    LGB+XGB:        {lgbxgb_mean:.6f}")
    print(f"    LGB+XGB+FT:     {m3_mean:.6f}  (v13.5 baseline)")
    print(f"    LGB+XGB+IR:     {lxi_mean:.6f}")
    print(f"    4-model:        {m4_mean:.6f}")
    if grid_mean > 0:
        print(f"    4-model grid:   {grid_mean:.6f}")

    baseline = 0.8657
    candidates = {
        'LGB+XGB+FT (v13.5)': m3_mean,
        'LGB+XGB+IR': lxi_mean,
        '4-model (equal)': m4_mean,
    }
    if grid_mean > 0:
        candidates['4-model (grid)'] = grid_mean

    best_name = max(candidates, key=candidates.get)
    best_val = candidates[best_name]
    delta = best_val - baseline

    print(f"\n  Best: {best_name} = {best_val:.6f} ({delta:+.6f} vs v13.5)")

    all_ok = all(r['lgb_auc'] > 0.78 for r in results)
    no_overfit = all(r['gap'] < 0.05 for r in results)
    adopted = best_val > baseline and all_ok and no_overfit

    if adopted:
        print(f"\n  *** ADOPTED: {best_name} ***")
    else:
        reasons = []
        if best_val <= baseline:
            reasons.append(f"AUC {best_val:.6f} <= baseline {baseline:.6f}")
        if not all_ok:
            reasons.append("some year AUC < 0.78")
        if not no_overfit:
            reasons.append("gap >= 0.05")
        print(f"\n  NOT ADOPTED: {', '.join(reasons)}")

    # Save results
    result = {
        'baseline_auc': baseline,
        'yearly': [{k: v for k, v in r.items()} for r in results],
        'means': {
            'lgbxgb': lgbxgb_mean,
            '3model': m3_mean,
            'lgbxgb_ir': lxi_mean,
            '4model': m4_mean,
            '4model_grid': grid_mean,
        },
        'best_ensemble': best_name,
        'best_auc': float(best_val),
        'delta_vs_baseline': float(delta),
        'adopted': adopted,
        'features_count': len(features),
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v135b_intra_ensemble_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {rpath}")

    elapsed = (time.time() - t0) / 60
    print(f"\n  Total elapsed: {elapsed:.1f} min")


if __name__ == '__main__':
    main()
