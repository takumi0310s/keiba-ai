#!/usr/bin/env python3
"""Fable sweep Phase 4: 全145特徴 リーク監査 (読み取り専用・本番不変)。
leak-free v2 cache 上で、過去4回のリーク類型 (dam_top3r / sib / SKB / ze) の統計署名を
全特徴に適用する総当たり検査。

検査項目 (特徴ごと):
 1. single-feature AUC (target=複勝圏)。単独特徴で AUC>0.70 は要精査 (SKB型)
 2. finish との spearman |corr| (post-race 直結署名。skb_kishi_code_3 は 0.137 だった)
 3. 勝者 vs 大敗者 (finish>=10) の非デフォルト率非対称 (SKB: 1着 0-rate 15% / 敗者 49%)
 4. 反市場テスト: 人気下位 (popularity>=6) かつ 特徴がレース内 top1 の馬の複勝率。
    base (人気6位以下全体の複勝率 ~10%) の 2.5 倍超は「結果を見ている」疑い (ze型: 44%勝ち)
出力: data/fable_override_audit.json + フラグ特徴の一覧 (重大度順)
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

CACHE = 'data/_v15_optuna_df_cache_leakfree_v2.pkl.gz'


def fast_auc(y, x):
    """rank-based AUC (Mann-Whitney)。NaN は除外。"""
    m = ~(np.isnan(x) | np.isnan(y))
    y, x = y[m], x[m]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or len(y) < 100:
        return float('nan')
    r = pd.Series(x).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    print('loading cache...')
    d = pd.read_pickle(CACHE)
    df, feats = d['df'], d['features']
    print(f'rows={len(df)} feats={len(feats)}')
    # 評価期間 = WF評価窓 2023-25。cache の year は2桁 (15-25)
    yr = pd.to_numeric(df['year'], errors='coerce')
    yr = yr.where(yr >= 100, yr + 2000)
    df = df[yr >= 2023].copy()
    print(f'2023-25 rows={len(df)}')
    assert len(df) > 50000, f'評価窓が空 ({len(df)}行) — year 列の形式を確認せよ'
    y = (pd.to_numeric(df['finish'], errors='coerce') <= 3).astype(float).values
    finish = pd.to_numeric(df['finish'], errors='coerce').values
    pop = pd.to_numeric(df.get('popularity'), errors='coerce').values if 'popularity' in df.columns else None
    # ★race_id は馬番込み行ユニーク。レースキーは race_id_str (検証済: 38,002レース)★
    rid = df['race_id_str'].values
    n_races = pd.Series(rid).nunique()
    assert n_races < len(df) / 5, f'レースキーがユニークすぎる: {n_races} groups / {len(df)} rows'
    print(f'races={n_races}')

    base_unpop_top3 = float(np.nanmean(y[(pop >= 6)])) if pop is not None else float('nan')
    print(f'base: 人気6位以下の複勝率 = {base_unpop_top3:.3f}')

    res = []
    g = df.groupby(rid, sort=False)
    for i, f in enumerate(feats, 1):
        if f not in df.columns:
            res.append({'feature': f, 'missing': True})
            continue
        x = pd.to_numeric(df[f], errors='coerce').values
        auc = fast_auc(y, x)
        # finish corr (sample 100k で十分)
        idx = np.random.RandomState(42).choice(len(df), size=min(100000, len(df)), replace=False)
        m = ~(np.isnan(x[idx]) | np.isnan(finish[idx]))
        sp = spearmanr(x[idx][m], finish[idx][m])[0] if m.sum() > 1000 else float('nan')
        # 勝者 vs 大敗 デフォルト(最頻値)率非対称
        v = pd.Series(x)
        mode = v.mode()
        mode = float(mode.iloc[0]) if len(mode) else float('nan')
        w_mask = finish == 1
        l_mask = finish >= 10
        asym = float('nan')
        if not np.isnan(mode) and w_mask.sum() > 500 and l_mask.sum() > 500:
            w0 = float(np.nanmean((x[w_mask] == mode)))
            l0 = float(np.nanmean((x[l_mask] == mode)))
            asym = l0 - w0  # SKB型: 敗者の方がデフォルトが多い → 正に大
        # 反市場テスト: レース内で特徴値が最大 かつ 人気>=6 の馬の複勝率
        anti = float('nan')
        if pop is not None and not np.isnan(auc):
            s = pd.Series(x, index=df.index)
            ranks = s.groupby(rid).rank(ascending=False, method='first')
            sel = (ranks.values == 1) & (pop >= 6)
            if sel.sum() > 300:
                anti = float(np.nanmean(y[sel]))
        flags = []
        if not np.isnan(auc) and (auc > 0.70 or auc < 0.30):
            flags.append('AUC')
        if not np.isnan(sp) and abs(sp) > 0.13:
            flags.append('FINISH_CORR')
        if not np.isnan(asym) and abs(asym) > 0.25:
            flags.append('ASYM')
        if not np.isnan(anti) and base_unpop_top3 > 0 and anti > base_unpop_top3 * 2.5:
            flags.append('ANTI_MARKET')
        res.append({'feature': f, 'auc': None if np.isnan(auc) else round(auc, 4),
                    'finish_sp': None if np.isnan(sp) else round(sp, 4),
                    'asym': None if np.isnan(asym) else round(asym, 4),
                    'anti_market_top3': None if np.isnan(anti) else round(anti, 4),
                    'flags': flags})
        if i % 20 == 0:
            print(f'  {i}/{len(feats)}')
    flagged = [r for r in res if r.get('flags')]
    out = {'n_features': len(feats), 'eval_window': '2023-25', 'n_rows': int(len(df)),
           'base_unpop_top3': round(base_unpop_top3, 4), 'flagged': flagged, 'all': res}
    json.dump(out, open('data/fable_override_audit.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"\nフラグ付き特徴: {len(flagged)}/{len(feats)}")
    for r in sorted(flagged, key=lambda r: -(abs(r.get('finish_sp') or 0))):
        print(f"  {r['feature']:36s} AUC={r.get('auc')} sp={r.get('finish_sp')} asym={r.get('asym')} anti={r.get('anti_market_top3')} {r['flags']}")
    print('-> data/fable_override_audit.json')


if __name__ == '__main__':
    main()
