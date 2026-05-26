"""オッズ依存分離分析: 騎手指数の能力 vs 人気代理を分解。

分析のみ。V15 production 完全不変。
"""
from __future__ import annotations
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 設定
# ============================================================
JOCKEY_FEATS = ['paci_jockey_exp_wr', 'paci_jockey_exp_3rd']
ODDS_FEATS   = ['paci_ninki_idx', 'oz_tansho_base_log', 'oz_base_pop_rank', 'oz_fukusho_base_log']
PERF_FEATS   = ['horse_career_top3r', 'horse_career_wr', 'avg_finish_3r', 'best_finish_3r']
MODEL_SCORE_FEATS = ['jrdb_idm', 'jrdb_ze_idm_avg', 'paci_jockey_exp_wr']

OUT = REPO / 'data' / 'odds_dependency_analysis.json'


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 1. データ読み込み
# ============================================================
section("1. データ読み込み")
raw = pd.read_pickle(REPO / 'data' / '_v15_train_df_cache.pkl')
df = raw['df'] if isinstance(raw, dict) else raw

# V15モデル読み込み (LGB 特徴量リストのため)
with gzip.open(REPO / 'models' / 'v15_full_candidate.pkl.gz', 'rb') as f:
    m15 = pickle.load(f)
v15_feats = m15['features']
lgb_model = m15['models_last_fold']['lgb']
xgb_model = m15['models_last_fold']['xgb']

print(f"df rows: {len(df):,}  |  V15 feats: {len(v15_feats)}")

# ============================================================
# 2. 騎手指数 × オッズ 相関
# ============================================================
section("2. 騎手指数 × 人気指標 相関")

results = {}

avail = [c for c in JOCKEY_FEATS + ODDS_FEATS + PERF_FEATS if c in df.columns]
sub = df[avail].copy()
# 欠損を中央値で埋め (相関測定用)
for c in avail:
    sub[c] = sub[c].fillna(sub[c].median())

corr_matrix = sub.corr(method='pearson')
print("\n相関行列 (pearson):")
print(corr_matrix.loc[JOCKEY_FEATS, [c for c in ODDS_FEATS + PERF_FEATS if c in avail]].round(4).to_string())

# 騎手指数 vs 人気 corr の抜粋
jockey_odds_corr = {}
for jf in JOCKEY_FEATS:
    if jf not in sub.columns:
        continue
    for of in ODDS_FEATS:
        if of not in sub.columns:
            continue
        r, p = stats.pearsonr(sub[jf].values, sub[of].values)
        jockey_odds_corr[f"{jf}_x_{of}"] = {'r': round(float(r), 4), 'p': float(p)}
        print(f"  {jf} × {of}: r={r:.4f} (p={p:.2e})")

results['jockey_odds_corr'] = jockey_odds_corr

# ============================================================
# 3. 残差分析: 騎手指数から人気成分を除去
# ============================================================
section("3. 残差分析: 騎手指数の人気独立成分")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

residual_results = {}

for jf in JOCKEY_FEATS:
    if jf not in sub.columns:
        continue
    odds_cols = [c for c in ODDS_FEATS if c in sub.columns]
    X_odds = sub[odds_cols].values
    y_jockey = sub[jf].values

    # 人気でどれだけ騎手指数を説明できるか
    reg = LinearRegression().fit(X_odds, y_jockey)
    r2 = r2_score(y_jockey, reg.predict(X_odds))
    residual = y_jockey - reg.predict(X_odds)

    # 残差の予測力 (target との相関)
    target = (df['finish'] <= 3).astype(int).values if 'finish' in df.columns else np.zeros(len(df))
    r_orig, _ = stats.pearsonr(y_jockey, target)
    r_resid, _ = stats.pearsonr(residual, target)

    print(f"\n{jf}:")
    print(f"  人気系(R²) で説明できる割合: {r2:.1%}  → 人気代理成分 {r2:.1%}")
    print(f"  残り純粋能力成分: {1-r2:.1%}")
    print(f"  元の target 相関: r={r_orig:.4f}")
    print(f"  残差の target 相関: r={r_resid:.4f}")
    print(f"  残差保持率: {abs(r_resid)/max(abs(r_orig),1e-9):.1%}")

    residual_results[jf] = {
        'r2_explained_by_odds': round(float(r2), 4),
        'popularity_proxy_pct': round(float(r2) * 100, 1),
        'ability_independent_pct': round((1 - float(r2)) * 100, 1),
        'original_target_corr': round(float(r_orig), 4),
        'residual_target_corr': round(float(r_resid), 4),
        'residual_retention_pct': round(abs(float(r_resid)) / max(abs(float(r_orig)), 1e-9) * 100, 1),
    }
    sub[f'{jf}_residual'] = residual

results['residual_analysis'] = residual_results

# ============================================================
# 4. V15 スコア算出 + 「戦績悪い人気馬」 分析
# ============================================================
section("4. 戦績が悪いのにV15スコア上位の馬")

feat_cols = [f for f in v15_feats if f in df.columns]
missing_feats = [f for f in v15_feats if f not in df.columns]
if missing_feats:
    print(f"  missing {len(missing_feats)} feats, using 0 fill")

df_score = df.copy()
for f in missing_feats:
    df_score[f] = 0.0
df_score[feat_cols] = df_score[feat_cols].fillna(0.0)

X_all = df_score[v15_feats].values.astype(np.float32)

import xgboost as xgb
p_lgb = lgb_model.predict(X_all)
p_xgb = xgb_model.predict(xgb.DMatrix(X_all, feature_names=v15_feats))
df_score['v15_score'] = 0.5 * p_lgb + 0.5 * p_xgb

# レース内スコアランク
df_score['race_id_str'] = df_score['race_id'].astype(str).str.zfill(12)
df_score['score_rank_in_race'] = df_score.groupby('race_id_str')['v15_score'].rank(ascending=False)
df_score['career_top3r'] = df_score.get('horse_career_top3r', pd.Series(np.nan, index=df_score.index))
df_score['actual_top3'] = (df_score['finish'] <= 3).astype(int) if 'finish' in df_score.columns else 0

# 「戦績悪い人気馬」: horse_career_top3r < 0.25 かつ V15スコア上位3位以内
low_career = df_score['career_top3r'].notna() & (df_score['career_top3r'] < 0.25)
high_score = df_score['score_rank_in_race'] <= 3
suspect = df_score[low_career & high_score].copy()

print(f"\n戦績悪い(top3r<25%) × V15スコア上位3位 N={len(suspect):,}")
if len(suspect) > 0:
    print(f"  実際の的中率: {suspect['actual_top3'].mean():.1%}")
    print(f"  平均 career_top3r: {suspect['career_top3r'].mean():.3f}")

    # これらの馬の特徴量 z-score 寄与
    all_mean = df_score[[f for f in v15_feats if f in df_score.columns]].mean()
    all_std  = df_score[[f for f in v15_feats if f in df_score.columns]].std().replace(0, 1)

    suspect_mean = suspect[[f for f in v15_feats if f in suspect.columns]].mean()
    z_contrib = ((suspect_mean - all_mean) / all_std).sort_values(ascending=False)

    print("\n  z-score 寄与 TOP15 (suspect馬 vs 全体平均):")
    for feat, z in z_contrib.head(15).items():
        print(f"    {feat:<35} z={z:+.3f}")

    print("\n  z-score 寄与 BOTTOM10 (suspect馬が低い特徴):")
    for feat, z in z_contrib.tail(10).items():
        print(f"    {feat:<35} z={z:+.3f}")

    results['suspect_horses'] = {
        'n': int(len(suspect)),
        'hit_rate': round(float(suspect['actual_top3'].mean()), 4),
        'avg_career_top3r': round(float(suspect['career_top3r'].mean()), 4),
        'top15_z_contrib': {k: round(float(v), 4) for k, v in z_contrib.head(15).items()},
        'bottom10_z_contrib': {k: round(float(v), 4) for k, v in z_contrib.tail(10).items()},
    }

# ============================================================
# 5. 「正常馬」との比較: 戦績良い × V15高スコア
# ============================================================
section("5. 参照: 戦績良い高スコア馬との比較")

good_career = df_score['career_top3r'].notna() & (df_score['career_top3r'] >= 0.35)
good_suspect = df_score[good_career & high_score].copy()
print(f"戦績良い(top3r>=35%) × V15スコア上位3位 N={len(good_suspect):,}")
if len(good_suspect) > 0:
    print(f"  実際の的中率: {good_suspect['actual_top3'].mean():.1%}")
    print(f"  平均 career_top3r: {good_suspect['career_top3r'].mean():.3f}")

    good_mean = good_suspect[[f for f in v15_feats if f in good_suspect.columns]].mean()
    z_good = ((good_mean - all_mean) / all_std).sort_values(ascending=False)
    print("\n  z-score 寄与 TOP10 (good馬):")
    for feat, z in z_good.head(10).items():
        print(f"    {feat:<35} z={z:+.3f}")

    results['good_horses'] = {
        'n': int(len(good_suspect)),
        'hit_rate': round(float(good_suspect['actual_top3'].mean()), 4),
        'top10_z_contrib': {k: round(float(v), 4) for k, v in z_good.head(10).items()},
    }

# ============================================================
# 6. オッズ除外 (V16相当) での影響測定
# ============================================================
section("6. オッズ系除外 (V16相当) の影響推計")

ODDS_DIRECT = ['paci_ninki_idx', 'oz_tansho_base_log', 'oz_fukusho_base_log',
               'oz_base_pop_rank', 'odds_change_rate', 'pop_rank_change']
ODDS_INDIRECT = ['paci_jockey_exp_wr', 'paci_jockey_exp_3rd']  # 人気代理成分あり

# 直接除外した場合の LGB gain 合計
lgb_imp = dict(zip(lgb_model.feature_name(), lgb_model.feature_importance(importance_type='gain')))
total_gain = sum(lgb_imp.values()) or 1

direct_gain = sum(lgb_imp.get(f, 0) for f in ODDS_DIRECT)
indirect_gain = sum(lgb_imp.get(f, 0) for f in ODDS_INDIRECT)
jockey_r2 = np.mean([residual_results[jf]['r2_explained_by_odds']
                     for jf in JOCKEY_FEATS if jf in residual_results])
indirect_odds_gain = indirect_gain * jockey_r2  # 人気代理成分分のみ

print(f"\n直接オッズ gain 合計: {direct_gain:,.0f}  ({direct_gain/total_gain:.1%} of total)")
print(f"騎手指数 gain 合計: {indirect_gain:,.0f}  ({indirect_gain/total_gain:.1%})")
print(f"  うち人気代理成分 ({jockey_r2:.1%}): {indirect_odds_gain:,.0f}  ({indirect_odds_gain/total_gain:.1%})")
print(f"オッズ依存合計 (直接+間接代理): {(direct_gain+indirect_odds_gain)/total_gain:.1%}")
print(f"V16 (直接除外のみ) で消えるオッズ依存: {direct_gain/(direct_gain+indirect_odds_gain):.1%}")
print(f"残留オッズ依存 (騎手経由): {indirect_odds_gain/(direct_gain+indirect_odds_gain):.1%}")

results['odds_dependency_summary'] = {
    'direct_odds_gain_pct': round(direct_gain / total_gain * 100, 2),
    'jockey_total_gain_pct': round(indirect_gain / total_gain * 100, 2),
    'jockey_r2_by_odds': round(float(jockey_r2), 4),
    'jockey_odds_proxy_gain_pct': round(indirect_odds_gain / total_gain * 100, 2),
    'total_odds_dependency_pct': round((direct_gain + indirect_odds_gain) / total_gain * 100, 2),
    'v16_removes_odds_pct': round(direct_gain / (direct_gain + indirect_odds_gain) * 100, 1),
    'remaining_odds_via_jockey_pct': round(indirect_odds_gain / (direct_gain + indirect_odds_gain) * 100, 1),
}

# ============================================================
# 7. 結果保存
# ============================================================
section("7. 結果保存")
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved: {OUT}")

# ============================================================
# 最終サマリー
# ============================================================
section("最終サマリー")
r2_avg = jockey_r2
r_resid_avg = np.mean([v['residual_retention_pct'] for v in residual_results.values()])
total_odds_pct = results['odds_dependency_summary']['total_odds_dependency_pct']
v16_removes_pct = results['odds_dependency_summary']['v16_removes_odds_pct']

print(f"\n騎手指数×人気 相関:")
for k, v in jockey_odds_corr.items():
    if 'ninki' in k or 'tansho' in k:
        print(f"  {k}: r={v['r']}")

print(f"\n騎手指数の人気代理成分: {r2_avg:.1%}")
print(f"残差 (純粋能力) の予測力保持率: {r_resid_avg:.1f}%")
print(f"\nオッズ依存 (直接+間接): {total_odds_pct:.1f}% of LGB gain")
print(f"V16 (直接除外) で消えるオッズ依存: {v16_removes_pct:.1f}%")

if 'suspect_horses' in results:
    sh = results['suspect_horses']
    print(f"\n戦績悪い人気馬 N={sh['n']}, 的中率={sh['hit_rate']:.1%}")
    top_feat = list(sh['top15_z_contrib'].items())[0]
    print(f"  最大 z-score 要因: {top_feat[0]} (z={top_feat[1]:+.3f})")

print("\n分離分析完了。 data/odds_dependency_analysis.json に保存。")
