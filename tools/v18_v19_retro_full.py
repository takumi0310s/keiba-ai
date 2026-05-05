"""v18 単勝 + v19 複勝 完全 retro (5/2, 5/3).

For each race in 5/2-5/3:
1. Build features (parse_shutuba + build_features + add_v17_features)
2. Run v18 LGB+XGB → P(1着)
3. Run v19 LGB+XGB → P(top3)
4. Match with odds_base_DATE.csv (jrdb 基準オッズ)
5. Compute EV = P × odds
6. Apply Phase 2 filter (単勝 p>=0.5 ev>=1.2, 複勝 p>=0.7 ev>=1.1)
7. Match with daily_results for actual is_win / is_top3
8. Compute realized ROI vs BT 2025 OOS (295% / 149%)
9. 楽観バイアス確定: BT × X% = 実 retro

出力: data/v18/v18_v19_retro_full_result.md + CSV
"""
import sys, os, io, contextlib, time, argparse
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

from predict_core import parse_shutuba, build_features, load_models, set_horse_defaults
from predict_v17_morning_pipeline import add_v17_features_to_race_df, predict_v17_for_race
import importlib.util
spec = importlib.util.spec_from_file_location('pred_top', 'tools/predict_v17_top_races_5_3.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
add_v162_safe = mod.add_v162_safe

DATES = ['20260502', '20260503']

# Phase 2 filter params
TANSHO_PROB_MIN = 0.5
TANSHO_EV_MIN = 1.2
FUKUSHO_PROB_MIN = 0.7
FUKUSHO_EV_MIN = 1.1

# Looser params for sweep
TANSHO_THRESH_GRID = [(0.3, 1.0), (0.4, 1.2), (0.5, 1.2)]
FUKUSHO_THRESH_GRID = [(0.5, 1.0), (0.6, 1.1), (0.7, 1.1)]


def maybe_normalize(df, prob_col, race_col='race_id', method=None, T=1.0):
    """Apply race-level normalization in-place if method is set.

    Returns the column name to use for downstream bet evaluation.
    """
    if not method or method == 'none':
        return prob_col
    # local import to keep retro lean if normalize not used
    from race_normalize import normalize_per_race
    norm_col = prob_col + '_norm'
    df[norm_col] = normalize_per_race(df, prob_col=prob_col, race_col=race_col, method=method, T=T)
    return norm_col


def load_v18_v19():
    v18_lgb = lgb.Booster(model_file='data/v18/models/v18_tansho_lgb.txt')
    v18_xgb = xgb.Booster(); v18_xgb.load_model('data/v18/models/v18_tansho_xgb.json')
    v19_lgb = lgb.Booster(model_file='data/v18/models/v19_fukusho_lgb.txt')
    # v19 xgb may not exist
    v19_xgb = None
    if os.path.exists('data/v18/models/v19_fukusho_xgb.json'):
        v19_xgb = xgb.Booster()
        v19_xgb.load_model('data/v18/models/v19_fukusho_xgb.json')
    return v18_lgb, v18_xgb, v19_lgb, v19_xgb


def predict_one_race(race_id, model_data, v18_lgb, v18_xgb, v19_lgb, v19_xgb):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            race_name, horses, _, race_info = parse_shutuba(race_id)
            for h in horses:
                set_horse_defaults(h)
            race_df = build_features(horses, race_info, model_data)
            race_df = race_df.rename(columns={'馬番': 'umaban', '馬名': 'horse_name', '血統登録番号': 'blood_num'})
            race_df['_nk_race_id'] = race_id
            race_df['race_id'] = race_id
            race_df['umaban'] = pd.to_numeric(race_df['umaban'], errors='coerce').astype('Int64')
            race_df = add_v162_safe(race_df)
            race_df = add_v17_features_to_race_df(race_df, has_nk_race_id=True)

            # v18 inference (DMatrix with feature_names for xgb)
            feats_18 = v18_lgb.feature_name()
            for f in feats_18:
                if f not in race_df.columns:
                    race_df[f] = 0
            X18df = race_df[feats_18].apply(pd.to_numeric, errors='coerce').fillna(0)
            p18_lgb = v18_lgb.predict(X18df.values)
            p18_xgb = v18_xgb.predict(xgb.DMatrix(X18df.values, feature_names=feats_18))
            p18 = (p18_lgb + p18_xgb) / 2

            # v19 inference
            feats_19 = v19_lgb.feature_name()
            for f in feats_19:
                if f not in race_df.columns:
                    race_df[f] = 0
            X19df = race_df[feats_19].apply(pd.to_numeric, errors='coerce').fillna(0)
            p19_lgb = v19_lgb.predict(X19df.values)
            if v19_xgb is not None:
                p19_xgb = v19_xgb.predict(xgb.DMatrix(X19df.values, feature_names=feats_19))
                p19 = (p19_lgb + p19_xgb) / 2
            else:
                p19 = p19_lgb

        return {
            'race_id': race_id,
            'race_name': race_name,
            'umaban_list': race_df['umaban'].tolist(),
            'p18': p18.tolist(),
            'p19': p19.tolist(),
        }
    except Exception as e:
        return {'race_id': race_id, 'error': str(e)[:120]}


def evaluate_from_csv(predictions_csv, normalize_method='none', T=1.0, output_md='data/v18/v18_v19_retro_full_result.md', BET=100):
    """既存 predictions CSV から normalize + ROI 評価のみ実行。inference スキップ。"""
    df = pd.read_csv(predictions_csv, dtype={'race_id': str})
    print(f"loaded {len(df)} rows from {predictions_csv}")

    # normalize
    p_tansho_col = maybe_normalize(df, 'p_tansho', 'race_id', normalize_method, T)
    p_fukusho_col = maybe_normalize(df, 'p_fukusho', 'race_id', normalize_method, T)
    if normalize_method and normalize_method != 'none':
        df['ev_tansho_eval'] = df[p_tansho_col] * df['odds']
        df['ev_fukusho_eval'] = df[p_fukusho_col] * df['odds'] * 0.3
        ev_t_col = 'ev_tansho_eval'
        ev_f_col = 'ev_fukusho_eval'
    else:
        ev_t_col = 'ev_tansho'
        ev_f_col = 'ev_fukusho'

    print(f"\n=== normalize: {normalize_method} (T={T}) ===")
    if normalize_method and normalize_method != 'none':
        g = df.groupby('race_id')[p_tansho_col].agg(['max','sum'])
        print(f"  tansho race max mean: {g['max'].mean():.3f}, sum mean: {g['sum'].mean():.3f}")

    df_w = df[df['winner_known'] == 1].copy()
    n_winner_known = df_w['race_id'].nunique()
    print(f"\nwinner_known: {df['winner_known'].sum()}/{len(df)} horses, {n_winner_known} races")

    print(f"\n=== v18 単勝 retro (winner_known races: {n_winner_known}) ===")
    tansho_results = []
    for prob_t, ev_t in TANSHO_THRESH_GRID:
        m = (df_w[p_tansho_col] >= prob_t) & (df_w[ev_t_col] >= ev_t) & (df_w['odds'] > 0)
        n_bet = int(m.sum())
        if n_bet == 0:
            print(f"  p>={prob_t} ev>={ev_t}: bet=0")
            tansho_results.append({'prob': prob_t, 'ev': ev_t, 'bet': 0, 'win': 0, 'inv': 0, 'pay': 0, 'roi': None})
            continue
        wins = m & (df_w['is_win'] == 1)
        n_win = int(wins.sum())
        inv = n_bet * BET
        pay = float((wins.astype(int) * df_w['odds'] * BET).sum())
        roi = pay / inv * 100 if inv > 0 else 0
        print(f"  p>={prob_t} ev>={ev_t}: bet={n_bet} win={n_win} hit={n_win/n_bet*100:.1f}% inv={inv} pay={int(pay):,} ROI={roi:.1f}%")
        tansho_results.append({'prob': prob_t, 'ev': ev_t, 'bet': n_bet, 'win': n_win, 'inv': inv, 'pay': pay, 'roi': roi})

    print(f"\n=== v19 複勝 retro (全レース、is_top3 from trio_result) ===")
    fukusho_results = []
    for prob_f, ev_f in FUKUSHO_THRESH_GRID:
        m = (df[p_fukusho_col] >= prob_f) & (df[ev_f_col] >= ev_f) & (df['odds'] > 0)
        n_bet = int(m.sum())
        if n_bet == 0:
            print(f"  p>={prob_f} ev>={ev_f}: bet=0")
            fukusho_results.append({'prob': prob_f, 'ev': ev_f, 'bet': 0, 'hit': 0, 'inv': 0, 'pay': 0, 'roi': None})
            continue
        hits = m & (df['is_top3'] == 1)
        n_hit = int(hits.sum())
        inv = n_bet * BET
        pay = float((hits.astype(int) * df['odds'] * 0.3 * BET).sum())
        roi = pay / inv * 100 if inv > 0 else 0
        print(f"  p>={prob_f} ev>={ev_f}: bet={n_bet} hit={n_hit} hit%={n_hit/n_bet*100:.1f}% inv={inv} pay~={int(pay):,} ROI~={roi:.1f}%")
        fukusho_results.append({'prob': prob_f, 'ev': ev_f, 'bet': n_bet, 'hit': n_hit, 'inv': inv, 'pay': pay, 'roi': roi})

    # write report
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('# v18/v19 完全 retro (5/2-5/3) — race-level normalization 版\n\n')
        f.write(f'生成: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## 設定\n\n- 期間: {DATES}\n- 全 horses: {len(df)}\n- winner_known races: {n_winner_known}\n- normalize method: {normalize_method} (T={T})\n\n')
        f.write('## 単勝 (v18) retro\n\n')
        f.write('| prob_min | ev_min | bet | win | hit% | inv | pay | ROI |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in tansho_results:
            if r['bet'] == 0:
                f.write(f'| {r["prob"]} | {r["ev"]} | 0 | - | - | - | - | - |\n')
            else:
                hit = r['win'] / r['bet'] * 100
                f.write(f'| {r["prob"]} | {r["ev"]} | {r["bet"]} | {r["win"]} | {hit:.1f}% | {r["inv"]:,} | {int(r["pay"]):,} | **{r["roi"]:.1f}%** |\n')
        f.write('\n## 複勝 (v19) retro (簡易: 複勝odds = tansho × 0.3 仮定)\n\n')
        f.write('| prob_min | ev_min | bet | hit | hit% | inv | pay~ | ROI~ |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in fukusho_results:
            if r['bet'] == 0:
                f.write(f'| {r["prob"]} | {r["ev"]} | 0 | - | - | - | - | - |\n')
            else:
                hit = r['hit'] / r['bet'] * 100
                f.write(f'| {r["prob"]} | {r["ev"]} | {r["bet"]} | {r["hit"]} | {hit:.1f}% | {r["inv"]:,} | {int(r["pay"]):,} | **{r["roi"]:.1f}%** |\n')
    print(f"\n[OK] {output_md}")
    return {'tansho': tansho_results, 'fukusho': fukusho_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', choices=['none','softmax','power','rank'], default='none',
                        help='race-level normalization method')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for softmax/power')
    parser.add_argument('--from-csv', default=None,
                        help='既存 predictions CSV から評価のみ (inference skip)')
    parser.add_argument('--output-md', default='data/v18/v18_v19_retro_full_result.md')
    args = parser.parse_args()

    if args.from_csv:
        evaluate_from_csv(args.from_csv, normalize_method=args.normalize, T=args.T, output_md=args.output_md)
        return

    print(f"=== v18/v19 full retro START {datetime.now()} ===")
    md = load_models()
    v18_lgb, v18_xgb, v19_lgb, v19_xgb = load_v18_v19()
    print(f"v18 features: {len(v18_lgb.feature_name())}, v19: {len(v19_lgb.feature_name())}")

    # Aggregate all 5/2-5/3 horses
    all_horse_rows = []
    t_start = time.time()

    for d in DATES:
        pred = pd.read_csv(f'data/daily_predictions/{d}.csv', dtype={'race_id':str})
        odds = pd.read_csv(f'data/odds_base_{d}.csv', dtype={'race_id':str})
        odds['horse_num'] = pd.to_numeric(odds['horse_num'], errors='coerce').astype('Int64')
        res = pd.read_csv(f'data/daily_results/{d}.csv', encoding='utf-8-sig', dtype={'race_id':str})
        res = res.drop_duplicates(subset=['race_id'], keep='last')

        # finish info: derive umaban-level is_win / is_top3 from trio_result and finishes
        # daily_results has top1_finish/top2_finish/top3_finish (V15軸の着順)
        # We need actual 1-3着 馬番 → trio_result has them
        for _, rr in res.iterrows():
            rid = str(rr['race_id'])
            tres = str(rr.get('trio_result',''))
            try:
                top3_nums = [int(x) for x in tres.split('-') if x.strip()]
            except: top3_nums = []
            if len(top3_nums) != 3:
                continue
            # 1着 inference: 着順が記録されてないので、payout map と一致しない
            # 簡易: trio_result の最初が必ず1着とは限らない。一般に sort されてる
            # Actually トリオは 1-2-3着のソートで保存されてる (馬番昇順) ので 1着特定不能
            # → 追加で trio_result_with_finish が必要
            # 一旦 trio set として is_top3 のみ判定可能、is_win は別ソース必要

        races_d = sorted(pred['race_id'].astype(str).unique())
        print(f"\n=== {d}: {len(races_d)} races ===", flush=True)

        for i, rid in enumerate(races_d):
            t0 = time.time()
            r = predict_one_race(rid, md, v18_lgb, v18_xgb, v19_lgb, v19_xgb)
            if 'error' in r:
                print(f"  [{i+1}/{len(races_d)}] {rid} ERROR: {r['error']}", flush=True)
                continue

            # Match odds
            odds_r = odds[odds['race_id']==rid][['horse_num','odds','pop_rank']]
            odds_map = {int(row['horse_num']): {'odds': float(row['odds']), 'pop': int(row['pop_rank'])}
                        for _, row in odds_r.iterrows()}

            # Match result (trio_result for is_top3, top1_finish for finish position of pred top1)
            res_r = res[res['race_id']==rid]
            if len(res_r) == 0:
                continue
            rr = res_r.iloc[0]
            tres_str = str(rr.get('trio_result',''))
            try:
                trio_set = set(int(x) for x in tres_str.split('-') if x.strip())
            except:
                trio_set = set()
            # 1着 馬番: trio_result の中で 1着 が直接わからない場合、top1_finish などから推定不能
            # → daily_predictions の top1_num + top1_finish=1 の場合のみ確定
            top1_pred_num = int(rr.get('top1_finish', 0))  # this is finish pos of pred top1 — not the winner

            # 1着馬番 unknown のため、is_win は trio_set 内かつ position==1 で判定が必要
            # 簡易: trio_set 含む = is_top3, 1着は不明 → 単勝 retro は trio_set 内でかつ "最低人気以外" でフォールバック
            # 厳密: daily_results にもう少し詳細な finish 情報が必要

            # 代替: daily_predictions の top1/top2/top3_num + top1_finish/top2_finish/top3_finish を使う
            # top1_finish=1 → pred top1 が1着
            top1_pred = pd.to_numeric(rr.get('top1_finish'), errors='coerce')
            top2_pred = pd.to_numeric(rr.get('top2_finish'), errors='coerce')
            top3_pred = pd.to_numeric(rr.get('top3_finish'), errors='coerce')
            # We need: 馬番 of 1着馬
            # 可能な情報: pred から top1_num/top2_num/top3_num と各馬の finish position
            # 1着馬番 = 馬番 with finish=1
            # daily_predictions lookup
            pred_r = pred[pred['race_id']==rid].iloc[0] if len(pred[pred['race_id']==rid]) else None
            winner_num = None
            if pred_r is not None:
                if int(rr.get('top1_finish', 0) or 0) == 1:
                    winner_num = int(pred_r.get('top1_num', 0) or 0)
                elif int(rr.get('top2_finish', 0) or 0) == 1:
                    winner_num = int(pred_r.get('top2_num', 0) or 0)
                elif int(rr.get('top3_finish', 0) or 0) == 1:
                    winner_num = int(pred_r.get('top3_num', 0) or 0)
                # else winner is some other horse

            # Per-horse rows
            for j, uma in enumerate(r['umaban_list']):
                uma_int = int(uma) if pd.notna(uma) else 0
                if uma_int == 0: continue
                p18 = float(r['p18'][j])
                p19 = float(r['p19'][j])
                o = odds_map.get(uma_int, {'odds':0,'pop':0})
                ev_t = p18 * o['odds']
                ev_f = p19 * o['odds']
                is_win_h = 1 if winner_num is not None and uma_int == winner_num else 0
                is_top3_h = 1 if uma_int in trio_set else 0
                all_horse_rows.append({
                    'date': d, 'race_id': rid, 'umaban': uma_int,
                    'p_tansho': p18, 'p_fukusho': p19,
                    'odds': o['odds'], 'pop_rank': o['pop'],
                    'ev_tansho': ev_t, 'ev_fukusho': ev_f,
                    'is_win': is_win_h, 'is_top3': is_top3_h,
                    'winner_known': 1 if winner_num is not None else 0,
                })

            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(races_d)}] {rid} t={elapsed:.0f}s p18_max={max(r['p18']):.3f} winner={winner_num}", flush=True)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f}min")
    print(f"Total horse rows: {len(all_horse_rows)}")

    df = pd.DataFrame(all_horse_rows)
    df.to_csv('data/v18/v18_v19_retro_full_predictions.csv', index=False, encoding='utf-8-sig')
    print(f"Saved data/v18/v18_v19_retro_full_predictions.csv")

    # === ROI computation per filter ===
    # 注: is_win は winner_num が pred top1-3 の中にいる場合のみ既知。
    #     その他のレースで実際に勝った馬が 4着以下のpred(top4-)場合、winner_known=0
    n_winner_unknown = (df['winner_known'] == 0).sum()
    n_horses = len(df)
    print(f"\nwinner_known: {df['winner_known'].sum()}/{n_horses} ({df['winner_known'].mean()*100:.1f}%)")
    print(f"  → winner_unknown のレースでは v18 単勝 retro 不正確 (実際の1着馬の馬番不明)")

    # フィルタ別 ROI
    BET = 100  # 100円ベース
    print("\n=== v18 単勝 retro (winner_known races only) ===")
    df_w = df[df['winner_known'] == 1].copy()
    print(f"winner_known horses: {len(df_w)}")
    for prob_t, ev_t in TANSHO_THRESH_GRID:
        m = (df_w['p_tansho'] >= prob_t) & (df_w['ev_tansho'] >= ev_t) & (df_w['odds'] > 0)
        n_bet = int(m.sum())
        if n_bet == 0:
            continue
        wins = m & (df_w['is_win'] == 1)
        n_win = int(wins.sum())
        inv = n_bet * BET
        pay = float((wins.astype(int) * df_w['odds'] * BET).sum())
        roi = pay/inv*100 if inv>0 else 0
        print(f"  p>={prob_t} ev>={ev_t}: bet={n_bet} win={n_win} inv={inv} pay={int(pay):,} ROI={roi:.1f}%")

    # 複勝
    print("\n=== v19 複勝 retro (全レース、is_top3 from trio_result) ===")
    for prob_f, ev_f in FUKUSHO_THRESH_GRID:
        m = (df['p_fukusho'] >= prob_f) & (df['ev_fukusho'] >= ev_f) & (df['odds'] > 0)
        n_bet = int(m.sum())
        if n_bet == 0:
            continue
        hits = m & (df['is_top3'] == 1)
        n_hit = int(hits.sum())
        inv = n_bet * BET
        # 複勝オッズは tansho odds の ~1/3 ベース推定。簡易: tansho odds × 0.3 を 複勝オッズとする
        pay = float((hits.astype(int) * df['odds'] * 0.3 * BET).sum())
        roi = pay/inv*100 if inv>0 else 0
        print(f"  p>={prob_f} ev>={ev_f}: bet={n_bet} hit={n_hit} inv={inv} pay~={int(pay):,} ROI~={roi:.1f}% (複勝odds=tansho×0.3 推定)")

    # === Save report ===
    report_path = 'data/v18/v18_v19_retro_full_result.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# v18/v19 完全 retro (5/2-5/3)\n\n')
        f.write(f'生成: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('## データ\n\n')
        f.write(f'- 期間: {DATES}\n')
        f.write(f'- 全 horses: {len(df)}\n')
        f.write(f'- winner_known horses: {df["winner_known"].sum()} ({df["winner_known"].mean()*100:.1f}%)\n')
        f.write(f'  - winner_unknown は実際の1着馬がpred top1-3外にいるレース → v18 retro 単勝判定不能\n\n')

        f.write('## 単勝 (v18) retro\n\n')
        f.write('| prob_min | ev_min | bet | win | inv | pay | ROI |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|\n')
        for prob_t, ev_t in TANSHO_THRESH_GRID:
            m = (df_w['p_tansho'] >= prob_t) & (df_w['ev_tansho'] >= ev_t) & (df_w['odds'] > 0)
            n_bet = int(m.sum())
            if n_bet == 0:
                f.write(f'| {prob_t} | {ev_t} | 0 | - | - | - | - |\n'); continue
            wins = m & (df_w['is_win'] == 1)
            n_win = int(wins.sum())
            inv = n_bet * BET
            pay = float((wins.astype(int) * df_w['odds'] * BET).sum())
            roi = pay/inv*100 if inv>0 else 0
            f.write(f'| {prob_t} | {ev_t} | {n_bet} | {n_win} | {inv:,} | {int(pay):,} | **{roi:.1f}%** |\n')
        f.write('\n')

        f.write('## 複勝 (v19) retro (簡易: 複勝odds = tansho × 0.3 仮定)\n\n')
        f.write('| prob_min | ev_min | bet | hit | inv | pay~ | ROI~ |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|\n')
        for prob_f, ev_f in FUKUSHO_THRESH_GRID:
            m = (df['p_fukusho'] >= prob_f) & (df['ev_fukusho'] >= ev_f) & (df['odds'] > 0)
            n_bet = int(m.sum())
            if n_bet == 0:
                f.write(f'| {prob_f} | {ev_f} | 0 | - | - | - | - |\n'); continue
            hits = m & (df['is_top3'] == 1)
            n_hit = int(hits.sum())
            inv = n_bet * BET
            pay = float((hits.astype(int) * df['odds'] * 0.3 * BET).sum())
            roi = pay/inv*100 if inv>0 else 0
            f.write(f'| {prob_f} | {ev_f} | {n_bet} | {n_hit} | {inv:,} | {int(pay):,} | **{roi:.1f}%** |\n')
        f.write('\n')

        # 楽観バイアス
        f.write('## 楽観バイアス (BT 2025 OOS vs 5/2-5/3 retro)\n\n')
        f.write('### 単勝\n\n')
        m = (df_w['p_tansho'] >= 0.5) & (df_w['ev_tansho'] >= 1.2) & (df_w['odds'] > 0)
        n_bet = int(m.sum())
        if n_bet > 0:
            wins = m & (df_w['is_win'] == 1)
            inv = n_bet * BET
            pay = float((wins.astype(int) * df_w['odds'] * BET).sum())
            roi_retro = pay/inv*100
            roi_bt = 295.1
            bias = roi_retro / roi_bt
            f.write(f'- BT 2025 OOS ROI (p>=0.5 ev>=1.2): **295.1%**\n')
            f.write(f'- 5/2-5/3 retro ROI: **{roi_retro:.1f}%** (n={n_bet})\n')
            f.write(f'- 楽観バイアス係数: **{bias:.2f}** (retro/BT)\n')
            if bias < 0.5:
                f.write(f'  → ⚠️ 大きな楽観バイアス、本番期待 ROI = BT × 0.5 程度が妥当\n')
            elif bias < 0.8:
                f.write(f'  → 🟡 中程度の楽観、本番期待 ROI = BT × 0.5-0.8\n')
            else:
                f.write(f'  → 🟢 楽観少、BT 値に近い実現性\n')
        else:
            f.write('- 該当 bet なし (sample 少)\n')
        f.write('\n')

        f.write('### 複勝\n\n')
        m = (df['p_fukusho'] >= 0.7) & (df['ev_fukusho'] >= 1.1) & (df['odds'] > 0)
        n_bet = int(m.sum())
        if n_bet > 0:
            hits = m & (df['is_top3'] == 1)
            inv = n_bet * BET
            pay = float((hits.astype(int) * df['odds'] * 0.3 * BET).sum())
            roi_retro = pay/inv*100
            roi_bt = 149.3
            bias = roi_retro / roi_bt
            f.write(f'- BT 2025 OOS ROI (p>=0.7 ev>=1.1): **149.3%**\n')
            f.write(f'- 5/2-5/3 retro ROI: **{roi_retro:.1f}%** (n={n_bet})\n')
            f.write(f'- 楽観バイアス係数: **{bias:.2f}**\n')

        f.write('\n## 結論\n\n')
        f.write('- 5/2-5/3 retro により BT 2025 OOS の楽観バイアス確定\n')
        f.write('- Phase 2.5 計画通り 5/16 以降 v18/v19 部分実弾投入時、本値で期待 ROI 修正\n')
        f.write('- Calibration: V18 mild under-conf (Phase 2 BT report)、Platt scaling で改善余地\n')

    print(f"\n[OK] Saved {report_path}")


if __name__ == '__main__':
    main()
