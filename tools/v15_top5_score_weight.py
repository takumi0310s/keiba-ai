"""V15 top-5 score weighting (paper only)

★ paper only — 実 cash 投票 logic は V15 + 既存 C4 のみ ★

概念:
  base_score = V15_prob (0.0-1.0)
  weighted_score = base_score × distance_factor × weight_change_factor × jockey_factor

  distance_factor: 将来実装 hook (現状 1.0 固定)
  weight_change_factor: ±5kg 以内=1.0、±10kg 以内=0.9、それ以上=0.8
  jockey_factor: 1番人気騎手=1.1、それ以外=1.0 (簡易版)

  ★ 現行 cumulative_results.csv に distance_factor の元データなし
     → 実装は簡易版 (weight_change のみ適用) + future hook

出力 formation: 重み付け後 top-5 の trio 7点 (top1軸 - top2,3 - top2~6)

race_notify_log_v2 連携:
  get_paper_formation(predictions_list) → str (formation string)
  predictions_list = [{num, score, name, weight_change}, ...] の sorted list
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Factor functions
# ---------------------------------------------------------------------------

def weight_change_factor(weight_change: Optional[float]) -> float:
    """馬体重変動に基づくファクター計算。

    Args:
        weight_change: 馬体重変動 (kg)。None または欠損の場合は 1.0 (no-op)。

    Returns:
        factor: 1.0 / 0.9 / 0.8
    """
    if weight_change is None:
        return 1.0
    try:
        wc = float(weight_change)
    except (TypeError, ValueError):
        return 1.0
    abs_wc = abs(wc)
    if abs_wc <= 5.0:
        return 1.0
    elif abs_wc <= 10.0:
        return 0.9
    else:
        return 0.8


def distance_factor(horse_data: dict) -> float:
    """距離適性ファクター (future hook — 現状 1.0 固定)。

    将来実装: 過去 N 走の距離別成績 (距離別 win_rate) を参照。
    現時点では累積結果 CSV に距離別馬履歴なし。
    """
    return 1.0


def jockey_factor(horse_data: dict) -> float:
    """騎手ファクター。

    horse_data に 'is_top_jockey' キーが True の場合 1.1。
    それ以外は 1.0。
    """
    if horse_data.get('is_top_jockey'):
        return 1.1
    return 1.0


# ---------------------------------------------------------------------------
# Core weighting
# ---------------------------------------------------------------------------

def compute_weighted_score(horse_data: dict) -> float:
    """1頭分の weighted_score を計算。

    horse_data keys:
        num          : 馬番 (int/str)
        score        : V15 base prob (float 0.0-1.0)
        name         : 馬名 (str, optional)
        weight_change: 馬体重変動 (float, optional)
        is_top_jockey: bool (optional)

    Returns:
        weighted_score (float)
    """
    base = float(horse_data.get('score', 0.0))
    wf = weight_change_factor(horse_data.get('weight_change'))
    df = distance_factor(horse_data)
    jf = jockey_factor(horse_data)
    return base * wf * df * jf


def rank_by_weighted_score(predictions_list: list[dict]) -> list[dict]:
    """predictions_list を weighted_score 降順でソートして返す。

    各要素に 'weighted_score' キーを追加する (in-place にはしない)。
    """
    if not predictions_list:
        return []
    scored = []
    for h in predictions_list:
        ws = compute_weighted_score(h)
        scored.append({**h, 'weighted_score': ws})
    return sorted(scored, key=lambda x: x['weighted_score'], reverse=True)


# ---------------------------------------------------------------------------
# Formation builder
# ---------------------------------------------------------------------------

def build_trio_formation(ranked: list[dict]) -> str:
    """top-5 から trio 7点 formation string を生成。

    formation: top1 軸 - top2,3 - top2~5
    形式: "1-3-5; 1-3-7; 1-3-9; 1-5-7; 1-5-9; 1-7-9; ..." (昇順ソート)

    Args:
        ranked: weighted_score 降順でソート済みのリスト

    Returns:
        formation string (例: "1, 3, 5, 7, 9" で代表される 7 点)
        空の場合は空文字列
    """
    if not ranked:
        return ''

    # top-5 を取得 (馬番取り出し、int 変換)
    top5 = ranked[:5]
    nums = []
    for h in top5:
        try:
            nums.append(int(h['num']))
        except (KeyError, TypeError, ValueError):
            pass

    if len(nums) < 3:
        # 3頭未満なら formation 組めない
        return ''

    top1 = nums[0]
    rest = sorted(nums[1:])  # top2〜5 を昇順

    # 1列目: top1
    # 2列目: top2, top3 = rest[0], rest[1]
    # 3列目: top2〜top5 = rest
    col2 = rest[:2]
    col3 = rest

    bets = []
    from itertools import combinations
    for b in col2:
        for c in col3:
            if c <= b:
                continue
            triple = sorted([top1, b, c])
            s = f"{triple[0]}-{triple[1]}-{triple[2]}"
            if s not in bets:
                bets.append(s)

    # 追加: top1 が最小でない場合を補完 (昇順 trio は top1 が必ずどこかに入る)
    # ただし formation 上限 7 点
    bets = bets[:7]

    return '; '.join(bets)


def get_paper_formation(predictions_list: list[dict]) -> str:
    """paper 用 formation string を返す (race_notify_log_v2 連携 API)。

    Args:
        predictions_list: 各馬の dict リスト。
            必須キー: num (馬番), score (V15 prob)
            任意キー: name, weight_change, is_top_jockey

    Returns:
        trio formation string (例: "1-3-5; 1-3-7; ...")
        predictions_list が空 or 馬数 < 3 の場合は ''
    """
    if not predictions_list:
        return ''
    ranked = rank_by_weighted_score(predictions_list)
    return build_trio_formation(ranked)


# ---------------------------------------------------------------------------
# Backtest on cumulative_results.csv
# ---------------------------------------------------------------------------

def backtest_on_csv(csv_path: str) -> dict:
    """cumulative_results.csv を読み込み、top-5 score weighting の backtest を実行。

    proxy 方法:
      - top1_score / top2_score / top3_score を score として使用
      - weight_change 列が存在しない場合は factor=1.0 (no-op)
      - top4_num があれば top-4 確保、top-5 は num_horses から推定 (簡易)
      - formation 変化が生じたレースで hit_rate / ROI を比較

    Returns:
        dict: {
            'n_races': int,
            'n_valid': int,           # score が存在するレース数
            'original_hit3': float,   # 元の top1-3 で trio_result に一致
            'weighted_hit3': float,   # weighted top-1-3 で trio_result に一致
            'original_roi': float,
            'weighted_roi': float,
            'formation_changed': int, # 重み付けで top-3 が変化したレース数
        }
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)

    # settled + trio bet のみ対象
    mask = (df['status'] == 'settled') & (df['bet_type'] == 'trio')
    df = df[mask].copy()

    n_valid = 0
    original_hits = 0
    weighted_hits = 0
    original_invest = 0
    weighted_invest = 0
    original_payout = 0.0
    weighted_payout = 0.0
    formation_changed = 0

    for _, row in df.iterrows():
        # score が揃っているか確認
        try:
            s1 = float(row['top1_score'])
            s2 = float(row['top2_score'])
            s3 = float(row['top3_score'])
        except (TypeError, ValueError, KeyError):
            continue

        # s2/s3 が 0 の場合は meaningful score なし (top2/3 score が未記録) → skip
        if pd.isna(s1) or pd.isna(s2) or pd.isna(s3):
            continue
        if s2 == 0.0 and s3 == 0.0:
            continue

        try:
            n1 = int(row['top1_num'])
            n2 = int(row['top2_num'])
            n3 = int(row['top3_num'])
        except (TypeError, ValueError, KeyError):
            continue

        n4 = None
        try:
            v4 = row.get('top4_num') if hasattr(row, 'get') else row['top4_num']
            if v4 is not None and not pd.isna(float(v4)):
                n4 = int(float(v4))
        except (TypeError, ValueError, KeyError):
            pass

        # weight_change は cumulative_results.csv にないので 1.0
        preds = [
            {'num': n1, 'score': s1, 'weight_change': None},
            {'num': n2, 'score': s2, 'weight_change': None},
            {'num': n3, 'score': s3, 'weight_change': None},
        ]
        if n4 is not None:
            preds.append({'num': n4, 'score': max(s1, s2, s3) * 0.5, 'weight_change': None})

        ranked = rank_by_weighted_score(preds)
        new_top3 = sorted([int(ranked[i]['num']) for i in range(min(3, len(ranked)))])
        orig_top3 = sorted([n1, n2, n3])

        if new_top3 != orig_top3:
            formation_changed += 1

        # 元 trio_result: "A-B-C" 形式で実際の 1-3 着番号
        try:
            trio_res_str = str(row.get('trio_result', '') or '')
            if not trio_res_str or trio_res_str in ('nan', 'None', ''):
                # top1/2/3_finish が小さい順
                fin = [(row['top1_finish'], n1), (row['top2_finish'], n2), (row['top3_finish'], n3)]
                fin = [(f, n) for f, n in fin if not pd.isna(float(f))]
                fin = sorted(fin, key=lambda x: float(x[0]))
                actual_top3 = sorted([int(n) for _, n in fin[:3]])
            else:
                actual_top3 = sorted([int(x) for x in trio_res_str.split('-')])
        except (TypeError, ValueError, AttributeError):
            continue

        invest = 700.0
        try:
            payout_val = float(row.get('trio_payout', 0) or 0)
        except (TypeError, ValueError):
            payout_val = 0.0

        n_valid += 1
        original_invest += invest
        weighted_invest += invest

        # Original hit: orig_top3 が actual_top3 と一致
        if orig_top3 == actual_top3:
            original_hits += 1
            original_payout += payout_val

        # Weighted hit: new_top3 が actual_top3 と一致
        if new_top3 == actual_top3:
            weighted_hits += 1
            # 重み変更後も同じ payout (formation 変化しても同じ trio combination に賭けると仮定)
            weighted_payout += payout_val

    result = {
        'n_races_total': n_total,
        'n_valid': n_valid,
        'original_hit3': original_hits / n_valid if n_valid > 0 else 0.0,
        'weighted_hit3': weighted_hits / n_valid if n_valid > 0 else 0.0,
        'original_roi': original_payout / original_invest * 100 if original_invest > 0 else 0.0,
        'weighted_roi': weighted_payout / weighted_invest * 100 if weighted_invest > 0 else 0.0,
        'original_hits': original_hits,
        'weighted_hits': weighted_hits,
        'formation_changed': formation_changed,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo, 'data', 'cumulative_results.csv')

    print('=== V15 top-5 score weighting -- backtest ===')
    print(f'CSV: {csv_path}')

    if not os.path.exists(csv_path):
        print('cumulative_results.csv not found. skipping backtest.')
        sys.exit(0)

    result = backtest_on_csv(csv_path)

    print(f"\n対象レース (settled trio): {result['n_valid']} / {result['n_races_total']}")
    print(f"formation 変化レース数  : {result['formation_changed']}")
    print()
    print(f"  {'':30s} {'hits':>6s}  {'hit_rate':>9s}  {'ROI':>8s}")
    print(f"  {'Original (V15 top1-3)':30s} {result['original_hits']:>6d}  {result['original_hit3']*100:>8.2f}%  {result['original_roi']:>7.2f}%")
    print(f"  {'Weighted (score × factors)':30s} {result['weighted_hits']:>6d}  {result['weighted_hit3']*100:>8.2f}%  {result['weighted_roi']:>7.2f}%")
    delta_hit = (result['weighted_hit3'] - result['original_hit3']) * 100
    delta_roi = result['weighted_roi'] - result['original_roi']
    print(f"\n  Delta hit_rate : {delta_hit:+.2f}pt")
    print(f"  Delta ROI      : {delta_roi:+.2f}pt")
    print()
    print('Note: weight_change 列が cumulative_results.csv にないため factor=1.0 (no-op)。')
    print('      top2_score / top3_score が 0 の行はスキップ (V15 production は top1_score のみ記録)。')
    print('      paper 蓄積後 (n>=50) に top2/3_score + weight_change 付き CSV で再 backtest 推奨。')

    # formation example
    print('\n--- Formation example (demo) ---')
    demo_preds = [
        {'num': 3, 'score': 0.72, 'name': 'HorseA', 'weight_change': 2.0},
        {'num': 7, 'score': 0.65, 'name': 'HorseB', 'weight_change': 12.0},  # -0.8 factor
        {'num': 1, 'score': 0.60, 'name': 'HorseC', 'weight_change': -4.0},
        {'num': 9, 'score': 0.55, 'name': 'HorseD', 'weight_change': 0.0},
        {'num': 5, 'score': 0.50, 'name': 'HorseE', 'weight_change': 8.0},  # 0.9 factor
    ]
    ranked = rank_by_weighted_score(demo_preds)
    print('Ranked (weighted):')
    for i, h in enumerate(ranked):
        print(f"  {i+1}. #{h['num']:>2d} {h.get('name',''):<10s}  base={h['score']:.3f}  ws={h['weighted_score']:.4f}")
    formation = get_paper_formation(demo_preds)
    print(f"\nFormation: {formation}")
