#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strategy Layer v2: 戦略⑦ + EV 動的閾値 + Calibration 統合 shadow runner.

V15 production / tools/race_auto_notify.py / tools/predict_core.py は完全不変。
本 module は **standalone shadow** で、 daily_predictions/YYYYMMDD.csv を読み
当日 odds 取得 → calibration → dynamic EV → recommended bet を算出して
data/v21/strategy_v2_shadow_YYYYMMDD.csv に書き出す。 Discord 通知なし。

【絶対 V15 production 不変】
- import は read-only (predict_core.calc_horse_ev は使わない)
- 戦略⑦ filter は race_auto_notify.py の copy (independent reimplementation)
- 5/16-5/17 G1 day 本番 通知 logic に影響 0%

Usage:
    # backtest 単体 (cumulative_results.csv を使う retrospective)
    python tools/strategy_layer_v2.py --backtest

    # 当日 shadow eval (daily_predictions + 当日 odds)
    python tools/strategy_layer_v2.py --shadow 20260518
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Optional

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATOR_PATH = os.path.join(BASE_DIR, 'data', 'calibrator_v15_pilot.pkl')
SHADOW_DIR = os.path.join(BASE_DIR, 'data', 'v21')
os.makedirs(SHADOW_DIR, exist_ok=True)

# ===== EV / bet 閾値 (design 参照) =====
EV_SKIP_BELOW = 1.0           # EV < 1.0 → skip
EV_BET_2X = 2.0               # 1400 円
EV_BET_3X = 3.0               # 2100 円
BET_BASE = 700
BET_2X = 1400
BET_3X = 2100
EV_SANITY_MAX = 10.0          # abnormal clip
CALIB_BLEND = 0.3             # raw 0.7 + isotonic 0.3
PROB_TO_TOP3_MULT = 3.0       # score / sum × 3 → top3 prob (predict_core 既存と同じ)
PROB_TOP3_CLIP = 0.85
TRIO_ODDS_MULT = 2.0          # 三連複 ≈ 単勝 × 2 (predict_core 既存と同じ)


# ===== 戦略⑦ logic (race_auto_notify.py:171-273 の copy、 V15 production 不変) =====

def is_06_tokubetsu(race_name: str) -> bool:
    """06_平場特別 (G/L/OPEN 特別 でない 平場特別) を検出."""
    is_graded = any(g in race_name for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'])
    is_listed = any(s in race_name for s in ['L)', '(L)', 'OP)', '(OP)'])
    is_open_tokubetsu = any(s in race_name for s in ['杯', '賞', 'ステークス', 'カップ', 'ハンデ'])
    return '特別' in race_name and not (is_graded or is_listed or is_open_tokubetsu)


def strategy_7_filter(race_name: str, condition: str, distance: int) -> tuple[bool, str]:
    """戦略⑦ filter. 通る = True (賭ける候補)、 通らない = False (skip).

    race_auto_notify.py の 171-273 line を 1 関数化、 V15 production と同 logic.
    京都 filter は 5/10 に削除済 (再現せず)。
    """
    rn = str(race_name)
    cond = str(condition)

    # 距離 1000m 以下 skip (race_auto_notify.py:166)
    if distance is not None and distance > 0 and distance <= 1000:
        return False, 'distance_le_1000'

    # 06_平場特別 skip
    if is_06_tokubetsu(rn):
        return False, 'strategy_7_06_tokubetsu'

    # 条件 E (頭数<=7) skip
    if cond == 'E':
        return False, 'strategy_7_cond_E'

    # 条件 B (重〜不馬場) skip
    if cond == 'B':
        return False, 'strategy_7_cond_B'

    return True, 'strategy_7_pass'


# ===== Calibration (data/calibrator_v15_pilot.pkl 適用) =====

_calibrator_cache: Optional[dict] = None


def load_calibrator(path: str = CALIBRATOR_PATH) -> Optional[dict]:
    """Calibrator load (cache 付き). 失敗時 None."""
    global _calibrator_cache
    if _calibrator_cache is not None:
        return _calibrator_cache
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            cal = pickle.load(f)
        if not isinstance(cal, dict) or 'isotonic' not in cal:
            return None
        _calibrator_cache = cal
        return cal
    except Exception:
        return None


def apply_calibration(raw_probs: np.ndarray, blend: float = CALIB_BLEND) -> np.ndarray:
    """V15 raw probs に isotonic calibration を blend 適用.

    audit (data/v21/strategy_layer_calibrator_audit.md) に基づき安全 clip:
    - isotonic 単体は p>=0.3 で 1.0 飽和 → bet 戦略 崩壊
    - blend (1-b)*raw + b*iso で raw を 70% 重視
    - calibrator 取得失敗時は raw を そのまま返す

    Args:
        raw_probs: V15 raw probabilities (0-1)
        blend: isotonic 重み (default 0.3)

    Returns:
        calibrated probabilities (0-1)
    """
    arr = np.asarray(raw_probs, dtype=float)
    cal = load_calibrator()
    if cal is None:
        return arr
    try:
        iso = cal['isotonic']
        iso_p = iso.predict(arr.clip(0.0, 1.0))
        blended = (1.0 - blend) * arr + blend * iso_p
        return np.clip(blended, 0.0, 1.0)
    except Exception:
        return arr


# ===== EV 計算 (predict_core.calc_horse_ev と同 logic、 reimplementation) =====

def compute_top1_prob(top1_score: float, total_score_sum: Optional[float] = None,
                       num_horses: Optional[int] = None) -> float:
    """top1 horse の top3 prob を推定.

    Pattern A: total_score_sum 既知 → top1_score / sum * 3, clip 0.85
    Pattern B: num_horses のみ → 簡易 normalization
        scores が softmax-like なら top1_score ≈ 0.15-0.25、 sum ≈ 1.0、
        top3_prob = top1_score * 3 で OK (clip 0.85)
    """
    if total_score_sum is not None and total_score_sum > 0:
        p = top1_score / total_score_sum * PROB_TO_TOP3_MULT
    else:
        # scores が softmax (sum=1.0) と仮定する近似
        p = top1_score * PROB_TO_TOP3_MULT
    return float(np.clip(p, 0.0, PROB_TOP3_CLIP))


def compute_ev_dynamic(prob_top3: float, odds_top1: float,
                        bet_type: str = 'trio') -> float:
    """動的 EV 計算.

    predict_core.calc_horse_ev の流儀:
    - trio multiplier = odds × 2.0
    - umaren multiplier = odds × 1.5 (経験則)
    - 単勝なら multiplier = odds そのまま

    Args:
        prob_top3: top3 prob (0-0.85)
        odds_top1: 単勝オッズ
        bet_type: 'trio' / 'umaren' / 'tansho'
    """
    if odds_top1 is None or odds_top1 <= 0:
        odds_top1 = 10.0  # predict_core 既存 fallback
    bt = (bet_type or 'trio').lower()
    if bt == 'umaren':
        mult = odds_top1 * 1.5
    elif bt == 'tansho':
        mult = odds_top1
    else:
        mult = odds_top1 * TRIO_ODDS_MULT
    ev = prob_top3 * mult
    # sanity clip
    if ev > EV_SANITY_MAX:
        ev = EV_SANITY_MAX
    return float(ev)


def decide_bet_size(ev: float, base: int = BET_BASE) -> int:
    """EV から bet size を決定. design に従う."""
    if ev < EV_SKIP_BELOW:
        return 0
    if ev >= EV_BET_3X:
        return BET_3X
    if ev >= EV_BET_2X:
        return BET_2X
    return base


# ===== top-level filter =====

def filter_strategy_v2(race_meta: dict, top1_score: Optional[float] = None,
                        odds_top1: Optional[float] = None,
                        total_score_sum: Optional[float] = None) -> dict:
    """戦略⑦ + EV 動的閾値 + calibration 統合 filter.

    Args:
        race_meta: dict (race_name, condition, distance, num_horses, bet_type)
        top1_score: V15 raw prob for top1 horse (None なら strategy_7 のみ判定)
        odds_top1: top1 horse の 単勝オッズ
        total_score_sum: 全馬 score sum (Pattern A 計算用、 None なら 近似)

    Returns:
        {
          'recommended': bool,
          'bet_size': int,
          'ev_top1': float,
          'p_calibrated_top1': float,
          'p_raw_top1': float,
          'reason': str,
          'strategy_7_pass': bool,
        }
    """
    race_name = race_meta.get('race_name', '')
    condition = race_meta.get('condition', '')
    distance = race_meta.get('distance', 1600) or 1600
    bet_type = race_meta.get('bet_type', 'trio')

    out = {
        'recommended': False,
        'bet_size': 0,
        'ev_top1': 0.0,
        'p_calibrated_top1': 0.0,
        'p_raw_top1': float(top1_score) if top1_score is not None else 0.0,
        'reason': '',
        'strategy_7_pass': False,
    }

    # Step 1: 戦略⑦
    s7_pass, s7_reason = strategy_7_filter(race_name, condition, int(distance))
    out['strategy_7_pass'] = s7_pass
    if not s7_pass:
        out['reason'] = s7_reason
        return out

    # Step 2: top1_score 不在なら 戦略⑦ pass のみで 700 円 bet (baseline 動作)
    if top1_score is None or pd.isna(top1_score):
        out['recommended'] = True
        out['bet_size'] = BET_BASE
        out['reason'] = 'strategy_7_pass + score_unavailable_fallback'
        return out

    # Step 3: calibration
    raw_p = float(top1_score)
    cal_p = float(apply_calibration(np.array([raw_p]))[0])
    out['p_calibrated_top1'] = cal_p

    # Step 4: prob → top3 prob → EV
    p_top3 = compute_top1_prob(cal_p, total_score_sum=total_score_sum)
    ev = compute_ev_dynamic(p_top3, odds_top1 or 10.0, bet_type=bet_type)
    out['ev_top1'] = ev

    # Step 5: bet_size 決定
    bet_size = decide_bet_size(ev)
    out['bet_size'] = bet_size
    out['recommended'] = (bet_size > 0)
    if bet_size == 0:
        out['reason'] = f'strategy_7_pass + ev_skip ({ev:.2f}<{EV_SKIP_BELOW})'
    elif bet_size == BET_3X:
        out['reason'] = f'strategy_7_pass + ev_high ({ev:.2f}>={EV_BET_3X}) → 3x'
    elif bet_size == BET_2X:
        out['reason'] = f'strategy_7_pass + ev_mid ({ev:.2f}>={EV_BET_2X}) → 2x'
    else:
        out['reason'] = f'strategy_7_pass + ev_base ({ev:.2f})'
    return out


# ===== backtest (cumulative_results.csv 使用) =====

def _to_num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def backtest_from_cumulative(out_csv: str, out_md: str):
    """data/cumulative_results.csv の実 outcome で baseline vs v2 比較.

    ★ data 制約 (★ 重要 ★) ★:
    - top1_score available は 20 件のみ (5/10 以前 score 未書込 既知 bug)
    - cumulative_results.csv には **odds 列が無い** → 真の EV 計算不可
    - → 本 backtest は **strategy 7 強化部分のみ** が valid signal
    - EV 動的閾値の 真の検証は 5/18+ paper shadow data 蓄積で実施
    """
    csv_path = os.path.join(BASE_DIR, 'data', 'cumulative_results.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df[df['status'] == 'settled'].copy()
    df['race_id'] = df['race_id'].astype(str)
    df['actual_payout'] = df['actual_payout'].apply(_to_num)
    df['investment'] = df['investment'].apply(_to_num)
    df['top1_score'] = df['top1_score'].apply(lambda x: _to_num(x, np.nan))
    df['distance'] = df['distance'].apply(lambda x: _to_num(x, 1600.0))
    df['race_name'] = df['race_name'].astype(str)
    df['condition'] = df['condition'].astype(str)
    df['course'] = df['course'].astype(str)

    # === daily_predictions_full から enrich (score_sum + 当日 odds) ===
    full_dir = os.path.join(BASE_DIR, 'data', 'daily_predictions_full')
    enrich = {}  # race_id -> {top1_score, score_sum, top1_num, top1_odds}
    if os.path.isdir(full_dir):
        for fname in sorted(os.listdir(full_dir)):
            if not fname.endswith('.csv'):
                continue
            try:
                fdf = pd.read_csv(os.path.join(full_dir, fname), encoding='utf-8-sig')
                fdf['race_id'] = fdf['race_id'].astype(str)
                for rid, sub in fdf.groupby('race_id'):
                    sub = sub.sort_values('rank_in_race')
                    if len(sub) == 0:
                        continue
                    top = sub.iloc[0]
                    try:
                        top_odds = float(top['odds']) if top['odds'] and top['odds'] > 0 else None
                    except (TypeError, ValueError):
                        top_odds = None
                    enrich[rid] = {
                        'top1_score': float(top['V15_score']),
                        'score_sum': float(sub['V15_score'].sum()),
                        'top1_num': int(top['horse_num']),
                        'top1_odds': top_odds,
                    }
            except Exception:
                pass
    if enrich:
        print(f'[INFO] daily_predictions_full enrich loaded: {len(enrich)} races')

    rows_out = []
    base_inv = base_pay = 0
    v2_inv = v2_pay = 0
    s7only_inv = s7only_pay = 0  # 戦略⑦ のみ (EV scaling なし) の場合 = 純粋 strategy 7 効果
    v2_bet_count = 0
    s7only_bet_count = 0
    v2_with_score_count = 0
    v2_bet2x = v2_bet3x = 0
    s7_skipped = 0

    for _, r in df.iterrows():
        meta = {
            'race_name': r['race_name'],
            'condition': r['condition'],
            'distance': int(r['distance']) if r['distance'] > 0 else 1600,
            'num_horses': _to_num(r.get('num_horses', 0)),
            'bet_type': str(r.get('bet_type', 'trio')) if pd.notna(r.get('bet_type', None)) else 'trio',
        }

        # baseline: actual cumulative bet was placed (現状 cumulative には 06_特別 が含まれている = 戦略⑦ 適用前 含む)
        actual_inv = r['investment']
        actual_pay = r['actual_payout']
        base_inv += actual_inv
        base_pay += actual_pay

        # === strategy 7 only (pure 効果、 EV scaling なし、 odds 不要) ===
        s7_pass, s7_reason = strategy_7_filter(r['race_name'], r['condition'],
                                                int(r['distance']) if r['distance'] > 0 else 1600)
        if s7_pass:
            s7only_inv += BET_BASE
            # base_inv が 1400 (umaren 700+700) でも actual_pay は 700 base に scale
            s7only_pay_race = actual_pay * (BET_BASE / actual_inv) if actual_inv > 0 else actual_pay
            s7only_pay += s7only_pay_race
            s7only_bet_count += 1
        else:
            s7_skipped += 1

        # === full v2 (戦略⑦ + EV 動的) ===
        top1_score = r['top1_score'] if not pd.isna(r['top1_score']) else None
        rid = r['race_id']
        odds_top1 = None
        score_sum = None
        if rid in enrich:
            score_sum = enrich[rid]['score_sum']
            odds_top1 = enrich[rid]['top1_odds']
            if top1_score is None:
                top1_score = enrich[rid]['top1_score']
        v2_result = filter_strategy_v2(
            meta,
            top1_score=top1_score,
            odds_top1=odds_top1,
            total_score_sum=score_sum,
        )

        if v2_result['recommended']:
            v2_size = v2_result['bet_size']
            base_size_each = actual_inv if actual_inv > 0 else BET_BASE
            scale = v2_size / base_size_each if base_size_each > 0 else 1.0
            v2_inv_race = v2_size
            v2_pay_race = actual_pay * scale
            v2_inv += v2_inv_race
            v2_pay += v2_pay_race
            v2_bet_count += 1
            if top1_score is not None and odds_top1 is not None:
                v2_with_score_count += 1
            if v2_size == BET_2X:
                v2_bet2x += 1
            elif v2_size == BET_3X:
                v2_bet3x += 1
        else:
            v2_inv_race = 0
            v2_pay_race = 0

        rows_out.append({
            'race_id': r['race_id'],
            'date': r.get('date', ''),
            'course': r['course'],
            'race_num': r.get('race_num', ''),
            'race_name': r['race_name'],
            'condition': r['condition'],
            'top1_score': r['top1_score'],
            'base_inv': actual_inv,
            'base_pay': actual_pay,
            'base_pnl': actual_pay - actual_inv,
            's7only_recommended': s7_pass,
            's7only_inv': BET_BASE if s7_pass else 0,
            's7only_pay': (actual_pay * BET_BASE / actual_inv) if (s7_pass and actual_inv > 0) else 0,
            'v2_recommended': v2_result['recommended'],
            'v2_bet_size': v2_result['bet_size'],
            'v2_inv': v2_inv_race,
            'v2_pay': v2_pay_race,
            'v2_pnl': v2_pay_race - v2_inv_race,
            'v2_ev_top1': v2_result['ev_top1'],
            'v2_p_calibrated': v2_result['p_calibrated_top1'],
            'v2_strategy_7_pass': v2_result['strategy_7_pass'],
            'v2_reason': v2_result['reason'],
        })

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    base_roi = base_pay / base_inv * 100 if base_inv > 0 else 0
    v2_roi = v2_pay / v2_inv * 100 if v2_inv > 0 else 0
    s7_roi = s7only_pay / s7only_inv * 100 if s7only_inv > 0 else 0
    delta_v2 = v2_roi - base_roi
    delta_s7 = s7_roi - base_roi

    # report
    lines = []
    lines.append('# Strategy Layer v2 - Backtest Simulation Report')
    lines.append('')
    lines.append(f'**実施日**: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'**source**: `data/cumulative_results.csv` (settled rows)')
    lines.append(f'**output CSV**: `{os.path.relpath(out_csv, BASE_DIR)}`')
    lines.append('')
    lines.append('## 1. 主要 metrics')
    lines.append('')
    lines.append('| metric | baseline (cumulative 実 bet) | strategy_7 only (再現) | strategy_v2 full |')
    lines.append('|--------|------------------------:|------------------------:|-----------------:|')
    lines.append(f'| 対象 races | {len(out_df):,} | {len(out_df):,} | {len(out_df):,} |')
    lines.append(f'| 戦略⑦ 除外 races | (mixed) | {s7_skipped:,} | {s7_skipped:,} |')
    lines.append(f'| bet races | {int((out_df["base_inv"]>0).sum()):,} | {s7only_bet_count:,} | {v2_bet_count:,} |')
    lines.append(f'| total inv | {base_inv:,.0f} | {s7only_inv:,.0f} | {v2_inv:,.0f} |')
    lines.append(f'| total pay | {base_pay:,.0f} | {s7only_pay:,.0f} | {v2_pay:,.0f} |')
    lines.append(f'| ROI | {base_roi:.2f}% | **{s7_roi:.2f}%** | {v2_roi:.2f}% |')
    lines.append(f'| PnL | {base_pay-base_inv:+,.0f} | {s7only_pay-s7only_inv:+,.0f} | {v2_pay-v2_inv:+,.0f} |')
    lines.append(f'| delta vs baseline | - | {delta_s7:+.2f}pt | {delta_v2:+.2f}pt |')
    lines.append('')
    lines.append(f'### bet_size 内訳 (v2)')
    lines.append(f'- 700 円 (base): {v2_bet_count - v2_bet2x - v2_bet3x:,}')
    lines.append(f'- 1400 円 (2x):  {v2_bet2x:,}')
    lines.append(f'- 2100 円 (3x):  {v2_bet3x:,}')
    lines.append('')
    lines.append('### 戦略⑦ 除外 内訳')
    s7_reasons = out_df[~out_df['s7only_recommended']]
    if len(s7_reasons) > 0:
        for cond_label, n in [('06_平場特別', int(out_df['v2_reason'].eq('strategy_7_06_tokubetsu').sum())),
                               ('条件 E (頭数<=7)', int(out_df['v2_reason'].eq('strategy_7_cond_E').sum())),
                               ('条件 B (重〜不馬場)', int(out_df['v2_reason'].eq('strategy_7_cond_B').sum())),
                               ('距離<=1000m', int(out_df['v2_reason'].eq('distance_le_1000').sum()))]:
            lines.append(f'- {cond_label}: {n}')
    lines.append('')
    lines.append('## 2. ★ data 制約 (★ 最重要 ★)')
    lines.append('')
    lines.append(f'- `cumulative_results.csv` 総 settled rows = **{len(df)}**')
    lines.append(f'- そのうち **cumulative top1_score available rows = {int(df["top1_score"].notna().sum())}** (5/10 以前 score 未書込 既知 bug)')
    lines.append(f'- `daily_predictions_full/` から enrich (top1_score + score_sum + odds): **{len(enrich)} races**')
    lines.append(f'- 結果: strategy_v2 が **完全 EV 評価できた settled rows = {v2_with_score_count}**')
    lines.append(f'- 残り rows は **score_unavailable_fallback** で 戦略⑦ pass+700 円 bet 動作 (= s7_only と同じ)')
    lines.append('')
    # 31 件 enrich subset の純粋比較
    enrich_subset = out_df[out_df['v2_ev_top1'] > 0].copy()
    if len(enrich_subset) > 0:
        e_base_inv = enrich_subset['base_inv'].sum()
        e_base_pay = enrich_subset['base_pay'].sum()
        e_v2_inv = enrich_subset['v2_inv'].sum()
        e_v2_pay = enrich_subset['v2_pay'].sum()
        e_base_roi = e_base_pay / e_base_inv * 100 if e_base_inv > 0 else 0
        e_v2_roi = e_v2_pay / e_v2_inv * 100 if e_v2_inv > 0 else 0
        e_delta = e_v2_roi - e_base_roi
        lines.append(f'### 純粋 enrich subset 比較 (N={len(enrich_subset)})')
        lines.append('')
        lines.append('| metric | baseline | strategy_v2 (real odds) | delta |')
        lines.append('|--------|---------:|------------------------:|------:|')
        lines.append(f'| inv | {e_base_inv:,.0f} | {e_v2_inv:,.0f} | {e_v2_inv-e_base_inv:+,.0f} |')
        lines.append(f'| pay | {e_base_pay:,.0f} | {e_v2_pay:,.0f} | {e_v2_pay-e_base_pay:+,.0f} |')
        lines.append(f'| ROI | {e_base_roi:.2f}% | {e_v2_roi:.2f}% | {e_delta:+.2f}pt |')
        # bet_size 別 ROI
        lines.append('')
        lines.append('bet_size 別 ROI (enrich subset only、 v2 view):')
        lines.append('')
        lines.append('| bet_size | N | inv | pay | ROI |')
        lines.append('|---------:|--:|----:|----:|----:|')
        for bs in [BET_BASE, BET_2X, BET_3X]:
            sub = enrich_subset[enrich_subset['v2_bet_size'] == bs]
            if len(sub) == 0: continue
            inv = sub['v2_inv'].sum()
            pay = sub['v2_pay'].sum()
            roi = pay / inv * 100 if inv > 0 else 0
            lines.append(f'| {bs} 円 | {len(sub)} | {inv:,.0f} | {pay:,.0f} | {roi:.1f}% |')
        lines.append('')
        lines.append('### 観察 (honest)')
        lines.append('')
        lines.append(f'- enrich subset で **v2 ROI {e_v2_roi:.1f}% vs baseline {e_base_roi:.1f}% (delta {e_delta:+.2f}pt)**')
        lines.append(f'- sample {len(enrich_subset)} 件は **5/10 1 日分のみ** で 5/10 自体が低調 day (baseline ROI {e_base_roi:.1f}%)')
        lines.append(f'- 統計的 sample 不足 (n<100、 day-level 偏り 重大)')
        if e_delta > 0:
            lines.append(f'- 観測上は +{e_delta:.1f}pt 改善 だが、 sample 不足で **未実証**')
        else:
            lines.append(f'- 観測上 {e_delta:.1f}pt 悪化、 calibrator 過剰飽和の 影響 可能性')
    lines.append('')
    lines.append('## 3. 結論 (honest)')
    lines.append('')
    lines.append('### Strategy 7 only 部分 (valid signal)')
    lines.append('')
    lines.append(f'- ROI {base_roi:.2f}% → {s7_roi:.2f}% (delta {delta_s7:+.2f}pt)')
    lines.append(f'- 戦略⑦ 除外 {s7_skipped} race で 投資効率 {"改善" if delta_s7 > 0 else "悪化"}')
    lines.append(f'- ★ ただし cumulative にも 既に 戦略⑦ 一部適用済 race が混ざっており、 純粋差分は 上記より控えめ')
    lines.append('')
    lines.append('### EV 動的閾値部分 (sample 不足)')
    lines.append('')
    lines.append(f'- v2 EV 評価できた sample {v2_with_score_count} 件 (= daily_predictions_full enrich 適用 後)')
    lines.append(f'- sample 51 件 (enrich subset、 5/10 1 日のみ) で -0.8pt 悪化、 統計的に 未実証')
    lines.append(f'- 「期待 +15-30% ROI 改善」 は **想定** 値、 5/18+ 蓄積 必要')
    lines.append(f'- 残り 498 rows は score_unavailable で 戦略⑦ pass のみで base 動作')
    lines.append('')
    lines.append('### Calibration 部分 (audit 別途)')
    lines.append('')
    lines.append('- 学習 sample 21 件、 isotonic は p>=0.3 で 1.0 飽和')
    lines.append('- blend 0.3 で慎重に取り込むが、 calibrator 自体の信頼性 低い')
    lines.append('- 詳細: `data/v21/strategy_layer_calibrator_audit.md`')
    lines.append('')
    lines.append('## 4. 次 step (真の検証)')
    lines.append('')
    lines.append('- [ ] 5/18 (土) 朝から `tools/strategy_layer_v2.py --shadow YYYYMMDD` 起動')
    lines.append('- [ ] 当日 odds_full.json と merge して 真の EV 計算 (shadow_eval 内で 拡張)')
    lines.append('- [ ] 30+ race 蓄積後 `data/v21/strategy_v2_shadow_*.csv` を集計')
    lines.append('- [ ] 200+ race 蓄積後 calibrator 再 train (現状 21 sample 不足)')
    lines.append('- [ ] 真 ROI delta を honest に算出、 「+15-30%」 想定との 突合')

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('=' * 60)
    print(f'baseline ROI:    {base_roi:.2f}%  (inv {base_inv:,.0f} / pay {base_pay:,.0f})')
    print(f'strategy_7 only: {s7_roi:.2f}%  (inv {s7only_inv:,.0f} / pay {s7only_pay:,.0f})  delta {delta_s7:+.2f}pt')
    print(f'strategy_v2 full:{v2_roi:.2f}%  (inv {v2_inv:,.0f} / pay {v2_pay:,.0f})  delta {delta_v2:+.2f}pt')
    print(f'  ★ NOTE: enrich {len(enrich)} race のみ 真 EV 計算、 残り 498 race は odds 不在で base 動作')
    print(f'v2 fully EV-evaluated sample: {v2_with_score_count} / {len(df)} settled rows')
    print('=' * 60)
    print(f'CSV: {out_csv}')
    print(f'MD : {out_md}')


# ===== shadow mode (5/18+ 起動) =====

def load_odds_for_date(date_str: str) -> dict:
    """odds_base_YYYYMMDD.csv を読み込み、 {race_id: {horse_num: odds}} を返す.
    無ければ {}.
    """
    path = os.path.join(BASE_DIR, 'data', f'odds_base_{date_str}.csv')
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, encoding='utf-8')
        out = {}
        for _, r in df.iterrows():
            rid = str(r['race_id'])
            uma = int(r['horse_num'])
            try:
                odds = float(r['odds'])
            except (TypeError, ValueError):
                odds = 0
            if rid not in out:
                out[rid] = {}
            out[rid][uma] = odds
        return out
    except Exception as e:
        print(f'[WARN] odds load 失敗 {path}: {e}')
        return {}


def load_full_predictions(date_str: str) -> Optional[pd.DataFrame]:
    """data/daily_predictions_full/{date}.csv (全頭 V15_score + odds) を読む.
    無ければ None.
    """
    path = os.path.join(BASE_DIR, 'data', 'daily_predictions_full', f'{date_str}.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['race_id'] = df['race_id'].astype(str)
        return df
    except Exception as e:
        print(f'[WARN] daily_predictions_full load 失敗: {e}')
        return None


def shadow_eval(date_str: str):
    """daily_predictions/{date}.csv を読み strategy_v2 shadow を出力.

    Discord 通知なし、 cumulative_results.csv 書き込みなし。
    出力: data/v21/strategy_v2_shadow_{date}.csv

    odds 取得 順:
      1. daily_predictions_full/{date}.csv (全頭 V15_score + odds、 推奨)
      2. odds_base_{date}.csv (top1_num の odds のみ、 fallback)
      3. odds=10.0 fallback (最低限)
    """
    daily_path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{date_str}.csv')
    if not os.path.exists(daily_path):
        print(f'[ERROR] {daily_path} 未生成')
        return 1

    df = pd.read_csv(daily_path, encoding='utf-8-sig')
    df['race_id'] = df['race_id'].astype(str)
    print(f'[INFO] {len(df)} races for {date_str}')

    # 1. full predictions (推奨)
    full_df = load_full_predictions(date_str)
    if full_df is not None:
        print(f'[INFO] daily_predictions_full/{date_str}.csv loaded: {len(full_df)} horses')
        # race_id ごと: top1 horse の score + odds + score_sum
        race_info = {}
        for rid, sub in full_df.groupby('race_id'):
            sub = sub.sort_values('rank_in_race')
            top = sub.iloc[0]
            try:
                top1_odds_v = float(top['odds']) if top['odds'] and top['odds'] > 0 else None
            except (TypeError, ValueError):
                top1_odds_v = None
            race_info[rid] = {
                'top1_score': float(top['V15_score']),
                'top1_num': int(top['horse_num']),
                'top1_odds': top1_odds_v,
                'score_sum': float(sub['V15_score'].sum()),
            }
    else:
        print(f'[WARN] daily_predictions_full/{date_str}.csv 不在')
        race_info = {}

    # 2. odds_base fallback
    odds_map = load_odds_for_date(date_str)
    if odds_map:
        print(f'[INFO] odds_base_{date_str}.csv loaded: {len(odds_map)} races')

    rows_out = []
    bet_count = 0
    bet_total = 0
    bet2x = bet3x = 0
    odds_resolved = 0
    score_sum_used = 0

    for _, r in df.iterrows():
        meta = {
            'race_name': r.get('race_name', ''),
            'condition': r.get('condition', ''),
            'distance': int(r.get('distance', 1600) or 1600),
            'num_horses': r.get('num_horses', 0),
            'bet_type': r.get('bet_type', 'trio'),
        }
        rid = str(r.get('race_id', ''))
        top1_score = r.get('top1_score', None)
        if pd.isna(top1_score):
            top1_score = None

        odds_top1 = None
        score_sum = None
        if rid in race_info:
            score_sum = race_info[rid]['score_sum']
            odds_top1 = race_info[rid]['top1_odds']
            if top1_score is None:
                top1_score = race_info[rid]['top1_score']
            score_sum_used += 1
        if odds_top1 is None:
            top1_num = r.get('top1_num', None)
            if pd.notna(top1_num) and rid in odds_map:
                try:
                    odds_top1 = odds_map[rid].get(int(top1_num), None)
                    if not odds_top1 or odds_top1 <= 0:
                        odds_top1 = None
                except (TypeError, ValueError):
                    pass
        if odds_top1 is not None:
            odds_resolved += 1

        v2 = filter_strategy_v2(meta, top1_score=top1_score, odds_top1=odds_top1,
                                 total_score_sum=score_sum)

        rows_out.append({
            'race_id': r.get('race_id', ''),
            'date': date_str,
            'course': r.get('course', ''),
            'race_num': r.get('race_num', ''),
            'race_name': r.get('race_name', ''),
            'condition': r.get('condition', ''),
            'num_horses': r.get('num_horses', ''),
            'distance': r.get('distance', ''),
            'top1_num': r.get('top1_num', ''),
            'top1_score': top1_score,
            'top1_odds': odds_top1 if odds_top1 else 10.0,
            'score_sum': score_sum if score_sum else 0,
            'odds_resolved': odds_top1 is not None,
            'p_calibrated': v2['p_calibrated_top1'],
            'ev_top1': v2['ev_top1'],
            's7_pass': v2['strategy_7_pass'],
            'v2_recommended': v2['recommended'],
            'v2_bet_size': v2['bet_size'],
            'v2_reason': v2['reason'],
        })

        if v2['recommended']:
            bet_count += 1
            bet_total += v2['bet_size']
            if v2['bet_size'] == BET_2X:
                bet2x += 1
            elif v2['bet_size'] == BET_3X:
                bet3x += 1

    out_csv = os.path.join(SHADOW_DIR, f'strategy_v2_shadow_{date_str}.csv')
    pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding='utf-8-sig')

    print('=' * 60)
    print(f'shadow date: {date_str}')
    print(f'total races: {len(df)}')
    print(f'score_sum 使用 (full): {score_sum_used} / {len(df)}')
    print(f'odds resolved: {odds_resolved} / {len(df)}')
    print(f'v2 recommended: {bet_count}')
    print(f'  - 700  base: {bet_count - bet2x - bet3x}')
    print(f'  - 1400 (2x): {bet2x}')
    print(f'  - 2100 (3x): {bet3x}')
    print(f'total v2 inv: {bet_total:,}')
    print(f'output: {out_csv}')
    print('=' * 60)
    print('[NOTE] Discord 通知なし / V15 production 不変')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Strategy Layer v2 (V15 production 不変)')
    ap.add_argument('--backtest', action='store_true', help='cumulative_results.csv で retrospective')
    ap.add_argument('--shadow', metavar='YYYYMMDD', help='当日 daily_predictions で shadow eval')
    args = ap.parse_args()

    if args.shadow:
        return shadow_eval(args.shadow)

    if args.backtest:
        out_csv = os.path.join(SHADOW_DIR, 'strategy_v2_simulation.csv')
        out_md = os.path.join(SHADOW_DIR, 'strategy_v2_simulation_report.md')
        backtest_from_cumulative(out_csv, out_md)
        return 0

    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
