"""race_notify_log v2 (3 phase) — 真の formation record

★ 既存 race_auto_notify / daily_predict logic 完全不変、 log 出力 file IO のみ ★

3 phase:
- phase 1: daily_predict 後 (朝 8:00) → ranking + 朝 odds + 予定 formation
- phase 2: 投票直前 (race -5min / race_auto_notify) → 投票 formation 確定 + 投票時 odds
           + 8 strategy paper formation を同時記録
- phase 3: 結果回収後 (20:00 daily_results) → 実 1-3 着 + 実配当 + hit/miss
           + 8 strategy paper hit/miss/pnl を同時記録

出力 path:
  data/race_notify_log_v2/{YYYYMMDD}/phase{1,2,3}/{race_id}.json

8 strategy paper tracking (strategy_formations / strategy_results):
  actual       実際の bet (V15 + 戦略⑦)
  c3           + C3 (pos2 除外: formation の 2 列目除外)
  c4           + C4 (条件 A 1600-1800m 除外)
  c3c4         + C3+C4 複合
  no_1pop      B-1: top1 予測馬が 1 番人気の場合は skip
  divergence   B-2: top1 予測馬が 3 番人気以内の場合は skip (発散 = 3 番人気外)
  ev_filter    C-1: 将来用 (現在は actual と同一、 EV 計算は aggregator で別途)
  odds_filter  C-2: top1 単勝 odds 1.5-20 倍 帯フィルタ

log fail 時は stderr 出力のみで exception を投げない (既存 logic 影響なし)。

Aggregator: tools/race_notify_log_v2_aggregator.py (DAILY 20:30)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / 'data' / 'race_notify_log_v2'

# 8 strategy keys
STRATEGY_KEYS = [
    'actual',
    'c3',
    'c4',
    'c3c4',
    'no_1pop',
    'divergence',
    'ev_filter',
    'odds_filter',
]


def _log_root() -> Path:
    """log 出力 root を返す (test 用に env で override 可能)。"""
    env_root = os.environ.get('RACE_NOTIFY_LOG_V2_ROOT')
    if env_root:
        return Path(env_root)
    return LOG_DIR


def _today_str() -> str:
    return datetime.now().strftime('%Y%m%d')


def _write_json(out_file: Path, data: dict) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _validate_race_id(race_id) -> bool:
    """race_id validation: digit 10-12 桁 必須。

    無効な race_id (None / 'None' / 'nan' / '' / 非 digit / 桁数不正) を
    検出して log skip するための gate。
    """
    if race_id is None:
        return False
    try:
        s = str(race_id).strip()
    except Exception:
        return False
    if not s:
        return False
    if s in ('None', 'none', 'NONE', 'nan', 'NaN', 'NAN', 'null', 'NULL'):
        return False
    if not s.isdigit():
        return False
    if not (10 <= len(s) <= 12):
        return False
    return True


def _normalize_formation(formation) -> list | str:
    """formation (list of tuple/list or string) を JSON serialise 可能形に正規化。"""
    if formation is None:
        return None
    if isinstance(formation, (list, tuple)):
        try:
            return [
                list(b) if isinstance(b, (list, tuple)) else b
                for b in formation
            ]
        except Exception:
            return str(formation)
    return formation


def build_strategy_formations(
    predictions: list,
    race_meta: dict,
) -> dict:
    """8 strategy の paper formation を計算して返す。

    Args:
        predictions: top N 予測馬 list。各要素は dict で以下のキーを期待:
            horse_num  (int)
            pop_rank   (int, optional) — 人気順位
            odds       (float, optional) — 単勝オッズ
        race_meta: race 情報 dict。以下のキーを参照:
            cond_key   (str) — 条件 A/B/C/D/E/X
            distance   (int) — レース距離
            venue      (str) — 開催場 (例: '京都')

    Returns:
        dict: strategy key → formation (list of sorted list) or None (skip)

    fail 時は空 dict を返す (例外を上位に伝播しない)。
    """
    try:
        if not predictions or len(predictions) < 3:
            return {k: None for k in STRATEGY_KEYS}

        # --- race meta ---
        cond_key = str(race_meta.get('cond_key') or race_meta.get('condition') or '')
        distance = int(race_meta.get('distance') or 0)
        venue = str(race_meta.get('venue') or race_meta.get('race_meta', {}).get('venue', ''))

        # top 6 馬番
        top6 = [h.get('horse_num') or h.get('umaban') for h in predictions[:6]]
        # None を除去
        top6 = [n for n in top6 if n is not None]
        if len(top6) < 3:
            return {k: None for k in STRATEGY_KEYS}

        # top1 の人気・オッズ
        top1_pop = int(predictions[0].get('pop_rank') or predictions[0].get('ninki') or 99)
        top1_odds = float(predictions[0].get('odds') or 99.0)

        # 戦略⑦ベース skip: B/E/X / 京都 は actual も skip
        base_skip = cond_key in ('B', 'E', 'X') or '京都' in venue

        def make_trio_bets(nums, exclude_c3=False):
            """top6 から trio 7 bet formation を生成。
            exclude_c3=True の場合 C3 (2 列目 = pos2 = top3) を除外した 6 bet。
            """
            if len(nums) < 6:
                # 頭数不足: 可能な範囲で生成
                if len(nums) < 3:
                    return None
                n = nums
                t1, rest = n[0], n[1:]
                bets = []
                for i in range(len(rest)):
                    for j in range(i + 1, len(rest)):
                        bets.append(sorted([t1, rest[i], rest[j]]))
                if exclude_c3 and bets:
                    # 2 列目除外 = (t1, pos2, pos4) のような bet を除く
                    # ここでは pos2 = rest[0], pos4 = rest[2] の bet を除外
                    if len(rest) >= 3:
                        to_remove = sorted([t1, rest[0], rest[2]])
                        bets = [b for b in bets if b != to_remove]
                return bets if bets else None

            t1, t2, t3, t4, t5, t6 = nums[:6]
            bets = [
                sorted([t1, t2, t3]),
                sorted([t1, t2, t4]),
                sorted([t1, t2, t5]),
                sorted([t1, t2, t6]),
                sorted([t1, t3, t4]),
                sorted([t1, t3, t5]),
                sorted([t1, t3, t6]),
            ]
            if exclude_c3:
                # C3 = (t1, t2, t4) を除外 (2 列目 t4 = pos4 bet)
                c3_bet = sorted([t1, t2, t4])
                bets = [b for b in bets if b != c3_bet]
            return bets

        # --- skip flags ---
        skip_c4 = (cond_key == 'A' and 1600 <= distance <= 1800)
        skip_no1pop = (top1_pop == 1)
        skip_divergence = (top1_pop < 3)  # 3 番人気以内 = 発散しない → skip
        skip_odds = not (1.5 <= top1_odds <= 20.0)

        result = {}

        # actual
        result['actual'] = None if base_skip else make_trio_bets(top6)

        # c3 (+ C3: pos4 bet 除外)
        result['c3'] = None if base_skip else make_trio_bets(top6, exclude_c3=True)

        # c4 (+ C4: 条件 A 1600-1800m skip)
        result['c4'] = None if (base_skip or skip_c4) else make_trio_bets(top6)

        # c3c4 (+ C3+C4 複合)
        result['c3c4'] = None if (base_skip or skip_c4) else make_trio_bets(top6, exclude_c3=True)

        # no_1pop (B-1: top1 予測馬が 1 番人気ならば skip)
        result['no_1pop'] = None if (base_skip or skip_no1pop) else make_trio_bets(top6)

        # divergence (B-2: top1 予測馬が 3 番人気以内ならば skip)
        result['divergence'] = None if (base_skip or skip_divergence) else make_trio_bets(top6)

        # ev_filter (C-1: 現在は actual と同一。EV 計算は aggregator 側で別途実装予定)
        result['ev_filter'] = None if base_skip else make_trio_bets(top6)

        # odds_filter (C-2: top1 単勝 odds 1.5-20 倍 帯)
        result['odds_filter'] = None if (base_skip or skip_odds) else make_trio_bets(top6)

        return result

    except Exception as e:
        print(f"[race_notify_log_v2 build_strategy_formations fail] {e}", file=sys.stderr)
        return {k: None for k in STRATEGY_KEYS}


def compute_strategy_results(
    strategy_formations: dict,
    real_top3: list,
    real_payout_trio: int,
    bet_cost_per_race: int = 700,
) -> dict:
    """8 strategy の paper hit/miss/pnl を計算して返す。

    Args:
        strategy_formations: build_strategy_formations() の返り値
        real_top3: 実 1-3 着 馬番 list [1着, 2着, 3着]
        real_payout_trio: 三連複実配当 (円, 100 円単位)
        bet_cost_per_race: 1 race あたり 投資額 (default 700)

    Returns:
        dict: strategy key → {
            'n_bets': int,        # 購入点数 (formation が None なら 0)
            'skipped': bool,      # True = formation None (skip)
            'hit': bool,          # 的中 True
            'payout': int,        # 受取配当 (hit なら real_payout_trio, else 0)
            'investment': int,    # 投資額
            'pnl': int,           # payout - investment
        }
    """
    try:
        top3_set = set(int(x) for x in (real_top3 or []) if x is not None)
        results = {}

        for key in STRATEGY_KEYS:
            formation = strategy_formations.get(key)
            if formation is None:
                results[key] = {
                    'n_bets': 0,
                    'skipped': True,
                    'hit': False,
                    'payout': 0,
                    'investment': 0,
                    'pnl': 0,
                }
                continue

            n_bets = len(formation) if isinstance(formation, list) else 0
            investment = bet_cost_per_race

            # hit: formation 中の任意の bet が top3 と完全一致
            hit = False
            if top3_set and isinstance(formation, list):
                for bet in formation:
                    bet_set = set(int(x) for x in bet if x is not None)
                    if bet_set == top3_set:
                        hit = True
                        break

            payout = int(real_payout_trio) if hit else 0
            results[key] = {
                'n_bets': n_bets,
                'skipped': False,
                'hit': hit,
                'payout': payout,
                'investment': investment,
                'pnl': payout - investment,
            }

        return results

    except Exception as e:
        print(f"[race_notify_log_v2 compute_strategy_results fail] {e}", file=sys.stderr)
        return {k: {
            'n_bets': 0, 'skipped': True, 'hit': False,
            'payout': 0, 'investment': 0, 'pnl': 0,
        } for k in STRATEGY_KEYS}


def log_phase1(race_id, race_meta=None, ranking_top5=None,
               formation_planned=None, morning_odds=None,
               date_str=None):
    """phase 1: daily_predict 後 (朝 8:00) log。

    Args:
        race_id: レース ID (str/int)
        race_meta: dict 形式の race 情報 (race_name, course, distance 等)
        ranking_top5: top1-5 馬番 list or 同等 string
        formation_planned: 予定 formation (trio_bets string 等)
        morning_odds: 朝時点の odds dict {umaban: odds}
        date_str: 日付 (YYYYMMDD)、 未指定なら本日

    fail 時は stderr 出力のみ、 例外を投げない。
    """
    if not _validate_race_id(race_id):
        print(f"[race_notify_log_v2 phase1 SKIP] invalid race_id: {race_id!r}",
              file=sys.stderr)
        return
    try:
        d = date_str or _today_str()
        out_dir = _log_root() / d / 'phase1'
        out_file = out_dir / f'{race_id}.json'

        data = {
            'phase': 1,
            'phase_name': 'morning_predict',
            'race_id': str(race_id),
            'timestamp': datetime.now().isoformat(),
            'race_meta': race_meta or {},
            'ranking_top5': ranking_top5 if ranking_top5 is not None else [],
            'formation_planned': formation_planned if formation_planned is not None else '',
            'morning_odds': morning_odds or {},
        }
        _write_json(out_file, data)
    except Exception as e:
        print(f"[race_notify_log_v2 phase1 fail] {e}", file=sys.stderr)


def log_phase2(race_id, race_meta=None, formation_actual=None,
               vote_time_odds=None, strategy_7c_skip=False,
               strategy_7c_reason=None, channel='bets',
               cond_key=None, bet_type=None,
               predictions=None,
               date_str=None):
    """phase 2: 投票直前 (race -5min / race_auto_notify) log。

    Args:
        race_id: レース ID
        race_meta: race 情報 dict
        formation_actual: 確定 formation (bets list of tuple or string)
        vote_time_odds: 投票時 odds dict
        strategy_7c_skip: 戦略⑦C で skip された場合 True
        strategy_7c_reason: skip 理由 (strategy_7_cond_E 等)
        channel: 'bets' / 'skip' / 'error'
        cond_key: 条件 (A/B/C/D/E/X)
        bet_type: trio / umaren / wide
        predictions: top N 予測馬 list (8 strategy paper 計算用、省略可)。
            各要素は dict: {horse_num, pop_rank, odds, ...}
        date_str: 日付 (YYYYMMDD)

    fail 時は stderr のみ。
    """
    if not _validate_race_id(race_id):
        print(f"[race_notify_log_v2 phase2 SKIP] invalid race_id: {race_id!r}",
              file=sys.stderr)
        return
    try:
        d = date_str or _today_str()
        out_dir = _log_root() / d / 'phase2'
        out_file = out_dir / f'{race_id}.json'

        normalized_formation = _normalize_formation(formation_actual)

        # 8 strategy paper formation 計算
        _meta = dict(race_meta or {})
        if cond_key:
            _meta['cond_key'] = cond_key
        strategy_formations = {}
        if predictions:
            raw_sf = build_strategy_formations(predictions, _meta)
            # JSON serialise 可能形に正規化
            for k, v in raw_sf.items():
                strategy_formations[k] = _normalize_formation(v)
        else:
            strategy_formations = {k: None for k in STRATEGY_KEYS}

        data = {
            'phase': 2,
            'phase_name': 'pre_vote',
            'race_id': str(race_id),
            'timestamp': datetime.now().isoformat(),
            'race_meta': _meta,
            'formation_actual': normalized_formation if normalized_formation is not None else '',
            'vote_time_odds': vote_time_odds or {},
            'strategy_7c_skip': bool(strategy_7c_skip),
            'strategy_7c_reason': str(strategy_7c_reason) if strategy_7c_reason else None,
            'channel': str(channel) if channel else 'bets',
            'cond_key': str(cond_key) if cond_key else None,
            'bet_type': str(bet_type) if bet_type else None,
            'strategy_formations': strategy_formations,
        }
        _write_json(out_file, data)
    except Exception as e:
        print(f"[race_notify_log_v2 phase2 fail] {e}", file=sys.stderr)


def log_phase3(race_id, real_top3=None, real_payouts=None,
               hit_miss=None, date_str=None):
    """phase 3: 結果回収後 (20:00 daily_results) log。

    Args:
        race_id: レース ID
        real_top3: 実 1-3 着 馬番 list [1着, 2着, 3着]
        real_payouts: 実配当 dict {'trio': 1500, 'umaren': 800, ...}
        hit_miss: 的中判定 dict {'trio_hit': True, 'umaren_hit': False}
        date_str: 日付 (YYYYMMDD)

    8 strategy paper results は phase2 の strategy_formations を読み込んで自動計算。

    fail 時は stderr のみ。
    """
    if not _validate_race_id(race_id):
        print(f"[race_notify_log_v2 phase3 SKIP] invalid race_id: {race_id!r}",
              file=sys.stderr)
        return
    try:
        d = date_str or _today_str()
        out_dir = _log_root() / d / 'phase3'
        out_file = out_dir / f'{race_id}.json'

        # 8 strategy paper results 計算 (phase2 log を読み込む)
        strategy_results = {}
        try:
            p2_file = _log_root() / d / 'phase2' / f'{race_id}.json'
            if p2_file.exists():
                with open(p2_file, 'r', encoding='utf-8') as fp:
                    p2_data = json.load(fp)
                sf = p2_data.get('strategy_formations', {})
                trio_payout = int((real_payouts or {}).get('trio') or 0)
                strategy_results = compute_strategy_results(
                    sf, real_top3 or [], trio_payout
                )
        except Exception as e2:
            print(f"[race_notify_log_v2 phase3 strategy_results fail] {e2}", file=sys.stderr)

        data = {
            'phase': 3,
            'phase_name': 'post_result',
            'race_id': str(race_id),
            'timestamp': datetime.now().isoformat(),
            'real_top3': real_top3 if real_top3 is not None else [],
            'real_payouts': real_payouts or {},
            'hit_miss': hit_miss or {},
            'strategy_results': strategy_results,
        }
        _write_json(out_file, data)
    except Exception as e:
        print(f"[race_notify_log_v2 phase3 fail] {e}", file=sys.stderr)


def read_phase(race_id, phase, date_str=None):
    """指定 race / phase の log を読み込む (None if not exist)。

    test / aggregator から利用。
    """
    try:
        d = date_str or _today_str()
        out_file = _log_root() / d / f'phase{phase}' / f'{race_id}.json'
        if not out_file.exists():
            return None
        with open(out_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[race_notify_log_v2 read fail] {e}", file=sys.stderr)
        return None
