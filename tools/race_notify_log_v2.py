"""race_notify_log v2 (3 phase) — 真の formation record

★ 既存 race_auto_notify / daily_predict logic 完全不変、 log 出力 file IO のみ ★

3 phase:
- phase 1: daily_predict 後 (朝 8:00) → ranking + 朝 odds + 予定 formation
- phase 2: 投票直前 (race -5min / race_auto_notify) → 投票 formation 確定 + 投票時 odds
- phase 3: 結果回収後 (20:00 daily_results) → 実 1-3 着 + 実配当 + hit/miss

出力 path:
  data/race_notify_log_v2/{YYYYMMDD}/phase{1,2,3}/{race_id}.json

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
               cond_key=None, bet_type=None, date_str=None):
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
        date_str: 日付 (YYYYMMDD)

    fail 時は stderr のみ。
    """
    try:
        d = date_str or _today_str()
        out_dir = _log_root() / d / 'phase2'
        out_file = out_dir / f'{race_id}.json'

        # formation_actual は tuple list or string、 JSON serialise 可能形に正規化
        normalized_formation = formation_actual
        if isinstance(formation_actual, (list, tuple)):
            try:
                normalized_formation = [
                    list(b) if isinstance(b, (list, tuple)) else b
                    for b in formation_actual
                ]
            except Exception:
                normalized_formation = str(formation_actual)

        data = {
            'phase': 2,
            'phase_name': 'pre_vote',
            'race_id': str(race_id),
            'timestamp': datetime.now().isoformat(),
            'race_meta': race_meta or {},
            'formation_actual': normalized_formation if normalized_formation is not None else '',
            'vote_time_odds': vote_time_odds or {},
            'strategy_7c_skip': bool(strategy_7c_skip),
            'strategy_7c_reason': str(strategy_7c_reason) if strategy_7c_reason else None,
            'channel': str(channel) if channel else 'bets',
            'cond_key': str(cond_key) if cond_key else None,
            'bet_type': str(bet_type) if bet_type else None,
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

    fail 時は stderr のみ。
    """
    try:
        d = date_str or _today_str()
        out_dir = _log_root() / d / 'phase3'
        out_file = out_dir / f'{race_id}.json'

        data = {
            'phase': 3,
            'phase_name': 'post_result',
            'race_id': str(race_id),
            'timestamp': datetime.now().isoformat(),
            'real_top3': real_top3 if real_top3 is not None else [],
            'real_payouts': real_payouts or {},
            'hit_miss': hit_miss or {},
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
