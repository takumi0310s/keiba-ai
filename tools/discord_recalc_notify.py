"""
P0-5 順3: -15 min 再計算 結果 Discord 速報通知

★ #updates channel 専用、 #買い目 channel 完全不変 ★
★ V15 production 投票判断 影響なし、 user 判断材料のみ ★

usage:
  python tools/discord_recalc_notify.py --race-id 202605020611 --mock
  python tools/discord_recalc_notify.py --race-id 202605020611 [--dry-run]

呼び出し元: tools/recalc_15min.py (★ 順 5 で 実装 ★)、 5/18+ paper shadow eval から

設計参照: docs/P0_5_RECALC_LOGIC_DESIGN_2026_05_17.md

【絶対遵守】
- DISCORD_WEBHOOK_UPDATES 環境変数のみ使用 (#updates channel)
- 買い目 channel webhook (BETS 系) は絶対 touch しない (read 含め 参照禁止)
- 戦略⑦案 C で skip 対象 race は 通知 skip
- 大変動 race のみ通知 (severity = minor 以上)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]


# ---------- core logic ----------
def detect_rank_change(
    original_ranking: Sequence[Tuple[int, float]],
    recalc_ranking: Sequence[Tuple[int, float]],
    top_n: int = 5,
) -> dict:
    """順位変動 detect.

    Args:
        original_ranking: 朝 8:00 base [(horse_num, score), ...] 降順
        recalc_ranking: -15 min recalc [(horse_num, score), ...] 降順
        top_n: 比較対象 top N (default 5)

    Returns:
        dict with keys:
            'top1_changed': bool
            'top3_swap_count': int (top3 内入替数: 元 top3 から消えた頭数)
            'severity': 'none' / 'minor' / 'major' / 'critical'
            'changes': list of {'rank', 'original', 'recalc'} (順位 ずれた箇所)
    """
    orig_top = [int(h) for h, _ in list(original_ranking)[:top_n]]
    recalc_top = [int(h) for h, _ in list(recalc_ranking)[:top_n]]

    if not orig_top or not recalc_top:
        return {
            'top1_changed': False,
            'top3_swap_count': 0,
            'severity': 'none',
            'changes': [],
        }

    top1_changed = orig_top[0] != recalc_top[0]
    orig_top3 = set(orig_top[:3])
    recalc_top3 = set(recalc_top[:3])
    top3_swap = len(orig_top3 - recalc_top3)

    if top1_changed:
        severity = 'critical'
    elif top3_swap >= 2:
        severity = 'major'
    elif top3_swap >= 1:
        severity = 'minor'
    else:
        severity = 'none'

    changes = []
    for i, (orig, recalc) in enumerate(zip(orig_top, recalc_top), 1):
        if orig != recalc:
            changes.append({'rank': i, 'original': orig, 'recalc': recalc})

    return {
        'top1_changed': top1_changed,
        'top3_swap_count': top3_swap,
        'severity': severity,
        'changes': changes,
    }


def is_strategy_7c_excluded(race_id: str, race_meta: dict) -> bool:
    """戦略⑦案 C で skip 対象か check (京都 / 条件 X、 G1/G2/G3/L 保護).

    ★ race_auto_notify.py の logic と整合 (tools/race_auto_notify.py L179-191, L285-290) ★
    """
    course = race_meta.get('course', '') or ''
    condition = race_meta.get('condition', '') or ''
    race_name = race_meta.get('race_name', '') or ''

    is_graded = any(g in race_name for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'])
    is_listed = any(s in race_name for s in ['L)', '(L)', 'OP)', '(OP)'])

    # 京都 filter (非 graded/listed のみ)
    if course == '京都' and not (is_graded or is_listed):
        return True
    # 条件 X filter (非 graded/listed のみ)
    if condition == 'X' and not (is_graded or is_listed):
        return True

    return False


def format_message(race_id: str, race_meta: dict, change_detail: dict) -> str:
    """Discord 通知 message format.

    ★ caveman mode: short word、 result only ★
    """
    severity_icon = {
        'none': '[OK]',
        'minor': '[WARN]',
        'major': '[WARN2]',
        'critical': '[CRIT]',
    }
    icon = severity_icon.get(change_detail.get('severity', 'none'), '[?]')

    race_name = race_meta.get('race_name', f'race_{race_id}')
    course = race_meta.get('course', '?')

    lines = [
        f"{icon} P0-5 recalc shadow: {course} {race_name} ({race_id})",
        f"severity: {change_detail.get('severity', 'none')}",
    ]
    if change_detail.get('top1_changed') and change_detail.get('changes'):
        first = change_detail['changes'][0]
        lines.append(
            f"top1 change: {first['original']} -> {first['recalc']}"
        )
    if change_detail.get('top3_swap_count', 0) > 0:
        lines.append(f"top3 swap: {change_detail['top3_swap_count']} 件")

    lines.append("(paper shadow eval / V15 production 不変)")
    return "\n".join(lines)


# ---------- discord ----------
def send_discord(message: str, dry_run: bool = False) -> bool:
    """Discord #updates webhook 送信.

    ★ DISCORD_WEBHOOK_UPDATES 環境変数のみ使用 ★
    ★ 買い目 channel webhook は絶対 touch しない (read 含め 参照禁止) ★
    """
    if dry_run:
        print(f"[DRY-RUN] would send to #updates:\n{message}")
        return True

    webhook = os.environ.get('DISCORD_WEBHOOK_UPDATES')
    if not webhook:
        print("[WARN] DISCORD_WEBHOOK_UPDATES 未設定、 send skip", file=sys.stderr)
        return False

    try:
        import requests  # type: ignore

        r = requests.post(webhook, json={'content': message}, timeout=10)
        return r.status_code in (200, 204)
    except Exception as e:
        print(f"[ERROR] discord send fail: {e}", file=sys.stderr)
        return False


# ---------- entry point ----------
def notify_recalc(
    race_id: str,
    original_ranking: Sequence[Tuple[int, float]],
    recalc_ranking: Sequence[Tuple[int, float]],
    race_meta: dict,
    dry_run: bool = False,
) -> dict:
    """recalc_15min.py から呼ばれる entry point.

    Returns:
        dict: {
            'sent': bool,
            'reason': 'strategy_7c_excluded' | 'no_significant_change' | 'notified' | ...,
            'change': detect_rank_change の結果 (or None),
            'message': format_message の結果 (notified 時のみ),
        }
    """
    # 戦略⑦案 C skip 適用
    if is_strategy_7c_excluded(race_id, race_meta):
        return {
            'sent': False,
            'reason': 'strategy_7c_excluded',
            'change': None,
            'message': None,
        }

    # 順位変動 detect
    change = detect_rank_change(original_ranking, recalc_ranking)

    # 大変動時のみ通知 (minor 以上)
    if change['severity'] == 'none':
        return {
            'sent': False,
            'reason': 'no_significant_change',
            'change': change,
            'message': None,
        }

    # 通知
    msg = format_message(race_id, race_meta, change)
    sent = send_discord(msg, dry_run=dry_run)

    return {
        'sent': sent,
        'reason': 'notified',
        'change': change,
        'message': msg,
    }


# ---------- CLI ----------
def _load_ranking_json(path: str) -> List[Tuple[int, float]]:
    """JSON file から ranking 読込.

    Expected format:
        [[horse_num, score], [horse_num, score], ...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(int(h), float(s)) for h, s in data]


def main() -> int:
    ap = argparse.ArgumentParser(description="P0-5 recalc Discord notifier (#updates)")
    ap.add_argument('--race-id', required=True)
    ap.add_argument('--original-rankings', help='JSON file with original ranking', default=None)
    ap.add_argument('--recalc-rankings', help='JSON file with recalc ranking', default=None)
    ap.add_argument('--race-meta', help='JSON file with race meta dict', default=None)
    ap.add_argument('--dry-run', action='store_true', help='実発火しない (print のみ)')
    ap.add_argument('--mock', action='store_true', help='mock data で 動作確認 (★ 発火なし ★)')
    args = ap.parse_args()

    if args.mock:
        # mock data で 動作確認 (★ 発火なし ★)
        original = [(5, 0.85), (3, 0.72), (1, 0.65), (7, 0.58), (4, 0.52)]
        recalc = [(3, 0.88), (5, 0.80), (1, 0.68), (4, 0.55), (7, 0.50)]
        meta = {
            'race_id': args.race_id,
            'race_name': 'mock race',
            'course': '東京',
            'condition': 'A',
        }
        result = notify_recalc(args.race_id, original, recalc, meta, dry_run=True)
        print("\n=== mock result ===")
        print(json.dumps(
            {k: v for k, v in result.items() if k != 'change'},
            ensure_ascii=False,
            indent=2,
        ))
        if result.get('change'):
            print("change:", result['change'])
        return 0

    # actual call (file 経由)
    if not args.original_rankings or not args.recalc_rankings:
        print(
            "[ERROR] non-mock mode requires --original-rankings + --recalc-rankings",
            file=sys.stderr,
        )
        return 1

    original = _load_ranking_json(args.original_rankings)
    recalc = _load_ranking_json(args.recalc_rankings)
    meta: dict = {}
    if args.race_meta and os.path.exists(args.race_meta):
        with open(args.race_meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    result = notify_recalc(args.race_id, original, recalc, meta, dry_run=args.dry_run)
    print(json.dumps(
        {k: v for k, v in result.items() if k != 'change'},
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
