"""P0-5 順3: discord_recalc_notify テスト

★ 全 mock、 実発火なし ★
★ DISCORD_WEBHOOK_UPDATES 環境変数 が未設定でも PASS する設計 ★

設計参照: docs/P0_5_RECALC_LOGIC_DESIGN_2026_05_17.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.discord_recalc_notify import (  # noqa: E402
    detect_rank_change,
    format_message,
    is_strategy_7c_excluded,
    notify_recalc,
)


# ---------- detect_rank_change ----------
def test_detect_rank_change_none():
    """変動なし -> severity='none'"""
    original = [(1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)]
    recalc = [(1, 0.85), (2, 0.72), (3, 0.62), (4, 0.52), (5, 0.42)]
    result = detect_rank_change(original, recalc)
    assert result['severity'] == 'none'
    assert result['top1_changed'] is False
    assert result['top3_swap_count'] == 0


def test_detect_rank_change_critical_top1():
    """top1 入替 -> severity='critical'"""
    original = [(1, 0.8), (2, 0.7), (3, 0.6)]
    recalc = [(2, 0.85), (1, 0.7), (3, 0.6)]
    result = detect_rank_change(original, recalc)
    assert result['severity'] == 'critical'
    assert result['top1_changed'] is True


def test_detect_rank_change_major_top3_swap():
    """top3 内 2 頭入替 -> severity='major'"""
    original = [(1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)]
    # 1 残留、 2 → 6、 3 → 7 (top3 から消えた = 2 頭)
    recalc = [(1, 0.85), (6, 0.7), (7, 0.6), (4, 0.5), (5, 0.4)]
    result = detect_rank_change(original, recalc)
    assert result['severity'] == 'major'
    assert result['top1_changed'] is False
    assert result['top3_swap_count'] == 2


def test_detect_rank_change_minor_one_swap():
    """top3 内 1 頭入替 -> severity='minor'"""
    original = [(1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
    # 1, 2 残留、 3 → 6 (top3 から 1 頭消えた)
    recalc = [(1, 0.85), (2, 0.72), (6, 0.6), (4, 0.5)]
    result = detect_rank_change(original, recalc)
    assert result['severity'] == 'minor'
    assert result['top1_changed'] is False
    assert result['top3_swap_count'] == 1


# ---------- is_strategy_7c_excluded ----------
def test_strategy_7c_kyoto_normal_excluded():
    """京都 R 平場 -> skip"""
    meta = {'course': '京都', 'condition': 'A', 'race_name': '京都 1R'}
    assert is_strategy_7c_excluded('202608030701', meta) is True


def test_strategy_7c_kyoto_g2_not_excluded():
    """京都 G2 -> skip しない (graded 保護)"""
    meta = {'course': '京都', 'condition': 'A', 'race_name': '京都記念 GⅡ'}
    assert is_strategy_7c_excluded('202608030711', meta) is False


def test_strategy_7c_kyoto_listed_not_excluded():
    """京都 listed -> skip しない"""
    meta = {'course': '京都', 'condition': 'A', 'race_name': 'なんとかS (L)'}
    assert is_strategy_7c_excluded('202608030711', meta) is False


def test_strategy_7c_condition_X_excluded():
    """条件 X 平場 -> skip"""
    meta = {'course': '東京', 'condition': 'X', 'race_name': '東京 1R'}
    assert is_strategy_7c_excluded('202605020701', meta) is True


def test_strategy_7c_condition_X_g1_not_excluded():
    """条件 X G1 -> skip しない (graded 保護)"""
    meta = {'course': '東京', 'condition': 'X', 'race_name': '日本ダービー G1'}
    assert is_strategy_7c_excluded('202605021011', meta) is False


def test_strategy_7c_tokyo_normal_not_excluded():
    """東京 A 平場 -> skip しない (production 通常 race)"""
    meta = {'course': '東京', 'condition': 'A', 'race_name': '東京 5R'}
    assert is_strategy_7c_excluded('202605020705', meta) is False


# ---------- format_message ----------
def test_format_message_contains_severity():
    """message に severity 文字が含まれる"""
    meta = {'course': '東京', 'race_name': 'test race'}
    change = {
        'top1_changed': True,
        'top3_swap_count': 1,
        'severity': 'critical',
        'changes': [{'rank': 1, 'original': 5, 'recalc': 3}],
    }
    msg = format_message('202605020611', meta, change)
    assert 'critical' in msg
    assert '202605020611' in msg


def test_format_message_paper_shadow_disclaimer():
    """V15 production 不変 文言 含まれる"""
    meta = {'course': '東京', 'race_name': 'test'}
    change = {
        'top1_changed': False, 'top3_swap_count': 1,
        'severity': 'minor', 'changes': [],
    }
    msg = format_message('test', meta, change)
    assert 'production' in msg or 'shadow' in msg


# ---------- notify_recalc (entry point) ----------
def test_notify_recalc_dry_run_critical():
    """notify_recalc dry-run、 critical で sent=True (dry-run)"""
    original = [(1, 0.8), (2, 0.7), (3, 0.6)]
    recalc = [(2, 0.85), (1, 0.7), (3, 0.6)]
    meta = {'course': '東京', 'condition': 'A', 'race_name': '東京 1R'}
    result = notify_recalc('test', original, recalc, meta, dry_run=True)
    assert result['sent'] is True  # dry-run でも True (print のみ)
    assert result['reason'] == 'notified'
    assert result['change']['severity'] == 'critical'


def test_notify_recalc_strategy_7c_skip():
    """京都 R で skip、 大変動 でも 通知しない"""
    original = [(1, 0.8), (2, 0.7), (3, 0.6)]
    recalc = [(2, 0.85), (1, 0.7), (3, 0.6)]  # critical だが
    meta = {'course': '京都', 'condition': 'A', 'race_name': '京都 5R'}
    result = notify_recalc('202608030705', original, recalc, meta, dry_run=True)
    assert result['sent'] is False
    assert result['reason'] == 'strategy_7c_excluded'


def test_notify_recalc_no_change_skip():
    """変動なし -> 通知 skip"""
    original = [(1, 0.8), (2, 0.7), (3, 0.6)]
    recalc = [(1, 0.85), (2, 0.72), (3, 0.62)]
    meta = {'course': '東京', 'condition': 'A', 'race_name': '東京 1R'}
    result = notify_recalc('test', original, recalc, meta, dry_run=True)
    assert result['sent'] is False
    assert result['reason'] == 'no_significant_change'


# ---------- false positive rate ----------
def test_false_positive_rate_under_threshold():
    """no-change pattern (微小 score 差) を 100 回 generate して、
    false positive (通知される) < 10% であることを 確認.

    ★ severity == 'none' の場合は確実に skip されるべき ★
    """
    import random
    random.seed(42)

    false_positive = 0
    n_trials = 100
    for _ in range(n_trials):
        # original: 安定した score
        original = [(i + 1, 0.9 - i * 0.05) for i in range(10)]
        # recalc: 微小 noise のみ (順位入替なし)
        recalc = [(i + 1, 0.9 - i * 0.05 + random.uniform(-0.005, 0.005))
                  for i in range(10)]
        meta = {'course': '東京', 'condition': 'A', 'race_name': 'test'}
        result = notify_recalc(f'test_{_}', original, recalc, meta, dry_run=True)
        if result['sent']:
            false_positive += 1

    fp_rate = false_positive / n_trials
    assert fp_rate < 0.1, f"false positive rate {fp_rate:.1%} >= 10%"


# ---------- production webhook isolation (★ critical ★) ----------
def test_does_not_use_bets_webhook(monkeypatch):
    """★ DISCORD_WEBHOOK_BETS は絶対 touch しない ★"""
    import tools.discord_recalc_notify as mod
    import inspect
    src = inspect.getsource(mod)
    # source 内で BETS webhook を 参照していないこと
    assert 'DISCORD_WEBHOOK_BETS' not in src, \
        "discord_recalc_notify は #買い目 webhook を 絶対 touch しない"
