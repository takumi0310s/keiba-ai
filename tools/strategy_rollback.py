"""
tools/strategy_rollback.py
Strategy rollback manager: anomaly detection → auto disable C4/C3

Rollback state persists via JSON file (data/strategy_rollback_state.json).
Standalone module — no predict_core dependency.

Usage:
    from tools.strategy_rollback import get_effective_flags, check_and_apply_rollback
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ROLLBACK_STATE_FILE = BASE_DIR / 'data' / 'strategy_rollback_state.json'

# Minimum days to keep rollback active before auto-restore
ROLLBACK_COOLDOWN_DAYS = 7

_DEFAULT_STATE: dict = {
    'c4_enabled': True,
    'c3_enabled': True,
    'reason': '',
    'timestamp': '',
}


def get_rollback_state() -> dict:
    """Load current rollback state. Returns default (enabled) if file missing or corrupt."""
    if not ROLLBACK_STATE_FILE.exists():
        return dict(_DEFAULT_STATE)
    try:
        with open(ROLLBACK_STATE_FILE, encoding='utf-8') as fp:
            data = json.load(fp)
        # Fill missing keys with defaults
        result = dict(_DEFAULT_STATE)
        result.update({k: v for k, v in data.items() if k in _DEFAULT_STATE})
        return result
    except Exception:
        return dict(_DEFAULT_STATE)


def set_rollback_state(c4_enabled: bool, c3_enabled: bool, reason: str) -> None:
    """Persist rollback state to JSON with timestamp."""
    state = {
        'c4_enabled': c4_enabled,
        'c3_enabled': c3_enabled,
        'reason': reason,
        'timestamp': datetime.now().isoformat(),
    }
    try:
        ROLLBACK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ROLLBACK_STATE_FILE, 'w', encoding='utf-8') as fp:
            json.dump(state, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[strategy_rollback] set_rollback_state failed: {e}")


def check_and_apply_rollback(anomaly_results: list) -> dict:
    """
    Given list of anomaly dicts from check_strategy_anomaly / run_strategy_anomaly_scan,
    determine if rollback is needed:
    - If any anomaly has strategy in ['c4', 'c3', 'c3c4'] → disable both C4 and C3
    - If no anomaly → restore to enabled only if cooldown (7 days) has passed
    Returns current state dict after applying rollback logic.

    Args:
        anomaly_results: list of dicts with 'strategy' key (from anomaly_auto_detector)

    Returns:
        dict: current rollback state {'c4_enabled', 'c3_enabled', 'reason', 'timestamp'}
    """
    ROLLBACK_STRATEGIES = {'c4', 'c3', 'c3c4'}

    # Check if any anomaly concerns C4/C3
    triggered = [a for a in anomaly_results
                 if a.get('strategy', '').lower() in ROLLBACK_STRATEGIES]

    current_state = get_rollback_state()

    if triggered:
        # Anomaly found: disable both C4 and C3
        details = '; '.join(a.get('details', a.get('strategy', '?')) for a in triggered)
        reason = f"anomaly detected: {details}"
        set_rollback_state(c4_enabled=False, c3_enabled=False, reason=reason)
        print(f"[strategy_rollback] ROLLBACK applied - {reason}")
        return get_rollback_state()

    # No anomaly: check cooldown before restoring
    ts_str = current_state.get('timestamp', '')
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str)
            age_days = (datetime.now() - ts).total_seconds() / 86400
        except Exception:
            age_days = float('inf')
    else:
        age_days = float('inf')

    already_enabled = current_state.get('c4_enabled', True) and current_state.get('c3_enabled', True)

    if already_enabled:
        # Already in normal state; nothing to do
        return current_state

    # Disabled state: restore only after cooldown
    if age_days >= ROLLBACK_COOLDOWN_DAYS:
        reason = f"auto-restored after {age_days:.1f}d cooldown (no anomaly)"
        set_rollback_state(c4_enabled=True, c3_enabled=True, reason=reason)
        print(f"[strategy_rollback] RESTORED - {reason}")
        return get_rollback_state()

    # Still in cooldown
    remaining = ROLLBACK_COOLDOWN_DAYS - age_days
    print(f"[strategy_rollback] Cooldown active - {remaining:.1f}d remaining, C4/C3 disabled")
    return current_state


def get_effective_flags() -> tuple[bool, bool]:
    """
    Returns (c4_enabled, c3_enabled) after applying rollback state.
    Falls back to (True, True) on any error.
    """
    try:
        state = get_rollback_state()
        return bool(state.get('c4_enabled', True)), bool(state.get('c3_enabled', True))
    except Exception as e:
        print(f"[strategy_rollback] get_effective_flags error (fallback True, True): {e}")
        return True, True
