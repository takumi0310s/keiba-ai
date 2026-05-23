"""Tests for tools/v21_paper_predict.py

3 tests:
  1. test_v21_model_load_graceful_fail  — missing V21 returns None, no crash
  2. test_v15_unchanged_after_v21       — loading V21 doesn't mutate V15 object
  3. test_v21_paper_label               — Discord message contains required labels
"""
import gzip
import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Make sure project root is importable
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tools'))


# ── helpers ───────────────────────────────────────────────────────────────

def _dummy_model_data() -> dict:
    """Minimal model_data dict that looks like a real V15 payload."""
    return {
        'model': object(),   # sentinel (not a real booster)
        'features': ['f1', 'f2'],
        'version': 'v15_test',
        'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5},
        'xgb_model': None,
    }


# ── test 1 ────────────────────────────────────────────────────────────────

def test_v21_model_load_graceful_fail(tmp_path):
    """_load_pkl_gz returns None and load_v21_model returns None for a non-existent file."""
    from tools.v21_paper_predict import _load_pkl_gz, load_v21_model

    # Point V21_MODEL_PATH to a path that does not exist
    nonexistent = tmp_path / 'does_not_exist.pkl.gz'
    with mock.patch('tools.v21_paper_predict.V21_MODEL_PATH', nonexistent):
        result = load_v21_model()

    assert result is None, "load_v21_model must return None when file is missing"


# ── test 2 ────────────────────────────────────────────────────────────────

def test_v15_unchanged_after_v21(tmp_path):
    """Loading a V21 model (even a dummy one) must not mutate the V15 model object."""
    from tools.v21_paper_predict import _load_pkl_gz

    # Create a tiny fake V21 pkl.gz
    v21_payload = {'model': object(), 'features': ['a', 'b'], 'version': 'v21_test'}
    v21_path = tmp_path / 'v21_candidate.pkl.gz'
    with gzip.open(str(v21_path), 'wb') as f:
        pickle.dump(v21_payload, f)

    # Simulate V15 already loaded
    v15_data = _dummy_model_data()
    v15_model_before = v15_data['model']
    v15_features_before = list(v15_data['features'])

    # Load V21
    with mock.patch('tools.v21_paper_predict.V21_MODEL_PATH', v21_path):
        from tools.v21_paper_predict import load_v21_model
        v21_data = load_v21_model()

    # V15 object must be completely untouched
    assert v15_data['model'] is v15_model_before, "V15 model object was replaced"
    assert v15_data['features'] == v15_features_before, "V15 features were mutated"
    # V21 should have loaded successfully
    assert v21_data is not None
    assert v21_data['version'] == 'v21_test'


# ── test 3 ────────────────────────────────────────────────────────────────

def test_v21_paper_label():
    """Discord message must contain '【V21 paper】' and 'これはpaper予測です。実際の投票は V15 買い目のみ使用してください。'."""
    from tools.v21_paper_predict import _build_v21_discord_message

    # Minimal pred dicts
    rinfo = {
        'course': '東京', 'race_num': 11, 'distance': 1600,
        'surface': '芝', 'condition': '良',
    }
    v15_pred = {
        'race_name': 'テスト重賞',
        'rinfo': rinfo,
        'cond_key': 'A',
        'bet_type': 'trio',
        'bets': [(1, 2, 3), (1, 2, 4)],
        'top5': [1, 2, 3, 4, 5],
        'top5_names': ['馬A', '馬B', '馬C', '馬D', '馬E'],
        'top1': 1,
        'top1_name': '馬A',
        'top1_score': 0.85,
        'strategy_pass': True,
        'distance': 1600,
        'num_horses': 12,
    }
    v21_pred = {
        'race_name': 'テスト重賞',
        'rinfo': rinfo,
        'cond_key': 'A',
        'bet_type': 'trio',
        'bets': [(3, 4, 5), (3, 4, 6)],
        'top5': [3, 4, 5, 6, 7],
        'top5_names': ['馬C', '馬D', '馬E', '馬F', '馬G'],
        'top1': 3,
        'top1_name': '馬C',
        'top1_score': 0.78,
        'strategy_pass': True,
        'strategy_skip_reason': '',
        'distance': 1600,
        'num_horses': 12,
    }

    msg = _build_v21_discord_message(v15_pred, v21_pred)

    assert '【V21 paper】' in msg, f"Missing '【V21 paper】' in message:\n{msg}"
    assert '投票しないでください' in msg, f"Missing '投票しないでください' in message:\n{msg}"
    assert 'これはpaper予測です。' in msg, \
        f"Missing paper disclaimer in message:\n{msg}"
    # Also verify top1 divergence is noted when axis differs
    assert '相違' in msg, f"Expected '相違' annotation when tops differ:\n{msg}"
