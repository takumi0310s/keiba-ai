"""T1: features 真値テスト (TYB merge bug 級事故 永久防止)

このテストは V15 cache (data/_v15_optuna_df_cache.pkl.gz) の 145 features を
audit し、 TYB merge bug (1 年以上 検出されずゼロ寄与) と同型の事故を防ぐ。

★ V15 model / production は完全不変。 read-only audit のみ ★
★ 既知 red flag (8 件) は documented として PASS 扱い (これらを fail させるには
   V15 再学習が必要なため、 monitor 段階では warning に留める) ★

usage:
    python -m pytest tests/T1_features_integrity_test.py -v
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CACHE_PATH = os.path.join(BASE_DIR, 'data', '_v15_optuna_df_cache.pkl.gz')
MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz')
AUDIT_DOC_DIR = os.path.join(BASE_DIR, 'docs')


def _find_audit_doc():
    """Return path to T1 audit doc, or None if not present."""
    if not os.path.isdir(AUDIT_DOC_DIR):
        return None
    for fname in os.listdir(AUDIT_DOC_DIR):
        if fname.startswith('T1_FEATURES_INTEGRITY_AUDIT_') and fname.endswith('.md'):
            return os.path.join(AUDIT_DOC_DIR, fname)
    return None

# Known red flag features as of 2026-05-17 (8 件、 V15 内 importance=0 のため model に害なし)
KNOWN_RED_CONSTANT_FEATURES = {
    'is_nar',
    'prev_odds_log',
    'prev_race_first3f',
    'prev_race_last3f',
    'prev_race_pace_diff',
    'sire_shinba_top3r',
    'pci',
    'gaisha_rank',
}


@pytest.fixture(scope='module')
def cache_data():
    """Load V15 cache once for all tests."""
    if not os.path.exists(CACHE_PATH):
        pytest.skip(f'Cache not found: {CACHE_PATH}')
    with gzip.open(CACHE_PATH, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def v15_importance():
    """Load V15 model importance once."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f'Model not found: {MODEL_PATH}')
    with gzip.open(MODEL_PATH, 'rb') as f:
        m = pickle.load(f)
    feats = m['features']
    try:
        lgb_imp = m['model'].feature_importance(importance_type='gain')
    except Exception:
        lgb_imp = [0.0] * len(feats)
    try:
        xgb_imp_raw = m['xgb_model'].get_score(importance_type='gain')
        xgb_imp_d = {}
        for k, v in xgb_imp_raw.items():
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feats):
                    xgb_imp_d[feats[idx]] = float(v)
            else:
                xgb_imp_d[k] = float(v)
    except Exception:
        xgb_imp_d = {}
    imp = {}
    for i, f in enumerate(feats):
        imp[f] = {
            'lgb_gain': float(lgb_imp[i]) if i < len(lgb_imp) else 0.0,
            'xgb_gain': float(xgb_imp_d.get(f, 0.0)),
        }
    return imp


def test_v15_cache_loaded_with_145_features(cache_data):
    """V15 cache に df と features が含まれ、 features は 145 個。"""
    assert 'df' in cache_data, 'cache missing df'
    assert 'features' in cache_data, 'cache missing features list'
    feats = cache_data['features']
    assert len(feats) == 145, f'V15 features count != 145, got {len(feats)}'


def test_v15_features_all_present_in_df(cache_data):
    """V15 で使用される 145 features 全てが df に存在する。"""
    df = cache_data['df']
    feats = cache_data['features']
    missing = [c for c in feats if c not in df.columns]
    assert not missing, f'V15 features missing from df: {missing}'


def test_v15_model_features_match_cache_features(cache_data, v15_importance):
    """V15 model の features と cache の features が完全一致。"""
    cache_feats = set(cache_data['features'])
    model_feats = set(v15_importance.keys())
    in_model_not_cache = model_feats - cache_feats
    in_cache_not_model = cache_feats - model_feats
    assert not in_model_not_cache, f'In model not cache: {in_model_not_cache}'
    assert not in_cache_not_model, f'In cache not model: {in_cache_not_model}'


def test_no_new_red_constant_features(cache_data):
    """新規 RED_CONSTANT (unique<=1) 検出時は FAIL。

    既知 8 件は許容 (V15 再学習なしでは修正不可)、 新たに発生したら TYB 級事故。
    """
    df = cache_data['df']
    feats = cache_data['features']
    new_red = []
    for c in feats:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=True) <= 1 and c not in KNOWN_RED_CONSTANT_FEATURES:
            new_red.append(c)
    assert not new_red, (
        f'NEW RED_CONSTANT features detected (TYB-class bug suspect): {new_red}. '
        f'これらは V15 内で全行 default value、 features 取得 pipeline の故障可能性大。'
    )


def test_no_red_imp_but_const(cache_data, v15_importance):
    """importance > 0 だが unique <= 1 = ★ critical ★ (model 入力 but 分散なし)。"""
    df = cache_data['df']
    feats = cache_data['features']
    violators = []
    for c in feats:
        if c not in df.columns:
            continue
        uniq = df[c].nunique(dropna=True)
        gain = v15_importance.get(c, {}).get('lgb_gain', 0.0)
        if gain > 0 and uniq <= 1:
            violators.append((c, gain))
    assert not violators, (
        f'RED_IMP_BUT_CONST detected: {violators}. '
        f'model に load されているが分散ゼロ → 学習時と production で齟齬。'
    )


def test_no_missing_features(cache_data):
    """V15 features が 1 件も missing でない。"""
    df = cache_data['df']
    feats = cache_data['features']
    missing = [c for c in feats if c not in df.columns]
    assert not missing, f'V15 features missing from cache df: {missing}'


def test_null_rate_acceptable(cache_data):
    """全 features の null_rate < 50%。"""
    df = cache_data['df']
    feats = cache_data['features']
    high_null = []
    for c in feats:
        if c not in df.columns:
            continue
        null_rate = df[c].isna().mean()
        if null_rate > 0.5:
            high_null.append((c, null_rate))
    assert not high_null, f'Features with null_rate > 50%: {high_null}'


def test_known_red_flag_features_documented():
    """既知 red flag 8 件が T1 audit doc に記載されている。"""
    doc_path = _find_audit_doc()
    if doc_path is None:
        pytest.skip('Audit doc not yet created (docs/T1_FEATURES_INTEGRITY_AUDIT_*.md)')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    missing = []
    for c in KNOWN_RED_CONSTANT_FEATURES:
        if c not in content:
            missing.append(c)
    assert not missing, f'Known red flags not documented in {doc_path}: {missing}'


def test_integrity_monitor_script_exists():
    """tools/features_integrity_monitor.py が存在する。"""
    p = os.path.join(BASE_DIR, 'tools', 'features_integrity_monitor.py')
    assert os.path.exists(p), 'features_integrity_monitor.py missing'


def test_tyb_known_suspects_not_in_v15_features(cache_data):
    """sub-task 6 で発見した TYB known suspects 5 件が V15 features に **含まれない** こと。

    含まれていたら V15 が壊れた特徴量を使っているため critical。
    """
    feats = set(cache_data['features'])
    tyb_known = {
        'jrdb_paddock_idx', 'jrdb_odds_idx', 'jrdb_body_code',
        'jrdb_demeanor_code', 'jrdb_live_composite_idx',
    }
    intersect = tyb_known & feats
    assert not intersect, (
        f'TYB-known-bad features found in V15 feature list: {intersect}. '
        f'V15 がこれらを使っていたら model 真値性が破綻する。'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
