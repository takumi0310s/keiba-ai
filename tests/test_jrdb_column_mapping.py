"""JRDB 日本語→英語スネークケースマッピングの整合性テスト。

- マッピング定義 (tools/jrdb_column_mapping.py)
- v2 CSV が英語カラムで出力されていること
- jrdb_features.py の _resolve_jrdb_csv が v2 優先でフォールバック可能なこと
- v2 カラムを既存 _rename (英→日) で復元できること
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
    sys.path.insert(0, os.path.join(BASE, 'tools'))

from tools.jrdb_column_mapping import (  # noqa: E402
    SED_JP_TO_EN, TYB_JP_TO_EN, CYB_JP_TO_EN,
    get_mapping, rename_jp_to_en, english_columns,
)
from tools.jrdb_features import _resolve_jrdb_csv  # noqa: E402

DATA_DIR = os.path.join(BASE, 'data')


# ===== マッピング定義 =====

@pytest.mark.parametrize("jrdb_type,expected_key", [
    ('SED', 'idm'), ('TYB', 'padock_idx'), ('CYB', 'train_type'),
])
def test_mapping_contains_key_fields(jrdb_type, expected_key):
    """代表的な英語キーが値として含まれる"""
    assert expected_key in get_mapping(jrdb_type).values()


def test_mapping_unique_english_values():
    """各タイプ内で英語値が一意（衝突なし）"""
    for t, m in [('SED', SED_JP_TO_EN), ('TYB', TYB_JP_TO_EN), ('CYB', CYB_JP_TO_EN)]:
        values = list(m.values())
        assert len(values) == len(set(values)), f"{t}: 英語値に重複"


def test_common_keys_consistent():
    """共通キー (jo_code, umaban, race_num, nichi, kai, year2) は3タイプで一致"""
    for key in ['jo_code', 'umaban', 'race_num', 'nichi', 'kai', 'year2']:
        s = SED_JP_TO_EN.get_val(key) if hasattr(SED_JP_TO_EN, 'get_val') else None
        # dict.values を比較
        assert key in SED_JP_TO_EN.values()
        assert key in TYB_JP_TO_EN.values()
        assert key in CYB_JP_TO_EN.values()


def test_get_mapping_unknown_raises():
    with pytest.raises(ValueError):
        get_mapping("ZZZ")


# ===== rename_jp_to_en =====

def test_rename_only_matched_columns():
    """日本語カラムだけ置換、英語カラムはそのまま"""
    df = pd.DataFrame({'IDM': [1], 'umaban': [1], 'unknown_col': ['x']})
    out = rename_jp_to_en(df, 'SED')
    assert 'idm' in out.columns
    assert 'umaban' in out.columns
    assert 'unknown_col' in out.columns
    assert 'IDM' not in out.columns


def test_english_columns_listing():
    cols = english_columns('TYB')
    assert 'idm' in cols
    assert 'padock_idx' in cols
    assert 'umaban' in cols


# ===== v2 CSV ファイル存在・カラム =====

@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, 'jrdb_sed_v2.csv')),
    reason="jrdb_sed_v2.csv not built"
)
def test_sed_v2_columns_english():
    df = pd.read_csv(os.path.join(DATA_DIR, 'jrdb_sed_v2.csv'),
                     encoding='utf-8-sig', nrows=1, low_memory=False)
    # 日本語カラムが残っていない
    jp_cols = [c for c in df.columns if any(ord(ch) > 127 for ch in c)]
    assert len(jp_cols) == 0, f"日本語カラム残存: {jp_cols}"
    # 必須英語カラム
    for c in ('idm', 'umaban', 'blood_num', 'yyyymmdd', 'jra_race_id', 'nk_race_id'):
        assert c in df.columns, f"{c} missing in sed_v2"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, 'jrdb_tyb_v2.csv')),
    reason="jrdb_tyb_v2.csv not built"
)
def test_tyb_v2_columns_english():
    df = pd.read_csv(os.path.join(DATA_DIR, 'jrdb_tyb_v2.csv'),
                     encoding='utf-8-sig', nrows=1, low_memory=False)
    jp_cols = [c for c in df.columns if any(ord(ch) > 127 for ch in c)]
    assert len(jp_cols) == 0, f"日本語カラム残存: {jp_cols}"
    for c in ('idm', 'umaban', 'padock_idx', 'odds_idx', 'sogo_idx'):
        assert c in df.columns


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, 'jrdb_cyb_v2.csv')),
    reason="jrdb_cyb_v2.csv not built"
)
def test_cyb_v2_columns_english():
    df = pd.read_csv(os.path.join(DATA_DIR, 'jrdb_cyb_v2.csv'),
                     encoding='utf-8-sig', nrows=1, low_memory=False)
    jp_cols = [c for c in df.columns if any(ord(ch) > 127 for ch in c)]
    assert len(jp_cols) == 0, f"日本語カラム残存: {jp_cols}"
    for c in ('umaban', 'train_type', 'train_comment'):
        assert c in df.columns


# ===== _resolve_jrdb_csv フォールバック =====

def test_resolve_prefers_v2(tmp_path, monkeypatch):
    """v2 と旧CSV 両方あれば v2 を優先"""
    import tools.jrdb_features as jf
    monkeypatch.setattr(jf, 'DATA_DIR', str(tmp_path))
    (tmp_path / 'jrdb_sed.csv').write_text('x')
    (tmp_path / 'jrdb_sed_v2.csv').write_text('x')
    assert jf._resolve_jrdb_csv('jrdb_sed') == str(tmp_path / 'jrdb_sed_v2.csv')


def test_resolve_falls_back_to_legacy(tmp_path, monkeypatch):
    """v2 がなければ旧CSV を返す"""
    import tools.jrdb_features as jf
    monkeypatch.setattr(jf, 'DATA_DIR', str(tmp_path))
    (tmp_path / 'jrdb_sed.csv').write_text('x')
    assert jf._resolve_jrdb_csv('jrdb_sed') == str(tmp_path / 'jrdb_sed.csv')


def test_resolve_missing_returns_legacy_path(tmp_path, monkeypatch):
    """どちらもなければ旧パスを返す（exists 判定は呼出側で）"""
    import tools.jrdb_features as jf
    monkeypatch.setattr(jf, 'DATA_DIR', str(tmp_path))
    result = jf._resolve_jrdb_csv('jrdb_sed')
    assert result == str(tmp_path / 'jrdb_sed.csv')


# ===== v2 -> 既存日本語互換 =====

@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, 'jrdb_sed_v2.csv')),
    reason="jrdb_sed_v2.csv not built"
)
def test_v2_back_to_japanese_compat():
    """v2 の英語カラムが既存 jrdb_features._rename 経由で日本語に戻せる"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'jrdb_sed_v2.csv'),
                     encoding='utf-8-sig', nrows=10, dtype=str, low_memory=False)
    _rename = {'race_id':'jra_race_id','umaban':'馬番','idm':'IDM','baba_sa':'馬場差',
               'furi':'不利','deokure':'出遅','ten_idx':'テン指数','agari_idx':'上がり指数',
               'pace_idx':'ペース指数','josho_code':'上昇度コード',
               'blood_num':'血統登録番号','yyyymmdd':'年月日'}
    for en, jp in _rename.items():
        if en in df.columns and jp not in df.columns:
            df.rename(columns={en: jp}, inplace=True)
    # 必須日本語カラムが揃う
    for c in ('血統登録番号', '年月日', 'IDM', '馬番'):
        assert c in df.columns, f"{c} not restored from v2"
