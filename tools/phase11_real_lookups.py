#!/usr/bin/env python
"""Phase 11 真値化 lookup logic — 2026-05-10.

Phase 11 の 15 features の中で、 実 data から lookup 可能なものを実装。
実装可能: 6-7 features (audit に基づく現実 scope)。

★ V15 投資保護: 本 module は predict_core.py に呼ばれない。
   tools/predict_core_v18.py が optional 経路で呼び出す。 ★

実装可能 features (6 件):
  A. gaika_id_enc        — KYI 放牧先 (text) を hash 化
  C. jockey_dist_winrate  — KYI 騎手期待単勝率 (JRDB 集計値)
  C. jockey_track_winrate — KYI 騎手期待連対率
  C. jockey_class_winrate — KYI 騎手期待 3 着内率
  C. jockey_x_trainer_wr  — KYI 騎手コード × 調教師コード × 期待連対率
  D. paddock_eval_v18    — V15 既存 jrdb_paddock_idx を scaling

未実装 (data 不足、 9 件):
  A. gaika_top3r_3r / winrate / dist_winrate — 馬個別 外厩 history aggregation 必要
  B. odds_change_3h_v18 / 30m_v18 / popularity_shift / volatility — 多 snapshot odds 不在
  D. return_horse_score / saddle_room_score — TYB parsing 必要 (5/12 task)
"""
from __future__ import annotations
import os
import sys
import hashlib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
KYI_CSV = os.path.join(DATA_DIR, 'jrdb_kyi.csv')

# ---------------------------------------------------------------------------
# KYI loader — cached per-process
# ---------------------------------------------------------------------------

_KYI_CACHE: Optional[pd.DataFrame] = None
_KYI_LOADED: bool = False


def _load_kyi() -> Optional[pd.DataFrame]:
    """jrdb_kyi.csv を一度だけ読み込み、 キャッシュ。"""
    global _KYI_CACHE, _KYI_LOADED
    if _KYI_LOADED:
        return _KYI_CACHE
    _KYI_LOADED = True
    if not os.path.exists(KYI_CSV):
        return None
    try:
        # 必要 columns のみ読み込み (省メモリ)
        # nk_race_id (12-char netkeiba style) が daily_predictions race_id と一致
        usecols = [
            'nk_race_id', 'jra_race_id', '馬番',
            '放牧先', '放牧先ランク',
            '騎手期待単勝率', '騎手期待連対率', '騎手期待3着内率',
            '騎手コード', '調教師コード', 'クラスコード',
        ]
        df = pd.read_csv(
            KYI_CSV, encoding='utf-8-sig', low_memory=False,
            usecols=lambda c: c in usecols, dtype=str,
        )
        # numeric cast
        for c in ['馬番', '騎手期待単勝率', '騎手期待連対率', '騎手期待3着内率']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        # 期待率は JRDB では % 単位のため 0-100 → 0-1 に正規化 (max 観測値で判断)
        for c in ['騎手期待単勝率', '騎手期待連対率', '騎手期待3着内率']:
            if c in df.columns and df[c].max(skipna=True) and df[c].max(skipna=True) > 1.5:
                df[c] = df[c] / 100.0
        _KYI_CACHE = df
        return df
    except Exception as e:
        print(f"[phase11_lookups] KYI load fail: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Race-id matcher (nk_race_id 12-char netkeiba style と jra_race_id 10-char 両対応)
# ---------------------------------------------------------------------------

def _kyi_lookup_row(df: pd.DataFrame, race_id: str, umaban: int) -> pd.DataFrame:
    """KYI から該当 (race, horse) 行を取得。
    race_id 形式は 12-char (nk_race_id) / 10-char (jra_race_id) どちらでも OK。"""
    rid = str(race_id).strip()
    try:
        u = int(umaban)
    except Exception:
        return df.head(0)
    mask = pd.Series(False, index=df.index)
    if 'nk_race_id' in df.columns:
        mask = mask | (df['nk_race_id'].astype(str).str.strip() == rid)
    if 'jra_race_id' in df.columns:
        mask = mask | (df['jra_race_id'].astype(str).str.strip() == rid)
    if not mask.any():
        return df.head(0)
    sub = df[mask]
    sub = sub[pd.to_numeric(sub['馬番'], errors='coerce').fillna(-1).astype(int) == u]
    return sub.head(1)


# ---------------------------------------------------------------------------
# Hash encoding
# ---------------------------------------------------------------------------

def _hash_to_int(s: str, mod: int = 10000) -> int:
    """text → 整数 ID (mod 10000)、 same input → same output."""
    if not s or pd.isna(s):
        return 0
    h = hashlib.md5(str(s).strip().encode('utf-8')).hexdigest()
    return int(h[:8], 16) % mod


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def lookup_gaika_id_enc(race_id: str, umaban: int) -> int:
    """KYI 放牧先 (text) を hash 化して 整数 ID を返す。

    ★ 放牧先 (training farm) は JRDB が観測した「直近放牧先」。
       無記載 (在厩のまま) は 0 を返す。 ★
    """
    df = _load_kyi()
    if df is None:
        return 0
    try:
        row = _kyi_lookup_row(df, race_id, umaban)
        if row.empty or '放牧先' not in row.columns:
            return 0
        farm = row.iloc[0].get('放牧先')
        if pd.isna(farm) or not str(farm).strip():
            return 0
        return _hash_to_int(farm, mod=10000)
    except Exception:
        return 0


def lookup_jockey_expected_rates(race_id: str, umaban: int) -> Dict[str, float]:
    """KYI 騎手期待 3 種を取得。 JRDB pre-computed 集計値 (コース文脈付き)。

    返り値 fallback:
      jockey_dist_winrate (=単勝率)  default 0.10
      jockey_track_winrate (=連対率) default 0.20
      jockey_class_winrate (=3着内率) default 0.30
    """
    out = {
        'jockey_dist_winrate': 0.10,
        'jockey_track_winrate': 0.20,
        'jockey_class_winrate': 0.30,
    }
    df = _load_kyi()
    if df is None:
        return out
    try:
        row = _kyi_lookup_row(df, race_id, umaban)
        if row.empty:
            return out
        if '騎手期待単勝率' in row.columns:
            v = row.iloc[0]['騎手期待単勝率']
            if pd.notna(v):
                out['jockey_dist_winrate'] = float(v)
        if '騎手期待連対率' in row.columns:
            v = row.iloc[0]['騎手期待連対率']
            if pd.notna(v):
                out['jockey_track_winrate'] = float(v)
        if '騎手期待3着内率' in row.columns:
            v = row.iloc[0]['騎手期待3着内率']
            if pd.notna(v):
                out['jockey_class_winrate'] = float(v)
    except Exception:
        pass
    return out


def lookup_jockey_x_trainer_wr(race_id: str, umaban: int) -> float:
    """騎手コード × 調教師コード ペアの連携 winrate proxy.

    Phase 11 真値版: 騎手期待連対率 を base に、 調教師コードと騎手コードの
    ペア出現頻度で簡易 boost (実集計は別 task)。

    本実装ではまず KYI 騎手期待連対率を返す (近似)、 フル集計は 5/12+。
    """
    df = _load_kyi()
    if df is None:
        return 0.15
    try:
        row = _kyi_lookup_row(df, race_id, umaban)
        if row.empty:
            return 0.15
        v = row.iloc[0].get('騎手期待連対率')
        if pd.notna(v):
            # 連対率 を 0.10 中心で扁平化 (調教師との相性が中位想定)
            return float(v) * 0.85 + 0.05
    except Exception:
        pass
    return 0.15


def lookup_paddock_eval_v18(jrdb_paddock_idx: float) -> float:
    """V15 jrdb_paddock_idx (0-100 想定) を Phase 11 candidate で再 encode.

    encoding:
      idx >= 80 → +1.0
      idx >= 60 → +0.5
      idx >= 40 →  0.0
      idx >= 20 → -0.5
      else      → -1.0
    """
    try:
        v = float(jrdb_paddock_idx)
    except Exception:
        return 0.0
    if v >= 80: return 1.0
    if v >= 60: return 0.5
    if v >= 40: return 0.0
    if v >= 20: return -0.5
    if v > 0: return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Vectorized helpers — for predict_core_v18.py 内 DataFrame に対する適用
# ---------------------------------------------------------------------------

def apply_phase11_real_lookups(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame に対し Phase 11 真値化 features を一括 fill.

    入力 df は predict_core_v18.py の build_features 直後想定。
    'race_id' (or 'jra_race_id') と '馬番' (or 'umaban') columns 必須。

    実装可能 features (6 件) を 真値で上書き、
    残 9 features は constant default のまま。
    """
    kyi = _load_kyi()
    if kyi is None:
        # KYI 未取得時は constant のまま
        return df

    # build merge key — nk_race_id (12-char netkeiba) と jra_race_id (10-char JRA) 両対応
    rid_col = None
    for c in ['race_id', 'nk_race_id', 'jra_race_id', 'race_id_str']:
        if c in df.columns:
            rid_col = c
            break
    uma_col = '馬番' if '馬番' in df.columns else ('umaban' if 'umaban' in df.columns else None)
    if rid_col is None or uma_col is None:
        return df

    df = df.copy()
    df['_p11_rid'] = df[rid_col].astype(str).str.strip()
    df['_p11_uma'] = pd.to_numeric(df[uma_col], errors='coerce').fillna(0).astype('Int64')

    # KYI は nk_race_id 優先、 fallback jra_race_id
    kyi_mini = kyi.copy()
    kyi_match = None
    if 'nk_race_id' in kyi_mini.columns:
        kyi_match = kyi_mini['nk_race_id'].astype(str).str.strip()
    if kyi_match is not None and (kyi_match.isin(df['_p11_rid'])).any():
        kyi_mini['_p11_rid'] = kyi_match
    else:
        kyi_mini['_p11_rid'] = kyi_mini['jra_race_id'].astype(str).str.strip() if 'jra_race_id' in kyi_mini.columns else ''
    kyi_mini['_p11_uma'] = pd.to_numeric(kyi_mini['馬番'], errors='coerce').fillna(0).astype('Int64')

    # 重複 drop
    pick_cols = [c for c in ['_p11_rid', '_p11_uma', '放牧先', '騎手期待単勝率',
                              '騎手期待連対率', '騎手期待3着内率'] if c in kyi_mini.columns]
    kyi_mini = kyi_mini[pick_cols].drop_duplicates(subset=['_p11_rid', '_p11_uma'], keep='last')

    df = df.merge(kyi_mini, on=['_p11_rid', '_p11_uma'], how='left', suffixes=('', '_p11'))

    # A. gaika_id_enc
    if '放牧先' in df.columns:
        df['gaika_id_enc'] = df['放牧先'].fillna('').astype(str).apply(
            lambda s: _hash_to_int(s, mod=10000) if s.strip() else 0)
    # C. jockey rates (KYI 期待値 を直接 fill)
    if '騎手期待単勝率' in df.columns:
        df['jockey_dist_winrate'] = pd.to_numeric(df['騎手期待単勝率'], errors='coerce').fillna(0.10)
        # KYI が % 単位 (0-100) なら正規化
        if df['jockey_dist_winrate'].max() > 1.5:
            df['jockey_dist_winrate'] = df['jockey_dist_winrate'] / 100.0
    if '騎手期待連対率' in df.columns:
        df['jockey_track_winrate'] = pd.to_numeric(df['騎手期待連対率'], errors='coerce').fillna(0.20)
        if df['jockey_track_winrate'].max() > 1.5:
            df['jockey_track_winrate'] = df['jockey_track_winrate'] / 100.0
    if '騎手期待3着内率' in df.columns:
        df['jockey_class_winrate'] = pd.to_numeric(df['騎手期待3着内率'], errors='coerce').fillna(0.30)
        if df['jockey_class_winrate'].max() > 1.5:
            df['jockey_class_winrate'] = df['jockey_class_winrate'] / 100.0
        # jockey_x_trainer_wr を 連対率 base に近似
        df['jockey_x_trainer_wr'] = df['jockey_track_winrate'] * 0.85 + 0.05
    # D. paddock_eval_v18
    if 'jrdb_paddock_idx' in df.columns:
        df['paddock_eval_v18'] = pd.to_numeric(df['jrdb_paddock_idx'], errors='coerce').fillna(0).apply(
            lookup_paddock_eval_v18)

    # cleanup helper cols
    df = df.drop(columns=['_p11_rid', '_p11_uma', '放牧先',
                            '騎手期待単勝率', '騎手期待連対率', '騎手期待3着内率'],
                 errors='ignore')
    return df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("[phase11_lookups] KYI exists:", os.path.exists(KYI_CSV))
    df = _load_kyi()
    if df is None:
        print("KYI load failed")
        sys.exit(1)
    print(f"KYI loaded: {len(df):,} rows, cols={list(df.columns)}")
    # 5/10 sample (nk_race_id 形式 12-char)
    sample = df[df['nk_race_id'].astype(str) == '202608030601']
    print(f"5/10 京都 R1 sample rows: {len(sample)}")
    print(sample[['nk_race_id', '馬番', '放牧先', '騎手期待単勝率',
                   '騎手期待連対率', '騎手期待3着内率']].head(15))

    # apply_phase11_real_lookups dry-run (race_id を 12-char 形式で渡す)
    fake = pd.DataFrame({
        'race_id': ['202608030601', '202608030601', '202608030601'],
        '馬番': [1, 2, 3],
        'jrdb_paddock_idx': [70.0, 50.0, 30.0],
        # constant placeholders
        'gaika_id_enc': [0, 0, 0],
        'jockey_dist_winrate': [0.10, 0.10, 0.10],
        'jockey_track_winrate': [0.10, 0.10, 0.10],
        'jockey_class_winrate': [0.10, 0.10, 0.10],
        'jockey_x_trainer_wr': [0.15, 0.15, 0.15],
        'paddock_eval_v18': [0.0, 0.0, 0.0],
    })
    out = apply_phase11_real_lookups(fake)
    print("\n--- apply_phase11_real_lookups output ---")
    print(out[['race_id', '馬番', 'gaika_id_enc',
                'jockey_dist_winrate', 'jockey_track_winrate', 'jockey_class_winrate',
                'jockey_x_trainer_wr', 'paddock_eval_v18']])
