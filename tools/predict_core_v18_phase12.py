#!/usr/bin/env python
"""V18 Phase 12: JRA-VAN DataLab 17 features (skeleton + 一部真値化).

Phase 12 (2026-05-10) で skeleton 設計 (commit b1751da5)。
本 module は Phase 12 PoC で 一部真値化 を加えた版:
  - 直近 1 ヶ月 backfill (4/10-5/10、 288 R) を data/jvlink/<year>/<month>/<rid>.json に出力済
  - per-race JSON が存在する場合、 race_class_detail を race_name から regex で抽出
  - 残 16 features は default fill 維持 (5/24+ JV-Link COM full backfill 待ち)

★ V15 投資保護: tools/predict_core.py / Phase 11 tools/predict_core_v18.py / V15 model 不変 ★

別 module 化の理由:
  Phase 11 (commit a2a2279b) で tools/predict_core_v18.py が 165 features 版に再定義された。
  Phase 12 17 features は本 module に切り出し、 caller (V18 学習 pipeline) で merge 想定。

honest report:
  真値化 features: 1/17 (jv_race_class_detail のみ、 race_name 解析 base)
  default 維持: 16/17 (5/24+ で JV-Link COM 経由 HY/WE/WH/UM/SK/BR parse 後 真値化)
"""
from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JVLINK_DIR = os.path.join(BASE_DIR, 'data', 'jvlink')

# =========================================================================
# Phase 12 17 features (Phase 12 commit b1751da5 から継承)
# =========================================================================

ODDS_EXPANSION_FEATURES: List[str] = [
    'jv_tansho_odds_open', 'jv_fukusho_low_open',
    'jv_umaren_top_odds', 'jv_trio_top_odds',
]

PROGRAM_INFO_FEATURES: List[str] = [
    'jv_race_class_detail', 'jv_prize_structure_total', 'jv_entry_condition_enc',
]

HARONTIME_FEATURES: List[str] = [
    'jv_lap_first3f_pred', 'jv_lap_last3f_pred', 'jv_race_pace_index',
]

WEATHER_BABA_FEATURES: List[str] = [
    'jv_baba_moisture', 'jv_baba_difference', 'jv_weather_change_score',
]

BLOODLINE_FEATURES: List[str] = [
    'jv_sire_dist_apt_score', 'jv_dam_sire_apt_score',
    'jv_sire_surface_apt_score', 'jv_ped_score_blend',
]

ALL_PHASE12_FEATURES: List[str] = (
    ODDS_EXPANSION_FEATURES + PROGRAM_INFO_FEATURES + HARONTIME_FEATURES
    + WEATHER_BABA_FEATURES + BLOODLINE_FEATURES
)

PHASE12_DEFAULTS: Dict[str, Any] = {
    'jv_tansho_odds_open': 10.0, 'jv_fukusho_low_open': 2.0,
    'jv_umaren_top_odds': 30.0, 'jv_trio_top_odds': 100.0,
    'jv_race_class_detail': 0, 'jv_prize_structure_total': 5000,
    'jv_entry_condition_enc': 0,
    'jv_lap_first3f_pred': 36.0, 'jv_lap_last3f_pred': 36.0,
    'jv_race_pace_index': 1.0,
    'jv_baba_moisture': -1.0, 'jv_baba_difference': 0.0,
    'jv_weather_change_score': 0,
    'jv_sire_dist_apt_score': 0.5, 'jv_dam_sire_apt_score': 0.5,
    'jv_sire_surface_apt_score': 0.5, 'jv_ped_score_blend': 0.5,
}

# =========================================================================
# Backfill JSON lookup
# =========================================================================


def _load_jvlink_backfill(race_id: str) -> Optional[dict]:
    """data/jvlink/<year>/<month>/<race_id>.json を読込 (存在しなければ None)."""
    if not race_id or len(race_id) < 8:
        return None
    year = race_id[:4]
    # JSON 内 date field から month 取得 path 確定 (race_id に month は埋込まれていない)
    # 試行: 04 / 05 month dir 順次 check
    for month in ('04', '05', '03', '06', '07', '08', '09', '10', '11', '12', '01', '02'):
        path = os.path.join(JVLINK_DIR, year, month, f'{race_id}.json')
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
    return None


# =========================================================================
# 真値抽出 (現状: 1/17 features のみ実 lookup 可能)
# =========================================================================

# G1/G2/G3/L/OP/3勝C/2勝C/1勝C/未勝利/新馬 の race_name pattern
_RACE_CLASS_PATTERN = [
    (re.compile(r'(?i)\bG ?I\b|G1|GⅠ'), 10),  # G1
    (re.compile(r'(?i)\bG ?II\b|G2|GⅡ'), 8),   # G2
    (re.compile(r'(?i)\bG ?III\b|G3|GⅢ'), 6),  # G3
    (re.compile(r'L\)|リステッド'), 5),          # L
    (re.compile(r'オープン|OP特別|OPEN'), 4),     # OP
    (re.compile(r'3勝クラス|3勝C|3勝'), 3),      # 3勝C
    (re.compile(r'2勝クラス|2勝C|2勝'), 2),      # 2勝C
    (re.compile(r'1勝クラス|1勝C|1勝'), 1),      # 1勝C
    (re.compile(r'未勝利'), 0),                   # 未勝利
    (re.compile(r'新馬'), 0),                     # 新馬
]


def _parse_race_class_from_name(race_name: str) -> int:
    """race_name 文字列から クラス detail を抽出 (G1=10..未勝利=0)."""
    if not race_name:
        return 0
    for pattern, code in _RACE_CLASS_PATTERN:
        if pattern.search(race_name):
            return code
    return 0


def fetch_phase12_features_with_backfill(
    race_id: str,
    umaban: int = 1,
    blood_num: Optional[str] = None,
    prev_race_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase 12 17 features を取得 (backfill 真値 + default fill).

    現状 (Phase 12 PoC):
      - jv_race_class_detail: backfill RA.race_name から抽出 (★ 真値 1/17 ★)
      - 残 16 features: default fill (5/24+ で full backfill)

    backfill data 不在 → 全 17 features default fill.
    """
    out: Dict[str, Any] = {f: PHASE12_DEFAULTS[f] for f in ALL_PHASE12_FEATURES}

    backfill = _load_jvlink_backfill(race_id)
    if backfill is not None:
        ra = backfill.get('ra', {})
        race_name = ra.get('race_name', '') or ''
        # 真値化 #1: race class detail
        out['jv_race_class_detail'] = _parse_race_class_from_name(race_name)
        # TODO 5/24+: 残 16 features を HY_DATA / WE / WH / UM_DATA / SK / BR parse 後 lookup

    return out


def get_real_value_status() -> Dict[str, str]:
    """各 features の real-value 化 status を返す."""
    status: Dict[str, str] = {}
    for f in ALL_PHASE12_FEATURES:
        if f == 'jv_race_class_detail':
            status[f] = 'POC_REAL_VALUE (race_name regex 抽出、 backfill JSON 利用時)'
        else:
            status[f] = 'DEFAULT_FILL (5/24+ JV-Link COM full backfill 待ち)'
    return status


def list_backfilled_races() -> List[str]:
    """data/jvlink/ に backfill 済の race_id list."""
    idx_path = os.path.join(JVLINK_DIR, 'phase12_poc_index.json')
    if not os.path.isfile(idx_path):
        return []
    with open(idx_path, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    return idx.get('race_ids', [])


# =========================================================================
# Self test
# =========================================================================

if __name__ == '__main__':
    print(f"[phase12] features: {len(ALL_PHASE12_FEATURES)} 件")

    backfilled = list_backfilled_races()
    print(f"[phase12] backfill 済 R: {len(backfilled)}")
    if backfilled:
        print(f"[phase12] sample race_ids: {backfilled[:3]}")

    # 1 R sample (4/26 東京)
    sample_rid = backfilled[0] if backfilled else '202605020611'
    feats = fetch_phase12_features_with_backfill(sample_rid)
    print(f"[phase12] {sample_rid} fetch: {len(feats)} features")
    print(f"[phase12]   jv_race_class_detail = {feats['jv_race_class_detail']} (★ real-value ★)")
    print(f"[phase12]   jv_tansho_odds_open  = {feats['jv_tansho_odds_open']} (default)")
    print(f"[phase12]   jv_baba_moisture     = {feats['jv_baba_moisture']} (default、 -1 = 不明)")

    status = get_real_value_status()
    real_count = sum(1 for v in status.values() if v.startswith('POC_REAL'))
    print(f"[phase12] ★ real-value 化: {real_count}/{len(ALL_PHASE12_FEATURES)} features ★")
    print(f"[phase12] ★ default 維持: {len(ALL_PHASE12_FEATURES) - real_count}/{len(ALL_PHASE12_FEATURES)} features ★")
    print("[phase12] OK: PoC 部分真値化 動作確認")
