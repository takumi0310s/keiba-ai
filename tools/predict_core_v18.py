#!/usr/bin/env python
"""V18 candidate predict core — Phase 12 DataLab 拡張 17 features skeleton.

Phase 12 (2026-05-10) で追加した 17 features の skeleton 実装。
本ファイルは V15 production (tools/predict_core.py) と完全に分離。
V15 投資保護: predict_core.py / daily_predict.py / app.py / V15 model file 不変。

Feature categories (Phase 12 user 指示の functional 分類):
  A. オッズ拡張 (4)  — JV-Link O1/O2/O3/O4/O5/O6 records 由来
  B. 番組情報 (3)    — JV-Link RA + BT records 由来
  C. ハロンタイム (3) — JV-Link SE record (区間タイム + 確定上がり)
  D. 天候馬場 (3)    — JV-Link WE / WH records (馬場差 + 含水率 + 天候)
  E. 血統拡張 (4)    — JV-Link UM / SK / BR records (父系/母父系 距離+馬場 適性)

Live activation timing:
  本 skeleton は default fill 値返却のみ。
  実 fetch は 5/24+ JV-Link 32-bit Python venv backfill 完了後に切替。
  Phase 3 後半 (5/24-6/8) で V20 構築時に 全 17 features 動的 fetch 化。

Imports / 依存:
  本 module は V15 既存 features と独立。 caller 側 (V18 学習 pipeline) で merge。
"""
from __future__ import annotations
import os
import json
from typing import Dict, List, Any

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
JVLINK_DIR = os.path.join(DATA_DIR, 'jvlink')

# =========================================================================
# Feature registry — 17 features 全 list
# =========================================================================

# A. オッズ拡張 (4 features) — JV-Link O1-O6 records
ODDS_EXPANSION_FEATURES: List[str] = [
    'jv_tansho_odds_open',     # 単勝オッズ (始値、 O1)
    'jv_fukusho_low_open',     # 複勝下限オッズ (始値、 O1)
    'jv_umaren_top_odds',      # 馬連 1 番人気オッズ (O2)
    'jv_trio_top_odds',        # 三連複 1 番人気オッズ (O5、 ★ V20 投資判断 EV 計算 base ★)
]

# B. 番組情報 (3 features) — JV-Link RA + BT records
PROGRAM_INFO_FEATURES: List[str] = [
    'jv_race_class_detail',    # レース格付け詳細 (RA、 G1=10/G2=8/G3=6/L=5/OP=4/3勝C=3/2勝C=2/1勝C=1/未勝利=0)
    'jv_prize_structure_total', # 総賞金 (RA、 1着〜5着 合計、 単位千円)
    'jv_entry_condition_enc',  # 出走条件 enc (BT、 牡牝混合=0/牝限=1/特定条件=2)
]

# C. ハロンタイム (3 features) — JV-Link SE record (区間タイム埋め込み)
HARONTIME_FEATURES: List[str] = [
    'jv_lap_first3f_pred',     # 前走前半 3F タイム (SE record 内 lap_pred、 確定値)
    'jv_lap_last3f_pred',      # 前走後半 3F タイム (SE record、 上がり 3F 確定)
    'jv_race_pace_index',      # レースペース指標 (前半 / 後半、 1.0 が標準ペース)
]

# D. 天候馬場 (3 features) — JV-Link WE / WH records
WEATHER_BABA_FEATURES: List[str] = [
    'jv_baba_moisture',        # 馬場含水率 (WE record、 % 単位、 不明 = -1)
    'jv_baba_difference',      # 馬場差 (WE、 内有利 = 正値、 外有利 = 負値、 中央 = 0)
    'jv_weather_change_score', # 天候変化スコア (WH 履歴差分、 急変 = 1、 安定 = 0)
]

# E. 血統拡張 (4 features) — JV-Link UM / SK / BR records
BLOODLINE_FEATURES: List[str] = [
    'jv_sire_dist_apt_score',     # 父系 距離適性 score (UM + SK 集計、 0-1 normalized)
    'jv_dam_sire_apt_score',      # 母父系 適性 score (UM + BR、 0-1)
    'jv_sire_surface_apt_score',  # 父系 馬場適性 (UM + SK、 芝 / ダート 別集計、 0-1)
    'jv_ped_score_blend',         # 血統 総合 score (sire 0.5 + dam_sire 0.3 + bms 0.2、 0-1)
]

# 合計 17 features
ALL_V18_PHASE12_FEATURES: List[str] = (
    ODDS_EXPANSION_FEATURES
    + PROGRAM_INFO_FEATURES
    + HARONTIME_FEATURES
    + WEATHER_BABA_FEATURES
    + BLOODLINE_FEATURES
)

# Default fill values (live fetch 失敗 / data 未到着時)
V18_PHASE12_DEFAULTS: Dict[str, Any] = {
    # A. オッズ
    'jv_tansho_odds_open': 10.0,    # 中央値想定
    'jv_fukusho_low_open': 2.0,
    'jv_umaren_top_odds': 30.0,
    'jv_trio_top_odds': 100.0,
    # B. 番組
    'jv_race_class_detail': 0,      # 未勝利 default
    'jv_prize_structure_total': 5000,  # 平場 5000 千円 想定
    'jv_entry_condition_enc': 0,    # 牡牝混合 default
    # C. ハロン
    'jv_lap_first3f_pred': 36.0,    # 1200m 前半 3F 平均
    'jv_lap_last3f_pred': 36.0,     # 後半 3F 平均
    'jv_race_pace_index': 1.0,      # 標準ペース
    # D. 天候馬場
    'jv_baba_moisture': -1.0,       # 不明
    'jv_baba_difference': 0.0,      # 中央
    'jv_weather_change_score': 0,   # 安定
    # E. 血統
    'jv_sire_dist_apt_score': 0.5,
    'jv_dam_sire_apt_score': 0.5,
    'jv_sire_surface_apt_score': 0.5,
    'jv_ped_score_blend': 0.5,
}

# =========================================================================
# Skeleton fetcher — 5/24+ で実 JV-Link fetch に切替
# =========================================================================


def _is_jvlink_available() -> bool:
    """JV-Link 32-bit Python venv backfill data 存在 check."""
    return os.path.exists(os.path.join(JVLINK_DIR, 'O1')) and \
           os.path.exists(os.path.join(JVLINK_DIR, 'RACE')) and \
           os.path.exists(os.path.join(JVLINK_DIR, 'WH'))


def fetch_odds_expansion(race_id: str, umaban: int) -> Dict[str, float]:
    """A. オッズ拡張 4 features (O1-O6 records 由来).

    Phase 12 skeleton: default fill のみ。
    5/24+ で tools/jvlink_fetcher_v2.py 経由 実 data fetch に切替。

    O1: 単複オッズ → tansho_odds_open / fukusho_low_open
    O2: 馬連オッズ → umaren_top_odds
    O5: 三連複オッズ → trio_top_odds (V20 投資判断 EV 計算 base)
    """
    if not _is_jvlink_available():
        return {f: V18_PHASE12_DEFAULTS[f] for f in ODDS_EXPANSION_FEATURES}
    # TODO 5/24+: data/jvlink/{O1, O2, O5}/<race_id>_parsed.csv から読み出し
    return {f: V18_PHASE12_DEFAULTS[f] for f in ODDS_EXPANSION_FEATURES}


def fetch_program_info(race_id: str) -> Dict[str, Any]:
    """B. 番組情報 3 features (RA + BT records 由来).

    RA record: 距離 / 馬場 / クラス / 賞金 (1-5 着)
    BT record: 番組テーブル (出走条件 / 牝限 / 特定 等)
    """
    if not _is_jvlink_available():
        return {f: V18_PHASE12_DEFAULTS[f] for f in PROGRAM_INFO_FEATURES}
    # TODO 5/24+: data/jvlink/RACE/<race_id>_parsed.csv から RA fields 抽出
    return {f: V18_PHASE12_DEFAULTS[f] for f in PROGRAM_INFO_FEATURES}


def fetch_harontime(race_id: str, umaban: int, prev_race_id: str | None = None) -> Dict[str, float]:
    """C. ハロンタイム 3 features (SE record 区間タイム).

    SE record (前走の馬毎レース情報) から:
      - 前走前半 3F (lap_first3f_pred)
      - 前走後半 3F (lap_last3f_pred、 = 上がり 3F 確定値)
      - レースペース指標 (前半 / 後半 比、 race_pace_index)
    """
    if not _is_jvlink_available() or prev_race_id is None:
        return {f: V18_PHASE12_DEFAULTS[f] for f in HARONTIME_FEATURES}
    # TODO 5/24+: data/jvlink/SE/<prev_race_id>_parsed.csv から umaban 行取得
    return {f: V18_PHASE12_DEFAULTS[f] for f in HARONTIME_FEATURES}


def fetch_weather_baba(race_id: str) -> Dict[str, float]:
    """D. 天候馬場 3 features (WE + WH records).

    WE record: 馬場状態 (含水率 / 馬場差 / 馬場種別)
    WH record: 重量天候履歴 (前 R 比 天候変化)
    """
    if not _is_jvlink_available():
        return {f: V18_PHASE12_DEFAULTS[f] for f in WEATHER_BABA_FEATURES}
    # TODO 5/24+: data/jvlink/{WE, WH}/<race_id>_parsed.csv から fields 抽出
    return {f: V18_PHASE12_DEFAULTS[f] for f in WEATHER_BABA_FEATURES}


def fetch_bloodline(blood_num: str) -> Dict[str, float]:
    """E. 血統拡張 4 features (UM + SK + BR records).

    UM record: 馬個体 (1936-2025 全 90 年分)
    SK record: 産駒情報 (種牡馬 → 産駒の距離 / 馬場別成績)
    BR record: 繁殖牝馬 (母系の産駒成績)

    Score 計算:
      sire_dist_apt = sire の同距離帯 top3 率 / 全産駒 top3 率
      sire_surface_apt = sire の同馬場 top3 率 / 全産駒 top3 率
      dam_sire_apt = 母父 同距離 + 同馬場 複合 score
      ped_blend = 0.5 * sire + 0.3 * dam_sire + 0.2 * bms_dist
    """
    if not _is_jvlink_available() or not blood_num:
        return {f: V18_PHASE12_DEFAULTS[f] for f in BLOODLINE_FEATURES}
    # TODO 5/24+: data/jvlink/{UM, SK, BR}/<blood_num>_parsed.csv 経由 集計
    return {f: V18_PHASE12_DEFAULTS[f] for f in BLOODLINE_FEATURES}


# =========================================================================
# 統合 fetcher (caller-friendly)
# =========================================================================


def fetch_all_phase12_features(
    race_id: str,
    umaban: int,
    blood_num: str | None = None,
    prev_race_id: str | None = None,
) -> Dict[str, Any]:
    """Phase 12 全 17 features を一括取得.

    Returns:
        Dict[str, Any]: 17 features 名 → 値.
                        live data 不在時は V18_PHASE12_DEFAULTS 値.
    """
    out: Dict[str, Any] = {}
    out.update(fetch_odds_expansion(race_id, umaban))
    out.update(fetch_program_info(race_id))
    out.update(fetch_harontime(race_id, umaban, prev_race_id))
    out.update(fetch_weather_baba(race_id))
    out.update(fetch_bloodline(blood_num or ''))
    assert len(out) == 17, f"expected 17 features, got {len(out)}"
    return out


def get_v18_phase12_feature_names() -> List[str]:
    """Phase 12 で追加された 17 features 名 list."""
    return list(ALL_V18_PHASE12_FEATURES)


def get_v18_phase12_defaults() -> Dict[str, Any]:
    """Phase 12 17 features の default 値 (live fetch 失敗時)."""
    return dict(V18_PHASE12_DEFAULTS)


# =========================================================================
# self test (skeleton 動作確認)
# =========================================================================

if __name__ == '__main__':
    print(f"[predict_core_v18] Phase 12 features: {len(ALL_V18_PHASE12_FEATURES)} 件")
    print(f"[predict_core_v18] JV-Link backfill 利用可: {_is_jvlink_available()}")

    # dummy fetch (default 動作確認)
    dummy = fetch_all_phase12_features(
        race_id='202605020611',
        umaban=1,
        blood_num='2023104705',
        prev_race_id='202605020512',
    )
    print(f"[predict_core_v18] dummy fetch keys: {len(dummy)} 件")
    assert set(dummy.keys()) == set(ALL_V18_PHASE12_FEATURES), 'feature 名 不一致'
    print("[predict_core_v18] OK: 全 17 features default 取得 成功")

    # category 別 内訳
    print(f"  A. オッズ拡張 ({len(ODDS_EXPANSION_FEATURES)}): {ODDS_EXPANSION_FEATURES}")
    print(f"  B. 番組情報 ({len(PROGRAM_INFO_FEATURES)}): {PROGRAM_INFO_FEATURES}")
    print(f"  C. ハロンタイム ({len(HARONTIME_FEATURES)}): {HARONTIME_FEATURES}")
    print(f"  D. 天候馬場 ({len(WEATHER_BABA_FEATURES)}): {WEATHER_BABA_FEATURES}")
    print(f"  E. 血統拡張 ({len(BLOODLINE_FEATURES)}): {BLOODLINE_FEATURES}")
