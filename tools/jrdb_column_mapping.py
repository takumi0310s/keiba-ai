"""JRDB SED/TYB/CYB 日本語カラム名 → 英語スネークケース マッピング

既存 data/jrdb_{sed,tyb,cyb}.csv は日本語カラム (scrape_jrdb.py 産) のまま。
v2 CSV では英語スネークケースに統一してメンテ性を高める。

ポリシー:
- jrdb_features.py 内の _sed_col_map / _tyb_col_map (英語→日本語) と整合する英語名を採用
- 先行する download_parse_jrdb_batch2.py 系の英語カラム (race_id, umaban, idm, yyyymmdd) を優先継承
- 未定義のフィールドは JRDB 用語のローマ字スネークで命名

使い方:
    from tools.jrdb_column_mapping import SED_JP_TO_EN, rename_jp_to_en
    df_en = rename_jp_to_en(df_jp, 'SED')
"""
from __future__ import annotations

from typing import Mapping


# 共通レースキー (KYI/TYB/SED/CYB で共通)
_COMMON_KEY_MAP: Mapping[str, str] = {
    "場コード": "jo_code",
    "年": "year2",
    "回": "kai",
    "日": "nichi",
    "R": "race_num",
    "馬番": "umaban",
    "jra_race_id": "jra_race_id",
    "nk_race_id": "nk_race_id",
}


# ===== SED (成績データ) =====
SED_JP_TO_EN: Mapping[str, str] = {
    **_COMMON_KEY_MAP,
    "血統登録番号": "blood_num",
    "年月日": "yyyymmdd",
    "馬名": "horse_name",
    "距離": "distance",
    "芝ダ障害コード": "surface_code",
    "右左": "lr_code",
    "内外": "inout_code",
    "馬場状態": "baba_state",
    "種別": "shubetsu",
    "頭数": "num_horses",
    "着順": "finish",
    "異常区分": "abnormal",
    "タイム": "time_min_sec",
    "斤量": "weight_carry",
    "確定単勝オッズ": "odds_final",
    "確定単勝人気": "popularity",
    "IDM": "idm",
    "素点": "soten",
    "馬場差": "baba_sa",
    "ペース": "pace",
    "出遅": "deokure",
    "位置取": "ichi_dori",
    "不利": "furi",
    "前不利": "mae_furi",
    "中不利": "naka_furi",
    "後不利": "ato_furi",
    "レース": "race_score",
    "コース取り": "course_dori",
    "上昇度コード": "josho_code",
    "クラスコード": "class_code",
    "馬体コード": "batai_code",
    "気配コード": "kehai_code",
    "レースペース": "race_pace",
    "馬ペース": "horse_pace",
    "テン指数": "ten_idx",
    "上がり指数": "agari_idx",
    "ペース指数": "pace_idx",
    "レースP指数": "race_pace_idx",
    "前3Fタイム": "first3f_sec",
    "後3Fタイム": "last3f_sec",
    "コーナー順位1": "corner1_pos",
    "コーナー順位2": "corner2_pos",
    "コーナー順位3": "corner3_pos",
    "コーナー順位4": "corner4_pos",
    "馬体重": "horse_weight",
    "馬体重増減": "horse_weight_diff",
    "天候コード": "weather_code",
    "レースペース流れ": "race_pace_flow",
    "馬ペース流れ": "horse_pace_flow",
    "4角コース取り": "corner4_dori",
}


# ===== TYB (直前データ) =====
TYB_JP_TO_EN: Mapping[str, str] = {
    **_COMMON_KEY_MAP,
    "IDM": "idm",
    "騎手指数": "jockey_idx",
    "情報指数": "info_idx",
    "オッズ指数": "odds_idx",
    "パドック指数": "padock_idx",
    "予備1": "reserve1",
    "総合指数": "sogo_idx",
    "馬具変更情報": "bagu_change",
    "脚元情報": "ashimoto",
    "取消フラグ": "cancel_flag",
    "馬場状態コード": "baba_state_code",
    "天候コード": "weather_code",
    "単勝オッズ": "tansho_odds",
    "複勝オッズ": "fukusho_odds",
    "馬体重": "horse_weight",
    "馬体重増減": "horse_weight_diff",
    "オッズ印": "odds_mark",
    "パドック印": "padock_mark",
    "直前総合印": "sogo_mark",
    "馬体コード": "batai_code",
    "気配コード": "kehai_code",
}


# ===== CYB (調教分析) =====
CYB_JP_TO_EN: Mapping[str, str] = {
    **_COMMON_KEY_MAP,
    "調教タイプ": "train_type",
    "調教コースタイプ": "train_course_type",
    "調教馬場": "train_baba",
    "追切指標": "train_mark",
    "仕上指標": "train_amount",
    "変化指標": "train_change",
    "調教コメント": "train_comment",
    "コメント年": "comment_year",
    "コメント回日": "comment_date",
    "調教評価": "train_eval",
    "調教コース": "train_course",
}


_TYPE_MAPS: Mapping[str, Mapping[str, str]] = {
    "SED": SED_JP_TO_EN,
    "TYB": TYB_JP_TO_EN,
    "CYB": CYB_JP_TO_EN,
}


def get_mapping(jrdb_type: str) -> Mapping[str, str]:
    """SED/TYB/CYB のいずれかの日本語→英語マッピングを返す"""
    t = jrdb_type.upper()
    if t not in _TYPE_MAPS:
        raise ValueError(f"unknown JRDB type: {jrdb_type} (expect SED/TYB/CYB)")
    return _TYPE_MAPS[t]


def rename_jp_to_en(df, jrdb_type: str):
    """DataFrame の日本語カラムを英語スネークケースに置換して返す (新 DataFrame)"""
    mapping = get_mapping(jrdb_type)
    # 既に英語のカラムは据え置き、日本語のみ置換
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def english_columns(jrdb_type: str) -> list[str]:
    """そのタイプの英語カラム名一覧 (値の集合)"""
    return sorted(set(get_mapping(jrdb_type).values()))
