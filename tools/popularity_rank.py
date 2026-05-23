"""
人気除外順位 (Popularity-Adjusted Rank, PAR) 計算モジュール

V15 production の inference は完全不変。
predict_race() を paci_ninki_idx を平均値で置換して再実行し、
「人気指数を除いた場合のスコア順位」を算出する。
通知・表示用の後処理のみ。モデル学習・評価には使わない。
"""
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

NINKI_FEATURE = 'paci_ninki_idx'
NINKI_STD_MIN = 10.0   # 全馬ほぼ同値 (JRDB未取得) のとき PAR 無意味
NINKI_DEFAULT  = 100.0  # JRDB未取得時の典型値


def calc_popularity_adjusted_rank(df_sorted, model_data, odds_available=False, race_info=None):
    """V15 スコアソート済み df を受け取り、人気除外順位列を付与して返す。

    Parameters
    ----------
    df_sorted : pd.DataFrame
        predict_race() が返した、スコアでソート済みの DataFrame。
    model_data : dict
        load_models() が返すモデルデータ。
    odds_available : bool
        predict_race() と同じ値。
    race_info : dict or None
        predict_race() と同じ値。

    Returns
    -------
    pd.DataFrame
        'PAR_rank' (int), 'PAR_label' (str), 'PAR_valid' (bool) 列を追加。
        PAR_valid=False なら JRDB 人気データなしで信頼性低い。
    """
    n = len(df_sorted)
    _fallback = df_sorted.copy()
    _fallback['PAR_rank']  = list(range(1, n + 1))
    _fallback['PAR_label'] = [f"#{i}" for i in range(1, n + 1)]
    _fallback['PAR_valid'] = False

    if NINKI_FEATURE not in df_sorted.columns:
        return _fallback

    std_ninki = df_sorted[NINKI_FEATURE].replace(0, np.nan).std()
    if pd.isna(std_ninki) or std_ninki < NINKI_STD_MIN:
        # 全馬ほぼ同値 → paci_ninki_idx に差異なし → PAR 無意味
        return _fallback

    try:
        from predict_core import predict_race as _predict_race
    except ImportError:
        return _fallback

    df_copy = df_sorted.copy()
    race_mean = df_copy[NINKI_FEATURE].replace(0, np.nan).mean()
    if pd.isna(race_mean):
        race_mean = NINKI_DEFAULT
    df_copy[NINKI_FEATURE] = race_mean

    try:
        df_reranked = _predict_race(df_copy, model_data, odds_available, race_info)
    except Exception:
        return _fallback

    score_map = dict(zip(df_reranked['馬番'].astype(int), df_reranked['スコア']))
    result = df_sorted.copy()
    result['PAR_score'] = result['馬番'].astype(int).map(score_map)
    result['PAR_rank']  = result['PAR_score'].rank(ascending=False, method='min').fillna(99).astype(int)
    result['PAR_valid'] = True

    # ラベル: スコア差 < 0.005 の近接順位は "2-3" 表示
    scores_desc = result.sort_values('PAR_score', ascending=False)['PAR_score'].values
    gaps = np.diff(scores_desc)

    def _label(rank_val):
        r = int(rank_val)
        prev_close = r > 1 and len(gaps) >= r - 1 and abs(gaps[r - 2]) < 0.005
        next_close = r < n and len(gaps) >= r   and abs(gaps[r - 1]) < 0.005
        if prev_close:
            return f"#{r-1}-{r}"
        if next_close:
            return f"#{r}-{r+1}"
        return f"#{r}"

    result['PAR_label'] = result['PAR_rank'].apply(_label)
    result = result.drop(columns=['PAR_score'], errors='ignore')
    return result


def format_par_horse_line(row, v15_rank):
    """1 馬分の通知行 (PAR 付き) を生成。

    Returns 例:
        "軸: 5 オオタニサーン (0.85) [能力:#1 / 人気:3位]"
        "2位: 10 ペッパーミル (0.72) [能力:#5 / 人気:1位] ★人気"
    """
    uma   = int(row.get('馬番', 0))
    name  = row.get('馬名', '?')
    score = float(row.get('スコア', 0))
    par_label = str(row.get('PAR_label', f"#{v15_rank}"))
    par_rank  = int(row.get('PAR_rank', v15_rank))
    par_valid = bool(row.get('PAR_valid', True))
    pop_rank  = int(row.get('pop_rank', 0) or 0)

    rank_labels = {1: '軸', 2: '2位', 3: '3位'}
    prefix = rank_labels.get(v15_rank, f"{v15_rank}位")
    pop_str = f"人気:{pop_rank}位" if pop_rank > 0 else "人気:?"
    par_str = f"能力:{par_label}" if par_valid else "能力:-(データ不足)"

    # 乖離フラグ
    diff = par_rank - v15_rank
    if diff >= 3:
        flag = " ★人気"
    elif diff <= -3:
        flag = " ★能力"
    else:
        flag = ""

    return f"{prefix}: {uma} {name} ({score:.2f}) [{par_str} / {pop_str}]{flag}"
