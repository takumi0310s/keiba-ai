"""T4 Leak Audit Gate — 学習前に必ず実行するリーク監査。

V22 leak incident (2026-05-26):
  track_lap__pace_diff_race / pace_second_half / lap_last_3f_race が
  現レースの race_id で JOIN されていた → POST-RACE leak。
  retro AUC delta = -0.0062 → V22 INVALID。

使い方 (学習スクリプトの冒頭で):
  from train.t4_leak_audit import run_leak_audit, REQUIRED_CLEAN_FEATURES
  run_leak_audit(df, features)  # FAIL なら sys.exit(1)

自動チェック:
  1. 禁止 feature が features に含まれていないか
  2. corr_target > 0.4 の feature を警告 (> 0.6 でエラー)
  3. JOIN が race_id 単独 (現レース) か否か (コード審査ルール: 手動 flag)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

# =====================================
# 絶対禁止 feature (POST-RACE 確定)
# =====================================
POST_RACE_CONFIRMED = {
    # V22 leak incident: 現レースのラップタイム (race_id で JOIN)
    'track_lap__pace_diff_race',
    'track_lap__pace_second_half',
    'track_lap__lap_first_3f_race',
    'track_lap__lap_last_3f_race',
    'track_lap__lap_volatility',
    # JRDB SKB: post-race 拡張データ (Session #38 確定)
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1', 'skb_baba_code_2', 'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
    # レース結果派生
    'finish', 'run_time', 'agari3f', 'pass3', 'pass4', 'popularity',
    'prize',  # 着賞金 → 結果確定後
}

# Pattern A で除外すべき当日情報 (Pattern B は OK)
PATTERN_A_LEAK = {
    'odds_log',           # 確定オッズ
    'horse_weight',       # 当日馬体重
    'condition_enc',      # 馬場状態
    'weight_change',
    'weight_change_abs',
    'weight_cat',
    'weight_cat_dist',
    'cond_surface',
    'paci_ninki_idx',     # オッズ派生 (2026-05-26 確認)
}

# corr_target 閾値
CORR_ERROR_THRESHOLD = 0.60    # これ以上は ERROR (強い leak 疑い)
CORR_WARN_THRESHOLD = 0.40     # これ以上は WARNING


def check_banned_features(features: list[str], mode: str = 'strict') -> list[str]:
    """禁止 feature が含まれているか確認。

    mode='strict': POST_RACE_CONFIRMED を全チェック
    mode='pattern_a': POST_RACE + PATTERN_A_LEAK をチェック
    Returns: 発見された禁止 feature のリスト
    """
    banned = POST_RACE_CONFIRMED.copy()
    if mode == 'pattern_a':
        banned |= PATTERN_A_LEAK
    found = [f for f in features if f in banned]
    return found


def check_corr_target(df: pd.DataFrame, features: list[str],
                       target_col: str = 'finish') -> dict[str, float]:
    """各 feature と target の相関を計算。

    target_col: 'finish' (着順) を使用。finish <= 3 の binarize も試す。
    Returns: {feature: corr} for corr > CORR_WARN_THRESHOLD
    """
    suspicious: dict[str, float] = {}
    if target_col not in df.columns:
        return suspicious

    y = df[target_col]
    if y.dtype == object:
        return suspicious

    for f in features:
        if f not in df.columns:
            continue
        x = df[f]
        if x.dtype == object:
            continue
        try:
            c = float(np.corrcoef(
                x.fillna(x.mean()).values,
                y.fillna(y.mean()).values
            )[0, 1])
            if abs(c) >= CORR_WARN_THRESHOLD:
                suspicious[f] = round(c, 4)
        except Exception:
            pass
    return suspicious


def check_join_safety(feature_name: str) -> str:
    """feature 名からJOINの安全性を推測 (ヒューリスティック)。

    Returns: 'safe' / 'warn' / 'danger'
    """
    name = feature_name.lower()
    # 前走を示すプレフィックス → 安全
    if any(name.startswith(p) for p in ('prev_', 'prev2_', 'prev3_', 'sire_', 'bms_',
                                         'sib_', 'horse_career', 'horse_dist', 'horse_surface',
                                         'horse_course', 'jockey_', 'trainer_',
                                         'frame_', 'jrdb_prev_')):
        return 'safe'
    # 現レース level の信号 → 要確認
    if any(s in name for s in ('_race', 'race_', '_current', 'current_')):
        return 'warn'
    # 既知 POST-RACE
    if feature_name in POST_RACE_CONFIRMED:
        return 'danger'
    return 'safe'


def run_leak_audit(
    df: pd.DataFrame,
    features: list[str],
    mode: str = 'pattern_a',
    target_col: str = 'finish',
    fail_on_error: bool = True,
) -> bool:
    """
    学習前リーク監査。

    Args:
        df: 学習 DataFrame
        features: 学習 feature リスト
        mode: 'strict' or 'pattern_a'
        fail_on_error: True の場合 ERROR 発見時に sys.exit(1)

    Returns:
        True if audit passed, False if warnings found (no errors)
    """
    print("\n" + "=" * 60)
    print("T4 LEAK AUDIT GATE")
    print("=" * 60)
    errors = []
    warnings = []

    # 1. 禁止 feature チェック
    banned = check_banned_features(features, mode=mode)
    if banned:
        for f in banned:
            if f in POST_RACE_CONFIRMED:
                errors.append(f"[ERROR] POST-RACE feature in feats: {f}")
            else:
                warnings.append(f"[WARN] Pattern-A leak in feats: {f}")

    # 2. corr_target チェック
    corr_suspicious = check_corr_target(df, features, target_col)
    for f, c in sorted(corr_suspicious.items(), key=lambda x: -abs(x[1])):
        if abs(c) >= CORR_ERROR_THRESHOLD:
            errors.append(f"[ERROR] corr_target={c:.3f} (>= {CORR_ERROR_THRESHOLD}): {f}")
        else:
            warnings.append(f"[WARN]  corr_target={c:.3f} (>= {CORR_WARN_THRESHOLD}): {f}")

    # 3. join safety ヒューリスティック
    danger_names = [f for f in features if check_join_safety(f) == 'warn']
    for f in danger_names:
        warnings.append(f"[WARN] suspicious name pattern (_race/*_current): {f}")

    # レポート
    for msg in errors:
        print(msg)
    for msg in warnings:
        print(msg)

    if not errors and not warnings:
        print("[PASS] No issues found.")
    elif not errors:
        print(f"\n[PASS with warnings] {len(warnings)} warnings, 0 errors.")
    else:
        print(f"\n[FAIL] {len(errors)} errors, {len(warnings)} warnings.")
        if fail_on_error:
            print("Aborting training. Fix leaks before proceeding.")
            sys.exit(1)
        return False

    print("=" * 60 + "\n")
    return True


# ==============================
# 新規 feature チェックリスト
# ==============================
FEATURE_DESIGN_RULES = """
新規 feature 設計チェックリスト (V22 leak 再発防止):

□ 1. JOIN が race_id 単独か?
     → 現レースの race_id で JOIN = 現レース情報 = POST-RACE
     → 前走の race_id でなければならない

□ 2. 取得タイミングを明示したか?
     → 発走前 X 時間で取得可能か?
     → 「レース結果として計算される」情報でないか?

□ 3. corr_target を事前確認したか?
     → 0.4 以上は WARNING
     → 0.6 以上は FAIL (run_leak_audit が検出)

□ 4. expanding window で計算されているか?
     → static CSV をそのまま使う = 未来情報混入リスク
     → cumsum.shift(1) で当該レース除外が必要

□ 5. V15/V20/V21 の既存 feature と役割が重複していないか?
     → prev_race_last3f_real 等と高相関なら不要
"""


if __name__ == '__main__':
    print(FEATURE_DESIGN_RULES)
    print("Banned POST-RACE features:", sorted(POST_RACE_CONFIRMED))
    print("Pattern A leak features:", sorted(PATTERN_A_LEAK))
