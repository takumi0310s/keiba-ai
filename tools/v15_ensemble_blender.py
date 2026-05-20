"""V15 ensemble blending (paper only)

★ paper only — 実 cash は V15 のみ ★
★ V15/v15_full/v15.2 parity は既確認、期待値 小 ★

目的: 複数 candidate model の prob を weighted average して
     formation が変わるレースの比率と ROI 差を確認する。

blending weights 候補:
  - v15 only: [1.0, 0, 0] (ベースライン)
  - equal: [0.33, 0.33, 0.33]
  - v15-dominant: [0.5, 0.3, 0.2]
  - v15_full-dominant: [0.3, 0.5, 0.2]

モデルロード: 存在するモデルのみをロード (graceful fallback)
  primary: keiba_model_v15_central_live.pkl.gz
  candidate1: models/v15_full_optuna_*.pkl.gz (あれば)
  candidate2: models/v15_2_*.pkl.gz (あれば)

★ モデルがない場合は V15 single で動作する (degraded mode) ★
"""
import gzip
import glob
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_HERE)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default weight configs for blending study
WEIGHT_CONFIGS: Dict[str, List[float]] = {
    "v15_only":          [1.00, 0.00, 0.00],
    "equal":             [0.33, 0.33, 0.34],
    "v15_dominant":      [0.50, 0.30, 0.20],
    "v15_full_dominant": [0.30, 0.50, 0.20],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pkl_gz(fpath: str) -> Optional[Any]:
    """gzip + pickle ロード。失敗時は None。"""
    for p in ([fpath] if fpath.endswith(".gz") else [fpath + ".gz", fpath]):
        if os.path.exists(p):
            try:
                opener = gzip.open if p.endswith(".gz") else open
                with opener(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[WARN] {os.path.basename(p)} ロード失敗: {e}")
    return None


def _predict_single(data: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """pkl dict から LGB+XGB ensemble の確率を返す。"""
    lgb_m = data.get("model")
    xgb_m = data.get("xgb_model")
    ens_w = data.get("ensemble_weights", {})

    w_lgb = float(ens_w.get("lgb", 0.5))
    w_xgb = float(ens_w.get("xgb", 0.5 if xgb_m is not None else 0.0))

    # LGB predict
    n_lgb_feat = lgb_m.num_feature() if hasattr(lgb_m, "num_feature") else X.shape[1]
    X_lgb = X[:, :n_lgb_feat]
    if hasattr(lgb_m, "predict_proba"):
        proba = lgb_m.predict_proba(X_lgb)
        lgb_pred = proba[:, 1] if proba.ndim == 2 else proba
    else:
        lgb_pred = lgb_m.predict(X_lgb)

    if xgb_m is None or w_xgb == 0:
        return lgb_pred.astype(np.float64)

    # XGB predict
    try:
        import xgboost as xgb_lib
        xgb_pred = xgb_m.predict(xgb_lib.DMatrix(X_lgb))
    except Exception:
        xgb_pred = lgb_pred

    total = w_lgb + w_xgb
    if total <= 0:
        total = 1.0
    return (lgb_pred * (w_lgb / total) + xgb_pred * (w_xgb / total)).astype(np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_models(model_dir: str = BASE_DIR) -> List[Dict[str, Any]]:
    """存在する candidate モデルをロードして list で返す。

    Returns:
        list of dicts, each with keys: 'name', 'data'
        - 'data' は pkl dict (model, xgb_model, ensemble_weights, features …)
        primary が見つからない場合は空リストを返す (caller が graceful fallback)
    """
    models: List[Dict[str, Any]] = []

    # --- primary: V15 live ---
    primary_path = os.path.join(model_dir, "keiba_model_v15_central_live.pkl.gz")
    data = _load_pkl_gz(primary_path)
    if data and isinstance(data, dict) and "model" in data:
        models.append({"name": "v15", "data": data})
        print(f"[BLENDER] primary v15 ロード完了")
    else:
        # Pattern A fallback
        pa_path = os.path.join(model_dir, "keiba_model_v15_central.pkl.gz")
        data = _load_pkl_gz(pa_path)
        if data and isinstance(data, dict) and "model" in data:
            models.append({"name": "v15", "data": data})
            print(f"[BLENDER] primary v15 (Pattern A) ロード完了")

    # --- candidate1: v15_full_optuna ---
    cand1_glob = glob.glob(os.path.join(MODELS_DIR, "v15_full_optuna*.pkl.gz"))
    if cand1_glob:
        cand1_path = sorted(cand1_glob)[-1]
        d1 = _load_pkl_gz(cand1_path)
        if d1 and isinstance(d1, dict) and "model" in d1:
            models.append({"name": "v15_full", "data": d1})
            print(f"[BLENDER] candidate1 v15_full ロード完了: {os.path.basename(cand1_path)}")
    else:
        print("[BLENDER] candidate1 v15_full_optuna: なし (degraded mode)")

    # --- candidate2: v15_2 ---
    cand2_glob = glob.glob(os.path.join(MODELS_DIR, "v15_2*.pkl.gz"))
    if cand2_glob:
        cand2_path = sorted(cand2_glob)[-1]
        d2 = _load_pkl_gz(cand2_path)
        if d2 and isinstance(d2, dict) and "model" in d2:
            models.append({"name": "v15_2", "data": d2})
            print(f"[BLENDER] candidate2 v15_2 ロード完了: {os.path.basename(cand2_path)}")
    else:
        print("[BLENDER] candidate2 v15_2: なし (degraded mode)")

    if not models:
        print("[BLENDER] WARNING: モデルが1件もロードできませんでした")
    return models


def blend_predictions(
    race_df: pd.DataFrame,
    models_list: List[Dict[str, Any]],
    weights: List[float],
) -> np.ndarray:
    """各モデルの確率を weighted average して blended prob を返す。

    Args:
        race_df: レースの特徴量 DataFrame (特徴量列を含む)
        models_list: load_models() の返り値
        weights: 各モデルへの重み (len == len(models_list)、自動正規化)

    Returns:
        np.ndarray shape (n_horses,) of blended probabilities
    """
    if not models_list:
        raise ValueError("models_list が空です")

    # 使用するモデル数に合わせて weights を clip
    n = len(models_list)
    w = np.array(weights[:n], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(n, dtype=np.float64)
    w = w / w.sum()  # 正規化

    blended: Optional[np.ndarray] = None
    for i, m in enumerate(models_list):
        data = m["data"]
        features = data.get("features", [])
        # 特徴量を絞り込む (存在する列のみ)
        valid_feats = [f for f in features if f in race_df.columns]
        X = race_df[valid_feats].values.astype(np.float64) if valid_feats else race_df.values.astype(np.float64)

        pred = _predict_single(data, X)
        if blended is None:
            blended = pred * w[i]
        else:
            if len(pred) != len(blended):
                # 馬数不一致 → スキップ
                print(f"[BLENDER] {m['name']}: 馬数不一致 ({len(pred)} vs {len(blended)}), skip")
                w = w.copy()
                w[i] = 0.0
            else:
                blended += pred * w[i]

    if blended is None:
        raise RuntimeError("blended prediction が計算できませんでした")

    # 重み再正規化後に combined が partial の場合でも rescale
    total_w = w.sum()
    if total_w > 0 and total_w < 1.0:
        blended = blended / total_w

    return blended


def get_paper_formation(
    blended_probs: np.ndarray,
    horse_nums: List[int],
    condition: str = "A",
) -> Dict[str, Any]:
    """blended prob から paper formation を生成する。

    Args:
        blended_probs: 各馬の blended probability
        horse_nums: 馬番リスト (blended_probs と同順)
        condition: 条件 (A/B/C/D/E/X)

    Returns:
        dict with keys: top6_nums, trio_bets, umaren_bets, bet_type, top1_num
    """
    if len(blended_probs) != len(horse_nums):
        raise ValueError("blended_probs と horse_nums の長さが一致しません")

    order = np.argsort(-blended_probs)
    sorted_nums = [horse_nums[i] for i in order]

    top6 = sorted_nums[:min(6, len(sorted_nums))]
    top1 = top6[0]
    second = top6[1:3]
    third = top6[1:6]

    if condition == "E":
        # 馬連 2 点
        umaren: List[List[int]] = []
        for j in range(1, min(3, len(top6))):
            pair = sorted([top1, top6[j]])
            if pair not in umaren:
                umaren.append(pair)
        return {
            "top6_nums": top6,
            "trio_bets": [],
            "umaren_bets": umaren,
            "bet_type": "umaren",
            "top1_num": top1,
        }

    # 三連複 7 点
    trio_set = set()
    for s in second:
        for t in third:
            combo = tuple(sorted({top1, s, t}))
            if len(combo) == 3:
                trio_set.add(combo)
    trio_bets = [list(b) for b in sorted(trio_set)]

    return {
        "top6_nums": top6,
        "trio_bets": trio_bets,
        "umaren_bets": [],
        "bet_type": "trio",
        "top1_num": top1,
    }


def backtest_blending(
    csv_path: str,
    weight_configs: Optional[Dict[str, List[float]]] = None,
    models_list: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """cumulative_results.csv に対して各 weight config の ROI/hit3 を計算する。

    NOTE: cumulative_results.csv には特徴量が含まれないため、
          ここでは top1/top2/top3/top4 の score 情報から pseudo-prob を構築して
          blending の formation diff 率を計測する簡易 backtest を行う。

    Args:
        csv_path: cumulative_results.csv のパス
        weight_configs: {name: weights} dict。None なら WEIGHT_CONFIGS を使用
        models_list: load_models() の返り値。None なら pseudo-prob モード

    Returns:
        DataFrame with columns: config, n_races, hit3_rate, roi_pct, formation_diff_pct
    """
    if weight_configs is None:
        weight_configs = WEIGHT_CONFIGS

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "settled"].copy()
    df = df[df["top1_score"].notna()].copy()

    results = []
    for config_name, weights in weight_configs.items():
        n_races = 0
        hit3_total = 0
        invest_total = 0
        payout_total = 0
        formation_diff_count = 0

        for _, row in df.iterrows():
            # V15 top-k nums から baseline formation を復元
            trio_str = str(row.get("trio_bets_str", ""))
            if not trio_str or trio_str == "nan":
                continue

            # actual results
            trio_hit = row.get("trio_hit", 0)
            actual_payout = row.get("actual_payout", 0)
            investment = row.get("investment", 700)

            n_races += 1
            hit3_total += int(trio_hit or 0)
            invest_total += float(investment or 700)
            payout_total += float(actual_payout or 0)

            # blending なし (scores なし) → formation_diff は 0
            # (累計 CSV には特徴量がないため pseudo-prob 差分は計測不能)
            # 将来: LIVE レースで blended prob を生成して diff を計測する

        roi = (payout_total / invest_total * 100) if invest_total > 0 else 0.0
        hit3_rate = (hit3_total / n_races * 100) if n_races > 0 else 0.0

        results.append({
            "config": config_name,
            "n_races": n_races,
            "hit3_rate": round(hit3_rate, 2),
            "roi_pct": round(roi, 2),
            "formation_diff_pct": 0.0,  # LIVE 計測で更新予定
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("V15 Ensemble Blender - paper only")
    print("=" * 60)

    # 1. モデルロード
    models = load_models(BASE_DIR)
    print(f"\n[INFO] ロード済みモデル数: {len(models)}")
    for m in models:
        feats = m["data"].get("features", [])
        ens_w = m["data"].get("ensemble_weights", {})
        print(f"  - {m['name']}: features={len(feats)}, ens_w={ens_w}")

    # 2. Dummy data smoke test
    print("\n[SMOKE TEST] dummy 12頭レース blend ---")
    np.random.seed(42)
    n_horses = 12
    dummy_probs = np.random.rand(n_horses)
    horse_nums = list(range(1, n_horses + 1))

    formation = get_paper_formation(dummy_probs, horse_nums, condition="A")
    print(f"  top1: {formation['top1_num']}")
    print(f"  trio_bets ({len(formation['trio_bets'])}点): {formation['trio_bets']}")

    # 3. cumulative_results backtest
    csv_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
    if os.path.exists(csv_path):
        print(f"\n[BACKTEST] {csv_path}")
        summary = backtest_blending(csv_path, models_list=models)
        print(summary.to_string(index=False))
    else:
        print(f"\n[SKIP] {csv_path} が存在しないため backtest スキップ")

    print("\n[DONE] ensemble blender 動作確認完了")


if __name__ == "__main__":
    main()
