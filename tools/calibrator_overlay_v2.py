"""
calibrator overlay v2 (paper only — production 投入は 6/17 採用判定後)

既存 calibrator_overlay.py を拡張 (上書きせず新規)。
V15 .pkl.gz / predict_core / daily_predict / app.py / race_auto_notify.py は一切改変禁止。

3 calibration 方式:
    a. IsotonicCalibrator  — sklearn IsotonicRegression (Platt scaling 比較用)
    b. VenueCalibrator     — per-場 calibration (累積データ 300+ 以降に本格 fit)
    c. OddsBinCalibrator   — per-odds-bin calibration (odds データ取得後に fit)

Author: 強-5 (2026-05-19)
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# scikit-learn は optional import (テスト環境での import 失敗を回避)
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

REPO = Path(__file__).resolve().parents[1]

# paper eval 用 calibrator 保存先 (production 不変)
V2_STATE_DIR = REPO / "data" / "v21"
V2_ISOTONIC_PKL = V2_STATE_DIR / "calibrator_v2_isotonic.pkl"
V2_VENUE_PKL    = V2_STATE_DIR / "calibrator_v2_venue.pkl"
V2_ODDSBIN_PKL  = V2_STATE_DIR / "calibrator_v2_oddsbin.pkl"

# odds-bin 境界 (単勝オッズ想定)
ODDS_BINS = [0.0, 1.5, 3.0, 5.0, 10.0, 20.0, float("inf")]
ODDS_BIN_LABELS = ["<1.5", "1.5-3.0", "3.0-5.0", "5.0-10.0", "10.0-20.0", "20.0+"]

# per-venue calibration に必要な最低サンプル数
MIN_VENUE_SAMPLES = 30
# per-odds-bin calibration に必要な最低サンプル数
MIN_ODDS_BIN_SAMPLES = 20

# probability bounds
PROB_MIN = 0.001
PROB_MAX = 0.999


# ============================================================
# 共通ユーティリティ
# ============================================================

def _clip_prob(x: np.ndarray) -> np.ndarray:
    """probability を [PROB_MIN, PROB_MAX] に clamp する."""
    return np.clip(x, PROB_MIN, PROB_MAX)


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn が必要です: pip install scikit-learn")


# ============================================================
# a. IsotonicCalibrator
# ============================================================

class IsotonicCalibrator:
    """isotonic regression calibration.

    sklearn IsotonicRegression を薄くラップ。
    monotonic non-decreasing 制約により over/under-confidence を補正。

    Platt scaling (sigmoid) との比較:
        - Isotonic は non-parametric、 flexible
        - Platt は 2-param sigmoid (少量データでは安定)
        - 153 サンプルでは Platt の方が汎化しやすい可能性 → 5-fold CV で比較

    Note:
        153 サンプル (cumulative_results.csv、 5/17 時点) は fit に十分だが
        per-venue / per-odds-bin の十分なサンプルがないため、
        global isotonic は paper eval のみ利用。
    """

    def __init__(self) -> None:
        _require_sklearn()
        self._iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._fitted = False
        self._n_train: int = 0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """calibrator を学習.

        Args:
            y_pred: raw probability scores (shape [N])
            y_true: binary labels 0/1 (shape [N])

        Returns:
            self (for chaining)
        """
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if len(y_pred) < 5:
            raise ValueError(f"fit に最低 5 サンプル必要 (got {len(y_pred)})")
        self._iso.fit(y_pred, y_true)
        self._fitted = True
        self._n_train = len(y_pred)
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """calibrated probability を返す.

        Args:
            y_pred: raw probability scores (shape [N] or scalar)

        Returns:
            calibrated probabilities in [PROB_MIN, PROB_MAX]
        """
        if not self._fitted:
            raise RuntimeError("fit() を先に呼んでください")
        scalar = np.ndim(y_pred) == 0
        arr = np.atleast_1d(np.asarray(y_pred, dtype=float))
        out = _clip_prob(self._iso.predict(arr))
        return float(out[0]) if scalar else out

    def brier_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """calibrated 後の Brier score."""
        _require_sklearn()
        calibrated = self.predict(np.asarray(y_pred))
        return float(brier_score_loss(y_true, calibrated))

    def log_loss_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """calibrated 後の log loss."""
        _require_sklearn()
        calibrated = self.predict(np.asarray(y_pred))
        return float(log_loss(y_true, calibrated))

    def save(self, path: Path | None = None) -> Path:
        target = path if path is not None else V2_ISOTONIC_PKL
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "IsotonicCalibrator":
        target = path if path is not None else V2_ISOTONIC_PKL
        with open(target, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        status = f"fitted (n={self._n_train})" if self._fitted else "unfitted"
        return f"IsotonicCalibrator({status})"


# ============================================================
# b. VenueCalibrator
# ============================================================

class VenueCalibrator:
    """per-場 (venue) calibration.

    東京/中山/阪神/京都/中京/福島/新潟/小倉 等、
    場ごとに別の IsotonicCalibrator を保持。

    ★ 注意 ★: 2026-05-19 時点で観測済みサンプル数:
        阪神 48 / 東京 47 / 新潟 38 / 中山 12 / 中京 8
        MIN_VENUE_SAMPLES = 30 を満たすのは 阪神・東京・新潟 のみ。
        小サンプル会場は global calibrator に fallback。

    データが 300+ に達した 6/15 頃に本格 fit 推奨。
    """

    def __init__(self, min_samples: int = MIN_VENUE_SAMPLES) -> None:
        _require_sklearn()
        self._calibrators: dict[str, IsotonicCalibrator] = {}
        self._global: IsotonicCalibrator | None = None
        self._min_samples = min_samples
        self._fitted = False
        self._venue_counts: dict[str, int] = {}

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        venues: np.ndarray,
    ) -> "VenueCalibrator":
        """per-venue calibrator を学習.

        Args:
            y_pred:  raw scores (shape [N])
            y_true:  binary labels (shape [N])
            venues:  venue names (shape [N], str array)

        Returns:
            self
        """
        y_pred  = np.asarray(y_pred, dtype=float)
        y_true  = np.asarray(y_true, dtype=float)
        venues  = np.asarray(venues, dtype=str)

        # global fallback calibrator (全データ)
        if len(y_pred) >= 5:
            self._global = IsotonicCalibrator()
            self._global.fit(y_pred, y_true)

        # per-venue calibrators
        unique_venues = np.unique(venues)
        for v in unique_venues:
            mask = venues == v
            n = mask.sum()
            self._venue_counts[str(v)] = int(n)
            if n >= self._min_samples:
                cal = IsotonicCalibrator()
                cal.fit(y_pred[mask], y_true[mask])
                self._calibrators[str(v)] = cal

        self._fitted = True
        return self

    def predict(
        self,
        y_pred: np.ndarray,
        venue: str | np.ndarray,
    ) -> np.ndarray:
        """calibrated probability を返す.

        当該 venue の calibrator があれば使用、なければ global fallback。

        Args:
            y_pred: raw scores (shape [N] or scalar)
            venue:  venue 名 (single str or array of str、 y_pred と同長)

        Returns:
            calibrated probabilities in [PROB_MIN, PROB_MAX]
        """
        if not self._fitted:
            raise RuntimeError("fit() を先に呼んでください")

        scalar_input = np.ndim(y_pred) == 0
        arr = np.atleast_1d(np.asarray(y_pred, dtype=float))

        # venue が単一 str の場合: 全要素に同一 venue を適用
        if isinstance(venue, str):
            venues = np.array([venue] * len(arr))
        else:
            venues = np.asarray(venue, dtype=str)

        out = np.empty_like(arr)
        for i, (p, v) in enumerate(zip(arr, venues)):
            v_str = str(v)
            if v_str in self._calibrators:
                out[i] = self._calibrators[v_str].predict(np.array([p]))[0]
            elif self._global is not None:
                out[i] = self._global.predict(np.array([p]))[0]
            else:
                out[i] = float(np.clip(p, PROB_MIN, PROB_MAX))

        return float(out[0]) if scalar_input else out

    def venue_summary(self) -> dict[str, Any]:
        """venue ごとのサンプル数と calibrator 有無を返す."""
        return {
            v: {"n": cnt, "has_calibrator": v in self._calibrators}
            for v, cnt in self._venue_counts.items()
        }

    def save(self, path: Path | None = None) -> Path:
        target = path if path is not None else V2_VENUE_PKL
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "VenueCalibrator":
        target = path if path is not None else V2_VENUE_PKL
        with open(target, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        status = f"fitted, venues={list(self._calibrators.keys())}" if self._fitted else "unfitted"
        return f"VenueCalibrator({status})"


# ============================================================
# c. OddsBinCalibrator
# ============================================================

class OddsBinCalibrator:
    """per-odds-bin calibration.

    単勝オッズ帯別に別の calibrator を保持。
    bin 定義: <1.5 / 1.5-3.0 / 3.0-5.0 / 5.0-10.0 / 10.0-20.0 / 20.0+

    ★ 注意 ★: 2026-05-19 時点で cumulative_results.csv に odds 列なし。
        - odds_base_*.csv or race_notify_log_v2 からマージが必要
        - 現時点では design-only; fit() は dummy data でも動作確認可能
        - 本格 fit は odds データ蓄積後 (2026-06 以降推奨)

    期待効果:
        高オッズ馬 (10倍+) は V15 が過小評価傾向、
        低オッズ馬 (<3倍) は過大評価傾向 → bin 別補正で Brier 改善見込み
    """

    def __init__(
        self,
        bins: list[float] | None = None,
        bin_labels: list[str] | None = None,
        min_samples: int = MIN_ODDS_BIN_SAMPLES,
    ) -> None:
        _require_sklearn()
        self._bins   = bins if bins is not None else ODDS_BINS
        self._labels = bin_labels if bin_labels is not None else ODDS_BIN_LABELS
        self._calibrators: dict[str, IsotonicCalibrator] = {}
        self._global: IsotonicCalibrator | None = None
        self._min_samples = min_samples
        self._fitted = False
        self._bin_counts: dict[str, int] = {}

    def _assign_bin(self, odds: float | np.ndarray) -> str | np.ndarray:
        """odds 値 → bin ラベルに変換."""
        scalar = not isinstance(odds, np.ndarray)
        arr = np.atleast_1d(np.asarray(odds, dtype=float))
        labels = np.full(len(arr), self._labels[-1], dtype=object)
        for i, (lo, hi, lb) in enumerate(
            zip(self._bins[:-1], self._bins[1:], self._labels)
        ):
            mask = (arr >= lo) & (arr < hi)
            labels[mask] = lb
        return str(labels[0]) if scalar else labels

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        odds: np.ndarray,
    ) -> "OddsBinCalibrator":
        """per-odds-bin calibrator を学習.

        Args:
            y_pred: raw scores (shape [N])
            y_true: binary labels (shape [N])
            odds:   単勝オッズ (shape [N], float)

        Returns:
            self
        """
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        odds   = np.asarray(odds, dtype=float)

        # global fallback
        if len(y_pred) >= 5:
            self._global = IsotonicCalibrator()
            self._global.fit(y_pred, y_true)

        # per-bin calibrators
        bin_assignments = self._assign_bin(odds)
        for label in self._labels:
            mask = bin_assignments == label
            n = mask.sum()
            self._bin_counts[label] = int(n)
            if n >= self._min_samples:
                cal = IsotonicCalibrator()
                cal.fit(y_pred[mask], y_true[mask])
                self._calibrators[label] = cal

        self._fitted = True
        return self

    def predict(
        self,
        y_pred: np.ndarray,
        odds: float | np.ndarray,
    ) -> np.ndarray:
        """calibrated probability を返す.

        当該 odds-bin の calibrator があれば使用、なければ global fallback。

        Args:
            y_pred: raw scores (shape [N] or scalar)
            odds:   単勝オッズ (scalar or array)

        Returns:
            calibrated probabilities in [PROB_MIN, PROB_MAX]
        """
        if not self._fitted:
            raise RuntimeError("fit() を先に呼んでください")

        scalar_input = np.ndim(y_pred) == 0
        arr = np.atleast_1d(np.asarray(y_pred, dtype=float))

        # odds が scalar の場合: 全要素に同一 odds を適用
        if np.ndim(odds) == 0:
            odds_arr = np.full(len(arr), float(odds))
        else:
            odds_arr = np.asarray(odds, dtype=float)

        bin_labels = self._assign_bin(odds_arr)
        if isinstance(bin_labels, str):
            bin_labels = np.array([bin_labels] * len(arr))

        out = np.empty_like(arr)
        for i, (p, lb) in enumerate(zip(arr, bin_labels)):
            if lb in self._calibrators:
                out[i] = self._calibrators[lb].predict(np.array([p]))[0]
            elif self._global is not None:
                out[i] = self._global.predict(np.array([p]))[0]
            else:
                out[i] = float(np.clip(p, PROB_MIN, PROB_MAX))

        return float(out[0]) if scalar_input else out

    def bin_summary(self) -> dict[str, Any]:
        """bin ごとのサンプル数と calibrator 有無を返す."""
        return {
            lb: {"n": self._bin_counts.get(lb, 0), "has_calibrator": lb in self._calibrators}
            for lb in self._labels
        }

    def save(self, path: Path | None = None) -> Path:
        target = path if path is not None else V2_ODDSBIN_PKL
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "OddsBinCalibrator":
        target = path if path is not None else V2_ODDSBIN_PKL
        with open(target, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        status = f"fitted, bins={list(self._calibrators.keys())}" if self._fitted else "unfitted"
        return f"OddsBinCalibrator({status})"


# ============================================================
# 統一エントリポイント
# ============================================================

def calibrate_v2(
    scores: float | list | np.ndarray,
    method: str = "isotonic",
    venue: str | np.ndarray | None = None,
    odds: float | np.ndarray | None = None,
    calibrator_path: Path | str | None = None,
) -> np.ndarray:
    """統一 calibration エントリポイント — paper eval hook 用.

    V15 production は一切不変。
    本関数は shadow eval / paper eval でのみ呼び出す。

    Args:
        scores:           raw V15 probability scores (scalar / list / array)
        method:           calibration 方式 ('isotonic' / 'venue' / 'odds_bin' / 'none')
        venue:            venue 名 (method='venue' 時に必要)
        odds:             単勝オッズ (method='odds_bin' 時に必要)
        calibrator_path:  保存済み calibrator .pkl のパス (None = default path)

    Returns:
        calibrated scores (np.ndarray, shape = input shape)
        - method='none' または calibrator 未 fit の場合は raw scores を返す
        - bounds: [PROB_MIN, PROB_MAX]

    Raises:
        ValueError: 不明な method が指定された場合
    """
    scalar_input = np.ndim(scores) == 0
    arr = np.atleast_1d(np.asarray(scores, dtype=float))

    if method == "none":
        out = _clip_prob(arr)
        return float(out[0]) if scalar_input else out

    if method == "isotonic":
        path = Path(calibrator_path) if calibrator_path else V2_ISOTONIC_PKL
        if not path.exists():
            warnings.warn(
                f"IsotonicCalibrator not found at {path}. "
                "Returning raw scores. Run fit() and save() first.",
                UserWarning,
                stacklevel=2,
            )
            out = _clip_prob(arr)
            return float(out[0]) if scalar_input else out
        cal: IsotonicCalibrator = IsotonicCalibrator.load(path)
        out = cal.predict(arr)
        return float(out) if scalar_input else np.atleast_1d(out)

    elif method == "venue":
        if venue is None:
            raise ValueError("method='venue' では venue 引数が必要です")
        path = Path(calibrator_path) if calibrator_path else V2_VENUE_PKL
        if not path.exists():
            warnings.warn(
                f"VenueCalibrator not found at {path}. Returning raw scores.",
                UserWarning,
                stacklevel=2,
            )
            out = _clip_prob(arr)
            return float(out[0]) if scalar_input else out
        vcal: VenueCalibrator = VenueCalibrator.load(path)
        out = np.atleast_1d(vcal.predict(arr, venue))
        return float(out[0]) if scalar_input else out

    elif method == "odds_bin":
        if odds is None:
            raise ValueError("method='odds_bin' では odds 引数が必要です")
        path = Path(calibrator_path) if calibrator_path else V2_ODDSBIN_PKL
        if not path.exists():
            warnings.warn(
                f"OddsBinCalibrator not found at {path}. Returning raw scores.",
                UserWarning,
                stacklevel=2,
            )
            out = _clip_prob(arr)
            return float(out[0]) if scalar_input else out
        ocal: OddsBinCalibrator = OddsBinCalibrator.load(path)
        out = np.atleast_1d(ocal.predict(arr, odds))
        return float(out[0]) if scalar_input else out

    else:
        raise ValueError(
            f"未知の method: '{method}'. "
            "使用可能: 'isotonic' / 'venue' / 'odds_bin' / 'none'"
        )


# ============================================================
# 評価ユーティリティ (paper eval 用)
# ============================================================

def evaluate_calibration(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    calibrated: np.ndarray | None = None,
) -> dict[str, float]:
    """raw vs calibrated の Brier score / log loss を計算.

    Args:
        y_pred:     raw scores
        y_true:     binary labels
        calibrated: calibrated scores (None = raw のみ評価)

    Returns:
        dict with keys:
            raw_brier, raw_logloss,
            cal_brier, cal_logloss  (calibrated が None の場合は NaN)
            delta_brier, delta_logloss
    """
    _require_sklearn()
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    raw_brier = float(brier_score_loss(y_true, y_pred))
    raw_ll    = float(log_loss(y_true, y_pred))

    if calibrated is not None:
        cal_arr   = np.asarray(calibrated, dtype=float)
        cal_brier = float(brier_score_loss(y_true, cal_arr))
        cal_ll    = float(log_loss(y_true, np.clip(cal_arr, 1e-7, 1 - 1e-7)))
        delta_b   = cal_brier - raw_brier
        delta_ll  = cal_ll - raw_ll
    else:
        cal_brier = float("nan")
        cal_ll    = float("nan")
        delta_b   = float("nan")
        delta_ll  = float("nan")

    return {
        "raw_brier":    raw_brier,
        "raw_logloss":  raw_ll,
        "cal_brier":    cal_brier,
        "cal_logloss":  cal_ll,
        "delta_brier":  delta_b,
        "delta_logloss": delta_ll,
    }


def crossval_brier(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """IsotonicCalibrator の k-fold CV Brier score を計算.

    小サンプル (N=153) での過学習リスクを評価するため、
    hold-out ではなく k-fold CV を推奨。

    Args:
        y_pred:       raw scores
        y_true:       binary labels
        n_splits:     fold 数 (default 5)
        random_state: random seed

    Returns:
        dict with keys:
            cv_brier, raw_brier, delta_brier,
            cv_logloss, raw_logloss, delta_logloss,
            n_samples, n_splits
    """
    _require_sklearn()
    from sklearn.model_selection import StratifiedKFold

    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    iso_preds = np.zeros(len(y_true))

    for tr_idx, te_idx in skf.split(y_pred.reshape(-1, 1), y_true):
        cal = IsotonicCalibrator()
        cal.fit(y_pred[tr_idx], y_true[tr_idx])
        iso_preds[te_idx] = cal.predict(y_pred[te_idx])

    cv_brier  = float(brier_score_loss(y_true, iso_preds))
    cv_ll     = float(log_loss(y_true, np.clip(iso_preds, 1e-7, 1 - 1e-7)))
    raw_brier = float(brier_score_loss(y_true, y_pred))
    raw_ll    = float(log_loss(y_true, y_pred))

    return {
        "cv_brier":     cv_brier,
        "raw_brier":    raw_brier,
        "delta_brier":  cv_brier - raw_brier,
        "cv_logloss":   cv_ll,
        "raw_logloss":  raw_ll,
        "delta_logloss": cv_ll - raw_ll,
        "n_samples":    int(len(y_true)),
        "n_splits":     n_splits,
    }


# ============================================================
# CLI (paper eval 用)
# ============================================================

def _main() -> None:
    """CLI: paper eval pipeline を実行."""
    import argparse, json, sys

    ap = argparse.ArgumentParser(
        description="calibrator_overlay_v2 — paper eval pipeline (V15 production 不変)"
    )
    ap.add_argument("--eval-cumulative", action="store_true",
                    help="cumulative_results.csv で CV Brier 評価")
    ap.add_argument("--fit-isotonic", action="store_true",
                    help="全データで IsotonicCalibrator を fit + save")
    ap.add_argument("--fit-venue", action="store_true",
                    help="全データで VenueCalibrator を fit + save")
    ap.add_argument("--output-json", type=str, default=None,
                    help="結果を JSON ファイルに出力")
    args = ap.parse_args()

    cumulative = REPO / "data" / "cumulative_results.csv"
    if not cumulative.exists():
        print(f"[ERROR] {cumulative} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(cumulative, encoding="latin-1", low_memory=False)
    settled = df[df["status"] == "settled"].copy()
    settled["top1_finish_num"] = pd.to_numeric(settled["top1_finish"], errors="coerce")
    settled["y"]     = (settled["top1_finish_num"] <= 3).astype(float)
    settled["score"] = pd.to_numeric(settled["top1_score"], errors="coerce")
    valid = settled.dropna(subset=["y", "score"])

    print(f"[INFO] valid samples: {len(valid)}")

    results: dict[str, Any] = {}

    if args.eval_cumulative or args.fit_isotonic or args.fit_venue:
        # CV eval
        cv_res = crossval_brier(valid["score"].values, valid["y"].values)
        results["cv_isotonic"] = cv_res
        print("[CV Isotonic]")
        for k, v in cv_res.items():
            print(f"  {k}: {v}")

    if args.fit_isotonic:
        cal = IsotonicCalibrator()
        cal.fit(valid["score"].values, valid["y"].values)
        path = cal.save()
        print(f"[OK] IsotonicCalibrator saved -> {path}")
        results["isotonic_saved"] = str(path)

    if args.fit_venue:
        valid_copy = valid.copy()
        valid_copy["course_dec"] = valid_copy["course"].apply(
            lambda x: x.encode("latin-1").decode("utf-8", errors="replace")
            if isinstance(x, str) else x
        )
        vcal = VenueCalibrator()
        vcal.fit(valid_copy["score"].values, valid_copy["y"].values, valid_copy["course_dec"].values)
        path = vcal.save()
        print(f"[OK] VenueCalibrator saved -> {path}")
        print(f"     venue summary: {vcal.venue_summary()}")
        results["venue_saved"] = str(path)
        results["venue_summary"] = vcal.venue_summary()

    if args.output_json and results:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] results -> {out_path}")


if __name__ == "__main__":
    _main()
