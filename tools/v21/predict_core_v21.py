"""V21 inference orchestrator (Phase D, 2026-05-16).

★ V15 production 完全不変保証 ★
- tools/predict_core.py は import で再利用するだけ (1 byte も変更しない)
- 動画 30 features は 既存 tools/predict_core_v21.py (Phase 16) の skeleton fetcher を再利用
- V21 = stacking(V15 final_score, video_30) by meta-LGB
- 動画 features 全 default (model 未学習 or 動画 不在) なら V21_score = V15_score の fallback path

Status: skeleton (import test までで OK、 meta-model は 6/1+ で学習)
"""
from __future__ import annotations

import os
import sys
import gzip
import pickle
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# === path setup ===
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_THIS_DIR)
_BASE_DIR = os.path.dirname(_TOOLS_DIR)
for _p in (_BASE_DIR, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# === V15 production inference (★ read-only re-use ★) ===
# tools/predict_core.py は 触らず、 関数を import するだけ。
from predict_core import (  # type: ignore  # noqa: E402
    load_models as _load_v15_models,
    build_features as _build_v15_features,
    predict_race as _predict_v15_race,
)

# === Phase 16 動画 features fetcher (既存、 不変) ===
# tools/predict_core_v21.py (top-level) を そのまま import。
from predict_core_v21 import (  # type: ignore  # noqa: E402
    ALL_V21_VIDEO_FEATURES,
    fetch_all_v21_video_features as _fetch_video_features,
    get_v21_video_defaults as _get_video_defaults,
)


# =========================================================================
# Constants
# =========================================================================

V21_META_MODEL_PATH = os.path.join(_BASE_DIR, 'models', 'v21', 'v21_meta_lgb.pkl.gz')

V21_STACK_INPUT_DIM = 1 + len(ALL_V21_VIDEO_FEATURES)  # V15 score (1) + video (30) = 31


# =========================================================================
# Meta-model loader
# =========================================================================


def _load_v21_meta_model(path: str = V21_META_MODEL_PATH) -> Optional[Any]:
    """V21 meta-model (LGB binary classifier) を load.

    file 不在 (未学習 phase) は None を返し、 fallback path で V15 互換 動作。
    """
    if not os.path.isfile(path):
        return None
    try:
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception:
        return None
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model']
    return obj


def _is_all_default_video(video_dict: Dict[str, float]) -> bool:
    """動画 features が全て default 値 (= 実動画なし) かを判定."""
    defaults = _get_video_defaults()
    if not video_dict:
        return True
    for k, dv in defaults.items():
        v = video_dict.get(k, dv)
        if abs(float(v) - float(dv)) > 1e-9:
            return False
    return True


# =========================================================================
# Orchestrator
# =========================================================================


class PredictCoreV21:
    """V21 inference orchestrator.

    V15 inference は変更せず、 V15 出力 + 動画 30 features を meta-LGB で
    stacking する。 動画 features 全 default なら V15 score を そのまま返す。
    """

    def __init__(
        self,
        meta_model_path: str = V21_META_MODEL_PATH,
        v15_model_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        # V15 model data (predict_core.load_models の dict)
        self.v15_model_data: Dict[str, Any] = (
            v15_model_data if v15_model_data is not None else _load_v15_models()
        )
        # V21 meta-model (None なら fallback)
        self.v21_meta_model = _load_v21_meta_model(meta_model_path)
        self.meta_model_path = meta_model_path

    # ---------------------------------------------------------------------
    # V15 path (★ 既存 logic 100% 不変、 re-use only ★)
    # ---------------------------------------------------------------------

    def predict_v15(
        self,
        horses: List[Dict[str, Any]],
        race_info: Dict[str, Any],
        race_id: Optional[str] = None,
        odds_dict: Optional[Dict[int, float]] = None,
        jra_track_info: Optional[Dict[str, Any]] = None,
        weather_info: Optional[Dict[str, Any]] = None,
        odds_available: bool = False,
    ):
        """V15 inference を そのまま 呼ぶ (★ 不変保証 ★).

        Returns: predict_race の出力 DataFrame (`スコア` / `AI順位` を含む)
        """
        df = _build_v15_features(
            horses,
            race_info,
            self.v15_model_data,
            race_id=race_id,
            odds_dict=odds_dict,
            jra_track_info=jra_track_info,
            weather_info=weather_info,
        )
        df = _predict_v15_race(
            df,
            self.v15_model_data,
            odds_available=odds_available,
            race_info=race_info,
        )
        return df

    # ---------------------------------------------------------------------
    # Video features gather
    # ---------------------------------------------------------------------

    def _gather_video_features(
        self,
        race_id: Optional[str],
        horses: Sequence[Dict[str, Any]],
        paddock_video_map: Optional[Dict[int, str]] = None,
        patrol_video_path: Optional[str] = None,
        chokyou_video_map: Optional[Dict[int, str]] = None,
    ) -> List[Dict[str, float]]:
        """Race 内 各馬の 動画 30 features を返す.

        - paddock / chokyou: 馬個別 動画 → 馬番 -> video path map
        - patrol: race 全体 1 本 (馬番ごとに同じ video を渡し、 features は同一)

        Phase D skeleton では default fill が返る (実 model は 7/1+ で fine-tune)。
        """
        paddock_video_map = paddock_video_map or {}
        chokyou_video_map = chokyou_video_map or {}

        out: List[Dict[str, float]] = []
        for h in horses:
            umaban = int(h.get('馬番', h.get('horse_num', 0)) or 0)
            paddock_v = paddock_video_map.get(umaban)
            chokyou_v = chokyou_video_map.get(umaban)
            feats = _fetch_video_features(
                paddock_video=paddock_v,
                patrol_video=patrol_video_path,
                chokyou_video=chokyou_v,
            )
            out.append(feats)
        return out

    # ---------------------------------------------------------------------
    # V21 path = V15 + video stacking
    # ---------------------------------------------------------------------

    def predict_v21(
        self,
        horses: List[Dict[str, Any]],
        race_info: Dict[str, Any],
        race_id: Optional[str] = None,
        odds_dict: Optional[Dict[int, float]] = None,
        jra_track_info: Optional[Dict[str, Any]] = None,
        weather_info: Optional[Dict[str, Any]] = None,
        odds_available: bool = False,
        paddock_video_map: Optional[Dict[int, str]] = None,
        patrol_video_path: Optional[str] = None,
        chokyou_video_map: Optional[Dict[int, str]] = None,
    ):
        """V21 inference: V15 → 動画 features stacking → V21 score.

        Fallback: 動画 features が全 default なら V15 score を そのまま返す
        (V15 完全互換 動作)。
        """
        # 1. V15 inference (★ 既存 logic 不変 ★)
        df = self.predict_v15(
            horses,
            race_info,
            race_id=race_id,
            odds_dict=odds_dict,
            jra_track_info=jra_track_info,
            weather_info=weather_info,
            odds_available=odds_available,
        )

        # 2. video features gather (馬番順 list-of-dict)
        # df は AI順位 で sort 済 → 馬番順に並べ直して video map と整合させる
        df_by_horse = df.sort_values('馬番') if '馬番' in df.columns else df.reset_index(drop=True)
        horse_records = df_by_horse.to_dict('records')
        video_feats_list = self._gather_video_features(
            race_id=race_id,
            horses=horse_records,
            paddock_video_map=paddock_video_map,
            patrol_video_path=patrol_video_path,
            chokyou_video_map=chokyou_video_map,
        )

        # 3. fallback 判定
        any_real_video = any(not _is_all_default_video(d) for d in video_feats_list)
        if not any_real_video or self.v21_meta_model is None:
            # V15 完全互換 path
            df['V21スコア'] = df['スコア']
            df['V21順位'] = df['AI順位']
            df['V21mode'] = 'fallback_v15'
            return df

        # 4. stacking 入力 構築: (n_horses, 31)
        video_matrix = np.array(
            [[d[k] for k in ALL_V21_VIDEO_FEATURES] for d in video_feats_list],
            dtype=np.float32,
        )
        v15_scores = df_by_horse['スコア'].values.astype(np.float32).reshape(-1, 1)
        X_meta = np.concatenate([v15_scores, video_matrix], axis=1)
        assert X_meta.shape[1] == V21_STACK_INPUT_DIM, (
            f'meta input dim mismatch: {X_meta.shape[1]} vs {V21_STACK_INPUT_DIM}'
        )

        # 5. meta-model predict
        try:
            if hasattr(self.v21_meta_model, 'predict_proba'):
                v21_scores = self.v21_meta_model.predict_proba(X_meta)[:, 1]
            else:
                v21_scores = self.v21_meta_model.predict(X_meta)
        except Exception:
            # meta predict 失敗時は V15 互換 fallback
            df['V21スコア'] = df['スコア']
            df['V21順位'] = df['AI順位']
            df['V21mode'] = 'fallback_meta_error'
            return df

        # 6. df に書き戻し (馬番 align)
        score_map = dict(zip(df_by_horse['馬番'].tolist(), v21_scores.tolist()))
        df['V21スコア'] = df['馬番'].map(score_map).astype(float)
        df['V21順位'] = df['V21スコア'].rank(ascending=False).astype(int)
        df['V21mode'] = 'stacked'
        return df


# =========================================================================
# Self-test (compile-check + fallback verify)
# =========================================================================


def _self_test() -> None:
    """import + V21 stack dim + default fallback の verify."""
    print('[predict_core_v21/orchestrator] self-test start')

    # V15 features defaults と 動画 fetcher を 触る程度の sanity
    from predict_core_v21 import get_v21_video_feature_names, get_v21_video_defaults

    names = get_v21_video_feature_names()
    defaults = get_v21_video_defaults()
    assert len(names) == 30, f'video features should be 30, got {len(names)}'
    assert set(names) == set(defaults.keys()), 'feature/default keys mismatch'

    # stacking dim check (V15 score 1 + video 30 = 31)
    assert V21_STACK_INPUT_DIM == 31, f'stack input dim must be 31, got {V21_STACK_INPUT_DIM}'

    # default-video → all_default detection
    dummy_default = dict(defaults)
    assert _is_all_default_video(dummy_default) is True
    dummy_real = dict(defaults)
    dummy_real['video_paddock_body_score'] = 90.0
    assert _is_all_default_video(dummy_real) is False

    # meta-model loader: 不在 file で None
    assert _load_v21_meta_model('/nonexistent/path.pkl.gz') is None

    print(f'[predict_core_v21/orchestrator] OK ({V21_STACK_INPUT_DIM} dim stack input)')


if __name__ == '__main__':
    _self_test()
