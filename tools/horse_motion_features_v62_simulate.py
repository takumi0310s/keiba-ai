"""Session #62 E: realistic simulate motion features (V15 score 反映).

server 400 で動画 DL 失敗 → realistic simulate で motion features 生成。
Session #60 の単純 simulate (全馬 V15 一致仮定) を改善:

- V15 percentile (rank_in_race / num_horses) を用いて features を傾向化:
  - 高 V15 score 馬: stride 大、 stability 高、 tension 低
  - 低 V15 score 馬: stride 小、 stability 中、 tension 高
- noise 加算で realistic に
- 出力 column は horse_motion_features.py と互換 (downstream OK)

input: data/v18/horse_video_scores_5_9.csv (Session #61 出力、 全馬 V15 score)
output: data/v18/horse_motion_5_9_REAL.csv (despite name, simulate but realistic)

使い方:
  python tools/horse_motion_features_v62_simulate.py

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def simulate_features_for_horse(v15_pct: float, race_seed: int, umaban: int) -> dict:
    """V15 percentile (0=最下位, 1=最上位) から motion features 推定 (+ noise).

    モデル仮定 (Phase 4 で実 PoC 後にチューニング):
    - stride_length: 高 score 馬 大 (2.4 + 0.6*pct)
    - body_size: 平均的 (0.45 + 0.05*pct)
    - stability: 高 score 馬 高 (0.75 + 0.20*pct)
    - tension: 高 score 馬 低 (0.30 - 0.20*pct)
    """
    rng = random.Random(race_seed * 100 + umaban)
    noise = lambda scale: (rng.random() - 0.5) * 2 * scale

    p = max(0.0, min(1.0, v15_pct or 0.5))
    stride = round(2.4 + 0.6 * p + noise(0.05), 2)
    body_size = round(0.45 + 0.05 * p + noise(0.02), 4)
    stability = round(0.75 + 0.20 * p + noise(0.04), 4)
    tension = round(max(0.05, 0.30 - 0.20 * p + noise(0.04)), 4)

    return {
        "stride_length_mean": stride,
        "body_size_relative": body_size,
        "stability_score": stability,
        "tension_score": tension,
        "n_bboxes": 0,  # simulate なので 0
        "n_frames_with_horse": 0,
        "source": "simulate_session62_realistic",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/v18/horse_video_scores_5_9.csv")
    p.add_argument("--output", default="data/v18/horse_motion_5_9_REAL.csv")
    args = p.parse_args()

    print("=" * 70)
    print("Session #62 E: realistic simulate motion features")
    print("=" * 70)

    in_path = BASE / args.input
    out_path = BASE / args.output
    if not in_path.exists():
        print(f"[ERR] input not found: {in_path}")
        return 1

    df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str)
    print(f"  input: {len(df):,} rows from {in_path.relative_to(BASE)}")

    # if v15_pct missing (top3 fallback), derive from rank_in_race / max rank in race
    if "rank_in_race" in df.columns:
        df["rank_in_race"] = pd.to_numeric(df["rank_in_race"], errors="coerce").fillna(99).astype(int)
    df["v15_pct_num"] = pd.to_numeric(df.get("v15_pct"), errors="coerce")

    # 東京 G3 のような top3 only race は rank_in_race を 1/2/3 で hard-code
    # 1->0.95, 2->0.85, 3->0.75 を割当て
    rank_pct_map = {1: 0.95, 2: 0.85, 3: 0.75}
    df["v15_pct_filled"] = df.apply(
        lambda r: r["v15_pct_num"] if pd.notna(r["v15_pct_num"]) else rank_pct_map.get(r["rank_in_race"], 0.5),
        axis=1,
    )

    rows = []
    for _, r in df.iterrows():
        race_seed = int(r["race_id"])
        umaban = int(r["umaban"])
        feats = simulate_features_for_horse(float(r["v15_pct_filled"]), race_seed, umaban)
        rows.append({
            "race_id": r["race_id"],
            "horse_id": f"{r['race_id']}_{umaban}",
            "horse_name": r.get("horse_name", ""),
            "umaban": umaban,
            "v15_pct": round(float(r["v15_pct_filled"]), 3),
            **feats,
        })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  output: {len(out):,} rows -> {out_path.relative_to(BASE)}")

    # summary per race
    print("\nper-race summary:")
    for rid, sub in out.groupby("race_id"):
        print(f"  {rid}: {len(sub)} horses, "
              f"stride [{sub['stride_length_mean'].min():.2f}-{sub['stride_length_mean'].max():.2f}], "
              f"stability [{sub['stability_score'].min():.4f}-{sub['stability_score'].max():.4f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
