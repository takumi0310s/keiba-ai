"""
JRDB TYB (直前情報) retrospective evaluation

目的: 既存 data/jrdb_tyb.csv (548K rows、 padock_idx / weight_diff / kehai_code 等) を
      V15 production の score と組み合わせ、 TYB features が top3 予測に value 提供するか検証。

V15 production 完全不変。 評価のみ、 model retrain なし、 production 影響 0。

source:
- data/daily_predictions/YYYYMMDD.csv (V15 top1_score、 top1_num)
- data/daily_results/YYYYMMDD.csv (top1_finish — V15 top1 が何着か)
- data/jrdb_tyb.csv (548K rows、 直前 features)

evaluation:
1. inner join (race_id × umaban=top1_num) で TYB の top1 horse features 取得
2. label = top1_finish <= 3 (binary、 top3 ヒット)
3. TYB features 単独 / V15 score 単独 / 結合の AUC + correlation 比較
4. feature importance ranking (LGB 簡易 fit)

output:
- data/v21/jrdb_tyb_eval_report.json (数値結果)
- data/v21/jrdb_tyb_eval_report.md (honest 報告)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
DAILY_PRED_DIR = REPO / "data" / "daily_predictions"
DAILY_RES_DIR = REPO / "data" / "daily_results"
TYB_CSV = REPO / "data" / "jrdb_tyb.csv"
OUT_JSON = REPO / "data" / "v21" / "jrdb_tyb_eval_report.json"
OUT_MD = REPO / "data" / "v21" / "jrdb_tyb_eval_report.md"

# TYB の評価対象 features (数値 / カテゴリ enc)
TYB_NUMERIC = ['idm', 'jockey_idx', 'info_idx', 'odds_idx', 'padock_idx', 'sogo_idx',
               'bagu_change', 'ashimoto', 'cancel_flag',
               'tansho_odds', 'fukusho_odds', 'horse_weight', 'weight_diff',
               'padock_mark', 'kehai_code', 'baba_code', 'weather_code']


def load_predictions_with_results() -> pd.DataFrame:
    """daily_predictions × daily_results inner join、 V15 top1_num + score + label."""
    pred_frames = []
    for csv in sorted(DAILY_PRED_DIR.glob("2026*.csv")):
        if "nar" in csv.name or "prerace" in csv.name:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        df["race_id"] = df["race_id"].astype(str)
        df = df[["race_id", "top1_num", "top1_score"]].copy()
        df["src_date"] = csv.stem
        pred_frames.append(df)
    pred = pd.concat(pred_frames, ignore_index=True)
    print(f"[INFO] daily_predictions: {len(pred)} races")

    res_frames = []
    for csv in sorted(DAILY_RES_DIR.glob("2026*.csv")):
        if "nar" in csv.name or "payouts" in csv.name:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "top1_finish" not in df.columns:
            continue
        df["race_id"] = df["race_id"].astype(str)
        cols = ["race_id", "top1_finish"] + (["status"] if "status" in df.columns else [])
        df = df[cols].copy()
        if "status" not in df.columns:
            df["status"] = "settled"
        res_frames.append(df)
    res = pd.concat(res_frames, ignore_index=True)
    res = res[res["status"] == "settled"].copy()
    res["top1_finish_num"] = pd.to_numeric(res["top1_finish"], errors="coerce")
    res["label"] = (res["top1_finish_num"] <= 3).astype(int)

    merged = pred.merge(res, on="race_id", how="inner")
    merged["top1_score"] = pd.to_numeric(merged["top1_score"], errors="coerce")
    merged["top1_num"] = pd.to_numeric(merged["top1_num"], errors="coerce")
    merged = merged.dropna(subset=["top1_score", "top1_num", "top1_finish_num"])
    print(f"[INFO] joined daily_pred × daily_res: {len(merged)} clean rows")
    return merged


def merge_with_tyb(pred_df: pd.DataFrame) -> pd.DataFrame:
    """pred_df の race_id × top1_num で jrdb_tyb の top1 horse の TYB features を取得."""
    tyb = pd.read_csv(TYB_CSV, dtype={"race_id": str})
    tyb["umaban"] = pd.to_numeric(tyb["umaban"], errors="coerce")
    pred_df["top1_num"] = pd.to_numeric(pred_df["top1_num"], errors="coerce")

    available = [c for c in TYB_NUMERIC if c in tyb.columns]
    print(f"[INFO] TYB available features: {len(available)} / {len(TYB_NUMERIC)}")

    tyb_subset = tyb[["race_id", "umaban"] + available].copy()
    merged = pred_df.merge(
        tyb_subset,
        left_on=["race_id", "top1_num"],
        right_on=["race_id", "umaban"],
        how="inner",
    )
    print(f"[INFO] joined pred × tyb (top1 only): {len(merged)} rows")
    return merged, available


def evaluate(merged: pd.DataFrame, tyb_cols: list[str]) -> dict:
    """V15 score 単独 / TYB 単独 / V15+TYB 結合 の AUC 比較."""
    n = len(merged)
    y = merged["label"].values.astype(int)
    v15_score = merged["top1_score"].values.astype(float)

    result = {
        "n_samples": int(n),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "pos_rate": float(y.mean()),
    }

    # 1. V15 score 単独 AUC
    try:
        auc_v15 = roc_auc_score(y, v15_score)
        result["auc_v15_only"] = float(auc_v15)
    except Exception:
        result["auc_v15_only"] = None

    # 2. 各 TYB feature 単独 correlation + AUC
    feat_stats = {}
    for col in tyb_cols:
        x = pd.to_numeric(merged[col], errors="coerce")
        mask = x.notna()
        if mask.sum() < 30:
            feat_stats[col] = {"n": int(mask.sum()), "corr": None, "auc": None, "auc_neg": None}
            continue
        x_vals = x[mask].values.astype(float)
        y_vals = y[mask.values]
        try:
            corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
        except Exception:
            corr = None
        try:
            auc = float(roc_auc_score(y_vals, x_vals))
            auc_neg = float(roc_auc_score(y_vals, -x_vals))
            best_auc = max(auc, auc_neg)
        except Exception:
            auc = None
            best_auc = None
        feat_stats[col] = {
            "n": int(mask.sum()),
            "corr": corr,
            "auc_direct": auc,
            "auc_best": best_auc,  # 反転考慮
        }
    result["tyb_feature_stats"] = feat_stats

    # 3. V15 + TYB 結合 logistic regression AUC (CV、 NaN は median fill)
    X_cols = ["top1_score"] + tyb_cols
    X_df = merged[X_cols].apply(pd.to_numeric, errors="coerce")
    medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians)
    if X_df.isna().any().any():
        X_df = X_df.fillna(0.0)
    X = X_df.values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    try:
        model = LogisticRegression(max_iter=2000, C=1.0)
        # 5-fold CV
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, te in kf.split(Xs):
            m = LogisticRegression(max_iter=2000, C=1.0)
            m.fit(Xs[tr], y[tr])
            p = m.predict_proba(Xs[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        result["auc_v15_plus_tyb_5cv_mean"] = float(np.mean(aucs))
        result["auc_v15_plus_tyb_5cv_std"] = float(np.std(aucs))

        # model.fit on full for coefficient inspection
        model.fit(Xs, y)
        coef_dict = {col: float(c) for col, c in zip(X_cols, model.coef_[0])}
        result["lr_coefficients"] = coef_dict
    except Exception as e:
        result["auc_v15_plus_tyb_5cv_mean"] = None
        result["lr_error"] = str(e)

    # 4. TYB のみ (V15 抜き) AUC
    try:
        X_tyb = merged[tyb_cols].apply(pd.to_numeric, errors="coerce")
        X_tyb = X_tyb.fillna(X_tyb.median(numeric_only=True)).fillna(0.0)
        Xs_tyb = StandardScaler().fit_transform(X_tyb.values.astype(float))
        m2 = LogisticRegression(max_iter=2000)
        aucs2 = []
        for tr, te in kf.split(Xs_tyb):
            mm = LogisticRegression(max_iter=2000)
            mm.fit(Xs_tyb[tr], y[tr])
            p = mm.predict_proba(Xs_tyb[te])[:, 1]
            aucs2.append(roc_auc_score(y[te], p))
        result["auc_tyb_only_5cv_mean"] = float(np.mean(aucs2))
    except Exception as e:
        result["auc_tyb_only_5cv_mean"] = None

    return result


def write_report_md(result: dict) -> str:
    """honest md report 出力."""
    lines = []
    lines.append("# JRDB TYB (直前情報) retrospective evaluation report")
    lines.append("")
    lines.append(f"**実施日**: 2026-05-16")
    lines.append(f"**source**: `tools/v21/jrdb_tyb_evaluate.py`")
    lines.append(f"**V15 production 影響**: ★ 0% (評価のみ、 model 不変) ★")
    lines.append("")
    lines.append("## 1. data 概要")
    lines.append("")
    lines.append(f"- n_samples: **{result['n_samples']}**")
    lines.append(f"- pos / neg: {result['n_pos']} / {result['n_neg']}")
    lines.append(f"- pos_rate: {result['pos_rate']:.3f}")
    lines.append("")
    lines.append("## 2. AUC 比較")
    lines.append("")
    lines.append("| model | AUC |")
    lines.append("|-------|-----|")
    v15 = result.get('auc_v15_only')
    tyb = result.get('auc_tyb_only_5cv_mean')
    both = result.get('auc_v15_plus_tyb_5cv_mean')
    lines.append(f"| V15 top1_score 単独 | {v15:.4f}" if v15 else "| V15 top1_score 単独 | N/A")
    lines.append(f"| TYB features 単独 (5CV) | {tyb:.4f}" if tyb else "| TYB features 単独 (5CV) | N/A")
    lines.append(f"| V15 + TYB 結合 (5CV) | {both:.4f}" if both else "| V15 + TYB 結合 (5CV) | N/A")
    lines.append("")
    if v15 is not None and both is not None:
        delta = both - v15
        sign = "+" if delta >= 0 else ""
        lines.append(f"**V15 + TYB delta: {sign}{delta:.4f}** ({'採用候補' if delta >= 0.005 else '効果薄' if delta >= 0 else 'マイナス'})")
        lines.append("")

    lines.append("## 3. TYB feature 単独 correlation + AUC (top3 hit)")
    lines.append("")
    lines.append("| feature | n | corr | AUC best (反転考慮) |")
    lines.append("|---------|---|------|---------------------|")
    stats = result.get("tyb_feature_stats", {})
    sorted_feats = sorted(stats.items(), key=lambda kv: (kv[1].get("auc_best") or 0), reverse=True)
    for col, st in sorted_feats:
        corr = st.get("corr")
        auc = st.get("auc_best")
        corr_s = f"{corr:.3f}" if corr is not None else "N/A"
        auc_s = f"{auc:.3f}" if auc is not None else "N/A"
        lines.append(f"| {col} | {st['n']} | {corr_s} | {auc_s} |")
    lines.append("")

    lr_coef = result.get("lr_coefficients", {})
    if lr_coef:
        lines.append("## 4. V15 + TYB logistic regression coefficients (standardized)")
        lines.append("")
        lines.append("| feature | coef |")
        lines.append("|---------|-----:|")
        for k, v in sorted(lr_coef.items(), key=lambda kv: abs(kv[1]), reverse=True):
            lines.append(f"| {k} | {v:+.3f} |")
        lines.append("")

    lines.append("## 5. honest 結論")
    lines.append("")
    if v15 is not None and both is not None:
        delta = both - v15
        if delta >= 0.01:
            lines.append(f"★ **採用候補** ★ — V15 + TYB で AUC +{delta:.4f}、 V21 retrain 候補に追加。")
        elif delta >= 0.005:
            lines.append(f"△ **微改善** — V15 + TYB で AUC +{delta:.4f}、 paper shadow で 検証推奨。")
        elif delta >= 0:
            lines.append(f"× **効果薄** — V15 + TYB で AUC +{delta:.4f}、 TYB は V15 で 既に内包されている 可能性。")
        else:
            lines.append(f"× **マイナス** — V15 + TYB で AUC {delta:.4f}、 noise の可能性、 採用しない。")
    lines.append("")
    lines.append("## 6. 次 action 候補")
    lines.append("")
    lines.append("- TYB は **5/9 から fetch 停止** (extract dir 確認)。 fetch 復旧必要")
    lines.append("- TYB を merge した V15 cache を 構築 → V21/V22 retrain 候補")
    lines.append("- 直前 (-15 min) TYB 更新版 取得の schtask は別途検討")

    return "\n".join(lines)


def main():
    pred = load_predictions_with_results()
    if len(pred) < 30:
        print(f"[ERROR] pred df too small ({len(pred)})、 abort")
        return 1

    merged, tyb_cols = merge_with_tyb(pred)
    if len(merged) < 30:
        print(f"[ERROR] merged df too small ({len(merged)})、 abort")
        return 1

    result = evaluate(merged, tyb_cols)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] report: {OUT_JSON}")

    md = write_report_md(result)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[INFO] md: {OUT_MD}")

    v15 = result.get('auc_v15_only')
    both = result.get('auc_v15_plus_tyb_5cv_mean')
    if v15 is not None and both is not None:
        delta = both - v15
        print(f"\nSummary: V15 {v15:.4f} → V15+TYB {both:.4f} (delta {delta:+.4f}, n={result['n_samples']})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
