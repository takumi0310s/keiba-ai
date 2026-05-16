"""TYB calibrator P0-3 leak 監査 analysis.

V15 production 不変、 read-only analysis のみ。
出力: docs/TYB_LEAK_AUDIT_2026_05_16.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[2]
DAILY_PRED_DIR = REPO / "data" / "daily_predictions"
DAILY_RES_DIR = REPO / "data" / "daily_results"
TYB_CSV = REPO / "data" / "jrdb_tyb.csv"

FEATURE_COLS = ['top1_score',
                'idm', 'jockey_idx', 'info_idx', 'odds_idx', 'padock_idx', 'sogo_idx',
                'bagu_change', 'ashimoto', 'cancel_flag',
                'tansho_odds', 'fukusho_odds', 'horse_weight', 'weight_diff',
                'padock_mark', 'kehai_code', 'baba_code']


def load_full() -> pd.DataFrame:
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
        pred_frames.append(df)
    pred = pd.concat(pred_frames, ignore_index=True)

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

    pr = pred.merge(res, on="race_id", how="inner")
    pr["top1_score"] = pd.to_numeric(pr["top1_score"], errors="coerce")
    pr["top1_num"] = pd.to_numeric(pr["top1_num"], errors="coerce")
    pr = pr.dropna(subset=["top1_score", "top1_num", "top1_finish_num"])

    tyb = pd.read_csv(TYB_CSV, dtype={"race_id": str})
    tyb["umaban"] = pd.to_numeric(tyb["umaban"], errors="coerce")

    full = pr.merge(tyb, left_on=["race_id", "top1_num"],
                    right_on=["race_id", "umaban"], how="inner")
    return full


def cv_auc(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in kf.split(X):
        m = LogisticRegression(max_iter=2000)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return np.mean(aucs), np.std(aucs)


def main():
    df = load_full()
    print(f"[INFO] n={len(df)}, pos={df['label'].sum()}, neg={len(df)-df['label'].sum()}")
    print(f"[INFO] pos_rate: {df['label'].mean():.4f}")

    y = df["label"].values.astype(int)
    X_df = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians).fillna(0.0)
    X = X_df.values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 1. baseline V15-only AUC analysis
    v15_only = Xs[:, 0:1]
    auc_v15, std_v15 = cv_auc(v15_only, y)
    # Direct AUC (no LR fit, just raw score)
    auc_direct_v15 = roc_auc_score(y, df["top1_score"].values)
    print(f"\n[BASELINE] V15 top1_score direct AUC: {auc_direct_v15:.4f}")
    print(f"[BASELINE] V15 top1_score LR 5CV AUC: {auc_v15:.4f} (std {std_v15:.4f})")

    # 2. Full model
    auc_full, std_full = cv_auc(Xs, y)
    print(f"[FULL]    V15+TYB LR 5CV AUC: {auc_full:.4f}")

    # 3. Train AUC (overfit check)
    m_full = LogisticRegression(max_iter=2000)
    m_full.fit(Xs, y)
    train_auc = roc_auc_score(y, m_full.predict_proba(Xs)[:, 1])
    print(f"[FULL]    train AUC: {train_auc:.4f} (gap = {train_auc - auc_full:.4f})")

    # 4. Per-feature: leave-one-out CV AUC
    print(f"\n=== leave-one-out (drop ONE feature, 5CV AUC) ===")
    loo_results = {}
    for i, fc in enumerate(FEATURE_COLS):
        mask = np.array([j for j in range(len(FEATURE_COLS)) if j != i])
        X_loo = Xs[:, mask]
        auc, std = cv_auc(X_loo, y)
        delta = auc - auc_full
        loo_results[fc] = {"auc_drop": float(auc), "delta_from_full": float(delta)}
        print(f"  drop {fc:18s}: AUC={auc:.4f} (delta={delta:+.4f})")

    # 5. Per-feature: add ONE to V15-only baseline (test marginal value)
    print(f"\n=== ADD-ONE: V15 + 1 TYB feature 5CV AUC ===")
    add_results = {}
    for i, fc in enumerate(FEATURE_COLS):
        if fc == 'top1_score':
            continue
        X_add = Xs[:, [0, i]]
        auc, std = cv_auc(X_add, y)
        delta = auc - auc_v15
        add_results[fc] = {"auc_add": float(auc), "delta_from_v15only": float(delta)}
        print(f"  +{fc:18s}: AUC={auc:.4f} (delta vs V15only={delta:+.4f})")

    # 6. tansho_odds alone (direct AUC, no LR)
    print(f"\n=== single-feature direct AUC (TYB-only, no V15) ===")
    single_results = {}
    for i, fc in enumerate(FEATURE_COLS):
        if fc == 'top1_score':
            continue
        x_raw = pd.to_numeric(df[fc], errors='coerce').fillna(X_df[fc].median()).values
        try:
            auc_p = roc_auc_score(y, x_raw)
            auc_n = roc_auc_score(y, -x_raw)
            best = max(auc_p, auc_n)
        except Exception:
            best = None
        single_results[fc] = {"auc_best": float(best) if best else None}
        print(f"  {fc:18s}: AUC_best={best:.4f}" if best else f"  {fc:18s}: N/A")

    # 7. Tansho_odds vs jra_payouts confirmed odds
    print(f"\n=== TYB tansho_odds vs jra_payouts confirmed odds (winners) ===")
    jp = pd.read_csv(REPO / 'data' / 'jra_payouts.csv', dtype=str, encoding='utf-8')
    course_to_code = {'札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
                      '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10'}
    jp['basho'] = jp['course'].map(course_to_code)
    jp['race_id'] = (jp['race_date'].str[:4] + jp['basho'].fillna('00') +
                     jp['kai'].str.zfill(2) + jp['nichi'].str.zfill(2) + jp['race_num'].str.zfill(2))
    jp = jp[jp['basho'].notna() & jp['tansho_nums'].notna() & jp['tansho_payout'].notna()].copy()
    jp['win_uma'] = pd.to_numeric(jp['tansho_nums'].str.split('/').str[0], errors='coerce')
    jp['confirmed_odds'] = pd.to_numeric(jp['tansho_payout'], errors='coerce') / 100.0
    tyb = pd.read_csv(REPO / 'data' / 'jrdb_tyb.csv', dtype={'race_id': str},
                      usecols=['race_id', 'umaban', 'tansho_odds'])
    tyb['umaban'] = pd.to_numeric(tyb['umaban'], errors='coerce')
    mt = jp.merge(tyb, left_on=['race_id', 'win_uma'], right_on=['race_id', 'umaban'])
    mt = mt.dropna(subset=['tansho_odds', 'confirmed_odds'])
    corr_tyb_conf = mt[['tansho_odds', 'confirmed_odds']].corr().iloc[0, 1]
    log_corr = np.corrcoef(np.log(mt['tansho_odds'].clip(0.1)), np.log(mt['confirmed_odds'].clip(0.1)))[0, 1]
    delta_v = mt['tansho_odds'] - mt['confirmed_odds']
    print(f"  matched winners: {len(mt)}")
    print(f"  raw correlation: {corr_tyb_conf:.4f}")
    print(f"  log correlation: {log_corr:.4f}")
    print(f"  delta (TYB - confirmed) median: {delta_v.median():+.3f}")
    print(f"  delta std: {delta_v.std():.3f}")
    print(f"  exact match (|d|<0.05): {(delta_v.abs() < 0.05).mean():.4f}")
    print(f"  close (|d|<0.5): {(delta_v.abs() < 0.5).mean():.4f}")
    print(f"  close (|d|<2.0): {(delta_v.abs() < 2.0).mean():.4f}")

    # 8. odds_base (morning) vs TYB vs confirmed comparison
    print(f"\n=== odds_base (08:00 morning) vs TYB tansho_odds ===")
    ob_frames = []
    for csv in sorted((REPO / 'data').glob('odds_base_*.csv')):
        try:
            d = pd.read_csv(csv)
            d['race_id'] = d['race_id'].astype(str)
            ob_frames.append(d)
        except Exception:
            pass
    if ob_frames:
        ob = pd.concat(ob_frames, ignore_index=True)
        ob['horse_num'] = pd.to_numeric(ob['horse_num'], errors='coerce')
        ob['odds'] = pd.to_numeric(ob['odds'], errors='coerce')
        mo = ob.merge(tyb, left_on=['race_id', 'horse_num'], right_on=['race_id', 'umaban']).dropna(subset=['odds', 'tansho_odds'])
        log_ob_tyb = np.corrcoef(np.log(mo['odds'].clip(0.1)), np.log(mo['tansho_odds'].clip(0.1)))[0, 1]
        print(f"  matched all-horses: {len(mo)}, races: {mo['race_id'].nunique()}")
        print(f"  log corr odds_base vs TYB: {log_ob_tyb:.4f}")
        d2 = mo['tansho_odds'] - mo['odds']
        print(f"  delta median: {d2.median():+.3f}, std: {d2.std():.3f}")

    out = {
        "n_samples": int(len(df)),
        "pos_rate": float(df["label"].mean()),
        "auc_v15_direct": float(auc_direct_v15),
        "auc_v15_lr_5cv": float(auc_v15),
        "auc_full_5cv": float(auc_full),
        "auc_full_train": float(train_auc),
        "train_cv_gap": float(train_auc - auc_full),
        "loo_results": loo_results,
        "add_results": add_results,
        "single_results": single_results,
        "tansho_audit": {
            "matched": int(len(mt)),
            "corr_raw": float(corr_tyb_conf),
            "corr_log": float(log_corr),
            "delta_median": float(delta_v.median()),
            "delta_std": float(delta_v.std()),
            "exact_match_rate": float((delta_v.abs() < 0.05).mean()),
            "within_0_5_yen": float((delta_v.abs() < 0.5).mean()),
            "within_2_0_yen": float((delta_v.abs() < 2.0).mean()),
        },
    }
    out_path = REPO / "data" / "v21" / "tyb_leak_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
