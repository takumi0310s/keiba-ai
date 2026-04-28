"""v16.1 単独評価スクリプト"""
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v16_and_am8_wf import build_or_load_cache, run_wf

print("=" * 70)
print(f"  v16.1 学習開始: {datetime.now()}")
print("=" * 70)

# STEP 1: キャッシュロード
print("\n[STEP 1] v15 キャッシュロード")
df, sire_map, bms_map, v15_features = build_or_load_cache()
print(f"  訓練データ: {len(df):,}行")
print(f"  v15 features: {len(v15_features)}個")

# STEP 2: v16 premium 計算
print("\n[STEP 2] v16 premium features 計算")
from features_v16_premium import compute_all_v16_premium_features, get_v16_premium_features, V16_PREMIUM_DEFAULTS
df = compute_all_v16_premium_features(df)
v16_feats = get_v16_premium_features()
print(f"  v16 premium features: {v16_feats}")

for f in v16_feats:
    if f in df.columns:
        df[f] = df[f].fillna(V16_PREMIUM_DEFAULTS.get(f, 0))

# STEP 3: training_eval_rank 確認
print("\n[STEP 3] training_eval_rank 確認")
if "training_eval_rank" not in df.columns:
    print("  ERROR: training_eval_rank が df にない")
    sys.exit(1)

nz = (df["training_eval_rank"] != 0).sum()
print(f"  非ゼロ: {nz:,} / {len(df):,} = {nz/len(df)*100:.1f}%")

# STEP 4: 特徴量構成
features = list(v15_features) + ["training_eval_rank"]
features = list(dict.fromkeys(features))
print(f"\n[STEP 4] 特徴量: {len(features)}個 (v15:{len(v15_features)} + 1)")

# STEP 5: WF 実行
print("\n[STEP 5] WF 実行")
t0 = time.time()
result = run_wf(df, features, "v161_training_eval_rank")
elapsed = (time.time() - t0) / 60

result["elapsed_min"] = elapsed
result["n_features"] = len(features)

mean_auc = result.get("mean_auc", 0)
baseline = 0.8856

print()
print("=" * 70)
print("  v16.1 結果")
print("=" * 70)
print(f"  mean WF AUC: {mean_auc:.6f}")
print(f"  baseline: {baseline}")
print(f"  diff: {(mean_auc - baseline) * 10000:+.0f}bp")
print(f"  経過: {elapsed:.1f}分")

if mean_auc > baseline:
    print("  判定: 採用")
else:
    print("  判定: 不採用")

# 保存
out = {
    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "baseline": baseline,
    "v161": result,
    "adopted": mean_auc > baseline,
}

with open("data/v161_wf_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

print(f"\n結果保存完了: data/v161_wf_results.json")
