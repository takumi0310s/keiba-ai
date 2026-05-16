# Phase D-2 — V21 architecture 設計 (純粋追加層)

date: 2026-05-16
session: Terminal D (Phase D)
status: 設計、 V15 production 完全不変保証

---

## 1. 設計原則 (絶対遵守)

1. **V15 inference は完全不変** — `tools/predict_core.py` を 1 byte も touch しない
2. V21 は V15 の **出力** を入力とする 追加 layer (純粋追加)
3. 動画 features 欠損 race では V15 score を そのまま返す (100% 互換)
4. V21 model file は **新規** (`models/v21/v21_meta_*.pkl.gz`)、 V15 file と混在しない
5. cron (schtasks) も 別系統 (DailyPredictV21 等、 後日 6/1+ で追加)

---

## 2. architecture diagram (ASCII)

```
┌────────────────────────────────────────────────────────────────────────┐
│ caller (daily_predict_v21.py / race_auto_notify_v21.py、 ★ 新規 ★)     │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
              ┌────────────────────────────────────────┐
              │ tools/v21/predict_core_v21.py          │
              │                                        │
              │   class PredictCoreV21:                │
              │     def predict_v21(race_id, horses):  │
              └────────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
     ┌──────────────────────┐              ┌────────────────────────┐
     │ V15 inference (★不変)│              │ Video features fetcher │
     │                      │              │                        │
     │ from predict_core    │              │ from predict_core_v21  │
     │ import (...)         │              │ (Phase 16、 既存)      │
     │                      │              │                        │
     │ load_models()        │              │ fetch_all_v21_video_   │
     │ build_features()     │              │  features(             │
     │ predict_race(df,...) │              │   paddock_video,       │
     │                      │              │   patrol_video,        │
     │ → df with スコア      │              │   chokyou_video)       │
     │                      │              │ → 30 features dict     │
     └──────────────────────┘              └────────────────────────┘
              │                                       │
              │ V15 final_score (1 dim)               │ video_30 (30 dim)
              │ per horse                             │ per horse
              │                                       │
              └────────────┬──────────────────────────┘
                           ▼
              ┌─────────────────────────────────────┐
              │ meta-model (V21)                    │
              │                                     │
              │ X_meta = [V15_score, video_30]      │
              │         shape: (n_horses, 31)       │
              │                                     │
              │ model: LGBMClassifier (binary)      │
              │ ↓                                   │
              │ V21_score = meta.predict_proba(X)   │
              │             [:, 1]                  │
              └─────────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────────────────┐
              │ V21 output                          │
              │   df['V21スコア'] = V21_score        │
              │   df['V21順位']  = rank(desc)        │
              └─────────────────────────────────────┘

  ★ fallback ★: video_30 が全 default (model 未学習 or 動画 不在) なら
     V21_score = V15_score (そのまま返却)、 V15 と完全互換
```

---

## 3. 補正方式 trade-off table

| 方式 | 数式 | pros | cons | 採用 |
|------|------|------|------|------|
| **a. linear ensemble** | `V21 = w·V15 + (1-w)·video_score` | 簡素、 weight 1 個 chuning | video_score を別 model で算出が必要 (二段階)、 表現力不足 | NG |
| **b. residual** | `V21 = V15 + Δ(video_30)` | V15 を centerline で安定、 video 効果が解釈可 | Δ を回帰 model で学習が必要、 動画欠損時 Δ=0 fallback 自然 | △ (案 c の特殊形) |
| **c. stacking (★ 採用 ★)** | `V21 = meta_LGB([V15_score, video_30])` | 31 dim 入力で meta model が non-linear 結合 学習可、 V15 + 動画 の interaction 捕捉、 動画欠損で V15 score をそのまま返す fallback 自然 | meta train data が必要 (V15 score + 動画 + 結果 で WF) | ★ **採用** ★ |
| d. full re-train | V15 features 150 + video 30 = 180 を end-to-end | 完全最適 | V15 不変保証 違反、 学習コスト 大、 ROI 既知の V15 を毀損 リスク | NG |

**選定理由 (c. stacking)**:
- V15 inference は完全不変 (import で結果のみ利用)
- meta LGB の学習は 動画 coverage が揃った時点で実施可能 (6/1+)
- 動画 features 全 default の race は V21 = V15 になる shortcut path で完全互換
- V20 でも同じ pattern を再利用可能 (V20 score を入力 1 dim に置換するだけ)

---

## 4. meta-model spec

| 項目 | 値 |
|---|---|
| algorithm | LightGBM (binary classification) |
| 入力 | (n_horses, 31) — `V15_final_score` (1) + video_30 (30) |
| 出力 | (n_horses,) — top3 probability |
| target | `finish <= 3` (V15 と同じ Pattern A target) |
| num_leaves | 31 (V15 より小、 overfitting 防止) |
| learning_rate | 0.03 |
| num_boost_round | 500 (early stop 50) |
| WF 学習 | 2023-2025 月単位 (動画 coverage に応じて拡張) |
| 出力 file | `models/v21/v21_meta_lgb.pkl.gz` |

---

## 5. ★ Fallback 仕様 ★ (V15 完全互換保証)

V21 inference 内部で 以下を判定:

```python
video_features_dict = fetch_all_v21_video_features(...)

# default 値 と全一致なら "動画なし" と判定
defaults = get_v21_video_defaults()
all_default = all(
    abs(video_features_dict[k] - defaults[k]) < 1e-9
    for k in defaults
)

if all_default or self.v21_meta_model is None:
    # V15 score をそのまま返す (V15 と完全同一 output)
    return v15_scores

# meta-model で stacking
X_meta = np.column_stack([v15_scores, video_matrix])
v21_scores = self.v21_meta_model.predict_proba(X_meta)[:, 1]
return v21_scores
```

→ 動画 features 1 個でも 実値 (non-default) があれば meta-model 経路、
   全 default なら V15 完全互換 path。
→ paper trade 初期 (6/1+) の動画 coverage 不足期間でも V15 同等 ROI 維持。

---

## 6. coverage 進化 plan

| 期 | 動画 coverage | V21 path | V15 path |
|---|---|---|---|
| 5/16 (今日) | 0% (skeleton) | V15 直結 fallback | 100% production |
| 6/1 | 重賞 R のみ ~3% | V21 paper shadow (記録のみ、 投票せず) | 100% production |
| 6/15 | 重賞 + 1 階級 ~15% | V21 paper trade (低額 trial) | 100% production |
| 6/30 | 全 R ~50%+ 目標 | V21 GO 判定 (本番候補) | 100% production |
| 7/1 (GO 時) | 全 R 80%+ | V21 production (V15 並行) | V15 並行 1 ヶ月 |
| 8/1 (順調時) | 全 R 90%+ | V21 production main | V15 archive 判定 |

---

## 7. 期待 AUC / ROI (★ 想定であって 実測ではない ★)

> 以下は 設計時の理論期待値、 実測は 6/1+ paper trade で確認。

| Metric | V15 (実測) | V21 (★ 期待 ★) |
|---|---|---|
| WF AUC | 0.8939 | 0.91-0.93 (動画 features 追加で +0.02-0.04 想定) |
| 実配当 ROI 全体 | **101.33%** (戦略⑦ applied 96.90% / ≤5/10) ※ 旧 119.2% / 140%+ は drift | 130%+ (paper trade で 確認必須) |
| winner_top1 率 | ~31% | +2-5pt 想定 |

**注**: V21 が V15 を超える 保証は **ない**。 6/30 GO/no-go 判定で 厳格に検証する。

---

## 8. risk 評価

| risk | 対策 |
|---|---|
| meta-model overfitting | num_leaves 31 / WF 学習 / early stop |
| 動画 coverage 不足 | fallback で V15 互換 path、 paper trade で curve 観察 |
| V15 inference 改変リスク | predict_core.py を import のみ、 1 byte も変更しない 確約 |
| 動画 features 質低下 (model 未学習) | default fill 経路で V15 互換 |
| 戦略⑦ 除外 R 対応 | sub-model 検討 (別 doc: phase_d_strategy_7_excluded_handling.md) |
| paper trade で V15 < V21 にならない | 6/30 GO/no-go で no-go 判定、 V15 単独継続 |
