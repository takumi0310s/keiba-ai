# Phase 19 完了: V18 真値版 学習 script + dataset 準備 (5/16 user CLI ready)

date: 2026-05-10 22:00
session: Phase 19 (Opus 4.7、 caveman mode、 AI session 内完結)

---

## 提供 script

### tools/train_v18_truevalue.py
V18 真値版 model 学習 (LGB + XGB、 WF 5 fold)

```bash
# 構造 + constant 検出 (即時、 ~5 sec)
python tools/train_v18_truevalue.py --check-only

# Full WF train (CPU、 ~30-60 min)
python tools/train_v18_truevalue.py

# GPU 利用 (RTX 4070 Ti SUPER、 ~15-30 min)
python tools/train_v18_truevalue.py --gpu

# AUC 改善時 model 保存
python tools/train_v18_truevalue.py --gpu --save-model
```

**features**:
- V15 base: 150 features (model file から読込)
- Phase 11/12/13 候補: 15 features (gaika_*, odds_change_*_v18, jockey_*_winrate, return/paddock/saddle)
- 計 165 features

**Phase 15 教訓 反映**:
- check-only mode で constant feature 自動検出
- 大半が constant なら 警告 + 学習 skip 推奨
- 5/12-5/13 真値化前に train を起動しても誤った AUC 改善期待を回避

### tools/v18_wf_evaluation.py
V15 vs V18 比較 + 5/10 全 35 R 仮想評価

```bash
# 全モード
python tools/v18_wf_evaluation.py

# 5/10 評価のみ
python tools/v18_wf_evaluation.py --mode 5_10

# score 帯別 hit 率のみ
python tools/v18_wf_evaluation.py --mode band
```

---

## 動作確認 (5/10 22:00)

### check-only 結果
```
V15 baseline: 150 features (AUC 0.8939)
[load] _v15_train_df_cache.pkl (917 MB)
[load] V15 cached df: (527280, 233)
[V18 candidate] 15/15 columns added (scaffold defaults)

=== Phase 11/12/13 候補 features constant 検出 ===
  constant (variance ~0): 15 -> 学習効果なし
  near_constant: 0
  real signal: 0

⚠ ★ Phase 15 教訓 警告 ★
  V18 候補 features の大半が constant (scaffold default のみ)。
  本 train で V15 比 AUC 改善は期待できません。
  → 5/12-5/13 で 各 feature の 真値化計算 logic を実装してから再実行を推奨。
```

### v18_wf_evaluation 結果 (5/10)
全 34 R 仮想:
- trio_hit: 11 (32.4%)
- ROI 113.8%

V15 score 帯別:
| band | N | hit | hit% | ROI% |
|---|---|---|---|---|
| ≥0.7 | 6 | 2 | 33.3% | 83.1% |
| 0.6-0.7 | 10 | 5 | **50.0%** | **242.1%** ★ |
| 0.5-0.6 | 7 | 2 | 28.6% | 39.8% |
| <0.5 | 11 | 2 | 18.2% | 61.0% |

→ ★ 5/10 day = 中位 score 0.6-0.7 が最強 (ROI 242%、 hit率 50%) ★
→ 上位 0.7+ は 6 R 中 hit 2 のみ (5/10 upset 多発の影響)
→ V15 baseline 案 B 改 strict (0.7+ 限定) は 5/10 で sub-optimal だった可能性

---

## 5/12-5/16 task plan (V18 真値化までの道筋)

### 5/12 (火) Phase 11b
- [ ] gaika_* 4 features 真値化
  - JRDB UKC.外厩 column lookup (馬基本情報)
  - 過去 3 R 外厩 top3 率 expanding window 計算
- [ ] odds_change_*_v18 4 features 真値化
  - save_odds_base 蓄積データから時系列差分
  - 3h前 / 30m前 vs 直前 odds 比率

### 5/13 (水) Phase 12b/13b
- [ ] jockey_*_winrate 4 features 真値化
  - JRDB KKA を distance/track/class/trainer で再集計
  - Bayesian alpha smoothing (alpha=10-20)
- [ ] return_horse_score / paddock_eval_v18 / saddle_room_score 真値化
  - JRDB CYB / TYB の詳細 column lookup

### 5/14 (木) V18 学習 data 構築
- [ ] data/v18/training_dataset_truevalue/ 整備
- [ ] V18 train_df_cache 再生成 (15 features 真値込み)

### 5/15 (金) V18 学習 + WF 評価
- [ ] python tools/train_v18_truevalue.py --gpu --save-model
- [ ] AUC 0.8939 vs V18 比較
- [ ] 採用判定 (AUC > V15 + 全年 > 0.85 + max gap < 0.05)

### 5/16 (土) V18 paper trade 開始
- [ ] V15 主軸維持
- [ ] V18 paper trade 並行 (paper_trade_engine_v22.py、 既存)
- [ ] 5/17 (土曜開催) 5/18 (日曜) で初評価

---

## 注意事項

### V15 投資保護
✅ V15 model file (keiba_model_v15_central_live.pkl.gz) 完全不変
✅ V18 model は新規 path (models/v18/) に保存、 V15 上書き禁止
✅ tools/predict_core.py / daily_predict.py / app.py 不変
✅ schtask 不変

### Phase 15 教訓 ('57 features constant') の反映
- Scaffold default のみ → variance 0 → LGB importance 0 = 学習効果ゼロ
- check-only mode で事前検出 → 真値化未完了で train 起動を防止
- 5/16 user CLI 実行時、 真値化完了確認後 train 起動

### 必要 environment
- Python 3.x (既存)
- lightgbm + xgboost (既存)
- torch + cuda (既存、 RTX 4070 Ti SUPER)
- 917 MB V15 cache (data/_v15_train_df_cache.pkl、 既存)

### 学習時間目安
- CPU: 30-60 min (5 fold WF + 165 features)
- GPU: 15-30 min (RTX 4070 Ti SUPER)

---

## 5/17 paper trade ready 状態

| model | status |
|---|---|
| V15 production | ✅ 動作中 |
| V18 真値版 | ⏳ 5/15 学習予定 |
| paper engine | ✅ tools/paper_trade_engine_v22.py 完備 |
| WF evaluation | ✅ tools/v18_wf_evaluation.py 完備 |

→ ★ 5/17 paper trade ready ★ (V18 学習 + 真値化完了が前提)
