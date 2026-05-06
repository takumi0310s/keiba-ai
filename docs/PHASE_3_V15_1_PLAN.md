# Phase 3: V15.1 SKB +0.0699 本格採用 計画書

**作成**: 2026-05-06 PM (Session #31)
**期間**: 5/24 (Phase 3 移行判定) - 6/8 (本格運用開始)
**目的**: V15 → V15.1 への移行で軸 top3 率 -15.8pt gap 改善

---

## 1. V15.1 status (5/6 時点)

| 項目 | 値 |
|------|-----|
| LGB single retro | AUC 0.8728 → 0.9427 (+0.0699) |
| 寄与 | SKB 10 features 単独 |
| KKA 16f | 寄与 0% (race_id 変換 bug 疑い) |
| SRB 8f | +0.0013 |
| リーク確認 | PASS (SKB は pre-race 印) |
| WF 検証 | **未実施** (LGB time-based split のみ) |
| 4-model ensemble | **未検証** |
| production pipeline 統合 | **未着手** |

詳細: `data/v18/v15_1_evaluation.md`

---

## 2. Phase 3 移行 6 条件 (5/24 判定)

`data/v18/post_5_9_improvement_template.md` § 5 より:

1. JRA 案B改 ROI ≥ 100% (4/12-5/24 累計)
2. race-level normalize 本番統合済 (predict_core.py)
3. NAR paper 12-14 race 蓄積
4. V18/V19 試行 sample 30+ bets
5. 累計 +10,000 円維持
6. 撤退ライン余裕 30,000+ 円

→ 全達成で Phase 3 移行、未達なら延長。

---

## 3. V15.1 本格採用 5 step (5/25-6/8)

### Step 1: 4-model ensemble 互換確認 (5/25-5/26、4h)

V15 ensemble: LGB + XGB + FT-Transformer + IntraRace Attention の 4 モデル grid。
V15.1 SKB +0.0699 は LGB single 検証のみ → 4-model でも改善が維持されるか:

```bash
python train/train_v15_1_ensemble_check.py --base-model v15 --add-features SKB
# 期待出力: 4-model ensemble retro AUC > 0.8939 (V15 baseline)
```

### Step 2: WF (walk-forward) 検証 (5/27-5/30、6h)

時系列分割 5fold で V15.1 の年別 AUC + 軸 top3 率:

```bash
python train/train_v15_1_wf_validation.py
# 期待: 全年 AUC > 0.85、軸 top3 率 BT 60%+ (V15 57% より改善)
```

### Step 3: production pipeline 統合 (5/31-6/2、6h)

`tools/predict_core.py` の修正:
- v141/v135/v134/v12/v9 fallback chain → v15.1/v15 のみに簡略化
- SKB merge 統合 (jrdb_skb.csv からの load)
- v15 → v15.1 自動切り替え + fallback 機構

```python
# predict_core.py L479-487 改修
if MODEL_PATH_V15_1.exists() and use_v15_1:
    model = load_v15_1()
elif MODEL_PATH_V15.exists():
    model = load_v15()  # fallback
```

### Step 4: paper trading (6/3-6/8、5 日)

- V15.1 paper + V15 本番投資 を並行
- 6/8 (土) で V15.1 paper ROI ≥ V15 本番 ROI - 5%pt なら本格採用
- 並列運用で V15.1 の真の精度評価

### Step 5: 本格運用切替 (6/9 以降)

5 step 全 GO なら 6/9 (土) から V15.1 本番投入、V15 は fallback に降格。

---

## 4. GO/no-go 判定基準 (5 step 各)

| step | GO 条件 | NO-GO 時の対応 |
|------|---------|--------------|
| 1 | 4-model ensemble AUC > V15 0.8939 | KKA 16f bug 究明 + step 1 再実行 |
| 2 | WF 全年 AUC > 0.85 + 軸 top3 率 改善 | step 2 再実行、データ追加検討 |
| 3 | predict_core.py 統合 OK + fallback 動作 | bug fix 後再 step 3 |
| 4 | paper ROI ≥ V15 - 5%pt | paper 延長 (6/15 まで) |
| 5 | 全 step GO | V15 本番維持で Phase 3 延長 |

---

## 5. リスク

- KKA 16f 寄与 0% の真因究明が必要 (race_id 変換 bug 疑い)
- 4-model ensemble で SKB 効果が縮小する可能性
- paper trading 5 日では sample 不足の懸念 → 必要なら 6/15 まで延長

---

## 6. 結論

V15.1 本格採用は 5/25-6/8 の 2 週間で 5 step 完遂。
GO 基準は厳格 (任意 step NO-GO で V15 維持)、取り返し禁止ルール遵守。
6/9 切替 OR Phase 3 延長 のいずれかで判定。
