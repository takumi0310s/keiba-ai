# Session #37 統合サマリー: B+C+D 並行 (Phase 3 大型前倒し)

**作成**: 2026-05-07 (Session #37 完了)
**所要**: 約 60 min (計画 3-4h より大幅短縮、 cache 既存 + ablation で完了)
**結論**: ✅ **3 領域 並行着手 + V15 動作不変 完全保証**

---

## 1. Session #37 deliverable 一覧

### A. V18/V19 sib抜き 単一-fold 再学習 (90 min plan → 5 min 実)
- ✅ `train/v18v19_no_sib/run_v18v19_no_sib_singlefold.py` 作成
- ✅ `data/v18/v18v19_retraining/v18_lgb_no_sib_v1.txt` (V18 sib抜き LGB)
- ✅ `data/v18/v18v19_retraining/v19_lgb_no_sib_v1.txt` (V19 sib抜き LGB)
- ✅ BT 2025 OOS 比較: V18 sib抜き AUC 0.8855 (-94bp vs Ens), winner_top1 45.76% (-2pt)
- ⚠️ 真価検証 = LIVE retro (5/2-5/3) は Session #38

### B. V15.1 LGB+XGB 互換確認 (60 min plan → 1 min 実)
- ✅ `train/run_v15_1_lgb_xgb.py` 作成
- ✅ `data/v15.1/v15_1_lgb_v37.txt` + `v15_1_xgb.json`
- ✅ WF (2024+2025) AUC: V15 0.8783 / V15.1 LGB 0.9470 / V15.1 Ens 0.9481 (**+698bp**)
- ✅ Ablation: SKB +675bp / SRB +5bp / KKA 0bp → SKB が改善源
- ⚠️ SKB リーク疑い 残存、 Session #38 で 検証

### C. V20 architecture 詳細設計 (45 min plan → 20 min 実)
- ✅ `docs/PHASE_3_V20_DETAILED_DESIGN.md` 拡張 (314 → ~580 行)
- ✅ Section 9-15 追加: 学習 data 構造 / 検証手順 / phase schedule / A/B test / リスク / JRA-VAN 計画 / file 構成
- ✅ Section 16: Session #37→#38 連携 plan

### D. V15 動作不変 final check (15 min plan → 5 min 実)
- ✅ `data/v18/session37_v15_safety_check.md` 作成
- ✅ V15 critical files (predict_core/daily_predict/race_auto_notify) 完全不変 (md5 一致)
- ✅ V15 model file unchanged
- ✅ schtasks 27 task 全て 不変
- ✅ 5/3 京都 12R 東大路S V15 score 0.6344 一致 (Session #32, #35 同値)

### E. 5 commits + push + Discord
- 進行中

---

## 2. 主要な発見 (Session #37 で確定)

### 2.1 V18/V19 sib抜き は BT で AUC 低下するが、 LIVE で逆転の可能性

| metric | V18 既存 (含 sib) | V18 sib抜き | Δ |
|--------|------------------|-----------|---|
| BT 2025 OOS AUC | 0.8948 | 0.8855 | -94bp |
| BT 2025 OOS winner_top1 | 47.79% | 45.76% | -2.03pt |
| (推定) LIVE shift_factor | 11x | 5-7x (期待) | -50% (期待) |

→ BT-LIVE gap の主因が sib_top3_rate / sib_shinba_wr のリークなら、 LIVE で sib抜き > 含 sib。 Session #38 で確定。

### 2.2 V15.1 改善は 99% SKB 由来 (KKA は coverage 0)

| feature set | AUC | Δ vs V15 |
|-------------|-----|---------|
| V15_only | 0.8765 | - |
| V15+KKA | 0.8765 | +0bp |
| **V15+SKB** | **0.9440** | **+675bp** |
| V15+SRB | 0.8771 | +5bp |
| V15+all | 0.9444 | +678bp |

→ V15.1 採用なら SKB 10 features (`skb_kishi_code_*`, `skb_baba_code_*`, `skb_kyaku_code_*`, `skb_turf_hoof`) が core。
→ SKB が JRDB pre-race release timing で legitimate なら 5/16 候補。 retroactive 集計なら除外。

### 2.3 V20 構築 plan 確定

- 6/9-6/30 の 3 週間で構築 (JRA-VAN 1 ヶ月再契約 ¥2,090)
- 4-model ensemble (LGB+XGB+FT+IR) 、 共通 80 / JRA 50 / NAR 12 features
- 7/1-7/14 paper trading → 7/15+ 段階投入 (10% → 50% → 100%)
- 6 GO 条件 (AUC + winner_top1 + shift_factor) で判断

---

## 3. 5/9 V15 投資への影響: 完全にゼロ

| protect 項目 | 状態 |
|-------------|------|
| V15 model file (md5) | 一致 |
| predict_core.py | 不変 |
| daily_predict.py | 不変 |
| race_auto_notify.py (戦略⑦) | 不変 |
| schtasks 27 task | 不変 |
| 5/3 京都12R V15 score 0.6344 | 一致 |
| 案B改 (12R 1勝クラスのみ 2,100 円) | 不変 |

→ Session #37 全作業は 隔離 dir に出力、 V15 production 経路と完全独立。

---

## 4. Session #38 plan (5/13-5/15)

### 4.1 高優先度

- [ ] V18/V19 sib抜き LIVE retro (5/2-5/3 + 4/26)
  - tools/v18_v19_retro_full.py 拡張: `--model-dir data/v18/v18v19_retraining`
  - shift_factor / winner_top1 比較
  - 期待: shift 11x → 5-7x、 winner_top1 34.5% → 38-42%
  
- [ ] V15.1 SKB リーク検証
  - JRDB SKB ファイル仕様確認 (release timing + 計算期間)
  - 5/2-5/3 race で `skb_kishi_code_*` を予測時 build 可能か
  - LIVE retro で V15.1 winner_top1 ≥ 50% 達成可能性

### 4.2 中優先度

- [ ] V18/V19 sib抜き 6-fold WF 拡大 (LGB+XGB)
- [ ] V15.1 4-model ensemble (FT-Transformer + IntraRace Attention) 追加
- [ ] V20 学習 data spec 確定 + JRA-VAN 再契約 timing 決定

### 4.3 低優先度

- [ ] tools/predict_v15_1.py 新規 (V15 並行 wrapper、 production 投入前)
- [ ] V20 学習 dry-run (small subsample で feature pipeline 確認)
- [ ] Session #38 末 で 5/16 GO/no-go 最終判断

---

## 5. 撤退ライン 確認

- 5/9 単日 ROI < 50% → 撤退
- 累計 -10,000 円 → 撤退
- **累計 -50,000 円 → 撤退** (現在 +13,530 円, 余裕 +63,530 円)
- Session #37 投資への影響: ゼロ

---

## 6. 結論

🟢 Session #37 は **B+C+D 並行着手 + V15 動作不変 完全保証** 達成。
時間 効率: 計画 3-4h → 約 60 min で完了 (cache 既存活用 + ablation で sib/SKB/SRB 切り分け)。
発見: V15.1 +697bp は SKB 単独 +675bp の貢献、 KKA / SRB はほぼゼロ。

5/9 V15 案B改 投資: **絶対遵守、 影響ゼロ確認**。
Session #38 で V18/V19 sib抜き LIVE retro + V15.1 SKB リーク検証 → 5/16 GO/no-go 最終判断。
