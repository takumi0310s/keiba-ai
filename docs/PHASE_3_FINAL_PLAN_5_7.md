# Phase 3 計画 final 修正 (Session #38 C)

**作成**: 2026-05-07 (Session #38)
**状態**: V15.1 NO-GO 確定、 V18/V19 LIVE retro 結果反映、 V20 plan 維持

---

## 1. Session #37 + #38 結果 統合

| 領域 | Session #37 結論 | Session #38 検証 | Final 判定 |
|------|----------------|----------------|-----------|
| **V15.1 SKB** | LGB+XGB +698bp 改善、 SKB 99% 寄与 | 🔴 **POST-RACE LEAK 確定** | ❌ **採用 NO-GO** |
| **V18/V19 sib抜き** | BT 2025 OOS -94bp / -2pt 劣化 | LIVE retro (5/2-5/3) → 仮説 a/b/c 判定 | (LIVE 結果次第) |
| **V20 統合** | 6/9-6/30 architecture 確定 | 6 月実装 plan 維持 | 🟢 **GO** |

---

## 2. V15.1 採用 NO-GO 確定

### 2.1 リーク機構 (Session #38 A 確定)

`data/v18/v15_1_skb_leakage_audit_5_7.md` 詳細:
- SKB ファイル = "成績拡張データ" = **post-race**
- `skb_kishi_code_3` 単独で +480bp の異常な AUC gain
- finish 順位と単調 monotonic 関係 (post-race 確定)
- 1着馬は 3.2x 多く 非ゼロ kishi_code_3 を持つ
- 既存 V18 LEAK list に skb_anshin / aisho / heavy_apt を含む = SKB は LEAK 既知

### 2.2 Plan からの 完全削除

- ❌ V15.1 4-model ensemble (FT-Transformer + IntraRace) 構築 → 中止
- ❌ tools/predict_v15_1.py wrapper → 中止
- ❌ V15.1 paper trading → 中止
- ❌ 5/16 V15.1 投入候補 → 完全削除
- ❌ 5/24+ Phase 3 で V15.1 line → 削除

### 2.3 影響

- V15 維持、 5/16 / 5/24+ 投資への 影響 ゼロ
- Phase 3 plan 軽量化 (V15.1 line 削除で V18/V19 + V20 に集中)

---

## 3. V18/V19 sib抜き LIVE retro 仮説 (Session #38 B で 確定)

### 3.1 仮説 (a/b/c)

| 仮説 | 説明 | LIVE 結果 patten |
|------|------|----------------|
| **a (sib リーク仮説 正しい)** | sib は BT に強く寄与 / LIVE で消失 | sib抜き winner_top1 ≥ 38% (旧 34.5% 比 +3.5pt 以上) |
| **b (sib は本番でも有効)** | sib は両 環境で同様に寄与 | sib抜き winner_top1 ≤ 30% (旧比 -5pt 以上) |
| **c (sib は不要)** | sib の寄与は ノイズ範囲 | sib抜き winner_top1 32-38% (旧と ±3pt) |

### 3.2 仮説別 Phase 3 plan

#### 仮説 a: sib リーク仮説 正しい (期待最高)

- **5/16 投入**: 暫定 GO 候補 (paper trading で 1 週間確認後)
- **5/24+ Phase 3**: V18/V19 sib抜き 6-fold WF 拡大 + 4-model ensemble
- **6/9+ V20**: V18/V19 sib抜き plan 継承、 SKB は 完全除外

#### 仮説 b: sib は本番でも有効 (悲観)

- **5/16 投入**: NO-GO 確定 (V18/V19 全面再検討)
- **5/24+ Phase 3**: V18/V19 plan 全面修正
  - sib_top3_rate / sib_shinba_wr の **expanding window 修正版** で 再学習
  - leak vs 本物 signal の精緻 切り分け
- **6/9+ V20**: sib 系 features の扱い 再検討

#### 仮説 c: sib は不要 (中立)

- **5/16 投入**: paper trading のみ (既定路線)
- **5/24+ Phase 3**: V18/V19 sib抜き plan 維持
  - 6-fold WF + LGB+XGB ensemble + FT-Transformer / IntraRace
- **6/9+ V20**: SKB 系 完全除外、 sib 系 expanding 修正版で 採用検討

---

## 4. 5/13-5/15 plan (Session #38 結果反映)

### 4.1 5/13 (火) — V15.1 削除 確定 + V18/V19 LIVE retro 結果 確認

- ✅ V15.1 SKB リーク確定 → V15.1 plan 全面 削除 (Session #38 A 完了)
- ✅ V18/V19 sib抜き LIVE retro 結果 確認 (Session #38 B 完了)
- 仮説 a/b/c 判定 → 5/16 GO/no-go 暫定判断

### 4.2 5/14 (水) — V18/V19 6-fold WF (仮説 a/c 時)

- 仮説 a: V18/V19 sib抜き 6-fold WF (LGB+XGB) 学習 着手
- 仮説 b: V18/V19 全面再検討 (expanding window 版 sib 設計)
- 仮説 c: V18/V19 sib抜き 6-fold WF (LGB only) 学習 着手

### 4.3 5/15 (木) — 5/16 最終判断 + V20 着手準備

- 5/15 22:00 5/16 GO/no-go 最終判断 (再仮説 a/b/c で 確定)
- V20 構築 plan 着手準備:
  - JRA-VAN 再契約 期日 確認 (6/9 予定)
  - 学習 data spec 確定
  - LEAK_FEATURES_V20 list 拡張 (SKB を 完全除外含む)

---

## 5. 5/16 投入 plan (3 通り、 仮説別)

### 5.1 仮説 a: paper + 試行投入

- 5/16-5/22 の 1 週間: V18/V19 sib抜き single-fold model で paper trading
  - paper のみ、 実投票は V15 案B改 維持
  - paper sample 30+ races で hit / ROI 測定
- 5/23 結果評価:
  - paper ROI ≥ 100% → 5/24 から 試行投入 (V18 200 円/レース)
  - paper ROI < 100% → 5/24 paper trading 継続、 Phase 3 6-fold 学習結果 待ち

### 5.2 仮説 b: 投入完全保留

- 5/16-5/22: V15 案B改 単独 維持
- V18/V19 全面再検討作業 (expanding window 版 sib + 各 feature audit)
- 5/24+ Phase 3 で 再 GO/no-go

### 5.3 仮説 c: paper のみ

- 5/16-5/22: V18/V19 paper trading のみ
- 5/24+ Phase 3 で 6-fold WF + 4-model ensemble 構築 後、 6/9+ で投入判断

---

## 6. V20 構築 plan (Session #36/37 plan 継続、 SKB 反映)

### 6.1 plan 維持事項

- 6/9-6/30 構築期間 (3 週間)
- JRA-VAN 1 ヶ月再契約 ¥2,090
- 4-model ensemble (LGB + XGB + FT-Transformer + IntraRace Attention)
- 共通 80 + JRA 50 + NAR 12 features
- 7/1-7/14 paper / 7/15+ 段階投入

### 6.2 plan 修正事項 (Session #38 結果反映)

- **LEAK_FEATURES_V20 拡張**:
  - 既存 18 features に **SKB 系 全 10 features** 追加
  - 全 SKB code (kishi_code_*, baba_code_*, kyaku_code_*, turf_hoof) を 完全除外
  - 計 28 features 除外
- **共通 80 features の見直し**:
  - SKB 由来 features ゼロ
  - sib_top3_rate / sib_shinba_wr は 仮説 a/c で expanding 版 採用検討
- **検証 metrics 追加**:
  - 各 feature の corr_target で >0.10 は LEAK 疑い 自動 alert
  - finish 順位との monotonic 関係 自動チェック

---

## 7. 撤退ライン 再確認

| trigger | action |
|---------|--------|
| 5/9 単日 ROI < 50% | V15 案B改 一時 縮小 |
| 累計 -10,000 円 | V15 案B改 投入額 半減 |
| **累計 -50,000 円** | **完全撤退** (V15 / V18 / V19 全停止、 paper のみ) |
| 5/16 V18/V19 試行で 1 週間 ROI < 50% | V18/V19 投入 即停止 |

現在 累計収支: **+13,530 円**、 撤退余裕 **+63,530 円**。

---

## 8. ファイル構成 (Phase 3 終了時、 Session #38 反映)

```
keiba-ai/
├── # === V15 既存 (絶対不変、 7 月以降も fallback として維持) ===
├── keiba_model_v15_central{,_live}.pkl.gz
├── tools/predict_core.py
├── tools/daily_predict.py
│
├── # === V15.1 SKB (削除 確定、 archive のみ) ===
├── data/v15.1/                          # archive
│   ├── v15_1_skb_drilldown.json
│   ├── v15_1_lgb_v37.txt                # 学習結果のみ保存、 production 未使用
│   └── ...
├── # → V15.1 production 経路は **完全削除**
│
├── # === V18/V19 sib抜き (5/24+ 仮説 a/c 時 採用) ===
├── data/v18/v18v19_retraining/
│   ├── v18_lgb_no_sib_v1.txt            # Session #37 A
│   ├── v19_lgb_no_sib_v1.txt
│   ├── v18_lgb_no_sib_v2_6fold.txt      # Session #38+ 6-fold WF
│   ├── v19_lgb_no_sib_v2_6fold.txt
│   ├── v18_xgb_no_sib_v2.json           # Session #38+
│   └── v18_4model_no_sib_v3.pkl         # Phase 3 末
├── train/v18v19_no_sib/
│
├── # === V20 統合 (6/28+ 採用) ===
├── keiba_model_v20.pkl.gz
├── tools/predict_v20.py
├── tools/predict_v20_orchestrator.py
├── train/train_v20_jra_nar_ensemble.py
│
└── # === 評価 + paper trading ===
└── tools/eval_v20_vs_v15.py
└── data/v20_paper_trading/
```

---

## 9. Session #38 → #39 連携

### 9.1 Session #38 完了 deliverable

- ✅ V15.1 SKB リーク 確定 + 採用 NO-GO 判定 (Session #38 A)
- ⏳ V18/V19 sib抜き LIVE retro + 仮説 a/b/c 判定 (Session #38 B、 進行中)
- ✅ Phase 3 plan final 修正 (本書、 Session #38 C)
- ⏳ 3 commits + push + Discord (Session #38 D)

### 9.2 Session #39 (5/13-5/15) plan

- [ ] V18/V19 sib抜き 6-fold WF 拡大 (仮説 a/c 時)
- [ ] V18/V19 sib抜き LGB+XGB ensemble (仮説 a/c 時)
- [ ] V20 学習 data spec 確定 (LEAK_FEATURES_V20 = 28 features)
- [ ] 5/15 22:00 で 5/16 最終判断
- [ ] sib_top3_rate / sib_shinba_wr の expanding window 版 設計 (仮説 b 時)

---

## 10. 結論

🟢 **V15.1 採用 NO-GO 確定** (SKB POST-RACE LEAK)、 V15 維持。
⏳ V18/V19 sib抜き LIVE retro 結果待ち → 仮説 a/b/c で 5/16 plan 確定。
🟢 V20 構築 plan 維持 (6/9-6/30、 SKB 完全除外で plan 修正)。

5/9 V15 案B改 投資: **完全保護**、 影響ゼロ。
取り返し禁止 / 累計損失拡大 NG / 撤退ライン -50K 円 全て遵守。
