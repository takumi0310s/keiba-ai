# Session #37 B 進捗: V15.1 LGB+XGB 互換確認 + Ablation

**作成**: 2026-05-07 (Session #37)
**結論**: 🟢 **V15.1 LGB+XGB ensemble 動作確認完了**、 **+697bp の 99% は SKB 由来 (3 種 code feature)**

---

## 1. WF (LGB+XGB) 結果 (200k subsample)

| metric | V15 baseline | V15.1 LGB | V15.1 LGB+XGB Ens |
|--------|-------------|-----------|-------------------|
| Mean WF AUC (2024+2025) | 0.8783 | 0.9470 | 0.9481 |
| Δ vs V15 | - | +687bp | **+698bp** |
| 2024 AUC | 0.8800 | 0.9496 | 0.9503 |
| 2025 AUC | 0.8765 | 0.9444 | 0.9458 |

LGB+XGB ensemble = LGB single + 11bp (微増、 ensemble 寄与小)。
V15.1 全体改善は LGB single でほぼ取れる、 XGB 追加で +1bp 程度の安定化。

---

## 2. ⚠️ Ablation 結果 (重要)

| feature set | AUC (2025) | Δ vs V15_only | n_features |
|-------------|------------|---------------|-----------|
| V15_only | 0.8765 | - | 145 |
| **V15 + KKA** (16 件) | 0.8765 | **+0.0bp** | 161 |
| **V15 + SKB** (10 件) | **0.9440** | **+675bp** | 155 |
| **V15 + SRB** (8 件) | 0.8771 | **+5bp** | 153 |
| V15 + all (34 件) | 0.9444 | +678bp | 179 |

### 2.1 解釈

- **KKA (16 件): 0bp 改善** = coverage 0% (data 0、 skb_jra_seiseki_* etc. が空)
- **SKB (10 件): +675bp 改善** = V15.1 改善のほぼ全て
- **SRB (8 件): +5bp 改善** = race-level corner / bias、 寄与小
- **all = SKB + SRB + KKA**: 678bp ≈ SKB + SRB (KKA 寄与なし)

→ V15.1 採用なら **SKB だけで十分**、 KKA / SRB は学習データ削減に削除可。

---

## 3. ⚠️ SKB リーク検証

SKB 10 features: `skb_kishi_code_1/2/3`, `skb_baba_code_1/2/3`, `skb_kyaku_code_1/2/3`, `skb_turf_hoof`

### 3.1 各 feature の意味 (JRDB ドキュメンテーション 推定)

- `skb_kishi_code_1/2/3`: JRDB 推奨騎手 ID top 3 (raw jockey ID, not aggregated WR)
- `skb_baba_code_1/2/3`: 馬場適性 code top 3
- `skb_kyaku_code_1/2/3`: 脚質 (running style) code top 3
- `skb_turf_hoof`: 芝向き 蹄種フラグ

→ **horse-race 単位で JRDB の事前分析 結果**、 release timing は **race 前** (期待)。

### 3.2 リーク疑い 残存

- 既存 `LEAK_TIME_INVARIANT` には `skb_anshin`, `skb_aisho`, `skb_heavy_apt` を含むが、 新 V15.1 features (`skb_kishi_code_*`, `skb_baba_code_*`, `skb_kyaku_code_*`) は **未含む**
- これらが「JRDB の 後追い retroactive 集計」 (例: 全期間データから全期間にわたる kishi 適性) なら **time-invariant leak**
- 「per-race-date 計算」 なら **legitimate pre-race signal**

### 3.3 検証方法 (Session #38)

3 つの確認:
1. **JRDB SKB 仕様確認**: SKB ファイルの release timing と計算 期間 (per-race date / 全期間)
2. **LIVE retro**: 5/2-5/3 race で SKB features を予測時に再構築可能か (predict_core / 当日 file から取得可能か)
3. **shift factor 確認**: BT vs LIVE で SKB 系 feature distribution 一致するか

---

## 4. 出力 file

- `train/run_v15_1_lgb_xgb.py` — LGB+XGB 学習 + WF
- `train/run_v15_1_ablation.py` — KKA/SKB/SRB 切り分け
- `data/v15.1/v15_1_lgb_v37.txt` — V15.1 LGB model (Session #37 B)
- `data/v15.1/v15_1_xgb.json` — V15.1 XGB model
- `data/v15.1/v15_1_lgb_xgb_results.json` — LGB+XGB metrics
- `data/v15.1/v15_1_wf_results.csv` — WF (per fold)
- `data/v15.1/v15_1_ablation_results.json` — KKA/SKB/SRB ablation

---

## 5. Session #38 残作業

### 5.1 SKB リーク 最終検証 (最優先)

- [ ] JRDB SKB ドキュメント確認 (取得 timing + 計算期間)
- [ ] 5/2-5/3 race で `skb_kishi_code_*` features の事前取得確認
- [ ] LIVE retro winner_top1 ≥ 50% 達成可能性 評価

### 5.2 4-model ensemble 統合

V15.1 LGB+XGB に FT-Transformer + IntraRace Attention 追加:
- v13.5b と同じ 4-model 構成
- Grid Search 重み最適化
- 期待 AUC ~0.95 (LGB+XGB 既に 0.948)

### 5.3 5/16 投入判断 (Session #38 末)

- SKB リーク クリア → V15.1 を 5/16 から段階投入候補
- リーク疑い残存 → V15 案B改 維持、 Phase 3 (5/24+) に sib抜き 等優先

### 5.4 V15.1 の予測時 features 整備

`predict_core.py` 不変原則の元、 V15.1 用の wrapper 必要:
- `tools/predict_v15_1.py` (新規, V15 と並行)
- 当日 SKB / SRB / KKA features を JRDB ファイルから build
- production 投入前に scrape / data flow 確認

---

## 6. 結論

🟢 V15.1 LGB+XGB ensemble は **動作確認完了**、 +698bp 改善 (BT 2025 OOS)。
⚠️ 改善のほぼ全 (+675bp) が **SKB 10 features 由来**、 リーク検証 必要。
SKB の release timing が clean なら 5/16 候補、 retroactive 集計なら除外。

5/9 V15 投資への影響: **完全にゼロ** (V15 model + predict_core 完全不変)。
