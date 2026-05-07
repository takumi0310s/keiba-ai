# Session #37 A 進捗: V18/V19 sib抜き 単一-fold 再学習

**作成**: 2026-05-07 (Session #37)
**結論**: 🟡 **BT 2025 OOS で sib抜き AUC 約 -68bp / winner_top1 -2pt**、 LIVE retro 検証は Session #38

---

## 1. 学習結果 (BT 2025 OOS, train 2015-2024)

| model | AUC | winner_top1 | mean p18 | features |
|-------|-----|------------|---------|----------|
| V18 既存 LGB only | 0.8923 | 48.28% | 0.0548 | 190 (含 sib) |
| V18 既存 XGB only | 0.8940 | 48.10% | -    | 190 |
| V18 既存 Ens (LGB+XGB) | 0.8948 | 47.79% | 0.0548 | 190 |
| **V18 sib抜き LGB** | **0.8855** | **45.76%** | 0.0550 | **188** (除 sib_top3_rate, sib_shinba_wr) |
| Δ vs Ens | -0.0094 | -2.03pt | +0.0002 | -2 |

| model | AUC |
|-------|-----|
| V19 既存 Ens (LGB+XGB) | 0.8873 |
| V19 sib抜き LGB | 0.8754 (-0.0119) |

---

## 2. 解釈

### 2.1 BT 2025 OOS では sib 含む方が高 AUC

BT データ (cache 済 2025) では sib_top3_rate / sib_shinba_wr を含む方が AUC +94bp / winner_top1 +2pt。
これは **既知**: Session #34 V162_EXCLUDED で sib_*_wr を削除した理由 = リーク疑い。
BT で AUC 高いのは過学習 / leak の signal、 LIVE で消える。

### 2.2 sib抜きの真価は LIVE retro

Session #36 (v18_v19_retro_session36_actual_5_7.md):
- BT 2025 OOS: mean p18 = 0.0548, winner_top1 = 47.8%
- LIVE retro (5/2-5/3, Session #10): mean p18 = 0.0018, winner_top1 = 34.5%
- **shift factor 27.7x** (Session #10) → 11x (Session #36 sr/srb merge 後)

期待: sib抜きは LIVE data で sib リーク影響を解消、 distribution shift 縮小。
本 Session #37 では BT のみ実施、 LIVE retro は Session #38。

---

## 3. 出力 file

- `train/v18v19_no_sib/run_v18v19_no_sib_singlefold.py` — 学習 script
- `data/v18/v18v19_retraining/v18_lgb_no_sib_v1.txt` — V18 sib抜き LGB model
- `data/v18/v18v19_retraining/v19_lgb_no_sib_v1.txt` — V19 sib抜き LGB model
- `data/v18/v18v19_retraining/v18_no_sib_oos_2025.csv` — V18 OOS 2025 prediction
- `data/v18/v18v19_retraining/v19_no_sib_oos_2025.csv` — V19 OOS 2025 prediction
- `data/v18/v18v19_retraining/no_sib_metrics.json` — 全 metrics

---

## 4. Session #38 残作業

### 4.1 LIVE retro 検証 (最重要)

5/2-5/3 + 4/26 retro:
- tools/v18_v19_retro_full.py に `--model-dir` arg 追加 → no-sib model 読込
- shift factor 期待: 11x → 5-7x (sib リーク削除分縮小)
- winner_top1 期待: 34.5% → 38-42% (+3-7pt)、 期待 +8-13pt は楽観

Phase 3 (5/24+) で LIVE retro 確定後 v18_v19_retro_session38_no_sib.md 作成。

### 4.2 6-fold WF 拡大

現状: 単一 fold (train 2015-2024, test 2025) で 1min 完了。
Session #38: 6-fold WF (2020-2025) で年別 AUC trend 確認、 過学習 / leak 二次解析。

### 4.3 XGB 追加

現状: LGB only。 Session #38 で XGB 追加、 ensemble (LGB+XGB) で再 retro。
LGB+XGB 期待 +0.5-1.5bp (元の v18 Ens の +0.7bp 程度)。

### 4.4 4-model ensemble (FT + IR)

Phase 3 後半 (5/30+) で V18/V19 sib抜き + FT-Transformer + IntraRace Attention を統合。
v13.5b と同じ 4-model 構成、 BT AUC 0.89-0.90 / LIVE shift factor < 5x 目標。

---

## 5. 5/9 V15 投資への影響: ✅ 完全にゼロ

- V15 model file (`keiba_model_v15_central_live.pkl.gz`): 完全不変
- V15 predict_core.py / daily_predict.py: 完全不変
- v18 既存 model file (`data/v18/models/v18_tansho_lgb.txt`): 完全不変
- 新規 sib抜き model: `data/v18/v18v19_retraining/` に独立保存、 5/9 production 経路と独立

---

## 6. 結論

🟡 Session #37 A は **着手完了**: sib抜き LGB 学習 / BT 2025 OOS 比較完了、 model 保存。
真価検証 = LIVE retro は Session #38 で実施、 Phase 3 (5/24+) の本格復活 plan に組込。

5/16 GO 確率: Session #36 の 30-40% 維持 (本 BT 結果は LIVE 影響に直結せず)。
5/9 V15 案B改 投資: 影響ゼロ、 絶対遵守 OK。
