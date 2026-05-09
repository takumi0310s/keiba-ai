# 5/16 V18 sib_w5 trial 最終 plan v5 (Session #74)

**作成**: 2026-05-09 (Session #74、 5/9 18:30 過ぎ)
**前版**: PLAN_5_16_V18_V19_DEPLOYMENT_v2.md (Session #42 H、 5/8)
**v5 (本ファイル)**: Session #43-#73 全発見反映、 5/16 直前確定 plan

---

## 0. 旧 plan (v2) との差分 (反映必要事項)

| 旧 plan の前提 | 結論 (Session #43-#73) | v5 反映 |
|--------------|----------------------|--------|
| V15.5 採用想定 | **Session #50 NO-GO 確定** | V15.5 完全除外 |
| 動画 features 期待 | **Session #62/63 netkeiba 動画 / 静止画 全 server block** | Phase 4 (JRA-VAN RV 7-9月) で再挑戦 |
| 7 vs 11 点 +2 頭検討 | **Session #69 NO-GO 確定** (案B改 strict 7 点 維持) | +2 頭 完全除外 |
| V18/V19 並列投入 (案 1) | sample 不足 + 並行リスク | V18 sib_w5 単独 trial (案 2 採用) |
| LIVE retro 1 週分のみ | Session #56 V20 ensemble AUC 0.90025 補強 | trial 信頼度 up |
| LEAK 既知 8 件 | **Session #51 LEAK 12 件発見** (追加 4 件) | V20 で完全除外、 V18 trial には影響なし |

---

## 1. 投入 model (5/16 trial)

### ✅ 投入する
- **V18 sib_w5** (主役、 期待 +0.0689 AUC vs no_sib)
  - sib_top3_rate_exp window=5 採用 (Session #42 F 最適化結果)
  - LIVE retro winner_top1: 31.03% (5/2-5/3 1 週分)
  - BT WF AUC 0.8845
- **V19 sib_w5** (補助、 相互強化観測のみ、 投票せず)

### ❌ 投入しない (理由付)
- **V15.5** — Session #50 NO-GO (改善 微小、 過学習 リスク)
- **動画 features** — Session #62/63 server block 確定、 Phase 4 で再挑戦
- **+2 頭 (9 点運用)** — Session #69 NO-GO、 ROI 改善 認められず
- **V20** — 6/8 投入候補、 5/16 段階では時期尚早

---

## 2. 投入 features (5/16)

### ✅ 採用
- **Session #65 + #72 Stage 2 二段階予測** (Stage 1 → 上位 6 頭抽出 → Stage 2 で IntraRace 再評価)
- **Session #71 全馬 score 完全保存** (5/10〜 蓄積開始、 5/16 までに 1 週分蓄積予定)
- **Sprint 1 全 5 idea** (軽量改善、 LEAK 監査 PASS 済)
- **Sprint 2 一部** (LEAK 監査 PASS 済 idea のみ、 maiden / jump v2 は別途確認後)

### ❌ 不採用 (理由付)
- **V15.5 features** — Session #50 NO-GO archive
- **動画 features** — server block (Session #62/63)
- **NAR V5 features** — NO-GO archive
- **V20 expanding sib all** — Session #55 delta -0.0000 (sib_w5 は特殊例で OK)
- **V20 interaction PoC** — Session #57 -2bp〜+1.8bp (V15 真の飽和)

---

## 3. 投票 strategy (5/16)

### V15 案B改 strict 維持確定 (Session #69 + #70)

| 項目 | 値 |
|------|----|
| 戦略 | 案B改 strict (戦略⑦込み、 7 点 trio) |
| 5 月 ROI 実績 (Session #70) | **125.1%** |
| 累計収支 | **+12,830 円** (5/9 終了時) |
| 撤退余裕 | +62,830 円 |
| 投資/race | 700 円 (戦略⑦ filter 後) |

**理由**:
- ROI 125.1% 安定 → 既に黒字 path 確立
- +2 頭 NO-GO 確定 (Session #69) → strict 7 点 が局所最適
- V18 trial は **paper trade のみ** (実投票せず V15 と並行観測)

### V18 sib_w5 trial (paper のみ)

| 項目 | 値 |
|------|----|
| trial mode | **paper trade** (実投票なし) |
| 観測対象 | top1 confidence ≥ 0.4 のレース |
| 想定 race 数 (5/16) | 2-4 race |
| 仮想投資/race | 500 円 (記録用) |
| 仮想投資 上限 | 1,000-2,000 円相当 |

**Eighth Kelly (2.2%) を仮想投資基準として記録のみ**。
実投票は V15 案B改 strict のみ。

### 期待結果 (5/16 〜 5/22)

- V18 sib_w5 paper hit rate +X% (V15 比較用)
- ROI 125% → 130-140% 改善 期待 (V18 統合 5/22+ で本投入時)
- 全馬 score 1 週間 蓄積 (Session #71)
- Stage 2 vs Stage 1 比較 data 蓄積

---

## 4. merge 対象 dev branch (5/15 22:00 実施)

### ✅ merge する (6 branch)
| branch | 内容 |
|--------|------|
| `dev/sprint1` | Sprint 1 5 idea (軽量改善) |
| `dev/sprint2` | 一部 idea (LEAK PASS のみ、 maiden / jump v2 は確認後) |
| `dev/training-poc` | Session #47-#72 PoC 集約 |
| `dev/two-stage` | Session #48 + #65 + #68 + #72 Stage 2 二段階予測 |
| `dev/audit-backtest` | AUDIT-1 + Session #69 (+2 頭) + #70 (5月 ROI) |
| `dev/sprint6-kka` | KKA parser 修復 (Session #53)、 race_id format 調整後 |
| `dev/video-poc` | 動画 pipeline (Phase 4 用、 5/16 では未活用、 保持目的) |

### ❌ merge しない (archive)
| branch | 理由 |
|--------|------|
| `dev/sprint4` | V15.5 NO-GO archive (Session #50) |
| `dev/nar-v5` | V5 NO-GO archive |
| `dev/v20-expanding` | NO-GO (Session #55) |
| `dev/v20-interaction` | NO-GO (Session #57) |

### ⚠ 保持 (merge せず branch 保持)
| branch | 理由 |
|--------|------|
| `dev/v20-ensemble` | 5/22+ Sprint 6 V20 構築 素材として保持 (Session #56 AUC 0.90025) |

---

## 5. 5/16 起床時 タスク (詳細は V18_TRIAL_5_16_CHECKLIST.md)

1. dev branch merge 確認 (5/15 22:00 実施済 確認)
2. main 更新確認 (V15 model 不変、 schtasks 連携のみ更新)
3. V18 sib_w5 paper trade 起動
4. 5/16 投票実行 (V15 案B改 strict のみ、 V18 は paper)
5. Discord 通知 (V18 trial 経過、 V15 結果)

---

## 6. 5/16 〜 5/22 観測項目

| 項目 | 目標 | 失敗時の判断 |
|------|------|------------|
| V18 sib_w5 paper winner_top1 | ≥ 30% (LIVE retro 31.03% 維持) | < 25% で 5/22+ 投入 NO-GO 再判定 |
| V18 sib_w5 paper shift | ≤ 12x (BT vs LIVE) | > 15x で hybrid LEAK 疑い再検証 |
| V15 案B改 strict ROI | ≥ 110% (5月 125% 継続) | < 100% で V15 単独継続再判断 |
| 全馬 score 蓄積 | 7 日分 | データ欠損なら Session #71 再点検 |
| Stage 2 vs Stage 1 | top1 一致率測定 | 大きな乖離は Stage 2 logic 再点検 |

---

## 7. 5/22 〜 6/8 V20 構築 (Session #56 ensemble base)

| 期間 | 内容 |
|------|------|
| 5/22-5/24 | KKA features 統合 (Session #53 修復済) |
| 5/25-5/28 | 4-model ensemble (Session #56 AUC 0.90025 PoC を本実装) |
| 5/29-6/1 | IntraRace Attention 主軸調整 |
| 6/2-6/5 | LEAK 12 件 完全除外 (Session #51) + sib_w5 expanding 統合 |
| 6/6-6/8 | V20 v1 6-fold WF + LIVE retro |

**V20 期待値**: WF AUC 0.90+、 LIVE winner_top1 35%+。

---

## 8. 6/8 V20 投入候補 (judgement)

GO 条件:
- WF AUC ≥ 0.890
- LIVE retro winner_top1 ≥ 33%
- shift ≤ 10x
- LEAK 監査 ALL PASS (Session #51 12 件 完全除外確認)
- paper trade 30 日 ROI ≥ 110%

GO 時:
- 投票 strategy 再評価 (V15 案B改 vs V20)
- Eighth Kelly 適用
- max ¥5,000-10,000/日

NO-GO 時:
- V15 案B改 strict 継続 (ROI 125%+)
- V20 v2 改善版 着手

---

## 9. 7/1 V20 production (paper trade 30 日後 GO 時)

- 週末のみ、 上限 5,000 円/日
- 7/15 順調なら 増額 (週末 1万円/日 + 平日 5,000 円/日)
- 8/1 V15 archive 判定 (1 か月並行運用後)

---

## 10. 7-9 月 Phase 4 (JRA-VAN RV 動画) ★Session #74 期待★

JRA-VAN RV = Phase 4 救世主 (動画 server block 回避):
- JRA-VAN NEXT 自動分配機能 で動画 取得
- Session #62/63 で確定した netkeiba 全 block 問題を JRA-VAN RV で迂回
- DLC HORSE-10 / YOLOv8 zero-shot で姿勢推定
- gait_symmetry / stride / head_bobbing / posture / ear_pos 5 features

---

## 11. 9/2 V21 候補 (V20 + Phase 4 動画 features)

判定基準: V20 比 +0.005 AUC、 LIVE +1pt 以上。

---

## 12. 10-12 月 Phase 5 (V22 RL + 30 年 backtest)

- 強化学習による 投票 strategy 最適化
- 1996-2025 30 年 backtest (TFJV 6 年分 → 全 30 年 拡張)

---

## 絶対遵守事項

🔴 NEVER:
- predict_core.py / daily_predict.py / app.py 変更 (5/16 まで)
- V15 model 変更 (5/16 まで)
- schtasks 既存 50 件 変更
- 既存 dev branch (上記 archive 除く) 強制削除
- destructive git op (reset --hard / push --force)

🟢 OK:
- main に 1 commit (docs/ のみ、 本 plan + merge plan + checklist)
- Discord 1 通通知 (dedup 適用)

---

## 累計 PnL 維持目標

| 日付 | 累計 | 撤退余裕 |
|------|------|--------|
| 5/9 終了 | +12,830 円 | +62,830 円 |
| 5/16 trial 後 (V15 のみ実投票) | ±0 〜 +5,000 円 想定 | +50,000 円 維持 |
| 5/22 V18 本投入時 | +20,000 円 想定 | +70,000 円 |
| 6/8 V20 投入時 | +30,000 円 想定 | +80,000 円 |
| 7/1 V20 production | +50,000 円 想定 | +100,000 円 |

撤退ライン: 累計 -50,000 円 (現 +12,830 円 → 余裕 +62,830 円)。

---

**Session #74 doc 完。 V18 sib_w5 trial ready。**
