# V18/V19 5/9 投入 GO/no-go 最終判定

**作成**: 2026-05-06 PM (Session #32 E、最終判定)
**結論**: 🔴 **NO-GO 確定** (4/6 NO、5/9 V15 案B改 単独維持)
**5/9 投資**: V15 案B改 (12R 1 勝クラスのみ、上限 2,100 円) のみ実行

---

## 1. 6 条件 達成 status

| # | 条件 | 値 | 判定 | 詳細 |
|---|------|-----|------|------|
| 1 | paper retro 5 月 100R で ROI ≥ 110% | sample 9-25 bets で表面 ROI 1450-2708% | 🔴 **NO** | sample 不足、過大評価リスク (`v18_v19_paper_retro_5_6.md`) |
| 2 | winner_top1 ≥ 45% | **34.5%** (BT 47.8% から -13.3pt 劣化) | 🔴 **NO** | calibration で改善せず、5/9 までに回復不可 (`v18_v19_winner_top1_audit_5_6.md`) |
| 3 | shift 真因 calibration で対応可能 | monotonic 変換で rank 不変 | 🔴 **NO** | feature shift が真因、calibration では本質改善不能 (`v18_v19_shift_resolution_5_6.md`) |
| 4 | 本番 pipeline 統合完了 | 設計のみ、predict_core 変更なし | 🟡 部分 | 隔離 module orchestrator で試作済 |
| 5 | fall-back 機構 (V18/V19 fail → V15) | 試作 skeleton、本実装未 | 🔴 **NO** | 本実装は 5/16+ (`v18_v19_fallback_design_5_6.md`) |
| 6 | 5/8 22:00 dry-run 完全動作確認 | 5/8 で実施予定 | 🟡 未実施 | 5/8 dry-run で確認 |

**達成数**: 0/6 (#4, #6 が部分/未実施、他 4 件 NO)

→ **GO 条件 全達成必須なので NO-GO 確定**。

---

## 2. 構造的 NO-GO 理由

### 2.1 winner_top1 の劣化が monotonic 変換で改善不可能

V18 BT ROI 295% は winner_top1 47.8% が根拠。 本番 34.5% (-13.3pt 劣化) では:
- 期待 payback = 0.345 × 平均オッズ 6.5 ≈ **224%** (表面 110% 超え)
- ただし sample 9-25 で CI 巨大 → 過大評価リスク 大

**calibration / softmax / 正規化は monotonic 変換で rank 不変**:
- raw 0.013 → cal 0.020 → norm 0.21 → どれも winner_top1 = 34.5% 不変
- 構造的に 5/9 までに 45% 復帰は不可能 (要 feature shift 修正、Phase 3+)

### 2.2 sample 不足

- 5/2-5/3 retro = 29 winner_known races / 9-25 bets
- 100R / 30+ bets には大幅不足
- 表面 ROI 1450-2708% も CI 巨大で信頼性低
- 過大評価のまま 5/9 投入は 取り返し禁止ルール違反

### 2.3 本実装未着手

V18/V19 を本番 pipeline に統合する作業 (`predict_core.py` 改修):
- 4-5h 工数、本セッションでは **絶対遵守ライン (predict_core 変更なし)** で禁止
- 5/8 までに別作業で実施可能だが、上記 #1-#3 NO で投入意義なし

---

## 3. 5/9 投資戦略 (確定)

### 3.1 V15 案B改 単独 維持

```
06:30 Morning_Sat → V15 11R/12R 軸候補通知
08:00 DailyPredict → V15 全 R 予測
09:30 MorningWeightCheck → V15 馬体重補正
10:00 MultiStagePredict_Test10 → V15 2R 補正 + 全 R 朝予測
14:50 MultiStagePredict_Race11_1450 → V15 全 11R 予測 (重賞含む観察)
15:45 MultiStagePredict_Race12_1545 → V15 12R + 案B改 採用 R 買い目
14:00-15:30 PAT 投票 → V15 案B改 12R 1勝クラスのみ 700×N 円 (上限 2,100 円)
```

V18/V19 投入なし、V15 単独で完了。

### 3.2 投資配分

| 項目 | 値 |
|------|-----|
| V15 案B改 | 採用 R × 700 円 (0-2,100 円) |
| V18 単勝 試行 | **0 円** (NO-GO) |
| V19 複勝 試行 | **0 円** (NO-GO) |
| **合計上限** | **2,100 円** |

→ 累計 +13,530 円から最悪 +11,430 円 (依然プラス維持)、撤退余裕 +61,430 円。

---

## 4. 5/16 GO/no-go 再判定 plan

5/16 (土) で再度 V18/V19 投入候補。 達成課題:

### 5/13 (火) - 5/15 (金) 平日 必須作業

| # | task | 工数 |
|---|------|------|
| (1) | feature shift 個別調査 (winner_top1 劣化の真因) | 90min |
| (2) | feature shift 修正 (data fix or 再学習) | 4-8h |
| (3) | V18/V19 model 再学習 (修正 features 反映) | 数時間 |
| (4) | retro 拡大 (4/11-5/15 paper、30+ bets) | 数時間 |
| (5) | predict_core 統合 (本実装) | 4h |
| (6) | fall-back 本実装 (orchestrator 拡張) | 2h |

→ 5/15 22:00 までに完了が前提。 工数 ~20-30h、平日 3 日では困難。

### 5/16 暫定判定

- (1)-(2) 完了 + winner_top1 ≥ 40% 達成: 🟡 paper trading 並行
- (1)-(6) 完了 + winner_top1 ≥ 45% + ROI ≥ 110%: 🟢 GO (1,000 円/日)
- 上記未達: 🔴 NO-GO 継続、Phase 3 (5/24+) で再検討

→ **5/16 でも GO 困難、Phase 3 (5/24+) が現実的タイミング**。

---

## 5. 5/9 当日のリスク監視 (V15 単独維持の前提)

V15 単独の継続稼働を最優先:

| 監視項目 | 確認方法 |
|---------|---------|
| V15 model 健全 | 09:30 MorningWeightCheck で予測成功 |
| 馬体重補正 動作 | 10:00 Test10 で 2R 補正 score 表示 |
| 11R 予測 (観察) | 14:50 Race11_1450 で全 11R 表示 |
| 12R 案B改 (主戦場) | 15:45 Race12_1545 で採用 R 買い目 |
| Discord 通知 | retry+log 強化済 (Session #31 A2) |
| schtasks 全 task | LastResult=0 確認 (5/9 17:00 以降) |

→ 全 OK なら 5/9 完遂、累計 +13,530 円維持。

---

## 6. 結論 (1 句)

🔴 **V18/V19 5/9 投入 NO-GO 確定** — winner_top1 34.5% (calibration で改善不能) + sample 不足 + 本実装未着手 + fall-back 試作のみ で **6 条件中 4 件明確 NO、2 件部分達成**。 取り返し禁止ルール下、5/9 は **V15 案B改 単独維持** で確実完遂。 5/16 GO 再判定は feature shift 修正完了が前提、Phase 3 (5/24+) が現実的タイミング。
