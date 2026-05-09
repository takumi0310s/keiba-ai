# 5/16 V18 sib_w5 trial checklist (Session #74)

**作成**: 2026-05-09 (Session #74)
**対象**: 2026-05-16 (土曜、 V18 paper trade trial 開始日)
**前提**: 5/15 22:00 dev branch merge 完了 (MERGE_PLAN_5_15.md 参照)

---

## 0. 前提確認 (5/15 23:00 時点)

- [ ] main HEAD が merge 完了 commit
- [ ] schtasks Keiba 系 50 件 不変
- [ ] V15 model load OK (`keiba_model_v135_central_live.pkl.gz`)
- [ ] 5 項目テスト PASS
- [ ] Discord 通知 受信 (5/15 22:00 merge 完了報)
- [ ] 累計収支 +12,830 円 確認

---

## 1. 5/16 06:00 起床時 (出発前)

### 1.1 system 状態 check (10 分)

- [ ] `python tools/pre_race_check.py` 実行 (8 項目)
- [ ] V15 model load 正常
- [ ] feature_lookups.pkl 存在 + キー数 ≥ 10
- [ ] netkeiba 出馬表 access OK
- [ ] JRA 馬場 data Shift_JIS パース OK
- [ ] 気象庁 API 応答 OK
- [ ] SQLite DB 過去 record 読込 OK
- [ ] 当日最初の race URL で 87/87 features 生成
- [ ] zero feature 5 個未満

### 1.2 V18 sib_w5 paper trade 起動 (5 分)

- [ ] V18 model load 確認 (sib_w5 weight 適用)
- [ ] V18 paper trade flag ON 確認 (実投票せず記録のみ)
- [ ] 出力 path 確認 (`data/v18/trial_5_16_paper.csv` 想定)

### 1.3 schtasks 動作確認 (5 分)

- [ ] DailyPredict 08:00 起動予定確認 (V15 path、 不変)
- [ ] RaceAutoNotify 08:45 起動予定確認 (V15 案B改 strict)
- [ ] DailyResults 18:00 起動予定確認

### 1.4 累計収支 確認 (2 分)

- [ ] 累計 +12,830 円 (5/9 終了時) 維持 確認
- [ ] 撤退余裕 +62,830 円 確認

---

## 2. 5/16 08:00 〜 (DailyPredict 自動実行 後)

### 2.1 V15 予測 確認

- [ ] `data/daily_predictions/20260516.csv` 生成 確認
- [ ] 全 R で score 計算 完了
- [ ] 全馬 score 完全保存 (Session #71) 動作確認
- [ ] Stage 2 二段階予測 (Session #65 + #72) 動作確認

### 2.2 V18 paper 予測 確認

- [ ] `data/v18/trial_5_16_paper.csv` 生成 確認
- [ ] V18 sib_w5 予測 全 R 完了
- [ ] V15 vs V18 top1 一致率 measure

---

## 3. 5/16 09:30 馬体重補正 (Session #44 所載)

- [ ] 馬体重補正機構 起動確認
- [ ] 朝予測 vs 09:30 再予測 diff 確認
- [ ] Discord アラート (体重急変 ±10kg 馬) 受信

---

## 4. 5/16 各 race 5 分前 (RaceAutoNotify 自動実行)

### 4.1 V15 案B改 strict 投票 (実投票)

- [ ] 戦略⑦ filter 適用確認 (06_特別 / 京都 / 条件E / 条件B 除外)
- [ ] Discord #買い目 通知 受信 (1 race 1 通)
- [ ] 投票実行 (700 円/race、 戦略⑦ 後の race のみ)
- [ ] 重複通知なし 確認 (Session #59 dedup logic)

### 4.2 V18 sib_w5 paper 記録 (実投票なし)

- [ ] V18 paper output 記録確認 (top1 confidence ≥ 0.4 の race)
- [ ] 仮想投資 500 円/race 記録
- [ ] V15 投票 race と V18 paper race の overlap 記録

---

## 5. 5/16 各 race 終了後

- [ ] V15 投票 結果 記録
- [ ] V18 paper 結果 記録 (仮想配当計算)
- [ ] hit / miss 即時 record

---

## 6. 5/16 18:00 (DailyResults 自動実行 後)

### 6.1 結果照合

- [ ] `data/daily_results/20260516.csv` 生成 確認
- [ ] V15 ROI (5/16 単日) 計算
- [ ] V18 paper ROI (5/16 単日) 計算
- [ ] 累計収支 update

### 6.2 Discord 通知

- [ ] V15 結果 通知 (#アップデート)
- [ ] V18 trial 結果 通知 (#アップデート、 dedup 適用)
- [ ] roi_monitor アラート 確認 (異常値時のみ)

---

## 7. 5/16 22:00 (1 日終了 review)

### 7.1 数値 record

| 項目 | 値 | 判定 |
|------|----|------|
| V15 案B改 strict 5/16 ROI | __% | ≥ 110% で OK |
| V15 hit count / 投票 race | __ / __ | — |
| V18 sib_w5 paper winner_top1 | __% | ≥ 30% で V18 健全 |
| V18 paper hit count / paper race | __ / __ | — |
| V15 vs V18 top1 一致率 | __% | reference |
| 累計収支 | +__ 円 | ≥ +10,000 円 維持 |
| 撤退余裕 | +__ 円 | ≥ +60,000 円 維持 |

### 7.2 異常検知

- [ ] V15 ROI < 100% → 5/17-22 観測継続、 5/22 V18 本投入再判定
- [ ] V18 paper winner_top1 < 25% → 5/22+ 投入 NO-GO 候補
- [ ] V18 shift > 15x → hybrid LEAK 疑い再検証
- [ ] 累計 -50,000 円 接近 → 即時 撤退 (絶対遵守)

---

## 8. 5/17 〜 5/22 (1 週間 trial 期間)

各日 同じ checklist 繰り返し。 1 週間で:

- [ ] V18 paper winner_top1 平均 ≥ 30%
- [ ] V18 paper shift 平均 ≤ 12x
- [ ] V15 ROI 平均 ≥ 110%
- [ ] 全馬 score 7 日分 蓄積完了 (Session #71)
- [ ] Stage 2 vs Stage 1 比較 data 蓄積

5/22 22:00 に V18 5/23+ 本投入 GO/NO-GO 判定。

---

## 9. 失敗時 rollback 手順

### 9.1 V18 paper trade 不調 (winner_top1 < 25%)

- 5/22+ 本投入 NO-GO
- V15 案B改 strict 単独継続 (ROI 125%+ 維持中、 撤退理由なし)
- V18 model 改善着手 (sib_w5 + Stage 2 + IntraRace 強化)

### 9.2 V15 案B改 strict 不調 (ROI < 100% 連日)

- 戦略⑦ filter 再点検
- 5/22+ 案B改 → 案A 切替 検討
- 累計 -10,000 円 で投資 半減 (撤退 第 1 段階)

### 9.3 system trouble (predict_core / daily_predict 異常)

- merge revert (MERGE_PLAN_5_15.md §6 参照)
- main HEAD を 5f5c3d43 (Session #71) に戻す
- V15 完全 restore 確認

### 9.4 schtasks 影響時

- schtasks /query で 50 件 確認
- 不足あれば setup_all_tasks.bat (管理者権限) 再実行

---

## 10. Discord 通知設計

### 10.1 通知 channel

| 内容 | channel | dedup |
|------|---------|------|
| 5/15 22:00 merge 完了 | #アップデート | 適用 (5min hash) |
| 5/16 06:00 pre_race_check 結果 | #アップデート | 適用 |
| 5/16 各 race 5 分前 V15 買い目 | #買い目 | 適用 (1 race 1 通) |
| 5/16 馬体重急変 アラート | #アップデート | 適用 |
| 5/16 18:00 結果 + V18 trial 経過 | #アップデート | 適用 |
| 5/16 22:00 1 日 review summary | #アップデート | 適用 |

### 10.2 重複防止

- Session #59 5min hash dedup logic 適用
- Session #64 全 R 通知 path 修正 適用

---

## 11. 絶対遵守事項

🔴 NEVER:
- predict_core / daily_predict / app.py 変更 (5/16 当日)
- V15 model 変更 (5/16 当日)
- schtasks 既存 50 件 変更
- V18 paper trade を 実投票化 (5/16 は paper のみ)
- 累計 -50,000 円 超過 (撤退ライン 絶対)
- destructive git op

🟢 OK:
- V18 paper trade 観測 + 記録
- Discord 通知 (dedup 適用)
- docs/ 追記 (review notes 等)

---

## 12. 5/16 終了時 期待値

| 項目 | 想定 |
|------|------|
| V15 投票 race 数 | 4-7 race (戦略⑦ filter 後) |
| V15 ROI (単日) | 100-140% (5月平均 125% 維持目標) |
| V18 paper race 数 | 2-4 race (top1 confidence ≥ 0.4) |
| V18 paper winner_top1 | 25-35% (LIVE retro 31% 中央値) |
| 5/16 単日 損益 (V15 実投票) | ±2,000 円 想定 |
| 累計収支 | +10,830 〜 +14,830 円 |

---

**Session #74 V18 trial checklist 完。 5/16 06:00 起床 ready。**
