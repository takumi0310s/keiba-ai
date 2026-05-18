# B-3: baseline_v15.json 5/18 17:00 真値 update

> ★ honest 厳守、 V15 production 完全不変 ★
> 全数値 cumulative_results.csv 実測、 想定値は明示

## 0. 結論

| 項目 | 値 |
|------|----|
| ROI | **95.67%** |
| PnL | **-¥19,080** |
| n_settled | **629** |
| 95% CI (bootstrap 10K) | **[64.92%, 134.03%]** |
| CI が 100% を含む | **YES** = 統計的有意 勝ち なし |
| 撤退余裕 (-¥50,000 まで) | **¥30,920** |
| 中間アラート (-¥30,000 まで) | **¥10,920** ★ 警戒水準 ★ |
| 一時停止 (-¥40,000 まで) | **¥20,920** |
| V15 production | **完全不変保証** ✅ |

---

## 1. 5/18 17:00 真値 (cumulative_results.csv 実測)

`data/cumulative_results.csv` を `status == 'settled'` 行のみ集計。

| metric | 値 |
|--------|----|
| total rows | 663 |
| settled | **629** |
| pending | 33 |
| other status (`20260505`) | 1 (除外) |
| investment 合計 | **¥440,300** |
| actual_payout 合計 | **¥421,220** |
| ROI | pay / inv = **95.6666%** |
| PnL | pay - inv = **-¥19,080** |
| latest settled date | **20260517** |

期間: **2026-03-14 〜 2026-05-17** (約 9 週間)

---

## 2. 真値推移

| date | ROI | PnL | n_settled | source |
|------|-----|-----|-----------|--------|
| 5/16 evening | 101.33% | +¥5,240 | 563 | P0-1 baseline |
| 5/17 audit-4 | 98.34% | -¥6,920 | 596 | V15-audit-4 (5/17 G1 day 反映) |
| **5/18 17:00** | **95.67%** | **-¥19,080** | **629** | **B-3 status_verify (★ adopted ★)** |

押し下げ要因 (audit-4 → 5/18 17:00):
- ROI: -2.67pt (98.34 → 95.67)
- PnL: -¥12,160 (-6,920 → -19,080)
- n: +33 settled

---

## 3. 5/17 G1 day +33R 反映 詳細

audit-4 (n=596) → B-3 (n=629) の delta は **+33 rows newly-settled**。
これらは 5/17 開催分で audit-4 時点では pending だった race の結果照合完了分。

| metric | delta |
|--------|-------|
| investment 増 | +¥23,100 |
| actual_payout 増 | +¥10,940 |
| PnL 変化 | **-¥12,160** |
| delta ROI | **47.36%** (大幅 underperform) |

5/17 G1 day 単日 (66 settled rows) 全体:
- inv ¥46,200 / pay ¥21,880 / ROI 47.36% / PnL **-¥24,320**

5/16 jackpot day 比較 (34 settled rows):
- inv ¥23,800 / pay ¥54,110 / ROI 227.35% / PnL +¥30,310

→ **5/16 の jackpot で稼いだ +¥30,310 が 5/17 の -¥24,320 で大半消失**

---

## 4. 95% CI bootstrap (10K iter, seed=42)

samples size = n_settled = 629、 投入金額・配当の resampling-with-replacement。

| percentile | ROI |
|------------|-----|
| 2.5% | **64.92%** |
| 50% (median) | 94.30% |
| mean | 95.67% |
| 97.5% | **134.03%** |

**CI [64.92%, 134.03%] 100% を含む** → ROI > 100% の統計的有意性 **なし**。

→ 現状の累積成績は **「赤字 でも 黒字 でも 統計的に判別不能」** の段階。
楽観も悲観も自重し、 短期挙動 (5/16 jackpot 等) に judgment を引きずられない。

---

## 5. 撤退余裕

| ライン | 閾値 PnL | 余裕 |
|--------|---------|------|
| 中間アラート | -¥30,000 | **¥10,920** ★ 警戒水準 ★ |
| 一時停止 | -¥40,000 | ¥20,920 |
| 完全撤退 | -¥50,000 | ¥30,920 |

5/22 (FRI) PM までに転回 必要だが、 **5/22 中央開催なし**。
**5/23 (SAT) が次の開催**。 5/23-5/24 の挙動が次の判定 trigger。

中間アラートまで残 ¥10,920 のみで、 5/23 単日 投票上限 ¥2,100 × 数日で容易に到達可能 → 慎重監視必須。

---

## 6. V15 production 不変保証 ✅

B-3 で実施した変更:
- `data/task_outcomes/baseline_v15.json` のみ update (記録 file)

**変更なし**:
- `keiba_model_v135_central.pkl.gz` (Pattern A)
- `keiba_model_v135_central_live.pkl.gz` (Pattern B)
- `tools/predict_core.py`
- `tools/race_auto_notify.py`
- `tools/daily_predict.py`
- `data/cumulative_results.csv` (読み取りのみ)

→ V15 prediction pipeline / Discord 通知 / 結果照合 全て従来通り動作。

---

## 7. 関連 docs

- `docs/V15_AUDIT_4_CUMULATIVE_ROI_5_17_2026.md` (5/17 audit-4、 prev baseline)
- `docs/V15_AUDIT_5_INTEGRATED_VERDICT_2026_05_17.md`
- `docs/MEMORY_DRIFT_FINAL_RESOLUTION_2026_05_17.md`
- `data/task_outcomes/baseline_v15.json` (本 update 対象)
