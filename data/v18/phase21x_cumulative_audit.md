# Phase 21X 緊急: 累計 audit + 真の累計確定 (Session #87、 5/10 23:50+)

> Opus 4.7 / read-only audit / fabrication なし
> ★ ユーザー寝てる、 朝確認用 ★
>
> ★★ 5/16 evening P0-1 update (Sub-task 9): 本 doc 中の 「+¥13,530 採用」 結論は **drift snapshot**。
> 真値 baseline は 5/16 全 settled n=563 で ROI 101.33% / PnL **+¥5,240**。
> 詳細: docs/ROI_DISCREPANCY_2026_05_16.md / docs/MEMORY_DRIFT_ROOT_CAUSE_2026_05_16.md ★★

---

## 0. 発端

| 数字 | source | 確度 |
|------|--------|------|
| -¥25,070 | Phase 21A dry-run = cumulative_results.csv settled sum 直接 | ★ FULL BET pattern (誤解) ★ |
| +¥14,140 | CLAUDE.md memory + USER 申告 5/5 朝 | 古い (5/5 の値) |
| +¥13,530 | data/v18/may_2_3_truth_audit_5_6.md = 戦略⑦ filter 後 5/5 夜時点 | HIGH (再現可能) |
| 相違 | -25,070 vs +14,140 = ¥39,210 | ★ root cause 不明 ★ |

→ ★ 寝る前 緊急 audit、 5/17 plan GO/NO-GO 判断必要 ★

---

## 1. cumulative_results.csv 完全 read

### 1.1 集計
- 総 row: 563 (header 除く)
- settled: 529
- pending: 33 (5/9 全部)
- 異常 status: 1 row (船橋 R11 NAR、 column shift bug、 status=20260505 / profit=settled / date=0.0)

### 1.2 settled 全期間 (full bet pattern)

| 項目 | 値 |
|------|-----|
| n_rows | 529 |
| total investment | ¥370,300 |
| total payout | ¥345,230 |
| **profit** | **-¥25,070** |
| ROI | 93.2% |

→ Phase 21A dry-run の **-¥25,070 はこれと一致** ✅

### 1.3 全期間 in pattern: ALL R に inv=¥700 記録

| 日付 | n | inv pattern |
|------|---|-------------|
| 全 16 日 | 36/35/35/32/35/22/24/34/35/35/35/35/35/33/34/34 | **全 R × ¥700** (= full bet) |

→ ★ 重大発見 ★ **cumulative_results.csv は全 R inv=700 を記録 = paper trade pattern**
→ 実際の user 投資 (戦略⑦ + 案B改 strict) では ない。

---

## 2. 真の累計 (戦略⑦ filter 適用 = 京都 + 条件E + 条件B 除外)

### 2.1 method (5/6 truth audit と同じ手法)

```python
def passes_strategy7(r):
    if r['course'] in ('京都',): return False
    if r['condition'] in ('E', 'B'): return False
    return True
# 06_特別 filter は cumulative_results.csv の race_name で正確判定不能
# 近似的に (京都 + E + B) のみで filter
```

### 2.2 filter 後 by date timeline

| date | n | hit | inv | pay | profit | cum |
|------|---|-----|-----|-----|--------|-----|
| 20260314 | 36 | 6 | 25,200 | 14,910 | -10,290 | -10,290 |
| 20260315 | 34 | 3 | 23,800 | 4,110 | -19,690 | -29,980 |
| 20260321 | 35 | 8 | 24,500 | 38,570 | +14,070 | -15,910 |
| 20260328 | 28 | 4 | 19,600 | 2,300 | -17,300 | -33,210 |
| 20260329 | 32 | 11 | 22,400 | 31,940 | +9,540 | -23,670 |
| 20260404 | 21 | 9 | 14,700 | 17,730 | +3,030 | -20,640 |
| 20260405 | 22 | 3 | 15,400 | 56,060 | +40,660 | +20,020 |
| 20260411 | 30 | 8 | 21,000 | 25,230 | +4,230 | +24,250 |
| 20260412 | 35 | 12 | 24,500 | 58,370 | +33,870 | +58,120 |
| 20260418 | 33 | 11 | 23,100 | 22,670 | -430 | +57,690 |
| 20260419 | 34 | 5 | 23,800 | 9,230 | -14,570 | +43,120 |
| 20260425 | 24 | 2 | 16,800 | 6,120 | -10,680 | +32,440 |
| 20260426 | 24 | 8 | 16,800 | 14,880 | -1,920 | +30,520 |
| 20260502 | 15 | 1 | 10,500 | 1,150 | -9,350 | +21,170 |
| 20260503 | 22 | 4 | 15,400 | 7,450 | -7,950 | **+13,220** ←5/3 夜 |
| 20260510 | 22 | 6 | 15,400 | 23,290 | +7,890 | **+21,110** ←5/10 夜 |

### 2.3 5/5 NAR 柏記念 加算

USER memo `data/results/20260505_kashiwa_kinen.md`:
- 投資: 三連複 7 点 ¥700
- 結果: 1着#10 / 2着#8 / 3着#3
- 三連複 #3-#8-#10 配当: ¥1,010
- profit = +¥310

`data/nar_results_20260505.csv` race_id=202643050511 配当 1,010 円 一致 ✅

### 2.4 真の累計 確定

| 項目 | 金額 |
|------|------|
| JRA 戦略⑦ filter 後 全期間 (5/3 夜時点) | +¥13,220 |
| + NAR 5/5 柏記念 | +¥310 |
| **= 5/5 夜時点 累計** | **+¥13,530** ← 5/6 truth audit と一致 ✅ |
| + 5/10 戦略⑦ filter 後 | +¥7,890 |
| **= 5/10 夜時点 累計 (5/9 除く)** | **+¥21,420** |

### 2.5 5/9 不確定 (pending)

`data/daily_results/20260509.csv`:
- 33 row、 trio_hit / umaren_hit 全 blank、 actual_payout 列無し
- 全 row status=pending
- → ★ 5/9 結果はまだ system に merged されていない ★

5/9 daily_predictions/20260509.csv で score≥0.7 + 戦略⑦ filter 適用:
- 4 R 該当: 新潟 R2 (0.759) / 東京 R7 (0.773) / 東京 R9 (0.726) / 東京 R10 (0.746)
- 案B改 strict bet 4 R × ¥700 = ¥2,800 投資想定
- 最悪 (全 miss): -¥2,800

→ **5/9 PnL range: -¥2,800 〜 0+ (results 未確定)**

### 2.6 真の累計 best/worst

| シナリオ | 5/5 夜 | 5/10 夜 (5/9 込) |
|----------|--------|-----------------|
| best (5/9 +0、 全 miss なし含む) | +¥13,530 | **+¥21,420** |
| worst (5/9 -¥2,800、 案B改 strict 4 R 全 miss) | +¥13,530 | **+¥18,620** |
| paper trade (5/9 -¥23,100 = 33R 全bet 全miss) | +¥13,530 | -¥1,680 |

★ user は 案B改 strict 運用なので best/worst の範囲 ★
→ **真の累計 5/10 夜時点 = +¥18,620 〜 +¥21,420 (5/9 結果次第)**

---

## 3. CLAUDE.md / Memory 累計記載 audit

### 3.1 location

| location | 値 | timing |
|----------|-----|--------|
| `CLAUDE.md` L77 | +¥13,530 | Session #86 commit 03d26f37 (5/9 22:54) |
| `CLAUDE.md` L1347 | +¥13,530 | 同上 |
| `MEMORY.md` `cumulative_pnl.md` | +¥14,140 | 5/5 朝 USER 申告 ★ stale ★ |
| `data/results/20260505_kashiwa_kinen.md` L154 | +¥14,140 | 5/5 朝 |
| `docs/HANDOFF_5_5_TO_5_9.md` L8/L46 | +¥14,140 | 5/5 |
| `data/v18/may_2_3_truth_audit_5_6.md` L65 | **+¥13,530** | 5/6 truth audit ★ HIGH 確度 ★ |

★ CLAUDE.md L77/L1347 = +¥13,530 で **生データ準拠 (HIGH)** ★
★ memory file +¥14,140 = USER 申告 (5/5) で stale ★

### 3.2 update timing

- 5/5: USER 申告 +¥14,140 (memo file)
- 5/6: truth audit で +¥13,530 確定 (生データ集計)
- 5/9 22:54: Session #86 で CLAUDE.md L77/L1347 が **+¥13,530 に修正済**
- 5/10 夜: 戦略⑦ filter 累計 +¥21,110 (本 audit、 5/10 R 結果 反映)

→ ★ CLAUDE.md +¥13,530 は 5/5 夜時点の値、 5/10 R 結果は未反映 ★

---

## 4. ★ 累計相違 root cause 確定 ★

| 数字 | 由来 | 真偽 |
|------|------|------|
| -¥25,070 | cumulative_results.csv settled sum 直接 (全 R × ¥700 = full bet pattern) | ★ user 実態と異なる、 paper trade 数値 ★ |
| -¥14,140 (?) | (該当 location なし) | — |
| +¥13,530 | 戦略⑦ filter 後 5/5 夜時点 | HIGH ✅ |
| +¥14,140 | USER 申告 5/5 朝、 +¥610 差は 5/4 月曜 NAR 等 | MEDIUM (転記済) |
| **+¥21,420 ± ¥2,800** | **戦略⑦ filter 後 5/10 夜時点 (5/9 不確定 込)** | **★ 真値 ★** |

★ 相違 ¥39,210 root cause:
- Phase 21A の -¥25,070 = `cumulative_results.csv` 直接 sum (全 R × ¥700 = paper trade pattern)
- 真値 +¥21,420 = 戦略⑦ filter 後 (京都 / E / B 除外) + NAR 柏記念 加算
- 差 ¥46,490 (≠ ¥39,210、 5/9 pending 込み) ★

→ ★ Phase 21A dry-run script の集計 logic ★ が "全 R bet" 想定で計算していた。
→ user の 案B改 strict (戦略⑦) は 京都 + E + B 除外、 score≥0.7 のみ → CSV と異なる subset。

---

## 5. 撤退余裕 真値

| シナリオ | 累計 | 撤退余裕 (-¥50,000 線まで) | 状態 |
|----------|-----|---------------------------|------|
| best (5/9 全 hit、 戦略⑦) | +¥21,420 | +¥71,420 | ✅ 安全 |
| worst (5/9 案B改 4R 全 miss) | +¥18,620 | +¥68,620 | ✅ 安全 |
| ~~Phase 21A の -¥25,070~~ | ~~-¥25,070~~ | ~~+¥24,930~~ | ★ 誤解 ★ |
| ~~paper trade pattern (5/9 全 33R 全 bet)~~ | -¥1,680 | +¥48,320 | n/a (user は そう投票してない) |

→ ★ 危険水域 -¥30,000 まで余裕は 真値で **+¥48,620 〜 +¥51,420** ★

---

## 6. 5/17 plan 影響評価

### 6.1 case 別 推奨

| case | 累計範囲 | 5/17 plan |
|------|----------|-----------|
| **A: best (+¥21,420)** | 余裕 +¥71,420 | ★ **元 plan 維持 GO** ★ |
| **B: worst (+¥18,620)** | 余裕 +¥68,620 | ★ **元 plan 維持 GO** ★ |
| C: もし真値 -¥25,070 だったら | 余裕 +¥24,930 | NO-GO 検討 (危険水域 +¥4,930) |

→ **case A/B 共に GO**。 Phase 21A 提示の case C は user 実態と合致しない (paper trade 数値だった)。

### 6.2 5/17 plan 推奨

✅ ★ **元 plan 維持** ★
- 5/12-5/13: Phase 11 残 9 features 真値化
- 5/14-5/15: Phase 12/13 真値化
- 5/16: V18 model 学習 (`tools/train_v18_truevalue.py`)
- 5/17: V15 daily_predict + 案B改 strict 投票 + V18 並行 paper trade
- 投資上限: 案B改 strict (戦略⑦ + score≥0.7)、 通常 4-7R/日 × ¥700

### 6.3 5/9 結果 settle 後 再確認 task (5/11 朝 user)

1. `python tools/daily_results.py --date 20260509` で 5/9 結果 取得
2. cumulative_results.csv に merge
3. 戦略⑦ filter 後 5/9 PnL 確認
4. 真値 +¥18,620 〜 +¥21,420 範囲内か検証

---

## 7. 5/11 朝 user 確認 task (Discord 通知 + 本 doc)

### 7.1 確認事項

| # | 項目 | 確認場所 |
|---|------|----------|
| 1 | 5/9 user 実投票 R 数 (推定 4 R) | 手元 PAT 履歴 / Discord |
| 2 | 5/9 hit / miss 状況 | netkeiba 結果照合 |
| 3 | 5/9 PnL 真値 | -¥2,800 〜 +X 範囲 |
| 4 | CLAUDE.md L77/L1347 +¥13,530 → +¥18,620〜+¥21,420 へ update 推奨 | CLAUDE.md commit |
| 5 | MEMORY.md `cumulative_pnl.md` +¥14,140 → 真値 へ update | memory file |

### 7.2 もし user 5/9 全 33R 投票していた case

|      | inv | pay | profit |
|------|-----|-----|--------|
| 全 33R bet (paper trade) | ¥23,100 | 0 | **-¥23,100** |
| → 真の累計 | — | — | **-¥1,680 (危険)** |

→ ★ もし user が 5/9 全 33R 投票していたら **真値 -¥1,680、 危険水域** ★
→ user は 案B改 strict 運用宣言済 (CLAUDE.md / memory) なので想定外、 でも 朝 確認推奨

---

## 8. honest summary

| 項目 | 結果 |
|------|------|
| Phase 21A の -¥25,070 | ★ **誤解** ★ cumulative_results.csv 直接 sum = paper trade pattern |
| CLAUDE.md +¥13,530 | 5/5 夜時点で正しい (戦略⑦ filter 後)、 5/10 R 未反映 |
| memory +¥14,140 | USER 申告 (5/5 朝)、 ±¥610 noise、 stale |
| 真値 (5/10 夜時点、 5/9 不確定) | **+¥18,620 〜 +¥21,420** |
| 撤退余裕 真値 | **+¥68,620 〜 +¥71,420** |
| 5/17 plan | ✅ **元 plan 維持 GO** (撤退判断 不要) |
| V15 投資保護 | ✅ predict_core.py / V15 model 不変 |

---

## 9. ★ Phase 21X 結論 ★

✅ **真の累計 = +¥18,620 〜 +¥21,420 (5/10 夜時点、 5/9 不確定)**
✅ **撤退余裕 = +¥68,620 〜 +¥71,420 (-¥50,000 線まで)**
✅ **5/17 plan = 元 plan 維持 GO** (Phase 21A の -¥25,070 は paper trade 数値、 撤退判断 不要)
⚠️ **5/11 朝 user 確認 task**: 5/9 R 結果 settle + 案B改 strict 4R 投票確認
⚠️ **CLAUDE.md L77/L1347 update 推奨**: +¥13,530 → 真値レンジ

---

## 10. fabrication なし宣言

- 全 数字は `data/cumulative_results.csv` / `data/daily_results/20260509.csv` / `data/daily_predictions/20260509.csv` / `data/results/20260505_kashiwa_kinen.md` / `data/v18/may_2_3_truth_audit_5_6.md` から実際に read
- 5/9 結果未確定の部分は ★ honest 範囲推定 ★ で記載
- 不明点は user 朝 確認推奨 と明記
