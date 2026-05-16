# ROI 乖離 真値確定 (P0-1 formal analysis)

作成: 2026-05-16
作成 source: 親 agent 指示 P0-1 (read-only formal analysis)
作業 mode: read-only。 cumulative_results.csv / track_record.csv / strategy_v2_simulation.csv 改変なし

---

## 0. 結論 (★ 真値 ★)

| 指標 | 値 | 根拠 / scope |
|------|---:|--------------|
| **真値 ROI (baseline、 全 settled)** | **101.33%** | `data/cumulative_results.csv` status=settled、 n=563、 2026-03-14〜2026-05-16 |
| **真値 PnL (baseline、 全 settled)** | **+¥5,240** | inv ¥394,100、 pay ¥399,340 |
| **5/16 を除いた直前 snapshot (≤5/10)** | **93.23%** / **-¥25,070** | n=529 (Terminal B 報告と完全一致) |
| **5/16 単日** | **227.35%** / **+¥30,310** | n=34、 hit 11/34 = 32.4% (1 日で baseline 大幅反転) |
| **3/14-4/18 dedup window (CLAUDE.md 引用元)** | **120.62%** / **+¥46,620** | n=323、 CLAUDE.md「324/120.2%/+45,920円」 と整合 (1 row 誤差は dedup 方式違い) |
| **戦略⑦ applied (Terminal B 定義、 ≤5/10)** | **96.90%** / **-¥10,120** | n=466、 06_平場特別+E+B+距離≤1000 除外 |
| **bootstrap 95% CI (baseline 全 563)** | **[66.83%, 145.36%]** | 10,000 resamples |

**採用 baseline**: **ROI 101.33% / PnL +¥5,240 (n=563、 全 settled、 ≤2026-05-16)**

★ honest 注記 ★:
- 1 日 (5/16) で +¥30,310 跳ねた事実は **統計的偶然の可能性大** (1 日 ROI 227% は CI 上限超え)。 翌週 1-2 日で baseline 100% 前後に戻る可能性が高い。
- CLAUDE.md 記載「ROI 119.2% / +¥13,530」 は **cumulative_results.csv から再現不能**。 別系統の手動集計 と 推定。

---

## 1. 119.2% の出典追跡

### 1.1 CLAUDE.md 該当箇所

| 行 | 引用 |
|---|------|
| 72 | `**現行モデル**: **V15** (本番、150 特徴量、AUC 0.8939、本番運用 ROI 119.2%、戦略⑦込み 140%+ 想定)` |
| 1271 | `**期待効果**: ROI 119.2% → 140.3% (+21.1pt)` |
| 1363 | `V15 (現状): 119.2% (戦略⑦込み 140%) → 月利 約 2-3 万円` |

### 1.2 推定 calculation source

CLAUDE.md row 1187 に類似値あり:
```
| **全体** | **324** | **76** | **23.5%** | **120.2%** (**+45,920円**) | 142.6% |
```
これは「実戦成績（2026-03-14〜04-18, dedup後 324レース）」。
当 docs 「真値 (3/14-4/18 window) = ROI 120.62% / +¥46,620 / n=323」 と **ほぼ一致** (1 row + ¥700 程度の差は dedup 方式違い)。

→ **119.2% の真の source**:
- 大部分は **3/14-4/18 dedup 集計** から派生 (CLAUDE.md row 1187/1189)
- 「119.2%」自体は row 72 / 1271 / 1363 で 引用回しされており、 row 1187 (120.2%) の **rounded** か **わずかな別計算** と推定
- 完全特定は **不能**。 5/5 以降の追加 race を含めると ROI 急落 (≤5/10 で 93.23%) のため、 CLAUDE.md の 119.2% は **4/18 以前の snapshot で固定** と推定

### 1.3 +¥13,530 の出典追跡

CLAUDE.md row 77 / 1347:
```
- 現行 累計収支: **+13,530 円** / 撤退余裕 +63,530 円
- **撤退ライン**: 累計 -50,000円 (現在 +13,530円、 撤退余裕 +63,530円)
```

cumulative_results.csv からの再現を試みた結果:

| date cutoff | n | PnL |
|---|---:|---:|
| 4/18 | 323 | +¥46,620 |
| 4/30 | 428 | +¥3,680 |
| 5/5  | 495 | -¥28,360 |
| 5/9  | 495 | -¥28,360 |
| 5/10 | 529 | -¥25,070 |
| 5/16 | 563 | +¥5,240 |

→ **+¥13,530 はどの cutoff でも cumulative から再現不能**。 別系統 (手動集計 / track_record サブセット / roi_monitor 等) と推定。 出典 **unknown**。

参考: MEMORY.md は `+14,140 円 (5/5 朝時点)` を記録するが、 cumulative の 5/5 時点は -¥28,360 で大きく乖離。 5/5 までに **大量の自動 paper-trade 行が cumulative に混入** していた可能性。

---

## 2. 93.23% の出典追跡

### 2.1 source

`data/v21/strategy_v2_simulation_report.md` (commit cea7c2d9 / Terminal B、 5/16 evening)

```
| 対象 races | 529 | 529 | 529 |
| total inv | 370,300 | 326,200 | 385,700 |
| total pay | 345,230 | 316,080 | 339,940 |
| ROI | 93.23% | **96.90%** | 88.14% |
```

### 2.2 真値検証 (cumulative_results.csv 直接集計)

```
status=settled、 date <= 2026-05-10
n=529、 inv=¥370,300、 pay=¥345,230、 pnl=-¥25,070、 ROI=93.2298% (完全一致)
```

→ Terminal B の 93.23% は **正確、 fabrication なし**。

---

## 3. 96.90% (戦略⑦ applied)

### 3.1 source

同 simulation_report.md。
戦略⑦ exclusion 内訳:

| 除外理由 | n |
|---|---:|
| 06_平場特別 | 36 |
| 条件 B (重〜不良) | 16 |
| 条件 E (頭数≤7) | 11 |
| **合計** | **63** |
| **bet races** | **466** |

注: **京都除外は含まれない** (5/10 に削除済、 tools/strategy_layer_v2.py の race_auto_notify.py logic と一致)。

### 3.2 真値検証

```
strategy_v2_simulation.csv 直接集計 (s7only_recommended=True、 n=466):
inv=¥326,200、 pay=¥316,080、 pnl=-¥10,120、 ROI=96.8976%
delta vs baseline 93.23% = +3.67pt (完全一致)
```

→ 96.90% は **正確、 fabrication なし**。 ただし sample 466 で **依然 PnL -¥10,120**、 100% を切る。

---

## 4. 月別 / 場別 / 条件別 真値テーブル (全 563 settled、 ≤2026-05-16)

### 4.1 月別

| 月 | n | inv | pay | PnL | ROI% | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| 2026-03 | 173 | 121,100 | 92,660 | -28,440 | 76.52 | [37.68, 128.98] |
| 2026-04 | 255 | 178,500 | 210,620 | +32,120 | 117.99 | [63.17, 197.98] |
| 2026-05 | 135 | 94,500 | 96,060 | +1,560 | 101.65 | [42.48, 190.51] |

★ 観察 ★:
- 3 月は最も低調 (76.5%、 1 か月 -¥28,440)
- 4 月は最も好調 (118%、 1 か月 +¥32,120) → CLAUDE.md 「119.2%」 はこの時期 snapshot 由来と推定
- 5 月は前半低調 (5/10 まで -¥3,290) を 5/16 単日 +¥30,310 で巻き返し

### 4.2 場別 (race_id 4-6 桁から場 code 復元)

| 場 | n | inv | pay | PnL | ROI% | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Fukushima (03) | 72 | 50,400 | 70,700 | +20,300 | 140.28 | [64.25, 235.74] |
| Hanshin (09) | 126 | 88,200 | 106,030 | +17,830 | 120.22 | [28.89, 269.20] |
| Niigata (04) | 40 | 28,000 | 30,410 | +2,410 | 108.61 | [35.25, 216.50] |
| Chukyo (07) | 59 | 41,300 | 44,210 | +2,910 | 107.05 | [43.05, 190.82] |
| Kyoto (08) | 69 | 48,300 | 47,320 | -980 | 97.97 | [13.75, 249.98] |
| Nakayama (06) | 125 | 87,500 | 68,850 | -18,650 | 78.69 | [34.28, 145.22] |
| Tokyo (05) | 72 | 50,400 | 31,820 | -18,580 | 63.13 | [31.17, 102.20] |

★ 観察 ★:
- 福島・阪神・新潟・中京 が ROI 100% 超え (+¥43,450)
- 東京・中山・京都 が 100% 切り (-¥38,210)
- **京都 ROI 97.97% (N=69)** は SYSTEM_MASTER doc の「京都 20%」 と乖離。 SYSTEM_MASTER の 20% は別 source (session_5_16 の特定 subset) と推定
- 東京・中山が大幅 negative → 5/18+ 戦略再評価時の最優先除外候補

### 4.3 条件別

| 条件 | n | inv | pay | PnL | ROI% | hit% | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 164 | 114,800 | 109,660 | -5,140 | 95.52 | 28.66 | [53.23, 150.13] |
| C | 152 | 106,400 | 133,150 | +26,750 | 125.14 | 19.74 | [55.78, 216.88] |
| D | 201 | 140,700 | 151,400 | +10,700 | 107.60 | 20.90 | [50.70, 202.52] |
| B | 16 | 11,200 | 3,020 | -8,180 | 26.96 | 18.75 | [0.00, 64.64] |
| E | 11 | 7,700 | 950 | -6,750 | 12.34 | 18.18 | [0.00, 35.45] |
| X | 19 | 13,300 | 1,160 | -12,140 | 8.72 | 5.26 | [0.00, 26.17] |

★ 観察 ★:
- C / D 良好 (+¥37,450 合計)
- A は ほぼ break-even (95.5%、 -¥5,140)
- B / E / X は致命的 negative (-¥27,070 合計、 戦略⑦ で除外済の 2 件 B+E は妥当、 X は **新たに除外候補**)

---

## 5. 統計的信頼区間 (bootstrap 95% CI、 10,000 resamples)

| scope | n | ROI | 95% CI |
|---|---:|---:|---|
| 3/14-4/18 (CLAUDE.md window) | 323 | 120.62% | [70.30, 186.61] |
| ≤5/10 (Terminal B baseline) | 529 | 93.23% | [61.16, 134.48] |
| All settled ≤5/16 | 563 | 101.33% | [66.83, 145.36] |
| Strategy⑦ applied ≤5/10 (Terminal B) | 466 | 96.90% | (Terminal B 公表値) |
| Strategy⑦ applied (本 doc 推定 ≤5/16) | (~497) | (~99-103%) | 推定 (race_name 不能のため近似) |

★ 重要 ★: いずれの scope でも **95% CI が 100% を含む** → 統計的有意な利益ではない。 「119.2%」 「101.3%」 共に CI 内変動。

---

## 6. 真値 ROI の採用根拠

### 6.1 採用 baseline

**ROI 101.33% / PnL +¥5,240 / n=563 (2026-03-14〜2026-05-16、 全 settled)** を採用。

### 6.2 採用理由

1. **完全 reproducible**: cumulative_results.csv status='settled' に対する単純集計
2. **最新 snapshot**: 5/16 を含む全 race
3. **fabrication なし**: 戦略⑦ 仮想適用や paper-trade exclude を含めない baseline raw 値
4. **honest**: 「119.2%」 は CLAUDE.md の 4/18 snapshot 残存値 と推定、 5/5-5/9 の不調を反映していない

### 6.3 補助 baseline

| 用途 | 採用値 | n |
|---|---:|---:|
| 5/18+ 戦略判断 baseline (戦略⑦ 込み) | **~99-103%** (要再計算 5/16+) | ~497 |
| 月次 trend 把握 | 月別 ROI (4 月 118% / 3 月 77% / 5 月 102%) | - |
| 場別フィルタ判断 | Fukushima 140% / Tokyo 63% / Nakayama 79% | - |

---

## 7. 5/18+ 戦略判断の前提

### 7.1 baseline_v15.json への入力値 (推奨)

```json
{
  "baseline_roi_pct": 101.33,
  "baseline_pnl_jpy": 5240,
  "baseline_n": 563,
  "baseline_period": "2026-03-14 to 2026-05-16",
  "baseline_source": "data/cumulative_results.csv status=settled",
  "ci_95_low": 66.83,
  "ci_95_high": 145.36,
  "strategy_7_applied_roi_pct_5_10": 96.90,
  "strategy_7_applied_n_5_10": 466,
  "computed_at": "2026-05-16",
  "notes": "CLAUDE.md の 119.2% は 4/18 snapshot 残存値、 cumulative では再現不能"
}
```

### 7.2 月予算 / 撤退ライン 再評価

| 項目 | 旧 (CLAUDE.md) | 新 (真値) |
|---|---|---|
| baseline ROI | 119.2% | **101.33%** |
| 月次 PnL 期待値 | 「+¥28,953」 (保守的 142.6%) | **±¥0-3,000** (CI 含 -¥15k〜+¥20k) |
| 累計 PnL | 「+¥13,530」 | **+¥5,240** |
| 撤退余裕 | 「+¥63,530」 | **+¥55,240** (撤退 -¥50,000 まで) |

### 7.3 5/18+ priority

1. ★ **CLAUDE.md row 72 / 77 / 1347 / 1363 の 119.2% / +¥13,530 を 101.33% / +¥5,240 に修正** (commit/push は親 agent 任せ)
2. **Tokyo / Nakayama 場 ROI** が pessimistic (-¥37,230 合計) → 戦略⑦ 拡張で除外検討 (要 race_name 復元データ整備)
3. **5/16 単日 +¥30,310 は統計偶然と推定** → 楽観に走らず baseline 100% 前後で動作確認

---

## 8. honest report 注記

- 全数値 source: `data/cumulative_results.csv` (encoding=utf-8-sig、 status='settled' filter) + `data/v21/strategy_v2_simulation.csv` + `data/track_record.csv`
- **course 名は race_id 4-6 桁 (JRA 場 code) から復元**。 csv 内の `course` 列は cp932/utf-8 混在 mojibake で直接利用不能
- **race_name 列も同様 mojibake** のため、 「06_平場特別」 の現スナップショット (5/16 含む 563 rows) への新規 strategy_7 適用は **不能**。 strategy_v2_simulation.csv の 466/63 値のみ使用
- **track_record.csv (n=518、 ROI 133.84%、 PnL +¥122,700)** という別 dataset 存在。 cumulative_results との差分 172 rows は 4/19・5/1-5/9 等の自動 paper-trade 行と推定 (track_record は 12 日分のみ記録)。 track_record は **手動投票 subset の候補** だが、 schema (hit/payout 列) と cumulative の差異 (trio_hit/actual_payout) が完全一致せず、 当 doc では **採用しない**
- 「真の手動投票 subset」 完全 identify は **当 read-only 作業範囲外**。 必要なら 別 task で track_record vs cumulative_results 突合 を実施推奨

---

## 9. 出力 file 一覧

| file | type | 状態 |
|---|---|---|
| docs/ROI_DISCREPANCY_2026_05_16.md | 本 doc | **新規作成** |
| data/cumulative_results.csv | source | read-only (改変なし) |
| data/v21/strategy_v2_simulation.csv | source | read-only (改変なし) |
| data/v21/strategy_v2_simulation_report.md | source | read-only (改変なし) |
| data/track_record.csv | source | read-only (改変なし) |

commit/push なし (親 agent 集中管理)。

---

end of doc.
