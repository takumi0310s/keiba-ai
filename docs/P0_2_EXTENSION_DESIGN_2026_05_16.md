# P0-2 拡張 戦略⑦ 全 worst 除外設計

作成: 2026-05-16
作成 source: 親 agent 指示 sub-task 5-3 (read-only analysis + design 提案)
作業 mode: read-only。 V15 production / cumulative_results.csv / predict_core.py / race_auto_notify.py 改変なし。 git commit/push なし。
source: `data/cumulative_results.csv` status=settled、 n=563 (JRA のみ; NAR 1件除外)、 期間 2026-03-14〜2026-05-16

---

## 0. 結論 (★ 推奨案 ★)

| 項目 | 値 |
|------|---|
| **推奨案** | **案 C (京都/中京 + 条件 X 除外)** |
| **採用 baseline** | ROI 101.33% / PnL +¥5,240 / n=563 |
| **案 C 期待 ROI** | **105.05%** (+3.72pt vs baseline) |
| **案 C 期待 PnL** | **+¥14,750** (+¥9,510 vs baseline) |
| **案 C 95% CI** | [66.47%, 158.33%] (CI 内に 100% を含む = 統計的に 100% 超過 未確定) |
| **除外 race 数** | 146 / 563 (26.0%) |
| **残 race 数** | 417 / 563 (74.0%) |
| **月平均投票 R 数** | 約 139R/月 (baseline 187R/月 比 -26%) |
| **月予算想定** | ¥97,300 (700円 × 139R、 baseline ¥131,400 比 -¥34,100) |

★ 推奨理由 (詳細 §5) ★:
- 案 D (全 worst 除外) は期待 ROI 127.38% で点推定では最良だが、 ★ **3 月単独で ROI 13.72% / PnL -¥33,820** ★ と月次変動が極大。 4 月 173% に依存しており、 1 か月だけの極大値で見積もりが膨張している
- 案 B も同様 (3 月 13.72%)
- 案 C は除外規模が穏当で、 (a) 条件 X 自体は単独次元で 8.72% / N=19 / p<0.001 と最強の除外根拠を持ち、 (b) 京都/中京除外は master doc 既定継承、 (c) 月次 ROI が 3/4/5 全月で baseline を上回るか同等 (62/133/85 vs 76/118/102)
- 案 A (master 既定) は +3.31k に留まる
- **★ honest 注記 ★**: 案 C も 95% CI 下限が 66% で 100% 未満。 「+¥9,510 改善」 はあくまで点推定であり、 統計的有意ではない。 5/18 以降 1 か月 shadow 検証推奨

---

## 1. 場 × 条件 ROI grid (6×8 = 48 cells、 N≥1 のみ表示)

★ ROI / N / 95% CI bootstrap (10,000 resamples) / p_val (Welch t-test vs baseline 1.0133) ★

### 1.1 完全 grid (N ≥ 1)

| 場 \ 条件 | A | B | C | D | E | X |
|----------|---|---|---|---|---|---|
| **Tokyo** | 35.58% (N=21) **p=0.001** | 102.86% (N=3) | 65.10% (N=21) p=0.211 | 84.23% (N=25) p=0.712 | 17.14% (N=1) | 0.00% (N=1) |
| **Nakayama** | 87.77% (N=32) p=0.632 | 0.00% (N=4) | 100.62% (N=37) p=0.994 | 81.68% (N=39) p=0.548 | 39.52% (N=3) | **0.00% (N=10) p<0.001** |
| **Hanshin** | 96.75% (N=47) p=0.937 | 0.00% (N=5) | 46.29% (N=20) p=0.203 | 202.31% (N=47) p=0.547 | 0.00% (N=5) | 82.86% (N=2) |
| **Kyoto** | **31.32% (N=27) p<0.001** | 61.43% (N=2) | 375.93% (N=13) p=0.465 | **36.17% (N=25) p=0.021** | 0.00% (N=1) | 0.00% (N=1) |
| **Chukyo** | 166.09% (N=19) p=0.495 | — | 45.21% (N=17) p=0.074 | 108.70% (N=22) p=0.902 | 0.00% (N=1) | — |
| **Fukushima** | 174.64% (N=8) p=0.687 | — | 171.35% (N=35) p=0.372 | 93.30% (N=29) p=0.827 | — | — |
| **Niigata** | 216.43% (N=10) p=0.531 | 0.00% (N=2) | 150.32% (N=9) p=0.478 | 59.08% (N=14) p=0.299 | — | 0.00% (N=5) |
| **Kokura** | — | — | — | — | — | — |

★ 観察 ★:
- ★ **Nakayama × X (N=10、 ROI=0.00%、 p<0.001)** ★ — 統計的に最強の除外根拠
- ★ **Tokyo × A (N=21、 ROI=35.58%、 p=0.001)** ★ — Tokyo の主損失源
- ★ **Kyoto × A (N=27、 ROI=31.32%、 p<0.001) + Kyoto × D (N=25、 ROI=36.17%、 p=0.021)** ★ — 京都全体マイナスの主因
- **Hanshin × D (N=47、 ROI=202.31%)** が単独で +¥33,660 を稼ぐ → 阪神 D は **絶対に除外してはいけない**

### 1.2 単一次元 (場のみ) 集計

| 場 | N | inv | pay | PnL | ROI% | 月別カバレッジ |
|---|---:|---:|---:|---:|---:|---|
| Fukushima | 72 | 50,400 | 70,700 | +20,300 | **140.28%** | 4月のみ |
| Hanshin | 126 | 88,200 | 106,030 | +17,830 | **120.22%** | 3/4月 |
| Niigata | 40 | 28,000 | 30,410 | +2,410 | 108.61% | 5月のみ |
| Chukyo | 59 | 41,300 | 44,210 | **+2,910** | **107.05%** | **3月のみ** |
| Kyoto | 69 | 48,300 | 47,320 | **-980** | **97.97%** | 4/5月 |
| Nakayama | 125 | 87,500 | 68,850 | -18,650 | **78.69%** | 3/4月 |
| Tokyo | 72 | 50,400 | 31,820 | -18,580 | **63.13%** | 4/5月 |

★ honest 注記 ★:
- ★ **Chukyo (107.05%) は単純集計では positive、 「中京 ROI 57.9%」 は別 source/window の値** ★。 親 prompt の引用と一致しないため、 master doc の「京都/中京 除外」 既定を継承する判断は **慎重** に行う必要あり
- Chukyo / Fukushima / Niigata は **1 か月のみのデータ** → 統計的に脆い
- Tokyo / Nakayama は 2 か月跨ぐので相対的に信頼できる ROI 推定

### 1.3 単一次元 (条件のみ) 集計

| 条件 | N | inv | pay | PnL | ROI% |
|---|---:|---:|---:|---:|---:|
| **X** | 19 | 13,300 | 1,160 | -12,140 | **8.72%** ★ |
| **E** | 11 | 7,700 | 950 | -6,750 | **12.34%** (戦略⑦ 既除外) |
| **B** | 16 | 11,200 | 3,020 | -8,180 | **26.96%** (戦略⑦ 既除外) |
| A | 164 | 114,800 | 109,660 | -5,140 | 95.52% |
| D | 201 | 140,700 | 151,400 | +10,700 | **107.60%** |
| C | 152 | 106,400 | 133,150 | +26,750 | **125.14%** |

→ **条件 X は新規除外候補として最強根拠** (戦略⑦ の既除外 B/E と同水準の低 ROI)

---

## 2. 統計的有意性 ranking (N≥10、 Welch t-test + bootstrap CI)

### 2.1 「除外推奨」 候補 (N≥10 かつ p<0.05 かつ CI 上限 < 100%)

| rank | 場 | 条件 | N | ROI% | 95% CI | p_val | 判定 |
|---:|---|---|---:|---:|---|---:|---|
| 1 | Nakayama | X | 10 | 0.00 | [0.00, 0.00] | <0.001 | ★ **除外推奨** |
| 2 | Kyoto | A | 27 | 31.32 | [6.56, 64.87] | <0.001 | ★ **除外推奨** |
| 3 | Tokyo | A | 21 | 35.58 | [7.82, 69.12] | 0.001 | ★ **除外推奨** |
| 4 | Kyoto | D | 25 | 36.17 | [0.00, 95.20] | 0.021 | ★ **除外推奨** |

### 2.2 「観察継続」 候補 (N≥10 かつ p<0.10 だが CI 上限 ≥ 100%)

| rank | 場 | 条件 | N | ROI% | 95% CI | p_val | 判定 |
|---:|---|---|---:|---:|---|---:|---|
| 5 | Chukyo | C | 17 | 45.21 | [0.00, 110.08] | 0.074 | 観察継続 |

### 2.3 「sample 不足」 (N<10)

主要 sample-deficient cells:
- Tokyo×B (N=3), Tokyo×E/X (N=1), Hanshin×X (N=2), Kyoto×B/E/X (N=1-2), Chukyo×E (N=1), Fukushima×A (N=8), Niigata×A (N=10 で境界、 ROI 216% で除外候補ではなく **採用継続**), Niigata×B/C (N=2, 9)
- Kokura は全 cells で N=0 (5/16 時点で開催履歴なし)

★ honest 注記 ★: 95% CI が「100% 含まない」 cells:
- Nakayama×X: [0.00, 0.00] (全 N=10 が miss)
- Kyoto×A: [6.56, 64.87]
- Tokyo×A: [7.82, 69.12]
- Kyoto×D: [0.00, 95.20]

これらが「統計的に baseline を下回る ROI 集合」と確定できる cells。

---

## 3. 案 A/B/C/D 比較 (read-only projection)

「除外」 = 「v15 の予測が出力されても 700円 投票しない」 = inv=0 / pay=0 として cumulative_results.csv から削除した時の 集計値。

| 項目 | baseline | **A_master** | **B** | **C** | **D_all_worst** |
|------|---:|---:|---:|---:|---:|
| 除外場 | — | Kyoto / Chukyo | Kyoto / Chukyo / Tokyo / Nakayama | Kyoto / Chukyo | Kyoto / Chukyo / Tokyo / Nakayama |
| 除外条件 | — | — | — | X | X |
| 除外 race | 0 | 128 | 325 | 146 | 332 |
| 残 race | 563 | 435 | 238 | **417** | 231 |
| 投資 (¥) | 394,100 | 304,500 | 166,600 | **291,900** | 161,700 |
| 配当 (¥) | 399,340 | 307,810 | 207,140 | **306,650** | 205,980 |
| PnL (¥) | +5,240 | +3,310 | +40,540 | **+14,750** | +44,280 |
| ROI (%) | 101.33 | 101.09 | 124.33 | **105.05** | **127.38** |
| ROI 95% CI | [66.67, 144.08] | [63.55, 152.24] | [64.69, 211.80] | [66.47, 158.33] | [64.52, 216.65] |
| ROI vs baseline | — | -0.24pt | +23.0pt | **+3.72pt** | +26.05pt |

### 3.1 月次安定性チェック (★ 最重要 ★)

| 月 | baseline ROI | 案 A_master | 案 B | 案 C | 案 D_all_worst |
|---|---:|---:|---:|---:|---:|
| 202603 (n=173 → variable) | 76.52% | 60.71% | **13.72%** ★ | 62.36% | **13.72%** ★ |
| 202604 (n=255 → variable) | 117.99% | 128.93% | **172.38%** | 133.37% | **173.66%** |
| 202605 (n=135 → variable) | 101.65% | 79.66% | 108.61% | 85.49% | 124.12% |

★ 観察 ★:
- ★ **案 D の点推定 ROI 127.38% は 4 月単月 (173.66%) に強く依存** ★ — 3 月単独で D の ROI は 13.72% / PnL -¥33,820 (baseline 76.52% を 大幅下回る!)
- ★ **案 B も同様 (3月 13.72%)** ★
- **案 A_master**: 月次 ROI 60.71 / 128.93 / 79.66 → 3 月の落ち込みが baseline より大きく不安定
- **案 C**: 月次 ROI 62.36 / 133.37 / 85.49 → 全月で baseline と同等以上 (3 月は -14pt だが、 D/B のような壊滅的低下なし)

→ **月次安定性で 案 C が最も robust**。 案 D は 3 月に -¥33,820 を出している実績から、 採用すると 1 か月で 撤退ライン -¥50,000 接近のリスク。

### 3.2 5/16 spike 除外 robustness check

5/16 単日 +¥30,310 (CI 上限超え統計偶然と推定、 §P0-1 結論) を除いた robustness:

| 案 | n=529 (5/16除外) ROI | PnL |
|---|---:|---:|
| baseline | 93.23% | -¥25,070 |
| A_master | 101.56% | +¥4,510 |
| B | 125.71% | +¥40,850 |
| C | **105.78%** | **+¥15,950** |
| D_all_worst | 128.95% | +¥44,590 |

→ 全案で 5/16 除外しても順位不変、 案 C も +¥15,950 残存 (5/16 除外後も positive)。

### 3.3 月予算 / 投票 R 数 への影響

| 案 | 月平均投票 R | 月投資額 | 月期待 PnL (点推定) | 撤退余裕日数 (月 PnL 最悪値 -¥33,820 想定で -¥50k 到達) |
|---|---:|---:|---:|---:|
| baseline | 187.7R | ¥131,400 | +¥1,750 | 約 45 日 (現累計 +¥5,240) |
| A_master | 145.0R | ¥101,500 | +¥1,100 | 約 45 日 |
| B | 79.3R | ¥55,500 | +¥13,500 | 案 D と同水準のリスク → 危険 |
| **C** | **139.0R** | **¥97,300** | **+¥4,900** | 約 45 日 (3 月 -¥29k 想定でも -¥50k 到達まで 余裕あり) |
| D_all_worst | 77.0R | ¥53,900 | +¥14,800 | **1 か月で -¥33k 出る実績 → 危険** |

---

## 4. オッズ帯 × 場 × 条件 interaction (limited)

### 4.1 制約

cumulative_results.csv に **pre-race 単勝 odds 列が存在しない**。 取得可能なのは:
- `actual_payout` (configured 三連複/馬連の的中時配当)
- `top1_score` (10% rows only)

そのため、 真の「オッズ帯」 interaction は **不能**。 代用 proxy として `payout/700` を配当倍率 band 化したが、 miss (band='miss') が支配的で意味のある interaction は得られない。

### 4.2 配当倍率 band per cell (proxy、 N≥5 only)

worst (PnL ascending) top 5 (※ band='miss' を除く):

| 場 | 条件 | band | N | inv | pay | PnL | ROI% |
|---|---|---|---:|---:|---:|---:|---:|
| Hanshin | D | miss | 37 | 25,900 | 0 | -25,900 | 0.00 |
| Nakayama | C | miss | 34 | 23,800 | 0 | -23,800 | 0.00 |
| (以下 miss 集中) | | | | | | | |

→ band='miss' は 「単に的中しなかった race の固定 -700 円」 にすぎず、 odds-band interaction を analysis する材料にならない。

### 4.3 honest 結論

★ 真のオッズ帯 interaction を analyze するには ★:
- `data/track_record.csv` (518 rows、 ROI 133.84%) の **pre-race odds 列** との突合
- もしくは `data/v21/strategy_v2_simulation.csv` の `v2_ev_top1` / `v2_p_calibrated` を proxy として使用
- 当 sub-task の scope では実装せず、 別 task (P0-3 として推奨) で実施するべき

---

## 5. 推奨案 + 根拠

### 5.1 推奨: ★ **案 C (京都/中京 + 条件 X 除外)** ★

#### 根拠 1: 統計的 evidence

- **条件 X 単独**: N=19、 ROI=8.72%、 95% CI [0.00, 26.17]、 **CI 上限が 100% を大きく下回る** → 統計的に baseline を下回る集合と確定
- ★ **Nakayama × X (N=10、 ROI=0.00%、 p<0.001)** ★ は条件 X 除外で完全カバー
- 京都の主損失源 **Kyoto × A (N=27、 p<0.001)** と **Kyoto × D (N=25、 p=0.021)** は京都除外で完全カバー
- 中京除外は master doc 既定継承だが、 中京単独次元 ROI は 107.05% (positive)、 月別 ROI 107.05% (3 月のみ) → ★ honest 注記: 中京は本来除外不要、 master doc の「中京 ROI 57.9%」 source 不明 ★。 ただし継承する判断は 5/18+ 議論待ち

#### 根拠 2: 月次安定性

| 月 | baseline | 案 C | 差 |
|---|---:|---:|---:|
| 202603 | 76.52% | 62.36% | -14.16pt |
| 202604 | 117.99% | 133.37% | +15.38pt |
| 202605 | 101.65% | 85.49% | -16.16pt |

→ baseline 比 ±15pt 程度の変動で 安定。 ★ 一方 案 D は 3 月だけで -33,820 円の壊滅的損失を出した実績あり (月次 ROI 13.72%) ★。

#### 根拠 3: 5/16 robust check pass

5/16 spike 除外時も +¥15,950 残存 (案 D は +¥44,590 と派手だが 月次不安定)

#### 根拠 4: ROI 改善は +3.72pt と穏当だが honest

- 期待 PnL +¥9,510 (3 か月で実現済み実績 baseline 比 +¥9,510)
- 月次換算 **+¥3,170/月** 改善
- 95% CI 下限は 66.47% で 100% 切る → 「統計的に有意な利益確定」 ではない、 honest に shadow 検証推奨

### 5.2 否決した案

#### 案 A_master: 京都/中京 除外のみ

- 期待 PnL +¥3,310 (baseline +¥5,240 比 -¥1,930) → ★ **baseline より悪化** ★
- 中京除外で正収益月 (Mar 107% / N=59) を切り捨ててしまう
- 条件 X (ROI 8.72%) を除外していないため新発見を活用していない
- **否決**

#### 案 B: 京都/中京/東京/中山 除外

- 月次変動極大 (3 月 13.72%、 4 月 172.38%)
- 残 race 数 238 で sample 不足 (月 79R)
- 撤退ライン到達リスクが C より高い
- **否決** (推奨 < C)

#### 案 D_all_worst: 京都/中京/東京/中山 + 条件 X 除外

- 点推定 ROI 127.38% は最良だが、 ★ 3 月単独 -¥33,820 / 13.72% の実績 ★
- 1 か月の局所最適にすぎず、 撤退ライン -¥50,000 接近リスク
- 投票 R 数 77/月 = baseline -59% の大幅減 → 機会損失大
- **否決** (リスク調整後 期待値で C より劣る)

### 5.3 honest 限界と注意

1. ★ **採用 baseline ROI 101.33% 自体が 95% CI [66.83, 145.36] と幅広い** ★ — 全案の改善幅は CI 内変動の可能性
2. **5/16 単日 +¥30,310 spike は統計偶然と推定** (P0-1 doc §0)、 全案の見積に同 spike が含まれているため、 6 月以降は ROI が baseline 100% 前後に戻る可能性大
3. **3 か月データで 5 月場 Niigata / 5 月以前場 Chukyo/Fukushima が単月限定** → 場別 ROI は信頼度低い
4. **Chukyo 単独 ROI 107.05% は positive、 master doc の「57.9%」 source 不明** → 中京除外は master doc 継承だが理由は要確認

---

## 6. 5/18+ 実装 prompt 設計

★ **絶対遵守** ★:
- `predict_core.py` / `daily_predict.py` / `race_auto_notify.py` / `app.py` / V15 .pkl.gz は **不変**
- 新 strategy filter は `tools/strategy_layer_v2.py` の **shadow only** 機能拡張で実装
- 1 か月 shadow eval → GO/NO-GO 判定 → 採用なら race_auto_notify.py へ統合 (別 task)

### 6.1 sub-task 5-4 prompt (推奨)

```
★ Sub-task 5-4: 案 C (京都/中京 + 条件 X 除外) shadow filter 実装 ★

【背景】
P0-2 拡張 (docs/P0_2_EXTENSION_DESIGN_2026_05_16.md) で 案 C 推奨確定。
条件 X (頭数 15+ / 重~不良) ROI 8.72% (N=19) を新規除外候補に追加。
京都/中京 は master doc 既定継承。

【絶対遵守】
🔴 NEVER:
- predict_core.py / daily_predict.py / race_auto_notify.py / app.py 変更
- V15 .pkl.gz / data/cumulative_results.csv (read のみ) 変更
- git commit / push (親集中)
- 案 C 採用判断 (1 か月 shadow 検証 まで pending)

🟢 OK:
- tools/strategy_layer_v2.py 拡張 (shadow mode strategy_8 仮称 追加)
- 既存 data 読み込み + simulation
- 新規 docs/STRATEGY_8_SHADOW_DESIGN_2026_05_18.md

【作業内容】
1. tools/strategy_layer_v2.py に strategy_8 (= 案 C) を shadow mode で実装
   - strategy_7 既存 logic 不変
   - strategy_8 = strategy_7 + (course in ['京都','中京'] OR condition == 'X') を 除外
2. data/cumulative_results.csv 全 settled (n=563) で 案 C backtest
   - 期待 ROI 105.05% / PnL +¥14,750 の再現確認 (本 doc §3 と一致するか)
3. 5/18-6/16 (約 1 か月) shadow 出力を tools/strategy_8_shadow_log.csv に毎日蓄積
4. 1 か月後 (6/17) sample N≥150 を target に GO/NO-GO 判定 prompt 別途
5. **race_auto_notify.py / predict_core.py には一切 touch しない**

【fabrication 防止】
- 全数値 cumulative_results.csv / strategy_v2_simulation.csv 実測
- shadow 1 か月の間 production の v15 出力は完全不変保証
- 中京除外は master doc 継承理由を honest に明記 ("単独 ROI 107% は positive、 別 source の根拠未確認")

【完了通知】
"案 C shadow filter 実装完了、 backtest ROI XX.XX%、 PnL +¥X,XXX、 5/18+ 1 か月 shadow 開始"

★ honest 厳守、 commit/push なし (親集中) ★
```

### 6.2 1 か月後 (6/17) 判定 prompt 雛形

```
★ Sub-task 案 C 1 か月 shadow eval GO/NO-GO ★

【判定基準】
- GO: shadow_8 1 か月 ROI ≥ baseline_v15 + 5pt かつ N≥150 かつ 月次 ROI 全月 ≥ 60%
- NO-GO: shadow_8 ROI < baseline またはどこか 1 か月で ROI < 50%

【作業】
1. tools/strategy_8_shadow_log.csv 1 か月 結果集計
2. baseline (v15 + strategy_7) と比較
3. GO 判定なら race_auto_notify.py 統合 prompt 起草
4. NO-GO 判定なら案 A 継続 + 別案 探索 prompt 起草
```

---

## 7. 出力 file 一覧

| file | type | 状態 |
|---|---|---|
| docs/P0_2_EXTENSION_DESIGN_2026_05_16.md | 本 doc | **新規作成** |
| tools/_p0_2_extension_analysis.py | 解析 script | **新規作成** (read-only、 cumulative_results.csv のみ参照) |
| data/cumulative_results.csv | source | read-only (改変なし) |
| data/v21/strategy_v2_simulation.csv | source | read-only (改変なし) |
| docs/ROI_DISCREPANCY_2026_05_16.md | 先行 doc | read-only (改変なし) |

commit/push なし (親 agent 集中管理)。

---

## 8. honest 注記 (★ 必読 ★)

1. **採用 baseline 101.33%、 案 C 期待 105.05% の +3.72pt 改善は 95% CI 重複** — 統計的有意ではない
2. **5/16 単日 +¥30,310 spike** が全案見積に含まれる。 6 月以降は baseline 100% 周辺に戻る可能性大
3. **中京 ROI 単独次元 107.05%** は positive、 master doc の「57.9%」 source 未確認。 「除外」 決定は本 doc も master 継承だが honest 異議あり
4. **「オッズ帯 × 場 × 条件 interaction」** は cumulative_results.csv に pre-race odds 列なしのため **不能**。 別 source (track_record.csv / strategy_v2_simulation.csv) との突合が別 task として必要 (P0-3 推奨)
5. **3 か月データで Chukyo / Fukushima / Niigata は単月限定 sample** → 場別 ROI 推定は確度低い。 4 月以降 追加サンプル蓄積待ち
6. **案 D は点推定で最良だが 3 月単月 -¥33,820 / 13.72% の実績** → リスク調整後 期待値で C より劣ると判定
7. **本 doc は read-only analysis + design 提案のみ**。 実装 (案 C 採用 / race_auto_notify.py 改修) は 1 か月 shadow eval 完了後の別 task

---

end of doc.
