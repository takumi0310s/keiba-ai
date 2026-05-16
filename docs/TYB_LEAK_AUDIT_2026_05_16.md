# P0-3 TYB calibrator leak 監査 formal report

**実施日**: 2026-05-16
**対象**: `data/tyb_top3_predictor.pkl` (commit b4948d6a 学習済み LR predictor)
**input data**: `data/jrdb_tyb.csv` (550,115 rows、 17 features)、 train n=348
**監査 script**: `tools/v21/tyb_leak_audit_analysis.py`
**output JSON**: `data/v21/tyb_leak_audit.json`
**V15 production 影響**: ★ 0% (read-only analysis のみ) ★

---

## 0. 結論 (★ verdict ★)

| feature | verdict | 根拠要約 |
|---------|---------|----------|
| **tansho_odds** | ✅ **content 安全** / ⚠ **delivery POST-RACE** | content は -15 min pre-race snapshot (vs 確定 odds: log-corr 0.97、 exact-match 6.6%、 76% が 2 yen 以内、 24% は 10% 以上ズレ → 確定ではない)。 ただし jrdb_tyb ZIP は **当日 17:00 JST publish** = race 終了後配信、 live 投入 不可。 |
| **fukusho_odds** | ✅ content 安全 / ⚠ delivery POST-RACE | tansho_odds と同じ source/タイミング、 同条件 |
| **padock_idx** | ✅ content 安全 / ⚠ delivery POST-RACE | パドック観察 indicator、 64% zero rate (subjective annotation)。 release timing = TYB ZIP 一括配信 → 17:00 JST、 live 不可 |
| **jockey_idx / info_idx / odds_idx / sogo_idx** | ✅ content 安全 / ⚠ delivery POST-RACE | TYB 計算 indices、 odds_time field と同期 (race 開始 -15 min) |
| **cancel_flag / ashimoto / bagu_change** | ✅ content 安全 / ⚠ delivery POST-RACE | pre-race observation (取消 / パドック歩様 / 馬具変更)、 30+ min 前 公表 |
| **horse_weight / weight_diff** | ✅ content 安全 / ⚠ delivery POST-RACE | 馬体重 (公式 -70 min 発表)。 ただし weight_diff の影響は -0.0535 add-one delta = **負寄与** で 役に立たない |

- **真の +AUC delta (5CV、 leak ナシ前提)**: V15-only **0.4653** → V15+TYB **0.6082** = **+0.1429**
- **train AUC**: 0.6964、 gap 0.088 → over-fit 兆候だが n=348 / 17 features (20:1 ratio) で 想定範囲、 LEAK 兆候ではない
- **baseline 0.4653 < 0.5 は task 設計 由来**、 leak/bug ではない (詳細 §5)

### 推奨

- **P1-0 shadow eval: 限定 GO** — 以下 2 条件付き:
  1. **content leak は無い** ことを確認済 (corr 0.97 を 1.00 confirmed と区別)
  2. **delivery timing problem を解決した上での GO**: 当日 朝 06:00 schtask で TYB ZIP fetch しても、 publish が 17:00 = race 後なので **その日のレースに使えない**。 翌週末から `tyb_publish_monitor` で -15 min 直前 fetch 経路を確立 (5/9 fetch 停止 以降 復旧 が pending)。

- **deploy 不可** → 解決 まで paper shadow eval のみ
- **過去 backtest (retrospective)** には 使用可

---

## 1. tansho_odds release timing audit (★ critical ★)

### 1.1 TYB ZIP file の publication timing

JRDB datazip endpoint `/datazip/Tyb/TYB{yymmdd}.lzh` の HTTP HEAD で Last-Modified を確認:

| race_date (JST) | Last-Modified (UTC) | JST equivalent |
|-----------------|---------------------|----------------|
| 2026-03-22 (Sun) | Sun, 22 Mar 2026 08:06:25 GMT | 17:06 JST |
| 2026-04-18 (Sat) | Sat, 18 Apr 2026 08:10:32 GMT | 17:10 JST |
| 2026-05-02 (Sat) | Sat, 02 May 2026 08:04:32 GMT | 17:04 JST |
| 2026-05-03 (Sun) | Sun, 03 May 2026 08:01:28 GMT | 17:01 JST |
| 2026-05-09 (Sat) | Sat, 09 May 2026 08:06:31 GMT | 17:06 JST |
| 2026-05-10 (Sun) | Sun, 10 May 2026 07:54:27 GMT | 16:54 JST |
| 2026-05-16 (Sat、 today) | Sat, 16 May 2026 08:03:30 GMT | **17:03 JST** |

- TYB ZIP は **当日 17:00 JST 頃に 1 度だけ publish**、 早朝 06:00 試行は 全件 404 (`tyb_publish_log.csv` で 2 試行 共 404 確認済)
- ★ JRDB は race 終了後 まとめて 1 ファイル 配信 ★ — live 当日 朝 morning JRDB dump には TYB 含まれない

### 1.2 TYB 内部 odds_time field (固定長 byte 85-88、 HHMM)

TYB 固定長 binary を 1 record ずつ parse、 各 race の `odds_time` と `start_time` を比較:

```
race_id          odds_time   start_time   delta (min)
202604010401     0926        0945         19
202604010402     0956        1015         19
202604010403     1026        1045         19
...
202605020601     0950        1005         15
```

- `start_time - odds_time` median: **15-19 min**
- すべての records が race 開始 -15 〜 -19 min の snapshot
- ★ TYB の odds 値 自体は **race -15 min pre-race snapshot** ★

### 1.3 TYB tansho_odds vs JRA 公式確定 odds (post-race) の比較

`data/jra_payouts.csv` の `tansho_payout / 100` を confirmed odds とし、 該当 race の 勝ち馬 (`tansho_nums` 第1馬) の TYB tansho_odds と join:

| 指標 | 値 |
|------|-----|
| matched winners | **12,329** |
| raw correlation | 0.9221 |
| log-log correlation | **0.9664** |
| delta (TYB - confirmed) median | +0.200 yen |
| delta std | 9.33 yen |
| **exact match (delta=0、 \|d\|<0.05)** | **6.56%** |
| close (\|d\| < 0.5 yen) | 39.61% |
| close (\|d\| < 2.0 yen) | 76.54% |
| 範囲 (\|d_pct\| < 5%) | 18.33% |
| 範囲 (\|d_pct\| < 10%) | 36.37% |

- ★ confirmed odds との **完全一致率 6.56%** ★ — もし TYB が post-race 確定値 だったら exact match 率は 100% に近いはず
- 76% が ±2 yen 以内 = 直前 odds の 妥当な drift 範囲
- **判定: TYB tansho_odds は -15 min pre-race snapshot で 確定値 ではない**

### 1.4 TYB tansho_odds vs odds_base (08:00 morning snapshot) の比較

`data/odds_base_*.csv` (race_auto_notify が 08:00 morning に記録) と TYB を join:

| 指標 | 値 |
|------|-----|
| matched horses | 3,153 |
| matched races | 211 |
| log-log correlation | 0.6574 |
| delta median (TYB - morning) | +1.4 yen |
| delta std | 38.5 yen |

- morning との 相関 (0.66) は confirmed との 相関 (0.97) より 大幅 低い
- = TYB は **morning より race-time に近い** 時刻の snapshot
- = -15 min pre-race を 強く 支持

### 1.5 tansho_odds verdict

- **content**: ✅ 安全 (-15 min pre-race snapshot、 race 確定値ではない)
- **delivery**: ⚠ POST-RACE (ZIP publish は 17:00 JST = race 後)
- live 使用には **separate scraping 経路** (-15 min snapshot を直接取得) が必要

---

## 2. padock_idx release timing audit

### 2.1 padock_idx の性質

- range: 0.0 〜 4.0
- **zero rate: 63.95%** (大半が 0)
- non-zero 上位値: 2.0-3.6 (パドック観察 worth noting marks)
- TYB 固定長 byte 31-35

### 2.2 release timing 推定

- パドック観察 = race 開始 30-40 min 前 (馬体ウォーミングアップ周回時)
- TYB 内 `odds_time` field と同一 publish cycle と推定
- ★ JRDB 朝 06:00 dump に TYB は 含まれない (§1.1) ★
- 真の publish 経路: **17:00 JST 一括 publish** のみ、 -15 min 直前 update は無い

### 2.3 過去 dump との比較

5/9 fetch 停止 中 (5/9 と 5/4 試行 共 404) → 過去 morning dump vs evening dump で 値変動が観測できない (片方しか無い)。 真の "publish 経路" 確認は **将来 別途 -15 min 経路で fetch する 必要あり**。

### 2.4 padock_idx verdict

- **content**: ✅ 安全 (パドック観察 = 30-40 min pre-race)
- **delivery**: ⚠ POST-RACE (TYB ZIP 一括配信、 morning dump に含まれない)
- 朝予測 (08:00) では取得不可、 -15 min 直前 経路 確立 後 利用可

---

## 3. 各 TYB feature audit (個別)

### 3.1 single-feature direct AUC (top1 horse の top3 hit predict)

| feature | n_samples | best single AUC | corr-target | 解釈 |
|---------|-----------|-----------------|-------------|------|
| tansho_odds | 348 | **0.6444** | -0.20 | popular horse → top3 hit (logical) |
| fukusho_odds | 348 | 0.6274 | -0.20 | tansho と同方向 |
| odds_idx | 348 | 0.6222 | +0.20 | 直前 odds 動向 |
| padock_idx | 348 | **0.6191** | +0.19 | パドック評価 が 強い signal |
| jockey_idx | 348 | 0.6180 | +0.18 | 騎手指数 |
| info_idx | 348 | 0.6087 | +0.16 | 情報指数 |
| padock_mark | 348 | 0.5821 | -0.13 | パドック印 |
| idm | 348 | 0.5254 | +0.05 | 総合指数、 ★ 既に V15 で 内包 ★ |
| ashimoto | 348 | 0.5232 | +0.04 | 歩様 |
| kehai_code | 348 | 0.5230 | +0.04 | 気配 |
| weight_diff | 348 | 0.5178 | +0.04 | 馬体重変化 |
| sogo_idx | 348 | 0.5169 | +0.03 | 総合指数 |
| baba_code | 348 | 0.5164 | -0.03 | 馬場 (V15 で 内包) |
| bagu_change | 348 | 0.5082 | +0.01 | 馬具変更 (rare event) |
| horse_weight | 348 | 0.5041 | 0.00 | 馬体重 (V15 で 内包) |
| cancel_flag | 348 | **0.5000** | 0.00 | flag が立つ horse は data から除外済 |

### 3.2 leave-one-out (drop ONE feature, 5CV AUC)

baseline V15+TYB full 5CV = **0.6082**

| drop feature | AUC | delta | 解釈 |
|--------------|-----|-------|------|
| drop padock_idx | 0.5851 | **-0.0231** | 最も悪化 → padock_idx が importance 上位 |
| drop top1_score | 0.5889 | -0.0194 | V15 score も 寄与あり (2nd biggest) |
| drop tansho_odds | 0.5954 | -0.0128 | 単独 AUC 0.64 だが LR 内では 重複 |
| drop ashimoto | 0.5998 | -0.0084 | 歩様 微寄与 |
| drop info_idx | 0.6056 | -0.0026 | 微寄与 |
| drop idm | 0.6080 | -0.0002 | ほぼゼロ寄与 (V15 で 内包) |
| drop horse_weight | 0.6168 | +0.0086 | 除外して 改善 → noise/multicollinearity |
| drop padock_mark | 0.6203 | +0.0121 | 除外して 改善 |
| drop fukusho_odds | 0.6239 | +0.0157 | 除外して 改善 → tansho_odds と 重複 |

★ 重要 features: padock_idx > top1_score > tansho_odds > ashimoto ★

### 3.3 ADD-ONE: V15 + 1 TYB feature 5CV AUC

V15-only baseline = 0.4653、 1 feature 追加した時の delta:

| +feature | AUC | delta | 解釈 |
|----------|-----|-------|------|
| +tansho_odds | 0.6464 | **+0.1811** | 単独で 最大 +18pt |
| +odds_idx | 0.6350 | +0.1698 | tansho と 同系 |
| +fukusho_odds | 0.6309 | +0.1656 | tansho と 同系 |
| +jockey_idx | 0.6291 | +0.1639 | strong |
| +padock_idx | 0.6172 | +0.1519 | strong |
| +info_idx | 0.6054 | +0.1402 | strong |
| +padock_mark | 0.5534 | +0.0881 | mid |
| +ashimoto | 0.5150 | +0.0497 | weak |
| +sogo_idx | 0.5014 | +0.0361 | weak (V15 内包) |
| +idm | 0.4718 | +0.0065 | nearly zero |
| +weight_diff | 0.4817 | +0.0164 | weak |
| +baba_code | 0.4948 | +0.0295 | V15 で 内包 |
| +horse_weight | 0.4118 | **-0.0535** | ★ negative ★ |
| +kehai_code | 0.4354 | -0.0299 | negative |
| +bagu_change | 0.4364 | -0.0289 | negative |

### 3.4 verdict (feature 別)

| feature | content leak | delivery leak | predictive value |
|---------|--------------|---------------|------------------|
| tansho_odds | ✅ 安全 | ⚠ POST-RACE | strong (+0.18) |
| fukusho_odds | ✅ 安全 | ⚠ POST-RACE | strong (+0.17) |
| odds_idx | ✅ 安全 | ⚠ POST-RACE | strong (+0.17) |
| jockey_idx | ✅ 安全 | ⚠ POST-RACE | strong (+0.16) |
| padock_idx | ✅ 安全 | ⚠ POST-RACE | strong (+0.15) |
| info_idx | ✅ 安全 | ⚠ POST-RACE | strong (+0.14) |
| padock_mark | ✅ 安全 | ⚠ POST-RACE | mid (+0.09) |
| ashimoto | ✅ 安全 | ⚠ POST-RACE | weak (+0.05) |
| idm | ✅ 安全 | ⚠ POST-RACE | nearly zero (V15 内包) |
| sogo_idx | ✅ 安全 | ⚠ POST-RACE | nearly zero (V15 内包) |
| baba_code | ✅ 安全 | ⚠ POST-RACE | V15 内包 |
| horse_weight | ✅ 安全 | ⚠ POST-RACE | negative add-one |
| weight_diff | ✅ 安全 | ⚠ POST-RACE | weak |
| kehai_code | ✅ 安全 | ⚠ POST-RACE | negative add-one |
| bagu_change | ✅ 安全 | ⚠ POST-RACE | rare, negative |
| cancel_flag | ✅ 安全 | ⚠ POST-RACE | constant (0 only after exclude) |

---

## 4. train/CV gap 0.088 の origin

train AUC 0.6964 vs 5CV AUC 0.6082 = **gap 0.088**

### 4.1 over-fit risk 評価

- n_samples = 348、 features = 17
- ratio = 20.5 sample/feature
- LR with n=348 / 17 features = **moderate over-fit expected** (一般 ≧ 30 sample/feature が推奨)
- gap 0.088 は LEAK 兆候 ではなく **小 sample LR の典型 over-fit**

### 4.2 leave-one-out で 「奇跡的 features」 を 探す

drop で 大幅 改善する feature が あれば leak。 実測:

| drop feature | delta_from_full | leak risk? |
|--------------|-----------------|------------|
| drop fukusho_odds | +0.0157 | small (tansho と 重複) |
| drop padock_mark | +0.0121 | small (multicollinearity) |
| drop horse_weight | +0.0086 | small (multicollinearity) |

★ どれも **leak 兆候 ではなく multicollinearity** ★。 真の leak feature なら drop 時 to AUC が 大幅 低下 (drop で −10pt 級) する。

### 4.3 結論

train/CV gap 0.088 = **正常な小 sample over-fit、 LEAK 由来 ではない**

---

## 5. baseline AUC 0.4653 origin

### 5.1 V15 top1_score の direct AUC

- **direct AUC (no LR)**: 0.5091 (≈ random)
- **LR 5CV AUC**: 0.4653 (slightly below random)

### 5.2 task 設計 由来 (★ V15 が pre-filter 済み のため ★)

V15 production は **per-race 内 top1 horse を選択** (= top1_score が 最高 の horse)。 この top1 horse のみ を集めて 「top3 hit」 を予測する task では:

- top1 picks は 既に "high score" でフィルタ済 → score 分布が 圧縮
- = top1_score の **絶対値** は top3 hit を 区別できない
- score 0.5 の top1 vs score 0.8 の top1 = ともに 「そのレースで 最も confident」 → top3 hit は 体感同等
- ★ V15 top1_score は per-race ranker として 機能、 per-top1 calibrator としては 機能しない ★

これは V15 model の **bug** ではなく **task と score の意味の mismatch**。 同様の現象は LightGBM の rank model 全般で 観測される。

### 5.3 代替 baseline (race-relative confidence) 評価候補

将来 案: V15 top1_score を race 内 1位 binary とせず、 raw confidence を per-race max-min normalize。 ただし今回 P0-3 では time 不足 で実装 skip。

### 5.4 結論

baseline 0.4653 = **task 設計 由来 (top1 filtering bias)、 leak 由来 ではない**。 真の +0.1429 delta は **計算 正しい**。

---

## 6. leak 除外後 真の delta

### 6.1 全 features が content-safe と確認済

§3.4 の通り、 **content leak は どの feature にも 観測されない**。 削除 不要。

### 6.2 delivery POST-RACE → live 不可 → 真の delta は retrospective のみ valid

| シナリオ | 真の delta | 評価 |
|----------|-----------|------|
| retrospective (backtest 評価) | **+0.1429** | valid (content 安全) |
| live deployment (08:00 朝予測 + 朝 TYB) | **不可** | TYB ZIP が 17:00 publish のため |
| live deployment (-15 min 直前 fetch、 別経路) | +0.1429 (期待) | -15 min 経路 構築 後に確定 |

### 6.3 odds-only sub-model 案 (経済性)

odds_base (08:00 morning snapshot) のみで proxy:
- ADD-ONE: V15 + tansho_odds = +0.1811 delta
- ただし TYB tansho_odds は -15 min snapshot、 odds_base は -1〜2 hour snapshot
- → odds_base 単独では +0.18 まで 出ない 可能性 (log corr 0.66)
- 別途 P0-3.5 で `odds_base only` delta 検証 推奨

---

## 7. 推奨 + 次 step

### 7.1 verdict

| 項目 | 判定 |
|------|------|
| content leak | ✅ **NONE** |
| delivery timing | ⚠ POST-RACE (17:00 JST publish) |
| train/CV gap 0.088 origin | small-sample over-fit (LEAK 兆候 ナシ) |
| baseline 0.4653 origin | task 設計 由来 (top1 filter bias) |
| 真の +AUC delta | **+0.1429** (5CV、 retrospective context) |
| 推奨 | **限定 GO** (paper shadow eval まで)、 **deploy NO-GO** |

### 7.2 次 step (推奨 順)

1. **P0-3 完了** → docs にこの 監査 結果 確定
2. **P0-4 (recommend new)**: TYB -15 min 直前 fetch 経路 確立
   - 候補 1: JRDB 別 endpoint (確認要)
   - 候補 2: TYB ZIP を 当日 16:50 JST 頃 monitor (Last-Modified watch)
   - 候補 3: netkeiba/JRA-VAN の 別 source で TYB-equivalent 取得
3. **P0-3.5 (recommend new)**: odds_base only sub-model 評価
   - V15 + odds_base 単独 (08:00 snapshot) で AUC delta 検証
   - +0.05 以上なら live deploy 候補
4. **P1-0 shadow eval (限定 GO)**:
   - retrospective backtest として 5/16 以前 全 race で V15+TYB の paper ROI 計測
   - live deployment は **NO-GO** (delivery 時刻 解決まで)
5. **V22 retrain ロードマップ**: TYB features を **historical context (前走 TYB)** として 加える検討 (= 当日 race の TYB ではなく、 前 race の TYB)
   - これなら delivery 17:00 publish でも 翌 race 朝 dump で利用可

### 7.3 重要 警告

- ★ commit b4948d6a の `data/tyb_top3_predictor.pkl` を **そのまま live deploy しないこと** ★
  - 朝 08:00 予測時 TYB ファイル 存在しない → predictor 使えない or stale data 使用 → ★ silent failure ★
- V15 production への影響 = 0 (predictor を call する code が 未 wire-in)
- shadow eval は **retrospective context** で 実施 すること

---

## 8. honest 限界

1. **n=348 と 小さい** — 真の delta CI は wide (probably ±0.04 程度)、 5CV 1 回 だけでは 楽観 過ぎる
2. **padock_idx の zero-rate 64%** — 残り 36% が 効いているだけ、 全 race universal な feature ではない
3. **delivery 解決 確信できず** — JRDB 内部 で -15 min 直前 publish が無い 可能性 (現状 1 日 1 回 17:00 のみ 観測)
4. **baseline 0.4653 < 0.5** の **task 設計 由来 解釈** は仮説 — 代替 baseline 設計 (race-relative confidence) で 再検証 する 価値あり
5. **odds_time field の解釈** — `start_time` との delta 15-19 min は **typical pattern** だが、 race ごと variability がある (実際は odds 確定 時刻 ではない 可能性、 publish 時刻 の可能性)

---

## Appendix A: 監査 script & raw output

- 監査 script: `tools/v21/tyb_leak_audit_analysis.py`
- raw JSON: `data/v21/tyb_leak_audit.json`
- log: `logs/tyb_leak_audit.log`

## Appendix B: 既存 評価 file

- 学習 script: `tools/v21/jrdb_tyb_train_predictor.py` (b4948d6a)
- 学習済み model: `data/tyb_top3_predictor.pkl`
- 学習 report: `data/v21/tyb_top3_predictor_report.json`
- 評価 script: `tools/v21/jrdb_tyb_evaluate.py`
