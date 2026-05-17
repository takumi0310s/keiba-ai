# data-audit-1: 朝 8:00 V15 予測 record 完全性 audit (2026-05-17)

> 範囲: 2026-04-01 〜 2026-05-17 (47 日)
> 対象: `data/daily_predictions/*.csv` (中央)、 `nar_*.csv` (地方)、 `data/daily_predictions_full/*.csv` (全頭 score)
> read-only audit。 V15 / predict_core / app.py 完全不変。

---

## 0. 結論

| 項目 | 値 |
|------|----|
| 真の record 完全性 | **❌ 不完全 (中央 weekend 4 日 欠落)** |
| 期待 日数 (4/1-5/17) | 47 日 |
| 任意 file 存在 日数 | 10 日 (中央) + 8 日 (NAR、 5/10 以降のみ) |
| 中央 weekend 期待 | 14 日 |
| 中央 weekend 実在 | 10 日 |
| **中央 weekend 欠落** | **4 日: 4/4, 4/5, 5/2, 5/3** |
| 平日 (中央 R なし) | 33 日 → file 不在は正常 |
| 重複 R | **15 R (4/26 のみ、 morning+partial 二重保存)** |
| 全頭 score (`daily_predictions_full`) | **3 日のみ (5/10, 5/16, 5/17)** |

---

## 1. file 一覧 (中央 daily_predictions、 4/1-5/17 範囲内)

| date | weekday | rows | unique_race_id | score_col | null_score | cols |
|------|---------|-----:|---------------:|-----------|-----------:|-----:|
| 20260411 | SAT | 35 | 35 | top1_score | 0 | 19 |
| 20260412 | SUN | 35 | 35 | top1_score | 0 | 19 |
| 20260418 | SAT | 35 | 35 | top1_score | 0 | 19 |
| 20260419 | SUN | 35 | 35 | top1_score | 0 | 19 |
| 20260425 | SAT | 35 | 35 | top1_score | 0 | 19 |
| 20260426 | SUN | **50** | **35** | top1_score | 0 | 19 |
| 20260509 | SAT | 34 | 34 | top1_score | 0 | 19 |
| 20260510 | SUN | 35 | 35 | top1_score | 0 | 19 |
| 20260516 | SAT | 35 | 35 | top1_score | 0 | 19 |
| 20260517 | SUN | 34 | 34 | top1_score | 0 | 19 |

注: 範囲外 (参考) — 20260314 (SAT, 36R), 20260315 (SUN, 35R), 20260321 (SAT, 35R)。

### 補助 file
- `20260418_prerace.csv` 存在 (pre-race snapshot、 audit 対象外)
- `20260426.csv.bak_morning_1414` / `20260426.csv.bak_partial_0854` (バックアップ 2 件、 audit 対象外)

---

## 2. 欠落 date

### 全 日 (47 日 expected、 中央 file 不在 = 37 日)
平日 33 日 + 中央 weekend 欠落 4 日 = 37 日。

### 中央 weekend 欠落 (4 日)
| date | weekday | 推定原因 |
|------|---------|---------|
| 20260404 | SAT | V15 投入 (2026-04-01) 直後の運用立ち上げ 期間、 daily_predict 未稼働の可能性 |
| 20260405 | SUN | 同上 |
| 20260502 | SAT | GW 期間、 SCRAPER-GUARD or daily_predict 未稼働 |
| 20260503 | SUN | GW 期間、 同上 |

注: 推定原因は file 不在のみから逆算した推定。 daily_predict ログ / Discord notification 履歴の照合は未実施。

### 平日 (NAR file 確認)
NAR file 存在: 5/10 - 5/17 のみ (8 日)。 4/1 - 5/9 の平日 NAR は file 不在。

| date | NAR file |
|------|---------|
| 20260510 (SUN) | ✓ |
| 20260511 (MON) | ✓ |
| 20260512 (TUE) | ✓ |
| 20260513 (WED) | ✓ |
| 20260514 (THU) | ✓ |
| 20260515 (FRI) | ✓ |
| 20260516 (SAT) | ✓ |
| 20260517 (SUN) | ✓ |

---

## 3. 各 weekend day R 数 (中央)

| date | weekday | 想定 R (~35) | 実 record | unique R | 重複 | missing |
|------|---------|------------:|----------:|---------:|-----:|--------:|
| 20260404 | SAT | ~36 | 0 | 0 | 0 | **~36** |
| 20260405 | SUN | ~36 | 0 | 0 | 0 | **~36** |
| 20260411 | SAT | ~35 | 35 | 35 | 0 | 0 |
| 20260412 | SUN | ~35 | 35 | 35 | 0 | 0 |
| 20260418 | SAT | ~35 | 35 | 35 | 0 | 0 |
| 20260419 | SUN | ~35 | 35 | 35 | 0 | 0 |
| 20260425 | SAT | ~35 | 35 | 35 | 0 | 0 |
| 20260426 | SUN | ~35 | **50** | 35 | **15** | 0 |
| 20260502 | SAT | ~36 | 0 | 0 | 0 | **~36** |
| 20260503 | SUN | ~36 | 0 | 0 | 0 | **~36** |
| 20260509 | SAT | ~34 | 34 | 34 | 0 | 0 |
| 20260510 | SUN | ~35 | 35 | 35 | 0 | 0 |
| 20260516 | SAT | ~35 | 35 | 35 | 0 | 0 |
| 20260517 | SUN | ~34 | 34 | 34 | 0 | 0 |

**合計 missing R (推定): ~144 R** (4/4, 4/5, 5/2, 5/3 各 ~36R)。

### 4/26 重複 detail
- 15 race_id が 2 行ずつ存在 (rows 50 / unique 35)
- 重複 race_id: 202605020201-04 (東京、 4 件) + 202608030201-12 (京都、 11 件)
- top1_score が 微差 (例: 202608030201 で 0.332565 vs 0.332268) → morning と partial の二重 保存
- バックアップ file (`bak_morning_1414`, `bak_partial_0854`) も同 dir 残存 → 該当 race は **取消発生 → 1 レース 再予測** の痕跡

---

## 4. daily_predictions_full の状況

| date | rows | unique_race_id | cols | 充足 |
|------|-----:|---------------:|-----:|------|
| 20260510 (SUN) | 489 | 35 | 17 | ✓ |
| 20260516 (SAT) | 487 | 35 | 17 | ✓ |
| 20260517 (SUN) | 456 | 34 | 17 | ✓ |

- **5/10 開始** (= 全頭 score 機能 投入日と推定)
- 4/1 - 5/9 期間は全頭 score 不在 = TOP3 のみの簡易 record
- 5/10 以降の weekend 3 日は完全 (中央 全 R × 全頭)

---

## 5. verdict + 5/18+ 仕組み提案

### verdict: ❌ 不完全
- 47 日中 file 存在 10 日 (中央)、 期待 14 weekend のうち **4 weekend 欠落 (推定 missing ~144 R)**
- daily_predictions_full は 5/10 開始のため、 4/1-5/9 の 8 weekend (推定 ~280 R) で全頭 score 不在
- 4/26 の 15 R 重複 (取消 再予測の痕跡、 file 整理不足)

### 5/18+ 仕組み提案 (read-only 範囲、 親集中 commit 前提)
1. **欠落 weekend の root cause 調査**: daily_predict.log / Discord 通知履歴を 4/4, 4/5, 5/2, 5/3 で照合し、 SCRAPER-GUARD 誤停止 / 起動失敗 / 手動停止のいずれか特定
2. **daily_predictions_full の遡及生成**: V15 + 4/11-5/9 の race 出馬表 + premium cache から、 過去 weekend の全頭 score を 後追い生成 (predict_core 不変 / dry-run)
3. **4/26 重複 cleanup**: morning vs partial を file 命名で区別 (現在の 1 file 上書き設計の改善)
4. **欠落 detect の自動化**: 毎晩 23:00 nightly_sanity で「翌日 weekend かつ daily_predict 未稼働」を Discord 警告

---

## 6. V15 production 不変保証 ✓

- read-only audit のみ実施
- predict_core.py / daily_predict.py / app.py / model file 一切 触れていない
- v15.2 training (PID 23528) 完全 不変
- 新規 file は本 docs (`docs/DATA_AUDIT_1_MORNING_PREDICTION_2026_05_17.md`) 1 件のみ

---

## 付録: fabrication 防止メモ
- 全 row 数 / unique race_id 数は `data/daily_predictions/*.csv` の pandas 実測値
- 中央 weekend 想定 R 数 (~34-36) は present file の実測値からのレンジ。 4/4/4/5/5/2/5/3 の正確な開催 R 数は本 audit で確認していない (推定 ~36R)
- 「推定原因」 (運用立ち上げ / GW) は file 不在からの状況推定であり、 ログ照合 未実施
