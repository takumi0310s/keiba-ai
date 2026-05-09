# Session #89 A: 5/10 (日) 開催 audit

Session #89 (2026-05-09 23:30 過ぎ)。

## 5/10 (日) 中央開催 (5/9 と 同 開催地 継続 想定)

| 開催地 | 開催コード | 5/9 (土) 確認 | 5/10 (日) 想定 |
|--------|----------|--------------|---------------|
| 京都 | 08 (8 回 3 日) | ✓ ([pre_race_predict_5_9_R10_京都_202608030510.json](pre_race_predict_5_9_R10_京都_202608030510.json) 等) | ✓ 8 回 4 日 (継続) |
| 東京 | 05 (5 回 2 日) | ✓ ([pre_race_predict_5_9_R10_東京_202605020510.json](pre_race_predict_5_9_R10_東京_202605020510.json) 等) | ✓ 5 回 3 日 (継続) |
| 新潟 | 04 (4 回 1 日) | ✓ ([pre_race_predict_5_9_R10_新潟_202604010310.json](pre_race_predict_5_9_R10_新潟_202604010310.json) 等) | ✓ 4 回 2 日 (継続) |

3 開催地 × 12 R = **36 R / 日 想定**。

## 5/10 (日) 重賞 (G1/G2/G3) 推定

JRA 過去 pattern + 開催 schedule から 推定 (確定は当日 daily_predict 出力):

| 場 | R | 推定 重賞 | grade | 備考 |
|---|---|----------|-------|------|
| 東京 | 11R | NHK マイルカップ | G1 | 5月 第 2 日曜 定番 |
| 京都 | 11R | 京都新聞杯 | G2 | 5月 第 2 土日 定番 |
| 新潟 | 11R | 新潟大賞典 | G3 | 5月 上旬 定番 |

★ ただし 開催年により 微調整あり。 14:00 投票候補確定通知 で 確認推奨。

## 案B改 strict 対象 (12R 1 勝クラス、 重賞除く)

|  場 | 12R 想定 | 案B改 strict 候補 |
|----|---------|-------------------|
| 京都 12R | 1 勝クラス (1600m or 1400m 想定) | ★ 候補 |
| 東京 12R | 1 勝クラス (1400m or 1600m 想定) | ★ 候補 |
| 新潟 12R | 1 勝クラス (1200m or 1400m 想定) | ★ 候補 |

→ **max 3 R 候補** ([session_89_5_10_candidates.md](session_89_5_10_candidates.md))

## 関連 doc

- [session_77_5_10_verification.md](session_77_5_10_verification.md) — 5/10 朝 fire schedule 動作保証
- [session_89_5_10_candidates.md](session_89_5_10_candidates.md) — 候補 R 推定
- [session_89_expected_roi.md](session_89_expected_roi.md) — ROI 期待値

## V15 投資保護

5/10 朝 fire schedule (Session #77 で 動作保証済):
- 06:30 Keiba-Morning_Sun
- 07:00 Keiba-MorningDigest
- 08:00 keiba-ai\DailyPredict ★最重要★
- 08:45 Keiba-RaceAutoNotify (土日 fire)
- 09:30 Keiba-SaveAllHorseScores_0930
- 09:30 Keiba-MorningWeightCheck_Sun
- 18:00 Keiba-RaceDayReport_Sun
- 20:00 keiba-ai\DailyResultsEvening

V15 model 完全不変、 累計 +¥14,140 維持、 撤退余裕 +¥64,140。
