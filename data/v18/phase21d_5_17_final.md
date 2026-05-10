# Phase 21D - 5/17 (土) final schedule

> 作成: 2026-05-11 (Phase 21D)
> 投資保護: V15 案 B 改 単独継続 (絶対)、 V18-V22 は paper trade only
> 中央: 東京 / 京都 / 新潟 開催想定
> 手動投票継続 (race_auto_notify Discord 受信 → 手動 IPAT)

## 03:00-23:00 全 schedule

### 早朝 (03:00-06:00)

| 時刻 | task | 内容 | 担当 |
|------|------|------|------|
| 03:00 | DailyPremiumScrape | netkeiba プレミアム + JRA-VAN data 事前取得 | scheduler 自動 |
| 03:30 | SCRAPER-GUARD 確認 | OPERATIONAL_CALLERS ホワイトリスト OK | watchdog |
| 04:00 | jrdb_kyi 取得 | KYI / TYB / SED / KKA 等 | DailyJrdbKyi |
| 05:00 | NAR 出馬表 取得 | NAR (地方) 当日 race | nar_shutuba_scraper |
| 06:00 | morning go check (準備) | morning go worksheet 自動 fill in | Phase 21A schtask |

### 朝 (06:00-09:00)

| 時刻 | task | 内容 | 担当 |
|------|------|------|------|
| 06:30 | morning_go_worksheet 確定 | 5/17 morning go 6 axis fill in | schtask |
| 07:00 | jrdb_health_check | JRDB 取得 健全性 audit | JrdbHealthCheck_Sat |
| 07:30 | nightly_sanity 翌朝確認 | 全 task scheduler 健全性 | nightly_sanity_check |
| 08:00 | DailyPredict | V15 全レース 当日予測 (中央 + NAR) | scheduler 自動 |
| 08:30 | 戦略⑦ + 案 B 改 適用 | 06_特別 / 京都 / 条件 E / 条件 B 除外 | race_auto_notify 内 |
| 08:45 | race_auto_notify | 5 分前 自動予測 + Discord 投稿 (#買い目) | scheduler 自動 |
| 09:00 | 手動確認 | Discord 買い目 受信 → 手動 IPAT 投票 | ★人間★ |

### 朝補正 (09:30-10:00)

| 時刻 | task | 内容 | 担当 |
|------|------|------|------|
| 09:30 | morning_weight_check | 馬体重 ±10kg alert + 予測 再実行 | morning_weight_check |
| 09:45 | weight diff alert | 馬体重急変 馬 を Discord alert | 自動 |
| 10:00 | 必要なら 投票修正 | 急変 馬 の 影響を 人間判断 | ★人間★ |

### 中央 開催 (10:00-17:30)

| 時刻 | task | 内容 | 担当 |
|------|------|------|------|
| 10:00-17:30 | 中央 全 R | 各 R 5 分前に Discord 受信 → 手動 IPAT | ★人間★ |
| 並行 | V15 production 投票 | 戦略⑦ + 案 B 改 (12R 1勝 cls 上限 2,100円) | ★人間★ |
| 並行 | V18-V22 paper trade | 自動的に 投票 score を 別 sheet に記録、 ★ 投票 しない ★ | paper_trade_logger |
| 並行 | live odds 監視 | nar_live_odds + 中央 odds 連続 取得 | scheduler 自動 |

### NAR (10:00-21:00)

| 時刻 | task | 内容 | 担当 |
|------|------|------|------|
| 10:00-21:00 | NAR 全 R | 通常 自動運用、 中央と独立 | scheduler 自動 |
| NAR 投票 | NAR は paper only (現状) | 人間判断、 production は 中央のみ | ★人間★ |

### 夕方 - 夜 (18:00-23:00)

| 時刻 | task | 内容 | 担当 |
|------|------|------|------|
| 18:00 | DailyResults | 中央 結果照合 + ROI 計算 + Discord 通知 | scheduler 自動 |
| 19:00 | NAR 結果照合 | NAR 結果照合 (中央と別) | nar_results_scraper |
| 20:00 | DailyResults (再) | 平日 含む 結果照合 (二重 safety) | scheduler 自動 |
| 21:00 | paper_trade_eval | V15 / V18-V22 全 model の paper ROI 集計 | paper_trade_eval |
| 22:00 | 累計 ROI 監視 | cumulative_results.csv 更新 + 撤退 ライン check | roi_monitor |
| 23:00 | nightly_sanity (翌日 5/18) | 5/18 task 事前確認 + Discord 通知 | nightly_sanity_check |

## paper trade 並行 spec

5/17 production は V15 案 B 改 単独。 V18-V22 は paper のみで以下を記録:

| model | 何を記録 | 投票するか |
|-------|---------|-----------|
| V15 case B改 | production 投票結果 | ★する★ (戦略⑦込み 上限 2,100円/R 1勝 cls) |
| V18 (5/14 学習) | predict score + paper ROI | しない |
| V20 (Session #44) | predict score + paper ROI | しない |
| V21 (5/15 動画 features) | predict score + paper ROI | しない |
| V22 RL 1M | bet action + paper ROI | しない |

→ 5/17 終了後、 paper ROI 集計で V18-V22 の本命 候補 1 つを 5/24+ Phase 3 で 拡張投入 検討

## ★手動投票継続★ (絶対)

- IPAT 自動投票 NEVER (Session #86 確定、 法規制 + リスク観点)
- race_auto_notify の Discord 通知 → 人間が IPAT で 手動投票
- 投票上限: 5/9 確定 通り 「12R 1勝 cls 上限 2,100円」 (戦略⑦ + 案 B 改)
- 4/19 教訓: SCRAPER-GUARD 誤停止 で 午前ロス → AM7:30 nightly_sanity 確認 で 防止

## 投資保護 (絶対遵守)

- 🔴 predict_core.py / V15 production model: 5/17 中も NEVER 触る
- 🔴 paper trade は production と完全分離 (別 sheet、 別 csv、 別 Discord channel)
- 🔴 撤退ライン: 累計 -50,000円 (現在 +13,530円、 撤退余裕 +63,530円)
- 🟢 戦略⑦ + 案 B 改 (5/9 確定) 単独継続

## 5/17 終了時 完了条件

1. V15 production: 戦略⑦ + 案 B 改 で 投票完了 + ROI 集計
2. V18-V22 paper trade: 全 model の paper score 記録 PASS
3. 累計 ROI 撤退ライン 影響なし (-50,000円 まで余裕)
4. nightly_sanity 5/18 翌日確認 PASS

## 失敗時 緊急対応

- 5/17 単日 ROI < 50% → 戦略⑦ + 案 B 改 そのまま継続 (パターン破棄しない、 1 日では判断しない)
- 累計 -10,000円 突破 → 翌週末 投票上限 半分 (1,050円/R)
- 累計 -50,000円 突破 → 即 投票停止、 V20+ 切替検討 (8/1 Phase 3 投入候補)
- SCRAPER-GUARD 誤停止 → AM7:30 nightly_sanity で発見 → 即 patch + 手動 救出

## 次週 5/18+ task

- 5/18 (日): 中央 開催継続、 同 schedule
- 5/19 (月、 平日): paper trade 集計 + V18-V22 比較 → 5/24 Phase 3 plan 確定
- 5/24+: Phase 3 (sib_*_exp 統合 + V20 構築) 開始
