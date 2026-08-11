# 通知系 棚卸し + 再整備 (2026-08-11)

予測ロジック不変・通知/表示レイヤーのみの変更。webhook: BETS(#買い目)/UPDATES(#アップデート)/TEST(テスト)/URL(fallback)。

## 開催日タイムライン (土日、8/11 再整備後)
| 時刻 | タスク | スクリプト | ch | 状態 |
|------|--------|-----------|----|------|
| 01:30 | TybPublishMonitor | tyb publish 監視 | updates | 生 (JRDB) |
| 02:55-08:50 | Pre/AM3/AM6/AM8 FireCheck | am*_fire_check.py | updates | 生 (健康通知4本=将来統合候補) |
| 03:00 | DailyPremiumScrape | daily_premium_scrape.bat | updates | ★8/11 スタブ化 (netkeiba premium)★ |
| 03:20 | JrdbSupplyDaily ★新★ | daily_jrdb_supply.py | (log) | 生 |
| 06:00 | DailyJrdbKyi | daily_jrdb_kyi.bat | updates | ★8/11 disable (0日間バグ)★ |
| 06:30 | Morning_Sat/Sun | morning_go_check | updates | 生 |
| 06:35 | JrdbSupplyWeekendAM ★新★ | daily_jrdb_supply.py | (log) | 生 |
| 07:00 | MorningDigest | morning_dashboard.py | updates | 生 |
| 07:30 | DataFreshnessMonitor | data_freshness_monitor.py | updates | 生 (--no-notify 箇所は要解除検討) |
| 08:00 | DailyPredict | daily_predict.py | updates | 生 (完了サマリ/取り漏れのみ・レース別買い目送信なし・T1v2ブロック付) |
| 08:45 | RaceAutoNotify_Sat/Sun | race_auto_notify.py (5分前ループ) | bets | ★8/11 batスタブ化 (item4 提案実装・復元1コマンド)★ |
| 08:50 | T1v2Audit ★新★ | t1v2_feature_audit.py | updates(異常時) | 生 |
| 09:00 | PaperS2BPredict | paper_trade_s2b.py | updates | 生 (8/15 再開) |
| 09:00 | JrdbRetryAm9_Sat/Sun | jrdb_retry_am9.bat | (log) | 旧経路 (supply代替済・整理候補) |
| 09:30 | ★MorningBatchNotify 新★ | morning_batch_notify.py + feature_health_report.py | bets + updates | ★新設 (本再整備の中核)★ |
| 09:30 | MorningWeightCheck_Sat/Sun | 馬体重再チェック | updates | 生 (公開weight・同時刻だが内容別) |
| 10:00-15:45 | MultiStagePredict×3 | multi_stage/stage2 | updates | 生 (paper eval) |
| 18:00 | DailyResults + RaceDayReport | daily_results.py | updates | 生 (公開result・台帳) |
| 20:30 | PaperS2BResults / RaceNotifyLogV2-Aggregator | s2b / aggregator | updates | 生 / 5分前停止中はphase1空 (無害) |
| 20:45-23:00 | PerRaceCoverage / CumulativeAudit / FeaturesIntegrity / NightlySanity | 各監査 | updates | 生 |
| 月 08:00 | WeeklyReport | weekly_report.py | updates | 生 (旧142.6%基準の誤DANGER既知) |

## 死んでいる/ゾンビ通知
| 項目 | 状態 | 対処 |
|------|------|------|
| Verdict_R11/R12系 ×6・Cumulative_1700_5_9・Summary_2030_5_9・VoteCandidates_1400_5_9 | next run N/A = 期限切れ one-off (5-6月) | 整理候補 (削除は要管理者・実害なし) |
| DailyPremiumScrape / FridayWeekendScrape | netkeiba premium 死 | 8/11 スタブ化済 (正式disableは tools/disable_netkeiba_tasks_admin.bat) |
| DailyJrdbKyi | 恒常0件バグ | 8/11 disable済 (JrdbSupplyDaily が代替) |
| update_jockey_wr.py | netkeibaスクレイパ+mojibakeクラッシュ | update_jockey_wr_jrdb.py が代替 (8/11) |
| 重複気味 | 朝の健康通知5本 (FireCheck×3/Digest/Freshness) + T1v2 + 特徴量健全性(新) | 将来統合候補 (今回はスコープ外) |

## 8/11 再整備の実装 (item2/3)
- **09:30 朝一括通知** `tools/morning_batch_notify.py` (#買い目):
  T1v2 PASS 確認後に送信 / BLOCKフラグ時は「監査NGのため予測停止」のみ /
  冒頭サマリー (総R数・買い/見送り・T1v2・供給鮮度 KYI/SED) + 買いレースごとに
  発走/場/R/クラス/頭数・V15上位6頭・フォーメーション (📝PAPERバッジ)・
  特徴量詳細 (非ゼロ/145・KYI結合率・gain上位6 (s2b同形式・位置対応済)・premium欠損数)。
  見送りは理由付き一覧でサマリーに圧縮 (36R→23メッセージ)。
  ★mojibake dump対策: 馬名が化けている場合 KYI から復元 (6/20-8/9 の歴史的破損に対応)★
- **特徴量健全性 日次レポート** `tools/feature_health_report.py` (#アップデート・別送1通):
  カテゴリ別生存率 (BASE75/JRDB系54/premium系16)・死亡中リスト・前回監査との差分 (新規死亡/復活)。
- タスク: `keiba-ai\MorningBatchNotify` 土日 09:30 (両スクリプトを連続実行)。

## item4: 5分前通知の現状と提案
- **現状**: race_auto_notify.py = 08:45 起動の常駐ループ。各レース発走5分前に
  独自予測 (netkeibaリアルタイムオッズ+TYB20分前) → #買い目 通知 + race_notify_log_v2
  phase1/2 記録 (8戦略 paper shadow)。実投票経路として設計されたもの。
- **提案 = paper期間中は停止 (実装済・復元容易)**:
  理由: ①実投票なし期間に直前オッズ再予測の実益なし ②通知量 (レース毎) が
  09:30 一括と重複 ③9月ゲートは cumulative (朝予測) で判定するため
  race_notify_log_v2 phase1 の欠落はゲートに影響しない。
  トレードオフ: 8戦略 paper shadow の記録が止まる (5-8月分は既に供給死で無効)。
  **復元**: `copy race_auto_notify.bat.bak_20260811 race_auto_notify.bat` (1コマンド) +
  9月GO検討会で実投票再開とセットで復活を推奨。
- タスク本体の disable は要管理者: `tools/disable_netkeiba_tasks_admin.bat` (追記済)。

## テスト送信 (2026-08-11 実施)
`--date 20260809 --test` で **DISCORD_WEBHOOK_TEST チャンネルに実送信済**
(朝一括: サマリー1+買い22embed / 健全性レポート1通)。スクショ確認可。
※ 8/9 データは供給死期間のため「KYI結合0%・premium欠損13」等の劣化表示は正 (当時の真実)。
   8/15 からは供給復旧+encoding修正済みの新規 dump で清浄な表示になる。
