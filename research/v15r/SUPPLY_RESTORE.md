# 供給復旧 (2026-08-11 実施) — s2b含む全パイプライン

## 根因 (3つの独立障害が同週に重なった)
| # | 障害 | 証拠 | 影響 |
|---|------|------|------|
| 1 | **金曜 FridayWeekendScrape 停止** (最終ログ 6/12) | logs/weekend_thisweek_*.log | 実供給源だった週次Paciバンドル(KYI等)が停止 → JRDB系全滅 (6/22+) |
| 2 | **daily_jrdb_kyi.bat の恒常0件バグ** | scrape_jrdb --date TODAY が「0日間」計算 → 毎日"Done"を報告しつつ0件 | 金曜停止を代替できず |
| 3 | **netkeiba charset 変更 (~6/20)** | parse_shutuba の EUC-JP ハードコード vs ページ utf-8 宣言・台帳の馬名化け開始 6/20 | 馬名 mojibake (U+FFFD 不可逆)・premium 特徴死と併発 |
| - | **監視封殺**: freshness monitor は STALL を検知していたが `--no-notify` | logs/daily_paci_refresh.log | 全障害が2ヶ月無警報 |

## 実施内容
1. **新 日次供給ジョブ** `tools/daily_jrdb_supply.py/.bat`: datazip日次zip(6種) + KTA/KKA日次lzh + JOA + 種別単独フル再構築(実証済ドライバをtools/昇格)。フルテスト ALL OK (443s)。
   - タスク登録: `keiba-ai\JrdbSupplyDaily` 毎日03:20 + `JrdbSupplyWeekendAM` 土日06:35 + `T1v2Audit` 毎日08:50
2. **旧チェーン無効化**: DailyJrdbKyi (disable) / WeeklyScrapeResume (kill+disable) / DailyPremiumScrape・FridayWeekendScrape (bat無効化スタブ、タスクdisableは要管理者= tools/disable_netkeiba_tasks_admin.bat)
3. **マスタ再同期**: 馬名=KYI 8/9まで100%解決(2歳デビュー込29,468行)。騎手= `tools/update_jockey_wr_jrdb.py` 新設(JRDB SED×paci、netkeibaスクレイパ廃止) → 145騎手更新(ルメール0.271等)。厩舎=KYI/paciがレース毎供給(別マスタ不要)
4. **T1v2 --source-check**: 平日=供給レベル監査(KYI/SED内容鮮度+馬名解決率≥99%) / 土日=dump監査。**8/11 PASS → BLOCKフラグ自動クリア=供給復旧(ソースレベル)**。dump完全確認は次開催 8/15
5. **s2b無効化**: 死亡窓で実行されたのは 8/9 のみ → `*.invalid_data` リネーム+INVALID_RANGES.json (集計glob から除外)。**paper評価は 8/15 から再スタート** (PaperS2B タスクは Ready のまま)
6. **表示修正**: parse_shutuba/oikiri の EUC-JP ハードコード → meta charset 検出 (db.netkeiba は現在も EUC-JP のため4箇所は不変)。実ページ検証: race_name/全馬名/race_info 完全正常

## 土曜 8/15 の自動チェーン (期待動作)
03:20 供給 → 06:35 供給2nd → 08:00 daily_predict (BLOCK解除済・shutuba公開ページ+修正エンコで動作・premium特徴はdefault) → 08:50 T1v2 dump監査 (JRDB定数≤10期待) → 09:00 PaperS2B再開
※ 実投票はユーザー判断 + item6 paper ゲート。premium系特徴(~gain9%)はdefault=劣化運転である点に注意。

## 残課題
- CHA (追切指数) 6/22+ 源なし (JRDBサイト側に日次無し・影響gain~0.1%)
- JV-Link 日次 (UM馬マスタ/SE結果/HR払戻) の常設ジョブ化 = v15r 差し替え設計と併せて次段
- freshness monitor の --no-notify 解除 (要検討: 誤報履歴とのバランス)
