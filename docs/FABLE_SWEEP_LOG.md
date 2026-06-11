# FABLE 全系統自律スイープ ログ (2026-06-11 開始)

> user委任: 「全部チェックして自分で探して自分で治して」。権限区分=🟢障害修正OK/🟡提案のみ/🔴NEVER。
> セッション落ち時はここから resume。

## 状態: Phase 0 進行中

## Phase 0 — per-race 二重マージ (最優先・承認済)

### merge_jrdb_predict_features 呼出 全列挙 (コード側、docs/report除く) — 総数17ファイル
| # | ファイル:行 | build_features後の再マージ? | 状態 |
|---|------------|---------------------------|------|
| 1 | tools/predict_core.py:2008 | 正規(build_features内) | 基準・不変 |
| 2 | tools/daily_predict.py:536 | merge_jrdb_once ガード済 | ✅修正済(7c1e86fb) |
| 3 | tools/race_auto_notify.py:353 | ★要確認(実投票経路)★ | 🔍 |
| 4 | tools/predict_one_race.py:104 | 要確認 | 🔍 |
| 5 | predict_one_race_v3 (存在確認要) | 要確認 | 🔍 |
| 6 | tools/paper_trade_s2b.py:417 | 要確認 | 🔍 |
| 7 | tools/v21_per_race_paper.py:331 | 要確認 | 🔍 |
| 8 | tools/v21_paper_predict.py:259 | 要確認 | 🔍 |
| 9 | tools/save_all_horse_scores.py:179 | 要確認 | 🔍 |
| 10 | tools/reprep_compare_20260530.py:91 | 検証用одно-off | 🔍 |
| 11 | tools/bulk_predict_20260531.py:61 | one-off | 🔍 |
| 12 | tools/feat_fill_20260530.py:51 | one-off | 🔍 |
| 13 | tools/feature_coverage_check.py:87 | 診断(直マージ・build_features無) | 🔍 |
| 14 | tools/predict_dryrun_compare.py:34 | 診断 | 🔍 |
| 15 | tools/predict_core_v18.py:2007 | v18クローン内正規 | 🔍 |
| 16 | tools/fable_dpfix_verify.py | 検証ツール(意図的) | OK |
| 17 | tools/kyi_health_check.py | 監視(参照のみ) | OK |

## Phase 0 実施記録 (6/11)

### 🟢 修正済: 二重マージ残存 10 ファイル (build_features後の再マージ=劣化)
共通ガード `jrdb_features.merge_jrdb_once` を新設し全置換 (daily_predict のローカルガードと同一ロジック):
1. tools/race_auto_notify.py:353 ★実投票経路・4/2起源★
2. tools/predict_one_race.py:104
3. predict_one_race_v3.py:102
4. tools/paper_trade_s2b.py:417 (_predict_via_scrape fallback)
5. tools/v21_per_race_paper.py:331
6. tools/v21_paper_predict.py:259
7. tools/save_all_horse_scores.py:179
8. tools/reprep_compare_20260530.py:91 (one-off。★歴史的出力 data/allscores/20260530_v2_repaired は二重マージで生成された点に注意★)
9. tools/bulk_predict_20260531.py:61 (one-off。同上 20260531_v2)
10. tools/feat_fill_20260530.py:51 (one-off。同上 _feat_matrix_20260530.csv)

### 再マージなし=元から健全と確認した経路
- app.py:5295(UI)/6300(batch) — build_features のみ。※6300 は race_id 無し渡し → Phase 1 で確認
- predict_one_race_v2.py / predict_one_race_test.py — build_features のみ
- predict_and_log.py — build_features/merge_jrdb 不使用(旧パイプライン) → Phase 1 で評価
- paper_shadow_v15_full.py — ダンプ読みのみ

### 再発防止
- kyi_health_check.py に source_guard_scan() 追加 (リポジトリ全 .py 走査・代入形再マージ検出)
- tests/test_no_double_jrdb_merge.py に 3 テスト追加 (全10ファイルのガード使用 / 共通ガード挙動 / 全走査ゼロ) → 6 passed
- 判別テスト: tools/fable_sweep_phase0_verify.py (旧経路の劣化再現 + 新経路 no-op + predict_core直一致 + 6/7 top変化) → 実行中

## Phase 0 判別テスト結果 (tools/fable_sweep_phase0_verify.py、6/6-7 ダンプ 43R)
1. 旧経路の劣化再現(障害実証): **41/43** (残2RはKYIデータ自体なし=劣化不能。docs §7.6 の 41/43 と一致)、旧KYIデフォルト率 100.0%
2. merge_jrdb_once no-op: **43/43**、新KYIデフォルト率 15.3%
3. 新経路スコア = predict_core直: **43/43 完全一致 (<1e-12)**
4. 影響量参考値 = data/fable_dpfix_discriminate.json (top1変化3R/top3変化14R/formation変化23R)

## スケジューラ・運用系の発見と修理 (Phase 2 先行実施)

### 🟢 修正済 4: bat ファイル LF改行地雷 (27/70 ファイル)
- **障害**: cmd.exe は LF-only bat でコマンド欠損/文字食いを起こす (判別テストで `'hcp' is not recognized` を再現、CRLF版はクリーン)
- **実害**: \Keiba-Morning_Sat/Sun (morning_top_races=V15/V17朝ダイジェスト)・\Keiba-JrdbRetryAm9_Sat/Sun (JRDB 9時retry)・\Keiba-MultiStagePredict_*(6タスク) が **一度もログを書けず exit 255 = 作成以来全滅**
- **修理**: 27 bat を CRLF に一括変換 (バイト保存・内容不変)。全70 bat 走査で残LFゼロ確認

### 🟢 修正済 5: Stage2 (1h前予測) タスク期限切れ = silent stop
- **障害**: \Keiba-PreRacePredict_Watchdog_5_9 が 5/9 ワンタイム+反復700h設定 → **6/7 17:00 で反復期限切れ・Next Run=N/A**。6/13 は発火しない状態だった
- **修理**: 同名タスクを 土日 9:00-17:00 / 30分間隔 で再登録 (Next Run 6/13 9:00 確認)。平日発火を排除 (旧設定は平日も発火し下記の毎R失敗を量産)
- **付随修理**: stage2_predict.py に朝予測CSV不在時の no-op exit ガード追加 (5/11-6/5 に `daily_predictions/<date>.csv 不在` エラー **2,723件** = 毎fire×毎R失敗。5/31 は開催日なのにCSV欠落=Stage2全死)

### 🟢 修正済 6: weekly_report 4連鎖バグ (6/8 のレポート死の根治)
1. `umaren_hit='3-8-10'` 文字列1行で TypeError 全停止 → _calc_stats を to_numeric 化
2. cp932コンソールへの ⚠ print で UnicodeEncodeError 途中死 → stdout/stderr を utf-8 reconfigure
3. np.bool_ の json.dump TypeError → bool() cast
4. drift audit log の cp932バイト混入で UnicodeDecodeError → errors="replace"
- 検証: `--date 20260608` で exit 0・レポート/Discord md 生成まで完走

### 🟢 修正済 7: cumulative_results.csv 列ズレ1行 (5/5 NAR かしわ記念)
- 行0 が列ズレ (trio_result が trio_bets に、payout が hit に等) + profit列に 'settled' 文字列 → weekly_report 死因
- 外科修復 (該当1行のみバイト置換、他行不変)。修復後: trio_hit=1/payout=1010/investment=700/profit=+310/date=20260505
- ★PnL影響: この行は従来 investment=310・profit=NaN として集計から漏れていた → 修復で正値が乗る★

## Phase 1 — 全経路スコア健全性マトリクス (6/11)

### 予測エントリポイント全列挙 (総数12) と判定
| 経路 | predict_core直一致 | _x/_y衝突 | 判定 |
|------|-------------------|-----------|------|
| tools/daily_predict.py (8:00本番) | ✅ 43/43 (7c1e86fb) | なし | 健全 (6/11修正済) |
| tools/race_auto_notify.py (★実投票) | ✅ ガード後=同一経路 (Phase0判別テスト43/43) | なし | 健全 (6/11修正) |
| tools/predict_one_race.py / _v3 (CLI) | ✅ 同上 | なし | 健全 (6/11修正。_v3 は gitignore=ローカルのみ) |
| predict_one_race_v2 / _test | 再マージ無し=merge#1のみ | なし | 元から健全 |
| app.py:5295 (UI単レース) | build_features のみ | なし | 元から健全 (静的確認) |
| app.py:6300 (UI一括予測) | race_id 渡さず → JRDB 全デフォルト | なし | 🟡 提案 (app.py=🔴不可侵) |
| tools/paper_trade_s2b.py (TEST1/2) | 主経路=ダンプ再利用 / fallback 修正済 | なし | 健全 (6/11修正) |
| tools/paper_shadow_v15_full.py | ダンプ読みのみ | なし | 健全 (上流=daily_predict 修正済) |
| tools/v21_per_race_paper.py / v21_paper_predict.py | ✅ ガード後 | なし | 健全 (6/11修正) |
| tools/save_all_horse_scores.py (9:30) | ✅ ガード後 | なし | 健全 (6/11修正) |
| predict_and_log.py (旧CLI) | ★predict_core 不使用の独自旧パイプライン★ | — | 🟡 deprecation 提案 (V8/V9時代の残骸、スコア体系が本番と別物) |
| tools/stage2_predict.py (1h前) | predict_one_race 経由 → ガード済 | なし | 健全 (6/11修正) |

### 特徴族デフォルト率 (6/6-7 復元ダンプ 43R = 修正後経路ベースライン)
| 特徴族 | デフォルト率 | 判定 |
|--------|------------|------|
| KYI | 14.7% | 健全 (修正前 100%) |
| PACI | 0.2% | 健全 |
| ZE | 7.8% | 健全 |
| OZ (odds_change) | 100% | 仕様 (8:00=base snapshot 自体。変動は per-race 時点で発生) |
| TYB | 100% | 仕様 (TYB は -15min 公開、8:00 ダンプに無いのは設計どおり。per-race は merge#1 で取得) |
| SED前走 | 15.3% | 健全 |
| netkeiba (wood/comment) | 44.3% | 許容 (コメント無し馬=0 が支配的。speed_index はモデル外=UI用) |

## ★経営的重大事実 (6/11 判明)★
- **累計 PnL = -49,240円 / 756R / ROI 90.7%** (weekly_report 6/8 週次より)
- **撤退ライン -50,000円 まで残り ¥760** — 6/13 の運用判断は user 必須

## 発見一覧 (随時追記)
| # | 発見 | 重大度 | 処置 |
|---|------|--------|------|
| 1 | per-race系 二重マージ残存 10 ファイル (race_auto_notify 含む) | 高(実投票) | 🟢 修正済 (Phase 0) |
| 2 | 5/30-31 one-off 出力 (allscores *_v2*, _feat_matrix) は二重マージ下で生成 | 中(検証データ) | 記録。consumer を Phase 2 で確認 |
| 3 | app.py:6300 batch 経路は race_id 無しで build_features → JRDB 全デフォルト | 中(UI一括予測のみ) | 🟡 提案 (app.py予測ロジック=🔴不可侵。_batch_score_race に race_id 引数追加を提案、呼出元 6411 に rid あり) |
| 4 | bat LF改行 27/70 (Morning/JrdbRetry/MultiStage 全滅の原因) | 高(運用) | 🟢 修正済 |
| 5 | Stage2 タスク 6/7 期限切れ silent stop + 平日毎R失敗 2,723件 | 高(6/13) | 🟢 再登録+ガード |
| 6 | weekly_report 4連鎖バグで 6/8 レポート死 | 中 | 🟢 修正済 |
| 7 | cumulative_results.csv 列ズレ1行 | 中(集計) | 🟢 外科修復 (commit は user 運用フローで) |
| 8 | 累計 PnL -49,240 = 撤退まで ¥760 | ★経営★ | 報告 (判断は user) |
| 9 | daily_predict 平日 exit 1 (非開催日) → タスク Last Result=1 | 低(cosmetic) | 記録のみ |
| 10 | RaceAutoNotify_Sat 6/6 = 0x40010004 (外部終了)・WeeklyScrapeResume 6/8 = 0xC000013A (console閉) | 低(単発) | 記録のみ。再発すれば調査 |

## Phase 進捗
- [ ] Phase 0: per-race二重マージ修正
- [ ] Phase 1: 全経路スコア健全性マトリクス
- [ ] Phase 2: データ配管全数点検
- [ ] Phase 3: 表示・通知の正直さ
- [ ] Phase 4: 145特徴 override総当たりリーク監査
- [ ] Phase 5: 総仕上げ・6/13リハーサル
