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

## Phase 2 本体 — 台帳・データ配管 (6/11 agent監査 + 修理)

### 🟢 修正済 8: 台帳 (cumulative_results.csv) 4系統の重大不整合 ★お金★
- **根本原因**: daily_results.py の (date, race_id) キーが float化date (`'20260607.0'`) で別キー扱い → dedup/settledスキップ両すり抜け → 同日2回実行 (土日18:00+20:00タスク) で全R二重計上
- **A. 6/7 全23R二重** (-6,570円の二重計上) → dedup で除去。※過去には 5/24 でも発生 (公式 commit 3c3c5412 の「n=697 / -35,280」も二重込み、当時真値 n=664 / -27,330)
- **B. 5/23 丸一日欠落** (33R, -10,040円) → daily_results/20260523.csv から復元
- **C. 3/14・3/15 的中17件が miss 記録** (+23,260円 過小計上、jra_payouts.csv と金額一致検証済) → 修正
- **D. 202606030509 の date 誤記** 20260405→20260411 → 訂正
- **修理後の真値: n=766 / PnL ★-29,450円★ / ROI 94.51%** (行内整合違反0・重複0。backup=cumulative_results.csv.bak_20260611_ledger)
- 再発防止: daily_results.py に _norm_key 正規化 (4箇所) + tests/test_cumulative_dedup.py (3 passed)

### 🟢 修正済 9: anomaly_auto_detector 3バグ
- 非開催日に「predictions不在=critical」誤警報 (5/11-6/10 で68日分、rollback提案文付き) → daily_predict の非開催記録参照で ok 化 (watchdogログ対応・utf-8/cp932両対応)
- strategy anomaly 検知が `roi` キー誤り (実キー roi_pct) + 閾値 -50 (スケール誤り) の二重バグで**永久不発** → roi_pct + 単日ROI<50% (設計意図=撤退3段階) に復元、n=0日ガード追加
- Discord に未解決プレースホルダ `git revert <Sub-task 8 commit hash>` を送信 → 手順参照に変更

### 🔴 DEAD データソース (agent監査・内容ベース確定)
| ソース | 停止日 | 影響特徴 |
|--------|--------|----------|
| jrdb_sed.csv | 5/9 | SED前走族6特徴 (5/10以降に前走がある馬で欠損) → ★6/11 復旧 (下記)★ |
| jrdb_kta.csv | 4/5 (67日) | jrdb_kta_idm/ten_pred/agari_pred 3特徴デフォルト化 |
| jrdb_kka.csv | 5/3 | jrdb_dam_rensho_avg/bms_rensho_avg |
| jrdb_sr.csv / srb.csv | 5/9 / 3/29 | tb/バイアス族 (V20学習側) |
| jrdb_skb.csv | 5/3 | jrdb_anshin/heavy_apt_skb (V20では除外予定族) |
| jrdb_tyb.csv | 5/17 | V15本番未使用 (V21候補/paper) |
| jrdb_cyb.csv | 4/19 | パーサ破損 (列ズレ蓄積)。本番未参照 |
| netkeiba_training_eval 2026年分 | 全行空ペイロード (ゾンビ) | V16学習側 |
- **検知ギャップ根治**: data_freshness_monitor に内容ベース停止検知 (KAB kaisai_key→開催日マップで各CSVの「中身の最新開催日」を判定) を追加。KTA 67日/SED 33日/SRB 74日等を即検知することを実測確認。非開催日も鮮度チェック継続に変更
- **SED 復旧**: バックフィル時に scrape_jrdb.save_csv の dedup キー不一致 (jra_race_id vs 旧 race_id) で旧548,780行が1行に潰れる事故が発生 → data/jrdb/extracted/Sed 全txt + jrdb_raw/sed lzh から完全再構築 (5/10-6/7 の欠落分も含めて回収)。save_csv に「両スキーマ実在時のみdedup + 行数半減拒否ガード」を恒久追加
- 🟡 KTA/KKA/SR/SRB/SKB/TYB のバックフィルは各fetcherの根本原因調査が必要 (次セッション最優先。監視は上記で稼働開始済)

### 通知の正直さ (Phase 3 agent監査 42件 → 主要対応)
- 🟢 stage2_predict: 「累計+13,530円死守」(drift値・真値は負) + 5/9固定投票指示 (新潟12R) を毎開催日 #買い目 に送信していた → 撤去
- 🟢 notify.py #買い目: ROI 205-330% 表示の正体は v12 backtest 値 → 「BT(v12)ROI」と世代明記 (値の変更はせず)
- 🟡 提案 (修正禁止領域): H4=142.6%基準の恒常誤DANGER (roi_monitor/weekly_report/roi_analysis、V15基準への差替は閾値変更=要承認) / H5=★app.py discovery が v22 (LEAK INVALID) を Pattern A スロットに掴む+「4-MODEL/LEAK-FREE」虚偽バッジ (app.py=不可侵、要承認で修正すべき)★ / H7=クラス別「ROI 455%」出典不明 / M15=kelly bankroll=50,000 出典不明 / 他 docs 参照
- 🟡 weekly_report のモデル健全性チェックが v12/v9 のみ (V15 を見ていない)・z検定 payout に umaren 不加算 (条件E過小)
- 🟢 daily_cumulative_audit.bat: >>redirect と内部 append_log の同一ファイル自己ロック (毎晩 PermissionError) → runner.log 分離

### Phase 4 — リーク監査 (統計+コード 両面完了)
- **統計 (tools/fable_override_audit.py)**: leak-free v2 / 2023-25 / 141,523行・10,365R で全145特徴に AUC・finish相関・非対称・反市場テスト → ★ze型 (反市場) 署名ゼロ★ (最大0.201 < 閾値0.233)。56フラグは全て合法的予測力 (オッズ/人気代理/過去成績)
- **コード (agent全145マップ、UNKNOWN 0件)**: SAFE 140 / SUSPECT 5
  - ★S1 (新発見・高): `odds_change_rate`/`pop_rank_change`/`odds_sharp_drop` が学習時に**確定オッズ・確定人気**使用 (train_v134_odds_change.py:165-199)。odds_log (LEAK_FEATURES_A筆頭) の派生が Pattern A に残存 = 自家ルール不整合 + train/serve不一致 (本番は base比のリアルタイム差・朝は0)。corr_target -0.323★ → 検証側 v3 評価 (3特徴を朝予測実態=0 に中和して V15 真値再計算) が必要
  - S2: jrdb_tb_homestr_inner = 当該レース事後SRを直マージ (race-level定数で corr -0.002 = 実害≈0) → V20で前走shift
  - S3: jockey_change_to_top の top20 集合が全期間定義 (軽微) → V20で expanding 化
  - 既知ze 4特徴 = v2 cache で merge_asof backward により中和済を確認
- 捨て列棚卸し (agent): 🟢候補2 = ①JOA馬場コードの参照先誤り (jrdb_joa.csv に列なし・実体は jrdb_kab.csv → 恒常0の死特徴、ただし学習も0なので train/serve skew なし・修正は再学習とセット=🟡扱い) ②blinker が kyi_key_cols から欠落 (consumer=検証側 build_competitor_gap_features が skeleton 止まり) → 🟡 V20データ再生成とセット
- 🟡 活用候補: SED朝10時オッズ・TYB odds_time・OT個別三連複オッズ・調教laps (docsに記録)

## ★経営的重大事実 (6/11 確定・台帳修理後)★
- **真の累計 PnL = -29,450円 / 766R / ROI 94.51%** (修理前の表示 -49,240 は二重計上+欠落+miss誤記の合成)
- 撤退ライン -50,000円 まで余裕 **¥20,550** (修理前の見かけ ¥760 から訂正)
- 公式記録の過去値も汚染: 5/26 commit「n=697/-35,280」は5/24二重込み (当時真値 -27,330)
- Streamlit TRACK RECORD (track_record.csv) は**+68,790円と符号逆の虚偽表示** (126重複) → 🟡 再構築要承認
- 8戦略paper (race_notify_log_v2) の phase3 が**未配線で恒久ゼロ** (formation喪失対策 5/18 Sub-task C が機能していない) → 🟡 配線要承認 (実投票経路への追記のため)

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

## Phase 4 S1 定量化 (tools/fable_v15_noS1_eval.py、leak-free v2 同一WFパイプライン)
| model | feat | AUC | 単勝 | 三連複t4 | 馬連box |
|-------|------|-----|------|---------|--------|
| V15 (S1込み=v2公表再現) | 145 | 0.8418 | 105.0% | 154.4% | 126.6% |
| V15_noS1 (S1除外) | 142 | 0.8381 | **108.4%** | **158.4%** | **128.7%** |
- **結論: S1 は AUC を +0.0037 嵩上げしていたが、ROI 評価はむしろ -3〜4pt 抑制側** → v2 公表の V15 ROI 105.0% は過大でなく保守側だった (糊塗不要)。**s2b は ODDS_REMOVE で S1 を元々除外 = s2b 真値・CI・GO/NO-GO 判定はすべて非汚染**
- V20 への提案 (🟡): odds_change_rate/pop_rank_change/odds_sharp_drop を LEAK_FEATURES_A に追加 (odds_log 派生の整合)

## Phase 5 — 回帰・リハーサル (6/11)
### 回帰: 584 tests / 8 failed / 0 errors
- 8 failed は**全て既存** (70ffe1ec 時点のソースで同条件確認済): test_features×3 (V8/V9時代のstaleテスト)・race_auto_notify_integration×3 + race_notify_log_v2×2 (v2配線不足の証跡=🟡発見6と同根)
- sweep で追加/修正したテスト (二重マージ5+台帳dedup3+anomaly9) は全 pass

### 6/13 リハーサル結果
- kyi_health_check: source_guard_scan=二重マージ疑いゼロ。dumpチェックは 6/7(修正前データ)が NG 表示=正しい挙動、6/13 の新 dump から OK 化見込み
- schtasks: DailyPredict 6/12 8:00 / Morning_Sat 6/13 6:30 (★CRLF修正後の初実走★) / RaceAutoNotify_Sat 8:45 / Stage2 9:00 (再арm済) / JrdbRetryAm9 9:00 (初実走) / SaveAllHorseScores 9:00 / PaperS2B 9:00 — 全て Next Run 確認済
- admin_verify_v2: bat/py 全 OK。schtask 7件 (AnomalyCheck×5・DailyCumulativeAudit・RaceNotifyLogV2-Aggregator) は**未登録のまま** (登録bat 3本がLF壊れで作成以来未登録だった。schtasks /Create が要管理者 → ★user action: 管理者で register bat 3本実行★)

## user 宛 action items
1. ★管理者 PowerShell で実行★: `tools\register_anomaly_detector_schtask.bat` / `tools\register_daily_cumulative_audit_schtask.bat` / `tools\register_race_notify_log_v2_aggregator_schtask.bat` (6/11 CRLF修正済・以前は壊れていて一度も登録されていない)
2. 6/13 は Morning_Sat (6:30 JRDB+V17ダイジェスト) と JrdbRetryAm9 (9:00) が**作成以来初めて実走**する — Discord に新しい通知が来るのは正常
3. 🟡要承認の判断: app.py v22 discovery 除外 / track_record.csv 再構築 / race_notify_log_v2 phase3 配線 / 142.6%基準の更新 / DEADデータ6源 (kta/kka/sr/srb/skb/tyb) の fetcher 修理

## Phase 進捗
- [x] Phase 0: per-race二重マージ修正 (PASS 判別テスト・commit cab8c396)
- [x] Phase 1: 全経路スコア健全性マトリクス (12経路・特徴族デフォルト率)
- [x] Phase 2: データ配管全数点検 (台帳修理・SED復旧・内容鮮度監視)
- [x] Phase 3: 表示・通知の正直さ (42所見→🟢7件修正/🟡記録)
- [x] Phase 4: 145特徴 override監査 (統計=ze型署名ゼロ / コード=S1発見+定量化)
- [x] Phase 5: 回帰・リハーサル・記録
