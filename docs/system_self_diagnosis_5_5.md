# keiba-ai システム 自己診断レポート

**作成**: 2026-05-06 00:30 (Claude Code Opus xhigh、独立走査)
**対象**: keiba-ai プロジェクト 全コードベース
**目的**: 「これがあったら更に良いのに」を率直に抽出

---

## 評価サマリー

| 領域 | 健全度 | 主な improvement opportunity |
|------|--------|----------------------------|
| B1. コードベース | 🟡 中 | Tests 19 ファイルでカバレッジ不明、error handling 不統一 |
| B2. データ冗長性 | 🟠 中-低 | netkeiba/JRDB 単一 source 依存、fallback 限定的 |
| B3. デプロイメント | 🟡 中 | 別 PC 移植性が手動依存、復旧 procedure 暗黙知 |
| B4. 観測可能性 | 🟢 中-高 | Discord 28 task + fire_check で良好、ダッシュボードは断片的 |
| B5. ドキュメント | 🟡 中 | docs 37 + data/v18 59 で過剰、CLAUDE.md 1325 行で肥大 |
| B6. 開発体験 | 🟠 低-中 | 新 Claude Code セッション の context 把握 = 5-10 min かかる |

---

## B1. コードベース監査

### 現状

- Python script: tools/ 100+ files, train/ 多数, root に主要 module
- テスト: `tests/` 19 ファイル (test_features, debug_all, regression_test_v15_final, fullclass_test_v15, validation_1〜13_*, test_jrdb_merge_strict, test_payout_integrity, test_pre_fire_check, test_predict_core, test_process_watchdog, test_scraper_guard など)
- error handling: try/except + Discord notify が 80% のパターン、残り 20% は silent fail

### アップデート候補

| 項目 | 詳細 | 緊急度 | 工数 | 着手難易度 |
|------|------|--------|------|-----------|
| **テストカバレッジ計測** | `pytest --cov=tools --cov=train` で実 % 不明 | 🟡 中 | 30min | 低 |
| **validation_1〜13 を v15 対応** | 全て 3/11 stale (v12 ベース)、リーク監査特に重要 | 🟠 高 | 6h | 中 |
| **error handling 共通化** | retry decorator + structured logging を 1 module に集約 (現状 20+ 箇所で個別実装) | 🟡 中 | 4h | 中 |
| **logging の一貫性** | `print(...)` と `logging.info(...)` が混在、structured JSON log への切替 | 🟡 中 | 6h | 高 |
| **silent fail 撲滅** | except: pass を grep で発見・対処 (現状 30+ 箇所) | 🟡 中 | 2h | 低 |
| **secrets 管理** | `.env` 1 箇所集中、git history scrubbing 済 (`docs/security.md`) | 🟢 低 | - | - |
| **dependency lock** | `requirements.txt` あるが pip-tools / poetry なし、再現性弱 | 🟡 中 | 1h | 低 |
| **type hints** | 主要 module は `from __future__ import annotations` 一部、全体カバレッジ 30% 程度 | 🟢 低 | 8h | 中 |

### 推奨優先度

1. テストカバレッジ計測 (30min で現状把握)
2. validation_1〜13 を v15 対応 (リーク監査、Phase 3 必須)
3. silent fail 撲滅 (Discord 通知漏れの真因かも)

---

## B2. データソース冗長性

### 現状

| データ | source | fallback |
|--------|--------|---------|
| 出馬表 | netkeiba (Cookie 必要) | db.netkeiba.com (premium 不要) |
| 結果 | netkeiba | db.netkeiba.com |
| 配当 | JRA 公式 DB scrape | 不在 (fail で終わる) |
| 馬場 | JRA 公式 (Shift_JIS) | 不在 |
| 天候 | 気象庁 API | 不在 |
| JRDB Advance 23 種 | jrdb.com | 不在 |
| premium (調教/コメント/指数) | netkeiba premium Cookie | A/B/C/D 推定 (4 段階フォールバック実装) |

### アップデート候補

| 項目 | 詳細 | 緊急度 | 工数 | 着手難易度 |
|------|------|--------|------|-----------|
| **JRA 配当 fallback** | netkeiba 結果ページから配当抽出 (現在 jra_payouts.csv 4/26 で stale 既知バグ) | 🟠 高 | 3h | 中 |
| **JRDB ダウン時の代替** | netkeiba から騎手・調教師統計を再構築 | 🟡 中 | 4h | 高 |
| **netkeiba ban 時の対策** | User-Agent rotation + IP 多重化、現状単一 IP/UA | 🟡 中 | 6h | 高 |
| **Cookie 失効時の自動復旧** | refresh_cookie.py --auto は実装済だが Discord alert 不在 | 🟠 高 | 30min | 低 |
| **TARGET 退会後の代替評価データ** | 学習データは 2025 年で固定、評価で stale 対応必要 | 🟡 中 | 8h | 高 (6 月 JRA-VAN 再契約で解決) |
| **JRDB 旧データ archive 戦略** | 半年分以上は archive/ へ移動して active を軽量化 | 🟢 低 | 2h | 低 |

### 推奨優先度

1. Cookie 失効時 Discord alert (30min、最大の運用リスク)
2. JRA 配当 fallback (jra_payouts 自動取得復旧)
3. netkeiba ban 対策 (UA rotation)

---

## B3. デプロイメント / 別 PC 移植性

### 現状

- 全パスが `C:\Users\takum\keiba-ai` にハードコード (BASE_DIR で集中化されてはいる)
- schtasks 28 件の定義は `tools/silentify_all_tasks.ps1` + `register_*_schtasks.ps1` 数本に分散
- `.env` は git ignore、別 PC 移植時は手動コピー必要
- pip dependencies: `requirements.txt` あり、Python 3.11+ 推奨
- migrate_to_new_pc.py スクリプト存在 (`tools/migrate_to_new_pc.py`)

### アップデート候補

| 項目 | 詳細 | 緊急度 | 工数 | 着手難易度 |
|------|------|--------|------|-----------|
| **別 PC 移植手順書** | migrate_to_new_pc.py の usage doc 不在 (script 単独で動作させる手順が暗黙) | 🟡 中 | 2h | 低 |
| **schtasks 一括登録 ps1** | 現状分散している ps1 を 1 つに統合 + idempotent 化 | 🟡 中 | 4h | 中 |
| **disaster recovery procedure** | PC 故障時の復旧手順書 (バックアップ → PC 再構築 → 動作確認) | 🟠 高 | 3h | 低 |
| **backup 戦略** | ローカル `data/_v15_optuna_df_cache.pkl.gz` 104MB + JRDB raw + premium CSV の自動バックアップなし | 🟠 高 | 2h | 中 |
| **クラウド同期** | 重要データを Google Drive / Dropbox に自動 sync (現状なし) | 🟢 低 | 4h | 中 |
| **Docker 化** | 環境再現性向上、ただし Windows + GUI Cookie + schtasks との相性悪い | 🟢 低 | 1 日 | 高 |
| **README に必須前提** | Python version / OS / 必要 Cookie の前提を明記 | 🟡 中 | 30min | 低 |

### 推奨優先度

1. disaster recovery procedure (3h、PC 故障 = 業務停止リスク最大)
2. backup 戦略 (Google Drive sync が現実解)
3. 別 PC 移植手順書

---

## B4. 観測可能性

### 現状

- Discord 3 channel 振り分け済 (#bets / #updates / fallback)
- 28 schtasks すべて Discord 通知 (notify_done.py / notify.py 経由)
- fire_check 4 種で発火確認 (pre/am3/am6/am8)
- KeibaAI_DriftDetector 月 08:30 で drift 検知
- nightly_sanity_check 23:00 で翌日 task 事前チェック
- ダッシュボード: tools/dashboard.py (一覧表示)、project_status.py (CLI)、Streamlit app.py

### アップデート候補

| 項目 | 詳細 | 緊急度 | 工数 | 着手難易度 |
|------|------|--------|------|-----------|
| **累計収支リアルタイム dashboard** | Streamlit app に "累計 +14,140 円 / 撤退余裕 +64,140 円" の表示なし | 🟠 高 | 1h | 低 |
| **schtasks 全 28 件 status 一覧 web UI** | 現状 PowerShell でしか見れない、Streamlit page 化 | 🟡 中 | 2h | 低 |
| **Discord alert 階層化** | CRITICAL/WARN/INFO の階層が watchdog v2 のみ、他は flat | 🟡 中 | 1h | 低 |
| **SLO 設定** | 「daily_predict が 09:00 までに完了」等の SLO 不在 | 🟢 低 | 2h | 中 |
| **drift detection 可視化** | data/drift_detection_log.json の Streamlit page 化 (現状 raw json) | 🟡 中 | 1h | 低 |
| **ROI 月次レポート** | weekly_report.py あるが月次集計なし | 🟡 中 | 2h | 低 |
| **撤退ライン Discord alert** | 累計 -10k / -30k / -50k 跨ぎで自動 push (現状手動目視) | 🟠 高 | 30min | 低 |

### 推奨優先度

1. 撤退ライン Discord alert (30min、心理安全装置の実装層)
2. 累計収支リアルタイム dashboard (1h、ユーザーが寝起きで開ける)
3. drift detection 可視化

---

## B5. ドキュメント体系

### 現状

- root: CLAUDE.md (1325 行) / README.md (218 行、5/5 18:35 更新で完璧)
- docs/ 37 ファイル (HANDOFF / lessons / sessions_recap / V162 / GW / V17 strategy など)
- report/ 40+ ファイル (4/19-4/26 の audit、Phase 2.5+ で意味失効が多い)
- data/v18/ 59 ファイル (Phase 2.5 進捗 + V18/V19 retro + NAR 設計)
- data/results/ 16+ ファイル (5/9 投票関連)

### アップデート候補

| 項目 | 詳細 | 緊急度 | 工数 | 着手難易度 |
|------|------|--------|------|-----------|
| **CLAUDE.md V15 中心に書換** | 「現行モデル v13.5b」のまま、V15/V15.1/V17/V18/V19/NAR 一切なし、v16 セクション二重重複 | 🟠 高 | 2h | 中 |
| **古い doc archive** | docs/ の GW 系 8 + V162 系 4 + 4/19-4/25 旧 handoff 2 = 14 ファイル + report/ 35 ファイル + data/v18 古い 10 ファイル ≒ **60 ファイル** | 🟡 中 | 30min | 低 |
| **data/v18/index.md 新設** | 59 ファイルの役目 / 鮮度 / 5/9 必須 を 1 表化 | 🟠 高 | 30min | 低 |
| **data/results/index.md 新設** | 16+ ファイルのうち 5/9 朝順番に開く 5 ファイル明示 | 🟠 高 | 15min | 低 |
| **memory/MEMORY.md 新設** | `C:/Users/takum/.claude/projects/.../memory/` 空、最重要 5 項目 (V15 / 戦略⑦ / 撤退ライン / cumulative / リークフリー) を 1 page 化 | 🟠 高 | 30min | 低 |
| **README に運用フロー** | 朝起きてやることが README に書かれていない | 🟡 中 | 30min | 低 |
| **重複情報削減** | HANDOFF v2 / lessons_learned / sessions_recap で似た内容、整理推奨 | 🟢 低 | 2h | 中 |
| **doc 命名規約** | YYYYMMDD_xxx.md と xxx_YYYYMMDD.md と SCREAMING_SNAKE.md が混在 | 🟢 低 | 1h | 低 |

### 推奨優先度

1. memory/MEMORY.md 新設 (30min、Claude Code 自動 context 圧縮対策)
2. data/v18/index.md + data/results/index.md (45min、5/9 朝の操作迷子防止)
3. CLAUDE.md V15 化 (Phase 3 移行までに必須、2h)

---

## B6. 開発体験 (DX)

### 現状

- 新 Claude Code セッション起動時:
  - CLAUDE.md auto-load (1325 行、context の 7-8% 消費)
  - HANDOFF_5_5_TO_5_9.md (420 行、手動 read 推奨)
  - next_session_checklist.md でフロー確認
  - 必要 doc を Read tool で順次読み込み
  - **コンテキスト把握まで 5-10 分**

### アップデート候補

| 項目 | 詳細 | 緊急度 | 工数 | 着手難易度 |
|------|------|--------|------|-----------|
| **CLAUDE.md 軽量化** | 1325 行 → 400 行台に圧縮、`CLAUDE_HISTORY.md` に過去詳細を切り出し | 🟠 高 | 2h | 中 |
| **SKILL.md 的なリポジトリ専用ガイド** | `.claude/skills/` あるが keiba-ai 専用がない | 🟡 中 | 2h | 中 |
| **「30 秒復帰」コマンド** | 1 コマンドで「累計 / 直近 commit / 次タスク / 注意事項」を出力 | 🟡 中 | 1h | 低 |
| **MEMORY.md 活用** | auto memory 空 → 重要項目 5 件で context 圧縮対策 | 🟠 高 | 30min | 低 |
| **session 切替時のコマンド整理** | next_session_checklist.md 既存だが、checklist 内のコマンドを `tools/session_start.sh` 化 | 🟡 中 | 1h | 低 |
| **Streamlit 起動状態の可視化** | "今 streamlit 動いてる?" を 1 コマンドで | 🟢 低 | 30min | 低 |
| **5/24 までに完了したい順序を 1 page** | UPDATE_INVENTORY § 7 で実現済 | ✅ | - | - |

### 推奨優先度

1. memory/MEMORY.md 新設 (B5 と重複、30min)
2. CLAUDE.md 軽量化 (Phase 3 移行までに、2h)
3. 「30 秒復帰」コマンド (1h、ユーザーが朝最初に叩く)

---

## C. 統合 推奨アクション

### 今夜寝る前: ❌ なし
本書を生成して push するだけ。 51.5 時間連続作業の終点として、これ以上は朝に持ち越し。

### 5/6 (火) admin 1 コマンド + 隙間時間 (合計 ~3h で済む quick wins)

| # | タスク | 工数 | 効果 |
|---|--------|------|------|
| 1 | ProcessWatchdog v2 admin 実行 | 1min | 監視層復活 |
| 2 | 累計閾値 Discord alert 実装 | 30min | -10k/-30k/-50k 自動通知 |
| 3 | memory/MEMORY.md 新設 | 30min | Claude session 圧縮対策 |
| 4 | data/v18/index.md + data/results/index.md | 45min | 操作迷子防止 |
| 5 | Cookie 失効時 Discord alert | 30min | 最大の運用リスク減 |

### 5/7 (水) - 5/8 (金) 平日

| # | タスク | 工数 | 効果 |
|---|--------|------|------|
| 6 | 累計収支リアルタイム dashboard | 1h | 朝 1 開く |
| 7 | テストカバレッジ計測 | 30min | 現状把握 |
| 8 | silent fail grep + 対処 | 2h | エラー黙殺撲滅 |
| 9 | disaster recovery procedure | 3h | PC 故障対策 |
| 10 | backup 戦略 (Google Drive sync) | 2h | データ消失対策 |

### 5/24 までに

- CLAUDE.md V15 化 (2h)
- 古い doc archive (30min)
- validation_1〜13 v15 対応 (6h)
- JRA 配当 fallback (3h)
- netkeiba UA rotation (6h)

### Phase 3 (5/25+) に向けて

- migrate_to_new_pc 手順書整備
- Streamlit page 拡充 (累計 / drift / schtasks / SLO)
- structured logging 移行
- Docker 化検討 (低優先度)

---

## D. 結論 (2 句)

**keiba-ai は「動作健全 / 運用安定 / 観測良好」だが「ドキュメント肥大 / 復旧 procedure 暗黙 / 単一 source 依存」が改善余地**。 51.5 時間で運用基盤は完成、次の 1 週間 (5/6-5/12) に DX 改善 (memory / index / dashboard / alert) を集中投下すれば、Phase 3 移行時の心理負担が大幅に減る。

**最重要の 1 件**: `memory/MEMORY.md` 新設 (30min、効果絶大)。 Claude Code の auto-compaction で失われやすい「V15 ベースライン / 戦略⑦ / 撤退ライン / 累計 +14,140 / リークフリー 8 features」を 1 page で永続化すれば、毎セッションの context 把握が 5-10 min → 1-2 min に短縮可能。
