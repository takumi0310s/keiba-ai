# Session #75: Dev Branch 一括 Audit

**実施日**: 2026-05-09 18:30+
**main HEAD**: `5f5c3d43` (Session #71)
**対象**: 13 dev branches

## サマリ

- **総 dev branch 数**: 13
- **merge 推奨**: 7 (sprint1, sprint2, sprint6-kka, training-poc, two-stage, audit-backtest, video-poc)
- **archive 推奨**: 5 (sprint4, nar-v5, v20-expanding, v20-interaction, session-58-audit)
- **V20 素材保持**: 1 (v20-ensemble、 6/8 まで)
- **conflict 状況**: 全 13 branch ALL CLEAN (main 単独 merge dry-run)

## 詳細 audit

| Branch | HEAD commit | ahead | behind | 主要内容 | 判定 |
|---|---|---|---|---|---|
| dev/sprint1 | 4f04bdd9 | 6 | 6 | Sprint 1 統合 (jump model / dynamic kelly / odds flow / race skip / auto ticket) | **MERGE** |
| dev/sprint2 | d8c39d1a | 9 | 5 | Sprint 2 統合 (jump v2 / maiden / paddock / horse weight / jockey net / running style / post race) | **MERGE** (LEAK 検証後) |
| dev/sprint4 | 466b7d65 | 9 | 3 | V15.5 PoC AUC 0.8685 (V15 0.8688 比 -0.0003) | **ARCHIVE** (NO-GO) |
| dev/sprint6-kka | 06dfe02a | 5 | 3 | Session #53 KKA parser 修復 + V20 候補 12-15 features | **MERGE** (race_id 調整) |
| dev/training-poc | da53ec35 | 38 | 5 | Session #47-#67 巨大累積 (horse motion / total score / video / 5system / KKA audit etc) | **MERGE** (基盤、 38 commits) |
| dev/two-stage | 531d80cb | 19 | 4 | Session #65 + #68 + #72 Stage 2 framework (stage_compare + two_stage_predict) | **MERGE** |
| dev/audit-backtest | be08b1a8 | 17 | 3 | Session #69 + #70 backtest + 三連複 7 vs 11 + LEAK verification | **MERGE** |
| dev/nar-v5 | e48ace0b | 5 | 3 | NAR V5 学習 + audit (NO-GO 確定) | **ARCHIVE** |
| dev/session-58-audit | f48972e7 | 1 | 3 | Discord 重複 audit + 公式 1 通 summary (1 file 一時 doc) | **ARCHIVE** |
| dev/v20-ensemble | f654a68c | 5 | 3 | V20 4-model ensemble + FT-Transformer + IntraRace (V20 本命素材) | **KEEP** (6/8 まで) |
| dev/v20-expanding | 19d25bfb | 6 | 3 | V20 expanding 化 PoC (delta -0.0000、 NO-GO) | **ARCHIVE** |
| dev/v20-interaction | facbdaed | 8 | 3 | V20 interaction PoC (V15 既に飽和、 -2bp〜+1.8bp) | **ARCHIVE** |
| dev/video-poc | 91226da7 | 1 | 4 | Phase 4 video pipeline (download / yolo / keypoint / aggregate) | **MERGE** (Phase 4 で活用) |

## Critical 安全確認

- **predict_core.py**: 全 13 branch で **未変更**
- **daily_predict.py**: 全 13 branch で **未変更**
- **app.py**: 全 13 branch で **未変更**
- **race_auto_notify.py**: 全 13 branch で **未変更**
- **train/features_v15_new.py**: 全 13 branch で **未変更**

→ **V15 production logic 完全不変保証 OK**。 merge しても本番影響ゼロ。

## 各 branch の影響範囲

### Tools 追加 (新規 only)
- sprint1: 7 tools (auto_ticket / dynamic_kelly / jump_race / odds_flow / race_skip + tests)
- sprint2: 9 tools (jump v2 / maiden / paddock / horse_weight / jockey_net / running_style / post_race + interval)
- sprint6-kka: 3 tools (kka_parser_v2 / kka_features / kka_quality_check)
- two-stage: 3 tools (stage_compare / two_stage_predict / pre_race_predict_runner.bat + test)
- audit-backtest: 6 tools (audit_combo / audit_top27 / sanrenpuku_7vs11 / session_70_filter / session_70_markdown / session_70_summary)
- training-poc: 多数 (horse motion / video downloader / 5system / process_watchdog_v2 etc)
- video-poc: 5 tools (video_pipeline/__init__ + download + features_aggregate + keypoint_extract + main_pipeline + yolo_inference)

### Data 追加 (docs / json / csv のみ)
- 全 archive 候補も含めて `data/v18/` 配下の md / json は merge OK
- model 追加: dev/nar-v5 の `data/nar/models/keiba_model_nar_v5.pkl` (archive 後も保持可能)
- V20 model: dev/v20-ensemble / v20-interaction の `data/v20/models/*.pkl` (V20 構築用、 6/8 まで保持)

## 結論

**5/15 merge 準備 完了**。
- 7 branch を順次 merge → 約 110 ファイル追加 (全て新規 tools / docs、 既存 file 改変なし)
- 5 branch archive → tag 化で履歴保持
- 1 branch (v20-ensemble) を 6/8 V20 投入まで保持
