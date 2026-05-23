# 2026-05-23 整合性監査レポート

作成日時: 2026-05-23  
監査スコープ: 本日 (2026-05-23 00:00 〜 23:59 JST) の全 git commit + V15 production 不変確認

---

## 1. 本日の commit 一覧 (時系列昇順)

| 時刻 (JST) | Hash | 内容 |
|-----------|------|------|
| 09:34 | `366a3b70` | [5/23 朝 9 時確認] 統合 status doc (read-only audit) |
| 09:49 | `8447fab4` | [critical fix] Python Store stub → pythoncore-3.14-64 (11 bat) / LiveOrchestrator + AnomalyCheck 修復 |
| 09:50 | `93f71d7a` | [5/23 stub fix doc] Python path 修正 + LiveOrchestrator 手動 fire 手順 |
| 09:57 | `76acd2ed` | [5/23 race_auto_notify 復旧] 09:53 再起動成功 / 34R 残存 / 2R 通知ミス |
| 10:06 | `1f9e3d60` | [確-2] 5/23 その他確認 (schtask/paper/TYB/通知) |
| 10:11 | `1a29c27c` | [確-1] V15 特徴量取得漏れ audit 5/23 |
| 11:00 | `1cd1d2ce` | [feature audit] OZ/PACI/KTA root cause + 修正方向 |
| 11:21 | `65a00bbc` | [JRDB再検証] KTA/KKA/PACI/OZ 取得側 bug vs 真の非提供 判定 |
| 12:36 | `bd394d16` | [data取得修正] daily_jrdb_kyi.bat に OZ/SR 追加 / PACI 週次取得開始 / V15 予測非反映 |
| 12:47 | `3d801790` | [schtask確認] AnomalyCheck 5件確認 / register bat 文字化け修正 |
| 12:54 | `33a0f8f8` | [stub fix 拡張] 8 bat 真 path 化 / WindowsApps 残存 0 件確認 |
| 13:23 | `ba5b0e44` | [V21-parallel] V15/V21 並行予測スクリプト + paper Discord 送信 / 実 cash は V15 のみ |
| 13:30 | `52c69214` | [V21-create] candidate model 作成 (V15+TYB10) WF AUC=0.8696 / V15 不変 |
| 13:48 | `4155f681` | [V21 per-race] 発走 17 分前 V21 paper 通知スケジューラ + TYB 取得 / V15 不変 |
| 13:58 | `35cb78f2` | [V21 paper live] 5/23 残り 11R で実機テスト / PID 20476 / V15 PID 28500 並行稼働 |
| 14:02 | `c9db163c` | [V21 paper] 送信先を #買い目 に変更 / 投票禁止を 🚫🚫🚫 で強調 |
| 14:30 | `93503344` | [V21 paper] 毎レース通知に変更 (filter 除外 R も送信) |
| 14:41 | `222bd514` | [V21 paper] 全馬スコア形式に変更 / V15 format 統一 |
| 15:29 | `2dec4da6` | [V21 paper] TYB fetch 修正 (JRDB_USER → JRDB_ID fallback) |
| 15:30 | `3d76c26d` | docs: 5/23 V21 paper 2 問題デバッグ記録 (重複/TYB 未取得) |
| 21:02 | `dbc258d8` | [V21 paper] TYB 完全修正 (7z path + inject tansho/fukusho) + 3doc |
| 21:21 | `25a3eca5` | [言語化] オオタニサーン / ペッパーミル 選定理由分析 (新潟 12R) |
| 21:38 | `a81c5a95` | [audit] paci_ninki_idx 正体確認 + V15 真の人気依存度 (~18.1%) 解明 |
| 22:04 | `778e64db` | [言語化] オオタニサーン vs ペッパーミル 根拠の質 対比 |

**合計 24 commit**

---

## 2. V15 production 不変確認

### 2-1. モデルファイル存在チェック

| ファイル | パス | サイズ | タイムスタンプ | 判定 |
|---------|------|--------|-------------|------|
| keiba_model_v15_central_live.pkl.gz | `/` 直下 | 2,099,610 bytes | **2026-04-08 23:32** | OK |
| keiba_model_v15_central.pkl.gz | `/` 直下 | 2,099,552 bytes | **2026-04-08 23:32** | OK |

- **本日 (5/23) の変更なし** — タイムスタンプが 4/8 のまま完全不変。

### 2-2. tools/predict_core.py 変更確認

```
git log --since="2026-05-23 00:00" -- tools/predict_core.py
→ 出力なし (変更 0 件)
```

- **本日の変更なし**。前回変更は 5/13 (`ff1ef02b`: FutureWarning 修正) のみ。

### 2-3. app.py 変更確認

```
git log --since="2026-05-23 00:00" -- app.py
→ 出力なし (変更 0 件)
```

- **本日の変更なし**。

### 2-4. app.py 構文チェック

```
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"
→ 正常終了 (exit code 0)
```

- **構文エラーなし**。

### V15 production 不変確認: **PASS**

---

## 3. V21 関連変更の整合性

### 3-1. 新規ファイル存在確認

| ファイル | パス | サイズ | タイムスタンプ |
|---------|------|--------|-------------|
| v21_per_race_paper.py | `tools/` | 31,155 bytes | 2026-05-23 21:00 |
| tyb_shadow_fetcher.py | `tools/` | 28,036 bytes | 2026-05-23 20:59 |
| v21_paper_predict.py | `tools/` | 23,498 bytes | 2026-05-23 13:23 |
| v21_candidate.pkl.gz | `models/` | 2,076,938 bytes | 2026-05-23 13:29 |
| train_v21_candidate.py | `train/` | — | 2026-05-23 |

全ファイル存在確認: **OK**

### 3-2. V21 → predict_core / app.py への影響

**v21_per_race_paper.py**:
- `predict_core` の import は **lazy import** (try ブロック内、関数呼び出し時のみ)
- `load_models()` を呼ばない。V21 candidate model (`models/v21_candidate.pkl.gz`) のみ使用
- V15 モデルを一切ロード・変更しない。コメント: *"V21 model dict is passed in — V15 load_models() is NOT called."*

**v21_paper_predict.py**:
- `predict_core.load_models()` を呼ぶが、predict_core 自体の変更なし → V15 production と同一ロジックで特徴量構築
- paper 送信のみ (実投票なし)。Discord メッセージに `🚫🚫🚫 投票禁止` 明示

**tyb_shadow_fetcher.py**:
- JRDB TYB データ取得専用モジュール。predict_core / app.py に対する副作用なし

**変更された非 V21 ファイル**:
- `tools/*.bat` (Python Store stub → pythoncore-3.14-64 への path 修正) — 予測ロジック無関係
- `friday_weekend_scrape.bat` / `tools/daily_jrdb_kyi.bat` — スクレイピング bat。predict_core 非依存
- `docs/*.md` (新規ドキュメントのみ)

### 3-3. V21 candidate model スペック確認

```
[V21-create] commit (52c69214) より:
- features: V15 145 + TYB 10 = 155
- 5-fold WF (2021-2025): 0.8667 / 0.8684 / 0.8704 / 0.8722 / 0.8704
- mean WF AUC = 0.8696 (+0.0018 vs V15 genuine 0.8678)
- LEAK gate: max |corr| = 0.4564 < 0.5 → PASS
- paper trading のみ、production 投入なし
```

### V21 整合性: **PASS** (V15 production に対する副作用なし)

---

## 4. 本日作成された docs/ ファイル一覧

| ファイル名 | 内容 |
|-----------|------|
| `5_23_MORNING_9AM_STATUS.md` | 朝 9 時統合 status 監査 |
| `5_23_PYTHON_STUB_FIX.md` | Python Store stub 修正手順 |
| `5_23_RACE_NOTIFY_RECOVERY.md` | race_auto_notify 09:53 復旧記録 |
| `確-1_V15_FEATURE_COMPLETENESS_5_23.md` | V15 特徴量取得漏れ audit |
| `確-2_5_23_OTHER_STATUS.md` | schtask/paper/TYB/通知 確認 |
| `5_23_FEATURE_MISSING_ROOTCAUSE.md` | OZ/PACI/KTA 取得漏れ root cause |
| `JRDB_AVAILABILITY_RE_AUDIT_5_23.md` | JRDB 提供可否 再検証 |
| `5_23_FEATURE_FETCH_FIX.md` | OZ/SR 取得追加 + PACI 週次化 |
| `5_23_ANOMALY_REGISTER_VERIFY.md` | AnomalyCheck schtask 確認 |
| `5_23_FRIDAY_SCRAPE_STUB_FIX.md` | FridayWeekend bat 修正 |
| `V21_PARALLEL_PAPER.md` | V15/V21 並行予測設計 |
| `V21_CANDIDATE_CREATED.md` | V21 candidate 作成記録 |
| `V21_PER_RACE_PAPER.md` | per-race paper 通知スケジューラ設計 |
| `5_23_V21_PAPER_LIVE_TEST.md` | V21 実機テスト記録 |
| `5_23_V21_PAPER_2ISSUES.md` | 重複/TYB 未取得 2 問題デバッグ |
| `5_23_NOTIFY_AUDIT.md` | 通知 audit |
| `ODDS_DEPENDENCY_ANALYSIS.md` | オッズ依存度分析 |
| `PREDICTION_REASONING.md` | 予測根拠 |
| `REASONING_OOTANI_PEPPER.md` | オオタニサーン / ペッパーミル 選定理由 |
| `PACI_NINKI_TRUTH.md` | paci_ninki_idx 正体 + V15 人気依存度 ~18.1% |
| `OOTANI_VS_PEPPER_BASIS.md` | オオタニサーン vs ペッパーミル 根拠の質 対比 |

**合計 21 ファイル (全て docs/*.md 追記のみ、コードの変更なし)**

---

## 5. テスト結果

```
python -m pytest tests/regression_test.py -v --tb=short
```

```
================ 23 passed, 1452 warnings in 201.85s (0:03:21) ================
```

- **23 tests PASSED** / 0 failed / 0 error
- warnings は predict_core.py の PerformanceWarning (DataFrame fragmentation) — 既知の既存 issue、機能への影響なし

---

## 6. 警告・注意事項

### [軽微] PerformanceWarning (既存問題)
- `tools/predict_core.py` の DataFrame fragmentation (insert 多用) が 1452 件 warning として検出
- 機能的影響なし。既存 issue (4/27 修正済みも一部残存)

### [注意] paci_ninki_idx = odds-derived 確認済み
- `a81c5a95` commit で PACI の `ninki_idx` フィールドが単勝人気 (= odds-derived) であることを確認
- V15 において `paci_ninki_idx` 特徴量は実質的に **人気順位の proxy** として機能
- V15 人気依存度 (広義) は **~18.1%** と算出 (重要: V15 本番ロジックへの即時変更なし)
- V20/V21+ での SKB 除外と同様の観点で V22+ 学習時に検討が必要

### [情報] race_auto_notify 通知ミス 2R
- 09:53 復旧前に新潟 1R / 東京 1R の通知が欠落
- V15 production の予測・投票ロジック自体には影響なし

---

## 7. 総合判定

| 項目 | 判定 |
|------|------|
| V15 モデルファイル不変 | **PASS** |
| predict_core.py 本日変更なし | **PASS** |
| app.py 本日変更なし | **PASS** |
| app.py 構文チェック | **PASS** |
| V21 ファイル存在確認 | **PASS** |
| V21 → V15 への副作用なし | **PASS** |
| regression tests 23/23 | **PASS** |

**総合: PASS — V15 production は完全不変。V21 は paper trading 専用として正常に分離稼働。**
