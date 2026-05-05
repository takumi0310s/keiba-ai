# 引き継ぎ書 v1 vs v2 差分 — 誤情報訂正

生成: 2026-05-05 17:30 (Phase 2.5+ Session #15)

**v1 source 候補**: `docs/weekly_handoff_20260425.md` + GW計画書群 (実体は session 内 oral handoff も含む)
**v2 target**: `docs/HANDOFF_5_5_TO_5_9.md`

---

## 1. 要旨

v1 (5/3 以前) で流通していた数字や前提に複数の誤情報があり、それを 5/3-5/5 の Session #1〜#14 で順次 検証/訂正した。本 doc は v1→v2 の主要差分を記録する。

**教訓**: 数字は **生データで再検証** しないと session 越しに誤情報が伝播する。引き継ぎ書 v2 では全ての数字に **データ source パスを併記** する。

---

## 2. 訂正された誤情報

### 2.1 training_times 行数

| 項目 | v1 (誤) | v2 (正) | source |
|------|--------|--------|--------|
| training_times.csv 全 rows | (記載なし or 過小) | **955,580** | `data/training_times.csv` 直接 wc -l |
| 2025+ rows | "2,551 件" | **192,296** | `pd.read_csv(...).query(date>=2025)` |

**訂正経緯**: Session #7 (commit e20bbc0c) `data_coverage_audit_5_4.md` で再カウント。
**含意**: 2025 訓練データは v1 主張の ~75 倍ある。V15 Pattern B 学習 base はこちらが正。

### 2.2 5/2 損失額

| 項目 | v1 (誤) | v2 (正) | source |
|------|--------|--------|--------|
| 5/2 USER 投資 損失 | "-23,800円" | **-8,820円** (実投資ベース) | USER 直接報告 |
| cumulative_results.csv 5/2 仮想ROI | -15,690円 (全33R) | 同 | `data/cumulative_results.csv` |

**差の正体**: cumulative_results.csv は **全 33R BATCH 仮想 ROI**、USER 実投資は **subset** (案B改 適用前の汚染日含む選択投資)。
**訂正経緯**: Session #5 で USER 報告と cumulative の乖離を発見。
**含意**: モデル評価 (cumulative theoretical) と USER 損益 (actual investment) を **絶対 混同しない**。

### 2.3 v15 batch ROI 31.3%

| 項目 | v1 解釈 (誤) | v2 解釈 (正) | source |
|------|------------|------------|--------|
| "5/3 v15 batch ROI 31.3%" | USER 実 ROI と誤解されてた | **全 34R 仮想ベース ROI** (BATCH 全買い理論値) | `data/cumulative_results.csv` 5/3 |

5/3 cumulative_results.csv: 34R 全買い → inv 23,800円 / pay 7,450円 / **ROI 31.3%**。
USER 実投資 (案B改 12R 1勝のみ 1R 採用): inv 700円 / pay 3,680円 / **ROI 525.7%** / **+2,980円**。

**含意**: "ROI 31.3%" は理論値であって 実 USER 損益ではない。**healthy 4日 案B改 ROI 161%** が USER 戦略の正しい期待値。

### 2.4 TYB publish 時刻

| 項目 | v1 (誤) | v2 (正) | source |
|------|--------|--------|--------|
| "TYB 17:00 公開" 想定 | 確実 17:00 | **不明 (5/3 14:50/17:00 試行 共に 404)** | `data/tyb_publish_log.csv` |

**訂正経緯**: Session #8 (commit 5262e0c0) `tyb_publish_monitor.py` で時刻分布観測開始 (5/4-5/10 蓄積中)。現時点 (5/5) で 5/4 12:25 / 5/9 12:25 共に 404。
**含意**: TYB midday 戦略は **5/9 では使わない**。5/4-5/10 観測完了後 (5/11 月) に再判定。

### 2.5 NAR モデル AUC

| 項目 | v1 (誤) | v2 (正) | source |
|------|--------|--------|--------|
| NAR モデル AUC | "0.789" 想定 | **0.8145** (v4 復活) / 0.8519 (OOS 2025) | `data/nar/models/keiba_model_nar_v4.pkl` |

**訂正経緯**: Session #12 (commit e5f71cfa) `archive/nar/keiba_model_nar_v4.pkl` を `data/nar/models/` に復活、5/5 柏記念で動作確認。OOS 再現は Session #13 (commit 57029ff1)。
**含意**: NAR は 0.789 想定で 5/16 試行を諦めるべきではない。0.8145 の Pattern B が利用可能、AUC OOS 0.8519 で実用域。

### 2.6 累計収支

| 項目 | v1 (誤) | v2 (正) | source |
|------|--------|--------|--------|
| 5/5 朝時点 USER 累計 | "約 -25,000円" (5/9 final_plan.md 旧) | **+14,140円** (USER 直接報告) | `data/results/20260505_kashiwa_kinen.md` 内記載 |
| cumulative_results.csv 4/12〜 全買い仮想累計 | 同上 | -28,360円 (495R 仮想) | `data/cumulative_results.csv` 集計 |

**差の正体**: USER 案B改 (subset 投資) では +14,140円、全買い仮想では -28,360円。
**訂正経緯**: Session #6 (commit 660b13a6) で `20260509_final_plan_v2.md` 作成時に修正。
**含意**: 撤退ライン -50,000円 まで余裕は **64,140円** (-50,000 - +14,140 の絶対値)。-25,000 想定だと 25,000円 余裕しかない誤った危機感だった。

### 2.7 5/3 USER 損失

| 項目 | v1 (誤想定) | v2 (正) | source |
|------|-----------|--------|--------|
| 5/3 USER 損益 | (未記録) | **-520円** (USER 報告) or **+2,980円** (案B改 1R 採用ベース) | USER 報告 |

5/3 の v1 推定は session #1 (commit 5fdfc2d0) で "26R 投資" 想定だった。実際 USER は 6R 程度の投資 で -520円。
**含意**: 5/3 は実損失 軽微、案B改 を 12R 1勝のみに絞れば **+2,980円** 利益も可能だった (結果 retrospective)。

---

## 3. v1 で正しかった情報 (継続使用)

| 情報 | v1 値 | 確認 source |
|------|-------|-------------|
| 撤退ライン | -50,000 円 | 全 session 一貫 |
| Anytime Fitness Japan の業務情報 | 業務時間/役割等 | USER 個別事情 (本書スコープ外) |
| GW 中の作業計画 大枠 | 5/3-5/5 集中作業 | 実施済 |
| V15 Pattern B 採用 (主モデル) | AUC 0.8939 | `keiba_model_v15_central_live.pkl.gz` |
| morning_top_races 06:30 自動化 | task 登録 | `Keiba-Morning_Sat` Ready 確認 (Session #14) |
| Cookie 自動 refresh 必須 | `tools/refresh_cookie.py` | Session #2 完了 |
| 静音化 必要性 | bat 直接呼び出し問題 | Session #9 (16 task 静音化) |

---

## 4. v2 で **新規** 確立された前提

| 項目 | 値 | source |
|------|----|--------|
| BT vs production prob distribution shift | **27.7x** scaling factor | `data/v18/distribution_shift_analysis.json` (Session #10) |
| race-level normalization | softmax T=1.0 推奨 | `data/v18/race_normalize_5_4_result.md` |
| v18/v19 calibration (Platt scaling) | max prob 0.154→0.213 (不十分) | `data/v18/calibration_5_4_result.md` (Session #8) |
| sc_score / ra_score 復活 | jra_races_full 2026年4-5月分追加で復旧 | Session #10 (b4c4894c, 6b5e4e7b) |
| jra_races_full 2026 missing | 38日 stale → 2026年4-5月分追加で復旧 | Session #10 |
| static FT/IR model state | None 許容 (v15以降) | `regression_test.py` PASS |
| 静音化 wscript ラッパー | `tools/silent_runner.vbs` | Session #9 |
| schtasks 23 件総合 | 16 既存 + NAR 5 + RaceDayReport 2 | Session #14 + #15 |

---

## 5. 重要な未解決 (v2 で記述、5/16+ で対応)

| 項目 | 内容 | 対応予定 |
|------|------|---------|
| winner_top1 rate 13pt 劣化 | BT 47.8% → 5/2-5/3 retro 34.5% (calibration では解消されない) | 5/16+ feature distribution 調査 |
| TYB 公開時刻 確定 | 5/4-5/10 観測完了後 (5/11 月) | TybPublishMonitor 結果待ち |
| ra_score / sc_score 完全復活 | 一部 race のみ復元、全レース は 5/12+ scrape 待ち | Session #10 部分復旧、継続 |
| chihou_races_2020_2025.csv 不在 | NAR strict OOS 評価 不能 | 別 session 60min |
| v18/v19 真の calibration | normalize は monotonic で根本治療ではない | 5/16+ paper trading 後判断 |

---

## 6. 結論

**v1 → v2 の主要訂正 7 件、新規前提 9 件、未解決 5 件**。

v2 (`docs/HANDOFF_5_5_TO_5_9.md`) では:
- 全数字に **source path** を併記
- 「USER 実投資」と「全買い仮想」を **明確に分離**
- 引き継ぎ書 v1 の旧情報を **そのまま再生産しない** (本書を参照)
