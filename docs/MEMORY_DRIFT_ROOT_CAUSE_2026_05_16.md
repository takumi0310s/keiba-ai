# memory drift root cause analysis (P0-1 派生 / sub-task 5-2)

**作成**: 2026-05-16 (Sub-task 5-2、 read-only audit、 fabrication なし)
**対象**: CLAUDE.md / memory に残った stale な数値 (+¥13,530 / 119.2% 等) の発生・伝播・update 失敗 root cause
**情報源**: git log / git show (直接引用)、 docs/ROI_DISCREPANCY_2026_05_16.md (P0-1)、 data/v18/phase21x_cumulative_audit.md (Session #87)、 data/v18/may_2_3_truth_audit_5_6.md (5/6 audit)

---

## 0. 結論

| 項目 | 値 |
|------|------|
| **+¥13,530 初出 commit** | `bc2d998c` (2026-05-06 17:38、 Session #31 E、 CLAUDE.md 緊急訂正) |
| **119.2% 初出 commit** | `364a9260` (2026-04-27 09:27、 v16 prep + strategy7) |
| **真の出典 (+¥13,530)** | `data/v18/may_2_3_truth_audit_5_6.md` (5/6 朝、 5/5 夜時点 戦略⑦ filter 後 真値) |
| **真の出典 (119.2%)** | CLAUDE.md L1187「実戦成績 3/14-4/18 dedup 324 R、 全体 120.2% (+¥45,920)」 を rounded 引用回し (P0-1 §1.2 で特定) |
| **root cause** | **仮説 D + E の併発**: 手動 update 慣行 (D) + 自動 drift 検知 機構の不在 (E)。 cumulative_results.csv は 自動更新されるが、 CLAUDE.md / memory への 反映は 全て **commit message 内手動転記**、 真値変動時の chained update なし |
| **影響期間** | **2026-05-06 〜 2026-05-16** (約 10 日、 CLAUDE.md +¥13,530 が stale で残存。 Session #87 (5/10 23:50) で update 推奨済だったが実行されず) |
| **drift 該当箇所** | CLAUDE.md 3 箇所 (L77 / L1271 / L1347) + data/v18/ 配下 21 docs (copy-paste 伝播) |

---

## 1. 初出 commit 追跡

### 1.1 119.2% (ROI)

```
$ git log -S "119.2" --oneline -- CLAUDE.md
c9c9d3d8 Session #39 H: CLAUDE.md 全面刷新
bc2d998c Phase 2.5+ E: CLAUDE.md 緊急訂正 (v15 反映 + v16 二重重複削除)
364a9260 v16 prep: strategy7 + course_renovated + FutureWarning + 1race-predict
```

| commit | 日付 | 役割 |
|--------|------|------|
| **`364a9260`** | **2026-04-27 09:27** | **初出**: race_auto_notify.py 戦略⑦適用時の commit message に `Expected ROI: 119.2% -> 140.3%` 記載、 同 commit で CLAUDE.md にも「本番運用 ROI 119.2% (298R)」を追加 |
| `bc2d998c` | 2026-05-06 17:38 | Session #31 緊急訂正で再記載 |
| `c9c9d3d8` | (Session #39 H) | CLAUDE.md 全面刷新で継承 |

→ **真の根 (P0-1 §1.2)**: CLAUDE.md L1187「実戦成績 3/14-4/18 dedup 324 R / 全体 120.2%」 を rounded して 119.2% に変換。 5/5 以降の追加 race を含めると ROI 急落 (≤5/10 で 93.23%、 P0-1 §2) のため、 119.2% は **4/18 以前の snapshot で固定**。

### 1.2 +¥13,530 (累計 PnL)

```
$ git log -S "13,530" --oneline -- CLAUDE.md
c9c9d3d8 Session #39 H: CLAUDE.md 全面刷新
bc2d998c Phase 2.5+ E: CLAUDE.md 緊急訂正 (v15 反映 + v16 二重重複削除)
```

| commit | 日付 | 役割 |
|--------|------|------|
| **`bc2d998c`** | **2026-05-06 17:38** | **初出** (commit message): 「累計 +13,530 円 / 撤退余裕 +63,530 円 を追加」 |
| `c9c9d3d8` | (Session #39 H) | 継承 |

→ **真の根**: `data/v18/may_2_3_truth_audit_5_6.md` L65 (5/6 朝 Session #27 audit) で 「JRA cumulative_results USER filter 後 全期間 +13,220 + NAR 柏記念 +310 = **5/5 夜時点 累計 +13,530 円**」を **生データ集計** で確定。 USER 申告 +14,140 円との差 ±610 円も同 doc で説明。 当該 audit doc は **戦略⑦ filter (京都 + E + B 除外) 適用後の戦略⑦ filter 累計**。

### 1.3 全 file 横断 (+¥13,530)

```
$ git log -S "13530" --oneline --all | head -3
db32bb2f 3 並列 sub-task 完了: P0-1 真値確定 + baseline 凍結 + master doc 更新
3f663d37 TOS review + .gitignore 強化 + 大 csv index 削除 (push 修復 Step A、 AI 自律 完了)
35edca11 Tier 1-1: netkeiba_master_index expanding 化 (LEAK-free 5 features)
```

→ 5/15 以降 docs 多数で言及するが、 全て CLAUDE.md からの **copy-paste** (転写)。 真値は 5/5 夜固定の stale 値。

---

## 2. data/cumulative_results.csv の snapshot

### 2.1 git history

```
$ git log --oneline -- data/cumulative_results.csv | head -5
8cee3543 Phase 9 強化版: 5/10 全 35R 完全照合 (csv は git tracked、 直接 modify なし)
66c78e9e Phase 2.5+ 総点検 (csv は git tracked、 直接 modify なし)
364a9260 v16 prep: strategy7 + course_renovated + FutureWarning + 1race-predict
d2a63752 fix: payout取得バグ完全修正 (actual_payout キー欠落)
ab110ab8 fix: daily_results.py CSV対応 + バックフィル機能 + ROI再計算 (4/18 23:23)
```

### 2.2 row 数 snapshot

| commit | 日付 | rows (header 込) | settled n |
|--------|------|-----|------|
| `ab110ab8` | 2026-04-18 23:23 | 325 | ~324 (4/18 末日) |
| `52ba0599` | 2026-04-19 00:56 | 325 | ~324 |
| `d2a63752` | 2026-04-23 21:02 | 360 | ~359 |
| `364a9260` | 2026-04-27 09:27 | 429 | ~428 |
| 現在 (5/16 18:00) | 5/16 | 598 | 563 |

→ csv は cron (daily_results.py) で **自動 update 継続**。 5/16 まで 1 度も停止せず。 つまり 真値計算の source は **常に更新されていた**。

### 2.3 CLAUDE.md 記載との整合性

| CLAUDE.md L1177 表 | csv 4/18 snapshot |
|-------------------|------------------|
| 「2026-03-14〜04-18, dedup後 324レース」 | settled n=324 (4/18) |
| 全体 76 hits / 23.5% / 120.2% / +¥45,920 | 一致 (P0-1 で再現確認) |

→ CLAUDE.md L1177 表は **4/18 snapshot で固定** され、 5/16 まで update なし。

---

## 3. memory entry 追跡

### 3.1 memory file 一覧 + drift 状況

| file | 該当数値 | source | drift 状態 |
|------|---------|--------|---------|
| `cumulative_pnl.md` | +¥14,140 (5/5 朝時点) | USER 手動申告 (5/5 Discord) | ★ **stale (10 日)** ★ 真値 5/10 +¥21,420、 5/16 +¥5,240 (cumulative) |
| `v15_baseline.md` | 119.2% (4/12-5/3、298R 未勝利除外) | CLAUDE.md からの転記 | ★ **stale (10 日)** ★ 5/16 真値 101.33% (n=563) |
| `strategy_7_planB.md` | 119.2% → 140.3% (+21.1pt) | race_auto_notify.py commit message (4/27) | ★ **stale (10 日)** ★ 出典自体が 4/18 snapshot |
| `v22_design.md` | quick fold AUC 0.8891 | V22 学習 result | OK (3 日前、 model 結果は drift しない) |
| `strategy_8_verified.md` | 69 件 / 53.6% top3 | 5/13 LIVE 検証 | OK (LIVE result snapshot) |
| `risk_management.md` | 撤退 -50,000 円 | 方針 (時不変) | OK |
| その他 12 file (session37/38/55/57/horse_id_mapper/leak_free_rules 等) | model 検証 / 設定値 | session-specific snapshot | OK (時不変または研究 result) |

→ ★ **drift がある memory entry は 3 件**: `cumulative_pnl.md` / `v15_baseline.md` / `strategy_7_planB.md` (いずれも累計 PnL / ROI 関連)。

### 3.2 自動更新 logic の有無

| memory file | 自動更新 | 手動更新 |
|------------|----------|----------|
| 全 17 file | **なし** (file system mtime ベース、 schtask 連携なし) | session 中の AI 手動 update のみ |

→ ★ memory file は AI が session 中に 必要時 手動 update する仕組み。 cron / schtask での 自動 sync は **未実装**。

---

## 4. root cause 分析 (仮説 A-E 検証)

### 仮説 A: daily_results.py の cumulative update logic に bug

```
$ grep -n "cumulative_results.csv" tools/daily_results.py
11:  - data/cumulative_results.csv          — 累積結果
57: CUMUL_CSV = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
418: """cumulative_results.csv を (date, race_id) キーでupsert
746: cumulative_results.csv から累計・条件別・直近30R ROI を再計算
```

検証: csv は 5/16 まで row 増加継続 (325 → 598)、 mtime 5/16 18:00 で更新。
→ ★ **NO**: csv update logic は正常動作。

### 仮説 B: WeeklyReport の数値 source が静的 (memory ハードコード)

```
$ grep -n "cumulative_results.csv" tools/weekly_report.py
158:    cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
180:    cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
```

検証: weekly_report.py は csv を 動的読み込み。 静的 hardcode なし。
→ ★ **NO**: source は動的。

### 仮説 C: 4/18 以降 cumulative 真値計算 schtask が止まった

検証: csv mtime 5/16 18:00、 settled row n=563 (5/16 京都/中山反映済)。 daily_results schtask は継続稼働。
→ ★ **NO**: schtask は止まっていない。

### 仮説 D: 手動 update 慣行で update し忘れ ★ 主因 ★

検証:
- CLAUDE.md +¥13,530 は `bc2d998c` (5/6) で書き込み、 5/10 Session #87 で「+¥13,530 → +¥21,420 に update 推奨」と明記 (`phase21x_cumulative_audit.md` §3.2 L153)
- にもかかわらず 5/10 〜 5/16 の 6 日間 で CLAUDE.md は **0 回 update** (git log --since="2026-05-11" -- CLAUDE.md → output なし)
- 5/15-5/16 の commit (LEVEL_UP_AUDIT / STACKING_HONEST / DISTILLATION_RESULT 等 21 docs) は 全て **CLAUDE.md +¥13,530 を copy-paste で参照** = 5/5 stale 値が伝播

→ ★ **YES**: AI session が CLAUDE.md update を「破壊的操作」として回避し、 docs にだけ追記する慣行が drift を増幅。

### 仮説 E: 複数 system が並行運用 (csv 真値 vs CLAUDE.md/memory snapshot) ★ 構造的主因 ★

検証:
- 真値 source: `data/cumulative_results.csv` (5/16 19:00 時点 settled n=563、 自動更新)
- snapshot source: `CLAUDE.md` (5/6 17:38 で固定、 手動 update のみ)
- memory: `cumulative_pnl.md` (5/5 朝 USER 申告で固定、 手動 update のみ)
- 真値→snapshot の 自動 sync layer **不在** → drift 検知 機構 **不在**

→ ★ **YES**: drift を 自動検知/警告する layer がない。 仮説 D の人的失敗を fail-safe で吸収する仕組みがゼロ。

### 採用 root cause

★ **仮説 D + E の併発**:
- E = 構造的: 自動 drift 検知/警告 layer が未実装、 真値→snapshot は 全て AI session 内手動 update に依存
- D = 運用的: AI session が CLAUDE.md 編集を回避 (Session #87 で update 推奨済も、 5/11-5/16 で 0 回 update)
- 結果: 真値が 5/16 までに +¥5,240 (n=563) / 101.33% に推移したのに、 CLAUDE.md は +¥13,530 / 119.2% (5/5 夜 snapshot) のまま 10 日間 残存。 約 1 ヶ月の意思決定が偽値ベースで実施された

---

## 5. drift 他例 (memory + 全 doc 横断)

### 5.1 memory file 3 件 (確定 drift)

| file | line | 記載 | 真値 (5/16 cumulative) | drift |
|------|------|------|----------|------|
| `cumulative_pnl.md` | L7 | +¥14,140 円 (5/5 朝) | +¥5,240 (5/16) | **-¥8,900** |
| `v15_baseline.md` | L13 | 本番運用 ROI 119.2% (4/12-5/3、298R) | 101.33% (n=563) | **-17.87pt** |
| `strategy_7_planB.md` | L10 | 119.2% → 140.3% (+21.1pt) | 戦略⑦ applied 96.90% (n=466) | **est. -22pt** |

### 5.2 CLAUDE.md (確定 drift)

| line | 記載 | 真値 |
|------|------|------|
| L72 | 本番運用 ROI 119.2% / 戦略⑦込み 140%+ 想定 | 101.33% / 96.90% (cumulative) |
| L77 | 累計収支 +13,530 円 / 撤退余裕 +63,530 円 | +¥5,240 / +¥55,240 |
| L1187 | 全体 324R / 76hits / 23.5% / 120.2% / +¥45,920 | 563R / 303hits / 53.82% / 101.33% / +¥5,240 (cumulative 全体、 ただし戦略⑦ off) |
| L1271 | 期待効果: ROI 119.2% → 140.3% | (est.) 93.23% → 96.90% |
| L1347 | 撤退ライン: 累計 -50,000円 (現在 +13,530円、 撤退余裕 +63,530円) | (現在 +¥5,240、 撤退余裕 +¥55,240) |
| L1363 | V15 (現状): 119.2% (戦略⑦込み 140%) | 101.33% / 96.90% |

### 5.3 data/v18/*.md (伝播 drift、 21 files)

```
$ grep -l "13,530" data/v18/*.md | wc -l
21
```

| file 例 | 影響 |
|---------|------|
| BUGFIX_5_13_PM.md, COMPREHENSIVE_AUDIT_5_15.md, DISTILLATION_RESULT_5_16.md, END_OF_DAY_5_13_SUMMARY.md, FRIDAY_5_16_READINESS_5_12.md, GIT_PUSH_FIX_5_15.md, JV_DATA_28_TYPES_SPEC.md, LEVEL_UP_AUDIT_5_15.md (×2), PREP_5_16_NEW_MODEL_5_14_AM.md, STACKING_HONEST_5_16.md, TOS_REVIEW_5_15.md, USER_SETUP_5_15_FINAL.md, V15_BEATING_ATTEMPT_5_16.md, V22_ENHANCED_RESULT_5_13_PM.md, V22_TOP100_RESULT_5_14_AM.md, V22_VS_V15_FULL_6FOLD_5_14_AM.md, V22_VS_V15_ROI_BACKTEST_5_14_AM.md, WHAT_REMAINS_5_14_AM.md, audit_roadmap_5_8.md, extended_retro_4_12_5_5_5_8.md, final_audit_remaining_tasks_5_6.md | 全て CLAUDE.md L77 / L1347 の +¥13,530 を **copy-paste で踏襲**。 5/5 stale 値が 21 doc に伝播 |

→ ★ 1 個の stale 値が 21 doc に伝播 = **single source of truth の崩壊**。

### 5.4 CLAUDE.md L1177 表 (3/14-4/18 snapshot、 stale)

| line | 記載 | 5/16 真値 (cumulative) |
|------|------|----------|
| 表全体 | 324R / 76hits / 23.5% / +¥45,920 / 120.2% | 563R / 303hits / 53.82% / +¥5,240 / 101.33% |

→ ★ 4/18 で凍結された snapshot を 「実戦成績」見出しで掲載し続け、 119.2% の引用元になっている。

### 5.5 場別 / 条件別 (P0-1 で確定済、 ここでは drift 観点で再記)

| 区分 | CLAUDE.md (旧) | 5/16 真値 (P0-1 §6) | drift |
|------|----------------|---------------------|-------|
| 京都 (戦略⑦既) | (記載なし) | 20% (N=58) | 新規確認 |
| 東京 | (記載なし) | 63.13% | 新規 worst 候補 |
| 中山 | (記載なし) | 78.69% | 新規 worst 候補 |
| 条件 X | 231.3% 保守的見積り | 8.72% (cumulative) | -222pt |
| 条件 E | 82.6% | 12.34% | -70pt |
| 条件 B | 165.8% | 26.96% | -139pt |

→ 条件別 ROI は **保守的見積りベース** (backtest × 0.7) で記載されており、 LIVE 真値とは別系統。 ただし「保守的見積り」も実測で大きく外れている事実が drift。

---

## 6. 再発防止策 (設計提案)

### 6.1 tools/daily_cumulative_audit.py (新規、 設計済)

**設計**:
- 配置: `tools/daily_cumulative_audit.py`
- 発火: schtask `Keiba-NightlySanity` の **23:00** に並列実行 (現状 nightly_sanity の後段に追加可能)
- 処理:
  1. cumulative_results.csv 真値計算 (全体 + 戦略⑦ applied 両方)
  2. CLAUDE.md / memory file の数値正規表現で抽出
  3. diff 計算 (絶対 / pt)
  4. drift があれば Discord #updates に「★ CLAUDE.md drift detected ★」 通知
- 出力: `data/_cumulative_audit_snapshot.json` (timestamp / 真値 / CLAUDE.md / memory の record)
- 工数: 約 2-3h 実装

### 6.2 tools/weekly_report.py 拡張

**追加 section**: 「真値 vs CLAUDE.md / memory drift」
- 週次 report 末尾に drift 表 + recommendation 列
- drift が ±5% pt 以上なら 赤色 warning
- 工数: 既存 weekly_report に 30 分 追加

### 6.3 CHANGELOG template ルール強化

**新 ルール** (CHANGELOG_*.md の冒頭テンプレ追加):
```
## 数値変更時の出典記載 (必須)
- 各数値 (PnL / ROI / 累計) を docs / memory に書く際は **必ず**:
  1. cumulative_results.csv の cutoff date を明記 (例: `as_of 2026-05-16`)
  2. filter 適用 (戦略⑦ etc.) の明示
  3. 再現用 1 行 code (例: `python -c "import pandas..."` で 同値が再現可能)
- 「累計 +X 円」のような bare statement は **禁止**、 必ず date + 計算 method を伴うこと
```

### 6.4 memory entry に `last_verified` field 追加

**追加 spec**:
```yaml
---
name: 累計収支 ...
description: ...
type: project
originSessionId: ...
last_verified: 2026-05-16   # ← 新規必須 field
verified_source: data/cumulative_results.csv (as_of 5/16, settled n=563)
expires_after_days: 7   # ← 推奨、 過ぎたら drift 警告
---
```

→ AI session 開始時に memory entry の `last_verified` が `expires_after_days` 超過なら 必ず 再検証してから quote すること。

### 6.5 single source of truth 化

**設計案**:
- `data/cumulative_results.csv` を **唯一の真値 source** と確定
- CLAUDE.md / memory / docs の累計 PnL / ROI 記載は **全て** csv 集計 wrapper 経由 (例: `tools/get_truth.py --metric pnl --cutoff today`)
- snapshot 形式の hard-coded 値を 全 doc から廃止
- 工数: 段階的 (P0 で CLAUDE.md / memory 5 件、 P1 で data/v18/ 21 docs)

---

## 7. 5/17+ 即時 action (recommend)

### 即日 (5/17 朝、 P0)

1. ★ CLAUDE.md L77 / L1271 / L1347 / L1363 / L1187 (5 箇所) の数値を 真値 (101.33% / +¥5,240 / 563R) に書き換え
2. ★ memory file 3 件 (`cumulative_pnl.md` / `v15_baseline.md` / `strategy_7_planB.md`) の数値書き換え + `last_verified: 2026-05-17` 追加
3. ★ data/v18/ 21 docs の +¥13,530 / 119.2% 引用に **「★ 5/5 stale 値、 5/16 真値 +¥5,240 / 101.33% に置換要 ★」** の注記追加 (各 doc の冒頭 1 行で OK、 retroactive 書き換えはせず履歴保持)

### 1 週間以内 (5/18-5/24、 P1)

4. `tools/daily_cumulative_audit.py` 実装 (約 2-3h)
5. `Keiba-NightlySanity` schtask に組込 + 5/24 まで dry-run
6. weekly_report.py に drift section 追加 (30 分)

### 1 ヶ月以内 (5/25-6/15、 P2)

7. memory entry 全 17 file に `last_verified` / `verified_source` / `expires_after_days` field 追加
8. CHANGELOG template に「数値変更時の出典記載必須」ルール導入
9. CLAUDE.md / data/v18/ から hard-coded 累計 / ROI を 全削除、 `tools/get_truth.py` wrapper 経由化

---

## 8. fabrication 防止 statement

本 doc 内の全数値は以下 source からの **直接引用**:
- git log / git show output: §1.1 / §1.2 / §1.3 / §2.1
- docs/ROI_DISCREPANCY_2026_05_16.md (P0-1): §0 / §1.2 / §5
- data/v18/phase21x_cumulative_audit.md (Session #87、 5/10 audit): §4D
- data/v18/may_2_3_truth_audit_5_6.md (5/6 audit): §1.2
- data/cumulative_results.csv (5/16 18:00 snapshot、 settled n=563): §2.2 / §5
- C:/Users/takum/.claude/projects/.../memory/*.md (5/16 read-only): §3.1

仮説検証は git log 直接出力 + 既存 audit doc 参照のみ。 推定箇所は「推定」明記。 unknown は honest に unknown 記載。
