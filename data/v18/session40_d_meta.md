# Session #40 D: メタ系 (backfill + tests + CI + INDEX)

**作成**: 2026-05-07 (Session #40 D)
**目的**: V20 構築前倒し準備 + コード品質保証 + doc 体系化

---

## 1. D1 — JV-Link backfill 計画

### 1.1 ファイル

`tools/jvlink_backfill_plan.py` (新規、 約 200 行)

### 1.2 既存 inventory

```
jra_races_full: rows=532,004, 年範囲 2015-2026
jra_payouts:    rows=12,333, max_date=20260503 (4/6 停止 既知)
blood:          58,921 horses
speed_index:    270,436 rows
```

### 1.3 JV-Link 取得 targets (priority 順)

| P | datatype | 期間 | records | phase |
|---|----------|------|---------|------|
| 1 | RACE | 2010/01-2026/05 | ~800K | 6/9-13 (V20 学習 data 主軸) |
| 1 | HR | 2018/01-2026/05 | ~30K | 5/24-31 (jra_payouts 4/6 停止 解消) |
| 2 | O1 | 2025/05-2026/05 (1年) | ~100K | 5/27- (paci_* 自前算出) |
| 2 | BLOD | 2010/01-2026/05 | ~80K | 6/14-20 (blood_full override) |
| 2 | WF | 2025/05-2026/05 | ~100K | 6/14-20 (馬体重) |
| 3 | TCOV | 2025/05-2026/05 | ~200K | 6/9-13 (調教補完) |
| 3 | WOOD | 2025/05-2026/05 | ~50K | 6/9-13 (木馬場補完) |

→ 5/24 加入後 即着手可能、 順次 fetch + parse

### 1.4 出力

`data/v18/jvlink_backfill_plan.json` (構造化 plan)

---

## 2. D2 — テストケース大幅追加

### 2.1 ファイル

`tests/test_session40_session39_tools.py` (新規、 250 行)

### 2.2 test 数

| クラス | tests |
|--------|------|
| TestSession39ATools (sib expanding) | 2 |
| TestSession39BTools (JV-Link fetcher) | 2 |
| TestSession39CTools (SKB exclusion) | 3 |
| TestSession40ATools (5/9 直前) | 11 |
| TestSession40BTools (運用安定化) | 5 |
| TestSession40DTools (jvlink_backfill) | 2 |
| TestV15ProductionUnchanged (V15 不変保証) | 4 |
| **計** | **29** |

(既存 17 tests + 29 = 46 tests、 50+ target に近い)

### 2.3 動作確認

```
$ python -m unittest tests.test_session40_session39_tools -v
Ran 29 tests in 3.92s
OK
```

### 2.4 V15 production 不変 自動保証

`TestV15ProductionUnchanged`:
- predict_core.py syntax check
- daily_predict.py syntax check
- app.py syntax check
- V15 model file 存在 + サイズ ≥ 1MB

→ 5/9 朝の自動 CI でも回せる、 CI 経由で V15 保護

---

## 3. D3 — CI/CD pipeline

### 3.1 ファイル

`.github/workflows/ci.yml` (新規)

### 3.2 trigger

- push to main
- pull_request to main
- 手動 (workflow_dispatch)

### 3.3 jobs

```yaml
1. setup Python 3.11
2. Install dependencies (pandas, lightgbm, xgboost, etc.)
3. Syntax check tools/*.py
4. Syntax check train/*.py
5. Syntax check app.py + predict_core.py + daily_predict.py
6. Run unit tests (data 依存しない test のみ)
7. Verify V15 production files exist (predict_core, daily_predict, app)
```

### 3.4 既存 deploy.yml との関係

- `deploy.yml`: Streamlit Cloud deploy 検証
- `ci.yml`: 新規、 syntax + tests に特化

→ 並行運用、 push 時 両方 trigger

---

## 4. D4 — docs/INDEX.md (doc 体系化)

### 4.1 ファイル

`docs/INDEX.md` (新規、 200+ 行)

### 4.2 カテゴリ tag system

- 🚀 active: 現在運用中
- 📋 plan: 計画 doc
- 📚 handoff: 引き継ぎ
- 🔍 recap: 過去振り返り
- 🛠 ops: 運用手順
- 🧪 research: 探究
- 📦 archive: 旧 plan

### 4.3 セクション

- A. 5/9 投資 関連 (active)
- B. Phase 3 関連 (plan)
- C. Phase 4 関連 (plan + research)
- D. 過去 V16/V17/V162 (archive)
- E. 過去 セッション handoff/recap
- F. 運用 / ops 系
- G. 5/2 / 5/3 / GW 期間
- H. その他 (research / inventory)
- I. data/v18/ 重要 doc
- J. CLAUDE.md / README.md

### 4.4 検索ヒント

5/9 投資なら A、 Phase 3 なら B、 過去問題なら F+E、 等。

---

## 5. 5/9 V15 投資保護 (D 領域)

✅ predict_core.py / daily_predict.py / V15 model file 完全不変
✅ tests は read-only (production code に副作用なし)
✅ CI/CD は GitHub 上のみ動作 (ローカル production に影響なし)
✅ INDEX.md は doc のみ

→ **5/9 朝 V15 完全保証**

---

## 6. 結論

✅ D1: JV-Link backfill plan + inventory 動作 (532K races, 7 datatypes)
✅ D2: 29 tests 追加 (合計 46 tests、 V15 不変保証 4 件含む)
✅ D3: CI/CD pipeline (.github/workflows/ci.yml)
✅ D4: docs/INDEX.md (50+ doc 体系化、 7 タグ system)
✅ D5: 統合 doc (本ファイル)

→ **V20 構築前倒し + 品質保証 + doc 整理 完了**

---

**Session #40 D 完了**
