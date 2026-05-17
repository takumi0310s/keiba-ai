# data-audit-4: 統合 verdict + 5/18+ 完全 record 仕組み 設計

**作成**: 2026-05-17
**目的**: audit-1/2/3 統合 verdict 確定 + 5/18+ 完全 record 仕組み (race_notify_log 拡張) 設計
**前提**: V15 / v15.2 training (PID 23528) / predict_core / daily_predict / race_auto_notify / app.py 完全不変保証

> ★ **honest 厳守** ★: audit-1/2/3 docs (DATA_AUDIT_1/2/3) は **未作成 (5/17 17:30 時点で docs/ に存在せず)**。
> 本 verdict は **provisional** とし、既知事実 (V15-audit、 commit history、 file inventory) から直接 verify した。

---

## 0. 結論 (TL;DR)

| 項目 | 完全性 | 5/18+ 仕組み 必要性 |
|------|--------|---------------------|
| 朝 8:00 V15 予測 (top1/2/3, formation, score) | ✅ **完全** | 既存 daily_predictions/ で OK |
| 実 race result (1-3着 馬番) | ✅ **完全** | 既存 daily_results/ で OK |
| 実配当 payout (trio/umaren actual) | △ **部分** (jra_payouts.csv は **5/3 で停止**) | netkeiba 経由で daily_results が補填中 |
| **投票実 formation (skip/bet 区分 + 確定 formation + 投票時 odds)** | ❌ **★ 記録なし ★** | **race_notify_log v2 拡張 必須** |

**5/18+ paper eval base**: **case B** (formation 記録なし)
→ 「仮想 formation hit rate 改善」 のみ計算可能、 「実投票 ROI 改善」 は **5/18 以降 race_notify_log v2 が稼働してから計測可能**。

---

## 1. 真の data 完全性 verdict (4 項目)

### 1-A. 朝 8:00 V15 予測 record

| 項目 | 値 |
|------|-----|
| source | `data/daily_predictions/{YYYYMMDD}.csv` |
| 5/17 行数 | 34 race (header+34) |
| 5/16 行数 | 35 race |
| schema | race_id, course, race_num, race_name, condition, num_horses, distance, surface, track_condition, top1_num, top1_name, top1_score, top2_num, top2_name, top3_num, top3_name, **trio_bets**, bet_type, investment |
| 完全性 | ✅ **完全** — top1-3 score + 予定 formation (trio_bets) + investment 含む |
| 観測 (5/17) | 京都 12R / 新潟 12R / 東京 10R = 34 race 全記録 |

**判定**: ✅ 朝予測は **完全記録**。 paper eval 用の **仮想 formation 計算可能**。

---

### 1-B. 実 race result

| 項目 | 値 |
|------|-----|
| source | `data/daily_results/{YYYYMMDD}.csv` + `data/cumulative_results.csv` |
| 5/17 行数 | 33 race (header+33) |
| schema | race_id..., top1_finish, top2_finish, top3_finish, **trio_result** (実 1-3着 馬番), distance, surface |
| 完全性 | ✅ **完全** — 実 1-3着 馬番 (trio_result) + top1-3 着順 含む |
| 観測 (5/17) | 京都 12R + 新潟 11R + 東京 10R = 33 race (新潟 12R 取得失敗 1 件) |

**判定**: ✅ 実 result は **完全に近い** (新潟 12R 1 件 miss あり、 後追い補填可)。

---

### 1-C. 実配当 payout

| 項目 | 値 |
|------|-----|
| source 1 | `data/jra_payouts.csv` — **5/3 で停止** (CLAUDE.md 既知バグ) |
| source 2 | `data/daily_results/{YYYYMMDD}.csv` の `trio_payout` / `umaren_payout` / `actual_payout` — netkeiba 経由補填 |
| 5/17 実 payout 合計 | ¥10,940 / 23 race (有効 33 race中) |
| 5/17 投資 | ¥23,100 / profit -¥12,160 |
| 完全性 | △ **部分** — jra_payouts は 5/3 から非更新、 daily_results が代替補填 |

**判定**: △ jra_payouts.csv 修復は 5/24+ JV-Link HR 経路で予定 (Session #39 B)。 **現状は daily_results で運用継続可能**。

---

### 1-D. 投票実 formation record (★ 最重要 ★)

| 項目 | 値 |
|------|-----|
| source | `data/race_notify_log/{YYYYMMDD}.json` (commit 1a76a3ff、 5/17 朝開始) |
| 5/17 5/17 行数 | **2 entries のみ** (race_id=null) |
| schema | race_id, race_name, notified_at, channel, strategy_7c_skip, strategy_7c_reason |
| 記録内容 | skip/notify 判断 + reason のみ |
| **欠落** | **確定 formation (trio_bets actual) / 投票時 odds / weight_diff / 投票成立フラグ** |

**判定**: ❌ ★ **投票実 formation 記録なし** ★
- race_notify_log は P0-5 用の skip 判定 log のみ
- 確定 formation = daily_predictions の trio_bets と **同一** だが、投票直前変更/odds 取得タイミング/weight_diff スキップは記録なし
- 5/17 朝の log は 2 entry (race_id=null) しか書かれていない → 機能不全の可能性 ★

**5/17 race_notify_log 内容**:
```json
[
  {"race_id": null, "race_name": null, "notified_at": "None", "channel": null, "strategy_7c_skip": false, "strategy_7c_reason": null},
  {"race_id": null, "race_name": null, "notified_at": "None", "channel": null, "strategy_7c_skip": false, "strategy_7c_reason": null}
]
```

→ ★ **既存 race_notify_log は呼び出されているが、 引数 None で 2 件しか記録されていない** ★

---

## 2. 5/18+ paper eval の真の base

### case A (全完全 record) — **非該当**

### case B (formation 記録なし) — **★ 該当 ★**

| paper eval 種別 | 計算可否 | 根拠 |
|------------------|----------|------|
| **仮想 formation の hit rate** | ✅ **可能** | daily_predictions の trio_bets + daily_results の trio_result から計算可 |
| **仮想 formation の ROI** | △ **部分可能** (jra_payouts 5/3 以降欠損、daily_results で補填) | daily_results の trio_payout 使用 |
| **実投票 formation の hit rate** | ❌ **不可** (5/17 まで) | race_notify_log の formation 未記録 |
| **実投票 formation の ROI** | ❌ **不可** (5/17 まで) | 同上 |
| **改善前後 比較 (alpha 戦略)** | ✅ **仮想のみ可能** | daily_predictions に対する仮想 formation で比較 |
| **実投票 ROI 改善計測** | ❌ **5/18 race_notify_log v2 稼働 以降のみ** | 設計のみ完了、 5/18+ 実装必要 |

### honest 結論

- **5/17 までの累計 ROI** = `data/cumulative_results.csv` の actual_payout 合計 (現実 = profit total **-¥30,020**、 5/17 単日 **-¥12,160**)
- **5/18+ paper eval** = 仮想 formation hit rate 改善 / 仮想 ROI 改善 のみ計算可能
- **実投票 ROI 改善計測** = **5/18 race_notify_log v2 稼働後 蓄積開始 → 1-2 週間後 比較可能**

---

## 3. 5/18+ 完全 record 仕組み 設計

### 3-1. 既存 race_notify_log (commit 1a76a3ff) の制約

| 項目 | 現状 | 5/18+ 必要 |
|------|------|-----------|
| 呼び出し phase | race_auto_notify 内 (≒ 投票通知時 1 phase) | **3 phase** (朝予測 / 投票直前 / 結果回収後) |
| 記録 schema | race_id + skip 判断のみ | **formation + odds + weight_diff + result + hit/miss** |
| file 構造 | `data/race_notify_log/{date}.json` 単 file | `data/race_notify_log_v2/{date}/{race_id}_{phase}.json` 分離 |
| 5/17 観測 | 2 entry (race_id=null) — 機能不全可能性 | 各 phase で正常書き込み |

### 3-2. race_notify_log v2 拡張仕様 (★ 設計のみ、 実装は別 sub-task ★)

```python
# tools/race_notify_log_v2.py 設計案
"""
P0-5 + 投票実 formation 完全 record 仕組み

3 phase 完全 log:
- phase 1: morning_predict (daily_predict.py 内 8:00) → 朝予測 ranking + 予定 formation
- phase 2: pre_vote (race_auto_notify.py 内 発走 5-15min 前) → 投票 formation 確定 + 投票時 odds + weight_diff
- phase 3: post_result (daily_results.py 内 結果回収後 20:00) → 実 1-3 着 + 実配当 + hit/miss
"""

def race_notify_log_v2(
    race_id: str,
    phase: str,  # 'morning_predict' | 'pre_vote' | 'post_result'
    timestamp: str = None,
    # phase 1 (morning_predict)
    ranking_top5: list = None,            # [(num, name, score)] * 5
    formation_planned: str = None,         # "1-2-3; 1-2-4; ..."
    # phase 2 (pre_vote)
    formation_actual: str = None,          # 戦略⑦案 C / weight_diff 反映 後
    odds_snapshot: dict = None,            # {num: odds} 投票時
    weight_diff: dict = None,              # {num: kg_diff} 09:30 取得
    strategy_7c_skip: bool = False,
    strategy_7c_reason: str = None,
    # phase 3 (post_result)
    result_1st: int = None, result_2nd: int = None, result_3rd: int = None,
    trio_payout: int = None, umaren_payout: int = None,
    actual_payout: int = None,
    hit: bool = None,
):
    """
    出力: data/race_notify_log_v2/{YYYYMMDD}/{race_id}_{phase}.json (新規 file)
    fail 時: stderr 出力のみ、 V15 投票 logic 影響なし
    """
    pass
```

### 3-3. 統合 file (1 race 全 phase) — phase 3 完了後 自動 merge

```python
# data/race_notify_log_v2/{date}/{race_id}_FULL.json
{
  "race_id": "202608030801",
  "race_name": "3歳未勝利",
  "date": "20260518",
  "phase_1_morning_predict": { ... },
  "phase_2_pre_vote": { ... },
  "phase_3_post_result": { ... },
  "summary": {
    "voted": true,
    "skip_reason": null,
    "formation_diff_planned_vs_actual": "...",
    "hit": false,
    "profit": -700
  }
}
```

### 3-4. ★ V15 production 不変保証 ★

- log は **file IO のみ**、 V15 投票判断 / predict_core / race_auto_notify の logic は **完全不変**
- log 書き込み fail 時 → stderr 出力のみ、 投票通知続行
- daily_predict / race_auto_notify / daily_results に **read-only な log hook 追加** (try-except wrap、 fail 時 silently pass)

---

## 4. 5/18+ 実装 step (★ 別 sub-task ★)

| step | 内容 | timing |
|------|------|--------|
| 1 | `tools/race_notify_log_v2.py` 新規作成 (race_notify_log v2 関数 + merge 関数) | 5/18 朝 admin schtask 登録後 |
| 2 | `daily_predict.py` 末尾に phase 1 log hook 追加 (try-except wrap) | 5/18 朝 |
| 3 | `race_auto_notify.py` の `predict_and_notify` 内 投票決定後に phase 2 log hook | 5/18 朝 |
| 4 | `daily_results.py` 末尾に phase 3 log hook + merge_to_FULL | 5/18 夜 |
| 5 | 5/19 朝 5/18 log の完全性 verify (3 phase 揃っているか) | 5/19 朝 |
| 6 | 1-2 週間蓄積後、 仮想 vs 実投票 ROI 比較レポート | 5/25+ |

★ **今回は設計のみ**、 実装は 5/18 朝 別 sub-task ★

---

## 5. V15 production 不変保証 ✅

| 項目 | 状態 |
|------|------|
| V15 model (`keiba_model_v135_central_live.pkl.gz` 等) | ✅ **完全不変** |
| `predict_core.py` | ✅ **read-only、 変更なし** |
| `daily_predict.py` | ✅ **本 audit-4 では変更なし** (5/18 実装時 log hook 追加予定、 try-except wrap で投票影響ゼロ保証) |
| `race_auto_notify.py` | ✅ **本 audit-4 では変更なし** (5/18 実装時 log hook 追加予定) |
| `app.py` | ✅ **完全不変** |
| v15.2 training (PID 23528) | ✅ **中断なし、 read-only access のみ** |
| commit / push | ✅ **未実施** (親 task 集中) |

---

## 付録: 観測 raw data

### A-1. file inventory (5/17 18:00 時点)

| file | size | last update |
|------|------|-------------|
| data/daily_predictions/20260517.csv | 34 race | 8:00 |
| data/daily_results/20260517.csv | 33 race | 18:00 |
| data/cumulative_results.csv | 631 rows total / 33 rows for 5/17 | 18:00 |
| data/race_notify_log/20260517.json | **2 entry (race_id=null)** ★ 機能不全可能性 ★ | 朝 |
| data/jra_payouts.csv | 12,334 rows / **last 5/3** | 5/3 停止 |

### A-2. 5/17 単日 + 累計 (5/17 18:00 時点)

| 項目 | 値 |
|------|-----|
| 5/17 投資 | ¥23,100 |
| 5/17 払戻 | ¥10,940 |
| 5/17 損益 | **-¥12,160** |
| 累計損益 (cumulative_results 全体) | **-¥30,020** |

### A-3. 既存 race_notify_log の呼び出し箇所 (race_auto_notify.py)

10 箇所で `_p0_5_notify_log()` 呼び出し:
- skip 判定 (no_horse_data / obstacle_race / distance_le_1000 / strategy_7_06_tokubetsu / strategy_7_kyoto_p0_2_5_17 / strategy_7_cond_E / strategy_7_cond_B / strategy_7_cond_X_p0_2_5_17)
- bets 通知 (channel='bets')
- error (channel='error')

→ ★ 5/17 朝 race_auto_notify が正常 fire したか **2 entry のみで疑問** ★ — 5/18 朝 schtask admin 登録後 再確認必要

---

**完了**: data-audit-4 統合 verdict 確定、 **case B (formation 記録なし)** 確定、 5/18+ race_notify_log v2 設計完了、 V15 production 不変保証。
