# Session #72 B: 全馬スコア順 通知設計

**作成**: 2026-05-09 18:02 (Session #72、 dev/two-stage)

---

## 旧通知 (Session #68 修復後、 top3 のみ)

```
## R12 新潟 1h 前予測 (Stage 2) — Stage 1 fallback 採用
発走: 16:10 / レース: 4歳以上1勝クラス

### Stage 2 失敗 (netkeiba_block)
error: ...

### 採用予測 = 朝予測 (Stage 1) top3 ★
1. 11 ハイクオリティ (score=0.648)
2. 12 マテンロウミラクル
3. 8 カレンラップスター
```

→ top3 のみで残り全頭の評価が見えない。

---

## 新通知 (Session #72、 全馬 V15 score 順 table)

### case 1: 5/10 以降 (data/daily_predictions_full/{date}.csv 存在)

```markdown
## R12 新潟 4歳以上1勝クラス (16:10 発走)
出走: 14 頭 / コース: 新潟 D1200m / 馬場: 良

### 全馬 V15 score 順 (Stage 1 = 朝予測 8:00)
| 順 | 馬番 | 馬名 | V15 score | 単勝オッズ |
|----|----|--------|------|------|
| 1 | 11 | ハイクオリティ | 0.648 | 5.2 |
| 2 | 12 | マテンロウミラクル | 0.612 | 8.4 |
| 3 | 8  | カレンラップスター | 0.587 | 12.1 |
| 4 | 3  | ホースD            | 0.523 | 18.3 |
| 5 | 7  | ホースE            | 0.498 | 22.5 |
| ...                                  |
| 14 | 5 | ホースN            | 0.215 | 89.2 |

### Stage 2 状況
- netkeiba server block 検知 (HTTP 400) → Stage 1 fallback 採用
- 当日体重統合: 未取得 (block のため)
- 次 fire (30 分後) で再試行

### V15 投票方針 (絶対遵守)
- 投票推奨: (該当 R なら明示) / (除外 R なら「投票なし」)
- Stage 2 は学習用、 投票推奨ではない
- 累計 +12,830 円 死守
```

### case 2: 5/9 以前 (daily_predictions_full 不在、 top3 のみ)

```markdown
## R12 新潟 4歳以上1勝クラス (16:10 発走)
出走: 14 頭 / コース: 新潟 D1200m / 馬場: 良

### 朝予測 (Stage 1) top3
| 順 | 馬番 | 馬名 | V15 score |
|----|----|--------|------|
| 1 | 11 | ハイクオリティ | 0.648 |
| 2 | 12 | マテンロウミラクル | - |
| 3 | 8  | カレンラップスター | - |

※ 全馬 score 表記は 5/10 以降 (Session #71 完了後) で対応。
※ 5/9 以前は朝予測 top3 のみ available。

### Stage 2 状況
- netkeiba server block 検知 (HTTP 400) → Stage 1 fallback 採用

### V15 投票方針 (絶対遵守)
- (各 R に応じた投票方針)
```

---

## Discord 2000 char 上限への対応

18 頭以上 で table が大判化:
- 全頭 row を保持
- 馬名は左 14 char で truncate
- score は 3 桁、 odds は 1 桁
- それでも 2000 超なら **末尾 truncate + "(以下 N 頭省略)" 表示**
- title + body 合計で計測 (Discord 仕様: title 256 / body 2000)

実測 (18 頭、 各 row ~50 char):
- header + meta: ~250 char
- table 18 row × 50 = 900 char
- Stage 2 status + 投票方針: ~300 char
- 合計 ~1450 char → OK

---

## 入力 schema (data/daily_predictions_full/{date}.csv)

Session #71 が生成する想定 schema:

```csv
race_id,course,race_num,race_name,num_horses,distance,surface,track_condition,
horse_rank,umaban,horse_name,score,odds
202604010312,新潟,12,4歳以上1勝クラス,14,1200,ダ,良,1,11,ハイクオリティ,0.6483,5.2
202604010312,新潟,12,4歳以上1勝クラス,14,1200,ダ,良,2,12,マテンロウミラクル,0.6124,8.4
...
```

→ stage2_predict.py の新 helper:
```python
def load_full_predictions(race_id: str, date: str) -> list[dict] | None:
    """daily_predictions_full/{date}.csv から race_id の全馬 sorted by rank.
    file 不在 or race_id 不在 → None (top3 fallback へ)"""
```

---

## flag 設計

```python
USE_FULL_PREDICTIONS = True  # 5/10 以降 default、 5/9 以前は file 不在で auto fallback
```

または build_message が:
- daily_predictions_full csv を try-load
- 取得できれば全馬 table
- 失敗 → 旧 top3 logic

→ 自動切替、 flag 不要。 シンプル。

---

## 実装方針 (C で実装)

1. `_load_full_predictions(race_id, date) -> list[dict] | None` 追加
2. `_build_horse_table(horses, max_chars=1700) -> str` (truncation 含む table 構築)
3. `build_message_all_horses()` 新規 (旧 build_message と並列)
4. `predict_one()` で `_load_full_predictions` 結果を見て分岐
5. 旧 build_message は削除せず維持 (互換性)
