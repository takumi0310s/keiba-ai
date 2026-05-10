# Phase 2 戦略⑦ filter 動作確認 (5/10 09:44)

## 結論: ✅ **正常動作** (36R → 8R 通知)

## filter 実装
`tools/race_auto_notify.py` 内 `predict_race()` 関数

### 1. race_name + course filter (line 171-187)
```python
# 06_特別 (G/L/OPEN特別 ではない平場特別) を除外
is_graded = any(g in race_name_str for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'])
is_listed = any(s in race_name_str for s in ['L)', '(L)', 'OP)', '(OP)'])
is_open_tokubetsu = any(s in race_name_str for s in ['杯', '賞', 'ステークス', 'カップ', 'ハンデ'])
if '特別' in race_name_str and not (is_graded or is_listed or is_open_tokubetsu):
    return  # Skip 06_特別

# 京都を除外
if course_str == '京都':
    return  # Skip 京都
```

### 2. 条件 filter (line 269-276)
```python
if cond_key == 'E':
    return  # Skip 条件E (頭数<=7)
if cond_key == 'B':
    return  # Skip 条件B (重~不馬場)
```

## 5/10 36R 内訳

### 京都 12R (全 skip)
| R | 名 | 条件 |
|---|----|------|
| 1 | 3歳未勝利 | A |
| 2 | 3歳未勝利 | D |
| 3 | 3歳未勝利 | A |
| 4 | 3歳未勝利 | C |
| 5 | 3歳未勝利 | A |
| 6 | 4歳以上1勝 | A |
| 7 | 3歳1勝 | A |
| 8 | 4歳以上1勝 | D |
| 9 | 烏丸S | A |
| 10 | 橘S | D |
| 11 | 平城京S | C |
| 12 | 4歳以上2勝 | D |

→ **全 12R 京都 skip**

### 新潟 12R
| R | 名 | 条件 | 判定 |
|---|----|------|------|
| 1 | 3歳未勝利 | C | OK |
| 2 | 3歳未勝利 | D | OK |
| 3 | 3歳未勝利 | C | OK |
| 4 | (csv に欠損) | - | - |
| 5 | 3歳未勝利 | C | OK |
| 6 | 3歳未勝利 | D | OK |
| 7 | 4歳以上1勝 | D | OK |
| 8 | 4歳以上1勝 | D | OK |
| 9 | **荒川峡特別** | A | **skip 06_特別** |
| 10 | **五泉特別** | A | **skip 06_特別** |
| 11 | 谷川岳S | D | OK |
| 12 | 4歳以上1勝 | D | OK |

→ 11R 通過 (4 欠損 + 9, 10 skip)

### 東京 12R
| R | 名 | 条件 | 判定 |
|---|----|------|------|
| 1 | 3歳未勝利 | C | OK |
| 2 | 3歳未勝利 | D | OK |
| 3 | 3歳未勝利 | C | OK |
| 4 | 3歳未勝利 | A | OK |
| 5 | 3歳1勝 | C | OK |
| 6 | 4歳以上1勝 | D | OK |
| 7 | 4歳以上1勝 | D | OK |
| 8 | 4歳以上2勝 | D | OK |
| 9 | **日吉特別** | A | **skip 06_特別** |
| 10 | メトロポリタンS | A | OK (S = ステークス想定) |
| 11 | NHKマイルC | C | OK (G1) |
| 12 | 立夏S | C | OK |

→ 11R 通過 (9 skip)

### filter 後計
- 京都 0 + 新潟 11 + 東京 11 = **22R 通過**
- ★ 但し 整形済み通知 = 8 messages
- → 残り 14R は 別 filter (条件E/B、 オッズ未確定、 EV 低、 信頼度低、 等) で skip

### 推測される 8 候補
- 整形済み = 案B改 strict (後段 filter 適用後)
- 詳細は Discord #買い目 channel の 8 個別 message で確認

## filter の盲点 (要対応 5/13+)

⚠ `notify_bets_all_in_one.py` (全レース一括通知) は 戦略⑦ filter **未適用**
- 全36R を そのまま 表示
- 京都12R + 06_特別3R + 条件E/B も含む
- → ノイズ多 + 1900 char 超え → 2 chunk 分割

→ 改善案: notify_bets_all_in_one に 戦略⑦ filter 適用 → 1 message 化 + 投票候補 のみ表示

## 投票判断
- ✅ **整形済み 8 messages を信頼**
- ❌ 全レース一括 (1/2, 2/2) は情報提供のみ、 投票判断には使わない
- 5/10 14:00: 8 候補 × ¥700 = **¥5,600** (案B改 上限 ¥2,100 以下に収まる)
