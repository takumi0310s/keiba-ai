# P0-2 案 C dry-run 結果 (5/16 today data)

作成: 2026-05-16
source: `data/daily_predictions/20260516.csv` (n=35)
作業 mode: read-only、 V15 model output 不変。 5/16 当日の prediction 出力を 案 C filter で再評価。

---

## 0. 結論

| 項目 | 値 |
|------|---|
| Total 5/16 predictions | 35 |
| Would skip (案 C) | **11** (京都 全 11R) |
| Would proceed | **24** (東京 12R + 新潟 12R + 京都 8R 内 0R は重賞 →なし) |
| Affected courses | 京都 のみ (東京/新潟 0 影響) |

★ 5/16 は案 C 適用済の dry-run であれば 京都 11R 分の投票 (¥7,700) が skip される ★

---

## 1. 全 35R の filter 動作

### 1.1 By course

| course | total | would skip | would proceed |
|--------|---:|---:|---:|
| 京都 | 11 | **11** | 0 |
| 東京 | 12 | 0 | 12 |
| 新潟 | 12 | 0 | 12 |

### 1.2 Skipped races (5/16 京都)

| race_id | course | race | 名前 | cond | graded |
|---|---|---|---|---|---:|
| 202608030701 | 京都 | 1R | 3歳未勝利 | D | False |
| 202608030702 | 京都 | 2R | 3歳未勝利 | A | False |
| 202608030703 | 京都 | 3R | 3歳未勝利 | D | False |
| 202608030704 | 京都 | 4R | 3歳未勝利 | A | False |
| 202608030705 | 京都 | 5R | 3歳未勝利 | A | False |
| 202608030706 | 京都 | 6R | 4歳以上1勝クラス | D | False |
| 202608030707 | 京都 | 7R | 4歳以上1勝クラス | A | False |
| 202608030709 | 京都 | 9R | あずさ賞 | A | False |
| 202608030710 | 京都 | 10R | 上賀茂S | C | False |
| 202608030711 | 京都 | 11R | 鞍馬S | D | False |
| 202608030712 | 京都 | 12R | 4歳以上2勝クラス | D | False |

★ 上賀茂S / 鞍馬S は OPEN特別だが `is_listed` (L) 判定対象外 → 案 C で skip 対象 ★

### 1.3 条件 X races (5/16)

5/16 当日の prediction で 条件 X に該当する race: **0 件**

→ 条件 X filter は 5/16 適用なし。

---

## 2. 5/17 影響 audit (★ 重要 ★)

5/17 race card (netkeiba race_list_sub 取得):

| course | total | would skip | would proceed |
|--------|---:|---:|---:|
| 東京 | 12 | 0 | 12 |
| 京都 | 12 | **12** (重賞なし想定) | 0 |
| 新潟 | 12 | 0 | 12 |

### 2.1 ヴィクトリアマイル (東京 11R G1) 影響 audit

| 項目 | 値 |
|------|---|
| race_id | 202605020811 |
| race_name | ヴィクトリアマイル |
| course | 東京 |
| is_graded (literal G1 文字 判定) | **False** (race_name に「G1」 literal がない) |
| 案 C action | **proceed** (東京は除外対象外) |

★ 重要 ★: ヴィクトリアマイル は race_name に「G1」literal がないため `is_graded=False` だが、 **東京は案 C 除外対象でないため そもそも skip されない**。 案 C 影響 **0%**。

### 2.2 京都 11R audit

| 項目 | 値 |
|------|---|
| race_id | 202608030811 |
| race_name | 栗東S (OPEN特別) |
| is_graded | False |
| is_listed | False |
| 案 C action | **skip** (Kyoto + 非重賞) |

→ 京都 11R は案 C で skip。 5/17 京都は重賞開催なしのため 12R 全 skip。

---

## 3. 機会損失/利益試算 (5/16 dry-run)

5/16 単日 京都 結果 (settled 後集計):

| 項目 | 京都 11R 計 |
|------|---:|
| 投資 (skip しなければ) | ¥7,700 (1,100 × 7R + 700 × 4R = 案 C で 全 skip) |
| 配当 (実績) | (5/16 settled 集計から京都分のみ抽出が必要、 別 task で) |

★ 5/16 は spike 日のため、 京都が +¥10,000 等の配当に貢献していた可能性あり。 P0-1 spike 起源は別 task で判明予定 ★

---

## 4. honest 注記

1. **「G1/G2/G3」 literal が race_name に含まれない G1 race (例: ヴィクトリアマイル)** が graded 保護対象から外れる潜在的問題。 ただし 案 C は 東京/新潟 等を除外しないため 5/17 G1 day 影響は **絶対 0**
2. **京都で重賞 (例: 京都新聞杯) があった場合**, `is_graded` 判定が hit するかは race_name の literal 次第。 鞍馬S は OPEN特別だが G3 ではない → 正しく skip
3. **5/16 京都 spike 寄与は別 task で audit 必要** (本 sub-task scope 外)

---

end of doc.
