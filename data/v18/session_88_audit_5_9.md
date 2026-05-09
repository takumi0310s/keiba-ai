# Session #88 A: 5/9 全 R 投票 + hit audit

> 作成: 2026-05-09 (Session #88 dev/audit-backtest)
> 目的: 5/9 の実際の投票・hit を生 data から確認し、 累計差分 +¥1,310 の root cause を特定。

---

## 1. data source

| file | 内容 |
|------|------|
| `data/v18/cumulative_monitor_5_9.md` | 5/9 累計報告 (投票額/払戻/損益) |
| `data/daily_results/20260509.csv` | 5/9 候補 R 一覧 (33 R、 全 status=pending) |
| `data/cumulative_results.csv` | 全 settled 結果 (497 行、 5/5 までで止まる) |
| `data/results/20260509_*.md` | 5/9 戦略 / pre_check / dry_run docs |

---

## 2. 5/9 cumulative_monitor 結果 ★決定的★

`data/v18/cumulative_monitor_5_9.md` (5 行のみ):

```
## 5/9 累計報告
投票額: ¥0 / ¥2,100
払戻: ¥0
損益: +¥0

投票 R: 0
```

→ ★ **5/9 は 投票 R = 0 (1 R も 投票していない)** ★
→ 投票額 ¥0 / 損益 +¥0 / 払戻 ¥0

---

## 3. daily_results/20260509.csv の意味

33 行 (京都 12 R + 新潟 9 R + 東京 12 R) 全て:
- `status = pending`
- `profit = -700` (default 候補列の予算割り当て)
- `trio_hit = 空欄`
- `actual_payout = 0`

→ これは **候補 R 一覧 (predictions)** であり、 **実投票記録ではない**。
→ 全 33 R を投票した場合の最大損失 ¥23,100 を示す list。

cumulative_monitor が ¥0 / 0 R を示すため **どの R も投票されていない** ことが確定。

---

## 4. Claude (私) の認識「5/9 新潟 12R MISS -¥700」 の検証

### 4.1 candidate 検索
`daily_results/20260509.csv` 行 22:
- race_id: 202604010312
- 新潟 12R (4 歳以上 1 勝クラス、 ダ 1800m、 条件 A)
- trio_bets: 6-8-11; 6-11-12; 8-9-11; 8-10-11; 8-11-12; 9-11-12; 10-11-12

→ 新潟 12R は **候補に存在**。 だが、 **実投票はされていない** (cumulative_monitor 示)。

### 4.2 結論
- ★ Claude の「5/9 新潟 12R MISS -¥700」 は ★誤認★ ★
- 実際は **5/9 投票 0 R / 損益 ±¥0**

---

## 5. 5/3-5/5 期間の settled rows

cumulative_results.csv の 5/3-5/5 (date filter):

| date | rows | 内容 |
|------|------|------|
| 20260503 | 36 行 | JRA 京都 + 東京 全候補 (全 settled、 trio_hit=0、 profit=-700) |
| 20260505 | 1 行 | NAR 船橋 11R かしわ記念 (settled、 trio_hit=1、 profit=+310) |

→ 5/3 全 36 R は **trio_hit=0 / profit=-700 全て** だが、 これも候補 list (実投票記録ではない)。
→ 5/5 かしわ記念 NAR 1 R のみ user 実投票 (note: "USER投資 三連複7点 #3-#8-#10 的中") → +¥310 確定。

---

## 6. 5/9 投票判定の理由 (推測)

ユーザー (れんはす) が 5/9 投票 0 R にした理由 (推測):
- 戦略⑦ (06_特別 / 京都 / 条件 E / 条件 B 除外) が 5/9 候補 R 全てを除外?
- 案B改 strict (1 勝クラスのみ上限 ¥2,100) の 厳格基準を満たす R が 1 つもなかった?
- ユーザー 自己判断 で skip?

→ いずれにせよ 5/9 の実損益は **±¥0**。

---

## 7. 結論

| 項目 | 値 |
|------|----|
| 5/9 投票 R | **0 R** |
| 5/9 投票額 | **¥0** |
| 5/9 払戻 | **¥0** |
| 5/9 損益 | **±¥0** |
| Claude の「-¥700」 認識 | ★誤り★ |
| root cause (this audit) | 5/9 -¥700 は私の誤認、 実損益 ±¥0 |

→ 累計差分 +¥1,310 のうち **+¥700** は この 5/9 -¥700 誤認 由来。
