# 5月 11R/12R (重賞除外) 統計サマリ (Session #70 D)

**source**: production_saved_score (リーク完全防止)

---

## 1. 全体

- 対象 R: **12 件**
- 平均出走頭数: 14.2 頭 (data 有 R のみ)
- 期間: 20260502 〜 20260509
- 日別内訳: {'20260502': np.int64(3), '20260503': np.int64(5), '20260509': np.int64(4)}

## 2. V15 production score 分布 (top1_score)

- 対象 R (score 有): **4** (= 5/9 daily_predictions のみ)
- top1_score 平均: **0.5972**
- 中央値: 0.6053
- 最大: 0.6614 / 最小: 0.5166
- 各 R: ['0.661', '0.517', '0.648', '0.562']

> 5/2, 5/3 の top1_score は cumulative_results.csv で NaN (95% 欠損 既知)。 score 値での分布は 5/9 4 R のみ。

## 3. V15 hit rate (production saved、 5/2-5/3 の cumulative + 5/9 案B改 投票結果)

- 対象 R: 8 (cumulative 11 + 5/9 案B改 投票 1 = 11 R は finish populated、 5/9 残り 3 R は本 session で finish 未取得)
- top1 が 1 着: **3/8** (37.5%)
- top1 が 3 着内: **5/8** (62.5%)
- top1/2/3 全員 3 着内 (perfect trio): **0/8** (0.0%)

## 4. 案B改 strict 7 点三連複 ROI (production、 確定済のみ)

### 4-1. 5/2 + 5/3 確定済 (cumulative_results.csv 由来、 V15 案B改 7 点 三連複 全 R 投票実行)

- 投票 R: **8**
- 投資: **¥5,600** / 払戻: **¥7,880** / 損益: **+¥2,280**
- ROI: **140.7%**
- hit: **3/8** (37.5%)

### 4-2. 5/9 案B改 strict 投票 (新潟 12R 1勝 のみ)

- 投票 R: **1** (新潟 12R 4歳以上1勝、 軸 11 ハイクオリティ)
- 結果: 軸 11 → **3 着**、 1-2-3 着 = `3-8-11` → 三連複 7 点 全 miss
- 投資 ¥700 / 払戻 ¥0 / 損益 **-¥700**
- (Session #67 確定値)

### 4-3. 5/9 verdict 用 (案B改 strict 除外、 投票なし)

- 対象 R: 3 (京都 12R 2勝 / 東京 12R 2勝 / 新潟 11R OP)
- 案B改 strict は 12R 1勝 のみ → 上記 3 R は filter で除外、 投票実行なし
- これらは Session #67 で 5 system 比較 / もし投票してたら ROI 算出 (本 session では集計外)

### 4-4. 5月 累計 (確定済 + 5/9 案B改 strict 1R)

- 投票 R: **9**
- 投資: **¥6,300** / 払戻: **¥7,880** / 損益: **+¥1,580**
- ROI: **125.1%**
- hit: **3/9** (33.3%)


## 5. クラス別

| クラス | R 数 | hit | hit 率 | 投資 | 払戻 | 損益 |
|---|---|---|---|---|---|---|
| 1勝 | 3 | 2 | 66.7% | ¥2,100 | ¥6,070 | +¥4,670 |
| 2勝 | 4 | 1 | 25.0% | ¥2,800 | ¥1,810 | +¥410 |
| OP/特別 | 5 | 0 | 0.0% | ¥3,500 | ¥0 | ¥-2,800 |

## 6. surface 別 (data 有 R のみ)

| 馬場 | R 数 | hit | hit 率 | 損益 |
|---|---|---|---|---|
| ダ | 9 | 2 | 22.2% | +¥1,290 |
| 芝 | 3 | 1 | 33.3% | +¥990 |

## 7. 5/16 V18 trial 含意 (production data から)

- V15 案B改 strict は 5月 12 R (重賞除外、 production saved) で **hit 率 / ROI 集計可能**。 上記 #4 を base data として 5/16 V18 trial GO/NO-GO の比較対象に使える
- 5/9 単独 -¥700 (投票 1R/MISS) は 5月全体の hit 率を歪める可能性 → 5/2, 5/3 を含めた集計が代表値
- ★ 5/2-5/3 の score 値欠損 (cumulative_results.csv バグ) は **production save logic の補修候補** → Session #65/68 の Stage 2 system 修復と並行で daily_predict.py の保存ロジック audit が必要
