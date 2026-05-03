# V17 ULTRA-CLEAN 5/3 直前予測 結果分析

生成: 2026-05-04 (Opus xhigh, Session#6)

## 結論サマリー

🔴 **V17 ULTRA-CLEAN 5/3 は実質 機能せず** — TYB260503 取得失敗で v17_morning と同等動作

| 項目 | 結果 |
|------|------|
| 実行時刻 | 14:50 (発走 15:40 京都11R の 50分前) |
| TYB260503.lzh 取得 | **HTTP 404** ✗ 未公開 |
| 7zip 解凍 | 失敗 (取得物なし) |
| jrdb_tyb.csv に 5/3 データ追加 | 0行 |
| v17 ULTRA-CLEAN 実行 | OK (但し TYB 6 features 全 NaN) |
| tyb_coverage | **0.0** (全レース) |
| 実質モデル動作 | v17_morning (190f) 相当 |

## 実行詳細

### Midday script ログ (`logs/midday_v17_ultra_20260503.log`)

```
=== STEP 1: TYB260503 取得 ===
  TYB260503.lzh: HTTP=404 Size=227B
  ⚠️ TYB 260503 取得失敗 (まだ未公開かも)。リトライ判断はユーザー
=== STEP 2: 7zip 解凍 ===
  TYB260503: 解凍失敗
=== STEP 3: parse TYB260503 (incremental, ~3秒) ===
ERROR: ...TYB260503.txt not found
=== STEP 4: v17 ULTRA-CLEAN 予測 (TYB含む 196 features) ===
[1/3] 京都11R 天皇賞(春) tyb_coverage=0.0 ✗ TYB欠
[2/3] 新潟11R 越後S tyb_coverage=0.0 ✗ TYB欠
[3/3] 東京11R プリンシパルS tyb_coverage=0.0 ✗ TYB欠
=== 完了 76s ===
```

→ **3レース全てで TYB 取得失敗**。v17 ULTRA-CLEAN model は実行されたが TYB 6 features は NaN。

## 5/3 主要 11R 軸変更

| レース | V15 軸 (朝) | V17 morning 軸 | V17 ULTRA 軸 (TYB欠) | 実3着 | V15 軸 結果 | V17 ULTRA 軸 結果 |
|--------|-----------|---------------|---------------------|------|----------|------------------|
| 京都11R 天皇賞春 | 1 ヴェルミセル | **7 クロワデュノール** | 7 (同) | 3-7-15 | 9着 ✗ | **3着内 ✓** |
| 新潟11R 越後S | 10 ユキマル | 15 ルディック | **3 ショウナンアビアス** | 3-4-10 | 3着 ✓ | **1着 ✓** |
| 東京11R プリンシパルS | 4 ヘイジュード | 2 レッドラージャ | **11 オルフセン** | 10-12-13 | 13着 ✗ | 圏外 ✗ |

→ **V17 系 (morning + ULTRA) が V15 を 1勝1分1敗** で軸選定優位。  
   ただし V17 morning と V17 ULTRA は **京都11R で同じ軸 (7)** を選んでおり、**TYB 寄与は不明**。

## V15 vs V17 軸 top3 含有 (主要3R)

| モデル | 軸 top3 含有 | 詳細 |
|--------|------------:|------|
| V15 batch | 1/3 (33%) | 新潟11R 軸=10 のみ |
| V17 morning | 2/3 (67%) | 京都11R, 新潟11R |
| V17 ULTRA (TYB欠) | 2/3 (67%) | 同上 |

→ V17 系の軸選定が V15 を上回る (3R中 2R で 3着内)。  
   ただし trio_hit には至らず (top2-top6 選定で外し or formation 制約)。

## trio_bets 分析 — 軸が3着内でも hit しない理由

### 京都11R 天皇賞春 (V17 ULTRA 軸=7)

V17 ULTRA top6: [7, 1, 3, 12, 2, 4]  
実3着: 3-7-15 (15が V17 ULTRA top6 外)

V17 ULTRA trio_bets (formation 7点):
```
1-2-7, 1-3-7, 1-4-7, 1-7-12, 2-3-7, 3-4-7, 3-7-12
```
実3着 (3,7,15) は (Top1=7 + Top3=3 + 15) = (3,7,15) パターンだが、15 が top6 外 → bet にない → hit せず。

### 新潟11R 越後S (V17 ULTRA 軸=3)

V17 ULTRA top6: [3, 1, 15, 10, 2, 4]  
実3着: 3-4-10 (全部 top6 内!)

V17 ULTRA trio_bets:
```
1-2-3, 1-3-4, 1-3-10, 1-3-15, 2-3-15, 3-4-15, 3-10-15
```
実3着 (3,4,10) パターン:
- (3, 4, 10) = (Top1, Top6, Top4) → **formation で対象外**
- bet にあるのは (1-3-4) や (1-3-10) や (3-4-15) 等
- → hit せず

→ **top6 全部含めても formation 制約で hit しない**。

## TYB 公開タイミングの真実

### 5/3 観測

```
14:50 → HTTP 404 (発走 50分前)
```

### 仮説

1. **TYB は post-race のみ公開** (17:00以降): 直前予測戦略は本質的に無効
2. **TYB は当日朝公開**: midday → morning 統合
3. **TYB は発走 30-60分前公開**: 14:50 は早すぎ、15:25 リトライ必要
4. **TYB 不安定**: 時々遅延

### 検証必要

5/4-5/10 で各日 06:00, 09:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00 の TYB fetch 試行 → publish 時刻分布を統計化。

(cf: `data/v18/v17_v15_improvement_proposals.md` 提案4)

## 5/3 直前予測の成果

✅ **CSV 出力完了**: `data/v17/predictions_5_3_ultra_top_races.csv` (3レース、TYB欠状態)  
✅ **Discord 通知**: #bets に送信成功  
❌ **TYB 寄与検証**: 不可能 (TYB 取得失敗で control 比較できず)  
❌ **直前予測の付加価値**: 5/3 では確認できず

## 結論

🟡 **直前予測 V17 ULTRA-CLEAN は 5/3 では機能せず** (TYB 取得失敗)  
🟢 **V17 系 (morning + ULTRA) は V15 より軸選定がやや優位** (主要3R 2/3 vs 1/3)  
🔴 **trio_hit までは到達せず** (top6 漏れ + formation 制約)

→ 5/9 では:
- TYB midday script は **実行しない** (v2 ルール)
- V15 batch 軸 + V15 trio_bets 7点 を維持 (現行)
- 改良は Phase 2.5 で別途検証

→ 詳細は `data/results/20260509_final_plan_v2.md`、改良提案は `data/v18/v17_v15_improvement_proposals.md` 参照。
