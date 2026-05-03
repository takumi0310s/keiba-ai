# 5/9 (土) 投資 final プラン v2

生成: 2026-05-04 (Opus xhigh, Session#6)

参照:
- `data/results/20260509_final_plan.md` (v1, Session#4)
- `data/results/v15_v17_5_3_comparison.md` (5/3 V15/V17 比較)
- `data/v18/v17_v15_improvement_proposals.md` (改良提案 5案)
- `data/v18/improvements_prototyped_5_3.md` (試作 retro 結果)

## v1 → v2 主要変更

| 項目 | v1 | v2 | 変更理由 |
|------|----|----|---------|
| 採用案 | 案B改 | **案B改 (維持)** | 改良提案 retro で全て 5/9 後送り判定 |
| Formation | V15 7点 | **V15 7点 (維持)** | 拡張 retro で ROI 改善せず |
| 軸選定 | V15 単独 | **V15 単独 (維持)** | V15+V17 アンサンブル効果限定 |
| TYB midday | 15:10-15:25 実行 | **実行しない** | 5/3 で TYB HTTP 404 確認、戦略無効化 |
| v18/v19 部分実弾 | 5/16- | **5/16- (但し calibration 修正後)** | retro で全 bet=0 確認 |
| 投資上限 | 2,100円 | **2,100円 (維持)** | 改良なし → 増額不可 |
| 期待 ROI | 161% | **161%** | 同 (healthy 4日 bootstrap) |

## TL;DR

| 項目 | 値 |
|------|---|
| **採用案** | 案B改 (12R 1勝クラスのみ) |
| 採用予定R | 0-3R (12R のみ、11R 全除外) |
| 投資額 | 0-2,100円 |
| 期待ROI | **161.0%** [95%CI 135.9-222.4%] |
| 期待収支 | +0-1,300円 |
| 最悪損失 | -2,100円 (全外し) |
| 撤退ライン | 5/9-5/10 累計 -10,000円, 累計 -50,000円 |

## 1. 採用方針 (5/9 確定、変更なし)

### 11R: 全場 除外 (確定)

| 場 | レース | 理由 |
|----|--------|------|
| 新潟 | 駿風S 芝1000m | 距離不適合 (条件C/D非該当) |
| 東京 | エプソムC G3 | 重賞除外 |
| 京都 | 京都新聞杯 G2 | 重賞除外 |

### 12R: 1勝クラスのみ採用 (5/8 夜 entries 確定後判定)

```
採用条件:
  if "1勝" in race_name:
      → ✅ 採用 (700円 × 三連複7点)
  elif condition == "D" and 1200 <= distance <= 1400:
      → ✅ 採用
  else (2勝/3勝/OP/未勝利/新馬):
      → ❌ 除外
```

### TYB midday script: 実行しない (v2 新規ルール)

🔴 **5/3 経験から TYB midday script 廃止**:
- 14:50 実行 → TYB260503 HTTP=404
- v17 ULTRA-CLEAN は実質 v17_morning と同等で動作
- 朝予測 (06:30 morning script) で確定、直前変更しない

## 2. 当日運用フロー (確定)

### 5/8 (金) 21:00 後

```bash
# friday_weekend_scrape ログ確認
ls -la C:/Users/takum/keiba-ai/logs/friday_weekend_scrape*.log

# 5/9 12R race_name 確認 (1勝クラス かどうか)
python -c "
import requests
r = requests.get('https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=20260509',
                 headers={'User-Agent':'Mozilla/5.0'})
# 12R race_name 抽出
"
```

### 5/9 06:30 (Keiba-Morning_Sat 自動)

```
✓ daily_predict.py V15 全レース (watchdog 経由なら 中断検知)
✓ V17_morning 11R/12R (参考用、5/9 では bet 判断には使わない)
✓ Discord 通知 (#bets)
```

### 5/9 08:00-15:00 (確認時間)

```bash
# V15 予測内容確認
python -c "
import pandas as pd
df = pd.read_csv('data/daily_predictions/20260509.csv', dtype={'race_id':str})
for _, r in df.iterrows():
    if int(r['race_num']) == 12:
        print(r['course'], r['race_num'], r['race_name'], 'top1:', r['top1_num'])
"
# 各場 12R を見て、1勝クラスのみ採用判定
```

### 投票 (15:00 前後)

```
新潟12R (1勝クラス想定):
  軸 = V15 top1 (現行 batch 結果)
  買い目 = V15 trio_bets 7点 (CSVそのまま)
  投資 = 700円

同様に 東京12R, 京都12R (race_name で 1勝確認後)
合計 = 700 × 採用R数 = 0-2,100円
```

### TYB midday script: 実行しない (v2)

5/3 で機能せず、Phase 2.5 で publish タイミング再検証完了するまで使わない。

## 3. healthy 4日 retro 統計 (Session#4-#6 一貫)

| 案 | n | ROI | 95%CI | 利益 |
|----|--:|----:|------:|-----:|
| A: 11R+12R全 | 23 | 86.6% | [62.6, 118.8] | -2,160 |
| B: 12R全+11R非重賞 | 22 | 90.5% | [65.3, 123.1] | -1,460 |
| C: 12Rのみ | 11 | 140.3% | [54.5, 225.1] | +3,100 |
| **B改 (採用)** | 10 | **161.0%** | **[135.9, 222.4]** | **+4,270** |

→ B改の95%CI 下限 135.9% > 100% で **唯一の統計的に確証されたプラス案**。

## 4. 撤退判定 (確定)

### 5/9 単日

| ROI 結果 | アクション |
|----------|-----------|
| ≥ 100% | ✅ 5/10 同戦略継続 |
| 50-99% | ⚠️ 警戒、5/10 控えめ運用 |
| < 50% | 🔴 5/10 投資停止 |
| 全外し (0%) | 🔴 即停止、5/10 ゼロ |

### 累計

| 累計 | アクション |
|------|----------|
| -10,000円 | 5/10 撤退検討 |
| -30,000円 | Phase 2.5 中断、構造再評価 |
| **-50,000円** | **完全撤退** (絶対遵守) |

現状想定: 累計 +14,140円 (USER 報告)。  
5/9 最悪 -2,100円 → 累計 +12,040円 (撤退ライン余裕大)。  
5/9-5/10 最悪 -4,200円 → 累計 +9,940円 (余裕大)。

## 5. Phase 2.5 タスク (5/4-5/15)

5/9 では使わないが、Phase 2.5 で対応:

### 緊急 (5/4 朝までに) — 一部 admin 必要

| 項目 | 状態 |
|------|------|
| Cookie 更新 | ✅ Session#2 完了 |
| morning タスク登録 (Sat/Sun 06:30) | ✅ Session#2 完了 |
| daily_predict_watchdog | ✅ Session#2 作成、admin 移行待ち |
| **DailyPredict task → watchdog 化** | ⚠️ admin 必要、`data/v18/daily_predict_watchdog_migration.md` 参照 |

### 高 (1週間以内、5/4-5/10)

| 項目 | 状態 |
|------|------|
| TYB publish タイミング 連続観測 | 🔧 5/4-5/10 で実施推奨 |
| odds_base daily 自動構築 | ✅ Session#4 で 5/2,5/3 retro 完了、未来日も daily_predict.py 内で生成 |
| netkeiba premium 再起動 | 🔧 ra_score, sc_score, ai_pos 等 |

### 中 (Phase 2.5 後半、5/11-5/15)

| 項目 | 状態 |
|------|------|
| v18/v19 calibration 修正 | 🔧 Platt scaling, race-level norm |
| v15.1 特徴量拡張 (157f想定) | 🔧 KKA/SKB/SR等を V15 に逆輸入 |
| Formation 改良 (Box5/2axis 試行) | 不採用 (retro で効果なし) |

### 5/16 以降 (Phase 3 候補)

| 項目 | 状態 |
|------|------|
| v18/v19 部分実弾 (calibration 修正後) | 🔧 retro で全 bet=0 だったので慎重 |
| EV>1 filter (calibration 後) | 🔧 |
| V15+V17 アンサンブル | 不採用 (5/3 で効果限定) |

## まとめ (5/9 GO 判定 v2)

✅ **5/9 投資 GO** — 案B改 (12R 1勝クラスのみ)、最大 3R × 700円 = **2,100円上限**

| 項目 | 値 |
|------|---|
| 期待ROI | 161% |
| 期待収支 | +0-1,300円 |
| 最悪 | -2,100円 |
| 撤退余裕 | -50,000円まで +14,140円 (=66,140円余裕) |
| 改良提案 | 全て 5/9 後 Phase 2.5 で対応 |

5/9 は守り、5/16 から Phase 2.5 改良の本格投入。
