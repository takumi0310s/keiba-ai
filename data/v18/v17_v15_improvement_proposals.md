# V15/V17 改良提案 5案 (5/3 分析ベース)

生成: 2026-05-04 (Opus xhigh, Session#6)

参照:
- `data/results/v15_v17_5_3_comparison.md` (5/3 詳細比較)
- `data/v18/formation_retro_summary.csv` (formation retro 結果)
- `data/v18/v18_v19_retro_full_result.md` (v18/v19 5/2-5/3 retro)

## 提案サマリー

| # | 提案 | 期待効果 | 実装工数 | 5/9 適用 |
|---|------|---------|--------:|---------|
| 1 | TOP2-TOP6 拡張 (formation) | hit率↑、ROI 中立 | 1h | ❌ retroで効果なし |
| 2 | V15+V17 アンサンブル | 軸選定 +0pt | 1日 | ❌ 5/9 では V15 単独維持 |
| 3 | 特徴量拡張 | 中期 (再 train 必要) | 1週 | ❌ |
| 4 | TYB 公開タイミング再調査 | 直前予測戦略の生死 | 1週 | ❌ |
| 5 | EV>1 フィルター実装 | bet 数絞込で ROI↑ | 半日 | ⚠️ v18/v19 calibration 待ち |

## 提案1: TOP2-TOP6 選定精度改良 (formation 拡張)

### 検証結果 (healthy 4日 137R)

| Formation | n_bets | ROI | hit_rate |
|-----------|------:|----:|--------:|
| **V15_baseline_7 (現行)** | 7 | **39.9%** | 16.1% |
| 2axis_4 (Top1-Top2 軸2頭流し) | 4 | **42.4%** ↑ | 11.7% |
| V15+T1-T4-T5 (8点) | 8 | 34.9% | 16.1% |
| V15+T14_T15_T16 (10点) | 10 | 27.9% | 19.7% |
| V15+T2-T3-T4 (8点) | 8 | 34.9% | 16.8% |
| Box5_10 | 10 | 20.7% | 16.8% |
| Box6_20 | 20 | 14.0% | 33.6% |

### 11R/12R のみ (主要レース 23R)

| Formation | ROI |
|-----------|----:|
| V15_baseline_7 | **86.6%** |
| 2axis_4 | 85.5% |
| V15+T14_T15_T16 | 60.6% |

### 評価

- **既存 V15 7点が ROI ベスト近傍** (拡張で hit 率↑だが投資額↑で相殺)
- 2axis_4 は ROI 微増だが分散大
- Box 系は投資額膨張で ROI 大幅悪化

→ **5/9 採用しない**。現行 V15 baseline 7点 維持。

## 提案2: V15+V17 アンサンブル軸選定

### 5/3 主要6R 軸top3率

| モデル | 軸 top3率 |
|--------|---------:|
| V15 | 3/6 (50%) |
| V17 morning | 3/6 (50%) |
| 一致セル数 | 4/6 |

→ V15/V17 軸が異なるレース 2件:
- 京都11R: V17 が当てた、V15 外し
- 新潟12R 1勝: V15 が当てた、V17 外し

→ **両モデルの軸選定優位性は相殺**。アンサンブル効果 limited。

### アンサンブル戦略試案 (実装せず)

```python
# 軸 = V15 top1 と V17 top1 が一致 → 採用 (高信頼)
# 不一致 → スキップ (or 投資額 半減)
if v15_top1 == v17_top1:
    bet_amount = 700
else:
    bet_amount = 0  # スキップ
```

5/3 6R で適用 → 軸一致 4R, 不一致 2R。  
不一致 2R (京都11R, 新潟12R 1勝): 1勝率 50% だがサンプル小。

→ **5/9 採用しない**。retro 結果不十分、運用ルール複雑化リスク。

## 提案3: 特徴量拡張 (中期)

### V17 ULTRA-CLEAN 196 features 中、V15 (150f) に未含 46f

主な追加 features:
- TYB (6): 直前データ
- KKA (4): 種牡馬・母父系統
- SKB (3): 脚質
- ZK (5): 前走補足
- SR/SRB (11): 前走バイアス
- netkeiba premium (4)
- 騎手×馬統計 (5)
- その他 (8)

### V15 への逆輸入候補

```
高優先度 (BT で V17 に effective):
  - kka_dam_rensho_avg, bms_rensho_avg
  - sr_first3f_avg, srb_bias_4corner
  - skb_baba_avg

中優先度:
  - 騎手×馬 (5): 5/2-5/3 で 0% 充足だが本来 3-5%
  - 兄弟血統 (sib_top3_rate, sib_shinba_wr)
```

→ **5/16 以降 v15.1 (157f想定) として再 train**。
   5/9 までに完了不可、Phase 2.5 後半タスク。

## 提案4: TYB 公開タイミング再調査 ★最重要

### 5/3 観測

```
14:50 midday script 実行 → TYB260503 HTTP=404
```

→ 5/3 京都11R (15:40 発走) の 50分前で TYB 未公開。

### 仮説と検証計画

| 仮説 | 検証方法 | アクション |
|------|---------|-----------|
| TYB は post-race (17:00以降) のみ | 5/4-5/10 連続 fetch test (5分間隔) | midday script **廃止** |
| TYB は当日朝 公開 (発走 4-6h前) | 同上 | midday → **morning** 統合 (8:00 fetch) |
| TYB は発走 30-60分前 | 同上 | midday script **設計変更** (15min前→60min前) |
| TYB 不安定 (時々遅延) | 同上 | midday + リトライ強化 |

### 検証スクリプト (試作必要)

```bash
# tools/check_tyb_publish_time.py
# 5/4-5/10 で各日 06:00, 09:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00 で TYB 取得試行
# → 公開時刻分布を統計化
```

→ **5/9 直前予測は実行しない** (5/3 の経験から信頼度低)。Phase 2.5 で検証完了後判定。

## 提案5: EV>1 フィルター実装 ★Phase 2.5 #5

### 前提条件

- ✅ v18/v19 model 復旧済 (session#5)
- ✅ odds_base 5/2, 5/3 構築済 (session#4)
- ⚠️ v18/v19 5/2-5/3 retro で全 filter bet=0 (probability 過小評価)

### 設計案

```python
# tools/ev_filter.py
def compute_ev(p_tansho, p_fukusho, base_odds, base_fukusho):
    ev_tansho = p_tansho * base_odds
    ev_fukusho = p_fukusho * base_fukusho
    return ev_tansho, ev_fukusho

# Filter: EV>1 のみ採用
def should_bet(p_tansho, base_odds, threshold=1.0):
    return p_tansho * base_odds >= threshold
```

### v18/v19 calibration 問題

5/2-5/3 retro で v18/v19 全予測が **probability 過小** (max 0.001-0.017)。
- BT 2025 OOS: max p ~0.4-0.9 (正常)
- 5/2-5/3 (2026): max p ~0.001-0.02 (異常)

→ **Distribution shift** または **feature pipeline 不整合** の疑い。
   そのまま EV>1 filter かけても全 bet ゼロ。

### Calibration 修正案

1. **Platt scaling**: BT で確率を実 win rate に較正
   ```python
   from sklearn.linear_model import LogisticRegression
   # p_raw → p_calibrated
   lr = LogisticRegression()
   lr.fit(p_raw_oos.reshape(-1,1), is_win_oos)
   p_cal = lr.predict_proba(p_raw_5_3)[:,1]
   ```

2. **Race-level normalization**: 各レース内で probability を sum=N (or 1) に正規化
   ```python
   df['p_norm'] = df['p_tansho'] / df.groupby('race_id')['p_tansho'].transform('sum')
   ```

3. **特徴量整合性検証**: 5/2-5/3 features 分布 vs 2024 features 分布の比較

→ **Phase 2.5 第2-3週 (5/16-5/24) で実装**。5/9 採用しない。

## 5/9 投資判断への影響

### 維持 (案B改 そのまま)

- V15 batch 軸 (現行)
- V15 trio_bets 7点 formation (現行)
- 12R 1勝クラスのみ採用 (案B改)
- 投資 700円/R, 上限 2,100円

### 採用しない (時期尚早)

| 提案 | 理由 |
|------|------|
| Formation 拡張 | retro で ROI 改善せず |
| V15+V17 アンサンブル | 5/3 で明確な優位性なし、ルール複雑化 |
| 特徴量拡張 | 中期タスク (再 train 必要) |
| TYB midday script | 公開タイミング不明、5/3 で機能せず |
| EV>1 filter | v18/v19 calibration 未解決 |

### 5/9 以降の Phase 2.5 タスク優先度

1. **TYB publish タイミング検証** (5/4-5/10 連続 fetch test)
2. **v18/v19 calibration 修正** (Platt scaling, race-level norm)
3. **特徴量拡張 v15.1** (5/16 以降)
4. **DailyPredict watchdog admin 移行** (重要、5/4朝までに)

## 結論

🟢 **5/9 案B改 維持**:
- 12R 1勝クラスのみ採用
- V15 batch 軸 + 現行 7点 formation
- 投資上限 2,100円
- 期待 ROI 161% (healthy 4日 retro)

🔴 **改良提案は全て 5/9 後**:
- 5/3 直前予測戦略は無効化 (TYB 未公開)
- formation 拡張は retro で効果なし
- v18/v19 は calibration 未解決
- アンサンブル/特徴量拡張は中期

→ **5/9 は守り**、**5/16 以降に Phase 2.5 改良を順次投入**。
