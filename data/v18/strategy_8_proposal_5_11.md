# 戦略⑧ proposal: 5/11 Phase 24 全 findings 統合 (V15 + Jackpot pattern 拡張)

## 現状: 戦略⑦ (V15 production、 ROI 140% 想定)

```
EXCLUDE:
- 06_特別 (G/L/OPEN 特別 ではない 平場 特別)
- 京都 (data 蓄積 待ち)
- 条件 E (頭数 ≤ 7、 サンプル少)
- 条件 B (重~不良 馬場、 サンプル少)

INCLUDE: 残り 条件 A / C / D / X (1200-1400m 除く) / +1000m 以下 非推奨
```

## 戦略⑧ 提案: 戦略⑦ + Jackpot pattern 拡張

### Layer 1: V15 base (戦略⑦ 維持)
- 既存 V15 案 B 改 + 戦略⑦ そのまま継続
- 月利 +¥28K (baseline)

### Layer 2: Jackpot pattern 自動検出 + 追加 bet
**4-way Jackpot 条件** (今夜検証済、 4 年 stable):
```python
is_jackpot = (
    class_down == 1 and
    horse_recent5_top3 >= top 20% (Q5) and
    jockey_recent30_top3 >= top 20% (Q5) and
    jockey_change == 0
)
```

### Jackpot 該当時 actions
| 条件 (戦略⑦ status) | Jackpot 該当時 |
|------|--------------|
| 条件 A (V15 bet あり) | V15 trio 7点 + 単勝 1500円追加 |
| 条件 C (V15 bet あり) | V15 trio 7点 + 単勝 1500円追加 |
| 条件 D (1200-1400m、 V15 bet あり) | V15 trio 7点維持、 単勝 控えめ 700円 |
| 条件 X (戦略⑦ 除外中、 重15+頭) | **Jackpot で復活 → 単勝 1500円** |
| **京都 (戦略⑦ 除外中)** | **Jackpot で復活 → 単勝 1500円** |
| 条件 E (戦略⑦ 除外、 7頭以下) | 通常通り 除外維持 (base ROI 高すぎて Jackpot 効果薄) |
| 条件 B (戦略⑦ 除外、 重 8-14頭) | Jackpot で復活 → 単勝 1000円 |
| 06_特別 (戦略⑦ 除外) | Jackpot 該当でも 除外維持 (条件 risk 大) |

## 期待 月利

| Layer | 月利 想定 |
|-------|---------|
| Layer 1: V15 戦略⑦ baseline | +¥28K |
| Layer 2: Jackpot 単勝 追加 | +¥9-15K (Jackpot 5-way ~10件/月、 4-way ~13件/月) |
| **合計 戦略⑧** | **+¥37-43K** (V15 単独 比 +¥9-15K) |

## ROI 実証 (4 年 stability)

### 4-way Jackpot (全コース・全条件・全年)
- 年 22: ROI 274.7% (n=73)
- 年 23: ROI 172.8% (n=134)
- 年 24: ROI 165.3% (n=180)
- 年 25: ROI 175.8% (n=209)

### コース別 (全 8 コース で base 超え)
- 函館: 430% / 中京: 234% / 小倉: 206%
- 中山: 184% / 札幌: 165%
- 阪神: 162% / 京都: 146% / 東京: 117%

### 条件別 Jackpot ROI
- 条件 X: 243% (戦略⑦ 除外中、 復活推奨)
- 条件 B: 213% (戦略⑦ 除外中、 復活推奨)
- 条件 A: 200%
- 条件 C: 197%
- 条件 E: 175% (戦略⑦ 除外、 既 base 高 で改善控えめ)
- 条件 D: 145%

## 実装 path (5/12-5/24)

### 5/12-5/14 (火-水): features 整備
1. `build_event_effect_features.py` 最新 data で 再 build
2. `build_hot_streak_features.py` 最新で 再 build
3. `live_jackpot_detector.py` の features 当日 朝 再生成 functionality 追加

### 5/15-5/16 (木-金): integration
1. `tools/race_auto_notify.py` に Jackpot detector hook 追加 (shadow mode)
2. 朝 7:00 で daily_predict.py 出力 + Jackpot detector を combine
3. Discord 通知 strategy:
   - V15 通常 通知 (現行 unchanged)
   - Jackpot 該当 race は **別 channel に 単勝 alert** (700 円 or 1500 円 推奨)

### 5/17 (土): 本番 試験 (shadow mode)
- V15 戦略⑦ 単独継続 (実投票)
- Jackpot detector 結果 = shadow log のみ
- 5/17 verdict で 5/24+ 戦略⑧ 統合判定

## V15 投資保護 (戦略⑧ でも 完全)

- predict_core / daily_predict / app.py / V15 model **不変**
- 戦略⑧ = V15 production + 別 hook layer
- Jackpot bet は 別 channel 通知、 V15 通知に影響しない
- 累計 -50K 撤退 line で Drawdown breaker 稼働継続

## 結論

戦略⑦ ROI 140% baseline に Jackpot pattern 拡張で **戦略⑧ ROI 150-160% 想定**、
月利 +¥9-15K 増。 V15 完全保護 + 4 年 stability + 全コース base 超え で 信頼性 高。

5/17 開催 verdict 後 5/24+ で 戦略⑧ 段階導入判定。
