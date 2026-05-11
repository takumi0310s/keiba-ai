# Phase 23 real data honest finding (5/11)

## 実 data 分析結果 (5/10 開催)

| 項目 | V15 (700 円固定 + 戦略⑦) | Phase 23 Shadow (Kelly + Cal) |
|------|-------------------------|------------------------------|
| 対象 races | 23 (戦略⑦ 適用後) | 23 |
| 総 投資額 | ¥16,100 | ¥34,500 |
| PnL | +¥7,890 | +¥7,170 |
| **ROI** | **149.0%** | **120.8%** |
| 差分 | baseline | **-¥720** |

## Honest 解釈

🚨 **Phase 23 Shadow が V15 + 戦略⑦ を下回る**:
- Kelly + calibration の 統合 → 投資量 増 (¥16K → ¥34K) だが PnL 改善せず
- 5/10 1 day では Phase 23 が **-720 円 / +PnL より worse ROI**

### 原因分析

1. **odds 推定 簡易**: COND_AVG_ODDS (条件別平均) を使用、 実 trio payout は race ごと大きく分散
   - 真の effect を見るには **race-specific real odds** が必要
2. **calibrator effect 微小**: 314 samples + Brier 0.290 → 0.236 は OK だが、 Kelly fraction 大した変化生まず
3. **V15 + 戦略⑦ が想像以上に強い**:
   - 1200-1400m 除外 / 06_特別 除外 / 京都 除外 / 条件 E/B 除外 で **高効率 race のみに集中**
   - 700 円固定 = 簡素だが robust
4. **Sample 不足**: 5/10 1 day = 23 races は小、 noise 大

### Phase 23 が本領 発揮する条件

1. **Race-specific real odds 取得後** (5min snapshot で計算)
2. **数百-数千 race 蓄積** (small sample noise 除去)
3. **Drawdown breaker** が trigger する場面 (連敗時 損失抑制)
4. **三連単 / 三連単 拡張** (pari-mutuel optimizer の真価)

## 結論 (5/12 user task に影響)

### Priority 変更

| 元 priority | 修正後 | 理由 |
|------------|-------|------|
| Phase 23 統合 → V15 改善 | **Phase 23 は long-term verify** | 1 day では V15 + 戦略⑦ が optimal |
| Kelly bet sizing 投入 | **保留** | 投資量増えるが ROI 改善せず、 risk 増 |
| Calibrator 整備 | **継続 OK** | 真値 prob は他で活用、 Brier 改善は事実 |
| Drawdown breaker | **稼働継続** | 連敗時 のみ役立つ、 cost 0 |
| Pari-mutuel optimizer | **検証 必要** | trio EV 最大化、 real odds で再判定 |
| 30y backtest | **継続 GO** | V20/V21 学習 data の基盤 |
| V21 multimodal | **継続 GO** | 動画 features で V15 飽和 突破 path |

### 5/24+ V20 投入 plan も再評価

V20 学習 + 検証は 継続。 ただし投入時の bet sizing は **V15 と同じ 700 円固定** で start、 Phase 23 tool は long-term shadow log で 検証継続。

→ Phase 23 を慌てて 統合せず、 **真のデータ蓄積後 (3 ヶ月 = 8/15+)** に判定が安全。

## V15 + 戦略⑦ が強い理由 (検証)

5/10 23 races 戦略⑦ 適用 後 ROI 149% = 想定 140%+ 超え

戦略⑦ 除外:
- 06_特別 (G/L/OPEN 特別 ではない 平場 特別): -9,470 円 損失源 除外
- 京都: data 蓄積待ち
- 条件 E (頭数 ≤ 7): サンプル少
- 条件 B (重~不良 馬場): サンプル少

→ 12/35 races 除外 = 約 34% で **高品質 race 選別**、 結果 ROI 149% に。

## 推奨 (修正版)

1. **5/17 開催** V15 案 B 改 + 戦略⑦ 単独継続 (確定)
2. **Phase 23 shadow log 蓄積 のみ** (実投票には適用しない)
3. **calibrator 整備** は 続行 (Brier 改善は事実、 他用途 で活用)
4. **5/24+ V20 投入** は計画 続行、 ただし **bet sizing は V15 同 700 円** で start
5. **3 ヶ月後 (8/15+) Phase 23 統合 再判定** で 真値 data 蓄積後

## V15 投資保護 完全 (再確認)

5/10 実 data で **V15 + 戦略⑦ が現状最強** であることが empirically 確認された。
predict_core / daily_predict / app.py / V15 model 不変、 戦略⑦ 維持 必須。
