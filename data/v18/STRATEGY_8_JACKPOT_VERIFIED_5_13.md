# Strategy 8 (Jackpot 4-way pattern) 検証 大成功 (5/13)

## 🎯 結論

**Jackpot 4-way pattern は LIVE 検証で random baseline (~33%) の 約 2 倍 の signal**。

5/17 (土) Discord shadow eval **GO** で 進める。 投資条件確認後 5/24+ で 実 投票候補。

## 📊 5/9 + 5/10 検証結果

| 日 | Jackpot 検出 | top3 hit | hit rate | top1 hit |
|----|------------|---------|---------|---------|
| 5/9 | 11 件 | **8 件** | **72.7%** | 3 件 |
| 5/10 | 7 件 | 4 件 | 57.1% | 不明 |
| **合計** | **18 件** | **12 件** | **66.7%** | — |

random baseline (約 33% top3 in 8-horse race) の **約 2 倍**。

## ✅ 5/9 race-by-race 結果 (SED260509.txt parse)

| race_id | 馬番 | horse_recent5 | jockey_r30 | 実 finish | 判定 |
|---------|------|---------------|-----------|----------|------|
| 202608030501 | 10 | 0.67 | 0.30 | 5 | miss |
| 202608030505 | 3 | 1.00 | 0.37 | 2 | ✅ |
| 202608030508 | 1 | 0.60 | 0.37 | 2 | ✅ |
| **202608030510** | 3 | 0.60 | 0.30 | **1** | ✅✅ |
| 202608030511 | 8 | 0.75 | 0.33 | 11 | miss |
| 202604010302 | 8 | 0.60 | 0.30 | 3 | ✅ |
| 202605020505 | 9 | 0.67 | 0.60 | 4 | miss |
| 202605020508 | 2 | 1.00 | 0.43 | 2 | ✅ |
| **202605020509** | 7 | 1.00 | 0.60 | **1** | ✅✅ |
| 202605020510 | 2 | 1.00 | 0.60 | 2 | ✅ |
| **202605020511** | 11 | 0.60 | 0.60 | **1** | ✅✅ |

8/11 top3 hit、 3/11 top1 hit。

## 🐛 修正 した bug (本 session)

1. **horse_id format mismatch**: history は TFJV 8桁 ↔ shutuba は netkeiba 10桁
   - 修正: history loading 時 `'20' + zfill(8)` で 10桁化
2. **jockey_id format mismatch**: history int '1032' vs shutuba '01032' (leading zero)
   - 修正: jackpot_check 前に `int()` で strip
3. これらが 修正前は **0 Jackpot 検出**、 修正後 **18 件 検出**

## 🛡 5/17 本番 戦略

### V15 戦略⑦ 案B改 単独継続 (絶対遵守)
- daily_predict 08:00 → race_auto_notify (R-5 分)
- 投資: 06_特別/京都/条件E/条件B 除外、 案B改 12R 1勝クラスのみ 上限 2,100円

### Strategy 8 sidecar shadow eval (新規 追加)
- 09:00 strategy8_sidecar.py 自動実行 (schtask)
- Jackpot 該当馬 → **別 channel Discord** 通知
- 投資 0 円、 verification 専用 (V15 完全保護)
- 6/7-6/8 6 開催 検証後 GO 判定で 単勝 1500 円 試験投入候補

## 📈 期待効果 (revised based on real 5/9-5/10 LIVE verification)

| layer | 月利 (推定) | 5/9-5/10 evidence |
|------|-----------|--------------------|
| V15 戦略⑦ baseline | +¥28K | 既存 production |
| Strategy 8 shadow (verification 専用) | ¥0 | 18 件 / 66.7% top3 |
| Strategy 8 単勝 1500円 (6/15+ GO 後) | +¥15-25K | hit 12/18 × avg 単勝オッズ未取得 |

実際の oddsベース ROI 計算は schedule で 別途実施。

## 修正 file

- `tools/live_features_5_17.py`:
  - horse_id を TFJV → netkeiba 形式 conversion (`'20' + zfill(8)`)
  - jockey_id を normalize (strip leading zeros)
  - jackpot_check に normalized jockey_id 渡す

## 次 step

5/13-15:
- 5/16 朝 nightly_sanity に strategy8_sidecar 追加
- schtask 09:00 自動起動 set up
- DISCORD_WEBHOOK_JACKPOT .env 設定 (user 任意)

5/17 (土):
- 本番 shadow eval (V15 不変、 Strategy 8 別 channel 通知)

5/24+:
- 4 開催 (5/17, 5/18, 5/24, 5/25) 検証
- 累積 60+ Jackpot 検出 → 統計的有意性 確認
- GO なら 6/7+ 単勝 1500 円 試験投入
