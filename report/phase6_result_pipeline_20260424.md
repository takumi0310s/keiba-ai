# PHASE 6: 本番後の結果照合準備 (2026-04-24 22:55)

## 6-1. payout バグ修正の最終確認
- `tools/daily_results.py` line 580: `actual_payout = trio_payout if trio_hit else (umaren_payout if umaren_hit else 0)`
- Line 596: `'actual_payout': actual_payout` → result_row に含まれる ✅
- Line 581: `profit = actual_payout - investment` ✅
- commit `ecbdf000` Merge fix/payout-bug-20260423 で修正済
- commit `d2a63752` fix: payout取得バグ完全修正 (actual_payout キー欠落)
- **修正完了** ✅

## 6-2. payout=0 安全装置
- Line 611-624: HIT時にpayout=0を検知したら Discord CRITICAL 通知
  ```python
  if (trio_hit or umaren_hit) and actual_payout == 0:
      print(f"  [ALERT] HITだがpayout=0 race_id={race_id} ({hit_combo})")
      from notify import send_discord
      send_discord(
          '🔴 CRITICAL payout=0 検知',
          ...,
          color='red', channel='updates',
      )
  ```
- **安全装置実装済** ✅
- 実動作テストは本番日 (4/25 18:00 DailyResults) で確認

## 6-3. cumulative_results.csv バックアップ
- `data/cumulative_results.csv` 本体: 60,730 bytes (4/23 21:05)
- `data/cumulative_results.csv.bak_20260418`: 83,971 bytes (4/18 23:18)
- **`data/cumulative_results.csv.bak_20260423_payoutbug`: 60,215 bytes (4/23 21:05)** ✅ 要件のバックアップあり

## 判定: 🟢 OK
- payout バグ修正コードを再確認: `actual_payout` キー含有・プロフィット計算・Discord通知すべて存在
- バックアップ2種類 (4/18 と 4/23) あり、ロールバック可能な状態
- 本番動作は 4/25 18:00 DailyResults で初検証
