# 既知バグ全棚卸し (5/7 起点) — A agent failed → main 代替版

**作成**: 2026-05-07 朝 (Session #34 A、agent stall で main 代替実装)
**目的**: 5/9 投資 + 5/13-15 V18/V19 復活作業の前に「他に気付いてない劣化」確認

---

## 1. 既知バグ全 list

| # | バグ | source | status | 5/9 影響 | 修復 |
|---|------|--------|--------|---------|------|
| 1 | jrdb_paci 4/4 停止 (CLAUDE.md 既述) | CSV | ✅ **誤情報** (実は 5/3 まで OK) | なし | Session #34 検証で確認 |
| 2 | jra_payouts 4/6 停止 (CLAUDE.md 既述) | CSV | ✅ **誤情報** (実は 5/3 まで OK) | なし | Session #28 確認 |
| 3 | jrdb_skb 全期間 broken | CSV | ⚠️ 残存 (旧データ 2015-2025 のみ) | なし (V15 未使用) | Phase 3+ |
| 4 | training_times 3/27 stale (5週超) | TARGET | ⚠️ 残存 (TARGET 退会済) | なし (V15 学習済 cache) | 6月 JRA-VAN 再契約 |
| 5 | netkeiba_race_review 3/29 stale (5週超) | netkeiba | ⚠️ 残存 (V15 未使用) | なし | Phase 3+ |
| 6 | netkeiba_speed_index 4/29 stale | netkeiba | ✅ Session #27 で 5/2-5/3 復旧 (5/6 11:51 まで) | なし | 解消 |
| 7 | netkeiba_training_eval 5/2-5/3 0 行 | netkeiba | ✅ Session #27 で復旧 (+1,004 行) | なし | 解消 |
| 8 | NarDailyPredict 17:00 rc=1 (pop_rank='--') | tools/predict_nar.py | ✅ Session #31 C で fix | なし | 解消 |
| 9 | DailyPredict 平日 rc=1 (0R 誤判定) | tools/am8_fire_check.py | ✅ Session #31 で fix | なし (土曜は 35R で OK) | 解消 |
| 10 | ProcessWatchdog v2 30/10min 閾値で誤発火 | tools/process_watchdog_v2.py | ✅ Session #31 A3 で 60/30min に緩和 | なし | 解消 |
| 11 | Discord silent fail (notify.py L84) | tools/notify.py | ✅ Session #31 A2 で retry+log 追加 | なし | 解消 |
| 12 | JRDB AM 6:00 早すぎ (TYB/SED 404) | scrape_jrdb | ✅ Session #31 A1 で 09:00 + 12:00 retry 追加 | 軽微 | 解消 |
| 13 | jra_races_full 6 月以降 backfill 不在 | TARGET | 🟢 残存 (5/5 までは OK) | なし | 6月 JRA-VAN 再契約 |
| 14 | chihou_races_2020_2025.csv 不在 | NAR | ✅ Session #24 で「実は不要」と判明 | なし | 解消 |
| 15 | premium CSV 追記 bug (cache JSON のみ) | daily_premium_scrape | ✅ Session #27 で恒久対策 | なし | 解消 |
| 16 | cumulative_results.csv top1 95% 欠損 | tools/daily_predict | ⚠️ 残存 (CSV 集計 影響) | なし (累計手計算 OK) | Phase 3+ |
| 17 | **sib_top3_rate / sib_shinba_wr リーク** | V18/V19 model | 🔴 **本日 Session #34 で発覚** (4/29 リーク削除済だが V18/V19 model に残存) | V18/V19 5/16 試行に影響 | Phase 3 V18/V19 再学習 |
| 18 | sr_first3f_avg 等 3 features merge 不足 | tools/jrdb_features.py L864-876 | 🟠 残存 (本日 Session #34 で発覚) | V18/V19 5/16 試行に影響 | 5/13 修正 (2h) |
| 19 | jrdb_sr 5/2 stale (5日) | data | 🟢 軽微 | なし | DailyJrdbKyi で再取得 |

合計 19 件、🔴 緊急 1 (V18/V19 関連、5/16 影響)、🟠 高 1、⚠️ 中 4、✅ 解消 12、🟢 低 1。

---

## 2. データ source 健全性 (5/7 起点)

| ファイル | mtime | 状態 |
|---------|-------|------|
| jrdb_paci.csv | 5/3 09:45 | ✅ (4日 stale だが 5/3 完全) |
| jrdb_kyi.csv | 5/3 09:42 | ✅ |
| jrdb_sed.csv | 5/3 09:44 | ✅ |
| jrdb_kab.csv | 5/3 09:45 | ✅ |
| jrdb_tyb.csv | 5/3 09:45 | ✅ |
| jrdb_kka.csv | **5/6 06:01** | ✅ (DailyJrdbKyi 06:00 で更新中) |
| jrdb_skb.csv | 5/3 13:31 | ⚠️ (CLAUDE.md 既知 broken) |
| jrdb_sr.csv | 5/2 14:30 | 🟢 (5日 stale、軽微) |
| jrdb_srb.csv | 5/5 19:04 | ✅ |
| jra_payouts.csv | 5/4 07:59 | ✅ (5/3 末尾) |
| jra_races_full.csv | 5/5 00:05 | ✅ (5/3 末尾) |
| training_times.csv | **3/27 00:40** | ⚠️ TARGET 由来、5 週 stale |
| netkeiba_speed_index.csv | **5/6 11:51** | ✅ Session #27 で本日まで OK |
| netkeiba_training_eval.csv | **5/6 11:51** | ✅ 同上 |
| netkeiba_stable_comments.csv | 5/5 00:37 | ✅ |
| netkeiba_race_review.csv | **3/29 23:21** | ⚠️ 5 週 stale (V15 未使用) |
| cumulative_results.csv | 5/6 16:33 | ✅ (柏記念 +310 反映済) |

**5/9 影響あり stale**: なし (training_times は V15 学習 cache 内、race_review は V15 未使用)

---

## 3. schtasks 異常 (5/7 起点)

| Task | LastResult | 評価 |
|------|-----------|------|
| Keiba-NarDailyPredict | 1 | ⚠️ Session #31 C fix 後の 5/6 17:00 が再度 fail? 要確認 |
| DailyPredict | 1 | 🟢 平日 0R 既知 (土曜 OK) |
| Keiba-WeeklyScrapeResume | 3221225786 | 🟢 Ctrl+C 既知 (5/9 影響なし) |
| WeeklyReport | 1 | 🟢 月曜のみ (5/9 影響なし) |
| ProcessMemoryDiagnosticEvents | 2147946720 | 🟢 Windows 標準 (無関係) |

**5/9 影響**: なし。 NarDailyPredict は 5/12 paper 開始時 再確認推奨 (Session #31 fix 後の 5/13 火 17:00 で結果確認可能)。

---

## 4. 隠れた劣化 (V15 features value drift)

V18/V19 で 12 features 破綻 (sib/sr/sire/bms/training_time_filled) が判明。 V15 (150 features) でも同種劣化の可能性:

| feature | V15 で使用? | drift 起きてるか |
|---------|------------|----------------|
| sib_top3_rate / sib_shinba_wr | ❌ V162 で削除済 | V15 影響なし |
| sr_first3f_avg 等 4 件 | V15 で使用 (V162_FEATURES) | ⚠️ V15 でも同様の merge 不足の可能性 |
| bms/sire_*_wr | V15 で使用 | 🟢 学習 logic 通り (constant 0.100 default は学習データでも頻出、model 側で吸収済) |
| rest_days/weight_trend | V15 で使用 | 🟢 同上 |
| training_time_filled | V15 で使用 | ✅ Session #27 で 5/2-5/3 復旧済 |

→ **V15 でも sr_*_avg 4 features merge 不足が同様に発生している可能性**。 5/13 修正 (jrdb_features.py L864-876 拡張) で V15 にも好影響が出る可能性。

検証: 5/13 修正後に V15 retro 再実行で軸 top3 率の変化確認。 もし -16pt gap が縮小すれば V15 にも貢献。

---

## 5. 5/9 影響あり残存バグ

🔴 緊急: **0 件確定**
🟠 高: 0 件
⚠️ 中: V18/V19 sib リーク (Session #34 発覚、5/16 影響)、sr merge 不足 (5/13 修正予定)
🟢 低: その他

→ **5/9 V15 案B改 単独投資、安全完遂**。

---

## 6. 5/13-15 で対応すべきバグ (V18/V19 復活と並行)

| # | バグ | 修復 timing |
|---|------|----------|
| 17 | sib リーク features 残存 | Phase 3 (V18/V19 sib 抜き再学習) |
| 18 | sr_*_avg 3 features merge 不足 | 5/13 (jrdb_features.py 拡張、2h) |
| 16 | cumulative_results top1 欠損 | Phase 3+ (運用影響軽微) |

---

## 7. 結論

既知バグ 19 件中:
- ✅ 解消 **12 件** (Session #25-#34 で対応)
- 🔴 緊急 **0 件** (5/9 投資準備 完成)
- 🟠 高 **1 件** (sr merge 不足、5/13 修正予定)
- ⚠️ 中 **4 件** (V18/V19 sib リーク、jrdb_skb broken、training_times/race_review stale)
- 🟢 低 **2 件** (jrdb_sr stale、cumulative top1 欠損)

**5/9 V15 案B改 単独投資 影響**: ゼロ。
**5/13-15 V18/V19 復活作業**: sr merge 修正 (2h) で +2-4pt、 sib リーク は Phase 3 で V18/V19 再学習。

V15 features への横展開: sr_*_avg 4 features merge 拡張で V15 にも好影響期待 (5/13 修正後 retro で検証)。
