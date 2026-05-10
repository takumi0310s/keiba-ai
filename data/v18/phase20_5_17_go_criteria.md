# Phase 20 B: 5/17 (土) 本番運用 GO 判定基準

**作成**: 2026-05-10 (Session #92 Phase 20 B、 ★ Opus 4.7 ★)
**前提**: V15 production 単独運用、 V18/V20/V21/V22 paper trade 並行
**目的**: 5/17 (土) 朝に本番運用を継続するか判定する事前 checklist

---

## 1. 5/17 GO 必須条件 (全 PASS で運用継続)

| # | 条件 | 確認方法 | 期待値 |
|---|------|---------|-------|
| 1 | V15 朝予測 (06:00 daily_predict) 完走 | `data/daily_predictions/20260517.csv` exists | 35 R 前後 |
| 2 | Stage 2 動作確認 | `data/daily_predictions_full/` 各 R の json | 各 R 1 件 |
| 3 | V15 model load OK | `python -c "from predict_core import load_models; load_models()"` | exit 0 |
| 4 | V15 model md5 不変 | `Get-FileHash keiba_model_v15_central.pkl.gz` | `842b9a5f...` |
| 5 | paper_trade_engine_v22.py 動作 | `python tools/paper_trade_engine_v22.py --date 20260517` | exit 0 |
| 6 | Discord webhook 動作 | `python tools/notify_done.py "test" "ping"` | OK |
| 7 | netkeiba Cookie 有効 | `python tools/refresh_cookie.py --check` | "Cookie有効" |
| 8 | 累計収支 +¥14,140 | `data/cumulative_results.csv` 末行 | >= +¥14,140 |
| 9 | scrape-status guard OK | `Skill scrape-status` | 緑 |
| 10 | nightly_sanity 23:00 (5/16 夜) PASS | Discord #アップデート 通知 | 確認 |

---

## 2. paper trade 5-model 動作確認 (5/17 朝、 V15 朝予測 後)

### 2.1 V15 + 4 candidate paper trade

```bash
# 5/17 朝 daily_predict 完了後 (08:00+)
python tools/paper_trade_engine_v22.py --date 20260517 --notify
```

### 2.2 期待出力 (V15 ROI は朝時点では 確定不可、 Discord は 20:30 結果照合後)

| model | 動作確認 (朝) | 動作確認 (夜) |
|-------|--------------|--------------|
| V15 | bet 数 表示 (data 確定後 ROI) | hit 数 + ROI |
| V18 cand | 0.75 threshold で bet 0-3 | hit 数 |
| V20 cand | 16 頭 R で bet 1000 円 | hit 数 |
| V21 cand | V20 と同 | hit 数 |
| V22 RL | model load OK | action 履歴 |

### 2.3 NG 時 fallback

| 症状 | 原因 | fallback |
|------|------|---------|
| paper engine 例外 | data race / 結果未確定 | engine は exit 0 で継続、 V15 production 不変 |
| V22 model load fail | stable_baselines3 環境 | V22 のみ skip、 V15-V21 paper 継続 |
| Discord 通知失敗 | webhook 期限切れ | 環境変数 `DISCORD_WEBHOOK_UPDATES` 確認 |
| daily_predict 失敗 | scrape error / cookie | `python tools/refresh_cookie.py --auto` で再実行 |

---

## 3. 5/17 本番運用 GO 判定 表

各項目 OK / NG / WARN を記入し、 全 OK で本番運用継続。 1 件 NG で 当日 skip。

```
================================================================================
5/17 (土) 朝 06:30 GO 判定 worksheet
================================================================================
[ ] 1. V15 朝予測 完走        判定:        / 備考:
[ ] 2. Stage 2 動作確認        判定:        / 備考:
[ ] 3. V15 model load OK      判定:        / 備考:
[ ] 4. V15 model md5 不変     判定:        / 備考:
[ ] 5. paper_trade_engine_v22 判定:        / 備考:
[ ] 6. Discord webhook 動作   判定:        / 備考:
[ ] 7. netkeiba Cookie 有効   判定:        / 備考:
[ ] 8. 累計収支 +¥14,140      判定:        / 備考:
[ ] 9. scrape-status guard    判定:        / 備考:
[ ] 10. nightly_sanity 5/16   判定:        / 備考:
================================================================================
```

---

## 4. 5/17 当日 NO-GO シナリオ (即停止 + Discord 通知)

| 検出 | 即停止 trigger | 対処 |
|------|---------------|------|
| V15 model md5 不一致 | 5/17 朝 hash 確認 | 即運用停止、 archive backup から復元 |
| 累計収支 -¥50,000 到達 | 5/16 夜 累計確認 | 撤退、 Phase 3 後半 まで paper のみ |
| Cookie BAN | 連続 fetch 401/403 | 全 scraper 停止、 cookie 再取得まで paper のみ |
| netkeiba 規約改訂 | 公式アナウンス | Phase 13/18 master 系 全停止 |
| データ整合性異常 | predict 結果 全 0 等 | 当日 skip、 root cause 修正後 翌週再開 |

---

## 5. 戦略 ⑦ + 案 B 改 strict (V15 production 維持)

paper_trade_engine_v22.py に既反映。 5/17 本番も同戦略維持:

- 06_特別 (G レース 以外の特別) 除外
- 京都 全 R 除外 (データ蓄積待ち、 5/24+ 再評価)
- 条件 E (頭数 ≤7) 除外
- 条件 B (重〜不良馬場) 除外
- score < 0.70 除外 (案 B 改 strict 閾値)
- 12R 1勝クラスのみ ¥2,100 上限 (5/9 案 B 改、 V15 朝予測 動作中のみ)

→ 5/17 (土) は通常 strategy_7 適用、 5/9 限定の 12R 例外 は 5/17 では unset

---

## 6. paper trade 累計 trace plan (5/17 以降)

| 日付 | 累計 paper data | V15 vs candidate roi |
|------|----------------|---------------------|
| 5/17 (土) | 1 日 (35 R 前後) | 初期 baseline |
| 5/18 (日) | 2 日 (70 R 前後) | 統計信頼性 低 |
| 5/24 (土) | 8 日 (280 R 前後) | candidate 本格評価 開始 |
| 5/30 (金) Phase 3 前半 | 14 日 (~500 R) | sib_w5 V18 candidate vs V15 |
| 6/8 (日) V20 GO 判定 | 23 日 (~800 R) | V20 4-model ensemble vs V15 |

→ paper_trade_rolling.csv で累計 trace、 weekly_report.py 拡張で週次 Discord 報告

---

## 7. V15 投資保護 (絶対遵守)

✅ V15 model file 不変 (md5 監視)
✅ predict_core / daily_predict / app.py 不変
✅ paper engine は read-only、 production 影響ゼロ
✅ V18-V22 candidate は paper のみ、 1 円も実弾投入しない
✅ 撤退ライン -¥50,000 厳守 (現状余裕 +¥64,140)

---

## 8. 結論

✅ 5/17 GO 判定 10 項目 確定
✅ paper trade 5-model 動作確認 工程 整備
✅ NO-GO シナリオ + 即停止 trigger 文書化
✅ 5/17 朝 06:30 GO 判定 worksheet 提供
✅ V15 投資保護完全

---

**Phase 20 B 完了** (Opus 4.7)
