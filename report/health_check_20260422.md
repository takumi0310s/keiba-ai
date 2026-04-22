# Health Check (2026-04-22 → 04-23)

## 1. Cookie 有効性 ✅

```
$ python tools/refresh_cookie.py --check
[OK] Premium認証OK: 調教タイムデータ取得確認
```

→ NETKEIBA_COOKIE 有効、Premium 取得可。リフレッシュ不要。

---

## 2. タスクスケジューラ 7本 全 Ready ✅

| タスク | State |
|--------|:---:|
| Keiba-AM3FireCheck | Ready |
| Keiba-AM6FireCheck | Ready |
| Keiba-AM8FireCheck | Ready |
| Keiba-FridayWeekendScrape | Ready |
| Keiba-MorningDigest | Ready |
| Keiba-NightlySanity | Ready |
| Keiba-PreFireCheck | Ready |

→ 全 7 タスク Ready 状態、来週末 4/25-26 本番運用に支障なし。

---

## 3. 回帰テスト ✅

```
$ python -m pytest tests/test_am3_fire_check.py tests/test_fire_check_common.py \
                  tests/test_pre_fire_check.py tests/test_morning_dashboard.py \
                  tests/test_discord_notifier.py tests/test_scraper_guard.py
82 passed in 1.30s
```

新規 / 既存 fire-check / scraper-guard 系の 82 tests 全 PASS.

AM3FireCheck の修正後にも regression なし。

---

## 4. 判定ゲート評価

| 条件 | 結果 | 判定 |
|:---|:---:|:---:|
| フェーズ1 で critical なし | ✅ (誤通知のみで本番影響ゼロ) | OK |
| フェーズ2 で v16 閾値達成 | ❌ master_index 2020-2022 が 0% | **NG** |
| フェーズ3 DailyResults リトライ成功 | ✅ 4/19 反映 (累積359R) | OK |
| フェーズ4 cookie 有効 + 全タスクReady + テストPASS | ✅ | OK |
| 現在時刻 23:30 未満 | ❌ 既に 03:25 を過ぎている | **NG** |

### 結論: フェーズ5 v16学習は **スキップ** (2項目NG)

#### スキップ理由
1. **v16 閾値未達** (master_index 2020-2022 が 0% のまま、3日経過もスクレイピング進捗ゼロ)
2. **時間制約超過** (作業開始時 22:30 想定が、調査完了時 03:25。AM2:00 リミット超過)

#### 来週月曜以降の再開プラン
- Mon 06:00 SCRAPER-GUARD 解除後にバックグラウンド再起動
  ```bash
  nohup python tools/scrape_missing_all.py --years 2020,2021,2022,2023 \
    > logs/scrape_missing_all_restart3.log 2>&1 &
  ```
- 週次でカバ率確認: `python tools/coverage_report.py`
- 閾値突破後 (推定 5/2 以降) に v16 学習再計画
- CatBoost WF の `race_id_unique` KeyError も同時修正必要
