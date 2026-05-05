# 5/9 (土) ドライラン リハーサル full (5/5 PM、Session #18)

**実行**: 2026-05-05 18:16
**目的**: 5/9 朝起きてから投票完了まで全 step を 5/3 データで模擬実行

---

## 1. 環境健全性 ✅ ALL PASS

| chk | 結果 |
|-----|------|
| Cookie | `[OK] Premium認証OK: 調教タイムデータ取得確認` |
| V15 model load | AUC 0.8939, features 150, 1.1s |
| NAR v4 model load | AUC 0.8145, features 22, 0.0s |
| 5/3 daily_predictions.csv | 35 rows, schema OK |
| schtasks 4 件 (Morning_Sat / DailyPredict / DailyResults_Sat / RaceDayReport_Sat) | 全 Ready |
| .env (DISCORD_WEBHOOK_URL/BETS, NETKEIBA_COOKIE) | 全 set |

## 2. PAT チェックリスト A.2 採用R抽出 dry-run (5/3 データで模擬)

```
=== 5/3 12R 一覧 (dry-run) === [3 R]
  [SKIP] 京都 12R | 東大路S | 条件=D | top1=1 | trio=1-5-7; ...
  [採用] 新潟 12R | 4歳以上1勝クラス | 条件=D | top1=3 | trio=3-4-5; 3-4-10; 3-5-7; 3-5-9; 3-5-10; 3-7-10; 3-9-10
  [SKIP] 東京 12R | 4歳以上2勝クラス | 条件=D | top1=5 | trio=4-5-9; ...
```

**5/3 実 結果**:
- 採用 1 R (新潟 12R 1勝クラス) で 700 円投資
- ROI 525.7%、+2,980 円利益 (race_day_report.py 確認済 Session #14)

→ **5/9 でも同じロジック動作する見込み**。

## 3. race_day_report.py dry-run

```
=== race_day_report 20260503 ===
[OK] data/results/20260503_summary_auto.md
```

→ 既存 手書き summary を上書きしない (`_auto` suffix で保護)、Discord 通知も `--no-discord` で suppression 可能。

## 4. 5/9 朝 想定タイムライン (5/5 18:00 動作確認 base)

| 時刻 | task | 想定動作 | 異常時 fallback |
|------|------|----------|------------------|
| 06:30 | Keiba-Morning_Sat (silent) | morning_top_races.bat → V17 11R/12R 推論 + Discord | logs/morning_top_races_wrapper_20260509.log 確認 |
| 07:00 | Keiba-MorningDigest | dashboard | - |
| 07:30 | JrdbHealthCheck_Sat | JRDB 鮮度 chk | - |
| 08:00 | DailyPredict (watchdog) | V15 全 35 races (土曜は 0 races でない、想定 fatal なし) | watchdog 3 retry → 失敗時手動 `python tools/daily_predict.py --date 20260509` |
| 09:00 | (人手) | 12R race_name 確認 + 採用 R 決定 | データ不正なら netkeiba 直接確認 |
| 14:00-15:30 | (人手 PAT 投票) | 採用 R × 700円、上限 2,100円 | 締切間に合わない R は除外 |
| 18:00 | DailyResults_Sat | 結果照合 | DailyResultsEvening (20:00) で再試行 |
| 18:00 | RaceDayReport_Sat | race_day_report.py + Discord 結果通知 | 手動 `python tools/race_day_report.py` |
| 23:00 | NightlySanity | 翌 5/10 task pre-check | - |

## 5. 想定 NG ケース

| 症状 | 対応 |
|------|------|
| Cookie 切れ | logs/morning で alert → `python tools/refresh_cookie.py --auto` (期限切れ時のみ自動 refresh) |
| daily_predict 中断 | watchdog 3 retry → 失敗時 手動 `python tools/daily_predict.py --date 20260509` |
| race_name "1勝" 識別失敗 | netkeiba 直接確認、shutuba.html で 目視 |
| 12R 全 1勝クラスでない | **5/9 無投資** (累計 +14,140円 維持) |
| Discord 通知来ない | `.env` の DISCORD_WEBHOOK_URL 直接 curl 確認、別端末で `python tools/notify_done.py "test" "test"` |
| 累計 -50,000円 接近 | **完全撤退** (絶対遵守) |

→ 全てのケースに `data/results/20260509_pat_checklist.md` の Section D で対応手順あり。

## 6. 5/8 (金) 21:00 後 確認 ステップ

```bash
# 1. friday_weekend_scrape ログ
ls -la logs/friday_weekend_scrape*.log

# 2. 5/9 12R race_name 確認
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}',
                     headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '→', (m.group(1).strip() if m else 'NOT_FOUND'))
"

# 3. Cookie chk
python tools/refresh_cookie.py --check
```

## 7. 結論

- **全 chk PASS**: Cookie OK / model load OK / schtasks Ready / 抽出ロジック動作
- **5/3 retro** で採用 R 抽出が完璧に動作 (1 件採用、ROI 525.7%)
- **race_day_report.py** dry-run で既存 summary 保護動作確認
- **5/9 朝起きたら `data/results/20260509_pat_checklist.md` 開いて順序通り進めるだけ**

万一の異常時 fallback も全て手順書化済。

---

## 関連 doc

- `data/results/20260509_pat_checklist.md` (5/9 朝の主役)
- `data/results/20260509_operation_guide.md` (時系列フロー)
- `data/v18/risk_management_5_9.md` (撤退ライン)
- `data/v18/post_5_9_improvement_template.md` (5/9 終了後の振り返りテンプレ)
