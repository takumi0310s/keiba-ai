# 5/24 (日) 段取り計画

**作成日**: 2026-05-23  
**目的**: JRA 中央競馬開催日 + V21 TYB 初実データ取得テスト日の段取り整理

---

## 1. 5/24 タイムライン (全自動タスク + 手動作業)

| 時刻 | タスク | 種別 | 備考 |
|------|--------|------|------|
| 02:55 | `\Keiba-PreFireCheck` → `pre_fire_check.bat` | 自動 | 夜間確認 |
| 03:15 | `\Keiba-AM3FireCheck` → `am3_fire_check.bat` | 自動 | |
| 06:15 | `\Keiba-AM6FireCheck` → | 自動 | |
| 06:30 | `\keiba-ai\Morning_Sun` → `tools/morning_top_races.bat` | 自動 | 開催日早朝レース情報 |
| 07:00 | `\Keiba-MorningDigest` → `morning_dashboard.bat` | 自動 | |
| 08:00 | `\keiba-ai\DailyPredict` → `daily_predict_watchdog.bat` | 自動 | 全レース朝予測 |
| 08:45 | `\keiba-ai\RaceAutoNotify_Sun` → `race_auto_notify.bat` | 自動 | **V15 本番** Discord 通知開始 |
| 08:50 | `\Keiba-AM8FireCheck` | 自動 | |
| **09:00** | `\Keiba-JrdbRetryAm9_Sun` → `tools/jrdb_retry_am9.bat` | 自動 | JRDB KYI リトライ |
| **09:00** | `\Keiba-SaveAllHorseScores_0930` → `save_all_horse_scores_runner.bat` | 自動 | 馬スコア保存 |
| **09:30** | `\Keiba-MorningWeightCheck_Sun` → `tools/morning_weight_check.bat` | 自動 | 馬体重急変チェック |
| **朝イチ (起床後)** | **TYB 観測結果確認** | **手動** | 下記「TYB 確認ポイント」参照 |
| **朝イチ (起床後)** | **V21 per-race paper 起動** | **手動** | schtask 未登録→ 手動起動必須 |
| 10:00 | `\Keiba-MultiStagePredict_Test10_Sun` → `tools/multi_stage_predict.bat test10` | 自動 | |
| 14:50 | `\Keiba-MultiStagePredict_Race11_1450_Sun` | 自動 | |
| 15:45 | `\Keiba-MultiStagePredict_Race12_1545_Sun` | 自動 | |
| 18:00 | `\keiba-ai\DailyResults_Sun` → `daily_results.bat` | 自動 | 結果照合・ROI更新 |
| 18:00 | `\Keiba-RaceDayReport_Sun` → `race_day_report.bat` | 自動 | 日次レポート |

---

## 2. V21 TYB 初テストの確認ポイント

### 2-1. 5/23 夜の観測ログ確認 (起動前に必須)

```bash
python -c "
from tools.tyb_shadow_fetcher import summarize_observe_log
import json
print(json.dumps(summarize_observe_log('20260523'), ensure_ascii=False, indent=2))
"
```

**GO 基準**:

| キー | GO 条件 |
|------|---------|
| `ok_count` | >= 1 (5/23 午後分が取れているか) |
| `error_count` | 全件 `[WinError 2]` → ★ Layer 3 修正後の fire (15:44〜16:13 の 3R) で取得できたか確認 ★ |
| `lzh_dl_ok` | >= 1 件 DL 成功 |
| `min_delta` | >= 5 (発走 5 分以上前に取得) |

**注意**: 5/23 の tyb_shadow_log.csv 最終 3 件は `[WinError 2]` (7z フルパス修正前)。  
→ Layer 3 修正 commit 後に再 fire したか確認すること。  
→ もし全件 ERROR なら 5/24 は TYB NO-GO = V21 paper は TYB なし (V15 同等スコア) で観測継続。

### 2-2. GO/NO-GO 判定と対応

| 判定 | 対応 |
|------|------|
| **GO** (>= 1 件 lzh DL + parse 成功) | `TYB_SHADOW_ENABLED = True` に変更 → race_auto_notify.py に TYB ブロック追加 (docs/観-2_5_24_IMPLEMENT_PLAN.md § 3 参照) |
| **NO-GO (404)** | 午前 R は JRDB 未掲載の可能性 → 午後 R のみ観測継続 |
| **NO-GO (認証失敗)** | .env の JRDB_ID / JRDB_PASSWORD を確認 |
| **NO-GO (全 WinError 2)** | Layer 3 修正が有効か確認 (`"C:\Program Files\7-Zip\7z.exe"` 存在 = True 確認済) |

### 2-3. V21 per-race paper 起動手順 (手動)

v21_per_race_paper.py は **schtask 未登録** = 毎回手動起動が必要。

```bash
# 1. 古い v21_per_race_paper プロセスを全 kill (重複防止)
#    PowerShell:
Get-Process python | Where-Object { $_.MainWindowTitle -match "v21" } | Stop-Process
#    または tasklist で PID 確認後 taskkill /F /PID <PID>

# 2. 新プロセス起動 (1R 前、できれば 8:30〜8:40 頃)
python tools/v21_per_race_paper.py --date 20260524 > logs/v21_paper_20260524.log 2>&1
```

**動作フロー**:
1. `load_v21_model()` → `models/v21_candidate.pkl.gz` (存在確認済) を読み込み
2. netkeiba から 5/24 全レース一覧取得 → 発走 -17 分の threading.Timer を設定
3. 各 Timer fire 時: `fetch_tyb_observe(race_id, start_time_str)` → TYB 取得 → V21 スコア計算 → Discord 送信 → `data/v21_paper_log/20260524/` に JSON 記録

**TYB fetch タイミング**: 発走 17 分前 (JRDB 直前ファイル配信は発走 ~15-20 分前)。

### 2-4. TYB 取得確認チェックリスト

- [ ] `TYB_SHADOW_OBSERVE_MODE = True` が tyb_shadow_fetcher.py に設定されているか
- [ ] `TYB_OBSERVE_LAUNCH_DATE = "20260523"` (5/24 >= 5/23 → gate 通過)
- [ ] `.env` に `JRDB_ID` キー存在 ✅ (確認済)
- [ ] `.env` に `JRDB_PASSWORD` キー存在 ✅ (確認済)
- [ ] `C:\Program Files\7-Zip\7z.exe` 存在 ✅ (確認済)
- [ ] `.env` に `TYB_SHADOW_OBSERVE_MODE` キー → **未設定** (py ファイル内 hardcode `True` で動作するため問題なし)

---

## 3. V15 本番予測の確認事項

### 3-1. schtask 登録状況 ✅

| タスク名 | 実行時刻 | コマンド | 状態 |
|---------|---------|---------|------|
| `\keiba-ai\RaceAutoNotify_Sun` | 08:45 | `race_auto_notify.bat` | Ready ✅ |
| `\keiba-ai\DailyPredict` | 08:00 | `daily_predict_watchdog.bat` | Ready ✅ |
| `\keiba-ai\DailyResults_Sun` | 18:00 | `daily_results.bat` | Ready ✅ |
| `\keiba-ai\JrdbHealthCheck_Sun` | (不明) | — | Ready ✅ |

### 3-2. 朝の確認コマンド (任意)

```bash
# V15 モデル整合確認
python -c "
import gzip, pickle
with gzip.open('keiba_model_v15_central.pkl.gz', 'rb') as f:
    v15 = pickle.load(f)
print('features:', len(v15['model'].feature_name()))  # 145 であること
"

# 構文チェック
python -c "import py_compile; py_compile.compile('tools/race_auto_notify.py', doraise=True)"
```

### 3-3. 本番予測の注意点

- V15 本番は `race_auto_notify.bat` (08:45 自動起動) で完全自動
- V21 paper とは **完全独立プロセス** — 干渉なし
- TYB を race_auto_notify.py に組み込む場合 (`docs/観-2_5_24_IMPLEMENT_PLAN.md`) は **V15 predictions を一切変更しない**:  
  - 投票 formation は V15+戦略⑦+C4 のまま
  - TYB は Discord への補足表示のみ (try/except 全 swallow)

---

## 4. schtask 登録状況まとめ (5/24 対象のみ)

### 自動 (Ready) ✅

| タスク名 | 時刻 |
|---------|------|
| `\Keiba-PreFireCheck` | 02:55 |
| `\Keiba-AM3FireCheck` | 03:15 |
| `\Keiba-AM6FireCheck` | 06:15 |
| `\keiba-ai\Morning_Sun` (morning_top_races.bat) | 06:30 |
| `\Keiba-MorningDigest` | 07:00 |
| `\keiba-ai\DailyPredict` | 08:00 |
| `\keiba-ai\RaceAutoNotify_Sun` (race_auto_notify.bat) | **08:45** |
| `\Keiba-AM8FireCheck` | 08:50 |
| `\Keiba-JrdbRetryAm9_Sun` | 09:00 |
| `\Keiba-SaveAllHorseScores_0930` | 09:00 |
| `\Keiba-MorningWeightCheck_Sun` | 09:30 |
| `\Keiba-MultiStagePredict_Test10_Sun` | 10:00 |
| `\Keiba-MultiStagePredict_Race11_1450_Sun` | 14:50 |
| `\Keiba-MultiStagePredict_Race12_1545_Sun` | 15:45 |
| `\Keiba-RaceDayReport_Sun` | 18:00 |
| `\keiba-ai\DailyResults_Sun` | 18:00 |
| `\Keiba-NarDailyScrape` | 16:30 |
| `\Keiba-NarDailyPredict` | 17:00 |
| `\Keiba-NarDailyResults` | 21:30 |

### 手動起動が必要 ⚠

| 項目 | 理由 | 手順 |
|------|------|------|
| `v21_per_race_paper.py` | schtask 未登録 | 上記 § 2-3 参照、8:30〜8:40 に手動起動 |

### TYB 関連 schtask

| タスク名 | 時刻 | 備考 |
|---------|------|------|
| `\Keiba-TybPublishMonitor` | 22:30 (5/23 夜) | tyb_publish_monitor.bat — 配信監視 (5/23 夜分) |

---

## 5. 残課題と優先度

### 高優先度 (5/24 朝に判断)

| # | 課題 | 状態 | アクション |
|---|------|------|-----------|
| 1 | **TYB GO/NO-GO 判定** | 5/23 夜観測結果待ち | `summarize_observe_log('20260523')` で確認 |
| 2 | **v21_per_race_paper 手動起動** | schtask 未登録 | 8:30〜8:40 に手動 `python tools/v21_per_race_paper.py --date 20260524` |
| 3 | **DISCORD_WEBHOOK_V21_PAPER 未設定** | fallback で `DISCORD_WEBHOOK_UPDATES` 使用中 | Discord で専用 webhook 作成 → `.env` 追加 (任意だが推奨) |

### 中優先度 (5/24 夕方〜)

| # | 課題 | 状態 | アクション |
|---|------|------|-----------|
| 4 | **v21_per_race_paper schtask 登録** | 未登録 | 5/24 夜に登録 (毎週 SUN 08:45 自動起動) |
| 5 | **TYB GO なら race_auto_notify.py に TYB ブロック追加** | 設計済 (docs/観-2) | § 3-2 の実装手順を実施 |

### 低優先度 (5/24 以降)

| # | 課題 | 状態 | アクション |
|---|------|------|-----------|
| 6 | **generate_horse_reason() 未実装** | v21_per_race_paper.py に呼び出しコードなし (未使用) | V21 paper が安定してから追加 |
| 7 | **人気除外順位表示** | 未実装 | V21 paper スコアテーブルには odds 列あり (現状で代替可) |

---

## 6. 重要な注意事項

- **V15 production は絶対に変更しない** — `keiba_model_v15_central.pkl.gz` (145 features, LGB+XGB) は不変
- **V21 は paper のみ** — 投票は V15 買い目のみ使用
- TYB NO-GO でも V15 本番は完全自動動作 (影響ゼロ)
- TYB 取得エラー時は v21_per_race_paper が swallow → V21 paper ログに記録して継続
- 重複プロセス防止: v21_per_race_paper を再起動するときは必ず旧プロセスを kill してから起動

---

*作成: 2026-05-23 | 調査ソース: schtasks, v21_per_race_paper.py, tyb_shadow_fetcher.py, .env key 確認, tyb_shadow_log.csv*
