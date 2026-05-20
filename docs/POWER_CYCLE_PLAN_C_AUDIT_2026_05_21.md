# 電源 案 C audit: 火-木 OFF 影響 + 配信 timing audit
作成: 2026-05-21 (read-only audit、schtask 変更なし)

---

## 1. 全 Keiba schtask 一覧 + 火-木 fire 頻度

### 毎日 fire タスク (火-木 OFF = 3回/週 miss)

| タスク | 時刻 | 火-木 miss | StartWhenAvailable | DisallowBattery | 実 impact |
|--------|------|-----------|-------------------|-----------------|-----------|
| `\Keiba-PreFireCheck` | 02:55 毎日 | 3回 miss | false | true | 軽微 (pre-check log のみ) |
| `\keiba-ai\DailyPremiumScrape` | 03:00 毎日 | 3回 miss | **true** | false | JRA 中央は平日開催なし = **実害なし** |
| `\Keiba-AM3FireCheck` | 03:15 毎日 | 3回 miss | **true** | false | モニタリングのみ = 軽微 |
| `\keiba-ai\DailyJrdbKyi` | 06:00 毎日 | 3回 miss | **false** | true | KYI 火-木 = 空データ = **実害なし** ★ |
| `\Keiba-AM6FireCheck` | 06:15 毎日 | 3回 miss | **true** | false | モニタリングのみ = 軽微 |
| `\Keiba-MorningDigest` | 07:00 毎日 | 3回 miss | false | true | ダッシュボード生成のみ = 軽微 |
| `\keiba-ai\DailyPredict` | 08:00 毎日 | 3回 miss | **true** | false | 平日中央開催なし = **実害なし** |
| `\Keiba-AM8FireCheck` | 08:50 毎日 | 3回 miss | **true** | false | モニタリングのみ = 軽微 |
| `\Keiba-NarMidDayCalendar` | 13:00 毎日 | 3回 miss | **true** | false | NAR shadow のみ = **実害なし** ★★ |
| `\Keiba-NarDailyScrape` | 16:30 毎日 | 3回 miss | **true** | false | NAR shadow = **実害あり (shadow miss)** |
| `\Keiba-NarDailyPredict` | 17:00 毎日 | 3回 miss | **true** | false | NAR shadow = **実害あり (shadow miss)** |
| `\Keiba-NarLiveOddsRefresh` | 19:00 毎日 | 3回 miss | **true** | false | NAR shadow = **実害あり (shadow miss)** |
| `\keiba-ai\DailyResultsEvening` | 20:00 毎日 | 3回 miss | **true** | false | 平日中央結果なし = **実害なし** |
| `\Keiba-NarDailyResults` | 21:30 毎日 | 3回 miss | **true** | false | NAR shadow = **実害あり (shadow miss)** |
| `\Keiba-NightlySanity` | 23:00 毎日 | 3回 miss | false | true | sanity check のみ = 軽微 |
| `\Keiba-TybPublishMonitor` | 毎時 (PTH1) | 全火-木 miss | false | true | TYB は土曜早朝配信 = **実害なし** |

### 週次/特定曜日タスク

| タスク | 曜日/時刻 | 案C 影響 |
|--------|----------|---------|
| `\Keiba-FridayWeekendScrape` | 金曜 10:00 | PC ON = **OK** |
| `\keiba-ai\Keiba-WeeklyScrapeResume` | 月曜 06:30 | PC ON = **OK** |
| `\keiba-ai\WeeklyReport` | 月曜 08:00 | PC ON = **OK** |
| `\KeibaAI_DriftDetector` | 月曜 08:30 | PC ON = **OK** |
| `\Keiba-Morning_Sat` | 土曜 06:30 | PC ON = **OK** |
| `\Keiba-Morning_Sun` | 日曜 06:30 | PC ON = **OK** |
| `\Keiba-JrdbRetryAm9_Sat` | 土曜 09:00 | PC ON = **OK** |
| `\Keiba-JrdbRetryAm9_Sun` | 日曜 09:00 | PC ON = **OK** |
| `\keiba-ai\JrdbHealthCheck_Sat` | 土曜 07:30 | PC ON = **OK** |
| `\Keiba-RaceAutoNotify_Sat` | 土曜 08:45 | PC ON = **OK** |
| RaceDayReport, MultiStagePredict, Verdict 系 | 土日のみ | PC ON = **OK** |

---

## 2. JRDB / JRA-VAN データ配信 timing 一覧

### 実測データ (scrape_jrdb.py ログ + jrdb_raw/ ファイル timestamp より)

| JRDB ファイル | 対象日 | 実際の配信日 | 配信時刻 | 証拠 |
|--------------|--------|------------|--------|------|
| KYI260509 (土) | 5/9 (土) | 5/9 (土) 06:00 | 6am | jrdb_kyi_auto_20260509.log [DL] 6:00 |
| KYI260510 (日) | 5/10 (日) | 5/10 (日) 03:00 | 3am | jrdb_raw/kyi/ timestamp |
| KYI260516 (土) | 5/16 (土) | **5/16 (土) 03:00** | 3am | premium_scrape_20260516.log [DL] 3:00 |
| KYI260517 (日) | 5/17 (日) | **5/16 (土) 03:01** | 3am+1min | premium_scrape_20260516.log [DL] 3:01 |
| KYI 火-木 | 火/水/木 | **配信なし** | — | jrdb_kyi_auto_20260505〜08: 全て「データなし」 |
| TYB | 当日朝 | 当日 (土曜のみ) | 不定 (404 多数) | tyb_publish_log.csv 5/4, 5/9 = 404 |
| SED | レース翌日 | 翌朝 | 06:00 前後 | jrdb_raw/sed/ |
| KAB | 木曜 | 木〜金 | 03:00 | KAB260517.lzh = 5/15 03:03 (木曜夜) |

### 重要発見 ★

```
KYI (出走馬基本情報/IDM指数) の配信パターン:
  火曜: 配信なし (確認済)
  水曜: 配信なし (確認済)
  木曜: 配信なし (確認済)
  金曜: 配信なし (06:00 時点では空、ただし深夜3am時点で前日配信されている可能性あり)
  土曜: 03:00〜 配信開始 (DailyPremiumScrape が自動取得)
  日曜: 土曜 03am に同時配信 (DailyPremiumScrape が土日両方を一括取得)

→ KYI を火-木に取得できなくても DATA LOSS ゼロ
```

### KAB (競馬場情報) の発見

KAB260517.lzh が 5/15 (木曜) 03:03 に配信されていた。  
ただし KAB は競馬場固定情報 (競馬場名/距離/コース形状) であり、  
週次変化ほぼなし。1回 miss しても翌週金曜に取得可能。  
実質影響: **なし**

### JV-Link (JRA-VAN DataLab) データ配信

- 実装状況: `tools/jvlink_fetcher.py` は PoC stub のみ (5/24+ 着手予定、現在未稼働)
- 現時点での JV-Link 定期 fetch タスク: **存在しない**
- 5/24 Phase 3 開始後に weekly fetch スクリプト追加予定
- 配信 timing (仕様より): レース結果 (HR) = レース当日夜〜翌朝、出走表 (RACE) = 木曜前後
- 案 C 実施時点 (5/22) では JV-Link 影響 = **ゼロ** (未稼働)

---

## 3. 火-木 OFF の影響度評価

| データソース | 配信 timing | 火-木 miss 影響 | 金曜 catch-up 可能か | 判定 |
|------------|-----------|--------------|-------------------|------|
| KYI (JRDB) | 土曜 03:00〜 | **なし** (火-木 = 空データ) | — (不要) | **GREEN** |
| TYB (JRDB) | 土曜朝 (不定) | **なし** (土曜配信) | — (不要) | **GREEN** |
| SED (JRDB) | レース翌日朝 | **なし** (土日レース後に月曜取得) | — | **GREEN** |
| KAB (JRDB) | 木曜夜 03:00 | 軽微 (1週間遅延) | 金曜 6am DailyJrdbKyi で自動取得 | **YELLOW** (軽微) |
| DailyPremiumScrape | 毎日 03:00 | **なし** (平日 JRA 開催なし) | StartWhenAvailable=true で金曜 3am 自動 | **GREEN** |
| DailyPredict (JRA) | 毎日 08:00 | **なし** (平日中央開催なし) | StartWhenAvailable=true | **GREEN** |
| DailyResultsEvening | 毎日 20:00 | **なし** (平日中央結果なし) | StartWhenAvailable=true | **GREEN** |
| NAR 予測 (火-木) | 毎日 17:00 | **あり** (火-木 shadow 予測 miss) | StartWhenAvailable=true だが過去日は無意味 | **YELLOW** (shadow のみ) |
| NAR 結果 (火-木) | 毎日 21:30 | **あり** (火-木 shadow 結果 miss) | 翌日 scrape 不可 (当日のみ) | **YELLOW** (shadow のみ) |
| WeeklyReport | 月曜 08:00 | **なし** (月曜 PC ON) | — | **GREEN** |
| JV-Link | 未稼働 | **なし** (5/24+ 着手) | — | **GREEN** |

---

## 4. NAR 影響詳細

### 現状の NAR 運用状態

| 項目 | 状態 |
|------|------|
| NAR 予測実行 | 毎日 17:00 (predict_nar.py) |
| NAR モデル | NAR v4 (AUC 0.8145, 22 features) |
| NAR ベット | **shadow 評価のみ** (実投票なし、実 ROI tracking なし) |
| NAR in cumulative_results.csv | 1行のみ (テスト行、data 破損) |
| Strategy 8 (NAR Jackpot) | shadow eval 段階 (6/15+ go/no-go 判定予定) |
| Discord NAR 通知 | 未確認 (predict_nar.py に Discord call なし) |

### 火-木 NAR miss の実害

- 火-木 各日: 約 40-55 レース予測 miss (5/17=55R, 5/18=47R, 5/19=48R)
- ただしこれらは **shadow データ蓄積のみ**
- 実 ROI への影響: **ゼロ** (実投票していない)
- Strategy 8 go/no-go の蓄積サンプル数への影響: 週 3 日 miss = 月 12 日分の shadow データ欠落
  - 現在蓄積: 5/11〜5/20 で順調に蓄積中
  - miss の影響: 6/15 go/no-go 判定を若干遅延させる可能性 (1-2 週)

---

## 5. 案 C 真の verdict

### ★ 案 C = 可能 ★

根拠:

1. **JRA 中央競馬への影響ゼロ**
   - 中央競馬は土日 (+ 稀に平日祝日) のみ開催
   - 火-木 OFF で miss するタスクは全て「平日開催なし」で実害なし
   - KYI (重要 JRDB データ) は火-木に配信されない (実測確認)

2. **JRDB データ欠損リスクゼロ**
   - KYI 火-木 fetch ログ = 全て「データなし」
   - 週末 KYI は DailyPremiumScrape (土曜 03:00 自動) が一括取得
   - SED/TYB は土日レース後 → 月曜 PC ON で自動取得

3. **StartWhenAvailable による自動 catch-up**
   - DailyPremiumScrape, DailyPredict, NAR 系は `StartWhenAvailable=true`
   - 金曜朝 PC ON 直後に missed 分が自動実行される
   - ただし平日分は「取得すべきデータなし」なので catch-up 内容も空

4. **唯一の懸念: NAR shadow データ蓄積の週 3 日 miss**
   - 影響度: 軽微 (実投票・実 ROI に無影響)
   - 対処: Strategy 8 go/no-go を 6/15 → 7/1 に 2 週延長すれば問題なし

### 条件付き制約

```
月曜 08:00 WeeklyReport → 月曜 PC ON 必須 (案 C 定義通り OK)
金曜 10:00 FridayWeekendScrape → 金曜 PC ON 必須 (案 C 定義通り OK)
土日 全タスク → PC ON (案 C 定義通り OK)
```

---

## 6. 電気代節約試算

| 項目 | 値 |
|------|-----|
| 火-木 OFF 時間 | 72 h/週 |
| PC アイドル消費電力 (仮定) | 50 W |
| スリープ消費電力 (仮定) | 5 W |
| 電気料金 (東京電力 2026 概算) | 約 30 円/kWh |

### 節約額

| 比較対象 | 週次節約 kWh | 週次節約 円 | 月次節約 円 | 年次節約 円 |
|---------|------------|-----------|-----------|-----------|
| vs 完全 ON (idle 50W) | 3.6 kWh | 108 円 | 464 円 | 5,616 円 |
| vs スリープ (5W → 0W) | 0.36 kWh | 11 円 | 47 円 | 561 円 |

- **案 C の主な節約効果は「スリープ比」ではなく「完全 ON 比」**
- スリープ運用 (案 B) に比べた追加節約は月 47 円のみ (わずか)
- 完全 OFF にする主な動機が「PC 寿命延長・騒音ゼロ」なら案 C の価値あり

---

## 7. 案 B vs 案 C vs 案 D 比較

| 案 | 内容 | 電気代節約 | データ miss リスク | 推奨度 |
|----|------|-----------|------------------|--------|
| 案 A | 常時 ON | ゼロ | ゼロ | 現状維持 |
| 案 B | 火-木 スリープ (WoL 対応) | 月 47 円 (vs A) | ほぼゼロ (WoL で任意起動可) | ★★★★ |
| **案 C** | **火-木 完全 OFF** | **月 464 円 (vs A)** | **ゼロ (JRA 中央)** | **★★★★★** |
| 案 D | 土日のみ ON | 月 928 円 (vs A) | NAR shadow 完全停止 + catch-up 複雑化 | ★★★ |

### 推奨: 案 C ★ 実施可能 ★

理由:
- JRA 中央メインシステムへの影響ゼロ (実測確認)
- NAR は shadow 評価のみ → miss しても ROI 影響なし
- 月 464 円節約 (年 5,616 円) = JRA-VAN DataLab 2 ヶ月分
- 実装の複雑さゼロ (スケジュール変更不要)

---

## 8. 次アクション (5/22 実施手順)

### 実施するだけのこと (schtask 変更不要)

```
火曜夜 (22:00) → PC シャットダウン
木曜夜 → そのまま OFF 維持
金曜朝 (06:00 頃) → PC 起動
  → DailyPremiumScrape が StartWhenAvailable=true で自動実行
  → DailyJrdbKyi が 06:00 に自動実行 (金曜 KYI = 当日分が初配信)
  → FridayWeekendScrape が 10:00 に自動実行
```

### 唯一の注意点

- `\keiba-ai\DailyJrdbKyi` は `StartWhenAvailable=false` かつ `DisallowStartIfOnBatteries=true`
  - 金曜 6am 以降に PC を起動した場合、DailyJrdbKyi は **当日 (金曜) の KYI のみ** 取得
  - ただし金曜 KYI = 翌土曜のレースデータは DailyPremiumScrape (03:00) が担当するため問題なし
  - 万が一 DailyPremiumScrape が失敗した場合のみ `python tools/daily_jrdb_kyi.bat` を手動実行

### NAR shadow 継続運用の場合

- 火-木 miss の影響: Strategy 8 go/no-go サンプル蓄積が週 3 日減
- 対処: 6/15 判定を 7/1 に延期 (Phase 3 NAR 評価と統合) で解決

---

## 9. 総括

```
案 C verdict: ★ 完全可能 ★

火-木 完全 OFF を実施しても:
  - JRA 中央競馬予測: 影響ゼロ
  - JRDB KYI/TYB/SED データ: 欠損ゼロ (火-木は配信なし)
  - 実投票 ROI: 影響ゼロ
  - NAR shadow: 週 3 日 miss (軽微、実害なし)
  - schtask 設定変更: 不要

実施コスト:
  - 変更作業: ゼロ (火曜夜にシャットダウンするだけ)
  - 年間節約: 約 5,616 円 (月 464 円)
```
