# NAR pipeline 設計 (Phase 2.5+ / 2026-05-05)

**対象**: NAR v4 (AUC 0.8145, 22 features Pattern B)
**目標**: 5/16 以降 V15 と並列で daily 自動運用

---

## 1. 全体アーキテクチャ

```
[03:00 〜 17:00] netkeiba.nar 当日カレンダー → race_id 列挙
       ↓
[17:00] tools/scrape_nar_today.py
       - data/nar_all_races.csv に append (results が確定した分)
       - data/nar_today_shutuba.csv に当日出馬表 (発走前)
       ↓
[17:30] tools/predict_nar.py --date YYYYMMDD
       - 22 features 計算 (odds_log は確定オッズ取得待ち、なければ前日オッズで仮計算)
       - LGB+XGB 推論
       - 条件分類 (A/B/C/D/E/X)
       - 三連複 7点 / 馬連 2点 buy 候補生成
       ↓
[各 race 発走 -10分] tools/race_normalize.py 適用 (将来 normalize 統合)
       - 当日確定オッズ更新 → odds_log 再計算 → 推論再実行 → Discord 通知
       ↓
[18:00 〜 21:30 race 終了後] tools/scrape_nar_results.py
       - 結果収集 → daily_results に append
       ↓
[翌日 02:00] tools/nar_daily_results.py
       - 前日 NAR ROI 計算
       - cumulative_results.csv に NAR 行 add
       - 累計 ROI Discord 通知
```

---

## 2. JRA との並列性

### 2.1 開催曜日

| 曜日 | JRA (V15) | NAR (v4) | 主役 |
|------|-----------|----------|------|
| 月 | (休) | 大井ナイター + 一部地方 | **NAR** |
| 火 | (休) | 名古屋・船橋 等 | **NAR** |
| 水 | (休) | 大井・川崎 ナイター | **NAR** |
| 木 | (休) | 浦和・笠松 等 | **NAR** |
| 金 | (休) | 川崎・名古屋 等 | **NAR** |
| 土 | 全国JRA 開催 | 一部 NAR (高知 等) | **JRA**, NAR 補助 |
| 日 | 全国JRA 開催 | 一部 NAR (高知 等) | **JRA**, NAR 補助 |

→ **平日 NAR メイン、土日 JRA メイン**。完全並列、衝突なし。

### 2.2 タスクスケジューラ slot 配置

| 時刻 | task | 内容 |
|------|------|------|
| 03:00 | DailyPremiumScrape (既存) | JRA 週末 premium |
| 06:00 | DailyJrdbKyi (既存) | JRA JRDB |
| 08:00 | DailyPredict (既存) | JRA 当日全レース |
| 13:00 | **(新) NarMidDayCalendar** | NAR 当日カレンダー取得 |
| 16:30 | **(新) NarDailyScrape** | NAR 当日出馬表 + 前夜オッズ |
| 17:00 | **(新) NarDailyPredict** | NAR 推論 + 候補抽出 (発走 -2h) |
| 19:00 | **(新) NarLiveOddsRefresh** | NAR 主要 race のみ確定オッズ取得 + 再推論 + Discord (時間幅で複数発走対応) |
| 21:30 | **(新) NarDailyResults** | NAR 結果収集 + ROI |
| 23:00 | NightlySanity (既存) | 翌日 task 健全性 |

→ JRA 既存 8 task + NAR 新規 5 task = 合計 13 task。
→ 既存 JRA 自動化と時間衝突なし。

---

## 3. オッズ取得経路

### 3.1 候補

| ソース | 信頼性 | スクレイピング難易度 | NAR カバレッジ |
|--------|-------:|------------------:|--------------:|
| **netkeiba.nar (db.netkeiba.com)** | 高 | 中 (実績あり) | 全場 |
| 楽天競馬 | 高 | 中 | 全場 |
| 地方競馬公式 | 高 | 高 (各場ごと別 site) | 各場 |

### 3.2 推奨

- **第1選択: netkeiba.nar**: 既に scrape_nar_all.py で実績あり、cookies 共有可
- **第2選択: 楽天競馬**: netkeiba がブロックされた場合のフォールバック
- 公式は最終フォールバック (場ごとの実装コスト高)

### 3.3 取得タイミング

| 段階 | 時刻 | 用途 | odds 性質 |
|------|------|------|-----------|
| 前夜 | 23:00 | 翌日 race 推論用 | 直前ではない、参考値 |
| **発走 -120分** | race 個別 | predict 第1発火 | **前売りオッズ** |
| **発走 -30分** | race 個別 | predict 第2発火 | 確定オッズに近い |
| **発走 -10分** | race 個別 | 最終 Discord 通知 | **最終オッズ** |

→ NAR は ナイター (19:00 等) が多く、JRA より時刻が分散。

---

## 4. 投資判断フロー (NAR 用 案B改 相当)

### 4.1 候補レース絞り込み (条件 A-X、JRA と同じ命名)

| 条件 | 内容 | NAR での想定 ROI 目安 |
|------|------|---------------------:|
| A | 8-14 頭/1600m+/良 | 100% (案B改 推定) |
| B | 8-14 頭/1600m+/重 | 90-110% (sample 小) |
| C | 15 頭+/1600m+/良 | 100-120% (NAR は 15 頭+ rare) |
| D | 1200-1400m | 90-100% (NAR の主流) |
| E | 7 頭以下 | 80% (NAR の少頭数も多い) |
| X | 15 頭+/重 | 100% (rare) |

**注**: NAR 実 ROI は backtest で再確認 (E task)。条件分類は JRA と共通でも、ROI 性質は異なる可能性大。

### 4.2 軸馬選定

LGB+XGB ensemble の p_ens TOP1 を軸。
ただし以下 制約:
- p_ens >= 0.30 (NAR 平均的な race 内 max prob)
- pop_rank が極端な大穴 (>10) の場合は除外候補

### 4.3 三連複 7点 / 馬連 2点

JRA と同じフォーメーション:
- 三連複 7点: TOP1軸 - TOP2,TOP3 - TOP2..TOP6 (1×2×5)
- 馬連 2点: TOP1-TOP2, TOP1-TOP3 (条件 E のみ)

### 4.4 上限金額 (1日)

| 段階 | 金額 | 備考 |
|------|------|------|
| 5/16-5/22 (paper / 試行) | 100 円/race × 5 race = **500 円/日** | 最小試行 |
| 5/23-5/29 (継続観察) | 200 円/race × 7 race = **1,400 円/日** | 慣熟 |
| 6月以降 (本格) | 300 円/race × 7 race = **2,100 円/日** | full ramp |

→ 累計上限 -50,000円 (JRA と合算)。

---

## 5. 撤退判定基準

### 5.1 即時停止 (1 race 単位)

- race 直前で odds_log の取得失敗 → そのレース見送り
- horse_weight 60% 以上欠損 → 推論信頼度低下、見送り

### 5.2 1日停止

- その日の 第1レース時点で Discord 通知不調 (3 連続 失敗) → 当日 NAR 停止
- 累計 1日 -1,000 円 以上 (試行期間) → 残り race 停止

### 5.3 週単位停止

- 週累計 -3,000 円 以上 → 翌週 NAR 完全停止、原因究明
- model AUC validation < 0.75 (新データ追加後) → モデル再学習まで停止

### 5.4 完全停止

- 累計 (JRA+NAR 合算) -50,000 円 → **全投資停止**、user 判断 wait

---

## 6. 5/9 (this Sat) 投入なし、5/16 から段階運用

5/9 は V15 案B改 単独維持。NAR は 5/12 (火) 以降 paper trading で観察開始。

| 日付 | NAR action |
|------|------------|
| 〜 5/11 | Pipeline 構築継続、admin schtasks 登録 |
| 5/12 (火) | NAR paper trading 開始 (Discord 通知のみ、investment 0) |
| 5/13-15 | paper 継続、毎日 ROI 評価 |
| 5/16 (土) | JRA 案B改 (継続)、NAR 試行 500円/日 開始 (条件 1日 -1,000円 で stop) |
| 5/17 (日) | 同上 |
| 5/18-22 (月-金) | JRA なし、NAR 500円/日 ramp |
| 5/23-24 (土日) | JRA + NAR 並列、累計判定 |
| 5/25 | Phase 3 移行判断 (NAR 条件 ROI > 100% なら 1,400円/日 へ) |

---

## 7. 関連ファイル想定

| 役割 | path | 備考 |
|------|------|------|
| 当日カレンダー | tools/scrape_nar_today.py (新) | 翌日 race_id 列挙 |
| 結果スクレイプ | tools/scrape_nar_results.py (新) | race 後 finish 取得 |
| 一括 daily wrapper | tools/nar_daily_pipeline.bat (新) | 各 step 順次実行 |
| schtasks 登録 | tools/register_nar_schtasks.ps1 (新) | admin |
| 汎用 predict | tools/predict_nar.py (新) | --date YYYYMMDD --race-id RID |
| config | tools/nar_predict_config.json (新) | 投資ルール |
| 結果整合 | tools/nar_daily_results.py (新) | 翌02:00 |
| Discord | (既存 notify_done.py 流用) | webhook 統合 |

→ 本 Phase で骨格作成 (D, C task)、本格実装は 5/12 までに継続作業。

---

## 8. リスクと留意

### 8.1 NAR 固有の留意

- **少頭数 race 多** (5-8 頭): 単勝/複勝の妙味が低い、三連複は組合せ不足
- **オッズ振れ大**: ナイター + 投票総額が JRA の 1/100、瞬間的に odds が動く
- **レース間隔 短い**: ナイター連続 race で連続予測の負荷
- **休止場 不定期**: 天候・場の都合で開催キャンセル → schtasks の race_id 0 件対応

### 8.2 model 限界

- AUC 0.8145 は Pattern B (odds_log 含む)、リーク特性
- pure 馬力予測 (Pattern A) ではなく、市場予測の上乗せ
- 大穴 (人気 8 番以下) で外す傾向

### 8.3 累計上限

- JRA + NAR 合算 -50,000 円ライン
- 累計 +14,140 円 を死守、減損 -64,140 円 まで耐える計算
- 2 週間試行で -10,000 円 超えた場合は早期 stop

---

## 9. 結論

- 平日 NAR + 土日 JRA で完全並列
- 5/16 から 500円/日 paper-light 試行、6 月本格化
- 投資判断は条件 A-X (JRA と同型)、ROI は backtest で再確認 (E task)
- pipeline は新規 5 task + 既存 JRA 8 task = 13 task
- 撤退基準 多段階 (race / 日 / 週 / 累計)
