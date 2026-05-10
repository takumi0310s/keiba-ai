# Phase 10 D: netkeiba マスターコース 完全 audit (5/10)

> Session #87 (2026-05-10 夜) Phase 10 D 領域
> 対象: ★ netkeiba マスターコース (¥4,980/月、 既加入) ★
> 趣旨: read-only audit、 V15 production 完全不変

---

## 1. 加入サービス概要

| 項目 | 値 |
|------|----|
| サービス名 | netkeiba マスターコース |
| 月額 | ¥4,980 (税込) |
| 加入状況 | ✅ 加入済 (2024-12 加入、 継続中) |
| Cookie 認証 | .env NETKEIBA_COOKIE で保存 |
| Cookie 自動更新 | tools/refresh_cookie.py (Playwright) |
| 対応 platform | iOS / Android アプリ + ★ PC web ★ |
| 公式 announce 2024-12 | アプリ + スマホ web 限定で開始 |
| 2025+ 状況 | ★ PC web 対応 拡大中 (race.netkeiba.com 経由 取得可) ★ |

---

## 2. 全機能 完全 list

### 2.1 ★ AI 系 (V20 統合最有力候補) ★

| 機能 | 内容 | V15 既統合 | priority |
|------|------|-----------|---------|
| ★ AI 展開予測 ★ | 各馬の通過順位 + 上がり 3F 予想 | ✅ 部分 (netkeiba_ai_position.csv 67,953 行) | ★★★★★ |
| ★ AI 波乱度予測 ★ | レースの波乱度 (低/中/高) | ❌ 未統合 | ★★★★★ |
| AI 馬予想 | 各馬 3 段階予想 (◎○▲) | ⚠ 部分 (newspaper_ai 経由) | ★★★ |
| AI 予想タイム | 各馬走破タイム + 個別ラップ | ✅ netkeiba_ai_predict_times.csv (17 行のみ) | ★★★★★ |
| AI レース分析 | レース全体 AI 解説 | ⚠ scrape 候補 | ★★★ |
| ★ AI レースグレード ★ | 50 レース/月 (条件分類拡張) | ❌ 未統合 | ★★★ |
| AI データシミュレーション | 月制限あり 2 種 | ❌ 未統合 | ★★ |
| AI アドバイザー | 馬券診断 (買い方推奨) | ❌ 未統合 | ★★ |

### 2.2 ★ オリジナルデータ (V20 統合候補) ★

| 機能 | 内容 | V15 既統合 | priority |
|------|------|-----------|---------|
| ★ 個別ラップ ★ | 全頭 1F-15F の高精度ラップ (動画解析) | ⚠ 部分 (netkeiba_race_lap.csv 24,900 行) | ★★★★★ |
| ★ トラックバイアス ★ | レース video から馬場の有利位置を測定 | ⚠ 部分 (netkeiba_track_bias.csv 25,838 行) | ★★★★★ |
| ★ 走行距離 / ポジション ★ | 各馬の実走行距離 + ポジションロス | ❌ 未統合 | ★★★★★ |
| ★ 個別ラップ premium model ★ | タイム指数 上位モデル (斤量補正) | ❌ 未統合 | ★★★★ |
| マスターインデックス | 拡張 race index | ✅ netkeiba_master_index.csv (139,674 行) | ★★★ |

### 2.3 機能 (既統合 / 部分統合)

| 機能 | 内容 | V15 既統合 |
|------|------|-----------|
| タイム指数 | speed.html | ✅ netkeiba_speed_index.csv (143K 行) |
| 厩舎コメント | comment.html | ✅ netkeiba_stable_comments.csv (857 行、 部分) |
| 調教タイム | oikiri.html | ✅ netkeiba_training_times.csv (2.6K 行) |
| レース短評 | 備考 | ✅ netkeiba_race_review.csv (277K 行) |
| 新馬評価 | shinba_eval | ✅ netkeiba_shinba_eval.csv (8K 行) |
| 母産駒成績 | siblings | ✅ netkeiba_siblings.csv (17K 母馬) |

### 2.4 リアルタイム機能

| 機能 | 内容 | V15 既統合 |
|------|------|-----------|
| リアルタイム更新 | 30 秒以内 update | ⚠ Stage 2 経由 |
| レース VTR | JRA 公式映像 clip | ❌ 未統合 (RV 経由 が主) |
| 速報 オッズ | 直前 オッズ | ✅ Stage 2 経由 |

### 2.5 投票補助 (V20 投資判断 用)

| 機能 | 内容 | V15 既統合 |
|------|------|-----------|
| みんなの馬券比較 | 集合知 集計 | ❌ 未統合 |
| オッズセンサー | オッズ変動 alert | ❌ 未統合 |
| レース相性度 | 馬同士の相性 | ❌ 未統合 |
| AI 馬券買い方診断 | フォーメーション推奨 | ❌ 未統合 |

### 2.6 PDF / カスタム

| 機能 | 内容 | V15 既統合 |
|------|------|-----------|
| PDF レポート | DL 可 | 不要 |
| カスタム通知 | 個別 alert | ❌ 未統合 (Discord で代替) |
| 馬メモ一括入力 | 個人メモ | 不要 (運用) |

---

## 3. PC 版対応状況 (★ 重要 ★)

### 3.1 公式アナウンス (2024-12)
- ★ 当初: アプリ + スマホ web 限定 ★
- 「PC 版のリリース時期は未定」 と公式 announce

### 3.2 2025-2026 現状 (ユーザー要望 audit)
- ★ race.netkeiba.com 経由で **多くの機能が PC web から取得可能** ★
- 2026-04-04 以降 race.netkeiba.com で HTTP 400 障害 → 復旧後 stable
- 既存 keiba-ai で取得済 csv (track_bias / race_lap / ai_position / ai_predict_times / master_index) は PC scrape で取得済
- ★ 残機能 (AI 波乱度 / 走行距離 / レース相性度) の PC scrape 経路 未確定 ★

### 3.3 fallback 経路
- スマホアプリ → CSV export → keiba-ai 取込 (今後検討)
- mobile API 経由 (規約注意)

---

## 4. data 取得方法

### 4.1 既実装 scraper (tools/scrape_master_course.py、 tools/scrape_master_index.py)
- 個別ラップ (laps): netkeiba_individual_lap.csv
- マスターインデックス: netkeiba_master_index_mc.csv (7-field schema)
- レースラップ: netkeiba_race_laps.csv
- トラックバイアス: netkeiba_track_bias.csv (25,838 行)
- ペース予測: netkeiba_pace_prediction.csv (★ AI 展開予測 統合の base ★)
- 波乱度: netkeiba_upset_level.csv (空ファイル → ★ scrape 必要 ★)
- AI 予想タイム: netkeiba_ai_predict_times.csv (17 行のみ → ★ 大量取得 必要 ★)

### 4.2 AI 関連 既取得
- AI 展開 (AI position): netkeiba_ai_position.csv (67,953 行)
- AI 意見: netkeiba_ai_opinion.csv (4,930 行)
- 新聞 AI: netkeiba_newspaper_ai.json
- レース分析: netkeiba_race_analysis.csv

### 4.3 ★ V20 統合 plan ★

| 取得対象 | source | 取得状況 | V20 投入 priority |
|---------|--------|---------|------------------|
| AI 展開予測 (各馬通過順) | netkeiba_ai_position.csv | ✅ 67K 行 | ★★★★★ |
| AI 波乱度予測 | netkeiba_upset_level.csv | ❌ 空、 scrape 必要 | ★★★★★ |
| AI 予想タイム (個別ラップ) | netkeiba_ai_predict_times.csv | ⚠ 17 行のみ、 大量取得必要 | ★★★★★ |
| トラックバイアス | netkeiba_track_bias.csv | ✅ 25K 行 | ★★★★★ |
| 個別ラップ | netkeiba_race_lap.csv | ✅ 24K 行 | ★★★★ |
| 走行距離 / ポジションロス | (未取得) | ❌ scrape 必要 | ★★★★ |
| みんなの馬券比較 | (未取得) | ❌ scrape 必要 | ★★ |
| レース相性度 | (未取得) | ❌ scrape 必要 | ★★ |

---

## 5. 規約 確認

| 行為 | 判定 | 備考 |
|------|------|------|
| 個人利用 (AI 学習) | ✅ OK | マスターコース 加入者範囲 |
| スクレイピング (自前 model 用) | ⚠ グレーゾーン | DELAY_SECONDS=10 conservative rate limit、 大量並列 NG |
| data 再配布 | ❌ NG | 規約違反 |
| 商用 | ❌ NG | 個人利用範囲 |
| 公開 (web / SNS) | ❌ NG | 著作権侵害 |

→ 既存 scraper は **DELAY_SECONDS=10 / RETRY_DELAY=60** で conservative rate、 規約遵守設計。

---

## 6. ★ V20 統合 features 候補 (期待 +30-40 features) ★

### 6.1 AI 系 (8-10 features)
- nk_ai_position_pass1/2/3/4 (通過順 予想)
- nk_ai_agari_pred (上がり 3F 予想)
- nk_ai_upset_score (波乱度 0-1)
- nk_ai_predict_time_total (走破タイム予想)
- nk_ai_predict_lap_first3f / last3f
- nk_ai_grade (レースグレード 50 段階)

### 6.2 ラップ / バイアス系 (10-12 features)
- nk_track_bias_inner / outer / center (枠別有利度)
- nk_track_bias_pace_speed (ペース速度バイアス)
- nk_individual_lap_avg / std (個別ラップ統計)
- nk_individual_lap_first3f / last3f (前後半)
- nk_running_distance_total (実走行距離)
- nk_position_loss (ポジションロス)
- nk_race_lap_pattern_enc (レースラップパターン分類)

### 6.3 投票補助 (5-8 features)
- nk_combinations_top_picks (集合知 上位)
- nk_odds_sensor_alert (オッズ急変)
- nk_horse_compatibility (相性度)

### 6.4 期待 V20 features 追加数
- ★ 合計 30-40 features ★ (V15 150 → V20 (master 単独) 180-190)

### 6.5 期待 AUC 寄与
- AI 展開予測: +0.005-0.012 (corr 高)
- AI 波乱度: +0.003-0.008 (波乱 R 識別)
- AI 予想タイム / 個別ラップ: +0.005-0.010
- トラックバイアス: +0.003-0.008 (枠 / 脚質 補正)
- 走行距離 / ポジションロス: +0.005-0.012 (前走補正)
- ★ 合計: +0.020-0.040 ★

---

## 7. 結論

✅ D1: PC 版対応状況 (race.netkeiba.com 経由 多機能 PC scrape 可、 残数機能は確認中)
✅ D2: 全機能 list (AI 系 8 / オリジナルデータ 5 / 既統合 6 / リアルタイム 3 / 投票補助 4 / PDF/カスタム 3)
✅ D3: data 取得 既実装 (scrape_master_course.py / scrape_master_index.py)、 既取得 csv 8 件
✅ D4: V20 統合 plan (★ 30-40 features 追加、 期待 AUC +0.020-0.040 ★)
✅ D5: 規約 確認 (DELAY_SECONDS=10 conservative rate、 個人 AI 学習範囲 OK)

→ **5/12-5/13 で AI 波乱度 + AI 予想タイム + 走行距離 大量取得 → 5/14-5/15 V20 学習投入**
→ **5/10 朝 V15 完全保証** (read-only audit、 V15 model 不変)

Sources:
- [マスターコースのご案内 - netkeiba](https://dir.netkeiba.com/master_course/)
- [netkeiba Master Course Update - PR](https://www.netdreamers.co.jp/news/release_20250401-7234/)
- [マスターコース大型アップデート](https://info.netkeiba.com/?pid=info_detail&id=1514)
- [AIで競馬を攻略 - netkeiba](https://race.netkeiba.com/AI/AI.html)
