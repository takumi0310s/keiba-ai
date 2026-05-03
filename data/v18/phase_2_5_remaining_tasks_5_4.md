# Phase 2.5 残タスク棚卸し (5/4 朝)

生成: 2026-05-04 (Opus xhigh, Session#7)

参照横断:
- `data/v18/system_audit_5_3.md` (14 改善ポイント)
- `data/v18/v17_v15_improvement_proposals.md` (改良5案)
- `data/v18/improvements_prototyped_5_3.md` (試作結果)
- `data/results/data_coverage_audit_5_4.md` (5/4 データ監査)

## 完了済タスク (Session#1-#6)

| # | タスク | コミット |
|---|--------|---------|
| 1 | Cookie 即時更新 | ccd0c890 (#2) |
| 2 | morning タスクスケジューラ登録 (Sat/Sun 06:30) | ccd0c890 (#2) |
| 3 | daily_predict watchdog 作成 (中断検知) | ccd0c890 (#2) |
| 4 | healthy 4日 ROI 再計算 (案B改 161%) | fcc4741d (#4) |
| 5 | 5/9 開催情報確定 (新潟駿風S/東京エプソムCG3/京都京都新聞杯G2) | fcc4741d (#4) |
| 6 | 5/9 final プラン v1 (案B改) | fcc4741d (#4) |
| 7 | odds_base retro 5/2,5/3 構築 | fcc4741d (#4) |
| 8 | v18/v19 retro 完全版 (バグ判明: 全 bet=0) | fcc4741d/660b13a6 (#4,#6) |
| 9 | DailyPredict watchdog 化 手順書 (admin elevation 必要) | fcc4741d (#4) |
| 10 | v18/v19/v17 LGB model CRLF→LF 復旧 (6 model) | 777cc08e (#5) |
| 11 | 5/3 直前予測 (V17 ULTRA-CLEAN) 分析 (TYB 404) | 660b13a6 (#6) |
| 12 | V15 vs V17 比較 + 改良5案 + 試作 retro | 660b13a6 (#6) |
| 13 | 5/9 final プラン v2 (案B改 維持) | 660b13a6 (#6) |
| 14 | Formation 拡張 retro (採用見送り) | 660b13a6 (#6) |

## 進行中タスク

なし (5/3 夜時点で全て完了 or 5/4 以降に持ち越し)

## 未着手タスク (優先度別)

### 🔴 緊急 (5/4 朝-夜)

| # | タスク | 工数 | 担当 | 状態 |
|---|--------|----:|------|------|
| 1 | **DailyPredict task watchdog 化** (admin) | 5min | ユーザー手動 | 待機 |
| 2 | **netkeiba_race_analysis 再起動** (32日 stale, ra_score) | 30min | 自動 | 5/4 中に着手可 |
| 3 | **netkeiba_stable_comments 再起動** (23日 stale, sc_score) | 30min | 自動 | 同上 |

### 🟠 高 (5/4-5/10)

| # | タスク | 工数 | 期待効果 |
|---|--------|----:|---------|
| 4 | **netkeiba_ai_position 再起動** (35日 stale) | 30min | ai_pos_left/top 復活 |
| 5 | **netkeiba_siblings 再起動** (35日 stale) | 1h | sib_top3_rate, sib_shinba_wr 復活 |
| 6 | **netkeiba_master_index 再起動** (18日 stale) | 1h | master データ更新 |
| 7 | **TYB publish タイミング 連続観測** | 7日 | midday 戦略の生死判定 |
| 8 | **jra_payouts 5/2-5/3 取得** | 10min | jra_payouts.py 実行 |
| 9 | **netkeiba_speed_index 再起動** (4日) | 1h | prev_index_* 計算用 |
| 10 | **.gitattributes models/*.txt -text** 設定 (CRLF再発防止) | 5min | 安全策 |

### 🟡 中 (Phase 2.5 後半 5/11-5/15)

| # | タスク | 工数 | 期待効果 |
|---|--------|----:|---------|
| 11 | **v18/v19 calibration 修正** (Platt scaling) | 半日 | 5/2-5/3 retro で 全 bet=0 解消 |
| 12 | **race-level probability normalization** | 半日 | 確率分布正規化 |
| 13 | **特徴量分布検証** (2026 vs 2024) | 半日 | distribution shift 確認 |
| 14 | **JRDB ot/ov/ow/oz 再取得** (33日 stale) | 1h | オッズ系拡張用 |
| 15 | **odds_history.csv 再取得** (54日 stale) | 半日 | Phase 2.5 BT拡張用 |
| 16 | **netkeiba_training_times date NaN 修復** | 1h | training feature 計算 |

### 🟢 低 (5/16以降 Phase 3 候補)

| # | タスク | 工数 | 期待効果 |
|---|--------|----:|---------|
| 17 | **v15.1 特徴量拡張 (KKA/SKB/SR 逆輸入)** | 2-3日 | V15 → V15.1 (157f) 再 train |
| 18 | **古いモデル削除** (v9-v141, leakfree, ~130MB) | 30min | リポジトリ整理 |
| 19 | **predict_*.py 整理** (13版 → 3版) | 1日 | versioning sprawl 解消 |
| 20 | **archive/ 移動** (古い静的 CSV) | 1日 | リポジトリ整理 |

### 🟢 低 (検証・改良不採用)

| # | タスク | 状態 |
|---|--------|------|
| 21 | Formation 拡張 (Top1-Top4-Top5 等) | ❌ 不採用 (retro で ROI 悪化) |
| 22 | V15+V17 アンサンブル | ❌ 効果限定 (5/3 で 1勝1分1敗) |
| 23 | EV>1 filter | ❌ v18/v19 calibration 後 (5/24-) |

## タスク依存関係

```
🔴 #1 DailyPredict watchdog 化
  ↓ 完了で 5/4 朝以降の事故防止保証

🔴 #2,#3 netkeiba premium 再起動
  ↓ V17 features (ra_score, sc_score) 復活
  ↓ 同列で
  
🟠 #7 TYB publish 観測
  ↓ midday 戦略 生死判定
  ↓ 結果次第で midday script 廃止 or 改良

🟡 #11-13 v18/v19 calibration
  ↓ 完了で v18/v19 部分実弾投入準備
  ↓ retro で ROI 確定
  
🟢 #17 v15.1 拡張
  ↓ Phase 3 再 train 開始
```

## 5/4 着手推奨タスク

1. **netkeiba_race_analysis 再起動** (30min) — V17 ra_score 復活
2. **netkeiba_stable_comments 再起動** (30min) — V17 sc_score 復活
3. **.gitattributes 設定** (5min) — CRLF再発防止
4. **jra_payouts 5/2-5/3 取得** (10min) — Session#1 ROI 集計補強
5. **TYB publish 観測スクリプト 作成** (1h) — Phase 2.5 準備
6. **DailyPredict watchdog 化** — ユーザー手動 (admin)

5/4 全部やっても 3-4時間。GW 中にゆっくり進める。

## 撤退ライン (再確認)

| 累計 | 状態 |
|------|------|
| 現在 (推定) | +14,140円 (USER 報告) |
| 5/9-5/10 最悪 | -4,200円 (累計 +9,940円) |
| **撤退ライン** | **-50,000円** (絶対遵守) |
| 余裕 | 約 +66,140円 |

## 結論

- **5/4-5/8** はレースなし、開発に専念できる5日間
- **致命的タスク 3件** は今日 5/4 中に着手可能
- **Phase 2.5 全体は 5/15 までに完了見込み**
- 5/9-5/10 は V15 案B改 で観察、5/16- v18/v19 部分実弾検討 (calibration 修正後)
- 5/2-5/10 累計損失 -50,000円 余裕大、安全運用可
