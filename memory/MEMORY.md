# keiba-ai MEMORY (毎セッション 1page で context 把握)

**目的**: Claude Code session 起動時に最初に読む 1 page。 5-10 min かかっていた context 把握を 1-2 min に短縮。
**最終更新**: 2026-05-06 (Session #27)
**最新 commit**: A 修正 (5/2-5/3 真相確定) → B 復活 (本書) → C premium bug → D JRDB retry

---

## ベースライン

| 項目 | 値 |
|------|-----|
| 現行モデル | **V15** (本番、AUC 0.8939、150 features) |
| 学習データ | 527,280 行 (2015/01-2025/12) |
| Pattern A (リークフリー評価) + Pattern B (実運用) | 別 pkl |
| 戦略 | **案B改** = 12R 1勝クラスのみ、上限 2,100円/日 |
| filter | 戦略⑦ (06_特別/京都/条件E/条件B 除外) |
| 撤退ライン | **-50,000 円** (絶対) |

## 累計収支 (5/6 真相確定)

- 生データ全期間: **+13,530 円** (5/3 終端 +13,220 + NAR 柏記念 +310)
- USER 申告: +14,140 円 (±610 円差、5/4 月曜の何か、要 USER 確認)
- 撤退余裕: **+63,530 円** (生データベース)
- 詳細: `data/v18/may_2_3_truth_audit_5_6.md`

## 直近 数字

| 日 | USER 実投資 | hits | 備考 |
|---|------------|------|------|
| 5/2 | -9,350 円 (15 R) | 1 | フローラS 軸 #1 が 17 番手など全滅 |
| 5/3 | -7,950 円 (22 R) | 4 | 4/26 の好調から 31% に低下 |
| 5/5 | +310 円 (NAR 1 R) | 1 | 柏記念 三連複 #3-#8-#10 配当 1,010円 |

## リークフリー features 8 件 (Pattern A 除外)

| # | feature | 由来 |
|---|---------|------|
| 1 | `odds_log` | 確定オッズ → 投票締切後 |
| 2 | `horse_weight` | 当日馬体重 → 発走 70 分前発表 |
| 3 | `condition_enc` | 馬場状態 → 当日朝発表 |
| 4 | `weight_change` | horse_weight 派生 |
| 5 | `weight_change_abs` | horse_weight 派生 |
| 6 | `weight_cat` | horse_weight 派生 |
| 7 | `weight_cat_dist` | horse_weight 派生 |
| 8 | `cond_surface` | condition_enc 派生 |

## 重要 commit (Phase 2.5+)

| commit | 内容 |
|--------|------|
| `bed809ec` | Phase 2.5+ 最終総括 (51.5h レポート + 自己診断) |
| `7358a74a` | 5/6 朝 (健康診断 + 5/2-5/3 反省 + 馬体重補正機構) |
| `86cd1da5` | 緊急 3 件対応 (ProcessWatchdog v2 + fire_check + chihou) |
| `f408d93d` | 5/5 柏記念 NAR v4 ハイブリッド予測 |
| `7c5ba9f8` | V15.1 SKB +0.0699 大発見 (Phase 3 候補) |
| `c106f66b` | NAR pipeline 5 task 本実装 (5/12 paper 開始) |
| `74eb10b7` | race-level normalization + V18/V19 retro |
| `2b6dc4eb` | 5/9 本番最終調整 |
| `9c88d27c` | 28 task 静音化 vbs ラッパー |
| `e20bbc0c` | .gitattributes (CRLF 再発防止) |

## 当日朝の運用フロー (5/9 土曜想定)

```
06:30 Keiba-Morning_Sat → Discord #bets (V17 11R/12R 軸候補)
08:00 DailyPredict       → data/daily_predictions/{ymd}.csv
08:50 Keiba-AM8FireCheck → 発火確認
09:00 12R race_name 確認 (1 勝クラスかどうか)
09:30 Keiba-MorningWeightCheck_Sat → 馬体重補正 + Discord (新規 Session #26)
14:00-15:30 PAT 投票 (採用 R × 700 円、上限 2,100 円)
18:00 DailyResults_Sat + RaceDayReport_Sat → Discord 結果通知
20:30 post_5_9_improvement_template.md 振り返り埋め
```

## 馬体重補正機構 (Session #26 新設)

- `tools/morning_weight_check.py` (395行) + `.bat` + `register_*.ps1`
- 09:30 で predict_one_race 再実行 → 朝予測との diff > 5% → Discord アラート
- 馬体重 ±15kg → 軸変更検討
- 5/9 試運転、5/16 本格運用予定
- 5/3 動作テストで V15 が馬体重を相当重視と判明 (TOP1 score 0.249→0.586 +0.337)

## 5/9 投資判断 (絶対遵守)

- ❌ 11R 投票禁止 (新潟駿風S 距離不適合 / 東京エプソムC G3 / 京都京都新聞杯 G2)
- ❌ 1R 700 円超え禁止
- ❌ 1日 2,100 円超え禁止
- ❌ V18/V19 投入禁止 (5/16 以降)
- ❌ NAR 投入禁止 (5/12 paper 開始)
- ✅ V15 案B改 のみ
- ✅ trio_bets 列をそのまま使う (再計算しない)

## 撤退基準 (多段階)

- 5/9 単日 ROI < 50% → 5/10 投資停止
- 5/9-5/10 累計 -10,000 円 → 翌週 (5/16) 投資停止
- 累計 -50,000 円 → 完全撤退

## 5/6-5/9 の追加 admin (累計 3 件)

```powershell
# 1. ProcessWatchdog v2 (commit 86cd1da5)
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1

# 2. 馬体重補正 (commit 7358a74a)
PowerShell -ExecutionPolicy Bypass -File tools\register_morning_weight_check_schtasks.ps1

# 3. JRDB AM 9:00 retry (Session #27 で新設、本日)
PowerShell -ExecutionPolicy Bypass -File tools\register_jrdb_retry_schtasks.ps1
```

## 5/6-5/8 平日 quick wins

| # | task | 工数 |
|---|------|------|
| 1 | premium CSV append bug 修正 (Session #27 C で対応中) | 1.5h |
| 2 | JRDB AM 9:00 retry (Session #27 D で対応中) | 1h |
| 3 | 5/8 21:00 後 12R race_name 確認 | 5min |
| 4 | Cookie --check | 1min |

## 引き継ぎ書 v2 ルール (絶対)

- 数字は必ず生データで再検証 (USER 申告 transfusion 禁止)
- USER 実投資 vs BATCH 仮想 を絶対混同しない
- 取り返し禁止 / 累計損失拡大 NG / 撤退ライン -50,000 円
- 60 時間目 連続作業の翌日朝、無理せず

## 詳細 doc

| 順 | doc | 用途 |
|----|-----|------|
| 1 | `docs/PHASE_2_5_PLUS_FINAL_RECAP_5_5.md` | 51.5h 全総括 |
| 2 | `docs/UPDATE_INVENTORY_20260505.md` | 6 領域棚卸し |
| 3 | `docs/HANDOFF_5_5_TO_5_9.md` | 5/9 投資詳細 |
| 4 | `docs/system_self_diagnosis_5_5.md` | 自己診断 |
| 5 | `docs/next_session_checklist.md` | 起動チェックリスト |
| 6 | `data/results/20260509_pat_checklist.md` | **5/9 朝 必読** |
| 7 | `data/v18/may_2_3_truth_audit_5_6.md` | 真相確定 (本日) |

---

迷ったら本書 → `docs/PHASE_2_5_PLUS_FINAL_RECAP_5_5.md` の順で読む。
