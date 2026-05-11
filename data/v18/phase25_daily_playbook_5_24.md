# Phase 25 detailed playbook (5/24+ V20 投入 step-by-step)

5/17 開催 verdict 後、 5/24 V20 投入までの 日次 action 詳細。 各 step に コマンド付き。

## 前提

- ✅ 5/17 開催 V15 案 B 改 + 戦略⑦ 単独継続 完了
- ✅ Phase 21-24 全 35+ tools 実装済
- ✅ V20 features 40 件 確定 (今夜 marathon)
- ✅ Jackpot pattern (top3 64.8% / 単勝 ROI 184%) verify 済

## Day 1: 5/18 (日) — 5/17 verdict + V20 学習 data 構築 開始

### 朝 (5m)
```bash
# 5/17 結果 確認
python tools/daily_results.py --date 20260517
python tools/daily_phase23_impact_report.py 20260517
```

### 夜 (2-3h)
```bash
# 1. 5/17 paddock archive build (前日分)
python tools/paddock_pipeline.py 20260517 --top-n 3

# 2. features 全 regenerate (5/17 data 含めて)
python tools/build_event_effect_features.py
python tools/build_pace_features.py --year-from 2020 --year-to 2026
python tools/build_pace_features_expanding.py
python tools/build_hot_streak_features.py
python tools/build_layoff_features.py
python tools/build_distance_surface_change_features.py
python tools/build_sire_class_down_features.py
python tools/build_remarks_features.py

# 3. 5/17 開催 verdict 集約
echo "5/17 V15 verdict" >> data/v18/5_17_verdict.md
# user 手動: V15 ROI / shadow log 比較 を記載
```

## Day 2: 5/19 (月) — JV-Link + 30y backtest

### 夜 (3-4h)
```bash
# JV-Link 32-bit (まだ未動作なら)
C:\Users\takum\jvlink-venv\Scripts\activate.bat
pip install pywin32
python tools/jvlink_parser.py --test-com
python tools/jvlink_parser.py --datatype RACE --from 20260503 --max 50

# 30y backtest 続行 (2006-2015)
python tools/backtest_30year_collect.py --year-from 2006 --year-to 2015 --datatype SE,HR

# 厩舎コメント scrape (rate limit 注意)
python tools/bulk_scrape_stable_comments_v2.py --year-from 2024 --year-to 2026
```

## Day 3-4: 5/20-21 (火-水) — V20 学習 data 構築 + V20 training

### 5/20 夜 (2-3h)
```bash
# V20 学習 data 構築 (全 features merge)
python tools/v21_training_data_builder.py --year-from 2020 --year-to 2026

# + 今夜の追加 features を 手動で merge
# tools/v21_training_data_builder.py に hot_streak / layoff / distance / sire を追加
# (user task: builder 拡張)

# 専門家 印 scrape
python tools/bulk_scrape_expert_marks.py --year-from 2024 --year-to 2026
```

### 5/21 夜 (3-4h)
```bash
# V20 training (実装、 user task)
# train/train_v135b_intra_ensemble.py をコピーして train/train_v20_*.py 作成
# + V20_ADDITIONAL_FEATURES (40 件) 投入
# + V20_LEAK_FEATURES 必ず除外

python train/train_v20_intra_ensemble.py
# 期待: WF AUC 0.900-0.920、 全条件 ROI ≥ 100%

# 結果 確認
cat data/v20_results.json
```

## Day 5: 5/22 (木) — V20 LIVE retro + 検証

### 夜 (2-3h)
```bash
# V20 LIVE retro (過去 5/17 5/10 5/9 で V20 が どう予測したか)
python tools/v20_live_retro.py --dates 20260510 20260517

# 22 項目 LEAK 監査
# (user task: 既存 validation_*.py に V20 用 ver 追加)
python tools/validation_1_v20_*.py
# ...

# 検証 sample (1 race)
# user 確認: V20 予測 vs V15 予測 比較
```

## Day 6: 5/23 (金) — V20 GO/no-go 判定

### 朝 (1h)
```bash
# 最終 全 check
python tools/morning_go_check.py
python tools/check_video_sources.py
python tools/phase23_smoke_test.py

# V20 vs V15 比較 final report
python tools/v20_vs_v15_comparison.py
# (user task: 比較 report 生成)
```

### 判定基準 (全 PASS で GO)
- [ ] V20 WF AUC ≥ 0.900
- [ ] V20 全年 AUC gap < 0.05
- [ ] V20 実 ROI 全条件 ≥ 100% (戦略⑦ 込み)
- [ ] V20 LIVE retro winner_top1 ≥ 30%
- [ ] V20 LEAK 監査 22 項目 ALL PASS
- [ ] V15 model file freeze 確認 (.pkl.gz 不変)

### NO-GO 時
- V15 案 B 改 + 戦略⑦ 単独継続
- V20 学習 data 更に蓄積 + 再 学習
- 6/8 再判定

## Day 7: 5/24 (土) — V20 段階投入 (GO の場合)

### 朝 06:30
- morning_go_check 自動 → Discord

### 朝 08:00
```bash
# V15 daily_predict (現行 unchanged)
python tools/daily_predict.py
```

### 朝 09:00
```bash
# V20 daily_predict (NEW、 別 file 出力)
python tools/daily_predict_v20.py
# → data/daily_predictions_v20/20260524.csv
```

### 各 R-5 分前
- V15 通知 (現行 unchanged)
- V20 shadow 通知 別 channel (推奨、 V15 と区別)

### Jackpot pattern alert
```bash
# Jackpot pattern 該当馬 自動検出
python tools/live_jackpot_detector.py 20260524 --verbose
# 該当馬 出たら 別 alert (700 → 1500 円 増額 検討)
```

### 投資額 上限 (V20 段階投入)
- V15 案 B 改: 700 円/race × 約 14 race = ~10K 円
- V20 試験: 700 円/race × 1-2 race = 1.4K 円
- Jackpot alert: +800 円 増額 (1500 円/race × 該当時のみ、 月 約 5-10 race)
- 合計: ~15-20K 円/日 (撤退余裕 ¥64K 内)

## Day 8+ (5/25+): 並行運用 1 ヶ月

### 毎開催 (土日)
- V15 通常運用 + V20 shadow 並行
- Jackpot alert 別 channel
- daily_results 自動

### 6/8 (日) - V20 1ヶ月 verdict
- V20 実 ROI vs V15 比較
- V20 winner_top1 / shift 数値確認
- GO なら V20 投資額 増額判定

## V21 投入 path (6/8+)

paddock 動画 features 蓄積後:
- 5/17-6/8 で paddock 約 360 馬 + 過去 archive で +1,500 = 1,860 records
- V21 stacking layer 追加 (V20 + video features)
- 6/15 V21 GO/no-go

## V22 RL 投入 path (9/1+)

V20 安定運用 3 ヶ月後:
- V22 RL bet sizing 学習 (paper trading 2 ヶ月)
- 8/15+ V22 paper → live 段階移行

## V15 投資保護 (5/24+ も継続)

- V15 model file (.pkl.gz) は **完全 freeze**
- V20/V21 は別 file (`keiba_model_v20_*.pkl.gz` 等)
- 1 ヶ月並行運用 6/24+ で V15 archive 判定
- V15 が依然 ROI 上回るなら V15 単独 復帰

## 撤退保護 (Drawdown breaker)

- 撤退 line: 累計 -¥50K
- 現状 (5/11): +¥14K / 撤退余裕 +¥64K
- breaker STOP 自動 trigger
- WARN/HALT で 手動判定

## 結論

5/17 開催 verdict 後 1 週間で V20 学習 → 5/24 段階投入。
今夜の 35+ tools / 40 features / Jackpot pattern を 活用し、 月利 +¥50-100K 達成。

V15 投資保護 完全 維持下で 段階的 V20→V21→V22 path で 市場 圧倒。
