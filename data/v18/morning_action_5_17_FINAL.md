# 5/17 (土) 開催 朝 action checklist (concrete instructions)

## Pre-開催 (5/16 金 夜)

### 必須 (admin 権限)
```bash
# admin PowerShell で:
python tools/register_all_phase24_schtasks.py
# 期待: YouTube×2 + Paddock×2 + MorningBriefing×2 全 6 schtask 登録

# 確認
python tools/register_all_phase24_schtasks.py --check
```

### features rebuild (5/17 race_id 反映)
```bash
# jra_races_full.csv が 5/17 race を含む 必要 (TFJV 抽出 後の最新)
python tools/rebuild_all_features.py
# event_effect / pace / hot_streak / layoff / etc 全 8 features 再生成
```

## 5/17 朝 (土)

### 06:30 - 自動 (schtask)
```
Keiba-MorningGoCheck → Discord 通知 1 通
Keiba-MorningBriefing-Sat → Discord 通知 1 通
```

### 07:00 - user 確認 (5 分)
```bash
# 朝の status 確認
python tools/morning_briefing_5_17.py
python tools/check_video_sources.py

# 期待 output:
# - breaker_status: WARN or GO
# - 累計 pnl 確認 (撤退 line +¥50K 内)
# - schtasks: 全 6 registered
```

### 08:00 - 自動 (現行)
```
Keiba-DailyPredict → V15 daily_predict.py
→ data/daily_predictions/20260517.csv 生成
```

### 08:30 - 手動 (5 分)
```bash
# V15 出力 確認 + 戦略⑦ shadow 試算
python tools/strategy8_shadow_runner.py 20260517
# data/strategy8_shadow/20260517.md 生成

# Jackpot 該当馬 確認
cat data/strategy8_shadow/20260517.md | grep "JACKPOT"
```

### 08:55 - 自動 (schtask)
```
Keiba-YouTubeLiveRecord-Sat → YouTube JRA 公式 LIVE 録画 開始
data/youtube_jra_live/20260517_{video_id}.mp4 生成 (約 8h、 2 GB)
```

### 09:00 - 自動 (LIVE 開始)
- JRA 公式 YouTube 開始
- 録画自動

### 09:30 - 手動 (現行)
```bash
python tools/morning_weight_check.py
# 馬体重 急変 検出
```

### 各 R-5 分前 - 自動 (現行)
```
Keiba-RaceAutoNotify → V15 通知 (現行 unchanged)
- 戦略⑦ 適用済 (除外 race は通知なし)
- trio 7点 / 馬連 2点
```

### 各 R-5 分前 - 手動 (5/17 試験用)
- Jackpot 該当馬 確認:
  ```bash
  python tools/strategy8_shadow_runner.py 20260517 | grep "JACKPOT"
  ```
- 該当 race で **手動 単勝 1500 円** 検討 (任意、 試験運用)

### 投票 (現行)
- JRA IPAT で V15 通知通り 手動投票
- Jackpot 該当時 単勝 追加 検討 (5/17 試験のみ、 統合は 5/24+ 判定)

### 21:00 - 自動 (現行)
```
Keiba-DailyResults → daily_results.py
→ data/daily_results/20260517.csv 生成
```

### 22:00 - 手動 (10 分)
```bash
# V15 ROI 確認
python tools/daily_phase23_impact_report.py 20260517

# Jackpot shadow vs 実 verdict
python tools/strategy8_shadow_runner.py 20260517
cat data/strategy8_shadow/20260517.md

# 累計 audit
python tools/drawdown_circuit_breaker.py
```

## 5/17 verdict (5/18 朝)

判定:
1. V15 戦略⑦ 単独 ROI 100%+ ✓ → V15 継続
2. Jackpot 該当 race の shadow ROI 確認
3. 5/24+ V20 投入 path 進行

## 投資額 上限

| 項目 | 金額/日 |
|------|---------|
| V15 戦略⑦ (700 円 × 約 14 race) | 約 ¥10,000 |
| Jackpot alert 試験 (手動、 1500 円 × 該当時) | ¥0-3,000 |
| **合計 上限** | **¥10-13K/日** |

撤退 line: 累計 -¥50K
現状 (5/11): +¥14K (撤退余裕 +¥64K)

## V15 投資保護 (絶対遵守)

- predict_core / daily_predict / app.py / V15 model 一切 不変
- Jackpot は 手動 検討、 V15 通知に影響しない
- 5/17 試験 = 実投票は V15 のみ、 Jackpot は shadow log

## V20 投入 path (5/24+)

5/17 verdict 良好 (ROI 100%+) → Phase 25 playbook で V20 学習 → 5/24 段階投入

## 結論

5/17 = V15 案 B 改 + 戦略⑦ 単独継続 (現行)、 Jackpot pattern は shadow log のみ。
5/17 verdict 後 5/24+ で 戦略⑧ 統合 判定。
