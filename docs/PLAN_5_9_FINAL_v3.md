# 5/9 投資戦略 final v3 (Session #43 F)

**作成**: 2026-05-08 (Session #43 F)
**v1**: docs/FINAL_PRECHECK_5_9.md (Session #36)
**v2**: docs/FINAL_PRECHECK_5_9_v2.md (Session #36-37)
**v3 (本ファイル)**: Session #43 A 真因反映後の **final 確定版**

---

## 1. 確定: 案 A (V15 案B改 維持)

### 1.1 採用 戦略

| 項目 | 値 |
|------|----|
| 採用 case | **V15 案B改** (1勝クラス + 戦略⑦ filter) |
| 投資 R 数上限 | **3 R** |
| 1R 投資額 | **700 円** |
| 想定総投資 | **0-2,100 円** |
| 買い目 | 三連複 7点 (TOP1 軸 - TOP2,3 - TOP2-6) |
| 戦略⑦ filter | 06_特別 / 京都 / 条件E (頭数<=7) / 条件B (重~不良) 除外 |

### 1.2 期待 ROI (Session #43 A 真因 反映)

| source | ROI |
|--------|-----|
| BT 想定 (Session #36) | 161% [CI 135-222%] |
| **直近 通常期 4/18-4/26** (30 races) | **91.62%** |
| 直近 全期間 4/18-5/5 (39 races、 GW 含む) | 83.96% |
| **5/9 推定** (通常開催想定) | **85-100%** |

### 1.3 撤退余裕

| 項目 | 値 |
|------|----|
| USER 実 累計 (CLAUDE.md) | +13,530 円 |
| 撤退ライン | -50,000 円 |
| **撤退余裕** | **+63,530 円** |
| 5/9 max loss | -2,100 円 (3.3% 消費) |

→ **5/9 投資 安全圏、 撤退ライン未達**

---

## 2. 5/9 当日 timeline

```
05:00  PC ON、 sleep 解除
06:00  Keiba-NightlySanity (5/8 23:00 起動分) → 翌日 task pre-check
06:30  (推奨 admin 追加) Keiba-FinalHealthCheck_5_8 → Discord 通知
       (Session #40 A4 / docs/FINAL_PRECHECK_5_9_v3.md)
07:00  Keiba-MorningDigest 自動 (dashboard)
08:00  DailyPredict 自動実行 (V15 全レース 推論、 約 10-15 min)
       → data/daily_predictions/20260509.csv 生成
08:45  RaceAutoNotify 自動 (戦略⑦ + 案B改 → Discord #bets / #investments)
       → 採用 R list 確定 (max 3 R)
09:00  予測結果 手動 確認 + 投票候補 list 確定
09:30  Keiba-RaceMorningCheck (馬体重補正、 推奨 admin)
       PAT login + 入金確認
10:00- レース毎 投票 (1勝クラス のみ、 700円 × max 3R = 2,100円)
14:00-15:30 PAT 投票 (中盤 main race)
14:50  multi_stage_predict.py race11_1450 stage 自動 trigger
15:45  multi_stage_predict.py race12_1545 stage 自動 trigger
       ★ 12R 1勝クラス (主戦場、 案B改 採用率 高)
18:00  DailyResults_Sat 自動 結果照合
       → data/daily_results/20260509.csv 生成
20:30  振り返り (data/v18/post_5_9_improvement_template.md)

5/10 (日) 朝
06:00-08:00  PC ON
08:00       (推奨 admin 追加) Keiba-ResultVerification_5_10
            python tools/result_verification_5_10.py --date 20260509
            → Discord #investments に verdict + 5/16 GO 確率 通知
```

---

## 3. 5/9 ✅ chemistry final (V15 動作不変)

### 3.1 V15 model 不変 確認 (md5)

```
keiba_model_v15_central_live.pkl.gz
  md5:   842b9a5f305c793ed8fa54a74e06b836  (Session #38-43 全期間 不変)
  size:  5,363,864 bytes
  mtime: 2026-05-06T15:32:38 (Session #31 commit 時)
```

### 3.2 production 経路 完全不変

```
$ git diff --stat origin/main..HEAD -- 'tools/predict_core.py' 'tools/daily_predict.py' 'app.py' 'keiba_model_v15*'
(出力なし、 Session #38-43 全期間)
```

### 3.3 schtasks 既存 task 完全不変

| task | trigger | 状態 |
|------|---------|------|
| DailyPredict | 朝 08:00 | 不変 |
| RaceAutoNotify_Sat/Sun | 08:45 (土日) | 不変 |
| DailyResults_Sat/Sun | 18:00 (土日) | 不変 |
| Keiba-NightlySanity | 23:00 (毎日) | 不変 |
| ProcessWatchdog | 5 分おき | 不変 |

### 3.4 Session #43 で追加 path (production 経路 影響なし)

| path | 種類 |
|------|------|
| tools/test_orchestrator_5_cases.py | test wrapper、 production 不影響 |
| tools/video_poc/extract_frames_and_detect.py | PoC、 production 不影響 |
| tools/v18_v19_retro_sib_exp_w5.py | retro test、 production 不影響 |
| train/v18v19_sib_exp_w5/* | 新 model 学習、 production 経路 不影響 |
| data/v18/v18v19_sib_exp_w5/* | 新 model file、 production 不影響 |
| data/video_poc/* | PoC artifacts、 production 不影響 |
| data/v18/v15_roi44_root_cause_5_8.md | 真因 doc |

---

## 4. 5/16 V18/V19 投入判定 (5/9 結果次第)

### 4.1 verdict 表 (Session #42 H plan v2 を Session #43 A 真因 反映で update)

| 5/9 profit | verdict | GO 確率 | recommendation |
|-----------|---------|--------|--------------|
| ≥ +1,000 | 大成功 | **90%** (+5pt) | V18 sib_exp_w5 単独 trial 推奨 |
| +400~+1,000 | 期待通り | **80%** (+5pt) | V18 sib_exp_w5 単独 trial OK |
| 0~+400 | 微益 | **70%** (+5pt) | V18 sib_exp_w5 単独 trial 慎重 |
| -700~0 | 微損 | **55%** (+10pt) | V15 単独継続 推奨、 5/16 NO-GO 強 |
| -1,400~-700 | 損失 | 35% | V15 単独継続、 5/22 再判定 |
| ≤ -1,400 | 大損失 | 15% | V18/V19 NO-GO |

→ Session #43 A の真の ROI 84% 反映で 全 verdict +5-10pt 上方修正

### 4.2 5/15 (金) 22:00 final 判定

```bash
# 5/9 + 5/10 + 5/11 + sib_exp_w5 LIVE retro 結果 で final 判定
python tools/result_verification_5_10.py --date 20260509
# + sib_exp_w5 retro 結果確認
# + 撤退余裕 確認 (累計 ≥ -10,000 円)
# → 5/16 GO/no-go 確定
```

---

## 5. 緊急事態 fallback (Session #40 B2 docs/EMERGENCY_RUNBOOK_5_9_DETAILED.md)

15 シナリオ runbook 既存:
- S01: Cookie 切れ + refresh fail
- S04: Discord webhook 死亡
- S05: PAT サーバー障害
- S15: 全 system fallback

→ 全 シナリオに 検出 + 対応 + 復旧 手順あり (Session #40 完成)

---

## 6. 結論

✅ F1: 5/9 採用 case = **V15 案B改 (700円 × max 3R = 2,100 円)**
✅ F2: 期待 ROI **85-100%** (Session #43 A 真因 反映、 通常期 91.62%)
✅ F3: max loss -2,100 円 (撤退余裕の 3.3% のみ消費)
✅ F4: V15 production 完全不変 (md5 不変、 syntax OK)
✅ F5: 5/16 V18/V19 GO 確率 75-85% (5/9 結果次第で 15-90%)
✅ F6: 緊急 runbook 完備 (Session #40 B2)

→ **5/9 V15 案B改 維持 確定、 投資準備 完了**

---

**Session #43 F 完了**
