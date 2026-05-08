# Session #47 完了 (調教解析 PoC + 5/9 全 R 予測 + 5/10 verify framework)

date: 2026-05-08
branch: dev/training-poc (main から分岐、 main 不変)
commits: 5 + 1 fix

## 完了 領域

| # | 領域 | 内容 | 出力 |
|---|------|------|------|
| A | 調教 features audit | V15 12 features + 拡張 15 候補 | `data/v18/training_audit_5_8.md` |
| B | training AUC test | V15 vs V15+8 features script | `tools/training_auc_test.py` |
| C | 5/9 全 R 予測 | 36 R V15 予測 | `data/v18/predictions_5_9_all.json` |
| D | 5/10 verify framework | 全 R 結果照合 + Discord | `tools/verify_all_5_10.py` |
| E | 動画解析 PoC | YOLOv8 skeleton (13:00 公開待ち) | `tools/video_analysis_poc.py` |

## 5/9 race 構成

- 京都 12R (1R-12R)、 東京 12R、 新潟 12R = 36 R
- 重賞: 東京 11R エプソムC (G3) / 京都 11R 京都新聞杯 (G2) / 新潟 11R 駿風 S (OP)
- 投資: 12R 1勝クラス のみ、 V15 案B改、 上限 2,100 円

## V15 model md5 (不変保証)

- 期待値 (CLAUDE.md 記載): `842b9a5f...` (古い)
- **実 md5**: `309dffc65504f056d233c65665c319d5`
- Session #47 期間中 model file 一切変更しない (達成)

## 投資保護 (絶対遵守、 達成)

- main branch 不変 ✅
- dev/sprint1 branch 不変 ✅
- predict_core.py 不変 ✅
- daily_predict.py 不変 ✅
- V15 model 不変 ✅
- schtasks 既存 38 件 不変 ✅
- 5/9 朝 V15 動作 不変 ✅

## 拡張調教 features 8 個 (B で AUC test)

1. training_time_5f / training_time_3f / training_pace_5f_3f
2. days_since_last_training / training_count_2w
3. cyb_train_baba_enc / cyb_train_amount / cyb_train_change_enc

期待 AUC delta: +0.001-0.008 (1勝クラス で大効果想定)

## 5/10 朝 運用

```bash
# 06:00 以降 (全 R 結果 公開後)
python tools/verify_all_5_10.py

# verdict 確認
cat data/v18/verification_5_10.md

# Discord #アップデート で受信確認
```

## 5/8 13:00 以降 任意 work

```bash
# 動画 PoC (重賞のみ、 失敗 OK)
python tools/video_analysis_poc.py --check-deps
python tools/video_analysis_poc.py --race-id <京都新聞杯>
```

## 採用判定 plan (5/10 D verdict 後)

| シナリオ | 判定 |
|---------|------|
| AUC delta +0.002+、 top1 hit ≥ 30%、 grade-monotonic | V20 候補追加 (Phase 3) |
| AUC delta +0.002+、 top1 hit < 30% | 5/16, 5/17 で再計測 |
| AUC delta < +0.001 | 棚卸しのみ、 V15 unchanged |

## 関連 commits (dev/training-poc)

```
c5cc5a83 Session #47 C fix: predict_race API 修正 + JSON top3 構築
1357981e Session #47 E: 動画解析 PoC skeleton
86ef47cf Session #47 D: 5/10 結果照合 framework
6e8c102f Session #47 C: 5/9 全 R 予測 tool
d060d349 Session #47 B: training AUC test
ca424708 Session #47 A: training audit
```

## PR draft

- url: https://github.com/takumi0310s/keiba-ai/pull/new/dev/training-poc
- base: main
- head: dev/training-poc
- title: "Session #47: 調教解析 PoC + 5/9 全R検証"
