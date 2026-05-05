# 5/9 (土) 運用ガイド — 案B改 完全フロー

**目的**: 朝起きてから投票完了まで迷わず動く。連続作業 50時間後でも判断ミスしない設計。

**採用方針**: V15 案B改 / 12R 1勝クラスのみ / 上限 2,100円 / 期待 ROI 161%
**累計**: +14,140円 (これを毀損しないことが最優先)

---

## 0. 前夜 (5/8 金) 21:00 後 — **必須確認**

```bash
# 1. friday_weekend_scrape ログ確認 (出馬表事前 scrape)
ls -la C:/Users/takum/keiba-ai/logs/friday_weekend_scrape*.log
# expected: 21:00 後に最新ログあり、エラー終了でないこと

# 2. 5/9 12R race_name 確認 (1勝クラスかどうか)
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}',
                     headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '→', (m.group(1).strip() if m else 'NOT_FOUND'))
"
# expected: 全 3R の race_name に "1勝" 含む
```

**判定**:
- 全 3R が 1勝クラス → 5/9 投資 GO (3R × 700円 = 2,100円)
- 一部のみ 1勝クラス → 該当 R のみ投資
- 全て 1勝クラスでない → **5/9 は無投資**、累計 +14,140円維持

---

## 1. 5/9 朝 06:30 — 自動 morning_top_races

タスクスケジューラ Keiba-Morning_Sat (06:30 daily) が自動発火。

| script | 役割 |
|--------|------|
| `tools/morning_top_races.bat` (wrapper) | Git Bash 経由 → morning_top_races.sh 起動 |
| `tools/morning_top_races.sh` | 11R/12R V17 morning 予測 + Discord 通知 |

**確認**: 06:35 頃に Discord #bets チャンネル に morning 通知が来ること。

---

## 2. 5/9 07:00 — Discord 確認

| 通知 | チャンネル | 内容 |
|------|-----------|------|
| morning_top_races 完了 | #bets or #updates | 11R/12R V17 予測 |
| Cookie 健全性 | #updates | refresh_cookie の last check |

**異常時**:
- 通知来ない → `logs/morning_top_races_wrapper_20260509.log` を確認
- Cookie 切れ → `python tools/refresh_cookie.py --auto` (対話なし、認証情報保存済の場合)

---

## 3. 5/9 08:00 — DailyPredict 自動発火 (watchdog 化済)

タスクスケジューラ keiba-ai\DailyPredict (08:00 daily) が watchdog 経由で発火。

| script | 役割 |
|--------|------|
| `tools/daily_predict_watchdog.py` | daily_predict.py を起動 + 死活監視 + 自動再起動 |
| `tools/daily_predict.py` | V15 全 35 races (3場 × 12R 仮想) 推論 |
| 出力 | `data/daily_predictions/20260509.csv` |

**完了 expected**: 08:30〜09:00 (35 races の inference)。

**確認**:
```bash
ls -la C:/Users/takum/keiba-ai/data/daily_predictions/20260509.csv
# expected: 35 行程度、最新タイムスタンプ
```

---

## 4. 5/9 09:00-10:00 — 投資判定 (手動 5 分)

```bash
cd C:\Users\takum\keiba-ai

# 12R のみ + 1勝クラス filter
python -c "
import pandas as pd
df = pd.read_csv('data/daily_predictions/20260509.csv', dtype={'race_id':str})
print('全レース:', len(df))
df12 = df[df['race_num'].astype(int) == 12]
print('12R 件数:', len(df12))
for _, r in df12.iterrows():
    name = str(r.get('race_name',''))
    cls = '1勝' in name
    cond = r.get('condition','')
    course = r.get('course','')
    top1 = r.get('top1_num','')
    print(f\"  {course} 12R | {name[:25]} | 条件={cond} | top1={top1} | 採用={'OK' if cls else 'SKIP'}\")
"
```

**判定**:
- "1勝" 含む 12R → 採用、各 700円
- それ以外 → 除外

---

## 5. 5/9 09:30-15:30 — PAT 投票

各採用 R で:
```
新潟 12R (1勝クラス想定):
  軸 = V15 top1 (predictions CSV の top1_num)
  買い目 = trio_bets 7点 (CSV の trio_bets 列、カンマ区切り)
  投資額 = 700円
  券種 = 三連複 (流し/フォーメーション どちらも可)

(同様に 東京 12R, 京都 12R が 1勝クラスなら)
```

**チェックリスト**: `data/results/20260509_pat_checklist.md` 参照。

**実投票時刻 目安**:
| 場 | 12R 発走 (経験則) | 投票締切 |
|----|------------------:|---------:|
| 京都 | 16:00 頃 | 15:55 |
| 東京 | 16:15 頃 | 16:10 |
| 新潟 | 15:30 頃 | 15:25 |

→ 15:00 前に全 R 投票完了が安全。

---

## 6. 5/9 18:00 — DailyResults_Sat 自動発火

| script | 役割 |
|--------|------|
| `tools/daily_results.py` | 当日結果収集 + ROI 計算 + cumulative 更新 |
| Discord 通知 | #updates |

**確認**: 18:30 に結果通知来ること。

---

## 7. 5/9 20:00 — 当日収支確認 + 5/10 判断

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/cumulative_results.csv', dtype={'date':str})
df_5_9 = df[df['date'] == '20260509']
if len(df_5_9):
    print('5/9 投資:', int(df_5_9['inv'].sum()))
    print('5/9 払戻:', int(df_5_9['pay'].sum()))
    roi = df_5_9['pay'].sum() / max(df_5_9['inv'].sum(), 1) * 100
    print(f'5/9 ROI: {roi:.1f}%')
"
```

**5/9 ROI による 5/10 判断**:

| 5/9 ROI | 5/10 アクション |
|--------:|----------------|
| ≥ 100% | 同戦略 継続 |
| 50-99% | 警戒、控えめ運用 |
| < 50% | 5/10 投資停止 |
| 0% (全外し) | 即停止、5/10 ゼロ |

詳細: `data/v18/risk_management_5_9.md`

---

## 8. 5/9 異常時 トラブルシューティング

| 症状 | 原因 | 対応 |
|------|------|------|
| daily_predict 完了せず | watchdog 監視失敗 | `python tools/daily_predict.py --date 20260509` 手動実行 |
| race_name 表示せず | scrape 失敗 | netkeiba.com で直接確認 |
| top1_num 数字不正 | feature 欠損 | 該当 R 除外 |
| Cookie 切れ通知 | premium scrape 失敗 | `python tools/refresh_cookie.py --auto` |
| Discord 通知来ない | webhook 切れ | `.env` の DISCORD_WEBHOOK_URL 確認 |
| daily_predictions/20260509.csv が空 | DailyPredict 自体が動いてない | schtasks 確認 → 手動実行 |

---

## 9. 重要ルール (5/9 当日 違反禁止)

| ルール | 理由 |
|--------|------|
| **11R 完全除外** (新潟駿風S/東京エプソムC/京都京都新聞杯) | 重賞 + 距離不適合 |
| **12R 1勝クラスのみ採用** | 4日 retro で唯一統計的にプラス確証 |
| **1R 700円固定** | 案B改 上限、増額禁止 |
| **採用 R 数 上限 3** | 3場 × 12R |
| **TYB midday script 実行しない** | 5/3 で 404 確認、Phase 2.5 観測中 |
| **v18/v19 投入しない** | 5/16 以降 |
| **NAR 投入しない** | 5/12 paper 開始、5/16 試行 |

---

## 10. 5/9 終了後

| step | 担当 | 内容 |
|------|------|------|
| 18:00 | 自動 | DailyResults_Sat → cumulative 更新 |
| 20:00 | 自動 | DailyResultsEvening (二重) |
| 20:30 | 自分 | 当日収支確認 + 翌日判断 |
| 21:00 | 自分 | `data/v18/post_5_9_improvement_template.md` を埋めて 5/16 改善材料化 |
| 23:00 | 自動 | NightlySanity (翌日 task pre-check) |

---

## 11. 関連 doc

- 採用方針: `data/results/20260509_final_plan_v2.md`
- 開催情報: `data/results/20260509_pre_check.md` / `20260509_race_card.md`
- リスク管理: `data/v18/risk_management_5_9.md`
- PAT chk: `data/results/20260509_pat_checklist.md`
- 振り返り: `data/v18/post_5_9_improvement_template.md`
