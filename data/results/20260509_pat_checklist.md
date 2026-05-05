# 5/9 (土) PAT 投票 チェックリスト

**目的**: 連続作業 50 時間後でも判断ミスしない人為ミス排除リスト

採用方針: V15 案B改 / 12R 1勝クラスのみ / 1R 700 円 / 上限 2,100 円

---

## A. 投票前 チェック (必ず順序通り)

### A.1 環境確認
- [ ] PC 起動確認 (.env 設定読み込まれている)
- [ ] daily_predictions/20260509.csv 存在確認 (`ls data/daily_predictions/20260509.csv`)
- [ ] CSV 行数 35 〜 45 (3場 × 12R 想定)
- [ ] morning Discord 通知 受信済 (#bets / #updates)
- [ ] DailyPredict (08:00 task) 完了通知 受信済

### A.2 採用 R 抽出
```bash
cd C:\Users\takum\keiba-ai
python -c "
import pandas as pd
df = pd.read_csv('data/daily_predictions/20260509.csv', dtype={'race_id':str})
df12 = df[df['race_num'].astype(int) == 12]
print('=== 12R 一覧 ===')
for _, r in df12.iterrows():
    name = str(r.get('race_name',''))
    is1 = '1勝' in name
    cond = r.get('condition','')
    course = r.get('course','')
    top1 = r.get('top1_num','')
    score = r.get('top1_score', r.get('score',''))
    bets = r.get('trio_bets', r.get('bets',''))
    flag = '採用' if is1 else 'SKIP'
    print(f'  [{flag}] {course} 12R | {name[:30]} | 条件={cond} | top1={top1} | trio={bets}')
"
```

- [ ] 12R 件数 = 3 (新潟/東京/京都)
- [ ] "1勝" を含む 12R を全て採用 R として list 化
- [ ] 各採用 R の top1 馬番 メモ
- [ ] 各採用 R の trio_bets 7 点 メモ
- [ ] 採用 R 数: __ R (0, 1, 2, 3 のいずれか)
- [ ] 投資総額: __ 円 (採用R × 700)

### A.3 11R は絶対除外
- [ ] **新潟 11R 駿風 S 芝1000m**: 距離不適合 → 除外 確認
- [ ] **東京 11R エプソムC G3**: 重賞 → 除外 確認
- [ ] **京都 11R 京都新聞杯 G2**: 重賞 → 除外 確認

→ **11R 投票しない**

### A.4 累計予算確認
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/cumulative_results.csv', dtype={'date':str})
print(f'累計収支: {int(df[\"pay\"].sum() - df[\"inv\"].sum()):+d} 円')
"
```
- [ ] 累計が +14,140 円 付近か確認 (4/12 〜)
- [ ] 撤退ライン (-50,000 円) まで余裕あり (60,000+ 円) 確認
- [ ] 当日投資 + 累計 が ライン超えない (例: 2,100 円投資後でも +12,000 円以上)

### A.5 重要原則 確認
- [ ] **TYB midday script は実行しない** (5/3 の 404 確認後 廃止)
- [ ] **v18/v19 は投入しない** (5/16 以降)
- [ ] **NAR は投入しない** (5/12 paper 開始)
- [ ] **増額禁止** (1R 700 円固定)

---

## B. 投票実施 (各採用 R で繰り返し)

### B.1 PAT ログイン
- [ ] PAT サイト ログイン済
- [ ] 投票締切 確認 (各 R 発走 1 分前)
- [ ] 残高 が投資総額 ≥ 2,100 円 確認

### B.2 R ごとの投票
**Race 1**: __ 場 12R (例: 新潟 12R)
- [ ] 場 + R 番号 確認
- [ ] race_name に "1勝" 含む 確認
- [ ] **券種: 三連複**
- [ ] **方式: フォーメーション or 7点指定**
- [ ] 軸 (1列目): top1 = __ 番
- [ ] 2列目: top2 = __ 番, top3 = __ 番
- [ ] 3列目: top2-top6 = __, __, __, __, __ 番 (5 通り)
- [ ] **金額: 100円 × 7点 = 700 円**
- [ ] 確認画面で 7 点 全て一致 → 投票確定

**Race 2** (もし 1勝): 同上
**Race 3** (もし 1勝): 同上

### B.3 投票確認
- [ ] PAT 履歴ページ で 採用 R 数 × 700 円 = 投資総額 一致
- [ ] 重複投票なし (同一 R に複数投票していない)
- [ ] 全採用 R で投票完了

---

## C. 投票後 確認

### C.1 投票内容と CSV 整合性
```bash
# data/daily_predictions/20260509.csv の trio_bets と PAT 履歴を 7点ずつ照合
python -c "
import pandas as pd
df = pd.read_csv('data/daily_predictions/20260509.csv', dtype={'race_id':str})
df12 = df[df['race_num'].astype(int) == 12]
for _, r in df12.iterrows():
    if '1勝' in str(r.get('race_name','')):
        print(f\"\n{r.get('course','')} 12R: {r.get('race_name','')}\")
        print(f'  軸 top1: {r.get(\"top1_num\")}')
        print(f'  trio bets: {r.get(\"trio_bets\")}')
"
```
- [ ] CSV の trio_bets と PAT 履歴 一致 (各 R)

### C.2 ログ記録
- [ ] (任意) 手元メモに 採用 R + 軸馬 + 投票時刻 + 投票額 記録

### C.3 結果待ち
- [ ] 18:00 DailyResults_Sat 自動発火 待ち
- [ ] 18:30 Discord #updates の結果通知 確認
- [ ] 5/9 ROI 確認 → 5/10 判断 (data/v18/risk_management_5_9.md)

---

## D. 中止/異常時の判断

### D.1 採用 R で異常検知
| 症状 | 対応 |
|------|------|
| 1勝クラス 0 R | **5/9 無投資**、累計 +14,140 円 維持 |
| daily_predictions/20260509.csv 不正 (空 or < 30 行) | watchdog ログ確認、手動 daily_predict 再実行 → 完了後 投票 |
| race_name 文字化け / 空 | netkeiba 直接 確認、shutuba.html で 1勝かどうか目視 |
| top1_num が NaN / 空 | 該当 R 除外 |

### D.2 PAT 異常時
| 症状 | 対応 |
|------|------|
| ログイン不可 | **当日無投資**、5/9 残額 0 |
| 投票確認画面で 7 点不一致 | 投票せず、CSV 再確認 → 修正後 投票 |
| 残高不足 | 投資 R 数 削減 (1-2 R) |
| 締切過ぎ | 該当 R 除外 |

---

## E. 5/9 終了後

- [ ] DailyResults_Sat 結果確認
- [ ] data/cumulative_results.csv の 5/9 行 追加確認
- [ ] data/v18/post_5_9_improvement_template.md を埋めて 5/16 の改善材料化
- [ ] 5/10 (日) 投資判断 (risk_management_5_9.md の表に従う)

---

## 重要原則 (再掲)

🔴 **絶対禁止**:
- 11R 投票
- 1R 700 円超え
- 1日 2,100 円超え
- TYB midday / v18/v19 / NAR 投入
- 累計 -50,000 円超え

🟢 **絶対遵守**:
- 12R 1勝クラスのみ
- 三連複 7 点 (CSV の trio_bets そのまま)
- 投票前チェックリスト全完了

---

📎 関連: `data/results/20260509_operation_guide.md`, `data/v18/risk_management_5_9.md`, `data/results/20260509_final_plan_v2.md`
