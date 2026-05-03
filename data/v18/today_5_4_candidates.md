# 5/4 (月) やれること候補リスト

生成: 2026-05-04 08:00 (Opus xhigh, Session#7)

## 状況

- 月曜 06:00 以降 → SCRAPER-GUARD 解除中 ✓
- レースなし (5/4-5/8)
- ユーザー GW 中、無理せず開発のみ
- 累計 +14,140円 (撤退ライン余裕大)

## 候補 (優先度順)

### 🔴 #1 .gitattributes 設定 (5min, リスクなし) ✅ 本セッションで実施

```bash
cat > .gitattributes <<EOF
data/v17/models/*.txt -text
data/v18/models/*.txt -text
data/_model_bak_*/*.txt -text
EOF
```

→ CRLF 再発防止。今後 model file が CRLF 化することを防ぐ。
   **本セッションで実施済**。

### 🔴 #2 DailyPredict watchdog 化 (5min, ユーザー手動) ⚠️ admin 必要

```powershell
# 管理者として PowerShell
$action = New-ScheduledTaskAction -Execute "C:\Users\takum\keiba-ai\daily_predict_watchdog.bat"
Set-ScheduledTask -TaskName "DailyPredict" -TaskPath "\keiba-ai\" -Action $action
```

詳細: `data/v18/daily_predict_watchdog_migration.md` (Session#4 作成)

### 🟠 #3 netkeiba_race_analysis 再起動 (30min, 自動可)

```bash
# 32日 stale → V17 features ra_score 全0 の主因
python tools/scrape_data_analysis.py --recent  # オプション要確認
# 出力: data/netkeiba_race_analysis.csv
```

### 🟠 #4 netkeiba_stable_comments 再起動 (30min, 自動可)

```bash
python tools/scrape_comments_bulk.py --recent
# 出力: data/netkeiba_stable_comments.csv
```

### 🟠 #5 jra_payouts 5/2-5/3 取得 (10min, 自動可)

```bash
python scrape_jra_payouts.py
# 5/2, 5/3 の JRA 公式配当データ追加
```

### 🟠 #6 netkeiba_speed_index 再起動 (1h, 自動可)

```bash
python tools/scrape_speed_index.py
# prev_index_* features の元データ更新
```

### 🟡 #7 TYB publish タイミング 観測スクリプト作成 (1h)

```python
# tools/check_tyb_publish_time.py を新規作成
# 5/4-5/10 各日 06:00, 09:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00 で
# JRDB Tyb 取得試行 → publish 時刻分布を統計
```

→ 実装し、cron/タスクスケジューラ登録。5/10 までに 7日分データ蓄積で
   midday 戦略の生死判定可能。

### 🟡 #8 netkeiba_ai_position 再起動 (30min)

```bash
python tools/scrape_ai_position.py  # スクリプト要確認
```

### 🟡 #9 netkeiba_siblings 再起動 (1h)

```bash
python tools/scrape_siblings.py
```

### 🟡 #10 netkeiba_master_index 再起動 (1h)

```bash
python tools/scrape_master_index.py
```

### 🟢 #11 古いモデル削除 (30min)

```bash
# CRLF backup 含めて削除可能
rm data/_model_bak_20260503/*.bak_crlf
# 古い v9-v141 model
rm keiba_model_v9_central*.pkl.gz
rm keiba_model_v12_central*.pkl.gz
rm keiba_model_v13[0-9]_central*.pkl.gz  # v131-v141
# data/v17/models/v17_leakfree_*, v17_lgb_fold5.txt
```

→ 約 130MB 削減

## 5/4 推奨実行順

```
07:55 (現在)
  ↓
A. .gitattributes 設定                 (5min)  ✅ 本セッション完了
B. jra_payouts 5/2-5/3 取得 起動       (10min) ← 即時可
C. netkeiba_race_analysis 起動         (30min) ← B 並列可
D. netkeiba_stable_comments 起動       (30min) ← B/C と並列可
E. TYB publish 観測スクリプト作成       (1h)    ← B/C/D と並列可

10:00頃 完了
  ↓
F. (ユーザー) DailyPredict watchdog 化  (5min)  ← admin 必要、ユーザー手動

午後 (任意):
G. netkeiba_speed_index 再起動         (1h)
H. netkeiba_ai_position 再起動          (30min)
I. netkeiba_siblings 再起動             (1h)
```

## 本セッション (Session#7) で実施可能な範囲

- ✅ A. .gitattributes 設定 (実施済)
- ⏳ B. jra_payouts 5/2-5/3 取得 (background 起動)
- ⚠️ C/D. netkeiba scrape は時間長く、別 session で実行推奨

## 撤退ライン (再確認)

5/4-5/8 は投資なし → 累計変動なし (+14,140円維持)
5/9-5/10 想定最悪 -4,200円 → 累計 +9,940円 (撤退ライン -50,000 まで余裕大)

## TL;DR

🟢 **本日 (5/4) の最優先**:
1. .gitattributes 設定 (5min) ✅ 完了
2. DailyPredict watchdog 化 ⚠️ ユーザー手動
3. netkeiba premium 再起動 (race_analysis, stable_comments)

5/4 中に 3-4時間で重要タスク完了見込み。GW中にゆっくり Phase 2.5 進行。
