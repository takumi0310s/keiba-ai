# 5/17 G1 day (ヴィクトリアマイル) 当日 安全 checklist

> Sub-task 12 deliverable (5/16 evening session)
> 目的: 5/17 (土) G1 day = ヴィクトリアマイル (東京 11R 15:40 発走) を 戦略⑦ 案 C 初日運用で 安全完走させる。

---

## 0. 前提

- 戦略⑦ 案 C 実装済 (Sub-task 8、 5/16 evening commit、 京都/条件 X 除外)
  - commit hash: 5/16 evening の git log で `[Sub-task 8]` 検索 (rollback 時 必要)
- V15 model 不変、 score 計算 不変、 strategy_7 filter 拡張のみ
- 5/16 evening commits 10+ 件 (local のみ、 push は親集中で 5/17 朝以降)
- 累計 真値 +¥5,240 (P0-1 確定、 撤退余裕 +¥55,240)
- 投票上限 ¥2,100/日 (案 B 改 strict)

---

## 1. 5/17 朝 timeline

### 5:30 起床 (早起き、 G1 day)
- [ ] PC 起動確認
- [ ] `git log -10 --oneline` で 5/16 evening commits 確認
- [ ] 戦略⑦ 案 C commit (Sub-task 8) hash 控える (rollback 用)

### 5:30-6:00 dry-run 動作確認
- [ ] `python -m py_compile tools/race_auto_notify.py` (syntax PASS)
- [ ] `python tests/test_features.py` (regression PASS)
- [ ] (もし `--dry-run` mode 実装済なら) `python tools/race_auto_notify.py --dry-run --date 20260517`

### 6:00 DailyJrdbKyi schtask 自動 fire 確認
- [ ] `logs/daily_jrdb_kyi_*.log` 確認 (06:00 fire 想定)
- [ ] `data/jrdb/extracted/Kyi/KYI260517.txt` 確認
  - 注: TYB は 17:00 publish なので朝は KYI のみ

### 8:00 DailyPredict schtask 自動 fire 確認
- [ ] `logs/daily_predict_*.log` 確認 (08:00 fire)
- [ ] `data/daily_predictions/20260517.csv` 確認
  - 行数: ヴィクトリアマイル週 想定 35 R (中央 3 場 × ~12 R)
  - top1_num filled 100%
  - top1_score filled 100%
- [ ] **京都 R が daily_predictions に含まれる** (★ daily_predict は予測する、 投票だけ skip ★)
- [ ] **ヴィクトリアマイル (東京 11R) 予測 sane**
  - top1 horse name 表示
  - top1_score range 0.3-0.7 期待
  - ★ 「予測ゼロ / score=0 / NaN」 が出たら 即 rollback ★

### 9:30 Discord #買い目 通知 受領確認 (★ critical ★)
- [ ] Discord #買い目 channel 確認
- [ ] 通知数: 通常 6-10 R 想定 (戦略⑦ + 案 C 適用後)
- [ ] **京都 R は通知に含まれない** (★ 案 C 動作確認 ★)
- [ ] **条件 X (15 頭+/重~不良) R は通知に含まれない**
- [ ] **東京 11R (ヴィクトリアマイル) 通知に含まれる** (★ G1 投票確保 ★)
- [ ] **中京 R は通常通り通知に含まれる** (案 C で 中京 除外しない)

### 9:30-14:00 投票候補 review
- [ ] 通知された買い目の 各 R で sanity check
- [ ] 買い目 horse 番号 が 出馬表 と一致
- [ ] EV 表示 が sane (1.0-10.0 range)

### 14:00 投票確定
- [ ] 通知された買い目 を IPAT で 手動投票
- [ ] 投票金額: 案 B 改 strict (700円 × 推奨 R 数、 上限 ¥2,100/日)
- [ ] 投票直後 領収書 screenshot or 確認

### 15:40 ヴィクトリアマイル G1 発走
- [ ] レース観戦
- [ ] 結果記録 (メモ可)

### 17:00 結果回収 (手動 trigger or 20:00 自動 待ち)
- [ ] (手動の場合) `python tools/daily_results.py --date 20260517`
- [ ] `data/daily_results/20260517.csv` 確認

### 20:00 DailyResultsEvening schtask 自動 fire
- [ ] `cumulative_results.csv` 自動 update 確認
- [ ] Discord #results 通知受領
- [ ] 当日 ROI / PnL 集計

### 21:00 DailyCumulativeAudit (Sub-task 10 で 5/18 登録予定の場合のみ)
- [ ] (5/18 user 判断後 schtask 登録の場合のみ fire、 5/17 時点 は skip)

### 23:00 Keiba-NightlySanity 自動 fire
- [ ] 翌日 (5/18) タスク事前検証
- [ ] Discord 通知受領

### 24:00 完了報告
- [ ] 当日 結果 (ヴィクトリアマイル + 他 R) を `docs/5_17_RESULT.md` (新規) に記録
- [ ] 戦略⑦ 案 C 初日 動作 verify (京都 skip / X skip / 他通常)
- [ ] 異常なし confirm

---

## 2. ★ 異常 detection 基準 ★

### 即 rollback trigger

| 症状 | 判断 timing | 対応 |
|------|------------|------|
| 8:00 daily_predict 出力 0R or score NaN | 8:30 まで | ★ 即 rollback ★ |
| 9:30 Discord 通知 0 件 | 10:00 まで | ★ 即 rollback ★ |
| ヴィクトリアマイル予測なし | 10:00 まで | ★ 即 rollback ★ |
| 京都以外で大量 skip (3 場以上 skip) | 10:30 まで | rollback 検討 |
| Streamlit dashboard 起動失敗 | 影響なし、 9:30 までに修復 | 投票 影響なし |

### rollback コマンド (1 行)

```powershell
# Sub-task 8 commit hash は 5/16 evening commit log で確認
git log --oneline | Select-Object -First 15  # hash 特定
git revert <Sub-task 8 commit hash> --no-edit
git log --oneline | Select-Object -First 5  # revert 確認
```

### rollback 後の対応

- ヴィクトリアマイル 投票 影響なし (revert で 戦略⑦ 旧 logic に戻る = 京都 含む 通常運用)
- 9:30 Discord 通知 再 fire 待ち (もしくは手動 `python tools/race_auto_notify.py --date 20260517`)
- 通常通り 14:00 投票確定

---

## 3. ★ 5/17 G1 day 投資保護 ★

- 投票上限 ¥2,100/日 (案 B 改 strict)
- 撤退ライン -¥50,000 (真値 +¥5,240 から余裕 +¥55,240)
- 取り返し禁止 (損切り後 翌日へ持ち越さない)
- 戦略⑦ 案 C 初日 = paper trade 並走 (5/18+ 30R 後 採用判定)

---

## 4. ★ 5/17 G1 day 期待値 (assumption) ★

> 以下 数値は paper backtest projection ベース、 当日 実績で update する

- ヴィクトリアマイル: G1、 通常通り 700円 投票
- 全 R: 案 C 適用後 6-10 R 想定 (京都 + 条件 X 除外)
- 投資合計: ¥4,200-7,000 (案 B 改 strict 上限 内)
- 期待 ROI: 案 C で 104-105% (背景 81 race / 月 の paper backtest projection)

---

## 5. ★ trouble shooting ★

### Discord 通知が来ない
1. Discord webhook 環境変数 確認 (`DISCORD_WEBHOOK_BETS`)
2. `logs/race_auto_notify_*.log` 確認
3. ProcessWatchdog 復旧 確認
4. 手動 fire: `python tools/race_auto_notify.py --date 20260517`

### daily_predict 出力 が空
1. `logs/daily_predict_*.log` 確認
2. `data/cookies.json` 期限切れ → `python tools/refresh_cookie.py --auto`
3. JRDB 取得失敗 → daily_jrdb_kyi 失敗 → 6:00 schtask log 確認

### 戦略⑦ 案 C 動作不確認
1. 京都 R が通知される → 案 C 適用されていない → ★ 即 rollback ★
2. ヴィクトリアマイル 通知されない → ★ 即 rollback ★
3. 中京 R が大量 skip → filter 誤判定 疑い → ★ 即 rollback ★

---

## 6. ★ 5/17 完了報告 template ★

`docs/5_17_RESULT.md` (5/17 23:00 までに作成):

```markdown
# 5/17 ヴィクトリアマイル G1 day 結果

## 当日 ROI
- 投資: ¥XXX
- 配当: ¥XXX
- PnL: ±¥XXX

## 戦略⑦ 案 C 動作 verify
- 京都 R skip: ✅ / ⚠
- 条件 X skip: ✅ / ⚠
- 中京 通常: ✅
- ヴィクトリアマイル 通常: ✅

## 異常事象
- なし or 詳細

## 5/18 への引継
- 戦略⑦ 案 C 継続 / rollback
- paper shadow eval 30R 蓄積 status
```

---

## 7. 整合性 note

- timeline は 既存 schtask 設定 と整合 (06:00 / 08:00 / 09:30 / 20:00 / 23:00)
- 期待 ROI / 期待投票数 は paper backtest projection ベース (assumption 明示)
- rollback は `git revert` 1 行で完結 (本番 production code への safe modification 設計)
- 本 doc は 5/17 朝 起床直後 (5:30) に最初に開く運用想定
