# 5/9 本番直前 リスク監視 + 想定外シナリオ 監査

**作成**: 2026-05-06 PM (Session #29)
**ベース commit**: c722d403
**ユーザー方針**: 取り返し禁止 / 累計死守 / 5/9 案B改 V15 維持 / 寝ても起きても一読把握

---

## 1. 想定外シナリオ 8 種 + 対策

| # | シナリオ | 確率 | 影響 | 既存実装 | 対策 |
|---|---------|------|------|---------|------|
| ① | Cookie 切れ | **中** | 中 | 部分対応 | refresh_cookie.py `--auto` あり (TEST_URL 認証ヘッダで判定)。.env mtime=5/3 10:51, NETKEIBA_COOKIE 行 1836 char (期待 1817 と微差、健全圏内)。**fail 時 Discord 通知なし → 09:30 MorningWeightCheck で fetch 失敗してから初めて検知** |
| ② | JRDB 9:00 retry 失敗 | 中 | **大** | あり | `jrdb_retry_am9.bat` で TYB/SED/KYI/KAB を `--force` で 1 回 retry → `jrdb_health_check.py --silent` で Discord。**retry 1 回のみ、404 継続なら諦め**。SED 不在で前走成績が直撃 → top1 score 信頼性低下。回避策: 5/9 朝 09:30 後 #updates で SED 取得 % 確認 |
| ③ | 馬体重 10:00 取得失敗 (一部開催場) | 中 | 小 | あり | multi_stage_predict.py L446-453: predict_one_race が None 返した場合 **朝予測 (480kg デフォルト) を使用継続**。残レース予測継続。fallback は CSV 上 `予測失敗 (朝予測使用)` と記録 → 朝予測 = 投票判断 base、被害最小 |
| ④ | 14:50 / 15:45 Discord 通知失敗 | **中** | **大** | **なし** | notify.py L84 silent fail (HANDOFF L389 既知)、retry なし、ログなし、戻り値 False のみ。**14:50 / 15:45 が無音だった瞬間に手詰まり**。回避: 投票前に必ず Discord 受信目視 → 来てなければ手動で `python tools/multi_stage_predict.py --stage race12_1545 --date 20260509` 再実行 |
| ⑤ | PAT 投票 入力ミス | 中 | 中 | あり | data/results/20260509_pat_checklist.md B.2 に券種=三連複, 軸 (1列目)=top1, 2列目=top2/top3, 3列目=top2-top6 の 5 通り、合計 7 点 100 円 = 700 円。**馬番昇順、CSV の trio_bets そのまま**。3 重チェック (R 数/総額/確認画面 7 点一致) |
| ⑥ | PAT サーバー障害 / オッズ更新終了 | 低 | 中 | 部分 | チェックリスト D.2: ログイン不可なら **当日無投資**。締切過ぎ R は除外。**5/9 オッズ最終更新 14:55 想定 → 15:45 多段予測後の投票余裕 25 分**、当日締切ギリギリで PAT 障害なら諦め |
| ⑦ | ProcessWatchdog v2 誤発火 | 低 | 中 | 安全機構あり | process_watchdog_v2.py L80-84: `is_active_hours` で **07:00-18:00 のみ再起動**。L146: 範囲外なら restart skip。プロセス検知は L122 PowerShell CIM (wmic 24H2 廃止対応済)。**alive な daily_predict を kill するリスクは低い (CommandLine 部分一致)**。HANDOFF L389 平日 fatal alert は 5/6 既知バグ (`daily_predict_watchdog_wrapper_20260506.log:24`) → 開催日 5/9 は無関係 |
| ⑧ | PC 落雷・停電 → 自動復旧 | 低 | **大** | 部分 | schtasks は OS 起動後 自動再開、Trigger 既存維持。**進行中タスクは強制終了 → resume なし** (daily_predict は `--resume` 引数あり、watchdog v2 が再起動時に付与 L64)。13:00 以降 PC 停電 → 投票見送り。06:30-12:00 停電 → 復旧後 手動 `python tools/daily_predict.py --date 20260509` |

---

## 2. 累計収支 final 確認

### 2.1 数値整合性

| ソース | 値 (累計収支 5/5 終時点) | 備考 |
|--------|------------------------:|------|
| **生データ** (`data/cumulative_results.csv` 集計) | **+13,530 円** ※dashboard data.json L18 と一致 | 4/12-5/3 settled 495R + 5/5 NAR +310 |
| **USER 申告** | **+14,140 円** | 5/3 13,830 + 柏記念 310 = 14,140 と表記 (PAT 履歴ベース) |
| **dashboard/data.json** | +13,530 円 | "5/9 投資前 (撤退余裕 +63,530)" |
| **PAT checklist L62** | **+14,140 円** | 「累計が +14,140 円付近か確認」 |
| **risk_management_5_9.md L3, L34, L37** | **+14,140 円** | "5/9 最悪 -2,100 → +12,040" 表記 |

### 2.2 cumulative_results.csv 実状

- 末尾 5 行 全て `20260503` の京都 8R-12R settled (再確認済)
- **5/5 NAR 柏記念 +310 は cumulative_results.csv に未反映** (race_id 202643050511 は data/daily_results/20260505.csv にのみ存在)
- dashboard data.json はその 5/5 の +310 を加味済 (L19) → 13,220 + 310 = **13,530**

### 2.3 USER 申告 +14,140 と生データ +13,530 の差 610 円

- 5/2-5/3 期間の集計差 (PAT 履歴 vs cumulative_results.csv) と推定
- どちらも 撤退ライン -50,000 から大幅余裕、5/9 撤退判定には影響なし

### 2.4 撤退基準計算 (各値で)

| 基準値 | 5/9 最悪 -2,100 後 | 5/9-5/10 最悪 -4,200 後 | 撤退ライン -50,000 までの余裕 |
|-------|--------------------:|------------------------:|------------------------------:|
| +13,530 (生) | +11,430 | +9,330 | **63,530** |
| +14,140 (USER) | +12,040 | +9,940 | **64,140** |

→ **5/9 撤退判定は「+13,530 (生データ)」を基準採用推奨**。理由: dashboard と data.json で確定済、PAT 履歴は手動記録で誤差混入リスクあり。差 610 円は撤退に無関係。

### 2.5 dashboard 整合性

- `data/dashboard/data.json` L23-29: withdraw line=-50000, current=13530, margin=63530, ratio=0.864 → 整合
- 文字化け: data.json は ASCII + JP UTF-8、修正完了済
- **要対処**: 5/5 NAR +310 行を `data/cumulative_results.csv` に追加 (race_id 202643050511)

---

## 3. 5/8 (金) 22:00 dry-run チェックリスト

```bash
cd C:\Users\takum\keiba-ai

# Step 0. 前提確認 (1 分)
python tools/refresh_cookie.py --check
ls data/daily_predictions/20260509.csv  # 5/8 朝発火後なら存在
git log --oneline -1                     # commit c722d403 以降確認

# Step 1. dry-run 3 stages (5/3 データ retro、各 30 秒)
python tools/multi_stage_predict.py --stage test10       --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race11_1450  --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race12_1545  --date 20260503 --dry-run

# Step 2. 期待結果
#  - test10: "対象なし or 京都 2R" + "3R-12R 朝予測 (10R 参考)" + Discord [SKIP] dry-run
#  - race11_1450: "全 3場 11R 予測 (重賞含む、採用 0/3)" + body 表示
#  - race12_1545: "全 12R 予測、採用 1 (新潟 12R 1勝クラス)" + trio_bets 7 点 + body
#  - 各 CSV: data/multi_stage_predict/20260503_{stage}.csv 生成

# Step 3. 失敗時 rollback
#  - Cookie 切れ: python tools/refresh_cookie.py --auto
#  - predict_one_race 例外: logs/multi_stage_*_20260503.log で stack trace、SED 不在なら諦めて朝予測のみ
#  - CSV 生成失敗: 権限確認 + dir test
#  - 全失敗時: bat ファイル単独実行 (tools/multi_stage_predict_test10.bat)

# Step 4. 結果記録
echo "[5/8 22:00 dry-run]" >> data/v18/dryrun_5_9_full.md
git status -s
```

---

## 4. 5/9 朝の起動チェックリスト (5 分で復帰)

| # | 確認項目 | コマンド | 期待結果 |
|---|---------|---------|---------|
| 1 | Discord 通知 受信 | スマホ確認 | 06:30 Morning_Sat / 08:00 DailyPredict / 08:50 AM8FireCheck の 3 件以上 |
| 2 | 朝予測 CSV 存在 | `ls data/daily_predictions/20260509.csv` | 35-45 行、3場×12R |
| 3 | Cookie 健全 | `python tools/refresh_cookie.py --check` | `[OK] Premium認証OK` |
| 4 | JRDB 9:00 retry 完了 | `ls -la logs/jrdb_retry_am9_20260509.log` | size > 0、Discord #updates 通知あり |
| 5 | 12R 1勝クラス 確認 | PAT checklist A.2 のスクリプト実行 | 0-3R 採用判定確定 |

**Discord 通知タイミング表 (5/9 (土))**

| 時刻 | 通知 | 不在時の判断 |
|------|------|-------------|
| 06:30 | #bets Morning_Sat 軸候補 | 寝てて見落としは OK、08:00 で再確認 |
| 07:00 | #updates MorningDigest | (任意) |
| 07:30 | #updates JrdbHealthCheck_Sat | 不在 → JRDB 取得異常、9:00 retry 期待 |
| 08:00 | #bets DailyPredict 完了 | 不在 → 重大、手動 `python tools/daily_predict.py --date 20260509` |
| 08:50 | #updates AM8FireCheck OK | 不在 → 朝の発火失敗、6 項目復帰 必要 |
| 09:00 | #updates JrdbRetryAm9 結果 | 不在 → SED/TYB 取得 % 不明、影響限定的 |
| 09:30 | #updates MorningWeightCheck | 不在 → 馬体重補正失敗、朝予測でそのまま投票 (許容) |
| 10:00 | #updates Test10 多段予測 | 不在 → multi_stage_predict 異常、14:50 まで様子見 |
| **14:50** | **#updates Race11_1450** | **不在 → 手動再実行 必須** |
| **15:45** | **#updates Race12_1545** | **不在 → 投票判断不可 → 手動再実行**、(2 回失敗時) 朝予測 CSV から 12R 1勝のみ手動投票 |
| 18:00 | #updates DailyResults_Sat | 不在 → 結果未照合、20:00 再試行待ち |

---

## 5. 5/9 投資 NEVER list (絶対遵守)

🔴 **絶対禁止**:
1. **11R 投票禁止**: 新潟 11R 駿風S 芝1000m (距離不適合)、東京 11R エプソムC G3、京都 11R 京都新聞杯 G2 (重賞) の 3 R 全除外
2. **1R 700 円超え禁止**: 三連複 7 点 × 100 円 固定
3. **1日 2,100 円超え禁止**: 採用 R 上限 3 (3場 12R が全て 1勝クラスの場合のみ最大)
4. **V18/V19 投入禁止**: 5/16 以降の試行枠、5/9 は V15 案B改 のみ
5. **NAR 投入禁止**: 5/12 paper 開始、JRA と完全分離
6. **累計 -50,000 円超え禁止**: 余裕 63,530 円、5/9 全外しでも到達しない
7. **増額禁止**: case-by-case 増減なし、Phase 2.5 完了 (5/24) まで固定
8. **TYB midday script 実行禁止**: 5/3 で 404 確認後 廃止済、再実行しない

🟢 **絶対遵守**:
- 12R 1勝クラスのみ (race_name に "1勝" 含む)
- 三連複 7 点フォーメーション (CSV trio_bets そのまま)
- 投票前 PAT checklist A 全完了
- 馬番昇順、カンマ間スペース区切り

---

## 6. 結論

### 6.1 想定外シナリオで未対策のもの

| # | 未対策 | 5/8 までに対応 |
|---|-------|---------------|
| **④** | Discord 通知失敗時の retry 機構ゼロ (notify.py L84 silent fail) | **未対応**。代替: 14:50 / 15:45 は通知不在を 5 分で気づき手動再実行 |
| **②** | JRDB 9:00 retry 1 回のみ。失敗時は SED/TYB 諦め | **未対応**。回避策: 朝予測ベースで投票判断、SED 不在は v15 で前走成績既存ファイル fallback |
| **⑧** | 進行中タスク resume なし (daily_predict のみ resume 対応) | **未対応**。停電は受容リスク、当日諦め判断 |
| **⑤ A.5** | 5/5 NAR +310 が cumulative_results.csv に未反映 | **要対処**。本日中に 1 行追加 (整合性のみ、撤退判定影響なし) |

### 6.2 5/8 22:00 dry-run で確認すべき最重要 3 件

1. **multi_stage_predict 3 stages の CSV + Discord body 想定通り出力** (5/3 retro で test10/race11/race12 全成功確認、5/9 はそのまま動作見込み)
2. **Cookie + DISCORD_WEBHOOK_URL/BETS/UPDATES 全 set 確認** (.env 既に 5/3 10:51 mtime、commit f408d93d 後変更なし)
3. **5/9 12R race_name 取得テスト** (新潟/東京/京都 12R に "1勝" が含まれるか目視) — 21:00 後 シャドー実行可能

### 6.3 5/9 投資 GO / no-go 判定基準

| 条件 | GO | no-go |
|------|----|-------|
| 朝予測 CSV 行数 | ≥ 30 | < 30 → 5/9 無投資 |
| Cookie 健全 | OK | 切れ → refresh_cookie auto 試行、失敗で無投資 |
| 12R 1勝クラス R 数 | 1+ | 0 → 5/9 無投資、累計 +13,530 維持 |
| Discord 受信 (08:00 / 14:50 / 15:45) | 全件 | 14:50 or 15:45 不在 → 手動再実行 1 回 → それでも無音なら無投資 |
| 累計 残高 (5/9 朝時点) | > -47,900 円 | ≤ -47,900 円 → 投資 してすぐ -50,000 抵触の可能性、無投資 |
| Test10 (10:00) Discord 通知 | あり | なし → 14:50 まで様子見、機構警戒 |

→ **6 条件 全 PASS** で GO。1 つでも no-go なら **5/9 無投資、累計 +13,530 死守**。

---

## 7. 寝起き 1 行サマリー

5/9 朝起きたら:
1. **Discord で 08:00 DailyPredict 完了通知 受信** 確認
2. **`data/results/20260509_pat_checklist.md` 開く** → A.1-A.5 順に進める
3. 12R 1勝クラス 0 件なら **5/9 無投資**、1-3 件なら 14:50/15:45 多段予測待ち → trio_bets そのまま PAT 投票
4. 14:50/15:45 の Discord が来なかったら `python tools/multi_stage_predict.py --stage race12_1545 --date 20260509` 手動実行
5. **絶対に 11R 投票しない** (重賞 3 件 + 距離不適合 1 件)
6. 18:00 結果通知で 5/9 ROI 確認 → 5/10 判断は risk_management_5_9.md 表
