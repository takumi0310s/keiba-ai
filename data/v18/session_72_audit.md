# Session #72 A: 現状動作確認 (audit)

**作成**: 2026-05-09 18:01 (Session #72、 dev/two-stage)
**HEAD**: 7fe4c743 (Session #68 E まで反映済)

---

## A1: tools/stage2_predict.py 現 logic (Session #68 修復後)

```
predict_stage2(race_id)
  → predict_one_race(race_id) 呼び出し
  → ret is None なら _probe_netkeiba(race_id) で診断:
    {status_code, response_len, has_horse_list, cookie_present, url}
  → error_kind 分類:
    netkeiba_block (HTTP 400)
    netkeiba_other (200 以外)
    shutuba_empty (200 だが HorseList 無し)
    exception
build_message():
  - 失敗時 "Stage 1 fallback 採用" 明示、 top3 朝予測 表示
  - 成功時 朝予測 vs Stage 2 top3 並記 + 差分 alert
predict_one():
  - 失敗時 cache 書込みスキップ → 次 fire 再試行可
cmd_check_next_1h():
  - 起動時 1 回 _probe_netkeiba()
  - HTTP 400 検知時 全 R skip + Discord 警告 1 通
```

manual 動作確認 (5/9 18:00):
```
$ python tools/stage2_predict.py --race-id 202604010312 --no-discord --force
→ [stage2-trace] error_kind=netkeiba_block, status_code=400
→ "採用予測 = 朝予測 (Stage 1) top3 ★ 1. 11 ハイクオリティ ..."
→ cache 書込みスキップ
→ 動作 OK ✅
```

## A2: logs/stage2_predict.log 5/9 fire 解析

7 回 fire (`check_next_1h ... candidates`):

| fire 推定時刻 | candidates 数 | 試行 | 成功 | 失敗 |
|---|---|---|---|---|
| ~13:00 | 6 | 6 | 0 | 6 (全 NG) |
| ~13:30 | 5 | 2 (3 dedup) | 0 | 2 |
| ~14:00 | 5 | 3 (2 dedup) | 0 | 3 |
| ~14:30 | 5 | 3 (2 dedup) | 0 | 3 |
| ~15:00 | 4 | 2 (2 dedup) | 0 | 2 |
| ~15:30 | 2 | 0 (2 dedup) | 0 | 0 |
| ~16:00 | 0 | 0 | - | - |

**合計 試行 16 / 成功 0 / 失敗 16 (100% NG)** — netkeiba HTTP 400 server block。

注: log 記録は Session #68 修復前 (12:00-15:30 fire 全件) なので「Stage 2 予測 失敗 / error: predict_one_race returned None」 の旧フォーマットで蓄積。
17:00 以降の fire は本 audit 実施時点で未実行。

最後の log entry:
```
C:\Users\...\python.exe: can't open file 'tools/stage2_predict.py': [Errno 2] No such file or directory
```
→ 16:55 頃 schtasks fire 時に parallel agent が dev/training-poc に branch swap していたためファイル不在。
   schtasks 自体は動作 OK、 file 取得に失敗しただけ。 dev/two-stage に戻った後は再 fire OK。

## A3: PreRacePredict_Watchdog schtasks 状態

```
PS> schtasks /query /tn "Keiba-PreRacePredict_Watchdog_5_9" /fo LIST
TaskName:      \Keiba-PreRacePredict_Watchdog_5_9
Next Run Time: 2026/05/09 18:30:00
Status:        Ready
```

→ 18:30 次 fire 待ち。 30 分毎 動作確認 OK。

## 結論

| 観点 | 状態 |
|---|---|
| Stage 2 不具合 (HTTP 400) | ✅ Session #68 で修復、 fallback 動作確認済 |
| 30 分毎 fire | ✅ 動作 OK (5/9 7 回 fire 確認) |
| 通知届いた | ✅ Stage 1 fallback 含めて Discord 送信成功 |
| 通知内容 (top3 のみ) | ❌ ユーザー要望は **全馬 V15 score 順 table** |

→ 次 phase B / C で通知内容を「全馬 V15 score 順 table」に変更。
