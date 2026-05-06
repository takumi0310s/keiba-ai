# 未対策 (デジタル系) リスク 対策完了レポート

**作成**: 2026-05-06 PM (Session #31)
**対象 commit**: c722d403 + Session #31 (本セッション 5 commits)

Session #25 自己診断 §B4 + Session #30 リスク監査 §C で指摘された未対策デジタル系リスクの対応。

---

## 1. JRDB retry 強化 (A1)

### 現状 (commit 95495268)
- 06:00 DailyJrdbKyi: 全 type DL
- 09:00 JrdbRetryAm9: TYB/SED/KYI/KAB --force retry (Session #25 で実装)

### 強化 (本 commit)
- **12:00 JrdbRetryPm12 新設**: 09:00 でも失敗時の最終 retry
- 失敗時 (jrdb_health_check 結果 NG) は Discord yellow 通知:
  「JRDB なしで投資判断、案B改 V15 maintain」と明示
- 5/9 までの追加 admin: `register_jrdb_retry_pm12_schtasks.ps1` (累計 5 件目)

### JRDB 3 段階 retry 体制
```
06:00 DailyJrdbKyi  (全 type)
  ↓ 失敗時
09:00 JrdbRetryAm9  (TYB/SED/KYI/KAB --force)
  ↓ 失敗時
12:00 JrdbRetryPm12 (TYB/SED/KYI/KAB --force + 健全性 check + Discord yellow)
  ↓ 失敗時
"JRDB なしで案B改 V15 投資判断"
```

→ 5/9 朝の前走成績取得を 3 段階で確実化。

---

## 2. Discord retry/log (A2)

### 現状 (Session #25 自己診断で指摘)
`tools/notify.py` L84 で `requests.post()` 失敗時 silent fail (return False のみ、log なし)。
障害時の検出が完全に不可能だった。

### 強化 (本 commit)
- `send_discord()` に retry logic 追加:
  - max_retries=3 default
  - exponential backoff (1s / 2s / 4s)
  - 4xx (429 除く) は retry 無意味で break
  - 429 (rate limit) は backoff で retry
- 失敗時 `logs/discord_failures.log` に記録:
  - timestamp / channel / reason (http_status or exception)
  - title / message preview
- url 未設定も log に記録 (silent fail 撲滅)

### 検証
```bash
$ python -c "from tools.notify import send_discord; ok = send_discord('test', '...', channel='updates'); print(ok)"
ok= True   # 成功 → log 不在で正常
```

5/9 朝に Discord 通知障害があれば `logs/discord_failures.log` で原因特定可能。

---

## 3. ProcessWatchdog v2 誤発火対策 + tuning (A3+D 統合)

### 現状 (Session #5 + 本日 admin 登録)
- 5 分間隔で logs mtime 監視
- daily_predict stale 30min / race_auto_notify stale 10min で再起動
- 07:00-18:00 のみ自動再起動

### 懸念 (Session #25/30 で指摘)
- daily_predict は 30min かかるので、ログ更新間隔次第で誤発火可能性
- race_auto_notify は通常 ~10min、閾値ぴったりで誤発火リスク

### 強化 (本 commit)
**閾値緩和 (誤発火防止)**:
- `daily_predict.stale_sec`: 30min → **60min** (実行時間 + 倍 余裕)
- `race_auto_notify.stale_sec`: 10min → **30min** (3 倍 余裕)

**誤発火 log 追加**:
- `_log_misfire()` 関数で再起動 / 時間外スキップを `data/v18/process_watchdog_v2_misfires.log` 記録
- 後の偽陽性判定検証 + 5/9-5/24 paper trading の knowledge

### 検証
```bash
$ python tools/process_watchdog_v2.py --once --dry-run
[watchdog_v2] daily_predict: MISSING (alive=False, stale=False, mtime=...)
[watchdog_v2] race_auto_notify: MISSING (alive=False, stale=False, mtime=...)
```

→ syntax OK、平日 (5/6) なので両方 MISSING 判定 (土曜想定外、これは正常)。

5/9 朝は daily_predict が 08:00-09:00 まで稼働、ログ更新が継続される想定 → stale 判定なし、誤発火しない。

---

## 4. 5/9 朝の安心感 向上

| 項目 | 改善前 | 改善後 (Session #31) |
|------|--------|---------------------|
| JRDB 取得失敗時 | 1 段階 retry のみ | **3 段階 retry + 失敗時 Discord 警告** |
| Discord 障害検出 | silent fail (検出不可) | **3 回 retry + 失敗 log で原因特定** |
| ProcessWatchdog 誤発火 | 30/10 min 閾値で誤発火リスク | **60/30 min に緩和 + 誤発火 log** |

---

## 5. 残課題 (5/16+ で対応)

| 項目 | 内容 | 緊急度 |
|------|------|--------|
| 停電時 resume | daily_predict のみ resume 対応、他は強制終了 | 🟢低 (発生極低) |
| メールフォールバック | Discord 24h 失敗継続時の代替通知 | 🟢低 (Discord 安定運用なら不要) |
| ProcessWatchdog 自分の監視 | 過剰、ログ確認で対応 | 🟢低 |

→ これらは 5/16+ で必要なら対応、5/9 投資判断には影響なし。

---

## 6. 結論

Session #25/30 で指摘されたデジタル系未対策 4 件のうち **3 件対応完了**:
- ① Cookie 切れ → refresh_cookie --auto (既実装)
- ② JRDB retry → **本日 3 段階化** ✅
- ③ 馬体重 fallback → multi_stage_predict L446 (既実装)
- ④ Discord retry/log → **本日 retry+log** ✅
- ⑤ NAR cumulative → **本日 1 行追記** (Session #30) ✅
- ⑥ PAT 障害 → 投票見送り判断 (人手、対策不要)
- ⑦ ProcessWatchdog 誤発火 → **本日 閾値緩和+log** ✅
- ⑧ 停電 resume → 5/16+ で対応 🟢低

5/9 投資準備、安心感 大幅向上。
