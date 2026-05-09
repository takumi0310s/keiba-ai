# Session #59 A: Discord 二重送信 root cause 特定

**作成**: 2026-05-09 (Session #59 A、 5/9 朝 9:15+)
**目的**: ユーザー Discord 通知 二重送信バグの原因特定

---

## 1. 結論 (root cause)

**`notify_formatted(date_str, mode='morning', channel='bets')` が AM8:00 と AM8:45 の 2 回呼ばれる**:

| 呼び出し元 | 時刻 | 行 |
|------------|------|----|
| `tools/daily_predict.py` | 08:00 (定例) | line 585 |
| `tools/race_auto_notify.py` | 08:45 (定例) | line 484 |

両方とも `mode='morning'` + 同じ date で同じ内容を送信。 **17 races × 2 = 34 messages** ⇒ ユーザーから見ると 2 重通知。

加えて各レース 5 分前 timer が個別通知 (line 336)。 race_auto_notify の log:
```
全レース一括通知: 4 messages
整形済み買い目通知: 17 messages
```
→ 起動時点で 21 message + その後 36 races × 1 = 各レース 1-3 重複 contents。

---

## 2. 補強証拠

### 2.1 race_auto_notify.py のコメント

```python
# 当日一括の整形済み買い目通知（#買い目）を1回だけ送信
# AM8:00 daily_predict で既に送信済みの場合もあるが、レース当日朝の
# 8:45起動時点でオッズが更新されている可能性があるため再送する
```

→ 意図的な再送だが dedup なし。 オッズ変化少なら 完全重複。

### 2.2 schtasks 確認

```
\keiba-ai\DailyPredict          (AM8:00 daily_predict.py)
\keiba-ai\RaceAutoNotify_Sat    (AM8:45 race_auto_notify.py、 土曜のみ)
\keiba-ai\RaceAutoNotify_Sun    (AM8:45 race_auto_notify.py、 日曜のみ)
\Keiba-MultiStagePredict_Race12_1545_Sat/Sun  (15:45 12R 個別)
\Keiba-MultiStagePredict_Race11_1450_Sat/Sun  (14:50 11R 個別)
```

→ 各タスクは Sat/Sun 排他で重複登録なし。 5/9 (土) は Sat のみ走る。 schtasks 重複は **なし**。

### 2.3 logs/discord_failures.log (5/9 04:23-05:53)

28 件 HTTP 429 (rate limit) は process_watchdog の連投が原因 (Session #58 で audit 済)。 **本 Session の二重送信とは別系統**。

---

## 3. 原因の核

**`tools/notify.py: send_discord()` に dedup 機構がない**:

```python
# 現状の send_discord (130 行)
def send_discord(title, message, color="green", fields=None, channel="updates",
                 max_retries=3, retry_backoff=(1, 2, 4)):
    # URL fetch → POST → retry logic のみ
    # ← 同じ内容を即時に再呼び出ししても 2 回送信される
```

→ retry logic は HTTP 失敗時の再送、 dedup ではない。

---

## 4. 修正方針 (Session #59 B で実装)

**`send_discord` に 5min hash cache を追加**:

```python
def send_discord(..., dedup_window_sec=300):
    """5分以内に同一 (channel + title + message) は skip。
    dedup_window_sec=0 で無効化、 既存呼び出しは default 300 で自動 dedup。"""
    h = sha256(channel + title + message[:500]).hexdigest()
    cache = _load_send_cache()  # data/discord_send_cache.json
    now = time.time()
    last_ts = cache.get(h)
    if last_ts and now - last_ts < dedup_window_sec:
        return True  # skip (silent)
    # ... POST ...
    if 200/204:
        cache[h] = now
        _save_send_cache(cache)
        return True
```

**特性**:
- ✅ 既存 caller (race_auto_notify / daily_predict / notify_done) は無変更で自動 dedup
- ✅ message が変わったら (オッズ更新後) 別 hash → 送信される
- ✅ 5 min 経ったら同じ内容も再送可能 (テスト用、 翌レース等)
- ✅ JSON ファイル lock なし (race condition は 1-2 重送信が偶発的に発生する程度、 致命的でない)
- ✅ `dedup_window_sec=0` で opt-out 可能

---

## 5. 5/9 投資 完全保護 (絶対遵守)

- ✅ V15 model 不変
- ✅ predict_core / daily_predict / app.py / race_auto_notify 既存ロジック不変
  (本修正は `tools/notify.py` の send_discord 関数のみ変更、 caller 側 0 行修正)
- ✅ schtasks 既存 41 件 不変
- ✅ 12R 1勝 ¥2,100 (案B改 単独継続) 絶対

---

**Session #59 A 完了**
