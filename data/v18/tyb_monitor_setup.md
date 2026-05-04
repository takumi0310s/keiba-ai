# C. TYB publish タイミング観測 自動化 完了

生成: 2026-05-04 12:25 (Opus xhigh, Session#8)

## 完了内容

### 作成ファイル

| ファイル | 役割 |
|---------|------|
| `tools/tyb_publish_monitor.py` | TYB.lzh HTTP HEAD 試行 + ログ記録 + 初公開時 Discord通知 |
| `tools/tyb_publish_monitor.bat` | schtasks wrapper (今日 or 次の土曜を観測) |
| `data/tyb_publish_log.csv` | 観測ログ (date, jrdb_date, fetch_time, http_status, size, first_publish) |

### スケジュール登録 ✅

```
TaskName: Keiba-TybPublishMonitor
NextRun: 2026/05/04 12:30 (毎時 実行)
Cmd: C:\Users\takum\keiba-ai\tools\tyb_publish_monitor.bat
```

`schtasks /Create /TN "Keiba-TybPublishMonitor" /TR "..." /SC HOURLY /MO 1 /ST 12:30 /F` で登録成功。

### テスト実行 ✅

```
=== TYB publish monitor ===
  date: 20260504 (jrdb=260504)
  HTTP=404 Size=0B (5/4 平日、TYB 公開予定なし → expected)

  date: 20260509 (jrdb=260509)
  HTTP=404 Size=0B (5/9 まだ早い)
```

→ 観測機能は正常動作。Discord 通知は **初めて 200 OK 検出時のみ** 発火。

## 動作仕様

### 観測対象

- 月-金: その日の日付 (JRDB Tyb 通常 publish しないが念のため)
- 土日: その日の race day

### 通知ロジック

```python
# 過去ログを検索
prev_200 = (jrdb_date が過去に 200 OK だったか)
if status == 200 and not prev_200:
    Discord 通知 (#updates green)
```

→ 5/9 朝に publish された瞬間に Discord 通知。これで真の publish 時刻判明。

### ログフォーマット (CSV)

```csv
date_iso,jrdb_date,fetch_time,http_status,size_bytes,first_publish
20260504,260504,12:25:19,404,0,no
20260509,260509,12:25:19,404,0,no
```

## 期待される運用 (5/4-5/10)

```
5/4 (月) 12:30 - 月-金の毎時 fetch (期待: 全 404)
5/5-5/8 同上
5/9 (土) 06:30 - 09:00 - 12:00 - ...
   → どこかで 200 OK に遷移 = TYB publish 時刻 確定
   → Discord 通知 ⏰ TYB 20260509 publish 検出
5/10 (日) 同様
```

## 結果次第での次アクション

| publish 時刻 | midday 戦略の生死 | アクション |
|-------------|------------------|-----------|
| 朝 6:00-9:00 | 朝 morning script に統合可 | midday 廃止、morning に TYB merge |
| 12:00-13:00 | 昼休みに取得可 | midday 早朝化 (12:30実行) |
| 14:00-15:00 | 発走 30-60min前 | midday 維持 (時刻調整) |
| 17:00以降 | post-race 公開のみ | **midday 戦略 廃止** |

## 失敗時のフォールバック

- schtasks 失敗 (admin 必要等) → 手動で 1日数回実行
- HTTP 認証失敗 → .env JRDB credentials 確認
- ログ CSV 書き込み失敗 → ファイル権限確認

## 次セッション継続課題

- [ ] 5/10 (日) 終了時点で `data/tyb_publish_log.csv` 解析
- [ ] publish 時刻分布から TYB 戦略 生死判定
- [ ] midday script の retry interval 調整 or 廃止判断
