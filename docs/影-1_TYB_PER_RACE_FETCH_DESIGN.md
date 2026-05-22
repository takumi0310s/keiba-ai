# 影-1: TYB Per-Race Fetch Chain — 設計書

**作成日**: 2026-05-22
**対象フェーズ**: Phase 3 (5/24+) 観測専用 shadow mode
**前提 docs**:
- `docs/TYB_RELEASE_TIMING_RE_AUDIT_2026_05_21.md` — field 全 26 件 PRE_RACE 確認、tyokuzen path 確認
- `docs/TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md` — per-race update 10/10 cross-day confirmed

---

## 0. TL;DR

| 項目 | 値 |
|------|----|
| **5/23 (初回開催日)** | TYB fetch **完全禁止** — shadow も含め無効 |
| **shadow default** | `TYB_SHADOW_ENABLED = False` |
| **V15 inference への影響** | **ゼロ** — fire-and-forget、例外でも主フローは継続 |
| **fetch 回数上限** | 12 fetch/day (1 race = 1 fetch、retry なし) |
| **integration point** | `process_race()` 内の fetch_pre_features 呼び出し **直前** (L103-124) |

---

## A. タイミング設計

### A.1 TYB ファイル更新タイミング (確認済)

JRDB 公式 spec より:
> 「直前累積データは直前データと同じタイミングで更新される。更新日時: 競馬開催日 **各レース出走 15 分前頃**」

10 開催日 cross-day 実測 (2026-04-12〜05-16、全件 R01 odds_time ≠ 最終 R odds_time):
- R01 fetch 可能時刻: ~09:20-09:30 JST (start ~09:45)
- 以降各 race: start_time - 15〜20 min で上書き更新
- URL: `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh`

### A.2 per-race fetch フロー

```
start_time - 20 min
     │
     ▼
[fetch_tyb_shadow(race_id, start_time)]
     │
     ├─ TYByymmdd.lzh を tyokuzen path から HTTP GET (1回のみ)
     ├─ 7z e → tmp_dir → TYByymmdd.txt
     ├─ parse_tyb_line() × 全行 → filter: race_num == current_race
     └─ 抽出: odds_idx, padock_idx, tansho_odds, fukusho_odds,
              horse_weight, weight_diff, padock_mark, ashimoto
```

### A.3 呼び出しタイミングの根拠

- `start_time - 20 min` = JRDB 公式「20 分前前後」中央値
- 早すぎる場合 (race データまだ未書込み) → HTTP 200 返るが race_num フィルタ後レコード 0 → silent skip
- 遅すぎると odds が更新され変動 → -20 min を基準とし retry しない

---

## B. Shadow Mode (CRITICAL)

### B.1 絶対ルール

```python
TYB_SHADOW_ENABLED = False  # ★ default DISABLED ★

# shadow mode で許可されること:
#   - tyokuzen path からの HTTP fetch
#   - .lzh extract → parse
#   - data/tyb_shadow/{date}/{race_id}_tyb.json への保存
#   - data/tyb_shadow_log.csv への append
#   - Discord へのオプション補足メッセージ (bet 通知とは別チャンネル)

# shadow mode で禁止されること (絶対):
#   - V15 inference の feature vector への混入
#   - predict_core.py / predict_one_race.py への引き渡し
#   - betting formation (trio/umaren) の変更
#   - race_auto_notify.py の buy signal への影響
#   - recalc_15min.py への引き渡し
```

### B.2 shadow が有効でも無効でも V15 は identical

`fetch_tyb_shadow()` は以下を保証する:
1. 戻り値を live_orchestrator_main.py は **使用しない** (fire-and-forget)
2. いかなる例外も catch して swallow し、main flow へ伝播しない
3. `TYB_SHADOW_ENABLED = False` なら関数冒頭で即 return

### B.3 5/23 (初回開催日) の特別ガード

```python
# 5/23 は絶対に fetch しない
TYB_SHADOW_LAUNCH_DATE = "20260524"  # この日以降のみ有効

def fetch_tyb_shadow(race_id: str, start_time: datetime, date_str: str) -> None:
    if not TYB_SHADOW_ENABLED:
        return
    if date_str < TYB_SHADOW_LAUNCH_DATE:
        log.warning("TYB shadow: before launch date, skip")
        return
    ...
```

---

## C. IP BAN 回避設計

### C.1 基本方針 (netkeiba 403 事件からの教訓)

> netkeiba: aggressive polling → IP BAN → Cookie refresh が必要になった。
> JRDB: 同様のリスクあり。1 shot per race を厳守。

| 制約 | 値 |
|------|----|
| fetch 回数 | **1 fetch/race** (retry なし) |
| 上限 | **12 fetch/day** (R01〜R12) |
| inter-fetch interval | レース間隔 (30〜45 min) = 自然に守られる |
| User-Agent | `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36` |
| 認証 | HTTP Basic Auth (`JRDB_USER` / `JRDB_PASSWORD`、.env から取得) |
| timeout | 15 sec |
| retry | **なし** — 失敗は silent skip |

### C.2 禁止事項

```python
# NG: retry loop
for attempt in range(3):
    resp = requests.get(url)

# OK: 1 shot only
try:
    resp = requests.get(url, timeout=15, auth=(user, pw), headers=HEADERS)
    resp.raise_for_status()
except Exception:
    log.warning("TYB fetch fail, skip")
    return  # V15 continues normally
```

### C.3 JRDB 認証情報

```python
import os
JRDB_USER = os.environ.get("JRDB_USER", "")
JRDB_PASSWORD = os.environ.get("JRDB_PASSWORD", "")
# .env に既存。不在時は credential error → shadow disabled for the day
```

---

## D. 失敗ハンドリング

| エラー種別 | 挙動 |
|-----------|------|
| Network timeout / connection error | `log.warning`, return, V15 継続 |
| HTTP 4xx (auth error 401/403) | `log.error("credential error")`, `_disable_shadow_today()`, V15 継続 |
| HTTP 404 (ファイル未公開) | `log.warning("TYB not yet published")`, skip, V15 継続 |
| .lzh extract failure (7z exit != 0) | `log.warning`, skip, V15 継続 |
| parse error (malformed bytes) | `log.warning(f"parse error row={i}")`, skip, V15 継続 |
| race_num filter → 0 rows | `log.info("TYB: race not yet in file")`, skip, V15 継続 |
| いかなる未補足例外 | `except Exception as e: log.error(e)`, return, V15 継続 |

```python
def fetch_tyb_shadow(race_id: str, start_time: datetime, date_str: str) -> None:
    """Fire-and-forget TYB shadow fetch. NEVER raises. NEVER affects V15."""
    try:
        _fetch_tyb_shadow_inner(race_id, start_time, date_str)
    except Exception as e:
        _log_warning(date_str, {"event": "tyb_shadow_uncaught", "error": str(e)})
```

---

## E. Integration Point (live_orchestrator_main.py)

### E.1 現行の process_race() 構造

```
tools/live_orchestrator_main.py : process_race() (L70-148)

L86-93  : import + is_strategy_7c_excluded()
L95-101 : strategy 7C skip check
L103-124: ← ★ ここに TYB shadow call を挿入 ★
          fetch_pre_features() (live_data_fetcher)
L126-148: recalc_15min.run_recalc()
```

### E.2 挿入コード

```python
def process_race(
    race_id: str,
    date_str: str,
    race_meta: dict,
    mock: bool = True,
    dry_run: bool = True,
) -> dict:
    """1 race 処理 (-20/-15 min 順次)."""
    log_event(date_str, {"event": "process_race_start", "race_id": race_id})

    # ... (既存の import / strategy 7C check) ...

    # ★ 影-1: TYB shadow fetch (fire-and-forget) ★
    # V15 inference に影響しない、例外は全て swallow
    if not mock and not dry_run:
        try:
            from tyb_shadow_fetcher import fetch_tyb_shadow
            start_dt = _parse_start_time(race_meta)  # datetime or None
            fetch_tyb_shadow(race_id, start_dt, date_str)
        except Exception as e:
            # import failure も swallow (tyb_shadow_fetcher 未実装時も安全)
            log_event(date_str, {"event": "tyb_shadow_import_fail", "error": str(e)})

    # -20 min: live_data_fetcher (既存、変更なし)
    try:
        from live_data_fetcher import fetch_pre_features
        fetched = fetch_pre_features(
            race_id, date_str=date_str, mock=mock, dry_run=dry_run
        )
        ...
```

### E.3 保証事項

- `mock=True` or `dry_run=True` の場合は TYB shadow call をスキップ (5/23 含む全 dry-run)
- `fetch_tyb_shadow` の import 失敗も `except Exception` で swallow → orchestrator に影響なし
- TYB shadow の処理時間が長くても (max 15 sec timeout) orchestrator は次へ進む
  - 注: synchronous call のため、失敗が 15 sec かかる可能性。許容範囲 (-20 min window は十分)
  - 実装時に thread/subprocess 化を検討可能 (Phase 3 後半)

---

## F. .lzh 展開

### F.1 7-Zip 使用 (batch pipeline と同じ)

```python
import subprocess, tempfile, shutil
from pathlib import Path

def _extract_lzh(lzh_bytes: bytes, date_str: str) -> Path | None:
    """バイト列 (.lzh) を tmp_dir に展開して txt path を返す。失敗時 None。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tyb_{date_str}_"))
    lzh_path = tmp_dir / f"TYB{date_str[2:]}.lzh"
    lzh_path.write_bytes(lzh_bytes)

    result = subprocess.run(
        ["7z", "e", str(lzh_path), f"-o{tmp_dir}", "-y"],
        capture_output=True, timeout=30
    )
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    txt_path = tmp_dir / f"TYB{date_str[2:]}.txt"
    return txt_path if txt_path.exists() else None
```

### F.2 Python lhafile (代替、オプション)

```python
# pip install lhafile — 未インストール時は 7z フォールバック
try:
    import lhafile
    USE_LHAFILE = True
except ImportError:
    USE_LHAFILE = False
```

7-Zip が既に batch pipeline (`tools/download_jrdb.py` 等) で動作確認済のため、**7-Zip を第一選択** とする。

---

## G. データ保存

### G.1 per-race JSON スナップショット

```
data/tyb_shadow/
  {YYYYMMDD}/
    {race_id}_tyb.json   — race ごとの全馬 TYB フィールド
```

```json
{
  "race_id": "202605020611",
  "fetched_at": "2026-05-24T14:05:12.345678",
  "odds_time_range": ["1405", "1420"],
  "start_time": "1425",
  "horses": [
    {
      "umaban": 1,
      "odds_idx": 72.3,
      "padock_idx": 68.1,
      "tansho_odds": 4.5,
      "fukusho_odds": 1.8,
      "horse_weight": 480,
      "weight_diff": 2,
      "padock_mark": "A",
      "ashimoto": 1
    }
  ]
}
```

### G.2 累積 CSV ログ

```
data/tyb_shadow_log.csv
```

| カラム | 型 | 内容 |
|--------|----|------|
| date | str | YYYYMMDD |
| race_id | str | 12桁 race_id |
| fetched_at | str | ISO 8601 |
| http_status | int | 200/404/401/etc |
| n_horses | int | フィルタ後レコード数 |
| odds_time_min | str | HHMM |
| start_time | str | HHMM |
| delta_min | float | start - odds_time (分) |
| error | str | エラーメッセージ (正常時 "") |

### G.3 ディレクトリ作成

```python
SHADOW_DIR = REPO / "data" / "tyb_shadow"
SHADOW_DIR.mkdir(parents=True, exist_ok=True)
```

`.gitignore` 対象 (data/ は既に .gitignore 管理済)。

---

## H. 実装ファイル構成

### H.1 新規作成ファイル (5/24+ Phase 3 実装時)

```
tools/tyb_shadow_fetcher.py   — 本設計の実装本体
```

**V15 production の既存ファイルは一切変更しない**:
- `tools/predict_core.py` — 変更なし
- `tools/daily_predict.py` — 変更なし
- `app.py` — 変更なし
- `tools/live_data_fetcher.py` — 変更なし
- `tools/live_orchestrator_main.py` — shadow call の 8 行挿入のみ (mock/dry-run guard あり)
- schtasks — 変更なし

### H.2 tyb_shadow_fetcher.py API

```python
# tools/tyb_shadow_fetcher.py

TYB_SHADOW_ENABLED: bool = False      # ★ default DISABLED ★
TYB_SHADOW_LAUNCH_DATE: str = "20260524"  # 5/23 は絶対 fetch しない
TYB_TYOKUZEN_URL = "http://www.jrdb.com/member/{date}/tyokuzen/TYB{ymd}.lzh"

def fetch_tyb_shadow(
    race_id: str,
    start_time: datetime | None,
    date_str: str,
) -> None:
    """Fire-and-forget TYB per-race shadow fetch.

    Rules:
    - TYB_SHADOW_ENABLED=False → immediate return
    - date_str < TYB_SHADOW_LAUNCH_DATE → immediate return
    - Any exception → swallowed, logged
    - NEVER raises, NEVER affects V15
    """
    ...

def _fetch_tyb_shadow_inner(race_id, start_time, date_str) -> None:
    """実 fetch ロジック (fetch_tyb_shadow の内部)."""
    ...

def _build_url(date_str: str) -> str:
    ymd = date_str[2:]  # YYYYMMDD → yymmdd
    return TYB_TYOKUZEN_URL.format(date=date_str, ymd=ymd)

def _filter_race(records: list[dict], race_num: int) -> list[dict]:
    return [r for r in records if int(r.get("race_num", -1)) == race_num]

def _race_num_from_id(race_id: str) -> int:
    """race_id 末尾 2 桁 = race_num (01-12)."""
    return int(race_id[-2:])

def _disable_shadow_today() -> None:
    """認証エラー時: 当日の残 fetch を無効化 (module-level flag)."""
    global TYB_SHADOW_ENABLED
    TYB_SHADOW_ENABLED = False
```

---

## I. Discord 補足通知 (オプション)

shadow mode が有効かつ取得成功した場合、**#アップデート** チャンネル (`DISCORD_WEBHOOK_UPDATES`) へ補足送信可能。

```
[TYB shadow] 202605020611 R06
馬番 1: odds_idx=72.3 padock_idx=68.1 tansho=4.5 ashimoto=良化
馬番 2: odds_idx=65.0 padock_idx=70.2 tansho=6.3 ashimoto=平行線
...
```

**#買い目** (`DISCORD_WEBHOOK_BETS`) には **絶対に送信しない**。
shadow data は betting decision に関与しないため。

---

## J. フェーズ別有効化計画

| フェーズ | 日付 | TYB_SHADOW_ENABLED | 内容 |
|----------|------|-------------------|------|
| 5/23 (初回) | 20260523 | **False (絶対)** | V15 production 単独運用 |
| Phase 3 前半 | 5/24-5/31 | **False** (観測のみ準備) | fetch 機能実装・dry-run テスト |
| Phase 3 中盤 | 6/1+ | **True** (手動有効化) | shadow 観測開始、betには影響なし |
| Phase 3 後半 | 6/15+ | True + 蓄積 | V21 学習 data 収集 |
| V21 production | 7/1+ | (V21 設計で別途判定) | retrain + feature 統合 |

---

## K. 設計判断のまとめ

| 設計選択 | 理由 |
|----------|------|
| default DISABLED | 5/23 事故防止。有効化は明示的 flag 変更が必要 |
| retry なし | IP BAN 回避 (netkeiba 教訓)。1 shot で失敗なら skip |
| fire-and-forget call | V15 inference に絶対影響しない保証 |
| 7-Zip (.lzh) | 既存 batch pipeline で動作確認済 |
| mock/dry-run guard | test 実行 / 5/23 開催日での誤 fetch を防止 |
| per-race 1 fetch | JRDB rate limit 配慮、最大 12/day は許容範囲 |
| shadow log only | V21 学習 data 収集が主目的。betting には不使用 |

---

*作成: Session 影-1 / 2026-05-22*
