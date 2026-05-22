# 影-3: 6/1 Shadow Observation Plan — TYB Per-Race Live Fetch

**作成日**: 2026-05-22
**対象日**: 2026-06-01 (日曜、JRA開催)
**目的**: tyokuzen path での TYB per-race 更新を AM R01 から全 R12 まで直接観測し、V21 production fetch の Go/No-Go 判定材料を収集する
**前提 docs**: `docs/TYB_RELEASE_TIMING_RE_AUDIT_2026_05_21.md` / `docs/TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md`
**制約**: V15 production コード変更なし。shadow のみ。

---

## 0. TL;DR (前提確認)

| 確認済事項 | 根拠 |
|-----------|------|
| TYB content: 全 26 fields PRE_RACE | `TYB_RELEASE_TIMING_RE_AUDIT_2026_05_21.md §2` |
| TYB per-race 更新: CONFIRMED (10/10 dates) | `TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md §3` |
| R01 fetch window: ~09:20-09:30 JST | `TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md §4` |
| odds_time delta: 37-53 min (全 R01-R12) | `TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md §2` |
| 未確認: tyokuzen path の R01 fetch (HTTPアクセス) | ★本観測で直接確認★ |

---

## 1. 事前準備 (5/31 土曜夜 または 6/1 日曜朝)

### 1.1 シャドウモードの有効化

**重要**: production コード (`race_auto_notify.py`, `predict_core.py`) は一切変更しない。
shadow 専用スクリプトのみで観測する。

```python
# tools/tyb_shadow_observer.py を作成 (6/1 当日のみ使用)
TYB_SHADOW_ENABLED = True   # ← このスクリプト内のみ。本番 .env / predict_core には追加しない
TYOKUZEN_URL = "http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh"
STANDARD_URL = "http://www.jrdb.com/member/data/Tyb/TYB{yymmdd}.lzh"
OUTPUT_DIR = "data/tyb_shadow/20260601/"
```

### 1.2 JRDB 認証確認

```bash
# .env に以下が設定済みか確認
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
u = os.getenv('JRDB_USER'); p = os.getenv('JRDB_PASS')
print('USER:', 'OK' if u else 'MISSING')
print('PASS:', 'OK' if p else 'MISSING')
"
```

- `JRDB_USER` / `JRDB_PASS` のどちらかが MISSING なら即座に停止。6/1 観測は実施しない。

### 1.3 7-Zip 動作確認

```bash
# Windows: 7-Zip が PATH に通っているか確認
7z --help 2>&1 | head -3
# または
& "C:\Program Files\7-Zip\7z.exe" --help | head -3
```

`.lzh` 展開が不可なら `lhaz` 等を代替確認。既存 batch pipeline が使用しているコマンドと同一ツールを使う。

### 1.4 事前パース動作確認 (前週データ)

```python
# 前週 (5/25) の TYB データで parse が通るか確認
python tools/parse_jrdb.py --mode tyb --file data/jrdb/extracted/Tyb/TYB260525.txt --races 01 06 12
# 期待出力: race_num / umaban / odds_idx / padock_idx / tansho_odds / horse_weight / start_time / odds_time が全フィールド揃うこと
```

エラーが出た場合は `tools/parse_jrdb.py` の TYB_COLUMNS 定義 (行 258) を確認。

### 1.5 出力ディレクトリ作成

```bash
mkdir -p data/tyb_shadow/20260601
```

---

## 2. 当日観測計画 (6/1 日曜)

### 観測対象レース (3 ウィンドウ)

| ウィンドウ | 対象 | fetch 実行時刻 | 観測目的 |
|----------|------|-------------|---------|
| 朝 (W1) | R01 | **09:20-09:30 JST** | 午前 race が tyokuzen path で取得できるか (最重要) |
| 中盤 (W2) | R06 | **約 12:10-12:15 JST** | 日中 update が cumulative に追加されているか |
| 終盤 (W3) | R12 | **約 16:10-16:20 JST** | 既知の 16:15-16:21 JST 更新と一致するか |

### 全レース観測 (W1-W3 の間、可能なら)

R02〜R11 は W1/W2/W3 の間に順次 fetch を実施。各 race start の -20 min を target に。
失敗しても W1/W2/W3 さえ取れれば十分。

---

## 3. fetch スクリプト設計 (当日スクリプト概要)

```python
# tools/tyb_shadow_observer.py — 6/1 当日実行用 (本番コードに組み込まない)

import requests, os, subprocess, time, json, csv
from datetime import datetime
from pathlib import Path

DATE_STR = "20260601"
DATE_SHORT = "260601"
OUT_DIR = Path(f"data/tyb_shadow/{DATE_STR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JRDB_USER = os.getenv("JRDB_USER")
JRDB_PASS = os.getenv("JRDB_PASS")

TYOKUZEN_URL = f"http://www.jrdb.com/member/{DATE_STR}/tyokuzen/TYB{DATE_SHORT}.lzh"
STANDARD_URL = f"http://www.jrdb.com/member/data/Tyb/TYB{DATE_SHORT}.lzh"

def fetch_tyb(url, label, race_filter=None):
    """1回 fetch、 parse、 save"""
    ts = datetime.now().strftime("%H%M%S")
    lzh_path = OUT_DIR / f"{label}_{ts}.lzh"
    txt_path  = OUT_DIR / f"{label}_{ts}.txt"

    resp = requests.get(url, auth=(JRDB_USER, JRDB_PASS), timeout=30)
    status = resp.status_code

    result = {
        "label": label, "url": url, "fetch_ts": ts,
        "http_status": status, "size_bytes": len(resp.content),
        "race_records": {}
    }

    if status == 200:
        lzh_path.write_bytes(resp.content)
        # 展開
        subprocess.run(["7z", "e", str(lzh_path), f"-o{OUT_DIR}", "-y"],
                       capture_output=True)
        # parse
        from tools.parse_jrdb import parse_tyb_file  # 既存 parser 流用
        records = parse_tyb_file(str(txt_path))
        for r in records:
            rn = r.get("race_num")
            if race_filter and rn not in race_filter:
                continue
            result["race_records"].setdefault(rn, []).append({
                "umaban":       r.get("umaban"),
                "tansho_odds":  r.get("tansho_odds"),
                "odds_idx":     r.get("odds_idx"),
                "padock_idx":   r.get("padock_idx"),
                "horse_weight": r.get("horse_weight"),
                "odds_time":    r.get("odds_time"),
                "start_time":   r.get("start_time"),
            })
        # JSON スナップショット保存
        snap_path = OUT_DIR / f"{label}_{ts}_snap.json"
        snap_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[OK] {label} status={status} races={list(result['race_records'].keys())}")
    else:
        print(f"[FAIL] {label} status={status}")

    return result
```

---

## 4. 観測チェックリスト (各レースウィンドウ)

### W1: R01 観測 (~09:20-09:30 JST)

- [ ] tyokuzen URL に HTTP GET → status=200 を確認
- [ ] `.lzh` 取得、7-Zip で展開成功
- [ ] parse → R01 レコード数を確認 (当日 R01 の馬数と一致するか)
- [ ] `odds_time` フィールドが `start_time - 15〜40 min` の範囲内か
- [ ] R02〜R12 のレコードが **まだない** こと (cumulative ではあるが R01 fetch 時点では R01 のみ)
- [ ] snapshot を `data/tyb_shadow/20260601/R01_{HHMMSS}_snap.json` に保存

### W2: R06 観測 (~12:10-12:15 JST)

- [ ] fetch → status=200
- [ ] R01〜R06 のレコードが全て揃っているか (cumulative 追加を確認)
- [ ] R06 の `odds_time` が `start_time - 15〜40 min` 範囲内か
- [ ] R07 以降のレコードがまだないこと
- [ ] snapshot 保存

### W3: R12 観測 (~16:10-16:20 JST)

- [ ] fetch → status=200
- [ ] R01〜R12 全馬レコード存在
- [ ] R12 の `odds_time` が `start_time - 15〜40 min` 範囲内か
- [ ] `fetch_ts` - `odds_time` の delta を計算して記録
- [ ] snapshot 保存

---

## 5. 検証項目 (観測全体)

### 5.1 TYB 取得可否

| 項目 | 期待値 | 判定基準 |
|------|--------|---------|
| R01 fetch status | 200 | 200 = PASS / 404 = 未公開 / 401 = 認証エラー |
| R06 fetch status | 200 | 同上 |
| R12 fetch status | 200 | 同上 (これまでの観測で確認済) |
| parse 成功率 | ≥ 95% | 取得レコード数 / 期待レコード数 |

### 5.2 odds_time フィールド検証

各 race について: `delta = start_time (HHMM) - odds_time (HHMM)` を分単位で計算。

| 期待範囲 | 根拠 |
|---------|------|
| 15〜60 分 | `TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md §2`: 実測 37-53 min (5/16) |
| R01: ~37 min | 過去 10 日平均 |
| R12: ~53 min | 過去 10 日平均 |

**delta が範囲外の場合**: fetch タイミングのずれ (遅れて fetch した) 可能性あり。odds_time 自体を信じる。

### 5.3 馬数完全性チェック

```python
# 各 race の期待馬数は netkeiba 出馬表から事前に取得しておく
# 例: R01 = 16頭なら、TYB に 16 レコード存在するか
expected_horses = {
    "01": None,  # 6/1 前日に netkeiba から確認して埋める
    "06": None,
    "12": None,
}
```

- parse 結果のレコード数 == 出馬表の頭数 → COMPLETE
- 欠損あり → PARTIAL (欠損馬番を記録)

### 5.4 IP/レートリミット確認

- 各 fetch の間隔: 最低 60 秒以上あける
- 連続 fetch でも 404/429 が出ないことを確認
- W1/W2/W3 の 3 回の fetch で問題なければ rate limit は OK と判断

---

## 6. データ収集・保存

### 保存先

```
data/tyb_shadow/20260601/
  R01_{HHMMSS}.lzh          # 生 lzh ファイル
  R01_{HHMMSS}.txt          # 展開後 txt
  R01_{HHMMSS}_snap.json    # parse 結果 JSON
  R06_{HHMMSS}_snap.json
  R12_{HHMMSS}_snap.json
  summary.csv               # 全レース集計 (当日最後に作成)
```

### summary.csv フォーマット

```csv
race_num, fetch_ts, http_status, horse_count, odds_time_min, odds_time_max, start_time, delta_min_min, delta_min_max, parse_ok
01, 092534, 200, 16, 0928, 0947, 1005, 18, 37, True
06, 121223, 200, 14, 1212, 1239, 1255, 16, 43, True
12, 161724, 200, 18, 1537, 1612, 1630, 18, 53, True
```

### 当日最後: odds_idx vs 着順の相関

```python
# 全レース終了後 (17:30以降) に実行
# TYB の odds_idx と当日実際の着順 (netkeiba から取得) を突き合わせ
# corr_target を計算 → 先行 audit の +0.42 と一致するか確認
```

---

## 7. 失敗時対応

### R01 fetch が 404 の場合

- 原因 A: まだ公開前 (09:20 では早すぎた) → 09:30, 09:35 で再 fetch (最大 3 回、各 60 秒間隔)
- 原因 B: tyokuzen path が R01 には使えない → standard path (17:00 公開) にフォールバック
- 判定: 3 回全て 404 → **R01 tyokuzen 未対応** と記録。W2/W3 は続行。

### 認証エラー (401/403) の場合

- `.env` の JRDB_USER/JRDB_PASS を確認
- 別の JRDB URL (standard path) で認証テスト
- 認証自体が失敗する場合は観測中止 (data 取得なし)

### parse エラーの場合

- `tools/parse_jrdb.py` の TYB_COLUMNS 定義 (L258) と実 txt の byte 長を比較
- フォーマット変更の可能性: raw txt の先頭 10 行を手動確認
- **積極的 retry はしない** — 観測結果に「parse エラー」と記録して次へ

### rate limit (429) の場合

- 即座に停止。30 分待機後に再試行 (最大 1 回)
- JRDB 規約上の許容 polling 間隔が不明なため慎重に対応

---

## 8. 成功基準

| 基準 | 値 | 判定 |
|------|----|------|
| R01 tyokuzen fetch | status=200 | **必須** — 午前 race の tyokuzen 可否が最重要 |
| R01-R12 全 parse 成功率 | ≥ 95% | PASS |
| odds_time delta 範囲 | 15-60 min (全 race) | PASS |
| IP 問題なし | 429/503 なし | PASS |
| 全 race 馬数完全性 | ≥ 90% | PASS |

**全 PASS → V21 live fetch GO 判定**
**R01 fetch 404 → tyokuzen 午前 race の fetch 戦略を見直し (standard path 使用に変更)**
**parse error ≥ 5% → フォーマット変更調査が必要**

---

## 9. 観測後アクション

6/1 観測完了後:
1. `data/tyb_shadow/20260601/summary.csv` を確認
2. 全基準 PASS なら: `docs/影-3_V21_TRAINING_PLAN.md` の timeline を維持 (6/9+ JV-Link + TYB merge)
3. R01 fetch 404 なら: V21 prediction fetch を R02 以降のみに制限する設計変更を検討
4. 結果を Discord #アップデート チャンネルに投稿: `python tools/notify_done.py "影-3 6/1 TYB観測" "結果サマリー"`

---

## 付記: 現行 production への影響

**0 件** — 本観測は shadow のみ。
- `race_auto_notify.py` 変更なし
- `predict_core.py` 変更なし
- V15 model / `.pkl.gz` 変更なし
- `data/jrdb_tyb.csv` / `data/jrdb/extracted/` への書き込みなし (shadow dir のみ)
