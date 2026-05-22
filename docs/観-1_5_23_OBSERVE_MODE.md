# 観-1: TYB 5/23 観測 mode 設計

**作成日**: 2026-05-22
**担当セッション**: Session #91 (観-1)

---

## 目的

5/23 (SAT) = 中央開催日 = LiveOrchestrator 初 fire 日。
TYB shadow fetch を ★ log 観測 only ★ で有効化し、
午前 R の取得可否・delta 分布・parse 成功率を確認する。

Discord 表示は 6/1+ まで行わない。
V15 production (keiba_model_v15_central.pkl.gz / predict_core.py / app.py) への影響は **ゼロ**。

---

## 観測 mode 設定値

| 定数 | 値 | 意味 |
|------|-----|------|
| `TYB_SHADOW_OBSERVE_MODE` | `True` | 観測 mode 有効 (5/23 観測のため) |
| `TYB_OBSERVE_LAUNCH_DATE` | `"20260523"` | 観測開始日 (それ以前はガード) |
| `TYB_SHADOW_ENABLED` | `False` | Discord 表示は 6/1+ まで無効 (変更なし) |
| `TYB_LAUNCH_DATE` | `"20260601"` | Discord 表示有効化日 (変更なし) |

### 定数の役割分離

```
TYB_SHADOW_ENABLED = False        ← 6/1 まで Discord 表示なし (既存)
TYB_LAUNCH_DATE   = "20260601"   ← Discord 有効化日 (既存)

TYB_SHADOW_OBSERVE_MODE = True    ← 5/23 観測のため追加
TYB_OBSERVE_LAUNCH_DATE = "20260523" ← 観測開始日ガード (追加)
```

`fetch_tyb_observe()` は `TYB_SHADOW_ENABLED` を **バイパス** して動作する。
ただし日付ガードにより `TYB_OBSERVE_LAUNCH_DATE` 未満の日付では即 None を返す。

---

## 観測 API

### `fetch_tyb_observe(race_id, start_time_str) -> Optional[dict]`

- **役割**: 観測 fetch 専用エントリポイント
- **動作**: `TYB_SHADOW_OBSERVE_MODE=True` かつ当日 >= `TYB_OBSERVE_LAUNCH_DATE` のとき動作
- **log 記録**: `data/tyb_shadow_log.csv` + `data/tyb_shadow/{date}/{race_id}_tyb.json`
- **Discord 表示**: **なし** (build_tyb_discord_block の結果は呼ばない)
- **V15 への影響**: **ゼロ** (返り値を V15 inference / 投票 formation に渡してはならない)
- **例外処理**: 全 swallow → LiveOrchestrator に一切 propagate しない

### `summarize_observe_log(date_str=None) -> dict`

- **役割**: 観測ログのサマリー集計
- **デフォルト**: `date_str=None` → today (JST)
- **返却フィールド**:
  - `date`, `total_fetches`, `ok_count`, `error_count`
  - `morning_ok` (R01-R06 いずれか成功)
  - `afternoon_ok` (R07-R12 いずれか成功)
  - `min_delta`, `max_delta` (発走前 delta 分)
  - `rows` (生データリスト)

---

## 安全保証 (絶対ルール)

| 項目 | 保証内容 |
|------|---------|
| V15 production 非影響 | `fetch_tyb_observe()` は V15 inference pipeline に一切接触しない |
| LiveOrchestrator 非影響 | 例外全 swallow → caller への propagate ゼロ |
| Discord 非表示 | `build_tyb_discord_block()` を呼ばない |
| 投票非影響 | 返り値を買い目生成・投票ロジックに渡してはならない |
| 日付ガード | `TYB_OBSERVE_LAUNCH_DATE` = "20260523" 未満の日付は即 None 返却 |
| 既存 disable 維持 | `TYB_SHADOW_ENABLED = False` は変更なし |

---

## 5/23 観測手順

### 前提確認 (5/23 朝 7:00 前)

```python
# Python で定数確認
import tools.tyb_shadow_fetcher as tsf
assert tsf.TYB_SHADOW_OBSERVE_MODE is True
assert tsf.TYB_SHADOW_ENABLED is False
assert tsf.TYB_OBSERVE_LAUNCH_DATE == "20260523"
assert tsf.TYB_LAUNCH_DATE == "20260601"
print("OK: observe mode ready")
```

### LiveOrchestrator からの呼び出し方 (fire-and-forget)

```python
# live_orchestrator_main.py 内 (既存コード変更なし — 参考のみ)
# LiveOrchestrator は race_auto_notify.py を経由するため
# fetch_tyb_observe() は独立した観測専用スレッドで呼ぶ想定。
# ★ 既存の LiveOrchestrator コードは一切変更しない ★

# 手動観測 (5/23 朝 CLI から):
from tools.tyb_shadow_fetcher import fetch_tyb_observe
result = fetch_tyb_observe("202605230101", "1030")
# → data/tyb_shadow_log.csv に記録される
```

### 午前レース観測 (R01-R06 対象)

```bash
# 5/23 各レース発走 30-45 分前に手動実行 (または LiveOrchestrator から自動)
python -c "
from tools.tyb_shadow_fetcher import fetch_tyb_observe
r = fetch_tyb_observe('202605230101', '1000')  # R01 例
print('Result:', r is not None)
"
```

### 観測サマリー確認 (5/23 夕方以降)

```python
from tools.tyb_shadow_fetcher import summarize_observe_log
summary = summarize_observe_log("20260523")
print(f"Total: {summary['total_fetches']}")
print(f"OK: {summary['ok_count']} / Error: {summary['error_count']}")
print(f"Morning OK: {summary['morning_ok']}, Afternoon OK: {summary['afternoon_ok']}")
print(f"Delta range: {summary['min_delta']:.1f} ~ {summary['max_delta']:.1f} 分")
```

---

## 観測成功判定基準

| 基準 | 合格 | 備考 |
|------|------|------|
| OK 率 | >= 80% | JRDB 配信遅延は許容 |
| 午前レース成功 | morning_ok = True | R01-R06 いずれか 1 件以上 |
| delta 分布 | 15-60 分前に集中 | 発走直前すぎない |
| parse 成功 | num_horses >= 8 | TYB ファイル破損チェック |
| V15 非影響 | エラーなし | cumulative_results.csv 変化なし |

### GO/NO-GO 判定 (5/24 朝)

- **全基準 PASS** → 観測成功。5/24+ の詳細分析ステップへ
- **OK 率 < 80%** → JRDB 認証・URL パターン確認
- **num_horses 不整合** → TYB ファイル構造変更の可能性、parse ロジック見直し
- **V15 影響あり** → 即 `TYB_SHADOW_OBSERVE_MODE = False` に戻す (緊急ロールバック)

---

## テスト (pytest)

```bash
python -m pytest tests/test_tyb_shadow_fetcher.py -v
```

追加テスト (Test 13-15):

| Test # | 内容 |
|--------|------|
| 13 | `TYB_SHADOW_OBSERVE_MODE` が True であること |
| 14 | 過去日付 (20260101) で `fetch_tyb_observe` が None を返す (日付ガード) |
| 15 | `fetch_tyb_observe` が例外を swallow し LiveOrchestrator を保護する |

---

## ロールバック手順

観測モードを無効化する場合:

```python
# tools/tyb_shadow_fetcher.py の定数を変更
TYB_SHADOW_OBSERVE_MODE: bool = False  # True → False
```

この変更のみで観測は即停止する。
V15 production / LiveOrchestrator / race_auto_notify.py への変更は不要。

---

## ファイル構成

| ファイル | 変更内容 |
|----------|---------|
| `tools/tyb_shadow_fetcher.py` | 定数追加 + `fetch_tyb_observe()` + `summarize_observe_log()` |
| `tests/test_tyb_shadow_fetcher.py` | Test 13-15 追加 |
| `docs/観-1_5_23_OBSERVE_MODE.md` | 本ドキュメント |

**変更なし**:

| ファイル | 理由 |
|----------|------|
| `tools/live_orchestrator_main.py` | 絶対変更禁止 |
| `tools/race_auto_notify.py` | 絶対変更禁止 |
| `tools/predict_core.py` | V15 production 保護 |
| `app.py` | V15 production 保護 |
| `keiba_model_v15_central.pkl.gz` | V15 production 保護 |

---

## 関連ドキュメント

- `docs/jrdb_tyb_verdict.md` — JRDB TYB 真の verdict (content 全 PRE-RACE 確認)
- `CLAUDE.md` Section 1 — JRA-VAN / JV-Link 環境
- `CLAUDE.md` リークフリールール — TYB は V15 145 features に未含有
