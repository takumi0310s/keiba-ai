# 影-2 TYB Shadow Fetcher — 設計・安全性ドキュメント

Session #91 実装。 2026-05-22 作成。

---

## モジュール概要

**ファイル**: `tools/tyb_shadow_fetcher.py`

JRDB TYB (直前情報) を観測専用 (shadow mode) で取得するモジュール。

| 項目 | 値 |
|------|----|
| データソース | JRDB tyokuzen TYB{yymmdd}.lzh |
| URL形式 | `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh` |
| 認証 | Basic auth (JRDB_USER / JRDB_PASSWORD via .env) |
| 取得タイミング | レース発走 15〜20 分前 (呼び出し側が制御) |
| レコード長 | 128 bytes (cp932 固定長) |
| パース対象 | race_num でフィルタ後: odds_idx / padock_idx / tansho_odds / horse_weight / padock_mark |

### 主要フィールド (TYB_COLUMNS 抜粋)

| フィールド | bytes | 内容 |
|-----------|-------|------|
| race_num | 7-8 | レース番号 |
| odds_idx | 26-30 | オッズ指数 |
| padock_idx | 31-35 | パドック指数 |
| tansho_odds | 73-78 | 単勝オッズ |
| horse_weight | 89-91 | 馬体重 |
| weight_diff | 92-94 | 馬体重増減 |
| padock_mark | 96 | パドック印 |
| cancel_flag | 48 | 取消フラグ |
| start_time | 100-103 | 発走時刻 HHMM |

---

## Default Disabled 安全保証

```python
# tools/tyb_shadow_fetcher.py 先頭
TYB_SHADOW_ENABLED: bool = False
```

`fetch_tyb_shadow()` の第3引数 `enabled` のデフォルト値は `TYB_SHADOW_ENABLED` (= `False`)。

```python
def fetch_tyb_shadow(race_id, start_time_str, enabled=TYB_SHADOW_ENABLED):
    if not enabled:
        return None   # ← 即返却、ネットワーク呼び出しなし、副作用なし
    ...
```

**disabled 時の保証**:
- ネットワーク接続 0 件
- ファイル書き込み 0 件 (data/tyb_shadow/ 以下 含む)
- 例外送出 0 件
- 実行時間 O(1) (定数)

---

## Shadow Mode が V15 投票に影響しない証明

1. **特徴量非使用**: V15 booster は 145 features で fix 済み。TYB フィールド (odds_idx, padock_idx 等) は V15 の feature list に存在しない。
2. **返り値の分離**: `fetch_tyb_shadow()` は独立した `dict | None` を返す。`predict_core.py` / `race_auto_notify.py` / `daily_predict.py` はこの返り値を受け取らない (shadow fetcher を import すらしない)。
3. **例外封鎖**: 内部の全例外は `try/except Exception` で捕捉され、`None` を返す。caller への例外伝播はゼロ。
4. **ログのみ**: 成功時も `data/tyb_shadow/{date}/{race_id}_tyb.json` と `data/tyb_shadow_log.csv` への書き込みのみ。既存 CSV/pkl/DB には書き込まない。

```
呼び出しフロー (shadow 有効時):
  [caller] → fetch_tyb_shadow(enabled=True)
           → download / parse / save to data/tyb_shadow/
           → return dict  ← 呼び出し側が Discord 補足に使うだけ
  V15 predict_core ← TYB 結果は一切到達しない
```

---

## 5/23 Fire 安全性証明

| 条件 | 保証 |
|------|------|
| `TYB_SHADOW_ENABLED = False` | モジュール import 時点で確定 |
| `enabled` 引数のデフォルト = `TYB_SHADOW_ENABLED` | 明示的に `enabled=True` を渡さない限り disabled |
| 既存ファイル (`race_auto_notify.py`, `live_orchestrator_main.py`, `daily_predict.py`) | tyb_shadow_fetcher を import していない |
| schtask / bat | tyb_shadow_fetcher を直接呼ばない |

**結論**: 5/23 の全自動タスクは tyb_shadow_fetcher を一切触れない。手動で `enabled=True` を明示しない限り発火不能。

---

## テストカバレッジ

**ファイル**: `tests/test_tyb_shadow_fetcher.py`

| # | テスト名 | 確認内容 |
|---|---------|---------|
| 1 | `test_default_disabled` | enabled=False → None 返却 |
| 2 | `test_disabled_no_side_effects` | enabled=False → data/tyb_shadow/ にファイル生成なし |
| 3 | `test_fetch_fail_returns_none` | ネットワークエラー (mock) → None 返却、例外なし |
| 4 | `test_fetch_fail_v15_unaffected` | fetch 失敗 → V15 pred dict に干渉しないことを確認 |
| 5 | `test_format_supplement_valid` | mock data → 非空文字列、馬体重大幅変化・取消・非影響テキスト含む |
| 6 | `test_no_523_impact` | `TYB_SHADOW_ENABLED == False` を assert / デフォルト enabled で None 返却 |

実行:
```bash
python -m pytest tests/test_tyb_shadow_fetcher.py -v
```

---

## 6/1 以降の観測開始方法

Shadow fetch を有効にするには、呼び出し時に `enabled=True` を明示する:

```python
from tools.tyb_shadow_fetcher import fetch_tyb_shadow, format_tyb_discord_supplement

# レース発走 ~15 分前に呼び出す
tyb = fetch_tyb_shadow(
    race_id="202606010611",
    start_time_str="1530",
    enabled=True,   # ← ここを True にする
)
if tyb:
    supplement = format_tyb_discord_supplement(tyb)
    # Discord に送るかどうかは呼び出し側が判断
    print(supplement)
```

または `tyb_shadow_fetcher.py` の先頭行を変更:
```python
TYB_SHADOW_ENABLED: bool = True   # 6/1 観測開始時
```

ただし **本番ファイル (race_auto_notify.py 等) は改変しない**。shadow 専用ルーティンから呼び出すこと。

---

## ファイル構成

```
tools/
  tyb_shadow_fetcher.py   ← 本体 (shadow only)
tests/
  test_tyb_shadow_fetcher.py  ← 6 テスト
data/
  tyb_shadow/             ← shadow JSON 保存先 (auto-created)
    {YYYYMMDD}/
      {race_id}_tyb.json
  tyb_shadow_log.csv      ← fetch 履歴 CSV
docs/
  影-2_TYB_SHADOW_LOGIC.md  ← このドキュメント
```
