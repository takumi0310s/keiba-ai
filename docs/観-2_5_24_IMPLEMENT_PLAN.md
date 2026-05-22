# 観-2: 5/24 TYB Discord 補足表示 実装計画

**作成日**: 2026-05-22  
**前提**: 5/23 (SAT) 観測 mode で TYB fetch log 収集 → 夜に結果確認 → GO なら 5/24 朝に実装  
**絶対ルール**: V15 production / predict_core.py / app.py / LiveOrchestrator / race_auto_notify.py の予測ロジックは変更しない

---

## 1. 5/23 観測結果の判定基準

| 観測項目 | GO 条件 | NO-GO 条件 |
|---------|---------|-----------|
| 午前 R (1-6R) 取得 | 少なくとも 1R が発走前に取得できた | 全 R 404 / timeout |
| 午後 R (7-12R) 取得 | 少なくとも 1R が発走前に取得できた | — |
| parse 成功率 | ≥ 80% | < 50% |
| lzh DL 成功 | ≥ 1 件 OK | 全件 ERROR |
| 5/24 実装 GO/NO-GO | 上記全 GO 条件を満たす | いずれかが NO-GO |

---

## 2. 観測結果確認コマンド

5/23 夜に実行するコマンド:

```bash
python -c "
from tools.tyb_shadow_fetcher import summarize_observe_log
import json
print(json.dumps(summarize_observe_log('20260523'), ensure_ascii=False, indent=2))
"
```

### 判断基準

| キー | GO 判定値 | 意味 |
|-----|---------|------|
| `morning_ok` | `True` | 午前 R (1-6R) 少なくとも 1R 取得成功 |
| `ok_count` | `>= 3` | 実用的な取得件数 |
| `min_delta` | `>= 5` | 発走 5 分以上前に取得 → ギリギリ使える |
| `error_count` | `<= (total_fetches * 0.2)` | parse 成功率 ≥ 80% |

**全条件 OK → 5/24 GO。いずれか NG → 下記 NO-GO 対応表を参照。**

---

## 3. 5/24 実装手順 (5/23 観測 GO が条件)

### 3-1. tyb_shadow_fetcher.py の設定変更

`tools/tyb_shadow_fetcher.py` の `TYB_SHADOW_ENABLED` フラグを変更:

```python
# TYB_SHADOW_ENABLED = False → True に変更 (Discord 表示有効化)
TYB_SHADOW_ENABLED: bool = True   # ← 変更
```

### 3-2. race_auto_notify.py への TYB ブロック統合

`tools/race_auto_notify.py` の `predict_and_notify()` 関数の `send_discord(title, msg, ...)` 直後に以下を追加:

```python
# ──── TYB 直前情報補足 (5/24 追加、 観測成功後のみ有効) ────
# ★ V15 予測ロジック・投票 formation は一切変更しない ★
try:
    from tools.tyb_shadow_fetcher import (
        fetch_tyb_observe, build_tyb_discord_block, TYB_SHADOW_ENABLED,
    )
    if TYB_SHADOW_ENABLED:
        _start_time = rinfo.get("start_time", "")
        _tyb = fetch_tyb_observe(race_id, _start_time)
        _top5 = [
            {"umaban": int(r.get("馬番", 0)), "horse_name": r.get("馬名", ""), "score": r.get("スコア", 0)}
            for _, r in df.head(5).iterrows()
        ]
        _tyb_block = build_tyb_discord_block(_tyb, _top5, race_num=race_info.get("race_num"))
        if _tyb_block:
            from notify import send_discord as _sd
            _sd(f"直前情報 R{race_info.get('race_num','?')}", _tyb_block, color="blue", channel="bets")
except Exception:
    pass  # TYB 失敗は V15 運用に影響しない
# ──────────────────────────────────────────────────────────────
```

### 3-3. 挿入箇所の特定

`tools/race_auto_notify.py` の以下の箇所の直後に挿入:

```python
send_discord(title, msg, color=color, channel="bets")
print(f"    Notified: {race_name} ...")
```

の `send_discord` 直後の行に挿入。

**注意**: `race_auto_notify.py` の V15 予測ロジック部分 (`predict_race()`, `build_features()`, `generate_trio_bets()`) は一切触れない。

### 3-4. 実装後の確認

```bash
# 構文チェック
python -c "import py_compile; py_compile.compile('tools/race_auto_notify.py', doraise=True)"

# V15 model 不変確認
python -c "
import gzip, pickle
with gzip.open('keiba_model_v15_central.pkl.gz', 'rb') as f:
    v15 = pickle.load(f)
print('features:', len(v15['model'].feature_name()))  # must be 145
print('version:', v15.get('version'))
"

# test
python -m pytest tests/test_tyb_shadow_fetcher.py -v
```

---

## 4. 5/23 観測 NO-GO の場合

| NO-GO 理由 | 次アクション |
|-----------|------------|
| 午前 R 404 (まだサーバー未掲載) | 午前 R は skip、午後 R のみで運用 |
| 全件 ERROR (認証失敗) | JRDB 認証確認 → 修正後 5/30 再挑戦 |
| parse 失敗 (format 変更) | フォーマット再解析 → 6/7 再挑戦 |
| delta < 5 min (ギリギリすぎ) | fetch タイミングを -25 min に変更 |

---

## 5. V15 非影響保証

```
V15 production 不変確認項目:
✅ keiba_model_v15_central.pkl.gz — 変更なし (features=145)
✅ tools/predict_core.py — 変更なし
✅ tools/race_auto_notify.py — TYB ブロック追加のみ (try/except で保護)
✅ 投票 formation — V15 + 戦略⑦ + C4 のまま
✅ TYB は表示 / alert のみ (AI 予測への組み込みなし)
```

### 安全設計の根拠

- `fetch_tyb_observe()` は例外を全 swallow → V15 caller に propagate しない
- `build_tyb_discord_block(None, ...)` → `""` を返す → 送信スキップ
- `TYB_SHADOW_ENABLED = False` のまま → `fetch_tyb_observe()` 内 `TYB_SHADOW_OBSERVE_MODE` ガードで即 `None` 返却
- 全処理を `try/except Exception: pass` で囲む → TYB 失敗時も V15 通知フローに影響なし

---

## 6. タイムライン

| 日時 | アクション |
|------|---------|
| 5/23 (SAT) 朝〜夕 | 観測 mode で TYB fetch log 収集 (`TYB_SHADOW_OBSERVE_MODE=True`) |
| 5/23 夜 | `summarize_observe_log('20260523')` で結果確認 → GO/NO-GO 判定 |
| 5/24 (SUN) 朝 | GO なら本手順書に従い実装 |
| 5/24 実装完了 | 当日 1R 前に `race_auto_notify.py` + `tyb_shadow_fetcher.py` 更新 |
| 5/24 夕 | 実際の Discord 補足表示を確認 |
| 5/24 NO-GO | 4節の対応表に従い次アクション実施 |

---

## 参照

- 設計 doc: `docs/TYB_HUMAN_JUDGMENT_SUPPORT.md`
- 実装済み関数: `tools/tyb_shadow_fetcher.py`
  - `fetch_tyb_observe(race_id, start_time_str)` — 観測 mode fetch
  - `build_tyb_discord_block(tyb_result, top_horses, race_num)` — Discord ブロック組み立て
  - `check_tyb_anomalies(tyb_result, top_horses)` — 気配急変 alert 検出
  - `summarize_observe_log(date_str)` — 観測ログ集計
- テスト: `tests/test_tyb_shadow_fetcher.py` (12 tests all PASS)

*V15 production 完全不変保証 — TYB は人間判断支援の表示のみ (2026-05-22)*
