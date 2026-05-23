# 5/23 V21 paper 2問題デバッグ

**実施時刻**: 2026-05-23 ~15:30  
**commit**: 49577719

---

## 問題① スコア重複 (新潟11R が 2 件)

### 原因
PID 20476 (14:00 起動) を kill せずに PID 2706 (14:41 起動) を開始。
両プロセスが 新潟11R@15:03 タイマーを保持 → 2 件の Discord 通知。

| PID | 起動時刻 | タイマー数 | 状態 |
|-----|---------|-----------|------|
| 20476 | 14:00 | 8 (14:38-16:13) | 15:20 頃に自然終了 |
| 22256 | (中間) | - | kill 済み |
| 2706 | 14:41 | 7 (14:48-16:13) | kill 済み (デバッグ時) |
| **2797** | **15:30** | **3 (15:44-16:13)** | **✅ 現在稼働** |

### 修正
再起動時は必ず古い process を全 kill してから新 process を起動する。

### 再発防止 (V21 paper 起動手順)
```bash
# 1. 古い V21 paper process を全 kill
kill $(ps aux | grep v21_per_race_paper | grep -v grep | awk '{print $1}')
# 2. 新 process 起動
nohup python -u tools/v21_per_race_paper.py > logs/v21_paper_YYYYMMDD.log 2>&1 &
```

---

## 問題② TYB 未取得 (V15 同等スコア)

### 症状
全 fire で `[TYB] not available (disabled or failed)` → TYB features 注入なし → V21 スコアが V15 と同等。

### 原因 (2 段階)

**Layer 1: fetch_tyb_for_race の gate 誤り**

```python
# 修正前 (NG)
from tyb_shadow_fetcher import fetch_tyb_shadow, TYB_SHADOW_ENABLED
if not TYB_SHADOW_ENABLED:  # ← TYB_SHADOW_ENABLED=False で即 return None
    return None
```

`TYB_SHADOW_ENABLED = False` (V15 production 保護用) → V21 paper も同じ gate で止まっていた。

**Layer 2: JRDB 認証 key mismatch**

| ファイル | 使用 key | .env の key |
|---------|---------|------------|
| `tyb_shadow_fetcher._load_env` | `JRDB_USER` | ❌ 未設定 |
| `scrape_jrdb.py` | `JRDB_ID` | ✅ 設定済み (`26037968`) |

`.env` に `JRDB_USER` が無く `JRDB_ID` のみ → `JRDB_USER / JRDB_PASSWORD not set` エラー。

### 修正内容

**1. `tools/v21_per_race_paper.py` — `fetch_tyb_for_race`**
```python
# 修正後
from tyb_shadow_fetcher import fetch_tyb_observe
result = fetch_tyb_observe(race_id, start_time_str)
```
- `fetch_tyb_observe` は `TYB_SHADOW_OBSERVE_MODE=True` + 日付 >= 20260523 でゲートが通る
- V21 paper 専用 (V15 inference に渡さない)

**2. `tools/tyb_shadow_fetcher.py` — `_load_env`**
```python
# 修正後
user = os.environ.get("JRDB_USER", "") or os.environ.get("JRDB_ID", "")
```
- `JRDB_USER` 未設定時に `JRDB_ID` を fallback 参照

---

## TYB 取得条件まとめ

| 条件 | 状態 |
|------|------|
| `TYB_SHADOW_OBSERVE_MODE` | `True` ✅ |
| 日付 >= `TYB_OBSERVE_LAUNCH_DATE` (20260523) | 今日 ✅ |
| `JRDB_ID` / `JRDB_PASSWORD` in `.env` | ✅ 設定済み |
| tyokuzen path `TYB{yymmdd}.lzh` 配信 | 各 R 発走 ~15-20 分前 ✅ |
| V21 fire タイミング | 発走 -17 分 → TYB 配信窓内 ✅ |

→ 修正後は TYB 取得が期待できる。

---

## V21 paper の意味

| TYB 状態 | V21 スコア | 比較価値 |
|---------|-----------|---------|
| **TYB 取得済** | V15+TYB10 features 反映 | ✅ V15 との差異が意味を持つ |
| TYB 未取得 | V15 と同一 features (145) | ❌ 比較の意味ゼロ |

**結論**: TYB なし V21 は V15 と同等。5/23 午後の残り 3 R (15:44/15:53/16:13) で TYB 取得を確認予定。

---

## 現在の状態 (修正後)

| 項目 | 値 |
|------|-----|
| V15 production | PID 28500 ✅ 不変 |
| V21 paper | PID **2797** ✅ (3 タイマー残) |
| 重複 process | なし (PID 1 個のみ) |
| TYB fix | ✅ commit 49577719 |
| 次の fire | 15:44 (新潟12R) |

*V15 production 完全不変 | V21 paper のみ*
