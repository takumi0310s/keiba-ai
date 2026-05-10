# Phase 13 — AI 展開予測 PoC

**date**: 2026-05-10
**target**: netkeiba マスターコース AI 展開予測 features 7 件

---

## 1. 取得対象 features (7)

| feature | type | range | 用途 |
|---------|------|-------|------|
| `master_pace_pred` | int | 0-2 | AI ペース予測 (slow/medium/high) |
| `master_pred_winner_score` | float | 0-100 | AI 1着馬 score |
| `master_pred_first3f_avg` | float | 25-45 | AI 予測前半 3F 平均 (秒) |
| `master_pred_last3f_avg` | float | 25-45 | AI 予測後半 3F 平均 (秒) |
| `master_pred_finish_time` | float | 60-200 | AI 予測走破タイム (秒) |
| `master_horse_aitenkai_score` | float | 0-100 | 当該馬 展開適性 score |
| `master_horse_pred_pos` | int | 1-18 | AI 予測 4 角通過順 |

---

## 2. URL 構造 (確認済)

```
GET https://race.sp.netkeiba.com/race/compatibility.html?race_id=202605020611
```

- `race_id` = 12 桁 (年 4 + 場 2 + 開催 2 + 日次 2 + R 2)
- 認証: Cookie (`nkauth`, `_nk_session`、 master 加入で master 機能 unlock)
- 鮮度: 朝発走 30 分前頃 確定、 当日 9:00 で取得可能想定

---

## 3. parser 設計

### selector 候補 (best-effort、 実 DOM 確定は Phase 13.5)

| field | 想定 selector |
|-------|--------------|
| ペース予測 | `.RaceData_PacePred`, `.pace_pred`, `#pace_prediction` |
| 走破タイム | `.RaceData_FinishTime`, `.pred_finish_time` |
| 前半 3F | `.first3f`, `.lap_first` (馬別 row 内) |
| 後半 3F | `.last3f`, `.lap_last` (馬別 row 内) |
| 馬別 score | `.score`, `.ai_score`, `.aitenkai_score` |
| 通過順 | `.pred_pos`, `.pass_pos`, `.pos_pred` |
| 馬番識別 | `.umaban`, `.horse_num` |

### parse 戦略

1. `compatibility.html` 単発 fetch で race-level + 全馬 data 一括取得
2. table tr ループで `.umaban` → 当該馬 row 抽出
3. 取れない field は `PHASE13_DEFAULTS` で fill
4. `fetch_status['tenkai'] = 'ok'/'fail'/'default'` で品質追跡

### 取得タイミング

| timing | 用途 |
|--------|------|
| 当日 朝 09:00 | daily fetch (全 R 一括) |
| 発走 30 min 前 | realtime fetch (最終確定) |

→ `daily_predict.py` の Phase A1/A2 と同 cadence。

---

## 4. PoC 動作確認

```bash
$ python tools/netkeiba_master_scraper.py --list
Phase 13 全 25 features:
  B. AI 展開予測 (7): ['master_pace_pred', ..., 'master_horse_pred_pos']
  C1. AI 波乱度 (3): [...]
  C2. 個別ラップ (10): [...]
  C3. トラックバイアス (5): [...]

$ python tools/netkeiba_master_scraper.py --status
disabled: False
cookie loaded: True
```

→ skeleton 動作 OK、 cookie 読込 OK、 kill switch 機能 OK。

実 fetch は Phase 13.5 で実 DOM 検証 + selector 確定後。

---

## 5. risk 残り

| risk | mitigation |
|------|------------|
| selector 名 推定 hit せず | PHASE13_DEFAULTS で fill、 回帰なし |
| fetch fail (rate / 認証 / network) | session 再生成、 default fill |
| netkeiba 規約改訂 | KILL_SWITCH (`data/netkeiba_master/.disabled`) で即停止 |
| Cookie 期限切れ | `tools/refresh_cookie.py` で更新 (既存 system) |

---

## 6. 次 step (Phase 13.5、 5/11+)

1. ★ user が master account ログイン状態で 1 R 手動 fetch (browser DevTools で HTML 取得) ★
2. 取得 HTML 内 selector 検証 → parser stub の selector を真値で更新
3. 1 R PoC fetch → 7 features 値妥当性検証
4. 当日全 R loop fetch → cache 保存
