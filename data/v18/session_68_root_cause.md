# Session #68 B: root cause 特定

**作成**: 2026-05-09 16:53 (Session #68、 dev/two-stage)
**対象**: Stage 2 全 R 失敗 (17/17、 100%) の真因

---

## 現在 (16:53) の netkeiba 反応

3 endpoint × cookie 有/無 で test:

```
race.netkeiba.com/race/shutuba.html?race_id=202604010312  → status 400, len 0  (User-Agent only)
race.netkeiba.com/race/shutuba.html?race_id=202604010312  → status 400, len 0  (Cookie len=1817)
race.netkeiba.com/race/shutuba.html?race_id=202604010312  → status 400, len 0  (full Chrome headers)
race.netkeiba.com/race/result.html?race_id=202604010312   → status 400, len 0
db.netkeiba.com/race/202604010312/                        → status 400, len 0
race.netkeiba.com/race/oikiri.html?race_id=202604010312   → status 400, len 0
```

**全 endpoint 全 header で HTTP 400 Bad Request、 body 空。**

## Session #62 / #63 の確定事項と整合

CLAUDE.md および Session #62-#63 commit log:

> Session #63 B: 静止画 DL (★ netkeiba 全 page server block 確定 ★)
> Session #62 D: 5/9 重賞動画 v2 全再 DL = 0/3 (server 400 依然)

→ 5/8 (Session #62) で server 400 block 既に発生、 5/9 (Session #63) でも継続確定。
   Stage 2 schtasks (5/9 中 30 分毎 fire) は **netkeiba block 環境下で 100% 失敗** が確定。

## 失敗の連鎖

```
netkeiba HTTP 400 (server block)
  ↓
predict_core.parse_shutuba()
  - resp.text == ''
  - soup.select("tr.HorseList") == []
  - return (race_name="レース", horses=[], horse_ids=[], race_info={...})
  ↓
predict_one_race.py:
  - if not horses: print('[NG] 出馬表取得失敗') / return None
  ↓
stage2_predict.py predict_stage2():
  - ret is None → return {"error": "predict_one_race returned None"}
  ↓
Discord 通知:
  - error: `predict_one_race returned None`
  - 朝予測 top3 のみ表示 (Stage 1 値 fallback として表示)
```

## 根本原因

**netkeiba 側の server block (Cloudflare / WAF / IP rate-limit)。 client 側修正で解決しない。**

ただし stage2_predict.py 側の挙動は改善余地あり:
1. **診断情報不足**: `predict_one_race returned None` のみで原因 (HTTP 400 か rate-limit か parse error か) 不明
2. **Discord 通知の有用性低い**: Stage 1 top3 表示はあるが、 「Stage 2 失敗 = Stage 1 を信用」 が明示されていない
3. **失敗時 cache 書込み**: 次 fire で dedup skip → 復旧後も再試行されない
4. **fallback 動作なし**: block 一時解除時 1 R でも成功すればその場で通知が望ましい

## 修復方針 (C で実装)

| 修復項目 | 対応 |
|---|---|
| 診断情報強化 | predict_one_race の各 step に `[stage2-trace]` log + parse_shutuba 戻り時に status_code / len 情報 propagate |
| fallback 通知 | Stage 2 失敗 → 「Stage 1 結果を採用」 を明示する body |
| cache の挙動 | 失敗時は cache 書込みしない (= 次 fire で再試行可) |
| netkeiba 接続診断 | 起動時 1 回 `_probe_netkeiba()` で server reachable か確認、 block 検知時は早期 skip + 1 通通知 |
| stage_compare 拡張 (D) | 失敗 R は朝予測のみで verdict、 hit rate 集計を 「朝のみ」 「Stage 2 込み」 「成功 R のみ」 で 3 系統 |

## 5/16 V18 trial への含意

netkeiba block が 5/16 までに解除されない可能性あり。
Stage 2 system は **block 解除前提なし** で動くこと (= Stage 1 fallback で必ず通知が機能する) を担保する。
