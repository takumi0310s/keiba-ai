# Session #65 B: tools/stage2_predict.py 実装

## 1. 名称変更 (重要)

当初 `tools/pre_race_predict.py` を新規作成予定だったが、 既存 file (前夜予測 = daily_predict ラッパー、 別目的) と衝突。 衝突回避のため:

- 新規 file: `tools/stage2_predict.py`
- 既存 `tools/pre_race_predict.py` 不変 (前夜予測モード継続)
- output cache / kill-switch 名は `pre_race_predict_*` を維持 (Session #65 A doc との整合)

## 2. 構成

| 部分 | 内容 |
|------|------|
| RACE_START_TIMES | 5/9 全 34 R 静的 schedule (Session #65 A doc 由来) |
| `races_in_next_window(60)` | 現在時刻 +60 min 内の race_id list |
| `load_cache()` / `save_cache()` | dedup state JSON 永続化 |
| `predict_stage2(race_id)` | predict_one_race.predict_one_race() を呼ぶ |
| `build_message()` | 朝 (Stage 1) vs Stage 2 比較 + Discord 用 markdown |
| `cmd_check_next_1h()` | watchdog 用 entry — 30 分毎 fire 想定 |

## 3. CLI

```bash
python tools/stage2_predict.py --race-id 202604010312     # 単一 R 予測
python tools/stage2_predict.py --check-next-1h            # 次 1h 内 R を順次予測
python tools/stage2_predict.py --no-discord               # dry-run
python tools/stage2_predict.py --race-id ... --force      # cache 無視
```

## 4. 動作確認 (5/9 13:05 dry-run)

### 4-1. schedule logic OK
```
now: 13:05
next 1h candidates (5):
  202608030507 -> 13:30  (京都 R7)
  202604010308 -> 13:30  (新潟 R8)
  202605020507 -> 13:45  (東京 R7)
  202608030508 -> 14:00  (京都 R8)
  202604010309 -> 14:00  (新潟 R9)
```

### 4-2. Stage 2 予測 fallback 確認 (新潟 12R で test)
```
=== Stage 2 predict 202604010312 (新潟 R12) ===
出馬表取得 ... [NG] 出馬表取得失敗
Stage 2 予測 失敗 (error: predict_one_race returned None)
```

→ predict_one_race の出馬表 fetch (netkeiba) が時刻早すぎ (発走 16:10 まで 3h+ あり、 当日体重 未公開) で fail。 build_message() が graceful fallback で 朝予測 + 失敗理由を表示する分岐に入り、 Discord 通知は問題なし。

実 1h 前 timing (例: 14:30 の 京都新聞杯) では出馬表 + 当日体重 (70min 前公開) 取得可能性高い。

## 5. fallback design (失敗時挙動)

| 失敗種別 | 挙動 |
|----------|------|
| 出馬表取得 NG | message 内に `Stage 2 予測 失敗` を表示、 朝予測 top3 のみ通知 |
| 当日体重 未公開 | predict_core が build_features で 0 fill (既存挙動)、 Stage 2 予測 自体は成功 |
| オッズ未確定 | predict_one_race が odds_dict={} で続行、 オッズ依存 features は デフォルト |
| Discord NG | console + json 出力のみ、 cache 記録は通常通り |

## 6. dedup 機構

- `data/v18/pre_race_predict_cache_5_9.json` に `{race_id: ISO timestamp}` 記録
- 既記録 race_id は `[skip dedup]` で skip
- watchdog 30 分毎 fire でも 1 R 1 通保証
- `--force` で cache 無視可

## 7. kill-switch (Session #64 patten 踏襲)

- `data/v18/pre_race_predict.kill` を touch すれば即 no-op exit
- main() 冒頭 + cmd_check_next_1h() 冒頭で check
- Admin 不要、 暴走時即停止可能

## 8. V15 投資保護 確認

- predict_core.py / V15 model file 触らない (関数のみ呼ぶ)
- daily_predict.py / race_auto_notify.py 一切 呼ばない (Session #64 spam 再発防止)
- 既存 schtasks 49 件 不変
- Stage 2 予測 は学習用、 message 内に「投票推奨ではない」 明記
- 5/9 投票方針 (新潟 12R ¥700) 不変

## 9. 出力 file naming

```
data/v18/pre_race_predict_5_9_R{race_num}_{course}_{race_id}.json
```

例: `pre_race_predict_5_9_R12_新潟_202604010312.json`
