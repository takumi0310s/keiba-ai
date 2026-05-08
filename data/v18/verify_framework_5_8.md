# Session #47 D: 5/10 朝 結果照合 framework (2026-05-08)

## 1. 目的

5/9 (土) 全 R 予測 (Session #47 C で save) と netkeiba 実結果を照合し
metrics + Discord 通知を生成。

## 2. tool

`tools/verify_all_5_10.py`

```bash
python tools/verify_all_5_10.py                 # 5/9 全 R 照合
python tools/verify_all_5_10.py --dry-run       # 動作確認 (fetch skip)
python tools/verify_all_5_10.py --no-discord    # Discord skip
```

## 3. 入出力

### 入力
- `data/v18/predictions_5_9_all.json` (Session #47 C 出力)

### 出力
- `data/v18/verification_5_10.json` — 数値結果
- `data/v18/verification_5_10.md` — Markdown verdict
- Discord 通知 (DISCORD_WEBHOOK_UPDATES)

## 4. metrics

| metric | 計算 |
|--------|------|
| top1 hit rate | V15 top1 が 1着 と一致した R 数 / 総 R |
| 複勝 hit rate | V15 top1 が 3着以内 だった R 数 / 総 R |
| top3 anywhere rate | V15 top3 のいずれか が 3着以内 |
| クラス別 (G1/G2/G3/OP/3勝/2勝/1勝/未勝利/新馬) | 同上 を grade ごと |

## 5. 結果取得 strategy (3 段階 fallback)

1. `daily_results.fetch_race_result()` 流用
2. `check_results.fetch_result()` 流用
3. 直接 netkeiba scrape (最低限)

→ 1 が動けば 1 で完結。 全部失敗時は metric=0 で出力。

## 6. dry-run 動作確認 (5/8 19:30)

```
[19:30:04] === Dry-run mode (sample) ===
[19:30:04] Verify 20260509 predictions
[19:30:04] Loaded 1 predictions
[19:30:04] Dry-run: skip actual fetch, simulate empty results
[19:30:04] Metrics: top1=0.0%, 複勝=0.0%, matched=0
[19:30:04] Saved: data/v18/verification_5_10.json
[19:30:04] Saved: data/v18/verification_5_10.md
```

→ tool は正常動作。 5/10 朝に full run。

## 7. 5/10 朝 運用 step

```bash
# 1. 5/10 06:00 以降 (全 R 結果 公開後)
python tools/verify_all_5_10.py

# 2. 結果確認
cat data/v18/verification_5_10.md

# 3. Discord 確認 (#アップデート チャンネル)

# 4. 投資 R (12R 1勝クラス) の verdict 別途確認
#   - V15 案B改 で投票した R の hit / miss
#   - 累計収支 +13,530 円 → 5/10 後の数値 update
```

## 8. 採用判定 (B AUC test 結果と統合)

5/10 verdict + Session #47 B AUC delta:

| 結果 | judgment |
|------|----------|
| AUC +0.002+ かつ top1 hit 30%+ かつ クラス別 monotonic | V20 候補 (Phase 3 追加) |
| AUC +0.002+ かつ top1 hit < 30% | 単独 sample 不足、 5/16, 5/17 で再計測 |
| AUC < +0.001 | 棚卸しのみ、 V15 unchanged |

## 9. リーク監査 (再確認)

- 結果取得は **レース完了後**
- 予測 fetch は **レース前**
- 両者 timestamp 別 → リークなし
- predictions_5_9_all.json に v15_md5 記録 → 再現性保証

## 10. 関連 tool

- `tools/predict_all_5_9.py` (C で実装)
- `tools/daily_results.py` (production、 read-only 利用)
- `tools/check_results.py` (production、 read-only 利用)
- `tools/notify_done.py` (Discord 通知 helper)
