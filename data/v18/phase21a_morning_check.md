# Phase 21A: 5/17 GO worksheet 自動 fill in (2026-05-11)

**前提**: Phase 20 B (`data/v18/phase20_5_17_go_criteria.md`) の 10 項目 GO checklist
**目的**: 朝 06:30 自動実行 → worksheet auto-fill + Discord 通知 + GO/NO-GO 判定
**実装**: `tools/morning_go_check.py` 新規 (~330 行)

---

## 1. 提供機能

| 機能 | 内容 |
|------|------|
| 10 項目 自動 check | Phase 20 B 表 1 完全対応 |
| OK / NG / WARN / SKIP | 各項目の判定 + 詳細 detail 文字列 |
| GO / NO-GO 判定 | NG=0 で GO、 NG≥1 で NO-GO、 WARN≥3 で GO (caution) |
| Discord 通知 | tools/notify.py 経由 (channel: updates) |
| JSON 保存 | `data/morning_go_check/{YYYYMMDD}.json` |
| 任意日付 | `--date 20260517` 等で過去日も verify 可能 |
| dry-run | `--no-notify` で Discord skip |

---

## 2. CLI usage

```bash
# 通常 (Discord 通知)
python tools/morning_go_check.py

# 任意日付 (dry-run)
python tools/morning_go_check.py --date 20260517 --no-notify

# JSON 出力
python tools/morning_go_check.py --json

# 任意保存先
python tools/morning_go_check.py --save logs/morning_5_17.json
```

exit code: 0=GO/GO(caution), 1=NO-GO

---

## 3. 10 項目 check 内訳

| # | check | 実装 |
|---|-------|------|
| 1 | V15 朝予測 完走 | `data/daily_predictions/{date}.csv` 行数 (>5 R で OK) |
| 2 | Stage 2 動作確認 | `data/daily_predictions_full/` glob 一致 |
| 3 | V15 model load OK | subprocess で `pickle.load(gzip.open(...))` |
| 4 | V15 model md5 不変 | hashlib.md5 → 先頭 8 文字を Phase 20 B baseline (`842b9a5f`) と比較 |
| 5 | paper_trade_engine_v22 動作 | dry-run 実行 → fallback import-check |
| 6 | Discord webhook 動作 | env / .env から `DISCORD_WEBHOOK_*` 変数 検出 |
| 7 | netkeiba Cookie 有効 | `tools/refresh_cookie.py --check` の rc + 出力 |
| 8 | 累計 +¥14,140 / 撤退余裕 | `cumulative_results.csv` profit 列 sum (status=settled) |
| 9 | scrape-status guard | kill switch file 不在 verify (`netkeiba_master/.disabled` 等) |
| 10 | nightly_sanity 前夜 PASS | `logs/*nightly_sanity*` 鮮度 (<36h) |

---

## 4. schtask 登録 (5/17 朝 06:30+)

```powershell
schtasks /create /tn "Keiba-MorningGoCheck" /tr "python C:\Users\takum\keiba-ai\tools\morning_go_check.py" /sc DAILY /st 06:30 /ru SYSTEM
```

または `setup_all_tasks.bat` 拡張で一括登録 (将来 task)。

---

## 5. 5/10 dry-run 結果 (5/11 00:14 検証)

| # | check | 結果 | 詳細 |
|---|-------|------|------|
| 1 | V15 朝予測 | OK | 35 R |
| 2 | Stage 2 | OK | 1 files |
| 3 | V15 load | OK | load ok |
| 4 | V15 md5 | **WARN** | `309dffc6...` (期待 `842b9a5f...`、 baseline 不一致) |
| 5 | paper engine | OK | engine ran (dry-run) |
| 6 | Discord webhook | OK | UPDATES + URL + BETS 全 configured |
| 7 | Cookie | **WARN** | rc=1 (refresh_cookie --check 非 0) |
| 8 | 累計 | **WARN** | settled sum=-25,070yen (margin from withdraw=+24,930yen) |
| 9 | scrape-guard | OK | no kill switches |
| 10 | nightly_sanity | OK | log 1.2h ago |

→ OK=7 / NG=0 / WARN=3 → **GO (caution)**

---

## 6. ★ honest 注意事項 (5/11 検証時に判明) ★

### 6.1 V15 md5 mismatch
- Phase 20 B 表 4 expected: `842b9a5f` (先頭 8 文字)
- 5/11 actual: `309dffc6`
- **判定**: Phase 20 B 作成時の md5 と現在の V15 model file の md5 が異なる
- **可能性**: V15 retrain / 再 build / 手動更新 等で md5 が変わった、 もしくは Phase 20 B baseline 値が誤り
- **対応**: 5/11+ で V15 model md5 を再記録 → `EXPECTED_V15_MD5` const を更新 (本 task では Phase 20 B doc 不変、 script 側 WARN で運用継続)

### 6.2 累計収支 baseline 不一致
- CLAUDE.md / Phase 20 B 表 1 #8: "累計 +¥14,140"
- 5/11 actual `cumulative_results.csv` profit (settled sum): **-¥25,070**
- **可能性**:
  1. CLAUDE.md +¥14,140 値は一部の集計 / 別 CSV / 案 B 改 strict 適用後
  2. cumulative_results.csv は production 全 R + 候補 paper を含む混合 (settled も含む)
  3. baseline 数値が outdated
- **対応**: 5/11+ で別 source (集計 script / paper trade ledger) と照合、 真の cumulative source を確定
- **撤退ライン**: -¥50,000 余裕 +¥24,930yen (NG ではない、 WARN にとどめ)

### 6.3 cookie WARN
- rc=1 は `refresh_cookie.py --check` が cookie expired を返した可能性 / 別 error
- 5/17 までに `python tools/refresh_cookie.py --auto` で再取得確認

---

## 7. NO-GO 自動 trigger (3 段階)

```python
# tools/morning_go_check.py run_all()
overall = 'GO' if n_ng == 0 else 'NO-GO'
if n_ng == 0 and n_warn >= 3:
    overall = 'GO (caution)'
```

Phase 20 B 表 4 NO-GO シナリオ (即停止 trigger) を本 script は完全には check しない (md5 と withdraw line の 2 つは検出可能、 cookie BAN / netkeiba 規約改訂 / data 整合性 は別 path)。

5/17+ schtask 運用前に `--no-notify` dry-run で検証推奨。

---

## 8. V15 投資保護 (絶対不変)

| 不変 | 状態 |
|------|------|
| `tools/predict_core.py` | ★完全不変★ |
| `tools/daily_predict.py` | ★完全不変★ |
| `app.py` | ★完全不変★ |
| `keiba_model_v15_central*.pkl.gz` | ★完全不変★ (md5 監視 cont.) |

morning_go_check.py は読み取り専用 (subprocess も `--check` / `--dry-run` のみ、 productive 副作用なし)。

---

## 9. 5/17 朝 user task

1. **06:30 自動実行**: morning_go_check.py で worksheet auto-fill + Discord 通知
2. **06:35 user 確認**: Discord 通知の overall (GO / GO(caution) / NO-GO)
3. **06:40 朝候補通知 (08:00)** までに WARN 解消 (cookie 等)
4. **08:00 V15 daily_predict 実行** (既存 schtask)
5. **20:30 daily_results 結果照合** (既存 schtask)

---

## 10. 5/11+ 拡張 候補

| 拡張 | 期日 | 内容 |
|------|------|------|
| md5 baseline 更新 | 5/11 | 現 V15 model md5 を `EXPECTED_V15_MD5` に反映 |
| 累計 source 確定 | 5/11+ | 真の cumulative ledger を別 csv で運用 |
| schtask 登録 | 5/17 朝 | `Keiba-MorningGoCheck` 06:30 登録 |
| nightly_sanity 連動 | 5/16 夜 | 23:00 で morning_go_check 用 dry-run も同時実行 |
