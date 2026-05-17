# Live Orchestrator 使用 guide

## 0. 役割
P0-5 -15 min 再計算 main orchestrator (5/18+ admin schtask 経由 fire)。
朝 8:30 (SAT/SUN) に発火し、 daily_predictions に並ぶ各 race に対して
-20 min (live_data_fetcher) → -15 min (recalc_15min) → Discord 通知の
shadow eval pipeline を順次実行する。

★ V15 production の投票判断 / .pkl.gz / cumulative_results.csv は完全不変 ★

## 1. file 構成
- `tools/live_orchestrator.bat` — schtask `/TR` ターゲット (Windows admin 登録)
- `tools/live_orchestrator.ps1` — PowerShell 代替 (Set-Location + python -u)
- `tools/live_orchestrator_main.py` — Python entry (本体)
- `data/live_orchestrator_log/{date}.log` — JSON Lines log

## 2. 動作 flow
1. SAT/SUN 08:30 fire (5/18+ admin schtask)
2. G1 day blocklist check (5/17 → 強制 skip)
3. `daily_predictions/{date}.csv` load (file 無 → no_races で early return)
4. race ごとに:
   - `recalc_15min.is_strategy_7c_excluded` で 京都 / 条件 X (G1/G2/G3/L/OP 除外) skip
   - `live_data_fetcher.fetch_pre_features(mock, dry_run)` — -20 min 取得
   - `recalc_15min.run_recalc(mock, dry_run)` — -15 min 再計算 + Discord notify delegate
5. results 集計 (ok / skip / fail) を log + stdout 出力

exit code:
- `0` = 全 race ok / skip のみ
- `1` = 1 件以上 fail (race の半分未満)
- `2` = race の半分以上 fail = critical

## 3. 5/17-5/23 (★ paper eval 開始前 ★)
- bat / ps1 は `--mock --dry-run` 強制
- 実 fetch 0、 実 Discord 発火 0 (recalc_15min は [DRY-RUN] でログ出力のみ)
- safety: G1 day blocklist `20260517` active (5/17 fire 時は orchestrator 早期終了)

## 4. 5/24+ paper eval 開始
- user が `tools/live_orchestrator.bat` の python 行から `--mock --dry-run` を手動解除
- 解除後: live fetch ON、 Discord `#updates` 通知 ON
- 解除タイミングは user 判断 (本 guide では強制しない)

## 5. dry-run コマンド (★ schtask 登録前の手動 verify ★)
```powershell
# G1 day blocked verify
python tools\live_orchestrator_main.py --date 20260517 --mock --dry-run
# 期待: "[INFO] 20260517 = G1 day blocked, skipping"

# 通常 SAT/SUN flow verify (5/16 で実 races 24+11skip 確認済)
python tools\live_orchestrator_main.py --date 20260516 --mock --dry-run
# 期待: "[OK] orchestrator done: {'ok': N, 'skip': M, 'fail': 0}"

# daily_predictions 未生成日 verify
python tools\live_orchestrator_main.py --date 20260524 --mock --dry-run
# 期待: "[WARN] 20260524 = no races (daily_predictions not found)"
```

## 6. trouble shooting
- log 確認: `data\live_orchestrator_log\{date}.log` (JSON Lines)
- stdout: `data\live_orchestrator_log\stdout.log` (bat / ps1 経由時)
- 全 race fail (exit 2): `schtasks /Delete /TN Keiba-LiveOrchestrator-15min /F` で一旦停止 → log 解析
- import_fail: tools/ の sys.path 注入は orchestrator 内で実施済、 venv 確認

## 7. V15 production 不変保証
- V15 .pkl.gz / cumulative_results.csv / predict_core.py / daily_predict.py / race_auto_notify.py / app.py — 一切変更なし
- orchestrator は shadow eval only、 投票判断は朝 8:00 race_auto_notify の V15 ranking が確定
- recalc_15min の出力は `data/recalc_15min/{date}/{race_id}.json` (V15 投票には未使用)
