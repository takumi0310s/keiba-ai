# outcome dashboard 動作確認 (2026-05-16)

## 0. 結論

- boot test: ✅ (port 8504 で 「You can now view your Streamlit app」確認、 15 秒 後 kill)
- 4-tab syntax: ✅ (py_compile pass、 4 tab すべて静的 OK)
- baseline 真値反映: ✅ (adopted_value 101.33% / PnL +¥5,240 / n=563 すべて表示 path 確認)
- V15 app.py 干渉なし: ✅ (app.py は port hardcode なし = streamlit default 8501、 dashboard は 8502 hardcode、 完全分離)

---

## 1. boot test 結果

### command
```powershell
Start-Process python -ArgumentList "-m","streamlit","run","dashboard/outcome_dashboard.py",
  "--server.port","8504","--server.headless","true","--server.runOnSave","false"
# 15 秒 後 Stop-Process で kill
```

note: 推奨 port 8503 は 既に使用中だったため 8504 に変更。 dashboard 自体の logic には影響なし (hardcode は 8502)。

### output (stdout)
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8504
Network URL: http://192.168.3.8:8504
External URL: http://126.147.102.14:8504
```

### syntax error: なし
- `python -c "import py_compile; py_compile.compile('dashboard/outcome_dashboard.py', doraise=True)"` → OK
- stderr 空、 起動成功

### 既知の動作
- `time.sleep(30)` + `st.rerun()` の 30 秒 auto refresh (line 214-219) は kill 前に 1 回も発火しない (15 秒で kill)、 boot test では問題なし。
- 実運用で long-running 起動した場合の rerun 動作は本タスク範囲外。

---

## 2. baseline_v15.json 反映確認

### data source
- file: `data/task_outcomes/baseline_v15.json`
- load: `load_baseline()` (line 27-31、 `@st.cache_data(ttl=30)` で 30 秒 cache)

### 各 値の dashboard 表示 path

| 値 | baseline_v15.json key | dashboard 表示箇所 | line |
|---|---|---|---|
| adopted_value 101.33% | metrics.actual_roi.candidate_values.raw_cumulative_5_16_evening.value | sidebar st.metric("raw ROI (5/16 ev)") + tab3 ROI trend base_roi + tab4 ROI gauge | 65-68, 154-160, 188-189 |
| adopted_pnl_yen 5240 | metrics.cumulative_pnl_yen.value_candidates.cumulative_csv_5_16_plus_5240 | tab4 cum pnl gauge cur_pnl | 191-193, 206-209 |
| WF AUC 0.8939 | metrics.wf_auc.value | sidebar st.metric("WF AUC") + tab4 AUC gauge | 62-64, 185, 196-199 |
| hit_rate_top3 0.5382 | metrics.hit_rate_top3.value | sidebar st.metric("top3 hit rate") + tab3 hit3 trend | 69-71, 160, 175 |
| n_settled 563 | metrics.n_settled | sidebar st.write("n settled") | 72 |
| honest_notes (9 件) | honest_notes[] | sidebar st.caption loop | 74-76 |

### 結論

- ✅ adopted_value 101.33% は sidebar + tab3 + tab4 で 表示確認 (3 箇所)
- ✅ adopted_pnl_yen +¥5,240 は tab4 cum pnl gauge で 表示確認
- ✅ adopted_n_races 563 は sidebar に出力
- ✅ honest_notes 9 件 (★ で markdown された 重要 caveat 含む) すべて sidebar に出力

---

## 3. 4-tab syntax check

| tab | 内容 | data source | line | 静的 syntax |
|---|---|---|---|---|
| tab1 timeline | outcome list を dataframe (task_id / phase / before_roi / after_roi / delta_roi / p_value / significant / status) | load_outcomes() で `data/task_outcomes/*.json` (baseline 除外) を全 load | 87-107 | ✅ |
| tab2 task cards | 各 outcome を expander、 4 cols で ROI / hit3 / n / p-value + expected + notes | load_outcomes() | 109-146 | ✅ |
| tab3 cumulative graph | baseline + outcomes を 時系列 sort、 ROI と hit3 を line_chart で 2 col 並列 | baseline + outcomes 合成 | 148-179 | ✅ |
| tab4 limit gauge | AUC / ROI / cum pnl を st.metric、 target との delta 表示 (target: AUC 0.9020+ / ROI 110%+ / cum pnl +50K) | baseline のみ | 181-209 | ✅ |

### tab1 timeline 詳細
- outcomes が空の場合: `st.info("no task record yet. baseline_v15.json only.")` 表示 (line 107) ← 現状はこの分岐 (task_outcomes/ に baseline 以外なし)
- 各 row: implemented_at desc sort
- 各 column 値は `(o.get("statistical_test") or {})` で None-safe

### tab2 task cards 詳細
- delta が None でも `delta=None` で st.metric は許容
- statistical_test.available が False なら reason caption 表示 (line 137-138)
- expected が dict なら description フィールド、 raw fallback (line 140-141)

### tab3 cumulative graph 詳細
- baseline row を 先頭に追加 (raw_cumulative_5_16_evening を base_roi として、 calculation_date '2026-05-16' を implemented_at に)
- 各 outcome の after.roi / after.hit_rate_top3 を 追加
- df を implemented_at asc sort で line_chart
- 現状: baseline 1 行のみ → 線にならない (point only) が syntax error ではない

### tab4 limit gauge 詳細
- AUC: target 0.9020 - cur_auc を delta 表示 (現状 0.8939 → 0.0081 不足を gauge 表示)
- ROI: target 110 - cur_roi を delta 表示 (現状 101.33 → 8.67pt 不足)
- cum pnl: target 50000 - cur_pnl を delta 表示 (現状 5240 → 44,760 円 不足)
- すべて None-safe (`if cur_xxx is not None`)

---

## 4. V15 干渉確認

### port 設定

| component | port | 設定箇所 |
|---|---|---|
| V15 production app.py | 8501 (streamlit default) | hardcode なし、 .streamlit/config.toml もなし、 streamlit default 採用 |
| outcome_dashboard.py | 8502 | docstring に明記 (line 4)、 caption に "V15 dashboard (port 8501)" 言及 (line 6, 211) |
| 今回の boot test | 8504 | 8503 が他で使用中だったため変更、 dashboard 本体に影響なし |

### 干渉判定
- ✅ port 競合なし (8501 vs 8502 vs 8504)
- ✅ predict_core / daily_predict / race_auto_notify / app.py は 一切 import されていない (`@st.cache_data` で json read のみ)
- ✅ baseline_v15.json は read-only、 V15 .pkl.gz / cumulative_results.csv に書き込まない
- ✅ V15 production state 完全 不変

### 同時起動可能性 (code 上)
- streamlit は port さえ違えば 同時起動可能 (子プロセスは独立)
- app.py を 8501、 outcome_dashboard.py を 8502 で同時起動しても conflict なし
- 但し ★ 本タスクでは実行しない ★ (long-running 起動を放置しない 絶対遵守)

---

## 5. 改善 提案 (任意)

### 5-1. 30 秒 auto refresh の問題
line 214-219 の `time.sleep(30) + st.rerun()` は **streamlit の意図しない使い方**:
- 通常 streamlit は user interaction で rerun する
- sleep を入れると thread が block され、 user の click 等 操作も 30 秒待ち
- 推奨: `st.button("refresh")` で manual refresh、 または `streamlit_autorefresh` lib を使用

★ 但し production impact = 0 (V15 と完全独立) なので 緊急修正は不要 ★

### 5-2. tab3 で baseline 1 行のみのときの線描画
現状 line_chart は 1 point の場合 線描画されない (point のみ表示)。
outcome が 1 件以上 増えてから 意味のある grafh になる。 修正不要。

### 5-3. limit gauge の target 値 hardcode
line 183: `target: AUC 0.9020+ / ROI 110%+ / cum pnl +50K yen`
将来 target 変更時は code 修正が必要。 baseline_v15.json に `targets` field を追加して driven 化が望ましい (将来 task)。
