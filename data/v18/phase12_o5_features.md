# Phase 12 A: オッズ拡張 features (4 件) 実装 (5/10)

> Session #87 Phase 12 A 領域 (2026-05-10 18:00+)
> 出力: tools/predict_core_v18.py の ODDS_EXPANSION_FEATURES (4 件)

---

## 1. user 指示 vs 実 JV-Link record mapping

| user 指示 | 実 JV-Link record | 用途 |
|----------|-------------------|------|
| 「O5 オッズ拡張」 | O1 (単複) + O2 (馬連) + O5 (三連複) | 4 features 抽出 |

★ 注: JV-Link 仕様上 O5 は **三連複オッズ専用 record**。 「オッズ拡張」 として複数 record を集約する設計に変更。 ★

---

## 2. 実装 4 features

| feature | source record | default |
|---------|--------------|---------|
| jv_tansho_odds_open | O1 (単複オッズ、 始値) | 10.0 |
| jv_fukusho_low_open | O1 (複勝下限、 始値) | 2.0 |
| jv_umaren_top_odds | O2 (馬連 1 番人気) | 30.0 |
| ★ jv_trio_top_odds ★ | O5 (三連複 1 番人気、 V20 投資判断 EV 計算 base) | 100.0 |

---

## 3. live activation 設計

### 3.1 Phase 12 (本日)
- skeleton 実装、 default fill のみ
- `_is_jvlink_available()` で `data/jvlink/{O1, O2, O5}/` 存在 check
- 不在時 → `V18_PHASE12_DEFAULTS` 返却

### 3.2 5/24+ Phase 3 後半
- 32-bit Python venv `C:\Users\takum\jvlink-venv\` 経由 fetch
- tools/jvlink_fetcher_v2.py で O1/O2/O5 records parse
- `data/jvlink/<datatype>/<race_id>_parsed.csv` 経由 読み込み

---

## 4. 動作 test

```
$ python tools/predict_core_v18.py
[predict_core_v18] Phase 12 features: 17 件
[predict_core_v18] JV-Link backfill 利用可: False
[predict_core_v18] dummy fetch keys: 17 件
[predict_core_v18] OK: 全 17 features default 取得 成功
  A. オッズ拡張 (4): ['jv_tansho_odds_open', 'jv_fukusho_low_open', 'jv_umaren_top_odds', 'jv_trio_top_odds']
```

---

## 5. V15 投資保護

✅ tools/predict_core.py 不変 (V15 production)
✅ tools/daily_predict.py 不変
✅ V15 model file 不変
✅ tools/predict_core_v18.py = 新規 file (V15 と完全分離)
✅ skeleton のみ、 live data fetch なし
✅ schtask 不変

---

## 6. 結論

✅ A1: O1/O2/O5 records 由来 4 features 定義
✅ A2: skeleton 実装 (default fill、 17 features assertion pass)
✅ A3: live activation は 5/24+ JV-Link backfill 後
✅ A4: V15 完全保護
