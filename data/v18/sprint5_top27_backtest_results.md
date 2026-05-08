# Session #51 B: AUDIT-1 Top 27 一括 backtest 結果

**作成**: 2026-05-08 (Session #51 B)
**tool**: tools/audit_top27_backtest.py
**JSON 出力**: data/v18/sprint5_backtest_metrics.json

---

## 0. 実装サマリ

- V15 cache (data/_v15_optuna_df_cache.pkl.gz、 145 features、 527,280 rows) を base
- 時系列 split: 2020-2023 train (189K rows)、 2024 valid (47K rows)
- LGB 軽量 (num_boost_round=200、 early_stopping=20)
- 単一 feature 追加 → AUC delta
- 18 件 即実装可能 features を 1 セッション (約 5 分 single thread) で 全検証

---

## 1. 結果テーブル (delta 降順)

| 順 | # | feature | coverage 2024 | AUC base | AUC new | delta | status | コメント |
|----|---|---------|--------------|---------|---------|-------|--------|--------|
| 1 | 18 | jrdb_sed_time | 100.0% | 0.8681 | **1.0000** | **+0.13188** | **★ LEAK 確定** | SED は post-race (finish/time_sec/abnormal) |
| 2 | 22 | race_review_score | 89.4% | 0.8681 | **0.9981** | **+0.12997** | **★ LEAK 確定** | review_score は post-race (v12.1 不採用 確認) |
| 3 | 11 | training_times_rank | 0.0% | 0.8681 | 0.8681 | +0.00000 | text encode 失敗 | 'A/B/C/D' label encode 必要 |
| 4 | 5 | ai_opinion_pace | 0.0% | 0.8681 | 0.8681 | +0.00000 | text encode 失敗 | 'H/M/S' label encode + 2024-2025 のみ |
| 5 | 9 | ai_position_pct | 46.0% | 0.8681 | 0.8681 | +0.00000 | 効果なし | left/top pct は target 無相関 |
| 6 | 8 | jrdb_jo_bb | 9.6% | 0.8681 | 0.8680 | -0.00011 | 低カバレッジ | gaisha_bb 9.6% で 不十分 |
| 7 | 6 | jrdb_cha_oikiri | 97.1% | 0.8681 | 0.8679 | -0.00017 | 効果ほぼなし | wood_best_4f 等 既存と重複 |
| 8 | 17 | jrdb_tyb_live | 100.0% | 0.8681 | 0.8678 | -0.00027 | live のみ | bagu_change/cancel_flag は Pattern B |
| 9 | 10 | jrdb_cyb_train | 100.0% | 0.8681 | 0.8677 | -0.00039 | 効果なし | training_intensity 等 既存重複 |
| 10 | 12 | race_analysis_score | 18.7% | 0.8681 | 0.8677 | -0.00040 | 低カバレッジ | 2025 中心 |
| 11 | 7 | speed_index_dist_course | 91.6% | 0.8681 | 0.8677 | -0.00040 | 既存重複 | index_max/avg5 既組込 |
| 12 | 21 | upset_level | 100.0% | 0.8681 | 0.8677 | -0.00041 | 効果なし | 単一値が 多い |
| 13 | 26 | jrdb_jo_odds | 100.0% | 0.8681 | 0.8676 | -0.00049 | 朝オッズ近似 | odds_log 既組込 |
| 14 | 23 | stable_comment_score | 35.0% | 0.8681 | 0.8675 | -0.00059 | 低カバレッジ | カバレッジ 60% 改善 必要 |
| 15 | 4 | jrdb_kka_seiseki | 0.0% | 0.8681 | 0.8681 | +0.00000 | **★ KKA parser 不全** | 9 cols 全て NaN (parser broken) |
| 16 | 28 | data_analysis_count | — | — | — | — | error | text-only CSV、 numeric 抽出不可 |
| 17 | 19 | jrdb_kyi_marks | — | — | — | — | error | jrdb_kyi.csv が Shift_JIS で col 名 化け |
| 18 | 24 | jrdb_kz_leading | — | — | — | — | error | jrdb_kz は master 表 (race_id なし) |

---

## 2. 重大発見 (★)

### 2-1. LEAK 確定 2 件 → V20 LEAK list 追加

| # | feature | AUC | コメント |
|---|---------|-----|--------|
| 18 | jrdb_sed (auto-extract numeric cols) | 1.0000 | SED は finish/time_sec/abnormal を含む完全 POST-RACE |
| 22 | netkeiba race_review_score | 0.9981 | review_score 自体が POST-RACE 評価 (v12.1 で 不採用済 を 再確認) |

→ **V20 構築時 NEVER USE** (以下を LEAK_FEATURES に追加 必須):
```python
LEAK_FEATURES_V20 = LEAK_FEATURES_A | SKB_LEAK_FEATURES | {
    'sed_time_sec', 'sed_first_3f', 'sed_last_3f', 'sed_finish', 'sed_abnormal',
    'review_score',  # netkeiba race_review.csv 由来
}
```

### 2-2. JRDB KKA parser 不全 (Sprint 5 阻害)

- `jra_seiseki_*`, `kyori_seiseki_*`, `track_seiseki_*`, `heavy_seiseki_*`, `class_seiseki_*` 全て 0% 非NaN
- ただし `dam_rensho_max/min/avg`, `bms_rensho_max/min/avg` は 100% 非NaN
- **AUDIT-1 期待値 +0.002-0.005 を 検証不能** (Sprint 5 では 不採用)

→ Sprint 5 では KKA を **見送り**、 Sprint 6 で parser 修復後 再評価。

### 2-3. 即実装 18 件 中 14 件 が delta ≤ 0

V15 145 features は既に 高カバレッジ。 単一 feature 追加 では 改善困難:
- training_intensity / index_max_filled / wood_best_4f_filled 等 が 既組込
- 新規 features は **組み合わせ (combo)** か **大規模 features (TFJV BS/BN/BR)** で 効果

→ Sprint 5 候補は 大幅縮小、 V20 統合 (Sprint 6+) が 本命に。

---

## 3. 残り Sprint 5 推奨候補 (修正後)

backtest 結果から Sprint 5 (5/16-5/22) では 以下を 着手:

| 順 | # | feature | 補正方法 | 工数 | 期待 |
|---|---|---------|--------|------|------|
| 1 | 11 | training_times_rank (label encode) | 'A/B/C/D' → 0/1/2/3 + WF AUC 再評価 | 2h | +0.001 |
| 2 | 5 | ai_opinion_pace (label encode) | 'H/M/S' → 0/1/2 + WF AUC 再評価 | 3h | +0.001 |
| 3 | 19 | jrdb_kyi_marks (encoding 修正) | encoding='cp932' で 再読み込み | 2h | +0.0003 |
| 4 | 17 | jrdb_tyb_live | Pattern B 専用 (Sprint 4 で別途) | 2h | live 改善 |
| 5 | 4 | jrdb_kka (parser 修復後) | parser fix 必要 → Sprint 6 へ後送 | 6h+ | +0.002 |

**Sprint 5 期待**: text encode 修正 + Pattern B 改善 で +0.002-0.005 AUC。
**Sprint 6 移送**: KKA parser 修復、 Sprint 6 で 再評価。

---

## 4. tool 詳細

```bash
python tools/audit_top27_backtest.py --features all --single
python tools/audit_top27_backtest.py --features 7,11,4 --single
python tools/audit_top27_backtest.py --features quick  # 1h/2h tier のみ
```

実行時間: 約 5 分 (single thread、 18 features × 約 17 秒/feature)
出力:
- data/v18/sprint5_backtest_metrics.json (JSON)
- 標準出力 (top 5 by delta)

---

## 5. 結論

✅ 18 件 即実装可能 features の 一括 backtest 完了
✅ **LEAK 2 件 確定** (#18 SED、 #22 review_score)
✅ KKA parser 不全 を 発見 (Sprint 5 阻害)
✅ V15 base AUC 0.8681 (200 rounds、 単 LGB) → 既存 145 features の 高カバレッジ確認

**主結論**:
1. Sprint 5 は **小幅** (text encode 修正のみ、 期待 +0.002-0.005)
2. Sprint 6 / V20 統合 が **本命** (TFJV BS/BN/BR + KKA parser 修復)
3. V20 LEAK list 追加: SED + review_score → 6 月 V20 学習時 必須
