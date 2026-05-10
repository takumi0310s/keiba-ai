# Phase 21D - 5/12 (月、 平日) 詳細 plan

> 作成: 2026-05-11 (Phase 21D)
> 投資保護: V15 案 B 改 単独継続 (predict_core.py 触らない)
> 5/12 は平日 (中央なし)、 NAR のみ → V15 production 影響ゼロ

## 目的

- 平日 PC フル活用 (Ryzen 7 7800X3D + RTX 4070 Ti SUPER 16GB + RAM 32GB)
- 4 Terminal 並行で 真値化 + V18 dataset + 戦略再評価 を一気に進める
- 中央 production には絶対 触らない (V15 case で 5/17 まで完全保全)

## 4 並行 task (Terminal A/B/C/D)

### Terminal A: Phase 11b 残 9 features 真値化 (90-120 min)

**目的**: JRDB KYI 残 9/15 features を 真値化 (現状 6/15 = 40%)

| 範囲 | 内容 |
|------|------|
| 入力 | jrdb_kka.csv (4/27-5/10 累計) |
| 出力 | data/jrdb_kka_features_v3.csv |
| 残 features | 9 個 (kka_renba_kyori_*、 kka_kakun_*、 kka_baba_keiken_* 等) |
| 検証 | corr_target & coverage > 80% / leak audit (Session #38 SKB 教訓) |

**実行**:
```
python tools/jrdb_kka_truth_v3.py --start 20260427 --end 20260510 --output data/jrdb_kka_features_v3.csv
python tools/audit_leak_v3.py --features data/jrdb_kka_features_v3.csv --target finish
```

**完了条件**: 9/9 真値化 + LEAK audit PASS + delta AUC ≥ +0.0010 (V15 PR 比)

### Terminal B: Phase 13b netkeiba master 25 features 真値化 (120-150 min)

**目的**: netkeiba スクレイピング master 25 features を 真値化 (現状 0/25)

| 範囲 | 内容 |
|------|------|
| 入力 | netkeiba shutuba HTML (5/10 4 場 約 144R) |
| 出力 | data/netkeiba_master_features_v1.csv |
| features | 上り 1F/3F、 走破 time、 通過 順、 厩舎 短評、 種牡馬 父系 等 |
| 検証 | coverage > 70% / 形式 audit |

**実行**:
```
python tools/netkeiba_master_truth_v1.py --date 20260510 --output data/netkeiba_master_features_v1.csv
```

**完了条件**: 25/25 抽出 + coverage > 70% (NaN handling 対応 込み)

### Terminal C: Phase 22 案 B 改 strict 戦略 再評価 実装 (60-90 min)

**目的**: Phase 21C で発見した 案 B 改 strict 重大訂正 を踏まえ、 戦略 再評価 logic を実装

| 内容 | 詳細 |
|------|------|
| 入力 | data/cumulative_results.csv + 5/10 帯別 深掘り 結果 |
| 修正 | score 帯 [0.50-0.55] 区間 cutoff、 1 勝 cls 上限 強化 |
| 実装 | tools/strategy_planB_strict_v2.py |
| 検証 | 4/27-5/10 retro で ROI 改善 ≥ +5pt 確認 |

**完了条件**: V15 production 触らず、 戦略 logic のみ追加 + retro PASS

### Terminal D: Phase 23 V18 dataset 整備 (90-120 min)

**目的**: V18 学習用 dataset (2020-2025、 sib_*_exp 込み) の 構築 + format check

| 範囲 | 内容 |
|------|------|
| 入力 | jra_races_full.csv + sib_expanding_v1 + JRDB KYI 真値 |
| 出力 | data/v18_dataset_v1.parquet |
| sample | 約 320K records (V20 構造ベース) |
| 検証 | NaN 率 / leak audit / coverage breakdown |

**実行**:
```
python tools/v18_dataset_build_v1.py --start 2020-01-01 --end 2025-12-31 --output data/v18_dataset_v1.parquet
```

**完了条件**: dataset 構築 + audit ALL PASS + 記録 (data/v18/v18_dataset_v1_audit.md)

## 投資保護 (絶対遵守)

- 🔴 predict_core.py / V15 model: NEVER 触る
- 🔴 5/12 中央レースなし (NAR のみ) → V15 自動運用 そのまま
- 🟢 戦略⑦ + 案 B 改 (5/9 確定) は次週末 5/17 まで継続

## 5/12 完了条件 (4 件 ALL PASS)

1. Phase 11b: 9/9 真値化 + LEAK PASS
2. Phase 13b: 25/25 抽出 + coverage > 70%
3. Phase 22: 戦略 v2 retro ROI 改善 ≥ +5pt
4. Phase 23: V18 dataset 構築 + audit ALL PASS

## 失敗時 fallback

- いずれか 1 件失敗 → 5/13 朝に optimal allocation 再計算
- V18 dataset NG → 5/13 を 修正 day に振替、 V18 学習は 5/14 に押し下げ (予定通り)
- 戦略 v2 NG → 5/9 案 B 改 そのまま継続 (絶対 安全)

## 次 step

- 5/13-5/14: data/v18/phase21d_5_13_5_14_plan.md (parser + backfill + V18 学習)
