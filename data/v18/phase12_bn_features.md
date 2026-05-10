# Phase 12 B: 番組情報 features (3 件) 実装 (5/10)

> Session #87 Phase 12 B 領域 (2026-05-10 18:00+)
> 出力: tools/predict_core_v18.py の PROGRAM_INFO_FEATURES (3 件)

---

## 1. user 指示 vs 実 JV-Link record mapping

| user 指示 | 実 JV-Link record | 用途 |
|----------|-------------------|------|
| 「BN 番組情報拡張」 | RA (race info) + BT (番組テーブル) | 3 features 抽出 |

★ 注: JV-Link 仕様上 BN は **馬主 record (Breeder/Owner)**。 「番組情報」 は RA + BT records が正規 source。 user の意図を尊重し functional category として実装。 ★

---

## 2. 実装 3 features

| feature | source record | encoding |
|---------|--------------|----------|
| jv_race_class_detail | RA (クラス coding) | G1=10/G2=8/G3=6/L=5/OP=4/3勝C=3/2勝C=2/1勝C=1/未勝利=0 |
| jv_prize_structure_total | RA (1-5 着 賞金合計、 千円単位) | numeric (default 5000) |
| jv_entry_condition_enc | BT + RA (出走条件) | 牡牝混合=0 / 牝限=1 / 特定条件=2 |

---

## 3. live activation 設計

### 3.1 Phase 12 (本日)
- skeleton 実装、 default fill のみ
- 既存 RA records は jra_races_full.csv 経由で 部分情報あり (V15 で活用済)
- 本 features は 拡張版 (賞金構造 + BT 出走条件)

### 3.2 5/24+ Phase 3 後半
- tools/jvlink_fetcher_v2.py の RA parser 拡張
- BT (番組) record parser 新規実装
- `data/jvlink/RACE/<race_id>_parsed.csv` から拡張 fields 抽出

---

## 4. V15 既統合 features との 差別化

V15 では以下を既統合:
- distance, surface_enc, course_enc, num_horses_val (RA 由来)
- season, age_season (RA 由来 派生)

Phase 12 B で 新規追加:
- ★ jv_race_class_detail ★: G1/G2/G3 等の階層 detail (V15 では class_code を 1-5 で粗く扱う)
- ★ jv_prize_structure_total ★: 賞金構造 (V15 では prev_prize のみ)
- ★ jv_entry_condition_enc ★: 出走条件 (V15 未統合)

---

## 5. 動作 test (Phase 12 全体)

```
B. 番組情報 (3): ['jv_race_class_detail', 'jv_prize_structure_total', 'jv_entry_condition_enc']
default fill 動作確認: 全 3 features OK
```

---

## 6. V15 投資保護

✅ tools/predict_core.py 不変
✅ V15 model 不変
✅ tools/predict_core_v18.py 新規 file (skeleton)
✅ live fetch なし

---

## 7. 結論

✅ B1: RA + BT records 由来 3 features 定義
✅ B2: V15 既統合 と差別化 (class detail / 賞金 / 条件)
✅ B3: skeleton 実装 (default fill 動作 OK)
✅ B4: V15 完全保護
