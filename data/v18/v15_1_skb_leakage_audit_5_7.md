# V15.1 SKB リーク検証 (Session #38 A)

**作成**: 2026-05-07 (Session #38)
**結論**: 🔴 **SKB は POST-RACE LEAK 確定、 V15.1 採用 NO-GO**

---

## TL;DR

V15.1 +675bp 改善 (Session #37 確認) の 99% は `skb_kishi_code_1/2/3` 由来。
これらは JRDB SKB ファイル (成績拡張データ = **post-race 成績データ**) から merge されており、 **当該レースの結果情報を学習時に含む 重大な target leak**。

**V15.1 採用 NO-GO**、 V15 維持。 5/16 投入候補から 完全削除。

---

## 1. 検証手順 (4 ステップ)

### Step 1: SKB ファイル仕様確認

`tools/parse_jrdb_missing.py:73-112` の `parse_skb_line()` 解析:

```python
# SKB パーサー (成績拡張データ、304バイト、ZKBと同一形式)
def parse_skb_line(line_bytes):
    ...
    # 騎手コード 6x3 (pos 27-44)
    for j in range(6):
        row[f'kishi_code_{j+1}'] = _safe_int(_field(line_bytes, 27 + j*3, 3))
    # 馬場コード 8x3 (pos 45-68)
    for j in range(8):
        row[f'baba_code_{j+1}'] = _safe_int(_field(line_bytes, 45 + j*3, 3))
    # 脚質コード 5x3 (pos 69-83)
    for j in range(5):
        row[f'kyaku_code_{j+1}'] = _safe_int(_field(line_bytes, 69 + j*9, 3))
    ...
    row['race_comment'] = _field(line_bytes, 234, 40).strip()  # ← 後評
```

JRDB ドキュメンテーション:
- **SKB = "成績拡張データ"** = **post-race**
- 同一構造の ZKB は "前走版のSKB" = 前走 (= 現走から見て post-race)
- 既存 `LEAK_TIME_INVARIANT` に `skb_anshin`, `skb_aisho`, `skb_heavy_apt` を含む = SKB は LEAK 既知

### Step 2: V15.1 features の SKB 取込ロジック

`train/v15_1_features.py:135-146` の `load_skb()`:

```python
def load_skb(path: str = 'data/jrdb_skb.csv') -> pd.DataFrame:
    use = ['race_id', 'umaban'] + list(SKB_SOURCE_COLS.values())
    df = pd.read_csv(...)
    ...
    df = df[df['race_part'] != ''].drop_duplicates(subset=['race_part','umaban'], keep='last')
    return df

# merge: race_part + umaban で 該当レースの SKB row を join
df_local = df_local.merge(skb, on=['race_part', 'umaban'], how='left')
```

→ **当該レースの SKB row そのまま** を学習 features に merge。 当該レースの post-race info そのもの。

### Step 3: 各 SKB feature の target との 相関

```
feature                         | corr_win | corr_top3 | non-zero
skb_kishi_code_1                |  +0.0240 |   +0.0140 |    99.8%
skb_kishi_code_2                |  +0.0717 |   +0.0793 |    85.0%
skb_kishi_code_3                |  +0.1369 |   +0.1614 |    53.8%  ← 異常高
skb_baba_code_1                 |  +0.0013 |   +0.0026 |    99.0%
skb_baba_code_2                 |  +0.0047 |   +0.0034 |    92.2%
skb_baba_code_3                 |  +0.0055 |   +0.0027 |    74.3%
skb_kyaku_code_1                |  -0.0187 |   -0.0206 |    22.7%
skb_kyaku_code_2                |  -0.0078 |   -0.0085 |    26.2%
skb_kyaku_code_3                |  -0.0072 |   -0.0078 |    25.7%
skb_turf_hoof                   |  +0.0033 |   +0.0028 |    56.6%
```

`skb_kishi_code_3` 単独で **target との相関 0.137** = 通常 pre-race feature では考えられない高さ。
比較: V15 の `jockey_wr_calc` (騎手勝率) corr ~ 0.05、 `pop_rank` (人気順) corr ~ 0.20。

### Step 4: ablation で各 SKB code の 寄与切り分け

`train/run_v15_1_skb_drilldown.py` 実行 結果 (200k subsample, test 2025):

| feature set | AUC | Δ vs V15 | n_features |
|-------------|-----|---------|-----------|
| V15_only | 0.8765 | - | 145 |
| V15+all_SKB (10件) | 0.9440 | **+675bp** | 155 |
| V15+SKB_no_kishi3 | 0.9251 | +485bp | 154 |
| V15+SKB_no_kishi23 | 0.8997 | +232bp | 153 |
| **V15+SKB_no_kishi (baba/kyaku/turf_hoofのみ)** | **0.8752** | **-13bp** | 152 |
| V15+kishi_only (3 codes) | 0.9439 | +674bp | 148 |
| V15+kishi3_only | 0.9246 | **+480bp** | 146 |
| V15+kishi1_only | 0.9006 | +240bp | 146 |
| V15+kishi2_only | 0.9182 | +416bp | 146 |

**重要発見**:
1. SKB 全体 +675bp → kishi_code_1/2/3 削除で **-13bp** (即 V15 baseline 以下)
2. kishi_code_3 単独で **+480bp** (1 feature で 4.8pp の AUC gain は **物理的にありえない**)
3. baba/kyaku/turf_hoof は 完全に無効 (-13bp、 むしろ若干悪化)

→ V15.1 改善 100% は kishi_code_1/2/3 由来、 これらは確実に LEAK。

---

## 2. リーク機構 (smoking gun 4 つ)

### 機構 1: skb_kishi_code_3 値の finish 順 単調変化

```
finish=1 (1着):  skb_kishi_code_3 mean = 364.5
finish=2:        291.7
finish=3:        279.0
finish=4:        257.8
finish=5:        243.2
finish=10:       175.6
```

→ **finish 順位が下がるほど skb_kishi_code_3 が小さい** ← pre-race feature では起き得ない単調関係。
これは **SKB の 6 kishi codes が当該 race の 結果含む 過去 winning history を encoding** している証拠。

### 機構 2: 0-rate の差

```
skb_kishi_code_3 == 0 rate:
  is_win=0 (敗者): 48.7%
  is_win=1 (勝者): 15.0%   ← 1着馬は 3.2x 多く 非ゼロ値を持つ
```

→ 1着馬は SKB に「特別な kishi_code_3」が記録される 確率が圧倒的に高い。
JRDB が当該レースで「勝った馬」を判別して特定 code を埋める処理が示唆。

### 機構 3: 1着馬と他馬で kishi_code_3 distribution が大幅に違う

```
1着馬 kishi_code_3 top: 177, 880, 732, 79, 874, 234, 415, 209, 215...
敗者  kishi_code_3 top: 441, 415, 130, 112, 171, 729, 435, 486, 177...
```

→ 完全に異なる distribution = 1着馬に固有の code が割り当てられる。 当該レース勝利後の post-process。

### 機構 4: JRDB SKB ファイル仕様

- ファイル名: SKB{YYMMDD}.txt (race day-level)
- パッケージ: `成績データ` group (SED, SRB, SKB は同梱)
- リリース timing: 当該 race day の **post-race** (JRDB 内部 analyst 解析後)
- 既存 V18/V19 LEAK list に `skb_anshin`, `skb_aisho`, `skb_heavy_apt` を含む = SKB ファイル全体が LEAK と既知

---

## 3. 真の改善幅 (SKB 削除後)

V15.1 採用するなら kishi_code_X 全削除必要:

```
V15+SKB_no_kishi (baba/kyaku/turf_hoof のみ): AUC -13bp = 改善ゼロ、 むしろ若干 悪化
```

→ 真の SKB 改善は **0bp 以下**、 V15.1 は V15 と同等 or 劣化。

V15.1 の他の data source (KKA / SRB) も Session #37 ablation で確認:
- KKA: +0bp (coverage 0%)
- SRB: +5bp (微増、 ノイズ範囲)

→ V15.1 全体の真の改善は ≤ 5bp、 V15 から 採用する 価値なし。

---

## 4. 5/16 投入判断: NO-GO

| 判定軸 | 結果 |
|-------|------|
| BT AUC | ~0.95 (但し +675bp は LEAK) |
| 真の AUC (LEAK 削除後) | ~0.876 (V15 0.8765 と同等) |
| LIVE 想定 winner_top1 | V15 と 同等 (改善なし) |
| 5/16 投入 効果 | ROI 改善 期待ゼロ |
| 投入 risk | LEAK 採用で 実 AUC は drop の可能性 |

→ **5/16 V15.1 投入 NO-GO**、 V15 案B改 維持。

---

## 5. Phase 3 (5/24+) への影響

### 5.1 V15.1 関連 plan 全削除

- ❌ V15.1 SKB 4-model ensemble (FT-Transformer + IntraRace 追加) → 中止
- ❌ tools/predict_v15_1.py 新規 → 中止
- ❌ V15.1 paper trading → 中止

### 5.2 V20 への影響 (limited)

V20 は JRA + NAR 統合 model、 SKB 系 features は **使わない予定**。
影響は限定的、 6/9-6/30 構築 plan は 維持。 但し:
- V20 で SKB 系 features 候補入れても 同 LEAK のため 全削除
- 共通 80 features に SKB 由来は ゼロにする

### 5.3 V18/V19 との関連

V18/V19 既存 model も SKB 派生 features を含むかも (要確認)。
既存 V18 LEAK list:
```
LEAK_TIME_INVARIANT = ["skb_anshin", "skb_aisho", "skb_heavy_apt", ...]
```

→ V18 では `skb_anshin/aisho/heavy_apt` のみ除外、 `kishi_code_*` / `baba_code_*` / `kyaku_code_*` は **未除外** (V18 は 元々これらを使ってない可能性)。

V18 features 確認 必要 (Session #38 B retro 後)。

---

## 6. 検証 file

- `train/run_v15_1_skb_drilldown.py` — drilldown ablation
- `data/v15.1/v15_1_skb_drilldown.json` — drilldown 結果
- `tools/parse_jrdb_missing.py:73-112` — SKB パーサー (成績拡張データ)
- `train/v15_1_features.py:135-146` — V15.1 SKB 取込 (LEAK 経路)
- `data/jrdb_skb.csv` — 元 data (post-race)

---

## 7. 結論 + 次 step

### 7.1 結論

🔴 **SKB は POST-RACE LEAK 確定**。
- skb_kishi_code_1/2/3 単独で +480bp+ の異常な AUC gain
- target との相関 0.14 (pre-race feature では不可能)
- finish 順位と単調関係 (post-race 確実)
- JRDB 仕様: 成績拡張データ = post-race file

V15.1 真の改善 = **≤ 5bp** (実質ゼロ)。

### 7.2 次 step

- **5/9 V15 案B改 投資**: 影響ゼロ、 完全保護 (V15 model + predict_core 完全不変)
- **5/16 V15.1 投入**: NO-GO 確定
- **5/24+ Phase 3**: V15.1 line を 削除、 V18/V19 sib抜き + V20 に focus
- Session #38 B (V18/V19 LIVE retro) で 5/16 / Phase 3 最終 plan 確定

5/9 V15 投資保護: **絶対遵守 OK**、 影響なし。
