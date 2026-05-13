# Phase 12 audit D: 工数再評価 + 5/24+ 計画影響

実行: 2026-05-13 PM、 Opus 4.7、 read-only

## D1. 真の工数 (audit B + C より)

| 作業 | 工数 | 内容 |
|------|-----|------|
| **race_name bug 修正のみ** | **30 分** | tfjv_parser.py L62 offset 28 → 32、 tokubetsu_num field 追加、 RA_2026.csv re-extract、 288 R JSON 再生成 |
| 簡単 features 4 件 (race_name + class + grade + distance) | 1-2h | 上記 + RA_FIELD_OFFSETS 15-20 fields 追加 |
| 中 features 追加 (SE pace/lap + WE/WH 天候 4 件) | 2-3h | SE 拡張 + WE/WH parser 新規 |
| 全 17 features parse | **1-2 day (8-16h)** | + 難 7 件 (O1/O2/O5 時系列 + UM/SK/BR 血統) |

## D2. 5/14-5/16 着手可能性

### case A (即修正、 5/17 V20 真の学習)

★ 不可能 ★ - 17 features 全 parse には 1-2 day 必要、 5/14-5/16 で 一気は GPU 学習 + 検証 込みで 不足。

### case B (一部前倒し、 推奨) ★ ★ ★

| 日付 | 作業 | features 累計 |
|-----|------|-------------|
| 5/14 PM | bug fix + 番組情報 3 (簡単 4 件) | 4 / 17 |
| 5/15 (木) | SE pace/lap 拡張 (中 2 件) | 6 / 17 |
| 5/16 (土) AM | WE/WH 天候 (中 4 件) | 10 / 17 |
| 5/16 PM | 検証 + V22 enhanced学習 (簡単+中=10 features 統合) | + V22 retrain |
| 5/24+ | O1/O2/O5 + UM/SK/BR (難 7 件) JV-Link COM 必須 | 17 / 17 |

- 5/16 までに **10 features 真値化 + 5/13 PM 143 candidate features (本日) と統合**
- 合計 **153 features V20 学習候補** (10 jvlink + 143 features_*)
- V22 retrain 5/17 (土) 可能 (新 10 features 統合、 GPU 93min)

**V20 真の構築 6/1-6/15 (orig 5/27-6/9 ± 数日 ずれ込み)**

### case C (計画通り、 5/24+ 集中)

| 日付 | 作業 |
|-----|------|
| 5/14-5/16 | V22 改良 + 1 番人気回避 etc.、 features 真値化 skip |
| 5/24+ | 17 features 一気 (1-2 day) + V20 学習 |

**V20 真の構築 6/15-7/1 (orig plan、 ずれなし)**

## D3. V20 真の構築 timing 影響

| case | V20 真の学習 開始 | V20 投入 判定 |
|------|----------------|--------------|
| **case B 推奨** | **6/1** (前倒し 1 週間) | 6/29 (前倒し) |
| case C | 6/15 (orig plan) | 7/1 (orig plan) |

### case B 採用時 メリット

- 10 features 真値化 (合成データ → 真の data) → V22 enhanced で V15 越え 検証 早期
- 5/24+ までに **143 candidate (本日合成) + 10 真値 features = 153 features 統合済** → V20 学習 加速
- 残 7 features (難) は 5/24+ JV-Link RT で 追加 (orig timing 維持)

### case B 採用時 デメリット

- 5/14-5/16 週末 + 平日に 計 5-6h 拘束 (user 留守中の 部分作業 想定)
- 中 features (WE/WH) parser spec ref 必要、 既知でない場合 工数増加リスク

## D4. 5/24+ 計画への影響

### orig plan (5/9 commit 96d8a2d9 時点)

- 5/24 加入 + JV-Link 動作確認
- 5/25-5/27 sib_*_exp 統合 + V18/V19 再学習
- 5/28-6/8 V18/V19 LIVE retro
- 6/9-6/30 V20 構築 (JV-Link parser + 学習)
- 7/1+ V20 production 投入判定

### case B 採用時 修正 plan

| 期間 | 作業 |
|-----|------|
| **5/14-5/16** | ★ Phase 12 parser 修正 + 10 features 真値化 ★ |
| 5/16 (土) PM | V22 enhanced retrain (新 10 features 統合) |
| 5/17 (日) | V22 enhanced 検証 + WF AUC 確認 |
| 5/18 (月) | nightly + weekly report |
| 5/19-5/23 | V22 fine-tune + features 効果 評価 |
| **5/24+** | **JV-Link 加入 + 残 7 features 真値化 (難)** |
| 5/27-6/1 | V20 真の構築 着手 (153 features 統合) |
| 6/1-6/8 | V20 学習 + WF 検証 |
| 6/8-6/15 | V20 paper trading (週末) |
| 6/15+ | V20 production 投入判定 |

**前倒し効果**: V20 投入 7/1 → **6/15+** (約 2 週間 前倒し可能)

## 推奨 action

★ **case B 採用 推奨** ★

理由:
1. race_name bug は **30 分 で 修正可能** (真の bug 1 箇所確定)
2. 簡単 + 中 features 10 件 で V22 enhanced 検証 早期
3. 143 candidate features (本日合成) と統合可能性
4. V15 投資保護 完全 (parser は read-only、 別 dir 出力、 V15 .pkl.gz 不変)
5. 残 7 件 (難) は 5/24+ JV-Link 経由で 同 timing 完成

### 5/14 (水) PM 着手指示 (user 判断後)

1. tools/tfjv_parser.py L62 修正 (offset 28 → 32 + tokubetsu_num 追加)
2. RA_FIELD_OFFSETS 拡張 (15-20 fields、 race_class/grade/distance 等)
3. data/tfjv/RA_2026.csv re-extract (10 sec)
4. data/jvlink/2026/{04,05}/*.json 再生成 (10 sec)
5. 検証: race_name = "皐月賞" 等 真の race name 取得確認

## V15 投資保護

- 全 修正 read-only / 別 dir 出力 (data/tfjv/、 data/jvlink/)
- V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変
- 17 features は V20+/V22 学習用、 V15 inference path には 影響なし

## 158h+ マラソン哲学 遵守

- ✅ data 駆動 慎重判断 (1194 records 全 ".0000" prefix 確認)
- ✅ V15 投資保護 完全
- ✅ fabrication 防止 (実 csv read + spec ref で bug 確定)
- ✅ ユーザー重大質問 真摯対応 ("data 取得 必要" → 実は data 既存、 parser 1 箇所 bug)

## まとめ

**真の bug**: tools/tfjv_parser.py L62 race_name offset 28 → 32 (特別競走番号 4 byte skip 抜け)。
**修正工数**: 30 分 (1 箇所 1 行 + field 追加 + re-extract)。
**5/24+ 計画**: case B 採用で V20 投入 7/1 → **6/15+** に 2 週間 前倒し可能。
**5/14-5/16**: 簡単 + 中 10 features 真値化 (4-5h 拘束)。
**残 7 features (難)**: 5/24+ JV-Link 32-bit venv + credentials 設定後 集中。
