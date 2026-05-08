# 5/9 予想 完全タイムスケジュール + 二段階予測 system 設計

**作成**: 2026-05-08 (Session #48 A、 main 直)
**目的**: 各 timing で取れる features と 二段階予測 system 設計の見える化

> **絶対遵守**: 5/9 投資 = V15 案B改 (12R 1勝のみ、 max 2,100円)
> 11R 重賞 + その他 R は **予測のみ、 投票しない** (本 doc は学習用)

---

## 1. 5/9 (土) タイムスケジュール

```
朝 - 06:30  起床、 MorningChecklist 自動 (Session #46)
朝 - 07:00  MorningDigest dashboard
朝 - 08:00  ★ V15 朝予測 (Stage 1) 自動実行
朝 - 08:45  RaceAutoNotify → Discord (案B改 候補)
朝 - 09:00  4 system 予測 通知 (重賞含む、 Session #48 D)
       09:30  PAT login + 入金

各 R 70 分前: ★ 当日馬体重 公式発表 (Stage 2 trigger)
各 R 30 分前: パドック動画 / 静止画 公開 (動画 PoC、 Session #48 C)
各 R  5 分前: 確定オッズ
各 R  発走 + 結果

夜 - 18:00  DailyResults 自動
夜 - 20:30  1 日 summary + 4 system verdict
夜 - 21:00  post_race_features_update (Sprint 2 H)
```

### 5/9 主要 race

| 場 | R# | レース | grade | 発走 |
|----|-----|--------|-------|------|
| 東京 | 11R | エプソムカップ | G3 | 15:45 |
| 京都 | 11R | 京都新聞杯 | G2 | 15:30 |
| 新潟 | 11R | 駿風 S | OP | 15:20 |
| 各場 | 12R | 1勝クラス | — | 16:00 前後 |

→ 12R 1勝クラス が 案B改 採用候補 (3 場 で 最大 3 R)
→ 11R 重賞 3 R は **観戦 + 4 system verdict 用** (投票なし)

---

## 2. 各 timing で取れる features (Tier 整理)

### Tier 1: 朝 06:00-09:00 (確定済 features)

| feature | source | V15 で使用? |
|---------|--------|------------|
| 過去成績 (finish, time, pass4 等) | jra_races_full.csv | ✅ |
| 騎手・調教師 expanding wr | 同上 | ✅ |
| 血統 (sire, bms) | blood_full.csv | ✅ |
| 調教タイム (4F best) | netkeiba | ✅ |
| 公式調教 (TFJV TM_DATA) | TFJV | ⚠ 部分 |
| JRDB 調教指数 (CYB) | jrdb_kyi.csv 等 | ✅ |
| 調教コメント | netkeiba | △ |
| sib_top3_rate_exp_w5 (Session #43 C) | netkeiba_siblings_expanding_w5.csv | ✅ |
| 出馬表 (8 確定後) | netkeiba | ✅ |
| 距離 / 馬場 / 性別 / 年齢 / 斤量 | 出馬表 | ✅ |

→ V15 朝予測 (Stage 1) で使う features

### Tier 2: 各 R 70 分前 (★ 二段階予測 trigger ★)

| feature | source | 公開 timing |
|---------|--------|------------|
| **当日馬体重** | netkeiba / JV-Link WF | 各 R 70 分前 ★ |
| 馬体重 変化 (kg, %) | 計算 | 同 |
| 過去 同距離 体重比較 | 計算 | 同 |
| 過去 3 走 体重 trend | 計算 | 同 |

→ Stage 2 で predict_core 再実行 (V15 + 当日体重 features)

### Tier 3: 各 R 5 分前

| feature | source | timing |
|---------|--------|--------|
| 確定オッズ (単勝/複勝/馬連/3連複) | netkeiba | 締切時 |
| odds flow (Sprint 1 D) | 直前 10 分 polling | 5/15+ schtasks |
| 人気 (popularity) | 同 | 同 |
| 直前 馬体重 補正 (緊急変更) | netkeiba | 締切後 |

→ Tier 3 features は Pattern B model (V15 確定オッズ込み) で使用、 case-by-case

### Tier 4: 動画解析 (各 R 30-60 分前)

| feature | source | 公開 timing | 動画範囲 |
|---------|--------|------------|---------|
| パドック静止画 (馬体検出) | netkeiba 画像 | 各 R 30 分前 | 全 R |
| パドック動画 (歩様 / pose) | JRA-VAN ネクスト / netkeiba | 各 R 20-30 分前 | G1 全頭、 重賞 注目馬、 一般 R なし |
| 調教動画 (歩様 / 仕上がり) | netkeiba | 金曜 13:00 (公開済) | 重賞 注目馬 |

→ Phase 4 (7-8 月) で本格、 本 Session C で PoC pipeline 設計

---

## 3. 二段階予測 system 設計

### 3.1 Stage 1 (朝 08:00、 現状)

```python
# tools/daily_predict.py (production、 不変)
features_stage1 = Tier 1 features
predictions_stage1 = V15.predict(features_stage1)
→ Discord 通知 (案B改 候補)
```

**現状で完成済**、 5/9 朝も そのまま使う。

### 3.2 Stage 2 (各 R 70 分前、 5/16 以降 trigger)

```python
# tools/two_stage_predict.py (本 Session B、 dev/two-stage)
# 各 R 70 分前 trigger (5/16+ schtasks)

features_stage2 = Tier 1 features + 当日馬体重 features (Tier 2)
predictions_stage2 = V15.predict(features_stage2)

# 朝予測との差分
diff = predictions_stage2[top1] - predictions_stage1[top1]
if abs(diff) > 0.05:
    Discord("Stage 2 で top1 変化: {old} → {new}")
```

**5/9 では未投入** (schtasks 追加なし)、 5/16+ V18 trial 後 検討。

---

## 4. 動画解析 pipeline 設計

### 4.1 pipeline (本 Session C、 dev/video-poc)

```
動画 source (netkeiba 調教 or JRA-VAN ネクスト パドック)
↓
download.py        : 動画 download
↓
yolo_inference.py  : YOLOv8 で馬体検出 (frame 単位)
↓
keypoint_extract.py: 歩様 keypoint (DLC SuperAnimal、 zero-shot)
↓
features_aggregate.py: stride / pose / 体格 features 化
↓
main_pipeline.py   : 統合 (1 race の全馬 features)
```

### 4.2 Phase 4 (7-8 月) plan との整合

Session #39 F + #42 E + #43 D で 環境構築 + PoC 完成。
本 Session C は **pipeline 統合**、 Phase 4 着手時に拡張。

### 4.3 動画範囲制約

| 動画種別 | カバレッジ |
|---------|-----------|
| 調教動画 (金曜 13:00 公開) | G1 全頭 / 重賞 注目馬 / 一般 R **なし** |
| パドック動画 (各 R 20-30 分前) | 重賞 全頭、 一般 R 部分的 |
| パドック静止画 | 全 R 全頭 (netkeiba Premium) |

→ V15 学習 data の 80%+ は動画なし、 動画 features は **重賞用 サブモデル** が現実的

---

## 5. 4 system 予測比較 (本 Session D)

### 5.1 4 system 構成

| # | system | 構成 | 用途 |
|---|--------|------|------|
| 1 | **V15 単独** | production current (Stage 1) | baseline、 5/9 投票用 |
| 2 | V15 + 拡張調教 | + Sprint 2 features | 学習用 |
| 3 | V15 + TM 公式調教 | + TFJV TM_DATA | 学習用 |
| 4 | V15 + 全部 + 当日体重 + パドック | Tier 1+2+3+4 | 学習用 (重賞 verdict) |

### 5.2 5/9 利用方法

```
朝 09:00: 4 system top3 + 合議 通知 (重賞 含む)
各 R 終了 5 分後: 4 system verdict
夜 20:30: 1 日 summary + 「もし重賞投票してたら ROI ○○%」
```

**重賞は予測 verdict のみ、 投票なし** (絶対遵守)

### 5.3 評価期間

5/9 + 5/16 + 5/22-23 = **約 9-12 R 重賞** で 4 system 比較。
継続検証で 4 system 合議 (TOP1 一致時の信頼度) を学習。

---

## 6. 5/16 V18 trial 含意

Session #43 C で V18 sib_w5 LIVE retro 完全回復 (+10.34pt)。
5/16 trial 投入候補:
- V15 単独継続 (案 A、 baseline)
- V15 + V18 sib_w5 並列 (案 B)

→ Session #48 4 system 比較 で 「重賞も V18 で投票」 候補は **5/22+ 検討**

---

## 7. 6/8 V20 含意

Session #44 で TFJV 6 年分 一括 parse 確認、 6/8 V20 投入候補 (1 ヶ月前倒し)。
Session #48 で 整備した:
- B Stage 2 (二段階予測) → V20 にも拡張可能
- C 動画 pipeline → V21 (V20 + 動画) で利用
- D 4 system → V20 検証用

→ V20 + V21 構築の 部品先行整備

---

## 8. 5/9 V15 投資保護 (本 doc 領域)

✅ V15 model md5: `842b9a5f305c793ed8fa54a74e06b836` 不変
✅ predict_core / daily_predict / app.py / schtasks 既存 38 件 不変
✅ 5/9 朝の V15 動作完全独立 (Stage 1 のみ稼働、 Stage 2 / 動画 / 4 system は学習用)
✅ 11R 重賞 + 他 R 投票なし、 12R 1勝 のみ

→ **5/9 朝 V15 案B改 完全保証**

---

## 9. 結論

✅ A1: 5/9 タイムスケジュール 整理 (Tier 1/2/3/4)
✅ A2: Stage 2 二段階予測 設計 (本 Session B 実装)
✅ A3: 動画 pipeline 設計 (本 Session C 実装)
✅ A4: 4 system 予測 plan (本 Session D 実装)
✅ A5: 5/16 V18 / 6/8 V20 含意

→ **本 doc は 5/9 + 5/16 + 6/8 の運用設計の中核**

---

**Session #48 A 完了 (main 直)**
