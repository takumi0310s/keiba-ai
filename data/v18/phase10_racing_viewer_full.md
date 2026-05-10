# Phase 10 B: JRA レーシングビュアー (RV) 完全 audit (5/10)

> Session #87 (2026-05-10 夜) Phase 10 B 領域
> 対象: ★ JRA-VAN レーシングビュアー (¥550/月、 5/10 新加入) ★
> 趣旨: read-only audit、 V15 production 完全不変

---

## 1. 加入サービス概要

| 項目 | 値 |
|------|----|
| サービス名 | JRA-VAN レーシングビュアー (RV) |
| 月額 | ¥550 (RV 単独) / ¥1,430 (NEXT 込み) |
| 加入状況 | ✅ 加入済 (2026-05-10) |
| 配信形式 | Mpeg4 ストリーミング |
| 視聴環境 | iOS / Android / PC ブラウザ |
| 連携 | JRA-VAN NEXT (オッズ + レース info) と 1 click 連携 |
| 配信 timing | 水曜・木曜 (重賞前々日〜前日) |

---

## 2. 取得可能 映像 全 list

### 2.1 ★ 契約者特権 (一般 web では取れない) ★

| 映像 | 内容 | カバレッジ | priority |
|------|------|-----------|----------|
| ★ 重賞調教動画 ★ | 出走予定馬 | 重賞のみ (週 2-4 R) | ★★★★★ |
| ★ パドック動画 ★ | 馬体充実度 / 発汗 / 蹄 | 全 R 一部 | ★★★★ |
| ★ パトロールビデオ ★ | 進路取り / 不利 / 接触 | 全 R | ★★★★★ |
| マルチカメラ | 多角 / スロー再生 | 重賞 | ★★★ |
| 過去レース映像 | 2002+ 全 R | 過去 23 年分 | ★★★★ |
| 競走中止映像 | 故障 / 落馬 | 該当 R | ★★ |
| 調教坂路映像 | 美浦 / 栗東 | 全頭 (一部) | ★★★ |

### 2.2 取得方法

| 取得経路 | 可否 | 備考 |
|---------|------|------|
| iOS / Android アプリ視聴 | ✅ OK | 同一 account login で同期 |
| PC ブラウザ視聴 | ✅ OK | RV web 経由 |
| ★ 個人画面録画 (iOS / Mac) ★ | ⚠ グレー | 個人 AI 学習なら問題薄、 公開 NG |
| ★ JV-Link 経由 直 download ★ | ❌ 不可 | JV-Link は **メタデータ + 数値のみ**、 動画は別経路 |
| ★ 公式 API ★ | ❌ 提供なし | 動画 endpoint は非公開 |
| 自動 大量 DL | ❌ NG | 規約違反、 access ban risk |

### 2.3 規約 注意点

| 行為 | 判定 | 備考 |
|------|------|------|
| 個人視聴 | ✅ OK | 規約遵守 |
| 個人録画 (iOS/Mac 画面録画) | ⚠ グレー | 個人 AI 学習なら問題薄、 公開禁止 |
| 商用利用 | ❌ NG | 規約違反 |
| 再配信 / 公開 | ❌ NG | 著作権侵害 |
| 自動 DL 大量保存 | ❌ NG | access ban 可能性 |

---

## 3. 動画 features 抽出 plan

### 3.1 抽出 model 候補 (Session #42 PoC 完了)
- ✅ ultralytics 8.4.47 install OK
- ✅ YOLOv8 馬体検出 動作確認 (CPU 138ms / GPU CUDA NMS 課題あり)
- ✅ NVIDIA RTX 4070 Ti SUPER (16 GB) 利用可能

### 3.2 ★ パドック動画 features (★候補★) ★

| feature | model | 期待 corr |
|---------|-------|----------|
| paddock_body_score | YOLOv8 + 馬体 keypoint | +0.005-0.010 |
| paddock_sweat_level | RGB 解析 + 段階分類 | +0.003-0.008 |
| paddock_hoof_condition | 蹄 keypoint + 角度 | +0.002-0.005 |
| paddock_hindleg_drive | DLC SuperAnimal HORSE-10 | +0.005-0.010 |
| paddock_calmness_score | 動画 frame 安定度 | +0.003-0.008 |

### 3.3 ★ パトロールビデオ features ★

| feature | model | 期待 corr |
|---------|-------|----------|
| patrol_furi_count | YOLOv8 馬群 + 接触検出 | +0.005-0.012 |
| patrol_route_efficiency | 軌跡 + 距離計算 | +0.003-0.008 |
| patrol_block_detection | 馬群 occlusion 解析 | +0.004-0.010 |
| patrol_pace_change | 通過時刻 + 加速度 | +0.003-0.007 |

### 3.4 調教映像 features

| feature | model | 期待 corr |
|---------|-------|----------|
| training_stride_length | DLC + フレーム計測 | +0.005-0.012 |
| training_gait_symmetry | 左右 step 比較 | +0.004-0.008 |
| training_finish_speed | 後半 frame 速度 | +0.003-0.007 |

### 3.5 V21 期待 AUC
- V20: 0.90025 (PoC 0.8752 + ensemble 効果)
- V21 = V20 + 動画 features: ★ 0.92-0.93 ★ (corr +0.020-0.030 想定)

---

## 4. V21 構築 plan (動画統合)

### 4.1 schedule (旧 plan 維持)

| 期間 | 内容 |
|------|------|
| 5/15-6/15 | RV trial (1 ヶ月)、 重賞調教動画 視聴 + 録画 試行 |
| 6/15-7/1 | 動画 features 抽出 logic 確定 (YOLOv8 + 5 features) |
| 7/1-9/2 | V21 動画統合 学習 |
| 9/2 | V21 投入候補 (V20 + 動画 features) |

### 4.2 カバレッジ戦略

| カテゴリ | source | カバレッジ想定 |
|---------|--------|--------------|
| G1/G2 重賞 | RV (主) + BS 録画 (副) | 100% |
| G3 重賞 | RV (主) + BS 録画 (副) | 100% |
| OPEN/L 特別 | BS 録画 のみ | 30-50% |
| 一般 R | パドック静止画 のみ | < 10% |

→ 重賞のみ 高カバレッジ → V21 は 重賞特化 model として運用候補
→ 一般 R は V20 single (動画 features 欠損) で運用

### 4.3 PoC 完了済 (Session #42)
- ultralytics 8.4 環境構築 (1-2 分)
- YOLOv8n model load + inference (138ms CPU)
- COCO horse class (17) 存在 確認
- 環境構築 35-75h 削減 (Phase 4 着手即可)

---

## 5. 撤退条件

RV trial 1 ヶ月 (5/15-6/15) で:
- 動画品質 不足 → 撤退、 BS 録画 + パドック画像 のみで再 plan
- features 抽出 困難 → 撤退、 PoC 縮小
- AUC 改善 < +0.005 → 撤退、 V21 計画 中止

→ いずれか発生 → V20 単独運用継続 (V15 → V20 のみ、 V21 不採用)

---

## 6. 結論

✅ B1: 取得可能映像 全 list (重賞調教 / パドック / パトロール / マルチカメラ / 過去レース)
✅ B2: 動画 download 方法 (アプリ視聴 OK、 直 download 不可、 個人録画グレー)
✅ B3: 動画 features 抽出 plan (パドック 5 / パトロール 4 / 調教 3 = 12 features)
✅ B4: V21 構築 plan (5/15 trial → 9/2 投入候補、 期待 AUC 0.92-0.93)
✅ B5: 撤退条件 明確化 (1 ヶ月 PoC で AUC < +0.005 なら V21 中止)

→ **5/15 RV trial 開始、 7/1-9/2 V21 学習、 9/2 投入判定**
→ **5/10 朝 V15 完全保証** (V21 は 完全に 動画上乗せ phase、 V20 fallback 必須)
