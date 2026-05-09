# Phase 4 動画解析 plan v2 (server block 後の救世主 plan)

> Session #80 (2026-05-09) 作成
> 旧 plan (netkeiba 動画 / 静止画 base) の崩壊 → JRA-VAN RV 主軸 への切替

## 旧 plan (崩壊)

```
Source: netkeiba 動画 / 静止画
       ↓
   Session #62, #63 で 全 client BLOCK 確定
       ↓
   Phase 4 base 完全崩壊
```

| Session | 内容 | 結果 |
|---------|------|------|
| #62 | netkeiba 動画 endpoint 検証 | 全 BLOCK |
| #63 | netkeiba 静止画 (パドック画像) 検証 | 全 BLOCK |
| 結論 | netkeiba route 完全 dead | 代替 必須 |

## 新 plan (★救世主★) — 複合 source 戦略

### Source 1: JRA-VAN RV (★主軸★)

| 項目 | 値 |
|------|----|
| 内容 | 重賞 出走予定馬 調教動画 (公式) |
| 画質 | 高 (Mpeg4) |
| 安定性 | ★★★★★ (公式 server) |
| カバレッジ | 重賞のみ (週 2-4 R 程度) |
| コスト | 550 円/月 (NEXT 込 1,430 円) |
| 規約 | 個人視聴 OK、 個人録画グレーゾーン、 公開 NG |
| 投入 priority | ★★★★★ |

### Source 2: BS / 地上波 競馬番組

| 項目 | 値 |
|------|----|
| 内容 | グリーンチャンネル / みんなのKEIBA / 中央競馬中継 |
| 録画方式 | DR (Direct Recording) - 合法 |
| カバレッジ | 重賞 + 一般 R 一部 |
| コスト | BS 受信料 のみ (既契約なら追加 0 円) |
| 規約 | 個人録画合法、 公開 NG |
| 投入 priority | ★★★★ |

### Source 3: パドック静止画 (公開分)

| 項目 | 値 |
|------|----|
| 内容 | 各社 SNS / 公式 BLOG 提供 部分 |
| カバレッジ | 限定的 (全 R カバーは無理) |
| コスト | 0 円 |
| 規約 | 公開分のみ、 各社 規約 遵守 |
| 投入 priority | ★★ (補助のみ) |

## 投入 timing (V21 ロードマップ)

| 期間 | 内容 |
|------|------|
| 5/15-6/15 | RV trial (1 ヶ月)、 動画視聴 + 録画 試行 |
| 6/15-7/1 | 動画 features 抽出 logic 確定 (YOLOv8 + 5 features) |
| 7/1-9/2 | V21 動画統合 学習 |
| 9/2 | V21 投入候補 (V20 + 動画 features) |

## 期待 AUC

| Model | AUC | 備考 |
|-------|-----|------|
| V15 (現行) | 0.8939 | 本番 |
| V20 (構築中、 7/1 投入候補) | **0.90025** | PoC AUC 0.8752 (Session #44)、 全 features で 0.900+ 想定 |
| V21 (V20 + 動画 features、 9/2 候補) | **0.92-0.93** | 動画 features の corr 想定 +0.020-0.030 |

## カバレッジ戦略

| カテゴリ | source | カバレッジ想定 |
|---------|--------|--------------|
| G1/G2 重賞 | RV (主) + BS 録画 (副) | 100% |
| G3 重賞 | RV (主) + BS 録画 (副) | 100% |
| OPEN/L 特別 | BS 録画 のみ | 30-50% |
| 一般 R | パドック静止画 のみ | < 10% |

→ **重賞のみ高カバレッジ → V21 は 重賞特化 model として運用候補**
→ 一般 R は V20 single (動画 features 欠損) で運用

## 投資保護

- ★ V15 production 完全不変 ★
- ★ V20 投入 (7/1) も 動画なし (V20 single) で確定 ★
- V21 は完全に 動画 features 上乗せ phase、 V20 単独 fallback 必須

## 撤退条件

- RV trial 1 ヶ月で:
  - 動画品質 不足 → 撤退、 BS 録画 + パドック画像 のみで再 plan
  - features 抽出 困難 → 撤退、 PoC 縮小
  - AUC 改善 < +0.005 → 撤退、 V21 計画 中止
- 上記いずれか発生 → V20 単独運用継続 (V15 → V20 のみ、 V21 不採用)

## 関連 doc
- [JRA_VAN_RV_TRIAL_GUIDE.md](JRA_VAN_RV_TRIAL_GUIDE.md) — RV trial 手順
- [VIDEO_FEATURES_EXTRACTION_DESIGN.md](VIDEO_FEATURES_EXTRACTION_DESIGN.md) — features 抽出 logic 設計
- [RV_TRIAL_5_15_CHECKLIST.md](RV_TRIAL_5_15_CHECKLIST.md) — 5/15 trial 開始 checklist
- [PHASE_4_VIDEO_AI_DESIGN.md](PHASE_4_VIDEO_AI_DESIGN.md) — 旧 design (Session #39)
- [PHASE_4_VIDEO_FEASIBILITY_5_8.md](PHASE_4_VIDEO_FEASIBILITY_5_8.md) — feasibility 検証 (Session #42)
