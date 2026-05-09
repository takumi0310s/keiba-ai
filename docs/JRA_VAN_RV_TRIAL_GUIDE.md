# JRA-VAN レーシングビュアー (RV) Trial 手順

> Session #80 (2026-05-09) 作成
> 重大発見: ★ JRA-VAN RV で重賞調教動画 公式提供 ★ → Phase 4 動画 plan 救世主
> netkeiba 動画 / 静止画 全 server BLOCK 確定後 (Session #62/63) の代替

## 概要

| 項目 | 値 |
|------|----|
| 提供元 | JRA-VAN (公式) |
| コンテンツ | 重賞 出走予定馬 調教動画 |
| 配信形式 | Mpeg4 ストリーミング |
| 月額 | 550 円 (RV 単独) / 1,430 円 (NEXT 込み) |
| 連携 | JRA-VAN NEXT から 1 click 再生 |
| 配信 timing | 水曜・木曜 (重賞前々日〜前日) |
| 視聴環境 | iOS / Android / PC ブラウザ |

## 1 ヶ月無料 trial 手順

1. **アクセス**: https://jra-van.jp/
2. **NEXT 申込み**: 「1 ヶ月無料 trial」 ボタン
3. **RV 連携 contract**: 月額 550 円 (NEXT 込み 1,430 円)
4. **アプリ install**:
   - iOS: App Store 「JRA-VAN レーシングビュアー」
   - Android: Google Play 同名
   - PC: ブラウザ視聴可
5. **PC + スマホ同期確認**: 同一 account login で OK

## 試用 plan (5/15-6/15、 1 ヶ月)

| 期間 | 内容 |
|------|------|
| 5/15 (木) | contract + install + 同期確認 |
| 5/16-5/22 | V18 trial と同時 重賞調教動画 視聴 (5/17 ヴィクトリアM、 5/18 京王杯SC、 5/24 オークス) |
| 5/23-5/30 | 動画 features 抽出 試行 (YOLOv8 + 5 features) |
| 6/1-6/15 | AI 学習 base 構築 (10-20 重賞 sample) |
| 6/15 | trial 終了判定: 継続 (550 円/月) or 解約 |

## 利用規約 注意点

| 行為 | 判定 | 備考 |
|------|------|------|
| 個人視聴 | OK | 規約遵守 |
| 個人録画 (iOS/Mac 画面録画) | グレーゾーン | 個人 AI 学習なら問題薄、 公開禁止 |
| 商用利用 | NG | 規約違反 |
| 再配信 / 公開 | NG | 著作権侵害 |
| 自動 DL 大量保存 | NG | 規約違反 risk、 access ban 可能性 |

## 推奨 approach (個人 AI 学習用)

1. **重賞前日 (金) に 調教動画 視聴**
2. **iOS / Mac 画面録画機能 で 個人録画** (1 動画 30 秒〜2 分)
3. **PC に転送** (AirDrop / iCloud / USB)
4. **YOLOv8 で 馬体 features 抽出** (frame 単位)
5. **数値 features (歩幅 / 体格 / 動作安定性 / 緊張度 / 歩様) を V20 model に投入**

## V20 / V21 ロードマップ 上での位置づけ

| Phase | 期間 | 内容 |
|-------|------|------|
| 5/15-6/15 | 試用 | RV trial (1 ヶ月)、 動画視聴 + 録画 試行 |
| 6/15-7/1 | 設計 | 動画 features 抽出 logic 確定 |
| 7/1-9/2 | 実装 | V21 動画統合 学習 |
| 9/2 | 投入候補 | V21 (V20 + 動画 features)、 AUC 0.92-0.93 想定 |

## 関連 doc
- [PHASE_4_VIDEO_REPLAN_v2.md](PHASE_4_VIDEO_REPLAN_v2.md) — Phase 4 動画 plan v2
- [VIDEO_FEATURES_EXTRACTION_DESIGN.md](VIDEO_FEATURES_EXTRACTION_DESIGN.md) — features 抽出 logic 設計
- [RV_TRIAL_5_15_CHECKLIST.md](RV_TRIAL_5_15_CHECKLIST.md) — 5/15 trial 開始 checklist
- [PHASE_4_VIDEO_FEASIBILITY_5_8.md](PHASE_4_VIDEO_FEASIBILITY_5_8.md) — feasibility 検証 (Session #42)
- [PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md](PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md) — 統合 roadmap v3
