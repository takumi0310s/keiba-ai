# 5/15 RV Trial 開始 checklist

> Session #80 (2026-05-09) 作成
> 5/15 (木) JRA-VAN RV 1 ヶ月無料 trial 開始 当日 step

## 当日 (5/15 木) の step

### Step 1: NEXT trial 申込み (5 分)
- [ ] https://jra-van.jp/ アクセス
- [ ] 「NEXT 1 ヶ月無料 trial」 ボタン
- [ ] account 作成 (メアド + パスワード)
- [ ] 支払い method 登録 (1 ヶ月後 自動課金、 中止可)
- [ ] trial 開始 確認 mail 受信

### Step 2: RV connect (5 分)
- [ ] NEXT 管理画面 から 「RV 連携」 選択
- [ ] 月額 550 円 (NEXT 込 1,430 円) 確認
- [ ] 連携 contract

### Step 3: アプリ install (10 分)
- [ ] iOS: App Store 「JRA-VAN レーシングビュアー」 install
- [ ] Mac: ブラウザ視聴用 link 確認 (PC 公式 page)
- [ ] iOS / Mac 同一 account login
- [ ] 同期 確認 (login 状態 共有)

### Step 4: 動画視聴 試用 (15 分)
- [ ] 直近重賞 (5/17 ヴィクトリアM、 5/18 京王杯SC) の出走予定馬 調教動画 list 確認
- [ ] 1 動画 視聴 (画質 / 音声 / 安定性 確認)
- [ ] 同一動画 を Mac で視聴 (同期確認)
- [ ] 動画長 確認 (30 秒〜2 分 想定)
- [ ] 動画品質 評価:
  - 画角 (馬体全体 が見えるか)
  - 解像度 (480p 以上か)
  - frame rate (30fps 想定)

### Step 5: 個人録画 試行 (15 分)
- [ ] iOS: 画面録画機能 (control center) で 1 動画 録画
  - 設定 → control center → 画面収録 ON
  - control center → 画面収録 ボタン 長押し → mic OFF → 開始
- [ ] Mac: QuickTime Player 画面収録 で 1 動画 録画
  - QuickTime Player → ファイル → 新規画面収録
- [ ] 録画 ファイル を Mac → PC 転送 (AirDrop / iCloud / USB)
- [ ] PC で 再生確認 (.mov / .mp4 形式)

### Step 6: YOLOv8 + features 抽出 試用 (20 分)
- [ ] tools/video_features_poc.py 起動 (Session #42 PoC base)
- [ ] 録画 動画 1 本 を input
- [ ] YOLOv8 馬体 detection 動作確認 (95-138ms / frame)
- [ ] 5 features 計算 確認:
  - stride_length
  - body_size_relative
  - stability_score
  - tension_score
  - pace_score
- [ ] 統合 score 出力 確認

### Step 7: 結果 記録 (5 分)
- [ ] data/v18/rv_trial_5_15_log.md 作成
- [ ] 動画品質 評価 記録
- [ ] features 抽出 結果 記録
- [ ] 6/15 継続/解約 判断 用 メモ

## 必要 資源

| 資源 | 状態 |
|------|------|
| JRA-VAN account | 未作成 (5/15 当日 作成) |
| 支払い method | クレカ準備 |
| iOS device | 既存 (画面録画 OK) |
| Mac | 既存 (QuickTime OK) |
| PC + YOLOv8 環境 | 既存 (Session #42 で構築済) |
| tools/video_features_poc.py | 既存 (Session #42 で実装済) |

## trial 期間 中 (5/15-6/15) 想定 video 取得 量

| 週 | 重賞 R 数 | RV 動画 (× 16-18 馬) | BS 録画 R 数 |
|----|----------|--------------------|-------------|
| 5/16-5/22 | 4-5 R (ヴィクトリアM、 京王杯SC、 オークス、 平安S 等) | 70-90 動画 | 10-15 R |
| 5/23-5/29 | 3-4 R (ダービー 含む) | 50-70 動画 | 8-12 R |
| 5/30-6/5 | 3-4 R (安田記念 含む) | 50-70 動画 | 8-12 R |
| 6/6-6/12 | 2-3 R (epsom S 等) | 35-55 動画 | 6-10 R |
| 6/13-6/15 | 0-1 R | 0-15 動画 | 0-3 R |
| **合計** | **約 12-17 R** | **約 200-300 動画** | **約 32-52 R** |

→ trial 1 ヶ月で 馬体 frame セット **300-500 件** 収集可能

## 6/15 trial 終了 判断

### 継続条件 (550 円/月 続ける)
- [ ] 動画品質: 480p 以上、 馬体全体 見える
- [ ] features 抽出: 5 features 安定計算 OK
- [ ] frame セット: 200+ 集約可能
- [ ] アプリ安定: クラッシュなし、 動画再生中断なし

### 解約条件
- [ ] 上記いずれか不足 → 解約、 BS + 静止画 のみで再 plan
- [ ] V21 期待 AUC < V20 + 0.005 想定 → 解約

## 関連 doc
- [JRA_VAN_RV_TRIAL_GUIDE.md](JRA_VAN_RV_TRIAL_GUIDE.md) — trial 手順 詳細
- [PHASE_4_VIDEO_REPLAN_v2.md](PHASE_4_VIDEO_REPLAN_v2.md) — Phase 4 plan v2
- [VIDEO_FEATURES_EXTRACTION_DESIGN.md](VIDEO_FEATURES_EXTRACTION_DESIGN.md) — features 抽出 設計
- [PHASE_4_VIDEO_FEASIBILITY_5_8.md](PHASE_4_VIDEO_FEASIBILITY_5_8.md) — feasibility 検証 (Session #42)
