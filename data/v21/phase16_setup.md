# Phase 16 A: RV 動画 download 環境 (5/10)

> Session #87 Phase 16 A 領域
> 出力: tools/rv_video_downloader.py (skeleton)

---

## 1. JRA-VAN ID + RV 連携状況

| 項目 | 値 |
|------|----|
| JRA-VAN ID | ✅ 加入済 (DataLab 経由) |
| RV 加入 | ✅ 2026-05-10 加入 (¥550/月) |
| trial 期間 | 5/15-6/15 (1 ヶ月) |
| 連携 method | NEXT account から 1 click 再生 |

---

## 2. 動画 download method 探索 結果

### 2.1 ★ 自動 download 経路: 全て不可 ★

| 経路 | 結果 | 理由 |
|------|------|------|
| ★ JV-Link 動画 API ★ | ❌ 不可 | JV-Link は メタデータ + 数値のみ、 動画 binary record なし (Session #42 確認済) |
| 公式動画 API | ❌ 不可 | 提供なし |
| RV 専用ソフト経由 | ❌ 不可 | アプリ内視聴のみ、 export 機能なし |
| ブラウザ scraping | ❌ NG | 規約違反 (mp4 ストリーミング、 DRM 不明、 アクセスブロック) |
| 自動大量 DL | ❌ NG | access ban 確実 |

### 2.2 ★ 個人視聴 + 手動録画 ★ (グレーゾーン、 個人 AI 学習限定)

| 方法 | 判定 | 備考 |
|------|------|------|
| iOS 画面録画 → AirDrop | ⚠ グレー | 個人 AI 学習なら問題薄、 公開 NG |
| Mac 画面録画 → USB 転送 | ⚠ グレー | 同上 |
| Android 画面録画 | ⚠ グレー | 同上 |
| PC ブラウザ + OBS | ⚠ グレー | 個人 録画範囲 |

→ 5/15-6/15 trial 中、 重賞 R のみ (週 2-4 R) 手動収集が現実的

---

## 3. tools/rv_video_downloader.py (skeleton 実装)

### 3.1 機能

| 機能 | status |
|------|--------|
| 動画 metadata 管理 (json) | ✅ 実装 |
| 配置済 動画 path 解決 | ✅ 実装 |
| category 別 list 集計 | ✅ 実装 |
| trial 進捗 status 出力 | ✅ 実装 |
| ★ 自動 DL ★ | ❌ **実装なし** (規約遵守) |

### 3.2 命名規則

```
data/rv_videos/
  paddock/<race_id>_u<umaban>.mp4   (パドック動画)
  patrol/<race_id>_u<umaban>.mp4    (パトロールビデオ)
  chokyou/<race_id>_u<umaban>.mp4   (調教映像)
  race/<race_id>.mp4                (過去レース、 馬番なし)
  multicam/<race_id>.mp4            (マルチカメラ)
```

### 3.3 self-test

```
$ python tools/rv_video_downloader.py
[rv_video_downloader] Phase 16 skeleton
[rv_video_downloader] VIDEO_DIR: .../data/rv_videos
[rv_video_downloader] VIDEO_DIR exists: False
[rv_video_downloader] trial status:
    trial_period: 2026-05-15 to 2026-06-15
    video_categories: ['paddock', 'patrol', 'chokyou', 'race', 'multicam']
    total_videos_collected: 0
    by_category: {'paddock': 0, 'patrol': 0, ...}
    note: 5/15+ で 重賞 trial 開始、 個人視聴 + 手動録画。 自動 DL は規約違反のため実装なし。
```

---

## 4. 5/10 全 35 R 動画 sample download (PoC)

### 4.1 ★ 結果: 5/10 当日収集 不可 ★

理由:
- 5/10 朝 RV 加入 完了
- ★ trial 開始は 5/15 木曜から ★ (RV 配信 timing が水・木の重賞前)
- 5/10 当日のレース (G1 NHKマイル C 含む) → 既に発走済、 録画 timing 失機
- 過去レース映像 (2002+) は trial 後に視聴可能

### 4.2 5/15+ trial 計画

| 期間 | 内容 | 動画予定 |
|------|------|---------|
| 5/15 (木) | RV trial 開始 | install + 同期 確認 |
| 5/16 (金) | 5/17 重賞 ヴィクトリアM 調教動画 視聴 + 録画 | 5-15 動画 |
| 5/17 (土) | ヴィクトリアM 当日 パドック + パトロール 録画 | 10-20 動画 |
| 5/18 (日) | 京王杯SC 同上 | 10-20 動画 |
| 5/22 (木) | オークス 調教 | 5-15 動画 |
| 5/24 (土) | オークス 当日 | 10-30 動画 |
| trial 終了 6/15 | 50-100 動画 蓄積 想定 | — |

---

## 5. V15 投資保護

✅ tools/predict_core.py / V15 model 不変
✅ rv_video_downloader.py = 新規 file、 V15 と完全独立
✅ ★ 自動 DL 実装なし ★ (規約遵守、 access ban risk なし)
✅ schtask 不変

---

## 6. 結論

✅ A1: RV 連携状況確認、 5/15+ trial 開始 plan 確定
✅ A2: 自動 DL 全経路 不可 確認、 個人視聴 + 手動録画 で trial
✅ A3: tools/rv_video_downloader.py skeleton (metadata 管理 + path 解決)
✅ A4: 5/10 当日収集 不可確認、 5/16+ で重賞動画手動収集 plan
✅ A5: V15 完全保護
