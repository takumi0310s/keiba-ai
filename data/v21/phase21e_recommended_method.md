# Phase 21E D: 推奨 method 確定 + 5/14 PoC plan

> Session #88 (2026-05-11) Phase 21E D
> 結論: ★ 推奨 method = 「OS 画面録画 (OBS Studio)」 単独 ★
> ★ 工数 中、 安定性 高、 規約 OK、 容量 中、 5/14 PoC 設計 → 5/15-5/18 trial 実行 ★

---

## 1. 結論 (一行)

★ **推奨 method = OS 画面録画 (OBS Studio + Mac QuickTime backup) で 個人視聴の延長として録画 → ffmpeg frame 抽出 → YOLOv8 features 抽出** ★

---

## 2. 3 method 比較 table

| 観点 | A. JV-Link API | B. RV ソフト 公式 DL | C. ブラウザ scraping (m3u8) | **D. OS 画面録画 (OBS) ★** |
|------|---------------|--------------------|-----------------------|---------------------------|
| **動画取得 可否** | ❌ 不可 (API なし) | ❌ 不可 (公式機能なし) | △ 技術的に可能 (DRM ない場合) | ★★★ 確実 |
| **工数 (初期)** | — | — | 5-10 日 (m3u8 解析、 ban 対策) | **2-3 日** (OBS 設定 + script) |
| **工数 (運用)** | — | — | 高 (UI 変更 / 認証期限切れで頻繁壊れる) | **中** (録画 = 視聴中のみ可、 半自動) |
| **安定性** | — | — | ★ (UI 変更で壊れる) | ★★★ (画面録画 = 不変) |
| **規約 risk** | — | — | ★ (大量 DL → ban) | ★★★ (個人視聴 = OK) |
| **法的 risk** | — | — | ★ (DRM 解除 = 違法) | ★★★ (私的複製 30 条 OK) |
| **容量 / 動画** | — | — | 元 file 直 DL (圧縮済) | **OBS 録画 (1080p/30fps): 1 分 = 約 30 MB** |
| **画質** | — | — | 元配信通り | OBS 設定で 自由 (元配信が上限) |
| **fps / 解像度 制御** | — | — | 元配信のみ | **OBS で自由設定** (機械学習向け) |
| **総合判定** | NG | NG | NG | **★ ★ ★ 採用** |

→ **A/B/C は全部 NG。 D (OS 画面録画) のみが現実解**。

---

## 3. 推奨 method 詳細: OBS Studio 画面録画

### 3.1 構成
```
[Browser (RV)] → [OBS Studio: Display Capture / Window Capture]
                      ↓ 録画
                 [ローカル mp4 / mkv file]
                      ↓ ffmpeg
                 [JPG frame list (1 fps)]
                      ↓ YOLOv8 + DLC SuperAnimal
                 [features 数値 (30 features)]
                      ↓
                 [V21 model 投入]
```

### 3.2 OBS 推奨設定
| 項目 | 値 |
|------|----|
| 出力 format | mp4 |
| 解像度 | 1920 × 1080 (元配信が高ければ それに合わせる) |
| fps | 30 (機械学習向け、 1 fps frame 抽出) |
| エンコーダ | x264 (CPU、 互換性高) または NVENC (GPU、 速い) |
| Bitrate | 約 6,000 kbps |
| 1 分動画 容量 | 約 30-50 MB |
| 録画 source | 「ウィンドウキャプチャ」 で ブラウザ window 指定 |

### 3.3 半自動化方針
- **完全自動化は困難** (画面録画 = 視聴中のみ可、 ブラウザを操作しないと動画再生されない)
- **半自動化** = OBS websocket + Playwright で ブラウザ操作 + 録画開始/停止
- **5/15 PoC** = まず手動で 1 重賞 (約 30 動画) 録画して工数感を確定
- **5/22+ 拡張** = 半自動化 script (tools/rv_video_capture.py) 実装

---

## 4. 5/14 PoC 実装 設計 (skeleton)

### 4.1 tools/rv_video_capture.py (5/14 設計、 5/15-5/18 PoC)
```python
# Skeleton 設計 (実装は 5/14 別 commit で)
import subprocess, time, os
from pathlib import Path
from playwright.sync_api import sync_playwright

def login_rv():
    """ Playwright で RV login。 cookie 保存 """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://prc.jp/jraracingviewer/")
        # 手動 login → cookie 保存 (手動 1 回のみ)
        input("login 完了したら Enter")
        page.context.storage_state(path="data/rv_cookies.json")

def open_video_page(race_id, content_type):
    """ RV の動画ページ open。 content_type: paddock / patrol / training """
    # Playwright で 動画 URL に navigate
    pass

def start_obs_recording(output_path):
    """ OBS websocket で録画開始 """
    # obs-websocket-py ライブラリ使用
    pass

def stop_obs_recording():
    """ OBS 録画停止 """
    pass

def extract_frames(video_path, fps=1):
    """ ffmpeg で frame 抽出 """
    cmd = f"ffmpeg -i {video_path} -vf fps={fps} {video_path}.frame_%04d.jpg"
    subprocess.run(cmd, shell=True)

def extract_features_yolo(frame_dir):
    """ YOLOv8 で 馬体検出 + features 数値化 """
    # 既存 tools/predict_core_v21.py の skeleton 活用
    pass

if __name__ == "__main__":
    # PoC: 1 重賞 30 動画 試行
    race_id = "202605030611"  # 5/17 ヴィクトリアM 想定
    for horse_no in range(1, 19):  # 18 頭分
        for content in ["training", "paddock"]:
            output = Path(f"data/v21/rv_videos/{race_id}_{horse_no:02d}_{content}.mp4")
            if output.exists():
                continue
            # 手動 step (PoC 段階)
            print(f"視聴: {race_id} #{horse_no} {content}")
            input("録画開始 → 視聴 → 録画停止 → Enter")
```

### 4.2 5/14 完成物
- [ ] tools/rv_video_capture.py skeleton
- [ ] obs-websocket-py 依存追加 (requirements.txt)
- [ ] data/v21/rv_videos/ ディレクトリ作成
- [ ] PoC 手順 doc (5 step、 1 ページ)

### 4.3 5/15-5/18 PoC TODO
| 日 | 作業 | 完了基準 |
|----|------|---------|
| 5/15 (木) | OBS install + RV login + 1 重賞 (例 5/17 ヴィクトリアM) 調教動画 5 本 録画 | mp4 file 5 件、 1 件あたり 30 秒〜2 分 |
| 5/16 (金) | DevTools で m3u8 抽出 試行 (DRM 有無 確認 only)、 ffmpeg frame 抽出 + YOLOv8 dry-run | 1 frame で 馬体検出 OK |
| 5/17 (土) | 重賞当日 パドック動画 録画 + パトロール動画 録画 | 重賞 1 件、 パドック + パトロール 各 14 頭分 |
| 5/18 (日) | 京王杯SC 同 試行 + 工数感計測 + 容量計測 | 累計 30-60 動画、 累計 1-2 GB、 1 重賞 工数 約 2-3 時間 |

---

## 5. リスク + mitigation

| リスク | 発生確率 | 対策 |
|--------|---------|------|
| RV が DRM 含む → 画面録画も画質劣化 | 低 | 5/15 録画試行で 確認、 劣化なら OBS 設定で対応 |
| 工数が予想超過 (1 重賞 5 時間+) | 中 | 5/18 計測後、 5/22+ 半自動化を必須化 |
| 規約変更で 録画 NG 化 | 低 | JRA-VAN 規約 monitor、 変更検知次第 即停止 |
| OBS websocket 接続失敗 | 中 | OBS GUI 手動録画 + Playwright 操作 で代替 |
| 動画 features の AUC 寄与 想定以下 | 中 | Phase 16 で +0.030-0.040 想定、 実 +0.015 でも accept (V21 投入条件: V20 + 0.005) |
| netkeiba 動画 同様に block される | 既 block | RV (公式) は契約内 = block されない |

---

## 6. 投資保護 (絶対遵守)

| 項目 | 状態 |
|------|------|
| V15 model 不変 | ✅ (Phase 21E は調査 + 設計のみ、 model 触らず) |
| predict_core.py 不変 | ✅ |
| 5/9 + 5/10 案 B 改 ROI | (別 phase で確定) |
| RV 月額 ¥550 | 5/10 加入済 |
| Phase 16 既存 skeleton 活用 | ✅ (rv_video_downloader.py / predict_core_v21.py 流用) |

---

## 7. 5/14 commit + Discord plan

- 5/14 commit: tools/rv_video_capture.py skeleton + obs-websocket-py + PoC 手順
- 5/15 PoC 開始 (5/17 重賞用 録画 試行)
- 5/18 結果 sum-up doc (data/v21/phase21f_poc_result.md 想定)

---

## 8. 関連 doc

- [phase21e_jvlink_video_api.md](phase21e_jvlink_video_api.md) — A 結果
- [phase21e_rv_software.md](phase21e_rv_software.md) — B 結果
- [phase21e_browser_scraping.md](phase21e_browser_scraping.md) — C 結果
- [phase16_summary.md](phase16_summary.md) — Phase 16 V21 candidate 237 features
- [JRA_VAN_RV_TRIAL_GUIDE.md](../../docs/JRA_VAN_RV_TRIAL_GUIDE.md)
