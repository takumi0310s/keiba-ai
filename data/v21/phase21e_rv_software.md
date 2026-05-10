# Phase 21E B: RV ソフト 経由 動画取得 調査

> Session #88 (2026-05-11) Phase 21E B
> 結論: ★ RV は Mpeg4 ストリーミング 視聴 only、 公式 download 機能なし、 FAQ で 「動画を保存することはできません」 明記 ★

---

## 1. 結論 (一行)

**RV (JRA レーシングビュアー、 ¥550/月、 5/10 加入済) は 専用ソフト 不要 ブラウザ視聴 のみ。 公式 download 機能 0、 ファイル保存 0。 PoC は画面録画 (D で詳述) で対応。**

---

## 2. RV service 仕様 (公式まとめ)

| 項目 | 内容 |
|------|------|
| 配信形式 | **Mpeg4 ストリーミング** (HLS / m3u8 ベース 推定、 公式記載なし) |
| 配信 timing | パドック: 締め切り 15 分前まで / レース: ゴール後約 3 分 / パトロール: ゴール後約 40 分 |
| 視聴環境 | ブラウザ (Edge/Chrome/Firefox/Safari) + iOS Safari + Android Chrome |
| **専用ソフト** | **なし** (Web ブラウザで完結) |
| **ダウンロード機能** | **なし** (公式 FAQ で禁止明記) |
| 月額 | ¥550 (RV 単独) / ¥1,430 (NEXT 込み) |

[公式 FAQ](https://prc.jp/jraracingviewer/support/qa.html) より:
> Q. 動画は保存できますか？
> A. 「動画を保存することはできません」

---

## 3. 動作環境 詳細 ([公式 環境ページ](https://prc.jp/jraracingviewer/intro/environment_mp4.html))

### PC
| OS | ブラウザ |
|----|---------|
| Windows 8.1/10/11 | Edge / Chrome / Firefox 最新 |
| macOS 10.12+ | Safari / Chrome / Firefox 最新 |

### モバイル
| OS | ブラウザ |
|----|---------|
| iOS / iPadOS 13+ | Safari 最新 |
| Android 7.0+ | Chrome 最新 |

### 共通要件
- JavaScript 有効
- Cookie 対応
- SSL (TLS 1.2) 対応
- 10 Mbps+ 高速回線推奨

---

## 4. 配信 content (V21 動画 features に必要なもの 全部 cover)

| content | RV 提供 | V21 features 必要性 |
|---------|--------|------------------|
| パドック動画 | ✅ (締め切り 15 分前まで) | ★★★ パドック CNN (12 features) |
| レース動画 (正面/側面) | ✅ (ゴール後約 3 分) | ★ レース後 features (post-race) |
| パトロール動画 (各 corner) | ✅ (ゴール後約 40 分) | ★★★ パトロール YOLO (8 features、 検証用) |
| 調教動画 (重賞 出走予定馬) | ✅ (水曜・木曜配信) | ★★★ 調教 keypoint (10 features) |
| 返し馬動画 | ❌ (RV では未提供、 NEXT 動画オプション に移行検討必要) | ★★ 返し馬 features (V21 候補) |

★ V21 動画 features (30 件) のうち **28 件は RV 配信で cover 可能** (返し馬 2 件のみ要追加検討)。

---

## 5. 「専用ソフト」 経由 download 試行 結果

### 5.1 専用 download ソフトの有無
- **JRA-VAN 公式 download ソフト**: 存在しない (RV は Web のみ)
- **third-party download tool**: 利用規約違反 + DRM 解除 risk → NG
- **Stream Recorder 系 ブラウザ拡張**: 動作可能性あるが規約違反 + ban risk

### 5.2 公式が推奨する視聴方式
**ブラウザ ストリーミング 視聴のみ**。

→ 公式手段では PC ローカル に video file 保存 0 件。

---

## 6. 個人 AI 学習用 PoC の現実解

### 公式範囲内 (推奨)
1. **ブラウザで Mpeg4 ストリーミング 視聴**
2. **OS 画面録画機能 で個人録画** (Mac: QuickTime / Windows: Game Bar / Xbox Game Bar)
3. **ローカル保存** → YOLOv8 / DLC で features 抽出
4. **私的複製範囲** (著作権法 30 条): 個人 AI 学習なら OK の可能性高 (ただし配布禁止)

### 規約違反 (NG)
- 自動 m3u8 download (大量 DL → ban)
- DRM 解除 (2012 改正で違法化)
- 公開 / 再配信 / 他者共有

→ **詳細は C (ブラウザ scraping) + D (推奨 method) で深掘り**。

---

## 7. 5/14-5/15 PoC plan (RV 視聴ベース)

| 日 | 作業 |
|----|------|
| 5/14 (水) | 重賞 出走予定馬の 調教動画 配信開始確認 (5/17 ヴィクトリアM など) |
| 5/15 (木) | 1 重賞 で 調教動画 5-10 動画 視聴 + Mac QuickTime 画面録画 試行 |
| 5/16 (金) | パドック動画 視聴 試行 (締め切り 15 分前まで) |
| 5/17 (土) | パドック + レース 視聴、 録画 → file format / 容量 / fps 確認 |
| 5/18 (日) | 京王杯SC でも同 試行、 weekend 累計 5-10 動画 + 累計容量 計測 |

→ PoC 結果は次回 Phase 21F で総括。

---

## 8. 関連 doc

- [JRA_VAN_RV_TRIAL_GUIDE.md](../../docs/JRA_VAN_RV_TRIAL_GUIDE.md)
- [PHASE_4_VIDEO_REPLAN_v2.md](../../docs/PHASE_4_VIDEO_REPLAN_v2.md)
- [phase16_setup.md](phase16_setup.md)
