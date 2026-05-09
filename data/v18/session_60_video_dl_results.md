# Session #60 B: 動画 download 結果

**作成**: 2026-05-09 (Session #60 B)
**実行**: `tools/video_downloader.py` を 5/9 重賞 3 race で実行

---

## 1. 結果サマリー

| race | race_id | 結果 | エラー |
|------|---------|------|--------|
| 京都新聞杯 G2 | 202608030511 | ❌ HTTP 400 | yt-dlp generic extractor で video URL 抽出不可 |
| エプソムC G3 | 202605020511 | ❌ HTTP 400 | 同上 |
| 駿風S OP | 202604010311 | ❌ HTTP 400 | 同上 |

**取得**: 0/3、 **失敗**: 3/3

---

## 2. root cause

`tools/video_downloader.py` は `https://race.netkeiba.com/race/movie.html?race_id=...` を yt-dlp の generic extractor に投げているが、 movie.html は **video player 込みの HTML page**。
yt-dlp はそこから video stream URL を抽出できず HTTP 400 を返す (Premium cookie あっても解決せず)。

正しい実装には以下が必要:
1. Playwright でページを開いて video tag / HLS URL を JS 実行後に抽出
2. または netkeiba の video API endpoint を逆engineering (`movie.html` 内部で叩いている JSON)
3. 抽出した stream URL を改めて yt-dlp に渡す

→ Session #52 / #60 の現実装ではこの 2-step が無く、 generic extractor 直接で **失敗確定**。

---

## 3. 本 Session の対応 (failure mode follow)

タスク仕様 「B 失敗時: 静止画 fallback (Session #52 実装済)」 を確認したが、 静止画 fallback は **実装されていない** (タスク説明の言及のみ)。

→ Area C は Session #52 の **simulate 値で 5 system v2 既定動作** に従う:
- horse_motion_features.py が動画なし → simulate モード
- predict_majors_5system v2 が System 5 を simulate 値で運用
- これが Session #52 D 時点の動作と同じ

---

## 4. NEXT (改善 candidate、 本 Session 範囲外)

| 改善 | 工数 | 効果 |
|------|------|------|
| Playwright で video tag 抽出 → yt-dlp | 1-2h | 動画 DL 実成功 |
| netkeiba video API 逆 engineering | 2-3h | より軽量 |
| 静止画 (パドック写真) fallback impl | 30min | 動画なし時の何らか signal |

→ 5/9 重賞 PoC は simulate 値で実施、 改善は Phase 4 (7-8 月) で別途。

---

## 5. 5/9 投資 完全保護

- ✅ V15 model 不変
- ✅ predict_core / daily_predict / app.py 不変
- ✅ schtasks 41 件 不変
- ✅ 12R 1勝 ¥2,100 (案B改 単独継続) 絶対
- ✅ 11R 重賞 投票しない (PoC verdict のみ)

---

**Session #60 B 完了 (動画 DL は failure、 simulate 値で C に進む)**
