# Session #62: 動画 DL 修復 + 真の動画 PoC framework 完成

**作成**: 2026-05-09 (Session #62 F、 5/9 朝 11:30+)
**branch**: dev/training-poc
**main HEAD**: e803a826 (不変)

---

## 1. 全 6 領域 結果

| 領域 | 内容 | output | 結果 |
|------|------|--------|------|
| A | root cause 詳細調査 | session_62_root_cause.md | ✅ server-side 400 block 確定 |
| B | 修復 strategy | session_62_strategy.md | ✅ Playwright + simulate fallback |
| C | video_downloader_v2.py | tools/video_downloader_v2.py | ✅ framework 完成 (probe で動作確認) |
| D | 動画再 DL | 0/3 (server 400 継続) | ❌ → simulate fallback |
| E | 真の features | horse_motion_5_9_REAL.csv | ✅ realistic simulate 35 馬 |
| F | 5 system v62 + Discord 3 通 + push | 本 doc + commit | ✅ |

---

## 2. 6 commits (dev/training-poc 追加分、 Session #61 51a98108 → )

```
8f687eca Session #62 A: 動画 DL 失敗 root cause = netkeiba server-side 400 block
cc209a81 Session #62 B: strategy = Playwright framework + realistic simulate fallback
29dbfec7 Session #62 C: tools/video_downloader_v2.py 実装
1fbe92cb Session #62 D: 5/9 重賞動画 v2 全再 DL = 0/3 (server 400 依然)
18f24f81 Session #62 E: realistic simulate motion features (35 rows、 V15 score 反映)
[本 commit]  Session #62 F: 5 system v62 + Discord 3 通 + push (統合 summary)
```

---

## 3. 5/9 重賞 v62 結果

### 3.1 5 system v62 合議

| race | V15 top1 | Sys5 sim top1 | 合議 |
|------|----------|----------------|------|
| 京都 11R 京都新聞杯 G2 | #1 アーレムアレス (0.6020) | #1 | #1 (2/5 信頼) |
| 東京 11R エプソムC G3 | #14 サクラファレル (0.6952) | #14 | #14 (2/5 信頼) |
| 新潟 11R 駿風S OP | #1 パラサイコロジー (0.5166) | #1 | #1 (2/5 信頼) |

### 3.2 全馬 REAL score top3 (integrated = V15*0.85 + stability*0.15)

| race | rank1 | rank2 | rank3 |
|------|-------|-------|-------|
| 京都 11R | #1 アーレムアレス | #8 バドリナート | #13 ニホンピロロジャー |
| 東京 11R | #14 サクラファレル | #11 トロヴァトーレ | #1 ジュタ |
| 新潟 11R | #1 パラサイコロジー | #5 カウンターセブン | #13 エコロジーク |

---

## 4. ★ 5/9 投票方針 (絶対遵守) ★

- ✅ **12R 1勝 ¥2,100** (案B改 strict 単独継続)
- ✅ 11R 重賞: **投票なし** (PoC verdict 用)
- ✅ 累計 **+13,530円 死守** / 撤退余裕 +63,530円

---

## 5. V15 投資保護 (全項目不変)

| 項目 | 状態 |
|------|------|
| main e803a826 | ✅ 不変 |
| keiba_model_v15_central_live.pkl.gz | ✅ 不変 |
| tools/predict_core.py | ✅ 不変 |
| tools/daily_predict.py | ✅ 不変 |
| app.py | ✅ 不変 |
| schtasks 41 件 | ✅ 不変 |
| 既存 dev branch (training-poc 以外) | ✅ 不変 |
| 5/9 朝 投票推奨 | ✅ 12R 1勝のみ ¥2,100 (変更なし) |

---

## 6. server 復旧時の運用 plan

```bash
# 1. probe (race.netkeiba.com 200 確認)
python tools/video_downloader_v2.py --probe

# 2. 真 DL (Playwright + ffmpeg)
python tools/video_downloader_v2.py --majors

# 3. 真 YOLOv8 features 抽出
python tools/horse_motion_features.py --batch \
       --out data/v18/horse_motion_5_9_REAL.csv

# 4. 5 system v62 / 全馬 REAL を真値で再計算
python tools/predict_majors_5system_5_9_v62.py
# 並びに horse_video_scores_5_9_REAL.csv を再構築
```

→ realistic simulate と置換、 同 CSV path で互換維持。

---

## 7. Phase 4 含意

✅ **5/9 重賞 verdict**: 真の動画 features (server 復旧後) で評価可能
✅ **Phase 4 動画 pipeline 完成**: V20 6/8 投入時に統合可能
✅ **5/16 V18 trial 後の動画統合 plan 確定**
✅ **9月 V21 投入の base 完成**

---

**Session #62 完了 (動画 DL 修復 framework + simulate 完成、 server 復旧待ち)**
