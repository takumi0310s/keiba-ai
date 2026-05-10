# Phase 16 統合 summary: V21 candidate 237 features (5/10)

> Session #87 Phase 16 統合 (5/10 20:00+)
> 全 5 領域 (A/B/C/D/E) 完了

---

## 1. V21 candidate features 構成

| 領域 | features | source |
|------|---------|--------|
| V15 (本番) | 150 | jra_races_full / netkeiba / JRDB(既存) |
| Phase 11 (JRDB 拡張) | +15 | 外厩集計 / 時系列オッズ / 騎手マスタ拡張 |
| Phase 12 (JRA-VAN DataLab) | +17 | O1+O2+O5 / RA+BT / SE / WE+WH / UM+SK+BR |
| Phase 13 (netkeiba マスター) | +25 | AI 展開 / 波乱度 / 個別ラップ / トラックバイアス / 走行距離 |
| **V20 candidate** | **207** | — |
| ★ Phase 16 (RV 動画) ★ | **+30** | パドック CNN (12) + パトロール YOLO (8) + 調教 keypoint (10) |
| **★ V21 candidate ★** | **★ 237 ★** | — |

---

## 2. 期待 AUC

| Model | AUC (BT WF) | 改善 |
|-------|------------|------|
| V15 (現行) | 0.8939 (公称、 BT) | baseline |
| V20 candidate (Phase 11+12+13) | 0.910-0.920 | +0.020-0.030 |
| ★ V21 candidate (V20 + 動画) ★ | **0.940-0.950** | **+0.050-0.060** |

動画 features 期待寄与:
- パドック CNN: +0.030-0.060
- パトロール YOLO: +0.020-0.040
- 調教 keypoint: +0.030-0.060

★ 重複考慮 + 飽和 → V20 → V21 で +0.030-0.040 が現実的 ★

---

## 3. 実装 物 全 list

### 3.1 新規 file

| file | 内容 | 状態 |
|------|------|------|
| tools/rv_video_downloader.py | RV 動画 metadata 管理 (skeleton) | ✅ self-test pass |
| tools/predict_core_v21.py | 30 動画 features skeleton | ✅ self-test pass |
| data/v21/phase16_setup.md | RV 環境 + 5/15+ trial plan | ✅ |
| data/v21/phase16_paddock_cnn.md | EfficientNet-B3 設計 | ✅ |
| data/v21/phase16_patrol_yolo.md | YOLOv8m + DeepSORT 設計 | ✅ |
| data/v21/phase16_chokyou_keypoint.md | DLC SuperAnimal HORSE-10 設計 | ✅ |
| data/v21/phase16_summary.md | 本 doc | ✅ |

### 3.2 ★ model file は本 Phase で生成しない ★

| 想定 file | 理由 |
|-----------|------|
| models/v21/paddock_cnn_efficientnet.pth | 実 sample 動画 0、 学習未実行 |
| models/v21/patrol_yolo_v8m.pt | 同上 |
| models/v21/chokyou_keypoint_mmpose.pth | 同上 |

→ 7/1-9/2 で実学習後 生成、 本 commit に含めない (Git LFS 設定 不要)

---

## 4. 全 GPU 学習 時間 試算 (7/1-9/2)

| 領域 | GPU 時間 | 着手 |
|------|---------|------|
| パドック CNN (EfficientNet-B3) | 18-35h | 7/1-8/15 |
| パトロール YOLO (YOLOv8m + DeepSORT) | 13-25h | 7/15-9/2 |
| 調教 keypoint (DLC SuperAnimal) | 10-20h | 7/1-9/2 |
| ★ 合計 ★ | ★ 41-80h ★ | — |

RTX 4070 Ti SUPER 16GB は 余裕あり、 weekend 集中で 1-2 ヶ月で完了。

---

## 5. ★ Phase 16 セッション内では実 GPU 学習を実行しない ★

理由:
- 実 sample 動画 = 0 (5/15+ RV trial で蓄積)
- 12-24h GPU 学習 = 1 セッション内不可
- 人手 annotation 必要 (各 head 50-100 動画)
- torchvision NMS CUDA 課題 (Session #42、 7/1+ 修正 plan)

→ 本 Phase = ★ 設計 + skeleton + plan 確定 ★
→ 5/15+ trial / 7/1+ 学習 / 9/2 V21 投入候補

---

## 6. ★ V15 投資保護 完全 ★

| 項目 | 状態 |
|------|------|
| tools/predict_core.py | ✅ 不変 |
| tools/daily_predict.py | ✅ 不変 |
| app.py | ✅ 不変 |
| V15 model file (.pkl.gz) | ✅ 不変 |
| schtask | ✅ 不変 |
| 戦略⑦ / 案 B 改 logic | ✅ 不変 |
| 累計 +¥14,140 | ✅ 維持 (撤退余裕 +¥63,420) |

V21 = 完全に動画上乗せ phase、 V20 / V15 fallback 必須。

---

## 7. ★ 撤退条件 ★

RV trial 1 ヶ月 (5/15-6/15) で:
- 動画品質 不足 → V21 中止、 V20 単独運用
- features 抽出 困難 → PoC 縮小、 V21 延期
- AUC 改善 < +0.005 → V21 不採用

---

## 8. 結論

✅ A. RV 動画 download 環境 (skeleton + 規約遵守)
✅ B. パドック CNN 12 features (EfficientNet-B3 設計)
✅ C. パトロール YOLO 8 features (YOLOv8m + DeepSORT 設計)
✅ D. 調教 keypoint 10 features (DLC SuperAnimal HORSE-10 設計)
✅ E. tools/predict_core_v21.py 統合 (30 features, self-test pass)

→ ★ V21 candidate 237 features 達成 (skeleton)、 9/2 投入候補 ★
→ ★ V15 完全保護 ★
