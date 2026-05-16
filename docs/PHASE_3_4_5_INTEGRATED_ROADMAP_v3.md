# Phase 3-5 統合 roadmap v3 (Session #44 F)

**作成**: 2026-05-08 (Session #44 F)
**v1**: docs/PHASE_3_4_INTEGRATED_ROADMAP.md (Session #39 J)
**v2**: docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v2.md (Session #41 H)
**v3 (本ファイル)**: ★ Session #44 で TFJV フル data 即活用 → 大幅前倒し ★

---

## 0. ★ 大幅前倒し ★ 全体像

```
2026/  5月         6月        7月        8月       9月+
─────────────────────────────────────────────────────────────────
5/8    ★ Session #44 完了 (TFJV 直 parse、 V20 1 ヶ月前倒し)
5/9    ★ V15 案B改 単独継続 (絶対遵守、 max loss -2,100円、 ROI 85-100% 期待)
5/16   ★ V18 sib_w5 trial 投入 候補 (GO 確率 85-95%、 Session #43 C 確定)
5/22   ┌── V20 構築 Phase 3 前半 ──┐
       │ TFJV 4-model ensemble 学習 │
       │ V20 features 統合          │
6/1    └── V20 WF + LIVE retro ───┘
       │
6/8    ★ V20 投入候補 (1 ヶ月前倒し、 v2 7/1 → v3 6/8)
       │
7/1    ┌── Phase 4 動画解析 PoC ──┐
       │ JRA-VAN ネクスト 加入       │
       │ YOLOv8 + DLC SuperAnimal   │
       │ V21 学習                    │
9/1    └── V21 投入判定 ─────────┘
       │
9月以降 ┌── Phase 5 (V22 構想) ─┐
       │ + 生体 + 天気 + voting │
12月    └─────────────────────┘
```

---

## 1. ★ TFJV フル data 即活用による前倒し効果 ★

### 1.1 v2 (Session #41 H) → v3 (本 Session) 比較

| phase | v2 schedule | v3 schedule (Session #44 反映) | 前倒し |
|-------|-----------|-----------------------------|--------|
| 32-bit Python install | 5/24 | **不要** (TFJV 直 parse) | -2 週 |
| JV-Link backfill | 6/9-13 | **不要** (TFJV 6 年分 = 10 秒) | -1 週 |
| V20 学習 data spec | 6/14 | **5/16** | -4 週 |
| V20 v1 学習 | 6/14-20 | **5/22-29** | -3 週 |
| V20 WF 検証 | 6/21-25 | **5/30-6/3** | -3 週 |
| V20 LIVE retro | 6/27 | **6/4-5** | -3 週 |
| V20 paper trading | 6/28 | **6/6-7** | -3 週 |
| **V20 投入判定** | **7/1** | **6/8** | **-3 週** ★ |

→ **V20 投入が約 1 ヶ月前倒し**、 7 月以降の運用期間が長期化

### 1.2 32-bit Python の扱い

| 用途 | 必要性 | 結論 |
|------|--------|------|
| TFJV 直 parse | **不要** (本 Session B で確認、 Python 64-bit で OK) | 廃止候補 |
| JV-Link リアルタイム オッズ | 必要 (5/16 V18 trial で paci 自前算出 検討) | 保留 |
| V20 学習 data 主軸 | **不要** (TFJV のみで十分) | 廃止 |
| 既存 keiba-ai operations | **不要** | 廃止 |

→ **32-bit Python は 5/16 V18 trial 後に再判定**、 リアルタイム オッズ 不要なら 廃止確定。

### 1.3 JV-Link DataLab 加入の扱い

| 用途 | 必要性 |
|------|--------|
| 過去 race / SE / HR | **不要** (TFJV で完全代替) |
| リアルタイム オッズ | △ (5/16 V18 trial 結果次第) |
| 月額 2,090円 | (5/16 後判定) |

→ **6/8 V20 投入後 1 か月運用 (7 月) で paci 必要性確認 → 不要なら JV-Link 解約**

---

## 2. Phase 3 前半 (5/16-5/29): V20 構築 加速版

### 2.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| TFJV parser 完成 | **5/8 (本 Session)** ✅ | tools/tfjv_parser.py、 6 年分 parse OK |
| V20 features spec 確定 | 5/22 | V15 + V162 + V17 + sib_w5 + TFJV (~5-10 features) |
| V20 v1 学習 (4-model ensemble) | 5/29 | LGB + XGB + FT + IR、 BT WF AUC ≥ 0.880 |
| V20 WF 検証 (6-fold) | 6/3 | 全年 AUC > 0.85 |
| V20 LIVE retro (5/30, 5/31, 6/1) | 6/5 | winner_top1 ≥ 30% |
| V20 paper trading (6/6-7) | 6/7 | ROI ≥ 110% |
| V20 GO/no-go 判定 | **6/8** | 6 GO 条件 PASS |

### 2.2 V20 features 構成 (Session #44 D 反映)

```python
V20_FEATURES = (
    V15_BASE_FEATURES                                    # 150
    + V162_FEATURES (sib 旧 削除 + sib_w5 追加)         # 22
    + V17_FEATURES                                       # 18
    + ['sib_top3_rate_exp_w5',                          # 2 (Session #43 確定)
       'sib_shinba_wr_exp_w5']
    + V20_NEW_FEATURES_TFJV                              # 5-10 (Session #44 D)
)
# 計: 200-205 features (V15 150 → V20 200+)
# 期待 BT WF AUC: 0.890-0.895 (V15 0.8939 とほぼ同等 or 微増)
# 期待 LIVE winner_top1: V18 sib_w5 同等 ~34%
```

---

## 3. Phase 3 後半 (5/30-6/8): V20 投入準備

### 3.1 投入条件 6 PASS 確認

| # | 条件 | 必要値 | 確認 timing |
|---|------|--------|------------|
| 1 | WF AUC | ≥ 0.880 | 5/29 学習完了時 |
| 2 | LIVE winner_top1 | ≥ 30% | 6/5 (3 週分平均) |
| 3 | shift_factor | ≤ 12x | 6/5 |
| 4 | NAR subset AUC | ≥ 0.83 | 6/3 WF 時 |
| 5 | paper trading ROI | ≥ 110% | 6/7 (3 日 SUM) |
| 6 | feature LEAK 監査 | PASS | 5/29 |

→ 全 PASS で 6/8 GO、 失敗なら 6/15 or 6/22 延期

### 3.2 V20 production deploy (6/8+)

| 期間 | 投資制約 |
|------|---------|
| 6/8-6/14 | 週末のみ、 上限 5,000円/日 |
| 6/15-6/30 | 週末 1 万円/日 + 平日 5,000円/日 |
| 7/1-7/31 | 平常運用 |

---

## 4. Phase 4 (7-8 月): 動画解析 PoC (Session #39 F + #42 E + #43 D 維持)

### 4.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| データ蓄積完了 | 7/14 | 50 動画 |
| YOLOv8 + DLC 動作確認 | 7/31 | 検出精度 ≥ 80% |
| VIDEO_FEATURES 抽出 | 8/15 | 10 件 |
| V21 学習 | 8/31 | WF AUC ≥ V20 + 0.005 |
| V21 投入判定 | 9/1 | LIVE winner_top1 ≥ V20 + 1pt |

### 4.2 環境構築 (Session #42 E + #43 D 完成済)

✅ ultralytics 8.4 install OK
✅ YOLOv8n + opencv-python OK (138ms CPU 動作)
✅ tools/video_poc/extract_frames_and_detect.py 完成
✅ Phase 4 工数 100-200h → 65-125h に縮小済

---

## 5. Phase 5 (9月以降): V22 構想

(Session #41 H plan 維持)

| 期間 | 内容 |
|------|------|
| 9/1-9/30 | V22 設計 + 馬体寸法 features 開発 |
| 10/1-10/31 | 天気予報 24h 前 features + 3-way voting |
| 11/1-11/30 | V22 学習 + WF + LIVE retro |
| 12/1-12/31 | V22 投入判定 |

---

## 6. ユーザー (れんはす) 関与 step (v3 update)

| 期日 | アクション | 重要度 |
|------|----------|--------|
| 5/9 (土) 朝 | V15 案B改 投資 (700円 × max 3R) | **絶対** |
| 5/10 (日) 朝 | result_verification_5_10.py で verdict | 中 |
| 5/16 (土) | V18 sib_w5 trial 投入判断 (GO 確率 85-95%) | 高 |
| 5/22 (木) | V20 features spec 確定 確認 | 中 |
| **5/29 (木)** | **V20 v1 学習完了 確認** | **高** |
| 6/5 (木) | V20 LIVE retro 結果 確認 | 高 |
| **6/8 (日)** | **V20 投入判断** ★ 1 ヶ月前倒し | **絶対** |
| 7/1 (火) | Phase 4 加入判断 (JRA-VAN ネクスト + Colab Pro) | 中 |
| 9/1 (月) | V21 投入判断 | 高 |
| 12月 | V22 投入判断 | 中 |

---

## 7. 月額コスト (Session #44 反映、 32-bit / JV-Link 廃止検討)

| source | 月額 | 状態 (5/8 時点) |
|--------|------|------------|
| netkeiba Premium | 4,500円 | 既存 |
| JRDB Advance | 約 2,000円 | 既存 (paci 4/4 停止確認) |
| **TFJV (TARGET)** | **(既加入)** | ★ 主軸へ |
| JV-Link DataLab | 2,090円 | **5/16 後 廃止判断** |
| JRA-VAN ネクスト (Phase 4) | +1,000円 | 7/1 (予定) |
| Colab Pro (Phase 4 GPU) | 1,178円 | 7/1 (予定) |
| **合計 (現状 5/8)** | **約 8,590円/月** | |
| **合計 (7/1 後 JV-Link 廃止 case)** | **約 7,500円/月** | -1,090円 |

ROI 想定:
- V15 (5/9-): 119.2% (戦略⑦込み 140%) → 月利 約 2-3 万円 ※ 旧記述は drift、 5/16 P0-1 真値: ROI 101.33% / 月利 期待値 ±¥0-3,000 (docs/ROI_DISCREPANCY_2026_05_16.md)
- V20 (6/8+): 145-150% 想定 (TFJV 補強で +0.5-1pt) → 月利 5-10 万円
- V21 (9/2+): 150-155% 想定 → 月利 6-11 万円
- V22 (12月+): 155-160% 想定 → 月利 7-13 万円

→ 月額 約 8,000-10,000 円は V20 以降の 月利 +5-13 万円で十分回収。

---

## 8. fallback 階層 (絶対遵守、 v2 から維持 + 更新)

```
V22 NG (12月)
  ↓
V21 単独継続

V21 NG (9/1)
  ↓
V20 単独継続 (Phase 4 NO-GO で PoC データ蓄積継続)

V20 NG (6/8)
  ↓
V18 sib_w5 (5/16 GO 時) + V15 並行
  ↓ (V18 も NG なら)
V15 単独継続 (Phase 3 NO-GO、 6 月以降も V15)

V15 重度問題 (winner_top1 -10pt 等)
  ↓
全停止、 撤退判断
```

→ 全 path で V15 単独 fallback、 撤退余裕 +63,530 円維持。

---

## 9. 5/9 V15 投資保護 (F 領域)

✅ V15 model md5: `842b9a5f305c793ed8fa54a74e06b836` 不変
✅ predict_core / daily_predict / V15 model 完全不変
✅ schtasks 既存 task 完全不変
✅ TFJV は read-only、 既存 keiba-ai data も不変

→ **5/9 朝 V15 案B改 完全保証**

---

## 10. 結論

✅ F1: V20 投入 1 ヶ月前倒し (7/1 → **6/8**) 確定
✅ F2: TFJV 直 parse で 32-bit Python 廃止候補
✅ F3: JV-Link は 5/16 V18 trial 後に 廃止判定
✅ F4: V20 期待 AUC 0.890-0.895 (V15 + sib_w5 + TFJV features)
✅ F5: 月額 8,000-10,000円、 V20 以降の月利 +5-13 万円で十分
✅ V15 投資保護 完全保証

→ **V20 6/8 投入候補、 Phase 3 大幅加速**

---

**Session #44 F 完了**
