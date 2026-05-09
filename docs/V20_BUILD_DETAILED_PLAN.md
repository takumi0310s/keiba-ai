# V20 構築 詳細設計

**作成**: Session #79 (2026-05-09)
**期間**: 2026-05-22 〜 2026-07-01
**目標**: V15 (AUC 0.886) → V20 4-model ensemble (期待 AUC **0.90025**)

---

## 1. 構築 timeline

| 日付 | task |
|------|------|
| 5/22 (木) | Sprint 6 開始 (V20 cache 構築) |
| 5/30 (金) | KKA features 統合 (Session #53 修復済) |
| 6/1 (日) | 4-model ensemble 統合 (Session #56 復活) |
| 6/8 (日) | V20 paper trade 開始 |
| 6/30 (日) | paper 評価 (30 日) |
| 7/1 (火) | V20 投入候補 判定 |

---

## 2. 構成要素 (確定済)

### 2-1. base data

| 項目 | 値 |
|------|----|
| source | TFJV 90 年分 (45,000+ files / 6 GB) |
| V20 PoC 確認 | 6 年分 320,000 records (10 秒 parse) |
| 蓄積開始 | 2026-05-10 (Session #71 全馬 score 保存) |
| base 完成 | 5/10-5/22 で 12 日分 + 既存 6 年 |

### 2-2. features (期待 200+)

| source | 件数 | 備考 |
|--------|------|------|
| V15 145 features | 145 | LEAK 除外後 (Session #51) の基盤 |
| KKA 12-15 件 | 12-15 | Session #53 修復済 (heavy / class / pace / season / dam_rensho) |
| 当日体重 | 1 | Session #48 B (70 分前公開) |
| **小計** | **約 158-161** | + interaction 拡張余地 |

### 2-3. LEAK 除外 (Session #51 list、 12 件)

```python
V20_LEAK_FEATURES = [
    'jrdb_sed_*',          # AUC 1.0 LEAK 確定
    'race_review_score',   # AUC 0.998
    # 他 10 件 (Session #51 全件 list)
]
```

学習時は `merge_v15_1_features(skip_skb=True, skip_leak=True)` で完全除外。

### 2-4. model 4-model ensemble (Session #56 復活)

| model | weight | 役割 |
|-------|--------|------|
| LGB (LightGBM) | 0.043 | base |
| XGB (XGBoost) | 0.043 | base |
| FT-Transformer | 0.087 | tabular DL |
| **IntraRace Attention** | **0.826** | ★主軸★ レース内相対関係 |

★ IntraRace Attention 単独 AUC 0.899 で V13.5b 全体 (0.8788) 超え ★

### 2-5. 学習環境

| 項目 | 値 |
|------|----|
| PyTorch | 2.11.0+cu126 |
| CUDA | enabled (RTX class GPU) |
| 総時間 | ~6 分 (Session #56 実績) |
| RAM | 32 GB 必要 |

---

## 3. 期待 AUC

| model | AUC |
|-------|----|
| V15 (現行 production) | 0.886 |
| V20 LGB alone (PoC) | 0.875 |
| V20 XGB alone | 0.870 |
| V20 FT alone | 0.866 |
| V20 IR alone ★ | **0.899** |
| **V20 4-model ensemble** ★ | **0.90025** ALL TIME BEST |

---

## 4. V20 投票 strategy (確定)

### 4-1. 案B改 strict 維持 (Session #69 確定)

| 項目 | 値 |
|------|----|
| 買い目 | 三連複 7 点固定 (1-2-5) |
| 投資 | ¥700 / R |
| max | 3 R / 日 = ¥2,100 |
| 戦略⑦ | 06_特別 / 京都 / 条件E / 条件B 除外 |

### 4-2. NO-GO 確定要素 (再採用しない)

| 項目 | session | 理由 |
|------|---------|------|
| V15.5 | #50 | NO-GO |
| expanding | #55 | delta -0.0000 |
| interaction | #57 | -2bp〜+1.8bp、 LGB 内部捕捉済 |
| +2 頭 | #69 | NO-GO |
| 動画 features | (server block) | Phase 4 で再挑戦 |

---

## 5. Eighth Kelly 適用 (V20 投入後)

| 段階 | 軍資金 | 1 R 投資 | max / 日 |
|------|--------|----------|----------|
| 現状 (V15) | +¥12,830 | ¥700 | ¥2,100 |
| V20 投入直後 | +12,830 + α | ¥700 | ¥2,100 (慎重維持) |
| paper 30 日 OK 後 | 増額判定 | ¥2,000-3,000 | ¥6,000-9,000 |

★ 7/1 投入時は paper 結果次第で慎重に検討 ★

---

## 6. 投資保護 (絶対遵守)

- 5/9-7/1 期間中も V15 production **完全不変保証**
- predict_core.py / daily_predict.py / app.py 一切変更しない
- schtasks 既存 50 件 変更しない
- 撤退ライン: 累計 -¥50,000 (現在 +¥12,830、 余裕 +¥62,830)

---

## 7. 関連 doc

- [SPRINT_6_PLAN.md](SPRINT_6_PLAN.md) — Sprint 6 詳細
- [V20_VS_V15_COMPARISON.md](V20_VS_V15_COMPARISON.md) — paper 比較 framework
- [V20_DEPLOYMENT_CHECKLIST.md](V20_DEPLOYMENT_CHECKLIST.md) — 7/1 投入 checklist
- [PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md](PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md) — Phase 3-5 全体
