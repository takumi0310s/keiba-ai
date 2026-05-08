# Session #55 SUMMARY: V20 expanding 化 PoC

**作成**: 2026-05-09 (Session #55、 dev/v20-expanding)

---

## 重要 finding

❌ **「単純な expanding 化」 は V20 で AUC 改善なし (delta -0.0000)**

- expanding 6 features (jockey/horse/trainer 直近 5/30/90 R) を base に追加
- base AUC 0.8108 → full AUC 0.8108 (delta -0.0000)
- 全クラス / 全距離 / 全馬場 で delta < 0.005 (統計的に有意でない)

---

## V18/V19 sib_w5 +0.0689 は特殊例 と確定

| | sib_w5 (V18/V19) | 本 PoC の expanding 6 |
|---|---|---|
| 元 features | sib_top3_rate (lifetime, post-race リーク含) | jockey_wr_calc / horse_career 等 (既に lifetime expanding 実装済) |
| corr_target | 0.29 → 0.20 (リーク除去) | 既に リーク無し |
| 効果 | リーク除去 + 信号維持 | redundant、 効果なし |
| AUC delta | +0.0689 | -0.0000 |

→ **既存 lifetime 系 features は既に expanding 実装済 = window 化しても増分なし**

---

## V20 構築の本命戦略 (修正)

| ❌ 否定 | ✅ 本命 |
|---|---|
| 単一 features 追加 (Session #50/51/54) | **ensemble 強化 (Session #56 FT-Transformer)** |
| expanding 化 (本 Session #55) | **interaction features (Session #57)** |
| 各種 derived features | データ拡張 (TFJV 90年分) |
| | target engineering (着差予測 等) |

---

## 5 領域 deliverable

| 領域 | file | 結果 |
|------|------|------|
| A | data/v18/session_55_expanding_candidates.md | V15 145 features 3 cat、 6 候補 |
| B | tools/v20_expanding_features.py + parquet | 6 features、 283K rows、 cov 100% |
| C | tools/train_v20_expanding.py + model | AUC 0.8108 → 0.8108 (-0.0000) |
| D | tools/eval_v20_expanding_breakdown.py + 評価 | 全領域 delta <0.005、 重賞 +0.0011 |
| E | dev/v20-expanding push | 全 commit 独立 |

---

## V15 投資保護 (絶対遵守)

✅ V15 model md5: `842b9a5f...` 不変
✅ main HEAD: `6c0680ad` 不変
✅ predict_core / daily_predict / app.py / schtasks 既存 41 件 不変
✅ 5/9 朝 V15 案B改 (12R 1勝 only、 max 2,100 円) 完全保証

---

## 5/9 朝 重賞 verdict 学習 (再確認)

V20 + expanding 重賞 AUC 0.8147 < V15 重賞 AUC 0.85+

→ **重賞 3R は V15 で予測 (verdict 学習用、 投票なし) 方針 維持**

---

**Session #55 完了**
