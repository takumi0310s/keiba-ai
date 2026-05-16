# Phase D-3 — 戦略⑦ 除外 R の V21 対応

date: 2026-05-16
session: Terminal D (Phase D)

---

## 1. 現状 (戦略⑦ on V15)

`tools/race_auto_notify.py` の `predict_and_notify` で 以下 4 categories を **投票対象から除外**:

| 除外条件 | 除外理由 | 月次 損失 削減 (試算) |
|---|---|---|
| `06_特別` (G/L/OPEN 外の 平場特別) | -9,470 円 損失源 | ~9,470 円 |
| `京都` | course_renovated 修復前 (4/27 修正済、 5/11+ 再評価予定) | data 蓄積待ち |
| 条件 E (頭数 ≤ 7) | sample 少、 統計信頼性低 (CI [103%, 133%]) | 過小 sample 投票回避 |
| 条件 B (重〜不良馬場) | sample 少 (N=847) | 過小 sample 投票回避 |

→ V15 production の投票 race 数を 約 19% 削減し、 ROI 119.2% → 140.3% (戦略⑦込み) を実現。

---

## 2. V21 で 解放したい R (target)

動画 features 30 が入ることで以下 が改善する 可能性:

| 除外 category | V21 で 解放できる 期待 | 確認方法 |
|---|---|---|
| 06_特別 (平場) | 動画 (パドック / 調教) で 馬体仕上がり 拾えれば 高 ROI 候補 抽出可 | 6/8-6/15 paper trade |
| 京都 | course_renovated は V15 already 修正、 動画 で 馬個別評価 上乗せ | 6/15-6/22 paper |
| 条件 E (少頭数) | 動画 で 馬体差別化 → 確度上昇 → 投票可 | 6/22-6/29 paper |
| 条件 B (馬場悪) | 動画 (パドック sweat / 蹄健全度) で 適応判定 | 6/22-6/29 paper |

---

## 3. ★ sub-model 候補 ★ (V21 を カバー外 R に拡張)

| sub-model | scope | data 量 | WF 可能性 | 採用優先度 |
|---|---|---|---|---|
| **重賞専用 model** (G1/G2/G3) | 年 ~200 R | 6 年 = 1,200 R → 動画 coverage 高 | 動画揃いやすい (RV 配信 必須) | ★ 高 ★ |
| **06_平場特別 専用** | 月 ~20 R | 6 年 ≈ 1,400 R | ◯ | 中 |
| **障害専用** | 年 ~120 R | 6 年 = 720 R | △ (sample 不足) | 低 (Phase 4+ 検討) |
| **短距離 < 1000m 専用** | 年 ~50 R | 6 年 = 300 R | × (sample 不足) | 採用 NG |
| **少頭数 (≤ 7 頭) 専用** | 年 ~200 R | 6 年 = 1,200 R | ◯ | 中 (条件 E 解放用) |

→ 採用 順位:
1. ★ **重賞専用 model** ★ (V21 stacking の WF 第一弾)
2. 06_平場特別 (06_special_meta_lgb.pkl.gz)
3. 少頭数 (E_meta_lgb.pkl.gz)
4. 障害 / 短距離 は data 不足 → 保留

---

## 4. ★ V21 投票 path 設計 ★ (戦略⑦除外 R 対応)

```
race_id 入力
   │
   ▼
classify_race_condition() → cond_key (A/B/C/D/E/X) + race_class (G1/G2/G3/06_特別/...)
   │
   ▼
┌──────────────────────────────────────────┐
│ V21 voting gate                          │
│                                          │
│   if race_class in (G1, G2, G3):         │
│       → V21 重賞 meta-LGB (新規)         │
│   elif race_class == '06_特別':           │
│       → V21 06_平場 meta-LGB (新規)      │
│   elif cond_key == 'E':                  │
│       → V21 少頭数 meta-LGB (新規)       │
│   elif cond_key == 'B':                  │
│       → V21 馬場悪 meta-LGB (新規)       │
│   else:                                  │
│       → V21 main meta-LGB (default)      │
└──────────────────────────────────────────┘
   │
   ▼
動画 features 取得 + V15 score → V21 score
   │
   ▼
GO/no-go: V21 paper trade で sub-model ごと validate
```

---

## 5. 戦略⑦ 切替 条件 (V21 投入後)

| 切替 trigger | 旧 (V15 戦略⑦) | 新 (V21 production) |
|---|---|---|
| 06_平場特別 投票 | 完全除外 | V21 06_平場 sub-model で paper ROI ≥ 130% 確認後、 復帰 |
| 京都 投票 | course_renovated 修復で復帰済 | そのまま (V21 で main meta-LGB 経由) |
| 条件 E 投票 | 完全除外 | V21 少頭数 sub-model で paper ROI ≥ 130% 確認後、 復帰 |
| 条件 B 投票 | 完全除外 | V21 馬場悪 sub-model で paper ROI ≥ 130% 確認後、 復帰 |

→ 復帰判定は **paper trade 4 週** (6/1-6/30) の sub-model 別 ROI で 厳格に。

---

## 6. coverage 制約

sub-model 採用には WF data + 動画 coverage の両方 が必要:

| sub-model | WF data (V15 features) | 動画 coverage |
|---|---|---|
| 重賞 | 6 年 1,200 R 即利用可 | RV 全 重賞配信、 6/1+ ほぼ 100% 取得可 |
| 06_平場特別 | 6 年 1,400 R 利用可 | RV 標準配信、 6/15+ で取得可 |
| 少頭数 | 6 年 1,200 R 利用可 | 動画 fast、 取得 priority 上げ |
| 馬場悪 | sample 少 (847 R)、 WF 注意 | 雨天 RV カバー次第 |

→ 重賞 → 06_平場 → 少頭数 の順 で sub-model 学習を進める。

---

## 7. 設計の不変保証

- V15 戦略⑦ 除外 logic は touch しない (`race_auto_notify.py` 改変 NG)
- V21 sub-model は すべて 新規 `tools/v21/` 配下 で実装
- paper trade engine (新規 `tools/v21/paper_trade_v21.py` 等) で sub-model 比較
- V21 GO 判定後、 race_auto_notify_v21.py (新規) で本番投票 path を切替

---

## 8. fabrication 防止

- 「戦略⑦ で +21.1pt 改善」は CLAUDE.md / cumulative_results.csv の 実測値 (4/27 適用後)
- 「V21 で 06_平場 復帰で +X pt」 等の数値は paper trade 実測後に追記する (現状空欄)
- sub-model の 期待 ROI は **設計時 想定**、 実測ではない
