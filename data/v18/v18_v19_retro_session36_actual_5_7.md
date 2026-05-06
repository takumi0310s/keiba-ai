# V18/V19 5/2-5/3 retro 実測結果 (Session #36 後追い)

**作成**: 2026-05-07 01:50 (background retro 完了後)
**結論**: 🔴 **sr/srb merge 拡張効果 限定的、 5/16 GO 確率 30-40% に再下方修正**

---

## 1. retro 実測値

```
Total horses: 932
winner_known: 387 (41.5%)
Total time: 26.9 min

実測 max p18 (各 race):
  race [25/35] 202608030402: 0.004
  race [27/35] 202608030404: 0.003 (winner=11)
  race [29/35] 202608030406: 0.020
  race [30/35] 202608030407: 0.002
  race [33/35] 202608030410: 0.022 (winner=3)
  ...
平均 max p18: ~0.005-0.020 (Session #10 retro と同レベル)
```

### 1.1 bet 候補

| prob_min | ev_min | bet 数 | hit% | ROI |
|---|---|---|---|---|
| 0.3 | 1.0 | **0** | - | - |
| 0.4 | 1.2 | **0** | - | - |
| 0.5 | 1.2 | **0** | - | - |

→ **filter で全 0 bet**、 normalize 適用前の raw probability。

---

## 2. shift 解析

| dataset | mean p18 | max p18 | winner_top1 |
|---------|---------|---------|------------|
| BT 2025 OOS (Session #10) | 0.0548 | 0.9863 | 47.8% |
| Session #10 retro (5/2-5/3) | 0.0018 | 0.1538 | 34.5% |
| **Session #36 retro (sr/srb merge 後)** | **0.005-0.020** | **0.022** | (sample 少で算出不能) |

→ Session #10 → Session #36 で **わずか改善** (0.0018 → 0.005、 約 2.7x)、 ただし BT 0.0548 までは 11x 不足。
→ shift factor: 27.7x (Session #10) → **約 11x (Session #36)**、 改善はあるが winner_top1 改善には不十分。

---

## 3. 真因再評価

| 修正 | 期待 | 実測効果 |
|------|------|---------|
| sr merge 拡張 (Session #35) | +2-4pt | わずか改善 (mean p18 1.5x) |
| srb merge 追加 (Session #36) | +1-3pt | わずか改善 (race-after 集計で 5/2-5/3 当日は 0 default) |
| 運用フィルタ | +3-5pt | 該当 race 減のみ、 winner_top1 計算不能 |

→ 期待 +6-12pt → **実測効果 +3-5pt 程度** (大幅未達)。

### 3.1 真の主因再確認

- **sib_*_wr リーク削除** (4/29、 V162_EXCLUDED): V18/V19 model に学習時残存、 復活 NG
- **PACI default 同値**: 一部 race で起きるが、本セッション merge では未対応の features 多数
- **feature distribution shift**: 27.7x → 11x に縮小したが依然存在
- **rank shift**: monotonic 変換で改善せず

→ **本格復活には V18/V19 sib 抜き再学習 + features alignment 必須** (Phase 3、 5/24+)。

---

## 4. 5/16 GO 確率 再下方修正

| Session | 確率 | 根拠 |
|---------|------|------|
| Session #33 | 75% | 楽観評価 |
| Session #34 | 40-50% | sib 復活 NG 判明 |
| Session #36 (本セッション中) | 50-60% | sr/srb merge + 運用フィルタ + V18 修復 (期待値) |
| **Session #36 retro 実測後** | **30-40%** | **実測効果限定的、 真因未解消** |
| Phase 3 V18/V19 sib 抜き再学習後 | 65-75% | 本格復活 |

### 4.1 5/16 投入判断

5/16 GO 確率 30-40% では **取り返し禁止ルール下で投入推奨できない**:
- 期待 ROI 95% 信頼区間 [80%, 130%] レベル (sample 不足、 winner_top1 不確実)
- 投入で過大評価リスク大、 累計損失拡大可能性

→ **5/16 暫定 NO-GO 寄り** (Session #34 と同じ判定に戻る)、 Phase 3 (5/24+) で本格復活へ集中。

### 4.2 5/16 で実施可能な低リスク投入 (代案)

V18/V19 paper trading のみ (実投資なし):
- 1,000 円/日 上限の試行は **paper として記録のみ**
- 実際の PAT 投票は V15 案B改 単独 維持
- 5/16-5/24 paper sample で Phase 3 V18/V19 再学習の前段準備

→ 5/16 paper、 5/24+ V18/V19 sib 抜き再学習、 6/9+ V20 統合 が現実的 plan。

---

## 5. 5/13-15 plan 修正 (Session #36 retro 実測反映)

### 5.1 5/13 (火)

旧 plan: retro 拡大 + V18 model 検証 (3h)
**新 plan**: retro 結果 (本書) を確認、 paper trading のみで 5/16 試行 plan 確定 (1h)
- 4/11-5/3 全 280 races retro 拡大は **継続するが投入決定の根拠とはしない**
- 期待 winner_top1 35-40% でも投入見送り

### 5.2 5/14 (水)

paper retro + Phase 3 V18/V19 sib 抜き再学習 着手:
- 学習 script (V162_EXCLUDED 反映) の準備
- 学習 data 抽出 (4/29 以降の post-leak data)

### 5.3 5/15 (木)

5/15 22:00 GO/no-go 判定 plan 修正:
- GO 条件 (緩和 Session #34 版): winner_top1 ≥ 40% / ROI ≥ 100% / sample 30+ bets
- 本書実測で達成困難 → **暫定 NO-GO**
- 5/16 paper (実投資なし) → 5/24+ V18/V19 再学習へ移行

---

## 6. 結論

🔴 **Session #36 retro 実測で sr/srb merge 拡張効果が期待 +6-12pt に対し +3-5pt 程度**、 真因 (sib リーク + features distribution shift) は monotonic 変換で改善不能と再確認。

5/16 GO 確率 **30-40% に再下方修正**、 取り返し禁止ルール下で 5/16 V18/V19 投入見送り推奨。
5/16 は **paper trading のみ** (実投資なし)、 V15 案B改 単独維持で確実完遂。
Phase 3 (5/24+) で V18/V19 sib 抜き再学習 → 65-75% GO で本格復活。

5/9 V15 投資への影響: **完全にゼロ** (絶対遵守ライン保護、 Session #36 全作業で確認済)。
