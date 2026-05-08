# Sprint 2 統合結果 (Session #47)

**実施**: 2026-05-08 (Session #47 Sprint 2、 dev/sprint2 branch)
**期間**: 約 4-5 h (元想定 30-40h、 大幅短縮)
**完了**: 8 idea 全完了、 8 commits
**merge 予定**: 5/15 22:00 (Sprint 1 と同時、 5/16 V18 trial 直前)

---

## 1. ★ 戦略 ★

main 不変 + dev/sprint2 で 8 idea 全実装 + 5/15 一括 merge (Sprint 1 と同時)。

---

## 2. 8 idea 完了 deliverable

| # | idea | 結果 | signal 強度 |
|---|------|------|----------|
| **A** | horse_weight_features | 4 features、 weight_std corr +0.0493 | weak |
| **B** | race_interval_features | 中2-4週 24.71% vs 連闘 17.65% (+7pt) | strong |
| **C** | running_style_change | 逃げ 49.42% vs 追込 6.57% (+42.85pt) | **超強** |
| **D** | maiden_race_model | 新馬戦専用 LGB AUC 0.8092 | model |
| **E** | jockey_network | 騎手 graph 248 nodes/30K edges | weak (要拡張) |
| **F** | jump_race_model_v2 | popularity 除外 AUC 0.6778 (vs 旧 0.7536) | leak 解消 |
| **G** | paddock_image_analyzer | YOLOv8 馬体検出 PoC 動作 | PoC |
| **H** | post_race_features_update | 5/9 後 features 即時更新 + 5/10 verify integrate | infra |

---

## 3. backtest 主要発見

### 3.1 ★ 超強 signal: 脚色 ★

```
逃げ:  49.42% (top3 率)
先行:  36.21%
差し:  19.26%
追込:   6.57%
→ 逃げ vs 追込 +42.85pt 差
```

### 3.2 strong signal: レース間隔

```
連闘 (1-7d):       17.65%
中1週:             20.29%
中2-4週 ★:         24.71%  (最高)
中5-8週:           23.03%
休み明け (57+d):   20.48%
→ 中2-4週 vs 連闘 +7.06pt
```

### 3.3 leak 解消: jump model

```
Sprint 1 E (popularity 含): AUC 0.7536 (リーク類似)
Sprint 2 F (popularity 除外): AUC 0.6778 ★ production candidate
```

---

## 4. V15 production 完全保護

```
main HEAD: 30b6c1bb (Session #46) 不変
V15 model md5: 842b9a5f305c793ed8fa54a74e06b836  (Session #38-47 全期間 不変)
predict_core / daily_predict / app.py / schtasks: 完全不変
```

→ ✅ **5/9 朝 V15 案B改 完全保証**

---

## 5. dev/sprint2 commits (8 件)

```
69412a66 Session #47 H: post_race_features_update
937e654f Session #47 G: paddock_image_analyzer PoC
3f423e57 Session #47 F: jump_race_model v2
245baba6 Session #47 E: jockey_network
7398c2d8 Session #47 D: maiden_race_model PoC
664b41d7 Session #47 C: running_style_change
52393272 Session #47 B: race_interval_features
fd602e7f Session #47 A: horse_weight_features
[次 commit] Session #47 I: SPRINT_2_RESULTS.md
```

---

## 6. 5/15 merge plan (Sprint 1 + 2 同時)

```bash
# 5/15 22:00 (5/16 V18 trial 投入直前、 ユーザー review 後)

# Sprint 1 merge
git checkout main
git pull origin main
git merge --no-ff dev/sprint1 -m "Merge Sprint 1: 5 軽量改善 (Session #45)"

# Sprint 2 merge
git merge --no-ff dev/sprint2 -m "Merge Sprint 2: 8 中規模改善 (Session #47)"

# push
git push origin main
```

→ 1 night で 13 idea (Sprint 1: 5 + Sprint 2: 8) を main に投入

---

## 7. merge 後の Phase 3 統合 (5/16+)

V20 構築 (Phase 3 後半 5/22-6/8):
- V15 base (150) + sib_w5 + Sprint 1 機能 + Sprint 2 features (~10-15)
- 4-model ensemble (LGB+XGB+FT+IR)
- 期待 AUC 0.890-0.895
- 6/8 V20 投入候補

---

## 8. 結論

✅ Sprint 2 8 idea 全完了 (4-5h で完走、 元想定 30-40h)
✅ 超強 signal: 脚色 +42.85pt (BT、 V15 と一部重複)
✅ strong signal: レース間隔 +7.06pt
✅ leak 解消: jump model v2 (popularity 除外)
✅ V15 production 完全不変、 main 不変
✅ 5/15 22:00 merge plan (Sprint 1 + 2 同時)

→ **Sprint 2 完了、 5/15 merge 待ち、 5/16 V18 trial + V20 構築素材 完備**

---

**Session #47 Sprint 2 完了 — 2026-05-08**
