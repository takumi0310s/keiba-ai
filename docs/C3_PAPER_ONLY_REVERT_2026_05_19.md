# C3 paper-only revert (2026-05-19)

## revert commit

`e1b16684` — [C3 paper-only revert] OOS +1.4pt 弱 signal、 5/24+ paper eval で見極め

- 対象: commit `5c8e9fd2` ([A-3] [C3] pos2 (T1-T2-T4) 除外 trio 7→6点 production)
- 変更: `tools/race_auto_notify.py` の C3 production ブロック削除 (12 行 --)

---

## revert 根拠 (Sub-task 2 hold-out validation 2026-05-19)

### 全期間 delta (n=515 strat_c)

| strategy | ROI | delta vs baseline |
|----------|-----|-------------------|
| baseline | 101.81% | — |
| C4 | 109.74% | +7.93pt |
| C3 | 114.76% | +12.95pt |
| C3+C4 | 123.31% | +21.50pt |

### OOS hold-out 平均 delta (3 split)

| strategy | OOS 平均 delta | 判定 |
|----------|--------------|------|
| **C4** | **+9.8pt** | 全 split 正 → **過学習なし / production 維持** |
| **C3 単体** | **+1.4pt** | Split3: -6.5pt (N=44 高配当ノイズ) → **弱 signal / paper-only に戻す** |
| C3+C4 | +9.5pt | in-sample 21.5pt → OOS 約半減 (軽度過大評価) |

### 統計的有意性

- bootstrap 95% CI は全 strategy で 100% 含む (N.S.)
- power analysis: +5pt delta 検出に N≈49,347 必要
- paper eval 4 週 (120-150R) では統計検定不能
- **6/17 判定の実質目的 = 「大崩れの早期検知」**

---

## 5/24 投票の真の formation (production)

| 要素 | 状態 |
|------|------|
| V15 (LGB+XGB 2-model) | production ✅ |
| 戦略⑦案 C (B/E/X/京都/06_特別 除外) | production ✅ |
| C4 (Cond-A + 1600-1800m → skip) | production ✅ (commit c36614b1) |
| **C3 (pos2 = T1-T2-T4 bet 除外)** | **paper-only ❌ (本 revert で production 停止)** |

**trio 買い目構成 (5/24+)**:
```
formation: trio 7点 (C3 適用なし)
1列目: TOP1
2列目: TOP2, TOP3
3列目: TOP2, TOP3, TOP4, TOP5, TOP6
--- bet 内訳 ---
bet1: T1-T2-T3
bet2: T1-T2-T4  ← C3 production 時は除外していたが、 paper-only 後は再び含む
bet3: T1-T2-T5
bet4: T1-T2-T6
bet5: T1-T3-T4
bet6: T1-T3-T5
bet7: T1-T3-T6
```

C4 適用時: Cond-A + 1600-1800m のレースは skip (bet 0 点)

---

## 5/24-6/16 paper eval での C3 継続観察

`tools/race_notify_log_v2.py` の strategy 'c3' / 'c3c4' は引き続き paper 記録を継続。

記録先: `data/race_notify_log_v2_summary/*.json` の `strategy_stats['c3']` / `strategy_stats['c3c4']`

---

## 6/17 採用判定 path

```bash
python tools/c3c4_adoption_test_v2.py
```

参照: `docs/6_17_ADOPTION_DECISION_GUIDE_v2.md`

### C3 GO 基準

| 基準 | 閾値 | 優先度 |
|------|------|--------|
| N | ≥ 24 (4 週末) | 必須 |
| delta | ≥ +5pt vs baseline | 主要 |
| 大崩れなし | 連続 -10pt 週 ≤ 1 | 安全弁 |
| C3 単体 vs C3+C4 | C3 単体が C4 単体と同等以上 | 参考 |

### C3 NOT-GO 基準

- OOS delta < 0pt / 2 週連続
- 連続 -10pt 2 週以上
- N < 24 (データ不足)

---

## rollback 後の確認コマンド

```bash
# C3 production ブロックが存在しないこと確認
grep "STRATEGY_C3_ENABLED" tools/race_auto_notify.py
# → _c3_effective 代入行 (rollback hook) のみ残る、production ブロックなし

# C3 paper 記録が継続していること確認
grep -n "c3" tools/race_notify_log_v2.py | grep -i "strategy\|STRATEGY_KEYS"
```

---

## verdict

- **C3 production**: ❌ (OOS +1.4pt 弱 signal、 honest 判断で paper-only)
- **C4 production**: ✅ (OOS +9.8pt 全 split 正、 維持確定)
- **5/24 投票**: V15 + 戦略⑦案 C + C4 のみ、 trio 7 点フォーメーション
- **6/17 再評価**: paper eval 4 週 delta で final 判定
