# Session #88 C: 累計差分 +¥1,310 root cause 確定

> 作成: 2026-05-09 (Session #88 dev/audit-backtest)
> 目的: Session #81 fork の +¥14,140 vs Claude 認識 +¥12,830 の差分 +¥1,310 を完全分解。

---

## 1. 差分 +¥1,310 の発生

| source | 累計 | 撤退余裕 |
|--------|------|---------|
| Session #81 fork 報告 | **+¥14,140** | +¥64,140 |
| Claude (Session #82 docs) | **+¥12,830** | +¥64,000 (∼) |
| **差分** | **+¥1,310** | (差分 ¥1,310) |

---

## 2. 差分 +¥1,310 の完全分解

### 2.1 +¥610 = baseline 認識違い

**Claude の baseline**: +¥13,530 (CLAUDE.md 旧 snapshot 由来)
**真の baseline (5/5 朝)**: +¥14,140 (user 報告、 memory `cumulative_pnl.md` 確定値)

差: **+¥610**

→ CLAUDE.md の "現行 累計収支 +13,530 円 / 撤退余裕 +63,530 円" は **古い snapshot** で、 5/5 朝の user 報告 +¥14,140 に更新されていなかった。

### 2.2 +¥700 = 5/9 vote MISS 誤認

**Claude の認識**: 5/9 新潟 12R MISS で -¥700
**実態**: 5/9 投票 0 R / 損益 ±¥0 (cumulative_monitor_5_9.md 確定)

差: **+¥700**

→ Claude は 「案B改 strict で 5/9 1 R 投票」 を assumption していたが、 user は実際には **5/9 0 R 投票** だった。

### 2.3 合計

| 要因 | 寄与 |
|------|------|
| baseline 認識違い (CLAUDE.md 古い snapshot) | +¥610 |
| 5/9 vote MISS 誤認 | +¥700 |
| **合計** | **+¥1,310** ✓ |

→ ★ root cause は ★case B + case C の MIXED★ ★
- **case A** (5/9 中 別 R で hit): NO (5/9 0 R 投票)
- **case B** (baseline 違い): YES (+¥610)
- **case C** (集計 logic 差 / 私の vote 誤認): YES (+¥700)

---

## 3. ★真の現在累計★ (Session #88 確定値)

cumulative_results.csv + 5/5 朝 baseline + 5/9 cumulative_monitor を統合:

| 項目 | 値 |
|------|----|
| **現在累計** | **+¥14,450** |
| **撤退余裕** | **+¥64,450** |
| 5/5 朝 baseline | +¥14,140 |
| 5/5 かしわ記念 NAR | +¥310 (settled) |
| 5/6-5/8 | ±¥0 (no JRA) |
| 5/9 投票 0 R | ±¥0 |
| 撤退ライン | -¥50,000 |

→ ★ Session #81 fork +¥14,140 vs Claude +¥12,830 の **どちらも 不正確** ★
→ ★ **真値は +¥14,450** (Session #88 が初めて確定) ★

---

## 4. 各 source の評価

| source | 値 | 正確性 | 根本原因 |
|--------|----|------|---------|
| memory `cumulative_pnl.md` | +¥14,140 (5/5 朝) | ★定義通り正確★ | 5/5 朝 snapshot |
| Session #81 docs | +¥14,140 (累計) | stale (-¥310) | かしわ +310 未反映 |
| CLAUDE.md (古い行) | +¥13,530 | stale (-¥920) | 5/5 朝 +14,140 へ未更新 |
| Session #82 docs | +¥12,830 | ★誤り★ (-¥1,620) | baseline -¥610 + 5/9 vote -¥700 誤認 |
| **Session #88 確定** | **+¥14,450** | **★現在の真値★** | 全 source 統合 |

---

## 5. 撤退余裕の真値

撤退ライン = -¥50,000

| 項目 | 値 |
|------|----|
| 真の累計 | +¥14,450 |
| 撤退ラインまでの距離 | +¥14,450 - (-¥50,000) = **+¥64,450** |

→ 撤退余裕 +¥64,450、 V15 投資保護 完全。

---

## 6. 改善 action

### 6.1 即時 (Session #88)
- ★ memory `cumulative_pnl.md` を **+¥14,450** に更新 (要 user 確認)
- ★ CLAUDE.md の +¥13,530 行を 次回 commit で +¥14,450 に置き換え (要 user 承認)

### 6.2 中期 (Sprint 5.5+)
- cumulative_results.csv から 累計を ★自動計算する script★ を実装 (memory `cumulative_pnl.md` で 2h 工数指摘済)
- 累計を毎晩 `cumulative_monitor_{date}.md` に **自動 snapshot**
- session 越し transfusion 防止 → 必ず CSV から 再計算

### 6.3 長期 (Sprint 6+)
- 累計監視を Discord Webhook で 自動通知 (毎晩 23:00、 翌日朝 8:00)

---

## 7. 結論

✅ **差分 +¥1,310 root cause 確定**:
- +¥610 = baseline 認識違い (CLAUDE.md 古い snapshot)
- +¥700 = 5/9 vote MISS 誤認 (実態は投票 0 R)

✅ **真の現在累計 = +¥14,450** (撤退余裕 +¥64,450)

✅ **V15 投資保護 完全** (撤退ライン -¥50,000 まで余裕 +¥64,450)

✅ **どの 既存 source も完全には正確でなかった**:
- Session #81: stale +¥310
- CLAUDE.md: stale +¥920
- Session #82 (Claude): 誤り +¥1,620
- Session #88 が **初めて 真値を確定**
