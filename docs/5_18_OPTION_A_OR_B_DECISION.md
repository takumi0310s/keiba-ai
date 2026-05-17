# 5/18 朝 admin 登録時 option A or B 判断

> 作成: 2026-05-17 (夜-4-B task)
> 対象: 5/18 (SUN) 朝 admin 登録時に 「初回 fire を当日にするか」 の user 判断
> 関連: `docs/5_18_ADMIN_TASKS.md` Section 0、 `docs/6_17_ADOPTION_DECISION_CHECKLIST.md`

---

## 1. 質問

5/18 (SUN) 朝 admin 登録時、 当日 fire するか?

| option | 初回 fire | 5/18 paper eval | sample N (5/18-6/16) |
|--------|----------|-----------------|---------------------|
| **A** (★ 推奨 ★) | **5/18 (SUN) 08:30 即 fire** | 5/18 当日から開始 | **~54 R** |
| B | 5/23 (SAT) 08:30 初回 fire | 5/18 skip | ~48 R |

---

## 2. 推奨理由 (option A)

- **sample N 増**: 54 R vs 48 R (+6 R、 +12.5%)
- **統計検定力 改善** (assumption、 sample 想定 power 計算):
  - +5pt detect: power ~0.5 (A) vs ~0.47 (B)
  - +10pt detect: power ~0.75 (A) vs ~0.72 (B)
- **採用判定 6/17 まで余裕**: 5/18-6/16 で 4 週末分 蓄積完了
- 5/17-5/23 は mock=True 強制 (夜-4-A 整合)、 5/18 fire でも実 fetch 0 = 影響最小
- 5/24 から実 paper eval 段階移行、 5/18 fire 込みでも safety 確保

---

## 3. option B 採用 path (★ 5/18 skip 希望時 ★)

### Step 1: 5/18 admin 登録 (default 通り)
`docs/5_18_ADMIN_TASKS.md` Section 3 の bat 4 件 実行。

### Step 2: 登録直後に LiveOrchestrator のみ一時停止
```powershell
schtasks /Change /TN "Keiba-LiveOrchestrator-15min" /Disable
```

### Step 3: 5/23 (SAT) 朝 再開
```powershell
schtasks /Change /TN "Keiba-LiveOrchestrator-15min" /Enable
```

★ 他 7 task (FeaturesIntegrity / AnomalyCheck×5 / CumulativeAudit) は 5/18 から動作継続。
★ Disable / Enable は admin 権限不要、 user 権限で実行可能。

---

## 4. 判断 step (★ 5/18 朝の動作 ★)

1. 5/18 朝 06:30-07:00 admin schtask 登録 (`docs/5_18_ADMIN_TASKS.md` Section 3)
2. ★ default = option A ★ (5/18 当日 08:30 fire)
3. option B 採用なら Section 3 の Disable コマンドを admin 登録直後に実行
4. 5/23 SAT 朝に Enable で 再開 (option B のみ)

---

## 5. デフォルト動作

admin schtask 登録 → ★ 5/18 当日 08:30 fire (option A) ★

★ option B を選ぶ理由がなければ option A で進行 ★

---

## 6. 整合性 (★ 夜-4-A live_orchestrator.bat mock 解除 plan ★)

- 5/17-5/23: mock=True 強制 (live_orchestrator.bat 内)
- 5/24 以降: user 手動で mock 解除 (実 fetch 開始)

option A (5/18 fire) の場合:
- 5/18 fire = mock=True で実 fetch 0 (paper test only)
- 5/24 から実 paper eval (実 fetch 開始)

★ 整合 ★: option A でも 5/18-5/23 は paper test 段階、 5/24 から実 paper eval。

---

## 7. fabrication 防止 (★ honest ★)

- sample N 計算 (54 R / 48 R) は実 schedule (5/18-6/16 weekend) × 6 R/day (戦略⑦案 C 適用後想定) ベース
- 検定力数値 (power ~0.5 / ~0.75) は assumption (sample 想定 power 計算)、 実 ROI 分散次第で変動
- user 判断 sentence (本 doc Section 1-4) で option A vs B 明示
- ★ 6 R/day は戦略⑦案 C 適用後の想定値、 実 race 数次第で 上下する ★

---

★ honest 厳守、 V15 完全不変、 admin 操作は docs/5_18_ADMIN_TASKS.md の手順のみ ★
