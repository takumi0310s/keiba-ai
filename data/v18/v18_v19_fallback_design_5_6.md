# V15 + V18/V19 fall-back 機構 設計 (GO 条件 #5)

**作成**: 2026-05-06 PM (Session #32 D)
**実装**: `tools/v15_v18v19_orchestrator.py` (試作、deploy しない)
**判定対象**: GO 条件 #5 「fall-back 機構 (V18/V19 fail → V15 自動切替)」
**結論**: **🟡 試作完了、本格 deploy は 5/16+**

---

## 1. 設計思想

5/9 当日 V15 単独投資の動作を **絶対保護**。 V18/V19 投入は ★追加★ の位置づけで、fail 時は無音で V15 単独に fall-back。

**絶対遵守ライン**:
- V15 model file 触らない (read のみ)
- predict_core.py に変更を加えない (新規 module で隔離)
- schtasks に登録しない (手動実行のみ)
- 5/9 V15 単独投資の動作には **完全無影響**

---

## 2. アーキテクチャ

```
┌──────────────────────────────────────────────────────┐
│ v15_v18v19_orchestrator.py (新規、隔離)              │
│                                                       │
│ 1. V15 predict (主)                                   │
│    └─ FAIL → 終了 + Discord red                       │
│    └─ OK → 続行                                       │
│                                                       │
│ 2. mode='v15_only': V15 単独で完了 (5/9 本番モード)   │
│                                                       │
│ 3. mode='v15_v18v19_parallel' (5/16+ 試行モード):    │
│    V18/V19 predict (副)                              │
│    └─ FAIL (model load / predict / normalize 等):    │
│        ⚠️ fall-back triggered                        │
│        Discord yellow 通知「V18/V19 fall-back to V15」│
│        V15 単独で完了                                  │
│    └─ OK:                                             │
│        並列通知 (主: V15、副: V18/V19)                │
└──────────────────────────────────────────────────────┘
```

---

## 3. fall-back 発火条件

| 条件 | 対応 |
|------|------|
| V18 model load 失敗 (file 不在 / corruption) | V15 単独 |
| V19 model load 失敗 | V15 単独 |
| V18 predict() exception | V15 単独 |
| V19 predict() exception | V15 単独 |
| race-level normalize 失敗 | V15 単独 |
| EV 計算 失敗 | V15 単独 |
| filter 後 bet=0 (案B改 V15 で代替) | V15 単独 (案B改 復活) |

**全パターンで V15 単独に fall-back**、 5/9 投資保護。

---

## 4. 動作確認 (試作)

```bash
$ python tools/v15_v18v19_orchestrator.py --race-id 202604010312 --mode v15_v18v19_parallel --dry-run

=== orchestrator mode=v15_v18v19_parallel race_id=202604010312 ===
[OK] V15 success: 4歳以上1勝クラス

--- V18/V19 並列予測 (試作) ---
⚠️ V18/V19 fail → fall-back to V15 単独
```

→ 試作 skeleton が想定通り動作 (V18/V19 部分は NotImplementedError 相当で fall-back 検証 OK)。
V15 単独 mode (`--mode v15_only`) も動作:
```bash
$ python tools/v15_v18v19_orchestrator.py --race-id 202604010312 --mode v15_only
[OK] V15 success: 4歳以上1勝クラス
mode=v15_only → V15 単独で完了
```

---

## 5. GO 条件 #5 判定

> #5: fall-back 機構 (V18/V19 fail → V15 自動切替)

| 観点 | 判定 |
|------|------|
| 設計 完了 | ✅ |
| 試作 動作確認 | ✅ (fall-back 発火 OK) |
| **V18/V19 本実装** | ❌ (deploy しない、5/16+ で実装) |
| 本格 production 統合 | ❌ (predict_core.py 統合 5/8 以降) |

**判定**: 🟡 **部分達成** (設計+試作 OK だが、V18/V19 本実装は未着手)

→ 5/9 投入 GO 条件としては **NO** (本実装が前提のため)。

---

## 6. 5/16+ 本格 deploy plan

### Step 1: V18/V19 本実装 (5/13-5/14、4h)

`predict_v18_v19()` 関数の実装:
1. V18 lgb + xgb load (data/v18/models/)
2. V19 lgb + xgb load
3. predict_core で features 構築済 df を引数に取る
4. v18 ensemble 予測 → P(1着)
5. v19 ensemble 予測 → P(top3)
6. race-level normalize (softmax T=1.0)
7. EV 計算 (P × オッズ)
8. filter (単勝 p_norm>=0.5 ev>=1.2、複勝 p_norm>=0.7 ev>=1.1)

### Step 2: race_auto_notify との連携 (5/15、2h)

5/16 (土) 本番で multi_stage_predict_race12_1545 と並列実行:
- V15 案B改 (主、12R 1勝)
- V18/V19 試行 (副、別 R で 500-1,000 円)

### Step 3: 5/16 paper trading (試行)

- 1,000 円/日 上限
- fall-back 動作確認、failure log 蓄積

---

## 7. 5/9 deploy しない理由

絶対遵守ライン:
- 本セッションは V15 単独投資保護
- predict_core.py 変更なし
- schtasks 変更なし
- → orchestrator は **隔離 module**、5/9 自動運用に組込まれない

→ 5/9 (土) は V15 単独 案B改 投資のみ、orchestrator は **手動実行のみ可能** (確認用)。

---

## 8. 結論

GO 条件 #5: 🟡 **部分達成** (設計+試作 OK、本実装は 5/16+)

5/9 投入 GO 条件としては **NO** (本実装未着手)。
ただし設計 doc + 試作 skeleton で 5/16+ 本格 deploy の準備完了。

**4 条件 NO 確定** (#1, #2, #3, #5) → V18/V19 5/9 投入 **NO-GO 確定**。
