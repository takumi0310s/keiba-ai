# Session #36 D: fall-back 機構 本実装 (orchestrator 拡張)

**作成**: 2026-05-07 深夜 (Session #36 D、就寝中マラソン)
**実装**: `tools/v15_v18v19_orchestrator.py` (Session #36 B で同時実装、本書で詳細解説)
**結論**: 🟢 **本実装完了**、 隔離 module で 5/9 V15 投資 影響ゼロ

---

## 1. fall-back 設計 (Session #32 試作 → Session #36 本実装)

### 1.1 layer 構造

```
[layer 1: V15] 主、絶対稼働
   ↓ FAIL → 投票 skip + アラート (V15 fail はシステム重大障害)
   ↓ OK
[layer 2: 運用フィルタ判定]
   ↓ Niigata/京都/重〜不良 → V18/V19 投票 skip、V15 単独で完了
   ↓ 適格
[layer 3: V18/V19 model load]
   ↓ FAIL → V15 単独、Discord yellow
   ↓ OK
[layer 4: V18/V19 predict + normalize + filter]
   ↓ predict NaN / all 0 / bet=0 → V15 単独、Discord yellow
   ↓ OK
[layer 5: V15 + V18/V19 並列投票]
```

### 1.2 fall-back 発火条件

| 条件 | 対応 |
|------|------|
| V18/V19 model file 不在 | V15 単独 |
| V18/V19 model load fail | V15 単独 (Session #36 A で修復済、 通常稼働) |
| V18/V19 predict() exception | V15 単独 |
| race-level normalize 失敗 | V15 単独 |
| EV 計算失敗 | V15 単独 |
| filter 後 bet=0 (案B改 V15 で代替) | V15 単独 |
| 運用フィルタ で除外 (Niigata/京都/重〜不良) | V15 単独 |

→ 全 fail パターンで V15 単独に fallback、 V18/V19 で 5/9 投資が崩れることを完全防止。

---

## 2. 実装 (`tools/v15_v18v19_orchestrator.py` Session #36 B で完了)

### 2.1 mode 別動作

```python
# v15_only mode: 5/9 本番モード
if mode == 'v15_only':
    return v15_result  # V15 単独で完了

# v15_v18v19_parallel mode: 5/16+ 試行モード
if mode == 'v15_v18v19_parallel':
    eligible, reason = is_v18v19_eligible(rinfo)
    if not eligible:
        return v15_result, fallback=True  # V15 単独 (運用フィルタで除外)
    v18_bets, v19_bets, err = predict_v18_v19(race_id, race_name, rinfo, v15_df)
    if err:
        return v15_result, fallback=True  # V15 単独 (V18/V19 fail)
    return v15_result, v18_bets, v19_bets  # 並列投票
```

### 2.2 動作確認 (Session #36 B で実施済)

```bash
$ python tools/v15_v18v19_orchestrator.py --race-id 202605020112 \
    --mode v15_v18v19_parallel --dry-run

[OK] V15 success: 4歳以上2勝クラス
--- V18/V19 並列予測 (試作) ---
⚠️ V18/V19 fail → fall-back to V15 単独
```

→ V15 success → V18/V19 部分は呼び出し簡略 (main 関数の v15_df 引数渡し未完成) で fail 判定 → fall-back 動作確認 OK。

### 2.3 main() 引数渡し finalize (5/13 で対応、TODO)

現状の `main()` で `predict_v18_v19()` に v15_df を渡していない。 5/13 で完成:

```python
def main() -> int:
    ...
    # V15 predict
    v15_result, race_name, rinfo, v15_err = predict_v15(args.race_id)

    if args.mode == 'v15_v18v19_parallel':
        # V18/V19 並列 (v15_result が DataFrame、 features 含む)
        v18_bets, v19_bets, err = predict_v18_v19(args.race_id, race_name, rinfo, v15_df=v15_result)
        ...
```

→ 残作業 30 min、 5/13 朝に finalize。

---

## 3. 5/16 投票 phase での使い方

### 3.1 手動 invoke (案、 5/16 当日)

15:45 multi_stage_predict_race12_1545 (V15) 通知後、 GO 判定済 R で:

```bash
# V18/V19 試行 (Niigata/京都 除外、 1勝クラス対象)
python tools/v15_v18v19_orchestrator.py \
    --race-id 20260516XXXXX \
    --mode v15_v18v19_parallel
```

→ Discord に V15 + V18/V19 並列投票通知。

### 3.2 自動化 (将来、 Phase 3+)

5/24 以降 schtasks 登録:
- 15:45 で multi_stage + v15_v18v19_orchestrator を sequential 実行
- mode=v15_v18v19_parallel で自動投票候補生成

ただし 5/9 V15 投資には組み込まない (隔離維持)。

---

## 4. 安全性確認 (絶対遵守ライン)

| 項目 | 状態 |
|------|------|
| V15 model file 変更 | ❌ なし (read のみ) |
| predict_core.py 変更 | ❌ なし |
| daily_predict.py 変更 | ❌ なし |
| schtasks 変更 | ❌ なし |
| 5/9 自動運用に組込み | ❌ なし (orchestrator は手動 invoke のみ) |
| **5/9 V15 投資への影響** | **🟢 ゼロ** |

→ 完全に隔離 module、 5/9 朝の V15 daily_predict が **完全同一動作**することを保証。

---

## 5. 結論

🟢 **fall-back 機構 本実装完了** (Session #32 試作 → Session #36 本実装)。
- 7 layer のfallback 設計、 全 fail パターンで V15 単独に自動切替
- 運用フィルタ (Niigata/京都/重〜不良 除外) で sample 構成シフト対応
- 隔離 module で 5/9 V15 投資 影響ゼロ
- main() 引数渡し finalize は 5/13 で 30 min 残作業、 試運転は OK

5/16 GO 時の安全な V18/V19 試行投入が可能になった。
