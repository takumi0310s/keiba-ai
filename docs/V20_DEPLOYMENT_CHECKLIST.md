# V20 投入 checklist

**作成**: Session #79
**判定日**: 2026-06-30 (paper 30 日 完了時点)
**投入候補日**: 2026-07-01

---

## 1. 6/30 paper 評価

### 1-1. 必須 metrics (全 5 項目 確認)

| # | 確認項目 | 閾値 | 結果欄 |
|---|---------|------|--------|
| 1 | WF AUC | ≥ 0.895 | [   ] |
| 2 | LIVE retro winner_top1 | ≥ 33% | [   ] |
| 3 | shift (BT → LIVE) | ≤ 12x | [   ] |
| 4 | paper ROI 30 日 | ≥ V15 + 5pt | [   ] |
| 5 | LEAK 監査 PASS | Session #51 + KKA | [   ] |

★ 5/5 PASS → GO ★
★ 4/5 → 再評価 (7/15) ★
★ 3/5 以下 → NO-GO 確定 (7/15+ 再構築) ★

### 1-2. 補助 metrics (参考)

- top3 inclusion rate
- class 別 AUC
- 当日体重 reaction (Session #48 B 効果)
- KKA features 寄与 (Session #53 修復効果)

---

## 2. 7/1 投入候補 判定 flow

```
6/30 22:00: paper 評価 完了
6/30 23:00: GO/NO-GO 判定 (上記 5 項目)
  ├─ GO: 7/1 朝 投入 logic 切替
  ├─ 再評価: paper +14 日 → 7/15 再判定
  └─ NO-GO: V15 継続、 V20 改修 sprint
```

---

## 3. 投票 strategy 設定 (V20 投入直後)

### 3-1. 段階的投入 (慎重維持)

| 期間 | 投資 / R | max / 日 | 備考 |
|------|---------|----------|------|
| 7/1-7/14 (week 1-2) | ¥700 | ¥2,100 | V15 と同額、 慎重維持 |
| 7/15-7/28 (week 3-4) | ¥700-1,000 | ¥2,100-3,000 | hit rate 安定確認後 |
| 8/1+ | ¥2,000-3,000 | ¥6,000-9,000 | Eighth Kelly 適用判定 |

### 3-2. 戦略⑦ 維持

- 06_特別 / 京都 / 条件E / 条件B 除外 継続
- KKA features で 京都 ROI 改善見込みあれば 8/1+ 解除検討

---

## 4. V15 → V20 切替 logic

### 4-1. ファイル切替

| 項目 | 旧 (V15) | 新 (V20) |
|------|---------|---------|
| model file | `keiba_model_v15_central_live.pkl.gz` | `keiba_model_v20_central_live.pkl.gz` |
| Pattern A | `keiba_model_v15_central.pkl.gz` | `keiba_model_v20_central.pkl.gz` |
| predict_core version check | `'v15'` | `'v20'` |
| feature lookups | (継続) | (V20 拡張版) |

### 4-2. 切替手順 (7/1 朝 06:00 想定)

1. **23:00 前夜**: V20 model file 配置確認、 predict_core.py V20 対応 commit (paper 中に既に commit 済の想定)
2. **06:00 朝**: production 切替 flag を `MODEL_VERSION=v20` に変更
3. **06:30**: daily_predict.py 実行 → V20 で予測生成
4. **06:35**: Discord 通知に "V20 投入初日" tag
5. **22:00**: 初日結果照合、 V20 hit rate 確認

### 4-3. flag 切替方法

```python
# tools/predict_core.py または config
MODEL_VERSION = os.getenv('KEIBA_MODEL_VERSION', 'v20')  # default v20 (7/1 以降)
```

`.env` に `KEIBA_MODEL_VERSION=v20` 追加で切替。

---

## 5. rollback 手順 (V20 投入後 異常時)

### 5-1. rollback trigger

| 条件 | 対応 |
|------|------|
| 7/1-7/7 で hit rate ≤ 25% (3 日連続) | rollback 検討 |
| 7/1-7/14 で ROI ≤ 50% | rollback 確定 |
| LEAK 検出 | 即時 rollback |
| Discord 通知 失敗連鎖 | 即時 rollback |

### 5-2. rollback 手順 (即時、 5 分以内)

```bash
# 1. .env 編集
echo "KEIBA_MODEL_VERSION=v15" >> .env

# 2. daily_predict.py 即時再実行 (cache clear)
python tools/daily_predict.py --force

# 3. Discord 通知 "V15 rollback"
python tools/notify_done.py "V20 rollback" "V15 復帰、 詳細調査開始"
```

### 5-3. rollback 後の対応

- V15 production 再開 (損失最小化)
- V20 改修 sprint (1-2 週間)
- 再判定 (7/15 + 2 週間 = 7/29 想定)

---

## 6. 7/1 投入時 communication

### 6-1. Discord 通知 (7/1 朝 06:30)

```
## V20 投入初日 (7/1)

予測 model: V20 4-model ensemble (期待 AUC 0.90025)
投票 strategy: 案B改 7 点 ¥700 / R (V15 と同額、 慎重維持)
比較対象: V15 paper 並行 (rollback 用)
監視: 22:00 daily report
```

### 6-2. user 確認事項 (れんはす)

- [ ] paper 30 日結果 報告 (6/30)
- [ ] GO/NO-GO 判定 確認 (6/30)
- [ ] 7/1 朝 投入 final 承認

---

## 7. 投資保護 (V20 投入後も遵守)

- 撤退ライン: 累計 -¥50,000 (現状 +¥12,830、 余裕 +¥62,830)
- 取り返し禁止 (損切り後 翌日へ持ち越さない)
- max 投資 / 日 は 7/1-7/14 は ¥2,100 厳守
- V20 paper 並行運用 で異常検知 体制維持 (1 ヶ月)

---

## 関連

- [V20_BUILD_DETAILED_PLAN.md](V20_BUILD_DETAILED_PLAN.md)
- [SPRINT_6_PLAN.md](SPRINT_6_PLAN.md)
- [V20_VS_V15_COMPARISON.md](V20_VS_V15_COMPARISON.md)
