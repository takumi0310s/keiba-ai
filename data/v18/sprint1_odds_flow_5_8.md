# Sprint 1 D: オッズ flow tracker (Session #45 D)

**作成**: 2026-05-08 (Session #45 D、 dev/sprint1)
**目的**: 直前 10 分の オッズ変動から 「隠れた人気変化」 検出
**ステータス**: ✅ 実装完了 + simulation 動作確認

---

## 1. 設計

### 1.1 データ取得 sequence

```
T-10min: 単勝オッズ snapshot 1
T-9min:  snapshot 2
...
T-1min:  snapshot 10
T-0min:  current snapshot
↓
features 計算:
- odds_change_5min_pct (5 分間 変動率)
- odds_change_10min_pct (10 分間 変動率)
- flow_score (-1.0 ~ +1.0)
```

### 1.2 flow_score 解釈

| change_5min_pct | 意味 | flow_score |
|----------------|------|-----------|
| ≤ -10% | オッズ急落 = 人気急上昇 | +1.0 |
| -10% ~ 0% | オッズ微減 | +0.0 ~ +1.0 (linear) |
| 0% | 変化なし | 0.0 |
| 0% ~ +10% | オッズ微増 | -0.0 ~ -1.0 (linear) |
| ≥ +10% | オッズ急騰 = 人気急落 | -1.0 |

→ flow_score 高 = 「直前で急に人気が出た馬」 = 投票検討。

### 1.3 永続化

```
data/odds_flow_snapshots/<race_id>.jsonl
  各行: {"timestamp": ISO8601, "odds": {"01": 4.5, "02": 8.0, ...}}
  → race ごと append、 1 分毎 snapshot 保存
```

---

## 2. 動作確認 (simulation)

```python
# T-10min: {"01": 5.0, "02": 8.0, "03": 12.0, "04": 20.0, "05": 50.0}
# T-5min:  {"01": 4.5, "02": 8.0, "03": 11.0, "04": 22.0, "05": 50.0}
# T-0min:  {"01": 4.0, "02": 9.0, "03": 10.0, "04": 22.0, "05": 50.0}

→ flow features:
  馬01: 5.0 → 4.0 (-20%) → flow_score = 1.0 (人気急上昇)
  馬02: 8.0 → 9.0 (+12.5%) → flow_score = -1.0 (急落)
  馬03: 12.0 → 10.0 (-17%) → flow_score = 0.91
  馬04: 20.0 → 22.0 (+10%) → flow_score = -0.0 (10% threshold で打止)
  馬05: 50.0 → 50.0 (0%) → flow_score = 0
```

→ 動作確認 OK

---

## 3. production 統合 plan (5/15+)

### 3.1 schtasks 追加候補

```cmd
# 5/9 投資 当日に schtasks 追加 NG → 5/15 merge 後に admin で追加
schtasks /Create /TN "Keiba-OddsFlowTracker" ^
    /TR "powershell -File C:\Users\takum\keiba-ai\odds_flow_minutely.ps1" ^
    /SC MINUTE /MO 1 /F
```

`odds_flow_minutely.ps1`:
```powershell
# レース時間帯 (10:00-16:30) のみ実行
$h = (Get-Date).Hour
if ($h -ge 10 -and $h -le 16) {
    cd C:\Users\takum\keiba-ai
    python tools\odds_flow_tracker_minute.py  # 別 wrapper、 netkeiba 直前オッズ取得
}
```

### 3.2 features 統合

V15 production には flow_score / odds_change_5min を追加 features として:
- 学習 data: 過去 retro で flow snapshot 不在 → V20 構築時に追加
- 予測時: race_auto_notify で flow_score を 補正係数として使用

→ 5/16+ V18 trial では試行的に flow_score を後段補正 (top1_p 高 + flow_score > 0.5 → 高自信時の boost)

---

## 4. caveat + 制限

- netkeiba 直前オッズ scraping は IP/cookie 制約 (大量取得で BAN リスク)
- 1 分毎 polling は schtasks で 1 day 約 600 calls (10:00-16:30、 6.5h) → 適度な負荷
- 過去 retro data に flow snapshot **不在**、 backtest は 5/15+ 蓄積後
- production では Cookie refresh 必要 (Session #38 既存 mechanism 流用)

---

## 5. 5/9 V15 投資保護

✅ 5/9 朝 odds_flow_tracker は **無効** (snapshot 不在、 5/9 当日 schtasks 追加なし)
✅ V15 production 完全独立、 main 不変
✅ V15 model md5: 842b9a5f... 不変

→ **5/9 朝 V15 完全保証**

---

## 6. 結論

✅ D1: tools/odds_flow_tracker.py (170 行)
✅ D2: 1 分 polling design + flow_score 計算
✅ D3: simulation で 動作確認 OK
✅ D4: 永続化 (jsonl 形式)
✅ D5: production 統合 plan (5/15+ schtasks 追加)
✅ V15 投資保護

→ **Sprint 1 D 完了、 5/15 merge 後 production 蓄積開始候補**

---

**Session #45 D 完了 (dev/sprint1)**
