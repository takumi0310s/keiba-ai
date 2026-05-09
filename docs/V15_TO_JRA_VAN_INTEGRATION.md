# V15 score → JRA-VAN NEXT 連携設計

Session #81 (2026-05-09 夜)。

## 課題

V15 (本番) / V20 (6/8 投入予定) の model score を
JRA-VAN NEXT 自動分配 + 1 click PAT 送信に 流したい。

## 現状の制約

- JRA-VAN NEXT は **JRA-VAN 独自予想** と連携前提
- V15 / V20 score を 直接 受け付ける API なし
- TARGET frontier JV 経由で features 投入できる可能性は あり (要検証)

## 解決策 (5/15-5/22 trial で 試用)

### Phase 1: 手動入力 + 自動分配 (5/16-)

```
V15 daily_predict (08:00)
    ↓
案B改 strict 候補 確定 (14:00)
    ↓
ユーザーが NEXT に 馬番 リスト 手動入力
    ↓
NEXT 自動分配 button push
    ↓
1 click PAT 送信
```

工数:
- 現状 (PAT 手動入力): 各 R 数分 (5-15 分 / 1 日 max 3 R)
- Phase 1 連携後: 各 R 数秒 (30 秒 / 1 日 max 3 R)
- **大幅時短 ★**

### Phase 2: TARGET frontier 経由 (要検証、 5/23+)

TARGET frontier JV に カスタム指数 を 取り込めるなら、
V15 / V20 score → カスタム指数 として 投入可能性。

- 5/23+: TARGET frontier カスタム指数 機能 調査
- GO なら NEXT 連携 完全自動化 へ
- NO-GO なら Phase 1 維持

### Phase 3: V20 production 連携 (7/1+)

V20 投入後:
- V20 case 1+4 候補 → NEXT 投入
- V20 winner_top1 ≥ 30% ならば 自動分配 EV モード で 1 click

### Phase 4: 完全自動化 (12 月、 Phase 5)

- RL 投票最適化
- Selenium / Playwright で NEXT 操作自動化
- ★destructive op 厳禁、 V15 投資保護 完全継続★

## V15 案B改 strict 入力 sample

```
レース: 2026/05/16 東京 11R
予算: 2,100 円
買い目: 三連複 7 点
  軸: 5
  相手 (2列目): 3, 8
  相手 (3列目): 3, 8, 11, 14, 2

NEXT 入力:
  馬券種: 三連複
  軸馬: 5
  相手1: 3, 8
  相手2: 3, 8, 11, 14, 2
  予算: 2,100 円
  → 自動分配 → 1 click PAT
```

## 安全性

- V15 model 変更 **なし**
- predict_core / daily_predict / app.py 変更 **なし**
- schtasks 変更 **なし**
- ユーザー手動操作 介在 (誤投票 防止)
- 累計 +14,140 円 / 撤退余裕 +64,140 円 死守

## 次 step

[JRA_VAN_NEXT_TRIAL_5_15.md](JRA_VAN_NEXT_TRIAL_5_15.md) で 5/15 trial 詳細。
