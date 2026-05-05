# v18/v19 統合プラン (Phase 2.5 / 2026-05-04 PM)

**前提**: race-level normalization (softmax T=1.0) で 5/2-5/3 retro の bet=0 問題は解決
**詳細**: `data/v18/race_normalize_5_4_result.md`

---

## 1. 投資 timeline

| 期間 | 主モデル | 補助 | 投資判断 |
|------|----------|------|----------|
| 〜 5/8 | V15 案B改 単独 | (なし) | 既存通り、Phase 2.5 諸スクリプト完成、観察のみ |
| **5/9 (Sat)** | **V15 案B改 単独** | (なし) | 案B改 維持 (累計 +14,140円、追加 risk なし) |
| **5/10 (Sun)** | **V15 案B改 単独** | (なし) | 同上 |
| 5/11-15 | V15 案B改 単独 | v18/v19 retro 蓄積 (paper) | normalize 効果を実 race で観察 |
| **5/16 (Sat)** | V15 案B改 (主) | **v18 単勝 paper / 部分実弾** | normalize T=1.0 retro が +ROI (>120%) なら 1日のみ部分実弾 |
| 5/17 | 同上 | v19 複勝 paper / 部分実弾 | tansho 結果次第で fukusho 追加 |
| 5/18-24 | A/B 比較期間 | full v18/v19 ramp-up 検討 | 累計 ROI と sample サイズ で判定 |

## 2. 5/9 (this Sat) 投資配分

**現状維持**: V15 案B改 単独 (700円 × 14条件)。v18/v19 一切投入なし。

理由:
- v18/v19 の 5/2-5/3 retro 結果は normalize 後でも sample 9-22 bets で広信頼区間
- normalize は monotonic、本質的 calibration 改善ではなく見せかけ scaling
- 5/9 で実弾投入する根拠は不十分
- 累計 +14,140円 を守るのが優先

## 3. v18/v19 部分実弾投入の前提条件 (5/16 〜)

以下 **すべて** 満たした場合のみ、5/16 から 1日 1〜2,000 円規模で開始:

| # | 条件 | 確認方法 | 状態 (5/4 時点) |
|---|------|----------|-----------------|
| 1 | race-level normalize を本番 pipeline に統合 | predict_core.py / race_auto_notify.py に softmax T=1.0 組み込み | ⏳ 未対応 |
| 2 | 5/2-5/15 paper retro で normalize 後 ROI >120% (tansho) | 各週末後に retro re-run | ⏳ 5/2-5/3 のみ |
| 3 | sample 30+ bets 累積 (paper) | 上記 retro の bet 累積 | ⏳ 5/2-5/3 で 9 bets |
| 4 | winner_top1 rate ≥ 40% (5/2-5/3 比 +5pt) | retro の rank 評価 | ⚠️ 現 34.5%、要改善 |
| 5 | feature distribution shift 調査済 | feature 値 BT vs production 比較 | ❌ 未着手 (別 session) |

**全条件揃わない場合**: 5/16 も V15 案B改 単独 維持。判断ミーティングを 5/15 夜。

## 4. 投資配分プラン (条件達成時)

### 4.1 5/16 (Sat) 試行

| 種別 | 上限 | 条件 | 想定 |
|------|------|------|------|
| V15 案B改 (主) | 既存通り (~9,800円/日) | 既存 14条件 | base ROI 維持 |
| **v18 単勝 (試行)** | **1,000 円/日** | normalize p≥0.5, ev≥1.2, **1 race 100円** | 10 races/日上限 |
| v19 複勝 (試行) | 0 (5/16 は tansho のみ) | - | - |

**最大 risk**: 1,000円/日 (v18 上限)。 V15 base に対して +10%。
累計許容: 5/16-5/24 期間 7日 × 1,000円 = 最大 7,000円 risk。

### 4.2 5/16 結果による分岐

| 5/16 v18 結果 | 5/17 アクション |
|---------------|-----------------|
| ROI ≥ 150% | v19 複勝 (1,000円/日) 追加投入 |
| 100% ≤ ROI < 150% | v18 維持、v19 さらに保留 |
| ROI < 100% | 即停止、原因分析、5/17 は V15 単独 |

## 5. normalize の本番 pipeline 統合 (上記 #1 の作業)

最小実装:

```python
# predict_core.py / race_auto_notify.py 内
from tools.race_normalize import normalize_per_race

# v18/v19 推論後
df['p_tansho_norm'] = normalize_per_race(df, 'p_tansho', 'race_id', method='softmax', T=1.0)
df['p_fukusho_norm'] = normalize_per_race(df, 'p_fukusho', 'race_id', method='softmax', T=1.0)
df['ev_tansho_norm'] = df['p_tansho_norm'] * df['odds']

# Phase 2 filter
m = (df['p_tansho_norm'] >= 0.5) & (df['ev_tansho_norm'] >= 1.2) & (df['odds'] > 0)
```

工数: 30 min (テスト含め). 5/15 までに完了が望ましい。

## 6. リスクと留意

### 6.1 取り返し禁止 (累計 +14,140円 を守る)

- v18/v19 試行で累計 -3,000円 以上の drawdown 発生 → 即停止、V15 単独復帰
- weekly cap: v18+v19 合計 -5,000円 で当該週中止

### 6.2 normalize の限界

- monotonic transform → 1着馬選定能力 (winner_top1 rate) は変わらない
- retro 34.5% (BT 47.8% から 13pt 劣化) の根本原因 (feature shift) は別問題
- normalize は filter 通過確保のみの「スケール調整」

### 6.3 sample サイズ

- 5/2-5/3: 67 races, 9-22 bets (filter 後)
- 5/16-5/24 (9 days × 36 races/day平均) ≈ 324 races 蓄積見込
- それでも統計的有意は 100+ bets 必要 → 6月以降の判定

## 7. 並行で必要な調査

別 session で以下を進めると本 plan 信頼性向上:

| 調査 | 工数 | 影響 |
|------|------|------|
| feature distribution shift 調査 (BT vs production) | 60min | winner_top1 13pt 劣化の根本原因 |
| 複勝 odds 実値で fukusho retro 再評価 | 30min | v19 ROI 推定の精度 |
| v18/v19 model 再学習 (2025年 Q4 含む) | 90min | scaling shift の根治 |

## 8. 結論

- **5/9: V15 案B改 単独 維持** (Phase 2.5 観察フェーズ継続)
- **5/16: 条件達成時のみ v18 単勝 1,000円/日 試行**、達成しなければ V15 単独維持
- **race-level normalize 統合は 5/15 までに完了** (本番 pipeline)
- **累計 +14,140円 死守**、損失拡大時は即停止

---

**関連 doc**:
- `data/v18/race_normalize_5_4_result.md` — 試作詳細
- `data/v18/distribution_shift_analysis.md` — shift 定量
- `data/v18/calibration_5_4_result.md` — Platt scaling 結果 (session#8)
- `data/v18/phase_2_5_progress_5_4.md` — Phase 2.5 全体進捗
