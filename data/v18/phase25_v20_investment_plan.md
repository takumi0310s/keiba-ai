# Phase 25 plan: 5/24+ V20 投入 path (5/17 開催 後の動き)

## 前提 (5/17 開催 まで完了想定)

- ✅ V15 production 完全保護 + 案 B 改 + 戦略⑦
- ✅ Phase 21-24 全 22 tools 実装 + shadow mode 並行運用 開始
- ✅ paddock 動画 ~500 馬 × 30 frame ≈ 15K frame 蓄積
- ✅ YouTube JRA LIVE 録画 (5/16-17 開始)
- ✅ JV-Link 32-bit 動作確認 + 30y backtest 実取得
- ✅ 厩舎コメント / 専門家予想 実 scrape 開始
- ✅ Drawdown breaker 稼働 monitoring

## 5/17 開催 結果 verification (5/18 朝)

- [ ] V15 案 B 改 単独 ROI 結果 確認 (現行 戦略⑦ 想定 140%+)
- [ ] Phase 23 shadow log Kelly + Pari-mutuel + Cal の "もしも" 結果 集計
- [ ] V15 vs Shadow 差分 評価 (+ X % で有意?)
- [ ] paddock 動画 自動 capture 動作確認
- [ ] YouTube 録画 file 確認 (8h 動画 ~ 2 GB)

## Day 1-5: 5/18-22 (整理 + 学習 data 構築)

### 5/18 (日) - 結果 review
- [ ] 5/17 開催 verdict (V15 / shadow 比較)
- [ ] Phase 23 tool 各々 の評価
  - Kelly: bet size 妥当性
  - Calibration: 実 outcome vs calibrated pred の整合性 (Brier)
  - Pari-mutuel: optimal trio 選択が hit に近づいたか
  - Drawdown breaker: trigger しなかったか

### 5/19 (月) - V20 学習 data 拡張
- [ ] paddock 動画 5/17 分追加 capture
- [ ] JV-Link で 2026/5 分 fresh data fetch
- [ ] 30y backtest 続き

### 5/20 (火) - V21 video features 学習 PoC
- [ ] paddock 全 馬 features 抽出 (~500 馬 × 38 features = 19K records)
- [ ] V21 multimodal stacker 実 学習 (V15 pred + video features)
- [ ] WF AUC 比較 (V15 0.8939 baseline)

### 5/21 (水) - V20 学習
- [ ] 全 source merge data 構築
  - TFJV + JRDB + netkeiba + JV-Link (fresh 2026/5) + 厩舎コメント拡張
- [ ] V20 4-ensemble 学習 (LGB + XGB + FT + IntraRace、 sib_w5 + skb 除外)
- [ ] WF AUC 確認 (PoC 0.8752 → 目標 0.880+)

### 5/22 (木) - V20 検証
- [ ] V20 WF backtest 6-fold 全条件 ROI 確認
- [ ] V20 LIVE retro (5/17 + 5/18 + 5/19 dry-run)
- [ ] winner_top1 ≥ 30% / shift ≤ 12x 検証

## Day 6-7: 5/23-24 GO/no-go 判定 + 投入

### 5/23 (金) - V20 GO/no-go 判定
判定基準 (全 PASS で GO):
- [ ] WF AUC ≥ 0.880 (V15 0.8939 比 ±)
- [ ] 全年 AUC ≥ 0.85 (gap < 0.05)
- [ ] 実 ROI 全条件 ≥ 100% (V15 baseline 維持)
- [ ] LIVE retro winner_top1 ≥ 30%
- [ ] LIVE retro shift ≤ 12x
- [ ] LEAK 監査 PASS (skb 除外 + sib_exp 修正)

NO-GO 時 → V15 案 B 改 + 戦略⑦ 単独 継続 (5/30, 5/31 開催)

### 5/24 (土) - V20 投入 開始 (GO の場合)

**段階的投入** (V20 投資保護 + V15 安全 net):
- 5/24 開催: 案 B 改 (V15) 7 race × 700 円 = 4,900 円 + V20 shadow 並行
- 5/25 開催: V20 shadow log 確認後、 案 C 改 (V20) 並行 試験 1-2 race × 700 円
- 投資額 上限 5,000 円/日

### 5/25 (日) - 続行

## V21 投入 path (6/8 → 前倒し候補)

V21 = V20 + video features 学習 (paddock 動画 数千 race 蓄積後)

### 6/1-6/7 (1 週間 学習 + 検証)
- paddock 動画 ~3,000 race × 5 馬 = 15K records 必要
- 5/17-5/30 で 2 週間 × 36 races × 5 馬 = 360 records 蓄積想定
- 不足分は 過去 archive build (~ 1,800 race)

### 6/8 (土) - V21 GO/no-go 判定 + 投入

## V22 RL 投入 path (9/1 → 7-8 月候補)

V22 = V21 + Reinforcement Learning bet sizing (V22 plan、 commit `109dacfe`)

### 7-8 月: V22 学習 + paper trading
### 9/1: GO/no-go 判定

## 月額 cost (5/24+ 想定)

| source | 月額 | 状態 |
|--------|------|------|
| netkeiba マスターコース | ¥4,980 | 維持 |
| JRDB Advance | ~¥2,000 | 維持 |
| JRA-VAN DataLab | ¥2,090 | **継続** (NEXT 変更 NG 確定済) |
| TARGET frontier JV | included | 維持 |
| **合計** | **約 ¥9,070/月** | 現状維持 |

YouTube 録画は無料、 アメダス無料、 JRA 公式無料。

## Risk + mitigation

| risk | mitigation |
|------|-----------|
| V20 LEAK 再発 (skb / sib類似) | 監査 22 項目 + LEAK_FEATURES + WF gap < 0.05 |
| V20 投入で V15 production 影響 | V15 model file 完全 freeze、 V20 は別 file |
| paddock 動画 蓄積 不足 (V21 用) | 過去 archive build 並行 |
| 30y backtest 容量 (135 GB) | 段階分割、 SSD 残量確認 |
| schtask 失敗連鎖 | nightly_sanity + morning_briefing 監視 |
| Kelly bet sizing で 過大 投資 | fractional 0.25x + max_bet 5% cap |

## V15 投資保護 (絶対遵守、 5/24+ も継続)

V20 投入 後 1 ヶ月 (6/24 まで) は V15 model file `keiba_model_v135_central_live.pkl.gz`、
`keiba_model_v135_central.pkl.gz` を **完全 freeze** で archive。

7/1+ V20 安定運用確認後、 V15 → archive/v15/ に移動判定。
それまで V15 と V20 並行 GUI (Streamlit) で 比較可能維持。

## 撤退 line

- 撤退 line: 累計 -50,000 円
- 5/17 現状: 累計 +14,140 円 (撤退余裕 +64,140 円)
- 月間 期待 利益: V15 +28,953 円、 V20 +50,000-100,000 円
- 撤退 trigger: Drawdown breaker STOP 自動 (-50,000)

## 結論

5/17 開催 を起点として、 V15 安定運用 + Phase 23 shadow mode で **真の比較 data 蓄積**。
5/24 に V20 段階投入、 6/8 V21、 9/1 V22 の path 明確化。

V15 完全保護 + 段階的 V20→V21→V22 投入 で **市場圧倒 5 領域 + 追加 2 領域** を達成。
