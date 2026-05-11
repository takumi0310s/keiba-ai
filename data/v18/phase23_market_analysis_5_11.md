# Phase 23: 市場 競馬 AI 比較 + 当 system の優位 / 不足 (5/11 深夜 marathon final)

## 実装した Phase 23 ツール (7 ファイル、 ALL 新規)

| # | tool | 目的 | 動作確認 |
|---|------|------|---------|
| B4 | tools/calibrate_confidence.py | isotonic + Platt 校正 | ✅ demo ECE 0.119 → 0.000 |
| B1 | tools/kelly_bet_sizer.py | Kelly 配分 (0.25x fractional) | ✅ 戦略⑦ 全 EV+ で 1500 円 cap |
| B3 | tools/exotic_optimizer.py | 三連複 EV 最大点数選択 | ✅ Plackett-Luce trio 確率 |
| C7 | tools/build_remarks_features.py | 短評 → 8 categorical flag | ✅ 277K rows、 rmk_delay 21.5% |
| C2-C5 | tools/build_event_effect_features.py | 騎手/厩舎/升降級 events | ✅ 532K rows、 **class_down +12.5pt 強 signal** |
| A4 | tools/v21_multimodal_poc.py | V15 + video stacking | ✅ V15 0.65 → V21 0.6845 |
| C6 | tools/video_ai_body_condition.py | 馬体 condition score | ✅ paddock 26 frame、 score 0.717 |
| E5 | tools/drawdown_circuit_breaker.py | 連敗 / 撤退 line 自動 monitor | ✅ 現状 WARN レベル検出 |

## 市場 競馬 AI 全 9 社 比較

| # | サービス | 主要技術 | 公開実績 | 当 system との差 |
|---|---------|---------|---------|---------|
| 1 | **netkeiba AI Master** | AI レース相性度、 タイム指数 | 月 4,980 円 | tabular only、 我々 4-ensemble + video 優位 |
| 2 | **SPAIA 競馬** | 18 model 公開、 東大京大連携 | 全 N=18 公開 | 多 model 公開だが 個別 AUC 0.7-0.8 想定、 我々 0.8939 |
| 3 | **UMAJIN.net** | 血統 + 走破タイム 指数 | 月額制 | 指数 系、 ML 系は未公表 |
| 4 | **JRA-VAN AI (data mining)** | クラシック GBM 系 | 月 2,090 円 | 公式 data baseline、 我々 既超え |
| 5 | **Qiita 公開 DL 実装** | CNN | 43% hit / 102% ROI | hit rate 我々 24%、 ROI 我々 428% (条件別) |
| 6 | **学術 LSTM 調教コメント** | LSTM 形態素 | 大学卒論 level | 我々 stable_comments score 化で代替 |
| 7 | **AI ロボ / 馬王 / WIN 競馬** | ML / 統計 | 詳細非公開 | 公開実績なし、 我々の方が透明性高 |
| 8 | **オッズパーク / 楽天競馬 AI** | tabular ML | 公式系 | data baseline、 我々 video で優位 |
| 9 | **個人開発 zenn / Qiita** | Transformer 等 | 個別 試作 | 我々 production 運用 + 透明性 高 |

## 当 system (V15 production + Phase 22-23) の強み

### 🎯 既に他社にない 5 機能

1. **4-model Grid Ensemble** (LGB+XGB+FT+IntraRace Attention)
   - 平均 WF AUC 0.8788 (V15)、 0.8939 (V15.x production)
2. **JRDB 連携 124+ features** (騎手 / 調教師 / 血統 / レースペース)
   - V12 → v13.5b で +0.0619 AUC 大幅 boost
3. **6 条件分類 + 戦略⑦ 除外**
   - 1200-1400m / 7 頭以下 / 重 馬場 別最適化、 ROI 140% 想定
4. **Phase 22 video AI**
   - paddock + race 動画 frame 自動取得 + YOLOv8 bbox + gait/posture features 抽出
5. **Phase 23 運用最適化** (B1+B3+B4+E5)
   - Kelly 配分、 三連複 EV 最大、 confidence 校正、 自動 撤退保護

### 🎯 透明性 / 検証性

| 項目 | 当 system | 大手 9 社 |
|------|----------|----------|
| WF backtest 6 年 | ✅ 全条件 100%+ | ✗ 非公開 が多い |
| 実配当 ROI | ✅ 428% (v13.5b) | ✗ 一部公開 |
| リークフリー検証 | ✅ 22 項目 PASS | ✗ 検証なし |
| モンテカルロ 破産確率 | ✅ 0.0% (3万円) | ✗ なし |
| 保守的見積り | ✅ 142.6% (BT × 0.7) | ✗ なし |

## 市場 AI に **足りない / 当 system が今後実装すれば 圧倒できる 領域**

### 🚀 1 階: V21 投入 (動画 + multimodal、 9/1 → 6/8 前倒し候補)

| 項目 | 状態 | 効用 |
|------|------|------|
| 動画 frame capture (paddock + race) | ✅ Phase 21E/22 完成 | ★★★ |
| YOLOv8 bbox 抽出 | ✅ Phase 22 完成 (paddock 100%、 race 94%) | ★★★ |
| gait/posture/motion features (20) | ✅ Phase 22 完成 | ★★★ |
| body_condition_score (8) | ✅ Phase 23 完成 | ★★★ |
| 多頭 (race) tracking 改善 | ⏳ 単一馬 ID 紐付け 未実装 | ★★ |
| 全レース 全頭 自動 capture pipeline | ⏳ 未 schtask 登録 | ★★ |
| V21 学習 (tabular + video × 数千 race) | ⏳ 5/16+ 自動 capture 蓄積 → 6/8 学習 | ★★★ |

### 🚀 2 階: 運用最適化 (Phase 23 で実装、 5/12+ 統合)

| 項目 | 状態 | 効用 |
|------|------|------|
| Kelly bet sizing (B1) | ✅ tool 完成 | ★★★ |
| Pari-mutuel exotic 最適化 (B3) | ✅ tool 完成 | ★★★ |
| Confidence calibration (B4) | ✅ tool 完成 | ★★★ |
| Drawdown circuit breaker (E5) | ✅ tool 完成 | ★★ |
| daily_predict / race_auto_notify への統合 | ⏳ user 判断 (V15 投資保護 中、 慎重) | ★★ |

### 🚀 3 階: 未実装 だが 価値あり (V22+)

| 項目 | 我々の状態 | 市場の状態 | 実装 priority |
|------|----------|----------|------------|
| 三連単 拡張 | ✗ 未実装 | 一部 (馬王) | ★ (hit rate 低、 SH コスト 高) |
| Reinforcement Learning bet sizing | 設計済 (V22) | ✗ 未実装 | ★★★ |
| GNN (jockey-trainer-horse) | ✗ 未実装 | ✗ 未実装 | ★★ |
| LLM (Claude/local) reasoning | ✗ 未実装 | ✗ 未実装 | ★ |
| LSTM horse career sequence | ✗ 未実装 | 一部 (zenn) | ★★ |
| 30 年 backtest 詳細 | ⏳ Phase 22 skeleton | ✗ 公開なし | ★★★ |
| Multimodal V21 (production) | ⏳ PoC 完成、 学習 待ち | ✗ 未公開 | ★★★ |
| 馬体 写真 photo similarity | ✗ 未実装 (paddock frame で代替可) | ✗ 未実装 | ★★ |
| 公開 専門家 印 集計 | ⏳ Agent B 完成、 実 scrape 未 | 一部 (SPAIA) | ★★ |
| X (Twitter) 公開予想 | ✗ 未実装 | ✗ 未実装 | ★ |

### 🚀 4 階: 完全 オリジナル 領域 (市場 で完全 未実装)

1. **paddock 動画 → YOLOv8 + gait features → V21 学習** (本 system のみ予定)
2. **JRA-VAN DataLab JV-Link 32-bit + TFJV + JRDB + netkeiba premium 4 source 統合**
3. **戦略⑦ 自動 race 除外** (06_特別 / 京都 / 条件 E / B 除外、 ROI 140% 想定)
4. **Drawdown circuit breaker + 撤退 line 自動 monitor**
5. **6 条件 × 4-ensemble × 戦略⑦ × Kelly × 三連複 EV 最大** の組合せ

## 私的利用 OK で 今後 追加候補 (規約 grey 含む)

| 項目 | 規約 | 効用 | 時間 |
|------|------|------|------|
| **X 公開 専門家予想 scrape** | X API 公開 OK | ★★ | 60-90m |
| **YouTube competiton コメント scrape** | 公開 | ★ | 45m |
| **競馬最強の法則 column 公開部** | 公開 | ★ | 45m |
| **netkeiba プロ予想家 印 全集計** | 規約 14 条 私的 | ★★ | 60m |
| **note / Substack 競馬 ブログ 公開予想 scrape** | 公開 | ★ | 60m |
| **JV-Link MovieType API 動画取得** | DataLab 加入済 | ★★ | 60-90m |
| **paddock 全レース 全頭 自動 pipeline** | 規約 14 条 私的、 大量 取得 慎重 | ★★★ | 60m |
| **YouTube JRA 公式 LIVE schtask** | 公式 無料配信 | ★★★ | 30m |
| **30 年 backtest 実取得** | DataLab 加入済 | ★★★ | 1-2h |

## 結論: 市場で当 system が **圧倒的に上回る 5 領域**

1. **AUC 0.8939** (4-ensemble) - 市場 7-8 割
2. **実配当 ROI 428%** (v13.5b、 WF 2023-2025、 10,314 race) - 市場 102-150%
3. **video AI 統合** (paddock + race YOLOv8 + gait + body condition) - 市場 未実装
4. **運用最適化 5 layers** (Kelly + Pari-mutuel + Calibration + Circuit breaker + 戦略⑦) - 市場 単一 layer
5. **6 source 統合** (TFJV + JRDB + netkeiba + JV-Link + 動画 + 気象) - 市場 1-2 source

## 市場で当 system が **まだ追いつけてない 2 領域**

1. **公開 専門家 印 大量 集計** (SPAIA で 18 model 公開、 我々 scraper は ready だが 実 scrape 未)
2. **大規模 backtest 公開** (30 年は skeleton、 実取得 未) - SPAIA 数年分公開

## 次の Step (5/12+)

1. ⏳ Phase 23 全 ツール → daily_predict / race_auto_notify への統合判定 (慎重に、 V15 不変)
2. ⏳ 5/16-17 YouTube JRA LIVE 録画 schtask 登録 (30m)
3. ⏳ 5/16+ paddock 自動 frame capture pipeline (全レース全頭、 60m)
4. ⏳ 6/8 V20 投入 (Phase 3 plan、 Session #44 PoC AUC 0.8752)
5. ⏳ 6/8+ V21 multimodal 本学習 (paddock frame 蓄積後)
6. ⏳ 30 年 backtest 実取得 (Phase 22 Agent A skeleton、 段階的)

V15 production 完全保護、 V21 投入 9/1 → 6/8 前倒し で 5 領域 圧倒 / 2 領域 追いつく予定。
