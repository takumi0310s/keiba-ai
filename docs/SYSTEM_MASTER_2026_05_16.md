# 競馬 AI 予測 system — 包括 master document (2026-05-16 evening 時点)

> **本 doc の目的**: Claude (またはユーザー) が **これ 1 つ読むだけで system 全体を完全把握** できるようにする。 CLAUDE.md / MEMORY.md / data/v21/ 等の各種 doc の **集約版**。
>
> **honest 厳守**: fabrication 0、 数値は全 出典付き、 想定値は明示、 「業界最高」 「最強」 表現 排除。
>
> **作成**: AI 自律 (claude-opus-4-7) 3 並行 agent + 親統合。 source: A_features_full.md / B_market_research.md / C_persona_swot.md (data/v21/inventory_5_16/)

---

## 0. Executive summary (★ 3 段落で 全体把握 ★)

**system 概要**: 中央 JRA 競馬予測 AI。 V15 (★ **LGB + XGB 2-model** production ★、 booster 145 features、 5/17 V15-audit-1 確定) を本番運用。 ※ 旧記述 「4-model ensemble (LGB+XGB+FT+IR)、 150 features」 は drift — FT/IR は v15_master の WF 評価専用で `.pkl.gz` には未保存。 リークフリー設計を厳守し、 朝 06:30 自動全 R 予測 → 5 分前 Discord 通知 → 夜 20:00 結果照合の自動 pipeline を 7 schtasks で構成。 戦略⑦ (06_平場特別 / 京都 / 条件 E / B 除外) + 案 B 改 strict で 投票精度向上。

**現状 数値 (★ 5/17 V15-audit 真値 ★)**: stored `.pkl.auc` 0.8939 は LGB train-set self-eval (in-sample)、 **genuine WF 6-fold mean LGB+XGB = 0.8678** / 4-model Grid 5-fold mean = 0.8858 (V15-audit-2)。 v13.5b backtest ROI 428.4% (歴史 reference)。 **実運用累計 ROI 98.34% / PnL ¥-6,920 / n=596** (≤2026-05-17、 V15-audit-4、 cumulative_results.csv settled)。 bootstrap 95% CI [66.33%, 138.05%] 100% 含む → ★ 統計的有意 勝ち なし ★。 撤退余裕 ¥43,080 (¥-50,000 ライン まで)。

**最大 opportunity と threat**: opportunity = **V21 動画 features** (パドック 12 + パトロール 8 + 調教 10 = 30 features)、 商用 競合不在で業界 frontier。 threat = **V15 plateau** (V22 / V20+ 改善試行 8 回全 fail) + **京都 ROI 20%** + **backtest 428% vs live 93% の大幅乖離**。 30 日 priority 3 件: ①京都/中京 戦略⑦再除外 (+5-7pt ROI 想定)、 ②calibrator v2 paper eval 30 R 後採用、 ③V21 動画 5/31+ production 化。

---

## 1. 実装済み 機能 (★ 本番運用中 ★)

### 1-1. モデル (8 versions、 V15 のみ production)

| モデル | Status | WF AUC | 実 ROI | 投入日 | 備考 |
|--------|--------|--------|----------|--------|------|
| **V15** | ★ **production** ★ | stored 0.8939 (= in-sample LGB train-set, LEAKY) / **genuine WF 0.8678** (LGB+XGB) / Grid 4-model 5-fold 0.8858 | **98.34%** (cumulative 真値、 n=596、 5/17 V15-audit-4) | 4/1 | ★ **LGB+XGB 2-model** ★ (mlp=None, FT/IR は .pkl 未保存)、 145 features booster |
| V15.1 | NO-GO | 0.8943 | — | — | SKB post-race leak 確定 (skb_kishi_code_3 +480bp) |
| V18/V19 | NO-GO | 0.886-0.887 | -10pt LIVE | — | sib 抜き hybrid、 5/16 投入 NO-GO 確定 |
| V20 | PoC 開発中 | 0.8752 | unknown | 6/8→6/30 GO 検討 | NAR+JRA、 320 features、 TFJV + JRDB 統合 |
| V21 | 設計済 | est. 0.91-0.93 | est. 130%+ | 6/1+ paper trade | V15 + 動画 30 features stacking、 V15 完全不変保証 |
| V22 base | NO-GO | 0.88 | — | — | 4-ensemble、 全 baseline 以下 |
| V22 distill | NO-GO | 0.886 | — | — | 5/15 alpha、 8 試行目 honest 不採用 |
| V22 enhanced | NO-GO | 0.887 | — | — | top100 features、 V15 越え未達 |

### 1-2. 予測 pipeline (7 components)

| Component | 役割 | 起動 | 出力 |
|-----------|------|------|------|
| `tools/predict_core.py` | V15 core inference (Pattern B list 150 / booster **145** truncate、 V15-audit-1) | オンデマンド | `df['final_score']` |
| `tools/daily_predict.py` | 朝 06:30 全 R 予測 | schtask daily | `data/daily_predictions/{ymd}.csv` |
| `tools/race_auto_notify.py` | 5 分前リアルタイム + Discord | schtask race-time | Discord #bets |
| `tools/daily_results.py` | 夜 20:00 結果照合 + 配当確定 | schtask daily | `data/daily_results/{ymd}.csv` |
| `tools/save_all_horse_scores.py` | V15 全馬 score 記録 | オンデマンド | `data/live_features/{ymd}.csv` |
| `tools/strategy_layer_v2.py` (5/16 NEW) | 買い目生成 + calibrator v1/v2 | shadow only | `data/v21/strategy_v2_shadow_{ymd}.csv` |
| `app.py` | Streamlit UI (netkeiba URL 入力) | 常時起動 | ブラウザ予測 |

### 1-3. データ取得 (8 source、 行数 + 更新)

| Source | 種別 | 行数 | 更新 | 用途 |
|--------|------|------|------|------|
| jra_races_full.csv | JRA 全レース履歴 | 781,161 | weekly | 基本レース情報 |
| training_times.csv | 調教タイム | 955,581 | daily | 調教 12 features |
| odds_history.csv | オッズ履歴 | 778,388 | daily | オッズ 8 features |
| JRDB (17 datatypes) | JRDB 公式 | 548K+ (各) | daily | 124 features (jo/sed/tyb/skb 等) |
| netkeiba (15+ source) | netkeiba.com | 531K+ | daily | 22 features (speed_index 等) |
| 気象庁 API | 天気・馬場 | 数千/年 | daily | 天候 4 features |
| TARGET JV / JV-Link | JRA-VAN 公式 | 43K files (TFJV) | daily | V20 候補、 parser 待ち |
| NAR (地方競馬) | NAR 公式 | 月 5K+ | weekly | V15 未使用、 V20 統合予定 |

### 1-4. 戦略 layer (6 種、 production 2 + dev 4)

| 戦略 | 種別 | 対象 | 効果 | Status |
|------|------|------|------|--------|
| **戦略⑦** | フィルタ | 06_平場特別 / 京都 / 条件 E / B 除外 | +3.67pt (cumulative 529 settled) | production |
| **案 B 改 strict** | 上限管理 | 1 勝クラス 12R のみ、 2,100 円/日上限 | リスク軽減 | production |
| calibrator v1 | 確率補正 | 全 R | 21 sample、 isotonic 飽和 problem | production |
| calibrator v2 | 確率補正 (5/16 NEW) | 全 R | 315 sample、 飽和解消、 iso(0.3)=0.59 | shadow eval (5/18+) |
| Strategy 8 Jackpot | 高配当狙い | top3 53.6% verified | shadow GO | 5/16 LIVE 検討 |
| V21 video stacking | meta-model | 動画 features 30 | est. +0.005 AUC | 6/1+ paper trade |

### 1-5. 自動運用 (7 schtasks)

| Task | Cron | 役割 |
|------|------|------|
| DailyPredict | 06:30 | 全 R 朝予測 |
| RaceAutoNotify | race 1h 前~ | 5 分前リアルタイム + Discord |
| DailyResults | 20:00 | 結果回収 + ROI 集計 |
| WeeklyReport | 日 19:00 | 週間集計 |
| MorningWeightCheck | 09:30 | 馬体重補正 ±15kg alert |
| ProcessWatchdog | 30min | Python/bat 監視再起動 |
| JRDB retry | 09:00 | JRDB 取得失敗 リカバリ |
| Keiba-NightlySanity | 23:00 | 翌日タスク事前検証 + Discord |
| SCRAPER-GUARD | 金 22:00-月 06:00 | 規約遵守 自動停止 |

### 1-6. 通知 (Discord 3 channel)

| Channel | 内容 | 頻度 |
|---------|------|------|
| #bets | 買い目 + フォーメーション + EV | 毎 R × 700円 |
| #updates | scrape 完了 / errors / alerts | event-driven |
| #results | 日間 / 週間 集計 (ROI/hits) | daily/weekly |

### 1-7. 検証・モニタリング (8 components)

cumulative_results.csv (564 rows、 投票実績) / monte_carlo_sim.py / leak_comparison_*.json / backtest_*.py / weekly_report / drawdown_analysis.json / drift_detector.bat / nightly_sanity_check.bat

### 1-8. データ資産 (15+ csv、 計 数 million rows)

jra_races (781K) / training (955K) / odds_history (778K) / netkeiba_siblings_exp (531K) / netkeiba_training_eval (531K) / jrdb (17 種 × 548K) / features_merged_all (467K) / calibration_full (315) / cumulative_results (564) / daily_predictions (40/day × 30 days)

---

## 2. 実装中 / paper eval 中 (8 items)

| # | Item | Status | 期限 | 投入判定 |
|---|------|--------|------|---------|
| A | V21 動画 features | Phase A POC、 coverage 0% | 6/30 → 7/1 GO | 1,000+ R coverage、 AUC +0.02 |
|   | — paddock 12 | 89 entries 解析済 | 5/31 1K | YOLOv8 / gait / body_condition |
|   | — patrol 8 | YOLO skeleton | 5/24 PoC | object detection 馬番 / 順位 |
|   | — chokyou 10 | keypoint 設計 | 5/31 PoC | stride / angle / posture |
| B | **calibrator v2 (5/16 NEW)** | 315 sample、 iso 飽和解消 | 5/24 paper eval 判定 | 30 R 蓄積 v1 vs v2 比較 |
| C | Strategy 8 Jackpot | 53.6% top3 verified | 5/31 paper trade | shadow ROI vs 実測 |
| D | V20 ensemble | 320 features、 LGB+XGB training | 6/8 GO 候補 | WF AUC > V15 + 120% ROI 継続 |
| E | V21 paper trade | architecture 完成、 6/1 開始 | 6/30 GO/no-go | V21 ROI > V15 |
| F | JV-Link COM unlock | 5/15 完了 | 6/1 production fetch | TFJV full 解析 → V20 |
| G | 完全自動化 plan | 5/15 80%、 9/2 90%、 12/1 100% | 9/2 review | admin touchpoint 削減 |
| H | **京都 再除外** (★ 5/16 NEW) | ROI 20% 発見 (N=58) | 5/31 判定 | +5pt ROI 想定 |

---

## 3. 設計済 / 未着手 (5 items)

| # | Item | 着手 | 工数 | 備考 |
|---|------|------|------|------|
| A | パドック video coverage 加速 (33→1K+ R) | 5/17-5/31 | 80h+ | OBS 録画 manual、 frame batch |
| B | パトロール YOLO PoC | 5/18-5/24 | 20h | 馬番 / 着順 / ペース |
| C | 調教 keypoint (10 features) | 5/25-5/31 | 25h | DLC HORSE-10 base |
| D | V21 production 投入判定 | 7/1 review | — | paper trade 6/1-6/30 後 |
| E | V20 NAR 統合 (地方競馬) | Q3 2026 | 40h+ | 別 pipeline 構築 |

---

## 4. 競合 AI サービス 比較 (11 services 調査)

| サービス | AI 機能 | データ | 月額 | 強み | 弱み |
|----------|---------|--------|------|------|------|
| **netkeiba AI** (公式) | AI 予想オッズ + 馬券診断 + IPAT 連携 | netkeiba 内部 + JRA | ¥690-1,490/30 日 | ユーザー基盤最大、 IPAT 連携 | model 不透明 |
| **SPAIA 競馬** | 18 種類 AI 並走 + IPAT + 京大/東大 研究系 | 過去 R + 騎手/馬場/天候 | ¥500-1,500 | 18 AI 並走、 14 日無料 | 選択困難 |
| **JRA-VAN ネクスト** | 公式 30 年 data + AI データマイニング | JRA 公式 30 年 | ¥880 | 公式信頼性最高 | UI 古い |
| **JRA レーシングビュアー** | パドック (15 分前) + 調教 + パトロール 全部入り | JRA 公式映像 | ¥550 | ★ 動画素材最強 ★ AI 解析なし | AI なし |
| **TARGET frontier JV** | 30 年 data ローカル分析 (AI なし) | JRA-VAN | 無料 + ¥2,090 | カスタム分析基盤 | AI なし |
| **競馬ブック Smart** | 取材一次情報 + TM 予想 + コンピュータ予想 | 自社取材 | ¥600 | 厩舎/調教取材 | AI 弱め |
| **ATHENA** | 全 R 全頭着順予想 + 各券種買い目 | 過去 3 万 R | **無料** | 全頭順位公開 | 中央平地のみ |
| **VUMA** | ワイド予想 AI | unknown | 14 日無料 | ワイド specialist 40% 的中 | ワイドのみ |
| **ROBOTIP スーパー** | カスタム予想エンジン (6 ファクター × 5 適性) + 自動投票 | U 指数 | ¥5,500 + 3% | カスタム + 自動投票 | 高、 ルールベース |
| **EquinEdge** (US) | Pace Figure 72.5%、 Ticket Generator | US racing data | $5.95-699.95 | Ticket Generator | 日本非対応 |
| **ChatGPT 系 LLM** | プロンプトベース picks + commentary | ユーザー提供 data | $20/月 | 自然言語対話 | hallucination リスク |

---

## 5. ★ 当 system gap 分析 (22 件、 深刻度別) ★

| # | feature | 競合 | 当 system | gap |
|---|---------|------|-----------|-----|
| 1 | **IPAT 自動投票連携** | netkeiba/SPAIA/ROBOTIP/KSC | × | **高** (運用効率直結) |
| 2 | **重賞 G1 専門 model** | DX 指数アプリ、 競馬ブック TM | × (戦略⑦で除外したまま) | **高** (重賞は高 ROI source) |
| 3 | AI 予想オッズ (出走想定段階) | netkeiba | × | 中 |
| 4 | 三連単 / WIDE / 馬単 拡張 | EquinEdge / ROBOTIP / 多数 | × (三連複 + 馬連) | 中 |
| 5 | リアルタイム odds 連続追跡 | オッズ期待値アナライザー / SPAIA | × (5 分前 snapshot のみ) | 中 |
| 6 | NAR 統合 | SPAIA 地方 / 楽天競馬 | × (V20 で着手) | 中 |
| 7 | LLM race commentary 解釈 | ChatGPT 系 | × | 中 frontier |
| 8 | payout 分布予測 (期待値精度) | netkeiba AI | × (固定 700 円) | 中 |
| 9 | pace 配分シミュレーション | EquinEdge Pace Figure | △ (pci feature あり) | 中 |
| 10 | **動画 AI 解析** (paddock/patrol/chokyou) | ★ 業界全体 未提供 ★ | △ (V21 開発中) | **当 system 先行 opportunity** |
| 11 | gait analysis / pose estimation | 学術のみ (DLC SuperAnimal) | △ (V21 設計) | frontier |
| 12 | 取材一次情報 (厩舎コメント) | 競馬ブック / KEIBABOOK | △ (netkeiba premium コメント取得済) | 低 |
| 13 | LINE 通知 | 一般予想サイト | × Discord のみ | 低 |
| 14 | AI 馬券診断 (ユーザー履歴) | netkeiba | △ (cumulative 集計) | 低 |
| 15 | 馬主 / 牧場 系統 | ROBOTIP / EquinEdge GSR | × | 低 |
| 16 | 障害競走 model | netkeiba 一部、 競馬ブック | × archive 化 | 低 |
| 17 | 馬体重当日補正高度化 | 競馬ブック パドック取材 | △ (09:30 ±15kg alert) | 低 |
| 18 | 強化学習 RL bet sizer | 学術 / 個人 | △ (V22 RL 試行済、 効果薄) | 低 |
| 19 | モバイル UI | 全サービス | × Streamlit のみ | 低 |
| 20 | 海外調教馬特殊扱い | 競馬ブック | × | 低 |
| 21 | マークシート出力 | KSC 自動投票 Plus | × | 低 |
| 22 | 複数 model A/B test 基盤 | SPAIA 18 model | × (model 単一切替) | 低 |

---

## 6. ★ 的中率/ROI 向上候補 (priority sort) ★

| # | feature | 期待 AUC | 期待 ROI | 工数 | risk | 優先度 |
|---|---------|---------|----------|------|------|--------|
| 1 | **V21 動画 features 完成** | est. +0.005 | est. +5pt | 5/31+ | 中 | ★★★★ |
| 2 | **京都 / 中京 戦略⑦再除外** | n/a | est. +5-7pt | 1 日 | 低 | ★★★★ |
| 3 | **重賞 G1 専門 model** | est. +0.003 | est. +10pt (重賞復帰) | 1-2 週 | 中 | ★★★ |
| 4 | Strategy 8 Jackpot LIVE 投入 | n/a | est. +5-10pt | 1 週 | 中 | ★★★ |
| 5 | **calibrator v2 採用** | n/a | est. +2-3pt | 5/18+ 検証中 | 低 | ★★★ |
| 6 | リアルタイム odds 連続追跡 | n/a | est. +2-5pt | 2-3 週 | 低 | ★★ |
| 7 | AI 予想オッズ自前推定 | n/a | est. +3pt | 2 週 | 中 | ★★ |
| 8 | 期待値連動投資額 (Kelly) | n/a | est. +3-5pt | 2 週 | 中 | ★★ |
| 9 | NAR 統合 (V20) | est. +0.002 | est. R 数 2x | V20 内 | 中 | ★★ |
| 10 | IPAT 自動投票連携 | n/a | 運用効率 | 1 週 | 低 | ★★ |
| 11 | 三連単 / WIDE 拡張 | n/a | +/-? (hit 率 risk) | 2-3 週 | 高 | ★★ |
| 12 | LLM race commentary 解釈 | est. ? | est. ? | 1-2 週 PoC | 中 | ★ |
| 13 | pace 配分シミュレーション | est. +0.001 | est. +1-2pt | 2 週 | 低 | ★ |
| 14 | 馬主 / 牧場 features | est. +0.0005 | est. +0pt | 1 週 | 低 | ★ |
| 15 | LINE 通知 | n/a | n/a | 1-2 日 | 低 | ★ |

---

## 7. 5 ペルソナ評価 (★ 平均 3.6/5 ★)

### 7-1. ペルソナ 1: 初心者 (馬券初購入、 月 1 万円) — ★★★☆☆ (3/5)

**強み**: Discord 通知が「買い目そのまま」 で コピペ購入可能、 戦略⑦ で 弱い R 自動除外、 monte_carlo 検証で 3 万円以上 破産率 0%。
**弱み**: ★ 月 1 万円 では 月 30R × 700 = 21,000 円必要 (予算 mismatch) ★、 「なぜこの馬?」 説明 薄い、 京都 ROI 20% を 自分で気づけない、 UI が開発者向け。
**最優先改善**: 月予算入力 → 自動 R 数調整 + 「なぜ」 1 行説明 (LLM 統合)。

### 7-2. ペルソナ 2: 中級者 (年 10-12 万円投資) — ★★★★☆ (4/5)

**強み**: 自動運用完成 (6+ schtasks)、 透明性 (cumulative_results.csv)、 戦略⑦ + 案 B 改で micro-management 不要、 genuine WF AUC 0.8678 (V15-audit-2、 LGB+XGB) 数値根拠。
**弱み**: ★ 実 ROI 98.34% (V15-audit-4、 5/17 反映)、 CI [66.33%, 138.05%] 100% 含む = 統計的有意なし ★、 戦略⑦ 除外 R で機会損失感、 カスタマイズ性低 (700 円 hardcoded)、 京都 ROI 20% 発見 lag 1 ヶ月超。
**最優先改善**: ROI 集計 統一 (乖離解消) + 月次 Discord ROI summary。

### 7-3. ペルソナ 3: プロ (年 100 万円+) — ★★★☆☆ (3/5)

**強み**: リークフリー設計監査可能 (LEAK_FEATURES 18 件)、 HONEST report 文化、 WF 6-fold + expanding window、 4 source data 統合。
**弱み**: ★ scaling 致命的 ★ (700 円/R × 50 倍 = odds 押し下げ)、 戦略⑦除外 R で 年 GI 24R 全 skip、 三連単未拡張、 リアルタイム odds 5 分前 snapshot のみ、 ★ V15 plateau (V22 8 fail) で技術天井 ★。
**最優先改善**: 重賞専門 model + 三連単拡張 + 投票額 Kelly criterion。

### 7-4. ペルソナ 4: データサイエンティスト — ★★★★☆ (4/5)

**強み**: ★ リークフリー設計 + 失敗教訓蓄積 ★ (odds_log / SKB / dam_top3r / sib hybrid 等)、 HONEST report 文化 (V22 8 fail 記述)、 v15_master では 4-model Grid ensemble IR 35% 貢献 (WF 評価専用、 production の .pkl は LGB+XGB のみ — V15-audit-1)、 expanding window 厳守、 WF 6-fold + 年別 gap 監視 (>0.05 過学習)。
**弱み**: ★ V22 / V20+ 8 fail → V15 plateau ★、 features 124→150 で AUC delta ~0 (diminishing returns)、 ★ backtest 428% vs live 93% **大幅乖離** ★、 v13.5b Grid 年ごと最適化 (test leak 疑い)、 calibrator v1 21 sample over-fit、 push 不能 (reproducibility risk)。
**最優先改善**: Grid Search 重み CV 固定 + backtest vs live ROI 乖離 formal analysis。

### 7-5. ペルソナ 5: 完全自動運用志望 (月 5 万円、 5 年持続 priority) — ★★★★☆ (4/5)

**強み**: 6+ schtasks 自動運用 + process_watchdog v2 + Keiba-NightlySanity (23:00) + SCRAPER-GUARD、 4/19 事故対応 17 task ALL PASS、 月額 ~6,500 円で 月 1-3 万 ROI 想定、 完全自動化 plan 策定済。
**弱み**: ★ マークカード / 自動投票 未連携 (0-touch 不可) ★、 Cookie 期限切れ risk、 データ source 規約変更 risk、 ★ V15 plateau で 5 年後 退化可能性 ★、 京都 ROI 20% を「手動 audit」 で発見 (自動監視不在)、 jrdb_paci.csv 4/4 停止 / jra_payouts 4/6 停止 等既知 bug。
**最優先改善**: 自動 model drift 検出 + 月次 alert (course/条件別 ROI < 80% で Discord)。

---

## 8. SWOT 完全版

### 8-1. Strengths (12 件、 evidence 付き)

| # | 強み | evidence |
|---|------|---------|
| S1 | genuine WF AUC 0.8678 (V15 LGB+XGB) / 0.8858 (Grid 4-model 5-fold) | V15-audit-2 (★ 旧記述 0.8939 は LGB train-set self-eval、 in-sample LEAKY ★) |
| S2 | v13.5b backtest ROI 428.4% | CLAUDE.md (JRA 公式配当 2023-2025、 10,314 R) ※ 歴史 reference、 V15 cumulative とは別 |
| S3 | リークフリー設計 — LEAK_FEATURES 18 件明示 | CLAUDE.md 8 章 |
| S4 | v15_master の 4-model Grid ensemble (IR 35% 貢献、 WF 評価専用) | V15-audit-1 (★ production .pkl は LGB+XGB only、 FT/IR 未保存 ★) |
| S5 | 7 schtasks 自動運用 | CLAUDE.md 定期タスク |
| S6 | Discord 3 channel リアルタイム通知 | CLAUDE.md |
| S7 | HONEST report 文化 (V22 8 fail / SKB LEAK / dam_top3r 等 documented) | CLAUDE.md |
| S8 | 4 source 統合 (JRDB + netkeiba + TFJV + JV-Link) | MEMORY |
| S9 | 撤退ライン -50,000 円 / monte_carlo 破産率 0% (3 万円+) | CLAUDE.md |
| S10 | process_watchdog v2 + Keiba-NightlySanity | CLAUDE.md 4/19 事故対応 |
| S11 | V21 動画 features architecture (V15 不変保証) | phase_d_v21_architecture_design |
| S12 | JV-Link COM unlock (5/15) | MEMORY |

### 8-2. Weaknesses (15 件)

| # | 弱み | evidence |
|---|------|---------|
| W1 | ★ 実運用 ROI 98.34% (V15-audit-4、 5/17 反映)、 CI [66.33%, 138.05%] 100% 含む = 統計的有意 勝ち なし ★ | cumulative_results.csv 実測 n=596 (旧 119.2% / 93.23% は何れも drift) |
| W2 | ★ 京都 ROI 20.0% (N=58) ★ | session_5_16 |
| W3 | 中京 ROI 57.9% (N=60) | 同上 |
| W4 | 中山 ROI 78.7% (N=125) | 同上 |
| W5 | 戦略⑦除外 R で機会損失 (重賞 / 06_平場 / 京都 / 条件 E) | strategy_7_planB |
| W6 | 動画 features 0% coverage (V21 未稼働) | phase_a_poc_result |
| W7 | calibrator v1 21 sample over-fit (v2 で発覚) | calibrator_v2_summary |
| W8 | ★ V22 / V20+ 8 回全 fail (V15 plateau) ★ | recent commits |
| W9 | push 不能 (114MB CSV blocking) | session_5_16 |
| W10 | LINE 通知 未対応 | CLAUDE.md 未解決 |
| W11 | 三連単 / WIDE / 馬単 未拡張 | CLAUDE.md |
| W12 | リアルタイム odds 5 分前 snapshot のみ | CLAUDE.md |
| W13 | マークカード / 自動投票 未連携 | inferred |
| W14 | 重賞専門 model なし | CLAUDE.md |
| W15 | NAR 統合 未完 (V20 で予定) | Phase 3 roadmap |

### 8-3. Opportunities (8 件)

| # | 機会 |
|---|------|
| O1 | V21 動画 features 完成 (5/31+ で 30 features 真値化、 V21 stacking est. +0.005 AUC) |
| O2 | 重賞専門 model 投入 (戦略⑦除外 R 復帰、 月 2-4 GI 取り込み) |
| O3 | calibrator v2 paper eval 30 R 後採用 (over-confidence 解消) |
| O4 | JV-Link production fetch unlock 完了 (5/15) — bug 復旧 path |
| O5 | 完全自動化 plan (12/1 100% 目標) |
| O6 | LLM 統合 (GPT-4o + race description で「なぜ」説明性向上) |
| O7 | NAR 統合 (V20 で 投票 R 数 2x 候補) |
| O8 | strategy_layer_v2 (calibrator + 京都/中京 除外で +5-7pt 想定) |

### 8-4. Threats (6 件)

| # | 脅威 |
|---|------|
| T1 | JRA-VAN / netkeiba 規約変更 risk (data source 停止) |
| T2 | 大規模 cloud AI (ファミ天等) の進化 (odds 押し下げ) |
| T3 | 競馬離れ / 市場縮小 (流動性低下) |
| T4 | AI 規制強化 (GenAI 政策、 LLM 制限) |
| T5 | ★ V15 plateau / V22 8 fail / saturation 仮説 ★ (中期 stagnation) |
| T6 | 動画解析規約 (RV / netkeiba 動画利用許諾 グレー) |

---

## 9. SWOT-based 戦略 4 マトリクス

### 9-1. S-O (強み × 機会、 攻撃戦略)

| # | 戦略 | 関連 |
|---|------|------|
| SO1 | 自動運用 (S5+S10) + V21 動画完成 (O1) → 業界 frontier maintain | S5 × O1 |
| SO2 | HONEST report (S7) + LLM 統合 (O6) → 説明性で差別化 | S7 × O6 |
| SO3 | リークフリー (S3) + 重賞 model (O2) → 戦略⑦除外 R 復帰 | S3 × O2 |
| SO4 | v15_master 4-model Grid (S4) を v15_full で production 化 (FT+IR 有効化、 +0.018 AUC 想定) + V21 stacking (O1) → meta-learning で plateau 突破 | S4 × O1 |
| SO5 | JV-Link unlock (S12) + production fetch (O4) → bug 復旧 path | S12 × O4 |

### 9-2. W-O (弱み × 機会、 改善戦略)

| # | 戦略 | 関連 |
|---|------|------|
| WO1 | 京都 ROI 20% (W2) + strategy_layer_v2 (O8) → 再除外で +5pt | W2 × O8 |
| WO2 | 中京 ROI 57.9% (W3) + strategy_layer_v2 (O8) → 除外検討で +2pt | W3 × O8 |
| WO3 | 動画 0% coverage (W6) + V21 完成 (O1) → 5/31+ 真値化 | W6 × O1 |
| WO4 | calibrator v1 over-fit (W7) + v2 paper eval (O3) → 30R 後採用 | W7 × O3 |
| WO5 | 重賞 model なし (W14) + 重賞専門 model (O2) → 戦略⑦除外 R 復帰 | W14 × O2 |
| WO6 | NAR 未完 (W15) + NAR 統合 (O7) → V20 で R 数 2x | W15 × O7 |

### 9-3. S-T (強み × 脅威、 防衛戦略)

| # | 戦略 | 関連 |
|---|------|------|
| ST1 | リークフリー (S3) + plateau (T5) → 既存 model 厳守、 risky 改善禁止 | S3 × T5 |
| ST2 | HONEST report (S7) + AI 規制 (T4) → コンプライアンス 強化 | S7 × T4 |
| ST3 | 4 source 統合 (S8) + 規約変更 (T1) → 冗長性で 1 source 停止耐性 | S8 × T1 |
| ST4 | 撤退ライン (S9) + 市場縮小 (T3) → 損切り規律で退化局面対応 | S9 × T3 |

### 9-4. W-T (弱み × 脅威、 撤退/縮小戦略)

| # | 戦略 | 関連 |
|---|------|------|
| WT1 | 動画 0% (W6) + 動画規約 (T6) → 早期 PoC + 規約整理並行 | W6 × T6 |
| WT2 | V22 8 fail (W8) + plateau (T5) → 大規模 architecture 変更を中期計画化 | W8 × T5 |
| WT3 | push 不能 (W9) + AI 規制 (T4) → reproducibility risk、 LFS migration 早期 | W9 × T4 |
| WT4 | リアルタイム odds 限定 (W12) + ML 化 (T2) → 直前 odds 取り込み改善 (5 分前 → 1 分前) | W12 × T2 |

---

## 11. ★ 30 日 priority (動画撤回 + TYB 反映、 5/16 改訂) ★

### P0 (5/17 G1 day 守備 + 5/17 21:00+ 監査 / 真値確定)
| # | task | 工数 | 期待効果 |
|---|------|------|---------|
| P0-1 | ROI 乖離真値確定 (read-only formal analysis) | 1-2h | 全戦略判断の前提整う |
| P0-2 | 京都/中京 戦略⑦再除外 (data 判断) | 1h | est. +5-7pt ROI |
| P0-3 ★NEW★ | TYB calibrator leak 監査 | 1-2h | production 投入可否判定 |

### P1 (5/18-5/24、 paper eval)
| # | task | 工数 | 期待効果 |
|---|------|------|---------|
| P1-0 ★NEW、 最優先★ | TYB calibrator paper shadow eval (P0-3 PASS が条件) | 5/18+ | 30R 蓄積後採用判定 |
| P1-1 | calibrator v1 paper eval (継続) | 5/18+ | 30R 蓄積 |
| P1-2 | JRDB tyb/cha feature engineering | 1 週 | est. +0.005 AUC |
| P1-3 | netkeiba マスターコース評価 | 1 週 | est. +1-2pt ROI |

### P2 (5/25-5/31)
| # | task | 工数 | 期待効果 |
|---|------|------|---------|
| P2-1 | v15.2 再学習 (TYB features 含む、 P0-3 PASS 前提) | 1-2 週 | est. +0.005-0.01 AUC |
| P2-2 | 市場依存度低減 | 1 週 | est. +1pt ROI |
| P2-3 | EV > 1 フィルタ + 案 B 改 strict 強化 | 1 週 | est. +2-3pt ROI |
| P2-4 ★NEW★ | TYB daily fetch schtask 登録 + monitor | 30 分 | 5/9-5/15 停止再発防止 |

### P3 (6 月以降)
| # | task | 工数 | 期待効果 |
|---|------|------|---------|
| P3-1 | 重賞専門 model 開発 (戦略⑦除外 R 復帰) | 1-2 週 | est. +10pt ROI (重賞復帰) |
| P3-2 | NAR 統合 (V20、 投票 R 数 2x) | 6 月内 | est. +月数千円 |
| P3-3 | backtest vs live ROI 乖離 formal analysis | 1 週 | DS 透明性確保 |

### P4 (インフラ、 随時)
| # | task | 工数 | 期待効果 |
|---|------|------|---------|
| P4-1 | LFS migration (push 復旧) | 1 日 | reproducibility 改善 |
| P4-2 | 自動 model drift 検出 + 月次 alert | 2 週 | 5/16 京都 ROI 20% 発見 lag 再発防止 |
| P4-3 | outcome dashboard (Sub-task 2 で着手) | 完了 | 各 task Before/After 可視化 |

---

## ★ 動画系 (Phase A) 永久放棄 (5/16 evening 追記) ★

### 根拠

| Source | 規約 | 状況 |
|--------|------|------|
| YouTube | 利用規約「ダウンロード禁止」明文 + AI学習禁止明示 | NG |
| JRA レーシングビュアー | 私的使用範囲外不可、 ストリーミングのみ | NG |
| netkeiba SP 動画 | 規約 + IP ban 経験あり | NG |
| JRA-VAN NEXT 動画 | レーシングビュアーと同条件 | NG |
| JRA アプリ | 私的利用範囲明文 | NG |

### V21 動画 AI 永久放棄

- パドック CNN (12 features)
- パトロール YOLO (8 features)
- 調教 keypoint (10 features)

全 30 動画 features 構想は ★ 永久放棄 ★。 既存 file (tools/v21/paddock_*, patrol_*, chokyou_*) は archive 化判断は後日。

### 代替 path

★ 動画なし 数値 source 最大化 ★ で plateau 突破を目指す:
- JRDB 26 datatypes 完全活用 (現状 TYB 0% 結合 bug 等)
- netkeiba SP テキスト (厩舎コメント / 追切コメント) 完全活用
- 戦略 layer 改善 (戦略⑦ / calibrator / EV 動的閾値)
- 真値確定 (ROI 乖離 解消)

---

## ★ JRDB TYB breakthrough (5/16 evening) ★

### 実装 (commit b4948d6a)

| metric | V15 only | V15 + TYB | delta |
|--------|---------:|----------:|------:|
| 5CV AUC | 0.4653 | 0.6082 | +0.1429 ★ |
| n_samples | 348 | 348 | — |
| pos_rate | 0.587 | 0.587 | — |

### 真の signal (LR coef、 standardized)

- padock_idx +0.44 (★ パドック指数 真の signal ★)
- tansho_odds -0.58 (単勝オッズ高 = top3 入らず)
- weight_diff -0.23 (馬体重 増減)
- top1_score +0.20 (V15 は補完的)

### ★ 過大評価 risk 認識 (commit d3b78683) ★

- baseline AUC 0.4653 = task 設計問題 (top1_score は race 内 normalize 済)
- +0.1429 改善幅は baseline の異常さ起因の可能性大
- train 0.696 vs CV 0.608、 gap 0.088 = over-fit 兆候
- n=348 は production 判断 不十分
- leak 監査 (P0-3) 未完了 — tansho_odds が -15 min snapshot か race 確定か audit 必須

### production 投入条件

★ P0-3 leak 監査 PASS + paper shadow eval 30R 統計的有意 (Welch's t-test p<0.05) ★ 両方満たすこと。

---

## 11. ★ critical 注意事項 ★

### 11-1. ROI 真値統一 (5/17 V15-audit-4 で確定)

- 旧 CLAUDE.md / MEMORY drift: 累計 +13,530 円、 ROI 119.2% (戦略⑦込 140%+)
- 5/16 P0-1 真値: ROI 101.33%、 PnL +¥5,240、 n=563 (≤2026-05-16)
- **★ 5/17 V15-audit-4 真値 (5/17 G1 day 反映後) ★: ROI 98.34%、 PnL ¥-6,920、 n=596**
- bootstrap 95% CI [66.33%, 138.05%] 100% 含む → ★ 統計的有意 勝ち なし ★
- 5/17 G1 day 単日 ROI 47.36% / PnL ¥-12,160 / hit 6/33 (18.18%) が baseline を押し下げ

**推定原因 (旧 drift)**:
- 戦略⑦ + 案 B 改 仮想適用後の集計と推定 (実際は betting 規律で実 bet は subset)
- 「+13,530 円」 は別系統 計算源 (manual 集計?) と思われる
- 検証 必要: backtest 428% vs live 98% の formal analysis (戦略 / 期間 / オッズ取得タイミング の差)

★ 5/17 audit 後の公式 baseline = ROI 98.34% / PnL ¥-6,920 / n=596 ★

### 11-2. V15 plateau 仮説

- V20、 V22 (base / distill / enhanced)、 V18/V19 (sib 抜き) **計 8 試行全 fail** (V15 の stored .pkl.auc 0.8939 = LGB train-set self-eval 越え未達、 真の genuine WF baseline は 0.8678 — V15-audit-2)
- 単純な features 追加 / re-architecture で 改善不可と確証
- ★ plateau 突破の唯一 path = ★ V21 動画 features (業界未踏 frontier) ★
- 5/31+ で coverage 1,000+ R 達成 + 6/1-6/30 paper trade で 真の効果検証

### 11-3. push 不能

- `data/v20_training_data_full.csv` 114MB が commit 8dfb595f に存在
- GitHub 100MB hard limit 抵触
- destructive op (filter-repo / lfs migrate / force push) は user 絶対 NG
- ★ local commit のみ継続中 ★、 6 月中に LFS migration 戦略 user と議論

### 11-4. 5/17 G1 day (ヴィクトリアマイル) 絶対遵守

- V15 + 戦略⑦ + 案 B 改 strict 単独本番
- shadow eval / v2 calibrator は schtasks 未登録、 Discord 通知 0、 production 影響 0
- 累計 PnL ¥-6,920 (5/17 V15-audit-4) → 撤退ライン -¥50,000 まで余裕 ¥43,080
- 投票上限 ¥2,100/日、 手動投票継続

---

## 13. ★ 当 system 立ち位置 (動画なし frontier path、 5/16 改訂) ★

### 当 system の strength (動画なし 競合比)

- ★ **JRDB TYB breakthrough (5/16 evening)** ★ — padock_idx / tansho_odds / weight_diff を V15 + TYB stacking で +0.143 AUC 改善 path 発見 (★ honest: paper eval + leak 監査 PASS 後 採用 ★)
- ★ **LGB+XGB 2-model production (V15)、 booster 145 features** ★ — 個人運用としては高度 (v15_master の 4-model Grid は WF 評価専用、 production .pkl は LGB+XGB のみ — V15-audit-1)
- ★ **JRDB 26 datatypes + netkeiba プレミアム + TFJV 統合** ★ — data source 業界平均以上
- 戦略⑦ で損失 source 除外 → 公開 ATHENA 単勝回収率 80% / VUMA ワイド 40% と比較で 競争力

### 当 system の weakness (動画なし 競合比)

- IPAT 自動投票連携 未実装 → 運用効率で劣る
- LINE 通知なし → 個人運用なら問題なし
- 重賞 G1 専門 model なし → 戦略⑦除外で機会損失
- 三連単 / WIDE 拡張なし → 券種多様性で劣る
- ROI 真値 = 98.34% (V15-audit-4、 5/17、 n=596)、 CI [66.33%, 138.05%] 100% 含む = 統計的有意 勝ち なし (旧 119.2% は drift、 P0-1 → V15-audit-4 で 真値確定)

### ★ 動画なし path での目標 ★

| 目標 | 期日 | 達成基準 |
|------|------|---------|
| ★ 動画なし限界点 突破 ★ | 6/30 | WF AUC 0.9020+ / 実 ROI 110%+ / 累計 +¥50K |
| 重賞 model + 戦略⑦除外 R 復帰 | 6/30 | 月 2-4 GI 取り込み + ROI 維持 |
| TYB production 投入 | 6/1+ | P0-3 PASS + paper eval 30R 統計的有意 |

### 「業界最強」 か?

**NO** (honest)。 動画ありの商用サービスは 進化中。 但し ★ 動画なし path の個人運用 frontier ★ では:
- JRDB 完全活用 + netkeiba SP テキスト + 戦略 layer breakthrough で
- **WF AUC 0.91+ / ROI 110%+** を 6/30 までに 達成可能性

★ 動画なし frontier への一歩 = JRDB TYB breakthrough (5/16) ★

---

## 13. ファイル構成 mini-map

```
C:\Users\takum\keiba-ai\
├── CLAUDE.md                              # Claude Code session guidance (last updated 2026-05-09)
├── docs/SYSTEM_MASTER_2026_05_16.md       # 本 doc (Claude 包括 master)
├── docs/MEMORY_INDEX.md                   # docs/ 全索引
├── docs/FULL_AUTOMATION_ROADMAP.md        # 5/15 80% / 9/2 90% / 12/1 100%
├── app.py                                 # Streamlit UI
│
├── keiba_model_v15_central.pkl.gz         # V15 本番モデル (★ 不変厳守 ★)
├── keiba_model_v15_central_live.pkl.gz    # V15 live (当日オッズ込み)
├── data/calibrator_v15_pilot.pkl          # calibrator v1 (orig 21 sample)
├── data/calibrator_v15_pilot_v2.pkl       # calibrator v2 (NEW、 315 sample)
│
├── tools/
│   ├── predict_core.py                    # V15 core inference (★ 不変厳守 ★)
│   ├── daily_predict.py                   # 朝 06:30 自動 (★ 不変厳守 ★)
│   ├── race_auto_notify.py                # 5 分前 + Discord (★ 不変厳守 ★)
│   ├── daily_results.py                   # 結果回収 (★ 不変厳守 ★)
│   ├── strategy_layer_v2.py               # 買い目 + calibrator v1/v2 (5/16 NEW)
│   ├── save_all_horse_scores.py           # V15 全馬 score 記録
│   └── v21/
│       ├── predict_core_v21.py            # V21 meta-stacking skeleton
│       ├── paddock_features_merger.py     # 12 features 抽出
│       ├── patrol_yolo_data_prep.py       # 8 features skeleton
│       ├── train_v21_paddock_poc.py       # V21 PoC trainer
│       └── calibrator_v15_retrain.py      # 5/16 NEW
│
├── data/
│   ├── jra_races_full.csv                 # 781K (2015-2026)
│   ├── training_times.csv                 # 955K
│   ├── odds_history.csv                   # 778K
│   ├── cumulative_results.csv             # 596 settled (実 ROI 98.34% / PnL ¥-6,920、 5/17 V15-audit-4 真値、 ★ formation drift: race-time formation 永久喪失、 trio_bets_str は AM 8:00 morning prediction のみ — data-audit-3 ★)
│   ├── jrdb/ × 17                         # JRDB datatypes (548K each)
│   ├── daily_predictions/                 # {ymd}.csv × 30 days
│   ├── daily_results/                     # {ymd}.csv × 30 days
│   ├── live_features/                     # all_horse_scores
│   ├── netkeiba_*                         # 15+ csv (speed_index / training / siblings 等)
│   ├── v20/
│   │   ├── v20_training_data_full.csv     # 114MB (★ push blocking ★)
│   │   └── v20_lgb_xgb_models.pkl.gz      # PoC ensemble
│   ├── v21/
│   │   ├── phase_a_poc_result.md          # 動画 0% coverage
│   │   ├── phase_d_v21_architecture_design.md
│   │   ├── calibrator_v2_summary.md       # 5/16 NEW
│   │   ├── calibrator_v1_v2_shadow_compare_20260516.md  # 5/16 NEW
│   │   ├── session_5_16_evening_summary.md  # 5/16 NEW
│   │   └── inventory_5_16/
│   │       ├── A_features_full.md         # 本 master の source #1
│   │       ├── B_market_research.md       # 本 master の source #2
│   │       └── C_persona_swot.md          # 本 master の source #3
│   └── [100+ csv/json/log]
│
└── models/
    ├── keiba_model_v15_central.pkl.gz, keiba_model_v15_central_live.pkl.gz   # ★ V15 production = LGB+XGB 2-model (mlp=None, FT/IR 未保存) ★、 V15-audit-1、 不変厳守
    └── v21/                               # V21 meta-model 格納予定
```

---

## 14. ★ 投資保護 (絶対遵守) ★

- ★ V15 production 完全不変 ★ — `predict_core.py / daily_predict.py / race_auto_notify.py / app.py / .pkl.gz` 全部 1 byte も touch しない
- schtasks 既存 不変
- destructive git op 永久 NG (filter-repo / push --force / lfs migrate)
- fabrication 0 厳守 (HONEST report 文化)
- 撤退ライン -50,000 円 (現累計 ¥-6,920、 撤退余裕 ¥43,080、 5/17 V15-audit-4 真値) ※ 旧 +13,530 / +63,530 / +5,240 / +55,240 は drift
- 取り返し禁止 (損切り後 翌日へ持ち越さない)
- 投票上限 ¥2,100/日 (案 B 改 strict)

---

## 15. honest stop note

- 本 master doc は 3 並行 agent (A_features_full / B_market_research / C_persona_swot) + 親統合
- 全数値は出典付き、 「期待」「推定」 値は明示、 「業界最高」 「最強」 表現 排除
- ★ critical 発見 ★: ROI 真値統一 (旧 CLAUDE.md 119.2% / +13,530 は drift、 5/17 V15-audit-4 真値 98.34% / PnL ¥-6,920 / n=596) — docs/V15_AUDIT_4_CUMULATIVE_ROI_5_17_2026.md
- ★ V15 plateau ★: V22 / V20+ 8 試行全 fail を honest 受容、 V21 動画 (frontier) が唯一 path
- ★ 30 日 priority 3 件 ★: 京都/中京 再除外 + calibrator v2 採用 + V21 動画 5/31+

**この doc 1 つで Claude が system 全体を完全把握できる構造**。 詳細は data/v21/inventory_5_16/ + CLAUDE.md + MEMORY.md。
