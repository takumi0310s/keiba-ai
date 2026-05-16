# 5 ペルソナ 評価 + SWOT 完全版

date: 2026-05-16
著者: AI 自律 review (claude-opus-4-7)
基準 data: CLAUDE.md / MEMORY.md / data/v21/*.md / data/cumulative_results.csv (実測)

★ honest report ★ — 「業界最高」「最強」 等の 誇大表現 排除、 全 数値は出典付き。

---

## 0. 評価前提 (★ 実測 数値 ★)

| 項目 | 値 | 出典 |
|------|---:|------|
| V15 WF AUC | 0.8939 | CLAUDE.md |
| v13.5b WF AUC | 0.8788 | CLAUDE.md (backtest 値) |
| v13.5b backtest ROI | 428.4% | CLAUDE.md (JRA 公式 配当、 2023-2025、 10,314 R) |
| **実運用 累計 ROI (戦略⑦前)** | **93.23%** | data/cumulative_results.csv (529 settled) |
| **実運用 累計 profit** | **-25,070 円** ※ | data/cumulative_results.csv |
| 戦略⑦ +pt (累計 baseline 比) | +3.67pt → 96.90% | session_5_16_evening_summary |
| CLAUDE.md 記載 累計 +13,530 円 | 注: 戦略⑦込み別系統 集計 / strategy_7_planB 適用後の 仮想値 と 推定。 cumulative_results.csv の生集計とは乖離 | MEMORY |
| 京都 ROI | 20.0% (N=58) | session_5_16_evening_summary |
| 阪神 ROI | 140.3% (N=72) | session_5_16_evening_summary |
| 東京 ROI | 120.2% (N=126) | session_5_16_evening_summary |
| 中山 ROI | 78.7% (N=125) | session_5_16_evening_summary |
| 中京 ROI | 57.9% (N=60) | session_5_16_evening_summary |
| 動画 features coverage | 0% (paddock 12 PoC のみ実装、 production 未稼働) | phase_a_poc_result |
| calibrator v2 sample | 21 → 315 (15x) | calibrator_v2_summary |
| schtasks 数 | 6+ (DailyPredict/Results/PremiumScrape/Watchdog/NightlySanity/RaceAutoNotify) | CLAUDE.md |
| V22 / V20+ 改善 試行 | 8 回全 fail (V15 越え 未達) | recent commits |

★ assumption: 「119.2% / +13,530円」は MEMORY 記載値だが、 cumulative_results.csv の 生 ROI 93.23% とは 乖離。 おそらく 戦略⑦ + 案 B 改 + 重賞除外 等を 仮想適用 した集計、 または別 source。 ペルソナ評価 では **両方併記**。

---

## 1. ペルソナ 1: 初心者 ユーザー (馬券初購入、 月 1 万円)

### 強み (この ペルソナにとって)
- Discord 通知が 「買い目そのまま」 形式 (買い目 7 点 / フォーメーション) → コピペで購入可能
- 戦略⑦ で 弱い R (06_平場 / 京都 / 条件 E / B) を自動除外 → 「全部買えばよい」 訳ではない 学習不要
- monte_carlo 検証 で 3 万円以上で 破産率 0% (CLAUDE.md) → 心理的安全感
- 700 円/R 固定 → 投入額の予測可能性 高

### 弱み (この ペルソナにとって)
- ★ 「なぜ この馬?」 説明が Discord 通知に 薄い ★ (指数 / 調教 / 厩舎 のみ、 ストーリー的説明 なし)
- 京都 ROI 20% / 中京 57.9% を ユーザー側で 認識する手段 限定 (週次レポート まで気づけない)
- 月 1 万円 では 戦略⑦ 適用後 でも 月 30R × 700 = 21,000 円必要 → ★ 予算 オーバー ★
- UI/UX が Streamlit 開発者向け (TRACK RECORD 等、 専門用語 多)
- 馬連 / 三連複 の違い 説明なし

### 改善要望 (assumption)
- 「初心者 mode」 — 月 5 R 厳選通知 (top 信頼度 のみ)
- 買い目に 「なぜ?」 1 行説明 追加 (LLM 統合 候補)
- 月予算 入力 → 自動で R 数調整

### overall: ★★★☆☆ (3/5) — 通知 形式は OK、 予算 整合性 + 説明性 改善余地

---

## 2. ペルソナ 2: 中級者 ユーザー (年 10-12 万円投資)

### 強み (この ペルソナにとって)
- 自動運用 schtasks 6 件 → 朝予測 / 5 分前通知 / 結果照合 全 自動 (hand-off OK)
- 透明性: 全買い目 cumulative_results.csv に 記録 → 後追い検証 可能
- 戦略⑦ + 案 B 改 で 個別 R 単位 micro-management 不要
- WF AUC 0.8939 / v13.5b backtest ROI 428% など 数値根拠あり

### 弱み (この ペルソナにとって)
- ★ 実運用 累計 ROI 93.23% (cumulative 生集計) と CLAUDE.md 記載 119.2% に **乖離** ★ → 中級者は気づく可能性
- 戦略⑦ 除外 R で 「機会損失感」 (06_平場 / 京都 / 重賞 等で V15 の score を見ても 購入できない 構造)
- カスタマイズ性 低: 投票額 700 円 hardcoded、 条件 7 種 hardcoded
- 京都 ROI 20% の 致命的 問題が 5/16 まで 発見されなかった → 監視機構 不足
- 「なぜこの買い目?」 への 突っ込んだ 説明 (feature importance) は app.py で見るしかない

### 改善要望 (assumption)
- 投票額 / 条件 enable/disable を Streamlit UI で 変更可能化
- 月次 / 週次 ROI summary を Discord に 自動 push (現状 週次 only)
- feature importance を 個別 R 単位で表示

### overall: ★★★★☆ (4/5) — 自動運用 完成度 高、 ただし 透明性 (ROI 集計 乖離) + 監視機構 改善余地

---

## 3. ペルソナ 3: プロ ユーザー (年 100 万円+、 統計 + AI 知識あり)

### 強み (この ペルソナにとって)
- リークフリー設計 (Pattern A、 LEAK_FEATURES 8 件 + V20 SKB 10 件 = 18 件 明示) → 監査可能
- HONEST report 文化 (CLAUDE.md の 「過去の失敗から学んだ教訓」、 SKB POST-RACE LEAK 検出経緯) → 信頼性
- WF 6-fold 時系列分割、 expanding window 厳守 → DS 観点で 妥当
- JRDB + netkeiba プレミアム + TFJV + JV-Link 4 source 統合 → データ独占性 中

### 弱み (この ペルソナにとって)
- ★ スケーラビリティ 致命的 ★ — 700 円/R × 月 30R = 21,000 円。 年 100 万円規模 では 投票額 50 倍必要 → odds 押し下げで ROI 崩壊
- ★ 戦略⑦ 除外 R (重賞含む) で 機会損失 — 年 GI 24 R 全 skip ★
- 全 R 対応 model なし (V15 は 全 R 学習だが、 運用 strategy は 戦略⑦ で絞り込み)
- 三連単 / 馬単 / WIDE 未拡張 → 高 EV ticket type 取りこぼし
- リアルタイム odds は 5 分前 snapshot のみ、 直前変動 取り込めず
- 重賞専門 model なし → プロは 重賞中心 投資 が多い、 致命的 mismatch
- ★ V22 / V20+ 8 回全 fail = V15 plateau 仮説 ★ → さらなる scaling 困難 (技術的天井)

### 改善要望 (assumption)
- 重賞専門 model 追加学習 (n=数千 R で 別 fold)
- 三連単 / 馬単 拡張 (高 EV 取り込み)
- 投票額 dynamic 化 (Kelly criterion / risk-adjusted sizing)
- 全 R production 化 (戦略⑦ は別 layer で 適用、 全 R 予測自体は出力)

### overall: ★★★☆☆ (3/5) — 監査性 高、 ただし scaling + 重賞 対応で プロ要件 未達

---

## 4. ペルソナ 4: データサイエンティスト

### 強み (この ペルソナにとって)
- ★ リークフリー設計 が 経験則 + 失敗教訓 で 強化 ★ — odds_log / horse_weight / SKB / dam_top3r / sib_top3_rate hybrid (Session #38) 等、 5 件 以上の 大規模 リーク発見・除去 documented
- ★ HONEST report 文化 ★ — V22 8 回失敗、 V20 expanding NO-GO、 V15.1 SKB LEAK を 隠蔽せず 記述
- 4-model ensemble (LGB+XGB+FT+IR) で IR (IntraRace Attention) が 35% 貢献 — レース内 相対性 を 真に学習 (zero-sum 問題への対応)
- expanding window 厳守 (cumsum - current) で 静的 CSV リーク を 防止
- WF 6-fold (2020-2025) + 年別 gap 監視 (>0.05 で 過学習 判定、 v12.1 で 実際に 不採用)
- 4 source data 統合 (JRDB / netkeiba / TFJV / JV-Link) で 多重独立検証 可能

### 弱み (この ペルソナにとって)
- ★ V22 / V20+ 8 試行 全 fail → V15 plateau (AUC 0.8939) ★ — saturation 仮説、 architecture breakthrough 不在
- features 数 124 → 150 (V15) → さらなる feature engineering で AUC delta ~0 → diminishing returns
- backtest ROI 428.4% vs 実運用 93.23% の **大幅乖離** → market impact / selection bias / live degradation の 説明不足
- v13.5b の Grid Search 重み が 年ごと最適化 → ★ test set leakage の可能性 ★ (年毎 grid は WF spirit に違反する 場合あり)
- code 品質: predict_core.py に FutureWarning 15+ (CLAUDE.md 既知)、 cumulative_results.csv に top1_num/score 95% 欠損 (既知 bug)
- calibrator v1 が 21 sample で over-fit (orig Brier 0.19 → v2 0.24 で 真値発覚) → ★ 統計的厳密性 軽視 ★ の前例
- push 不能 (114MB CSV blocking) → version control 不完全、 reproducibility リスク

### 改善要望 (assumption)
- Grid Search 重みを **CV** で固定 (年別 grid 廃止)
- backtest vs live ROI 乖離 の formal analysis (selection bias / odds shift quantify)
- model card / data sheet 作成 (Mitchell 2019 / Gebru 2018)
- features の causal graph / DAG documentation
- LFS migration で push 復旧

### overall: ★★★★☆ (4/5) — リークフリー + HONEST report は ★★★★★、 ただし backtest 乖離 + plateau で −1

---

## 5. ペルソナ 5: 完全自動運用 志望 ユーザー (月 5 万円、 5 年 持続 priority)

### 強み (この ペルソナにとって)
- ★ 6+ schtasks 自動運用 — DailyPredict / Results / PremiumScrape / Watchdog / NightlySanity / RaceAutoNotify ★
- process_watchdog v2 (ログ鮮度 ベース) + Keiba-NightlySanity (23:00 自動検証)
- SCRAPER-GUARD (金 22 時〜月 6 時) で データ取得 規約遵守
- 4/19 事故 (Sun 03:00 誤停止、 機会損失 +2,745 円) → 11 commits で 完全修正、 検証 17 task ALL PASS
- 月額 約 6,500 円 (netkeiba 4,500 + JRDB 2,000) — 月 1-3 万 ROI で 十分回収
- 完全自動化 plan (5/15 80% → 9/2 90% → 12/1 100%) 策定済 (FULL_AUTOMATION_ROADMAP.md)

### 弱み (この ペルソナにとって)
- ★ マークカード / 自動投票 未連携 ★ — Discord 通知見て 手動購入必要 (0-touch 不可)
- ★ Cookie 期限切れで netkeiba 取得停止 risk ★ — refresh_cookie.py で 手動 (--auto オプションあり、 ただし credential 保管要)
- データ source 規約変更 risk (JRA-VAN / netkeiba) は ユーザー側で 対応不能
- V15 plateau (V22 8 fail) → 5 年後 同 model 運用は ROI 退化 (市場 学習) 可能性
- 5/16 京都 ROI 20% の発見 が 「手動 audit」 で行われた → 自動監視 不在
- jrdb_paci.csv 4/4 更新停止 / jra_payouts.csv 4/6 更新停止 etc 既知 bug — 自動復旧機構 限定的

### 改善要望 (assumption)
- IPAT API 連携 / 自動投票 (法令確認要)
- course / 条件 / 月別 ROI 自動 alert (閾値 < 80% で Discord 通知)
- データ source 多重化 (JRA-VAN 規約変更時の fallback)
- model 自動再学習 (月次 / 季節別、 model drift 検出)

### overall: ★★★★☆ (4/5) — schtasks + watchdog は強い、 ただし自動投票 + 自動 model 更新で △

---

## 6. SWOT 完全版

### Strengths (12 件、 evidence 付き)

| # | 強み | evidence |
|---|------|------|
| S1 | WF AUC 0.8939 (V15) | CLAUDE.md |
| S2 | v13.5b backtest ROI 428.4% (JRA 公式 配当) | CLAUDE.md |
| S3 | リークフリー設計 — LEAK_FEATURES 18 件明示 | CLAUDE.md 8 章 |
| S4 | 4-model ensemble (LGB+XGB+FT+IR、 IR 35% 貢献) | CLAUDE.md |
| S5 | 6+ schtasks 自動運用 | CLAUDE.md 定期タスク |
| S6 | Discord リアルタイム通知 3 channel | CLAUDE.md |
| S7 | HONEST report 文化 (V22 8 fail / SKB LEAK / dam_top3r 等 fabrication 防止) | CLAUDE.md 失敗教訓 |
| S8 | 4 source data 統合 (JRDB + netkeiba + TFJV + JV-Link) | MEMORY subscriptions |
| S9 | 撤退ライン -50,000 円 / monte_carlo 破産 0% (3 万円+) | CLAUDE.md |
| S10 | process_watchdog v2 (ログ鮮度 ベース) + NightlySanity | CLAUDE.md 4/19 事故対応 |
| S11 | V21 動画 features architecture (純粋追加 layer 設計、 V15 不変保証) | phase_d_v21_architecture_design |
| S12 | JV-Link COM unlock (5/15) | MEMORY |

### Weaknesses (15 件、 evidence 付き)

| # | 弱み | evidence |
|---|------|------|
| W1 | 実運用 累計 ROI 93.23% (cumulative 529 settled) vs CLAUDE.md 119.2% の **乖離** | data/cumulative_results.csv 実測 |
| W2 | 京都 ROI 20.0% (N=58、 致命的) | session_5_16_evening_summary |
| W3 | 中京 ROI 57.9% (N=60) | 同上 |
| W4 | 中山 ROI 78.7% (N=125) | 同上 |
| W5 | 戦略⑦除外 R (重賞 / 06_平場 / 京都 / 条件 E) で 機会損失 | strategy_7_planB |
| W6 | 動画 features 0% coverage (paddock PoC 89 entries のみ) | phase_a_poc_result |
| W7 | calibrator v1 21 sample over-fit (v2 315 で発覚) | calibrator_v2_summary |
| W8 | V22 / V20+ 改善 試行 8 回全 fail (V15 plateau) | recent commits |
| W9 | push 不能 (114MB CSV blocking) | session_5_16 |
| W10 | LINE 通知 未対応 (Discord のみ) | CLAUDE.md 未解決 |
| W11 | 三連単 / WIDE / 馬単 未拡張 | CLAUDE.md |
| W12 | リアルタイム odds 5 分前 snapshot のみ | CLAUDE.md |
| W13 | マークカード / 自動投票 未連携 | inferred (no IPAT mod) |
| W14 | 重賞専門 model なし | CLAUDE.md |
| W15 | NAR 統合 未完 (V20 で予定) | Phase 3 roadmap |

### Opportunities (8 件)

| # | 機会 |
|---|------|
| O1 | V21 動画 features 完成 (5/31+ で 30 features 真値化、 V21 stacking +0.005 AUC 想定) |
| O2 | 重賞専門 model 投入 (戦略⑦除外 R 復帰、 GI 月 2-4 R 取り込み) |
| O3 | calibrator v2 paper eval 30 R 後採用 (over-confidence 解消で over-bet 防止) |
| O4 | JV-Link production fetch unlock 完了 (5/15) — jrdb_paci / jra_payouts 代替 path |
| O5 | 完全自動化 plan (12/1 100% 目標、 ロードマップ 策定済) |
| O6 | LLM 統合 (GPT-4o + race description で「なぜ」説明性 向上、 初心者 ペルソナ 改善) |
| O7 | NAR 統合 (V20 で着手、 投票 R 数 2x 候補) |
| O8 | strategy_layer_v2 (calibrator + 京都/中京 除外で +5pt 想定) |

### Threats (6 件)

| # | 脅威 |
|---|------|
| T1 | JRA-VAN / netkeiba 規約変更 risk (data source 停止 リスク) |
| T2 | 大規模 cloud AI (ファミ天等) の進化 (競合 ML 化、 odds 押し下げ) |
| T3 | 競馬離れ / 市場縮小 (流動性 低下で 戦略⑦ R 数減) |
| T4 | AI 規制強化 (GenAI 政策、 LLM 統合 制限可能性) |
| T5 | ★ V15 plateau / V22 8 fail / saturation 仮説 ★ (model 進化 停滞 = 数年後 ROI 退化) |
| T6 | 動画解析 規約 (RV / netkeiba 動画 利用許諾 グレーゾーン) |

---

## 7. SWOT-based 戦略 4 マトリクス

### S-O (強み × 機会、 攻撃戦略)

| 番号 | 戦略 | 関連 |
|------|------|------|
| SO1 | 自動運用 (S5+S10) + V21 動画完成 (O1) → 業界 frontier maintain | S5 × O1 |
| SO2 | HONEST report (S7) + LLM 統合 (O6) → 説明性で 差別化、 初心者 ペルソナ 評価 +1★ | S7 × O6 |
| SO3 | リークフリー (S3) + 重賞 model (O2) → 戦略⑦除外 R 復帰、 プロ ペルソナ 評価 +1★ | S3 × O2 |
| SO4 | 4-model ensemble (S4) + V21 stacking (O1) → meta-learning で plateau 突破 (assumption) | S4 × O1 |
| SO5 | JV-Link unlock (S12) + JV-Link prod (O4) → bug 復旧 path 確保 | S12 × O4 |

### W-O (弱み × 機会、 改善戦略)

| 番号 | 戦略 | 関連 |
|------|------|------|
| WO1 | 京都 ROI 20% (W2) + strategy_layer_v2 (O8) → 再除外で +5pt 想定 | W2 × O8 |
| WO2 | 中京 ROI 57.9% (W3) + strategy_layer_v2 (O8) → 除外検討で +2pt | W3 × O8 |
| WO3 | 動画 0% coverage (W6) + V21 完成 (O1) → 5/31+ 真値化で coverage 100% | W6 × O1 |
| WO4 | calibrator v1 over-fit (W7) + v2 paper eval (O3) → 30R 後採用で 健全化 | W7 × O3 |
| WO5 | 重賞 model なし (W14) + 重賞専門 model (O2) → 戦略⑦除外 R 復帰 | W14 × O2 |
| WO6 | NAR 未完 (W15) + NAR 統合 (O7) → V20 で R 数 2x | W15 × O7 |

### S-T (強み × 脅威、 防衛戦略)

| 番号 | 戦略 | 関連 |
|------|------|------|
| ST1 | リークフリー (S3) + V15 plateau (T5) → 既存 model 厳守、 risky 改善で AUC 下げない | S3 × T5 |
| ST2 | HONEST report (S7) + AI 規制 (T4) → コンプライアンス 強化、 第三者 audit 可能化 | S7 × T4 |
| ST3 | 4 source 統合 (S8) + データ source 規約変更 (T1) → 1 source 停止でも 3 残る 冗長性 | S8 × T1 |
| ST4 | 撤退ライン (S9) + 市場縮小 (T3) → 損切り規律で 退化局面 でも 損失制限 | S9 × T3 |

### W-T (弱み × 脅威、 撤退/縮小戦略)

| 番号 | 戦略 | 関連 |
|------|------|------|
| WT1 | 動画 0% (W6) + 規約変更 risk (T6) → 早期 PoC 達成必要、 規約整理 並行 | W6 × T6 |
| WT2 | V22 8 fail (W8) + plateau (T5) → 大規模 architecture 変更 (graph NN / transformer 系) を中期 計画化、 短期は 戦略 layer 改善優先 | W8 × T5 |
| WT3 | push 不能 (W9) + AI 規制 (T4) → reproducibility リスク、 LFS migration 早期実施 | W9 × T4 |
| WT4 | リアルタイム odds 限定 (W12) + 競合 ML 化 (T2) → 直前 odds 取り込み 改善 (5 分前 → 1 分前) | W12 × T2 |

---

## 8. ペルソナ別 ★ 最優先 改善 ★

| ペルソナ | 最優先改善 | 理由 |
|---------|----------|------|
| 1. 初心者 | ★ 月予算入力 → 自動 R 数調整 + 「なぜ」 1 行説明 (LLM) ★ | 月 1 万円 vs 必要 21,000 円の 予算 mismatch 致命的、 SO2 で 解決 |
| 2. 中級者 | ★ ROI 集計の 統一 (cumulative 93.23% vs CLAUDE 119.2% 乖離 解消) + 月次 Discord ROI summary ★ | 透明性 = 中級者の信頼の源、 W1 解消 |
| 3. プロ | ★ 重賞専門 model + 三連単拡張 + 投票額 Kelly criterion ★ | scaling + 戦略⑦除外 R 復帰、 W5+W11+W14 解消、 SO3 |
| 4. データサイエンティスト | ★ Grid Search 重み CV 固定 + backtest vs live ROI 乖離 formal analysis ★ | v13.5b の test leak 疑い解消、 W1 への 科学的説明、 STat 厳密性 確保 |
| 5. 完全自動運用 志望 | ★ 自動 model drift 検出 + 月次 alert (course/条件別 ROI < 80% で Discord) ★ | 京都 ROI 20% を 自動発見できなかった (5/16 まで 1 ヶ月超 放置) の再発防止、 WO1+WO2 |

---

## 9. overall 評価 + 結論

### overall ★ 平均: ★★★☆☆ (3.6/5)

| ペルソナ | 評価 |
|---------|-----:|
| 初心者 | 3/5 |
| 中級者 | 4/5 |
| プロ | 3/5 |
| データサイエンティスト | 4/5 |
| 完全自動運用 志望 | 4/5 |
| **平均** | **3.6/5** |

### 強みの本質 3 件
1. ★ リークフリー + HONEST report 文化 ★ (S3 + S7) — 競合 と 差別化 する 最大の moat
2. 自動運用 6+ schtasks + watchdog (S5 + S10) — 0-touch 運用 の 基盤
3. 4-model ensemble + IR 35% 貢献 (S4) — レース内 相対性 を 捕捉する 設計

### 弱みの本質 3 件
1. ★ V15 plateau (V22 8 fail) ★ (W8) — 中期 5 年 持続性 への 最大 threat
2. ★ 京都 ROI 20% / backtest vs live 乖離 (428% vs 93%) ★ (W1 + W2) — 数値 信頼性 への 直撃
3. 戦略⑦ 除外 R の機会損失 + 重賞 model 不在 (W5 + W14) — プロ pathway 阻害

### 次の 30 日 (5/16-6/15) 最優先 アクション 3 件

| 優先度 | アクション | 期待 効果 |
|-------|-----------|---------|
| 1 | 京都 / 中京 を 戦略⑦ で 再除外 (WO1+WO2) | +5-7pt ROI |
| 2 | calibrator v2 paper eval 30 R → 採用 (WO4) | over-bet 解消、 ROI 安定化 |
| 3 | V21 動画 features 5/31+ production 化 (WO3 + O1) | plateau 突破の唯一 path |

### 中期 (6 月以降) 計画

- 重賞専門 model 投入 検討 (SO3)
- backtest vs live ROI 乖離 formal analysis (DS ペルソナ要件)
- LFS migration (W9 解消)
- 自動 model drift alert (WO5、 5/16 京都 発見 ラグ 再発防止)

---

★ honest report 終了 ★

注 1: 「業界最高」「最強」 表現 排除済。
注 2: 数値 全 出典付き、 想定 / assumption は明示。
注 3: 119.2% vs 93.23% の 乖離 は CLAUDE.md と 実測値 の 差異 を そのまま記述 (推定 解釈 並記)。

