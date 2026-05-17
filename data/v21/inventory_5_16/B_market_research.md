# 競馬予想 AI 市場 包括調査 + 当 system gap 分析

**作成日**: 2026-05-16
**調査方法**: WebSearch (Anthropic web search) + WebFetch
**当 system 基準 (5/17 V15-audit 真値)**: V15 本番 (★ **LGB+XGB 2-model**、 booster 145 features ★、 stored `.pkl.auc` 0.8939 = LGB train-set self-eval (in-sample LEAKY)、 **genuine WF 0.8678** (LGB+XGB) / Grid 4-model 5-fold 0.8858)、 累計 ROI **98.34%** / PnL **¥-6,920** / n=596 (5/17 V15-audit-4、 CI [66.33%, 138.05%] 100% 含む = 統計的有意 勝ち なし) ※ 旧 119.2% / 140%+ / 101.33% は drift、 5/16 P0-1 → 5/17 V15-audit-4 で 真値確定
**honest 原則**: 確認できない情報は "unknown" / "未確認"。 推測は "est." 表記。 「業界最強」 は使わない。

---

## 0. 調査対象 サービス 一覧

調査済 (11 サービス):
1. netkeiba AI (公式)
2. SPAIA 競馬
3. JRA-VAN ネクスト
4. JRAレーシングビュアー (動画)
5. TARGET frontier JV
6. 競馬ブック (KEIBABOOK)
7. ATHENA (アテナ)
8. VUMA (ヴーマ)
9. ROBOTIP スーパー (ウマニティ)
10. EquinEdge (US)
11. ChatGPT / LLM 系 horse racing predictor (海外)

未着手 (時間切れ / 個別ヒットなし):
- ファミ天 / 馬王Z / KEIBA-NN (具体 hit なし、 同名類似ソフトは情報量薄)
- 競馬らぼ AI (検索 hit なし)
- umaaa (UMAI app は確認、 umaaa.ai は予想ビルダー)
- TwinSpires / TVG / AmWager (米国、 言及のみ)

---

## 1. 競合 AI サービス 比較 (11 サービス)

| サービス | 提供機能 | AI 推定 | データソース | 月額 | 強み | 弱み |
|----------|----------|---------|--------------|------|------|------|
| **netkeiba AI** | 全 R AI予想オッズ + 出走想定オッズ + AI馬券診断 + コンパチ + 1000+ data 分析 | unknown (LGB/DL est.) | netkeiba 内部 + JRA 公式 | プレミアム ¥690/30日、 スーパープレミアム ¥14,900/年 (¥1,490/30日) | ユーザー基盤最大、 出馬表前から AI予想オッズ、 IPAT 連携、 票数 data 持ち | model 内部不透明、 全頭着順予想は公開せず |
| **SPAIA 競馬** | 18 種類 AI予想、 全 R 自動予想、 IPAT 連携、 収支管理、 追切診断 AI、 京大/東大 competition AI 込み | 複数 model アンサンブル (est.) | 過去レース + 騎手/調教師/馬場/天候 | ゴールド ¥500、 プラチナ ¥1,500 (税別) | 18 種類 AI を結果公開、 14 日無料、 大学研究系 | サンプル多すぎて選択困難、 IPAT 連携の質 unknown |
| **JRA-VAN ネクスト** | 30 年 data 閲覧 + AIデータマイニング (公式 AI 予測) + パドック映像 | 公式 data mining (詳細 unknown) | JRA 公式 30 年 | ¥880/月 | 公式 30 年 data 完全、 AI 予測 標準搭載、 信頼性最高 | PC 専用、 UI 古い、 自分でモデル組めない |
| **JRA レーシングビュアー** | パドック (15分前 配信) + 調教映像 + パトロールビデオ + マルチカメラビュー + 過去 G1 1984+ | 動画配信 (AI 解析なし) | JRA 公式映像 | ¥550/月 | 公式パドック / 調教 / パトロール **全部入り**、 4K 高画質 | AI 解析機能なし → ★ 自分で動画AI を載せる素材として最強 ★ |
| **TARGET frontier JV** | 30 年 data ローカル分析、 条件抽出、 血統/騎手/脚質/種牡馬 分析 | データ分析 ツール (AI なし) | JRA-VAN Data Lab | ソフト無料 + DataLab ¥2,090/月 | 完全カスタム分析、 個人 AI の基盤 | AI 機能なし、 学習コスト高 |
| **競馬ブック (Smart)** | 調教取材 + 厩舎取材 + パドック情報 + ギリギリ情報 + TM予想 + ブック指数 + コンピュータ予想 | コンピュータ予想 (詳細 unknown) | 自社取材 | Smart ¥600/月 (旧 ¥480、 2025/9 改定)、 Web は 5 プラン | 取材一次情報、 厩舎/調教関係者コメント | AI 弱め、 取材依存 |
| **ATHENA (アテナ)** | 全 R 全頭着順予想 + 各券種買い目 + リアルタイム更新、 完全無料 | 過去 3 万 R + 50 万馬 学習 (詳細 unknown) | netkeiba 系? unknown | **無料** (広告収益) | 全頭順位 公開、 登録不要、 直前まで更新 | 中央平地のみ、 障害 / 地方 なし、 internal 不透明 |
| **VUMA (ヴーマ)** | ワイド予想 AI、 全会場 全 R | unknown | unknown | 14 日無料、 月額 unknown | ワイド specialist、 的中率 40% (報告ベース) | ワイドのみ、 model 詳細 unknown |
| **ROBOTIP スーパー (ウマニティ)** | カスタム予想エンジン作成 (競走馬/騎手/血統/調教師/馬主/生産者 6 ファクター × 距離/トラック/馬場/周回/坂 5 適性)、 U指数 ベース | カスタム数値モデル | ウマニティ 独自 U指数 | スタンド ¥5,500/月 + 予想燃料費 (馬券購入額の 3%) | 自分で エンジン 組める | 高い、 ML ではなく ルールベース、 自動投票 連携あり |
| **EquinEdge (US)** | EE Win % (top win 32.9% 的中)、 Pace Figure (top pace 1-2 位 72.5%)、 Genetic Strength Rating、 Ticket Generator (exacta/trifecta/superfecta) | ML (詳細 unknown) | US racing data | $5.95/day, $24.95/weekend, $49.95/month weekend, $699.95/year unlimited | Ticket Generator が予算最適化、 GSR 系統 rating ユニーク | US 専用、 日本 競馬 非対応 |
| **ChatGPT / LLM 系** | プロンプト ベース 出走表 解釈、 picks 生成、 質疑応答 | GPT-4 / Claude 等 (汎用 LLM) | ユーザー が提供する data | ChatGPT Plus $20/月 等 | 自然言語 対話、 follow-up 可、 race commentary 解釈 | 統計精度 unknown、 hallucination リスク |

---

## 2. 当 system gap 分析 (★ 足りない機能 list ★)

### 当 system 現状機能 (再掲、 5/17 V15-audit 真値)
- ★ **LGB+XGB 2-model production** ★ (v15_master の 4-model Grid (LGB+XGB+FT+IR) は WF 評価専用、 production .pkl は LGB+XGB のみ — V15-audit-1)、 ★ **booster 145 features** ★ (Pattern B list 150 だが truncate)、 genuine WF AUC **0.8678** (LGB+XGB) / Grid 5-fold mean 0.8858 — V15-audit-2
- 6 条件分類、 三連複 7点 / 馬連 2点 自動生成
- 朝 daily_predict + 5 分前 race_auto_notify
- **Discord** 通知のみ
- 戦略⑦ (06_平場特別 / 京都 / 条件E/B 除外)、 案 B 改 strict
- データ: TARGET JV / JRDB / netkeiba premium / JRA 公式 / 気象庁
- V21 動画 (パドック/パトロール/調教) 開発中

### Gap table

| # | feature | 競合での提供 | 当 system | gap 深刻度 |
|---|---------|--------------|-----------|-----------|
| 1 | **IPAT 自動投票連携** | netkeiba, SPAIA, ROBOTIP, KSC, 各種 ツール | × | **高** (運用効率 直結) |
| 2 | **LINE 通知** | ファミ天系 / 多くの一般予想サイト (推定) | × Discord のみ | 中 (個人運用なら Discord で十分) |
| 3 | **AI 予想オッズ** (出走想定段階) | netkeiba (中央 + 地方) | × | 中 (オッズ予測 → 期待値計算 高度化) |
| 4 | **AI 馬券診断** (ユーザー履歴) | netkeiba | △ (cumulative_results は集計のみ) | 低 |
| 5 | **重賞 G1 専門 model** | DX指数アプリ、 競馬ブック TM予想 | × (戦略⑦ で 06_特別 除外しただけ) | **高** (重賞は ROI 高、 取り返し 可能性) |
| 6 | **三連単 / WIDE / 馬単 拡張** | EquinEdge ticket generator、 ROBOTIP、 多くのサイト | × (三連複 + 馬連 のみ) | 中 (三連単 hit率 risk、 WIDE は安牌) |
| 7 | **障害競走 model** | netkeiba (一部)、 競馬ブック | × archive 化 | 低 (頻度少) |
| 8 | **地方 (NAR) 統合** | SPAIA 地方、 netkeiba TCK、 楽天競馬 福来エマ、 ATHENA は中央のみ | × (V20 で着手予定) | 中 |
| 9 | **動画 AI 解析** (パドック/パトロール/調教) | ★ **業界 全体 未提供** ★ JRA レーシングビュアー が素材提供のみ | △ (V21 開発中) | 競合に差をつけられる ★ 最大の opportunity ★ |
| 10 | **gait analysis / pose estimation** (DLC SuperAnimal、 YOLOv8) | 学術研究 (DeepLabCut + MatLab) のみ、 商用 まだなし | △ (V21 で設計中) | 同上 ★ frontier ★ |
| 11 | **リアルタイム odds 連続追跡** (5 分前 snapshot のみ → 連続) | オッズ期待値アナライザー、 異常オッズシンドローム、 SPAIA | × snapshot 1 回 | 中 (異常 odds 検知で期待値 補正) |
| 12 | **強化学習 RL bet sizer** | 学術 / 個人 ブログ で言及、 商用 ほぼなし | △ (V22 RL 試行済、 効果薄 baseline 以下) | 低 (試行済 honest 不採用) |
| 13 | **LLM race commentary 解釈** | ChatGPT Horse Race Predictor 系 (GPT 汎用) | × | 中 ★ 未着手 frontier ★ |
| 14 | **取材一次情報** (厩舎コメント / 馬体評価) | 競馬ブック、 KEIBABOOK、 netkeiba プレミアム コメント | △ (netkeiba プレミアム コメントは取得済 scoring -3〜+3) | 低 |
| 15 | **馬主 / 牧場 系統 features** | ROBOTIP は 6 ファクター中 2 つ、 EquinEdge GSR | × | 低 |
| 16 | **pace 配分シミュレーション** | EquinEdge Pace Figure (top pace 72.5%)、 SPAIA | △ (pci feature あり、 配分なし) | 中 |
| 17 | **馬体重 当日 補正高度化** | 競馬ブック パドック情報 (取材) | △ (09:30 補正 / 朝予測 diff で アラート) | 低 |
| 18 | **payout 分布予測** (期待値 精度向上) | netkeiba AI 予想オッズ、 オッズ期待値アナライザー | × (固定 投資額 700円) | 中 |
| 19 | **モバイル UI** | 全 サービス | × Streamlit のみ | 低 |
| 20 | **遠征 / 海外調教馬 特殊扱い** | 競馬ブック 一次取材 | × | 低 |
| 21 | **マークシート 出力** | KSC自動投票 Plus | × | 低 (IPAT 連携で迂回可) |
| 22 | **複数 model A/B test 基盤** | SPAIA は 18 model 並走で実質 A/B | × (model 単一切替) | 低 |

---

## 3. 的中率/ROI 向上候補 (priority sort)

| # | feature | 期待 AUC delta | 期待 ROI delta | 工数 | risk | 採用優先度 |
|---|---------|---------------|----------------|------|------|-----------|
| 0 ★NEW★ | **JRDB TYB breakthrough** (commit b4948d6a + d3b78683) | +0.1429 (5CV, n=348、 baseline 異常で過大評価 risk) | est. +2-5pt (P0-3 PASS 後) | 完了 (実装)、 5/18+ paper | 中 (P0-3 leak 監査 必須) | ★★★★ (P0-3 PASS 条件) |
| 1 | ~~動画 features 完成 (V21)~~ ★ **永久放棄** ★ | — | — | — | 規約 NG (YT/RV/SP/NEXT/アプリ 全部) | — (撤回) |
| 2 | **重賞 G1 専門 model** | est. +0.003 (重賞のみ) | est. +10pt (重賞 復帰) | 1-2 週間 | 中 (data 少、 過学習 risk) | ★★★ |
| 3 | **strategy 8 Jackpot LIVE 投入** | n/a | est. +5-10pt (低投資 高配当) | 1 週間 (shadow → live) | 中 | ★★★ |
| 4 | **calibrator v2** | n/a (AUC 不変) | est. +2-3pt (期待値 精度) | 5/18+ 検証中 | 低 | ★★★ |
| 5 | **リアルタイム odds 連続追跡** + 異常 odds 検知 | n/a | est. +2-5pt (期待値 補正) | 2-3 週間 | 低 | ★★ |
| 6 | **三連単 model + WIDE** 拡張 | n/a | est. +/-? (hit率 risk) | 2-3 週間 | **高** (三連単 hit率 低、 試験的) | ★★ |
| 7 | **AI 予想オッズ** 自前推定 → 期待値 補正 | n/a | est. +3pt | 2 週間 | 中 | ★★ |
| 8 | **NAR 統合** (V20) | est. +0.002 (共通 features 効果) | est. +月数千円 (地方 day) | V20 内 | 中 | ★★ |
| 9 | **IPAT 自動投票連携** | n/a | est. 運用 効率化、 ROI 直結なし | 1 週間 (team-nave API) | 低 (JRA 公認外) | ★★ |
| 10 | **LLM race commentary 解釈** | est. ?? (未検証) | est. ?? | 1-2 週間 (PoC) | 中 (hallucination) | ★ (PoC 価値) |
| 11 | **pace 配分シミュレーション** (EquinEdge 風) | est. +0.001 | est. +1-2pt | 2 週間 | 低 | ★ |
| 12 | **payout 分布予測** (固定 700円 → 期待値 連動) | n/a | est. +3-5pt | 2 週間 | 中 | ★★ |
| 13 | **馬主 / 牧場 features** | est. +0.0005 | est. +0pt | 1 週間 | 低 | ★ |
| 14 | ~~gait analysis / pose estimation~~ ★ **永久放棄** ★ | — | — | — | 動画 source 規約 NG | — (撤回) |
| 15 | ~~動画 AI 解析 (paddock/patrol/chokyou)~~ ★ **永久放棄** ★ | — | — | — | 同上 | — (撤回) |
| 16 | **障害競走 model 再構築** | n/a | est. 機会少 | 2-3 週間 | 低 | ★ (不要) |
| 17 | **LINE 通知** | n/a | n/a (運用便利のみ) | 1-2 日 | 低 | ★ |

★ 動画系 3 件 (V21 paddock/patrol/chokyou + gait analysis + 動画 AI 解析) は ★ 永久放棄 ★ (YouTube / JRA RV / netkeiba SP / JRA-VAN NEXT / JRA アプリ 全 規約 NG)。 5/16 evening の master doc で確定。

---

## 4. 業界 frontier (★ honest 評価 ★)

### 4.1 動画 AI 解析 (★ 当 system が先行可能 ★)
- **学術**: DeepLabCut SuperAnimal で 26 keypoint markerless pose estimation、 388 動画 15 分で処理
- **学術**: 三日間 競技馬 gait 解析で duty factor / speed / forelimb swing range 検出 (Madbarn 2025)
- **商用 horse racing 専用 動画 AI**: ★ **発見できず** ★ → 当 system V21 が業界初の可能性
- 当 system: V21 で YOLOv8 + DLC + 30 features 設計済、 5/31+ 着手

### 4.2 LLM race description 解釈 (★ 未着手 frontier ★)
- ChatGPT で race description / 厩舎コメント / 戦評 を解釈する custom GPT 多数存在
- 信頼性 unknown (hallucination)、 統計 model と併用が現実的
- 当 system 未着手 → PoC 価値あり、 工数 1-2 週間

### 4.3 強化学習 RL bet sizer (★ 商用 ほぼなし、 当 system V22 で試行済 不採用 ★)
- 学術: state = funds + odds + win prob、 reward = payout で policy network 学習
- non-stationary data 困難、 初期 投資 大
- 当 system V22 RL: 3 alpha 全 baseline 以下、 8 試行目 honest 不採用 (CLAUDE.md より)

### 4.4 自動投票 + API 連携 (★ 既に成熟、 当 system は未連携 ★)
- team-nave JRA-IPAT API: 単勝/複勝/連系/三連単/三連複/ワイド/WIN5 全対応
- KSC 自動投票 Plus: JRA-VAN DataLab 込み、 オッズ連動 資金分配
- 個人開発 で Python + Selenium が一般的 (HTML 構造変更 risk)
- 当 system: 朝予測 + 5 分前 Discord 通知 → 手動 IPAT 投票

### 4.5 異常 odds / オッズ期待値分析 (★ 部分採用済 ★)
- 異常オッズシンドローム: G1 解析、 リアルタイム 得票率
- オッズ期待値アナライザー (iny-keiba.com): 勝率 入力で買い目 + 資金 自動分配
- 当 system: 5 分前 snapshot のみ、 連続追跡 未実装

### 4.6 federated learning / 個人 data 連携 (★ 完全未着手 ★)
- 学術 概念止まり、 競馬 業界 適用例 unknown

---

## 5. 当 system 推奨 next 5 actions (★ honest 推奨 ★)

### Action 1: **V21 動画 features 完成** (継続、 最優先)
- 業界 frontier、 商用 競合 なし
- 5/31+ 着手予定、 期待 AUC +0.005 / ROI +5pt
- リスク: data 量 / 学習 リソース。 fallback: zero-shot DLC SuperAnimal で機能 縮小版

### Action 2: **重賞 G1 専門 model 開発** (新規、 高優先)
- 戦略⑦ で 06_平場特別 除外したが、 重賞 (G1/G2/G3) は依然 ROI source
- 期待 ROI +10pt (重賞 復帰)、 工数 1-2 週間
- リスク: 重賞 data 少、 過学習。 cross-validation 厳格運用

### Action 3: **calibrator v2 + 期待値 連動投資額** (検証中、 完成優先)
- 現在 5/18+ 検証中
- 固定 700円 → 期待値 連動投資額 (Kelly criterion 等) で ROI +3-5pt
- リスク: 低 (検証 phase 完了 後 実装)

### Action 4: **strategy 8 Jackpot LIVE 投入** (Session #?? 検証済)
- 53.6% top3 / 21.7% top1 LIVE verified
- shadow GO 済、 5/16 以降 small bet で LIVE 開始
- 期待 +5-10pt (低投資 高配当)、 リスク 中

### Action 5: **IPAT 自動投票連携** (新規、 運用効率)
- ROI 直接効果なし だが運用効率 化 (手動投票 mistake 防止)
- team-nave API (有料) または Selenium 自前実装
- 工数 1 週間、 リスク 低 (JRA 公認外、 利用規約 確認必須)

### Action 6 以降 (補欠)
- LLM race commentary PoC (1-2 週、 frontier 探索)
- リアルタイム odds 連続追跡 (2-3 週、 期待値 補正)
- WIDE 拡張 (1-2 週、 安牌 hit率 向上)
- NAR 統合 (V20 内、 既定 スケジュール)

---

## 6. ★ honest 評価 ★

### 当 system の strength (競合比)
- ★ **動画 AI 解析 (V21)**: 業界 全体 未提供、 競合 不在 ★
- ★ **LGB+XGB 2-model production (V15)、 booster 145 features**: 個人運用としては 高度、 SPAIA 18 model と比肩 ★ (v15_master の 4-model Grid (FT+IR 込み) は WF 評価専用、 production .pkl は LGB+XGB のみ — V15-audit-1。 v15_full で FT+IR 有効化 +0.018 AUC が次の improvement path)
- ★ **netkeiba プレミアム + JRDB + TARGET JV + JRA-VAN 統合**: data source は業界平均 以上 ★
- 戦略⑦ で 損失 source 除外 → 旧記述「実 ROI 140% 想定」 は drift、 真値は 戦略⑦ applied 96.90% (n=466、 ≤5/10)。 競合 公開数値 (ATHENA 単勝回収率 80% / VUMA ワイド 40% 等) との比較 は 真値 baseline で再評価必要

### 当 system の weakness (競合比)
- **IPAT 自動投票連携 未実装** → 運用 効率 で劣る
- **LINE 通知 なし** (Discord のみ) → general user 向け なし、 個人運用なら問題なし
- **重賞 G1 専門 model なし** → 戦略⑦ で 除外したまま (機会損失)
- **三連単 / WIDE 拡張なし** → 券種 多様性 で劣る
- **UI**: Streamlit のみ、 モバイル UI 弱い (個人運用なら無視 可能)
- **ユーザー基盤 ゼロ** (個人運用) → 公開 サービス と比較不能

### 業界最強 か?
- **NO** (honest)。 netkeiba / SPAIA / 競馬ブック 等 商用 サービスは ユーザー 体験 / 取材一次情報 / IPAT 連携 で先行
- **ただし**: AI モデル の精度 / 動画 AI / data 統合 では ★ 個人運用 トップクラス ★ の可能性
- V21 動画 AI 完成 + IPAT 連携 + 重賞 model で「個人運用 業界 frontier」 を狙える

---

## 7. 未確認 / 追加調査 必要事項

- **ファミ天 / 馬王Z / KEIBA-NN**: 個別 hit なし、 別 名称 か古いソフト の可能性
- **競馬らぼ AI**: 検索 hit なし、 サービス自体 存在 確認必要
- **VUMA 月額**: 14 日無料 のみ確認、 正規 月額 unknown
- **DX指数アプリ**: 重賞 専門 と確認、 月額 unknown
- **海外 (UK/HK/AUS)**: 検索範囲外、 別 機会で 調査
- **TwinSpires / TVG**: 米国 ベッティング platform、 内部 AI 詳細 unknown

---

## Sources

- [netkeiba AI](https://race.netkeiba.com/AI/AI.html)
- [SPAIA 競馬 料金](https://spaia-keiba.com/billing-course)
- [JRA-VAN ネクスト](https://jra-van.jp/nx/)
- [JRA レーシングビュアー](https://jra-van.jp/rview/)
- [TARGET frontier JV 料金](https://jra-van.jp/target/ryokin.html)
- [競馬ブック Smart](https://s.keibabook.co.jp/)
- [ATHENA](https://keiba-ai.jp/)
- [VUMA](https://vuma.ai/)
- [ROBOTIP スーパー](https://umanity.jp/robotip_super/)
- [EquinEdge](https://equinedge.com/)
- [team-nave JRA-IPAT API](https://www.team-nave.com/system/jp/products/ipatapi/)
- [KSC 自動投票 Plus](https://jra-van.jp/dlb/sft/lib/kscjidou.html)
- [DeepLabCut horse gait paper (Madbarn 2025)](https://madbarn.com/research/ai-assisted-digital-video-analysis-reveals-changes-in-gait-among-three-day-event-horses-during-competition/)
- [HoofBeat AI Motion Analysis](https://hoofbeat.net/ai-motion-analysis/)
- [ChatGPT Horse Race Predictor](https://chatgpt.com/g/g-APqptK62c-horse-race-predictor)
- [SI: $9 to $3,044.51 with ChatGPT](https://www.si.com/onsi/horse-racing/news/how-chatgpt-helped-me-turn-9-into-3-044-51-betting-on-horse-racing)
- [umabi AI ランキング 2026/05](https://umabi.jp/ai-ranking/)
- [AI vs Tipsters (horse.bet)](https://horse.bet/ai-vs-tipsters-who-predicts-horse-races-better-top-ai-prediction-tools/)
- [TwinSpires AI 2025 Kentucky Derby](https://www.twinspires.com/edge/racing/kentucky-derby/experts-vs-ai-revisited-who-will-win-the-2025-kentucky-derby/)
- [ポケットモンタロウ: 中央競馬×AI 完全攻略](https://note.com/gaisenmontaro/n/nde9b4d0a03b2)
