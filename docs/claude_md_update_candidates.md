# CLAUDE.md 訂正候補リスト

**作成**: 2026-05-06 00:30 (Session #25)
**対象**: `C:\Users\takum\keiba-ai\CLAUDE.md` (1325 行)
**実施タイミング**: **Phase 3 移行 (5/24+) で正式更新**、本書は事前棚卸し

> ★★ 5/17 V15-audit final resolution (Sub-task D): 本 doc 中の **119.2% / 140.3% / 月利 2-3 万円 / 0.8939 / 4-model / 150 features** 等は全て drift。
> 真値 (n=596、 ≤2026-05-17、 V15-audit-4): **ROI 98.34% / PnL ¥-6,920 / 撤退余裕 ¥43,080** / CI [66.33%, 138.05%] 100% 含む = 統計的有意 勝ち なし。
> V15 真値 (V15-audit-1/2/3): architecture = LGB+XGB 2-model production、 booster 145 features、 genuine WF AUC 0.8678 (LGB+XGB) / 0.8858 (Grid 4-model 5-fold)、 stored .pkl.auc 0.8939 は LGB train-set self-eval (in-sample LEAKY)。
> 詳細: docs/V15_AUDIT_1〜5_2026_05_17.md / docs/MEMORY_DRIFT_FINAL_RESOLUTION_2026_05_17.md。 ★★

---

## 1. 緊急訂正候補 (誤情報、5/9 までに直したい)

| # | 行 | 現状 | 訂正後 |
|---|----|------|--------|
| 1 | header | "Last updated: 2026-04-19" | **"Last updated: 2026-05-06"** (実際は 4/27 v16/strategy⑦ 等が含まれる) |
| 2 | §1 概要 | "現行モデル: v13.5b (124 特徴量、4 モデル Grid Ensemble、WF AUC 0.8788)" | **"現行モデル: V15 (150 特徴量、AUC 0.8939、本番運用 ROI 119.2%、戦略⑦込み 140%+ 想定)"** |
| 3 | §1 概要 | "2 段階モデル: Pattern A (リークフリー評価用) + Pattern B (当日情報込み実運用)" | (維持、ただし V15 ベースに更新) |
| 4 | §6 テスト結果 | "v13.5b 実 ROI 428.4%" 等 | V15 実運用 ROI 119.2% (298R) + 戦略⑦込み 140%+ 想定に更新 |
| 5 | §11 ファイル構成 | v13.5b/v141/v134 の pkl 列挙 | V15 + V15.1 + V17 + V18/V19 + NAR v4 反映 + archive されたモデル除去 |
| 6 | 既知バグ | "jrdb_paci.csv が 4/4 から更新停止" | **5/3 09:42 で解消済** (jrdb_kyi.csv も同) |
| 7 | 既知バグ | "jra_payouts.csv が 4/6 で更新停止" | **5/4 朝 07:59 で 5/3 まで更新済** (実態は 4/6 stale ではない) |
| 8 | 既知バグ | "JRDB データの 2026 年分が未取得 (race_id 列なしの可能性)" | 5/2-5/3 まで raw 取得済、要確認 |

---

## 2. 削除候補セクション

### 2.1 v16 Development Status 二重重複 (1177-1240 行 + 1241-1325 行)

emoji 化けあり版となし版が両方残存。 **片方を完全削除**。
合計約 150 行削減可能。

### 2.2 v12/v13.5b ROI table

§6 「テスト結果一覧」内の v12 (LGB 単体) WF 2020-2025 ROI table と v13.5b (4-model grid) WF 2023-2025 ROI table は historical reference として残してもいいが、**現行は V15** であることを冒頭に明記してから残置 (or `CLAUDE_HISTORY.md` 切出し)。

### 2.3 古い特徴量探索結果

V12.1 で不採用とした prev_review_score / shinba_eval_score / dam_top3r の詳細 (§9 内) は 2026-03-29 の検証結果、現行 V15 では別の判定済 → archive 候補。

### 2.4 V12 学習パラメータ詳細

§4 内の "V12 LGB パラメータ" "V12 XGB パラメータ" "Optuna 結果 (100 試行、不採用)" は historical、V15 ではパラメータ違うため誤解の元 → 削除 or archive。

---

## 3. 追加候補内容

### 3.1 V15 セクション (新規、§4 の冒頭)

```markdown
### V15 (本番、現行) スペック
- ファイル: `keiba_model_v15_central_live.pkl.gz` (Pattern B、150 features)
        + `keiba_model_v15_central.pkl.gz` (Pattern A、リークフリー評価用)
- AUC: **0.8939** (Booster) / 0.8858 (4-model ensemble)
- 訓練データ: 527,280 行 (2015/01 - 2025/12)
- 本番運用 ROI: 119.2% (4/12-5/3、298R、未勝利除外)
- 戦略⑦ (06_特別 + 京都 + 条件 E + 条件 B 除外) 込み想定 ROI: 140.3%
- アンサンブル: LGB Booster (主) + 補助
```

### 3.2 V15.1 セクション (新規、Phase 3 候補)

```markdown
### V15.1 (試作、Phase 3 候補) スペック
- ファイル: `data/v15.1/v15_1_lgb.txt` (1.8MB、5/5 19:00 学習完了)
- 寄与: SKB (専門家印) 10 features 単独で AUC 0.8728 → 0.9427 (+0.0699)
- KKA 16 features は寄与 0% (race_id 変換失敗の疑い、要究明)
- SRB 8 features は +0.0013 (微小)
- リーク確認: PASS (SKB は pre-race 印)
- 本格採用: Phase 3 (5/24+)、4-model ensemble + WF + leak audit が前提
```

### 3.3 V17/V18/V19 セクション (新規)

```markdown
### V17 (morning ULTRA-CLEAN)
- ファイル: `data/v17/models/v17_morning_pipeline.txt` 等 6 LGB ファイル
- CRLF 復旧済 (commit 777cc08e)
- 5/9 では使わない (TYB 観測継続中、5/11 月に midday 戦略生死決定)

### V18 (単勝) / V19 (複勝)
- ファイル: `data/v18/models/v18_tansho_lgb.txt` + `_xgb.json`、`v19_fukusho_lgb.txt` + `_xgb.json`
- AUC: V18 0.8954 / V19 0.8787
- BT ROI: V18 295.1% / V19 149.3%
- 状態: distribution shift 27.7x、5/16 試行は条件 5 件未達で no-go 寄り
- race-level normalize で bet>0 化 ROI 復活、winner_top1 -13.3pt は別問題
```

### 3.4 NAR v4 セクション (新規)

```markdown
### NAR v4 (地方競馬、5/12 paper 開始)
- ファイル: `data/nar/models/keiba_model_nar_v4.pkl` (167 KB、archive→active 復活)
- AUC: 0.8145 (reported) / 0.8519 (OOS 2025)
- 学習データ: 4,821 races / 49,213 rows (NAR 2020-2024、nar_all_races.csv のみ実使用)
- 5/5 柏記念 (船橋 11R Jpn1) で 0.777 完全再現確認
- 5/12 paper 開始 → 5/16 試行 500 円/日 (paper 良好なら)
```

### 3.5 戦略⑦ セクション (新規)

```markdown
### 戦略⑦ (4/27 適用済)
- 実装: `tools/race_auto_notify.py` L171-187, L269-276
- フィルタ: 06_特別 / 京都 / 条件 E / 条件 B 除外
- 期待効果: ROI 119.2% → 140.3% (+21.1pt)
- シミュレーション: 298R → 242R, 損益 +28,240 円改善
- 京都は 5/11 以降、course_renovated 永久化効果で再評価
```

### 3.6 累計収支 + 撤退ライン (新規)

```markdown
## 累計収支 + 撤退ライン

| 日付 | 累計 | 備考 |
|------|------|------|
| 5/5 朝 | **+14,140 円** | 現状値 |
| 撤退ライン | -50,000 円 | 絶対 |
| 撤退余裕 | +64,140 円 | 5/5 朝時点 |

### 撤退判定基準
- 5/9 単日 ROI < 50% → 5/10 投資停止
- 5/9-5/10 累計 -10,000 円 → 翌週投資停止
- 累計 -50,000 円 → 完全撤退
```

### 3.7 Phase 2.5+ 完了 + Phase 3 移行基準 (新規)

```markdown
## Phase 2.5+ 完了 (2026-05-05)
- 51.5h 連続作業、35 commits、24 セッション
- 詳細: `docs/PHASE_2_5_PLUS_FINAL_RECAP_5_5.md`

### Phase 3 移行 6 条件 (5/24 判定)
1. JRA 案B改 ROI ≥ 100% (4/12-5/24 累計)
2. race-level normalize 本番統合済
3. NAR paper 12-14 race 蓄積
4. V18/V19 試行 sample 30+ bets
5. 累計 +10,000 円維持
6. 撤退ライン余裕 30,000+ 円
```

---

## 4. 構造リファクタ案

### 4.1 行数目標: 1325 → 400 行台

切り出し方:
- `CLAUDE_HISTORY.md` (新規): V8/V9/V12/V13.5b 詳細、V12 不採用特徴量、Optuna 結果、Phase 1-A 詳細
- `CLAUDE.md` (新): 現行 V15 + V15.1 + V17 + V18/V19 + NAR v4 + 戦略⑦ + Phase 2.5+ + Phase 3 + 撤退ライン + リーク厳禁ルール + コマンド集

### 4.2 セクション順序 (新)

```markdown
1. プロジェクト概要 (V15 中心)
2. 現行モデル (V15 / V15.1 / V17 / V18 / V19 / NAR v4)
3. 戦略⑦ + 累計収支 + 撤退ライン
4. Phase 2.5+ 完了 + Phase 3 移行基準
5. データ資産 (TARGET 退会済 + JRDB + netkeiba premium)
6. リーク厳禁ルール (8 features、過去の失敗教訓)
7. 自動化体制 (28 schtasks + watchdog v2 + Discord)
8. コマンド集
9. 実戦前チェックリスト
10. 重要ファイル一覧 (V15 ベースに更新)
```

過去詳細 (V8/V9/V12/V13.5b) は `CLAUDE_HISTORY.md` へ。

---

## 5. 実施計画

### Phase 3 着手前 (5/24 まで)

| 工数 | タスク |
|------|--------|
| 30min | header + § 1 概要 訂正 (緊急 8 件のうち 1-3) |
| 30min | 既知バグ訂正 (緊急 8 件のうち 6-8) |
| 1h | v16 二重重複削除 (削除候補 § 2.1) |
| 1h | V15/V15.1/V17/V18/V19/NAR v4 セクション追加 (追加候補 § 3.1-3.4) |
| 30min | 戦略⑦ + 累計 + Phase 3 セクション追加 (§ 3.5-3.7) |

### Phase 3 移行時 (5/25 - 6/8)

| 工数 | タスク |
|------|--------|
| 2h | CLAUDE_HISTORY.md 切り出し (V8/V9/V12/V13.5b 詳細) |
| 1h | 削除候補 § 2.2-2.4 整理 |
| 1h | 構造リファクタ (§ 4.2 新セクション順序) |
| 30min | 行数確認 + 整合性チェック |

合計 6h で完全更新可能。

---

## 6. 結論

CLAUDE.md は 1325 行で V13.5b 中心の記述、**Phase 3 移行までに V15 中心に書換が必須**。 ただし 5/9 (土) 投資セッションは README + HANDOFF + UPDATE_INVENTORY + RECAP で完結するため、CLAUDE.md は読まなくて良い。

緊急訂正 8 件は 1h で対応可能、Phase 2.5+ 完了で時間ある時 (5/10-5/12 隙間) に着手推奨。 完全更新 (構造リファクタ含む) は Phase 3 着手後に。
