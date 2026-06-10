# リーク監査 → s2b → 3大規模タスク 結論集約 (2026-06-02〜04)

> **目的**: 2026-06-02〜03 の長い探索の結論を1つに集約。次に見返したとき「何が発見され、何が確定し、s2bは結局何で、何が宿題か」が一望できるように。
> **性質**: 検証・記録専用。**本番 V15/V16 は完全不変・投票未使用**。データ/モデル/cache(.pkl.gz)は .gitignore 対象でコミットしない。
> **姿勢**: ★ 美味しい数字は疑って訂正した経緯(リーク・反市場29%の幻・Optuna過最適化)も正直に残す ★

---

## 0. TL;DR(一望)

- **V15 は本物**。実運用 ROI 98〜146% は leak の影響を受けない(本番 live は元々 leak-free)。leak-free 真値 = 単勝 ~105%・AUC ~0.842。
- **backtest/cache に leak があった**(`jrdb_ze_*` 4特徴)。リーク版の数字(単勝156%・AUC0.8678・反市場29%)は **無効**。leak-free v2 cache で再評価して全結論を引き直した。
- **s2b は「穴特化」ではない**。実体は「配当効率寄りの**全体ランキング**モデル」。leak-free v2 で単勝 111%・三連複t4 207%(高分散)と V15 を上回るが、**反市場好走率 16.8% < base 22% = 穴を当てる力は無い**。当初の "反市場29%" はリーク版の幻だった。
- **市場は効率的**。見えやすい情報(脚質/馬具/枠/血統)で穴を当てるのはほぼ不可能、と蘇生特徴2回・過去モデル・Optuna の全敗で繰り返し確認。
- **本番昇格は時期尚早**。s2b の ROI 優位は前向き paper trading でしか確証できない(backtest はリーク混入の前科あり)。

---

## 1. 発見の核心(結論ベース)

### 1.1 騎手指数 `paci_jockey_exp` の正体 = 93%人気代理
- JRDB「騎手期待率」。市場人気(オッズ)との相関が極めて高く、**残差はわずか6〜7%**(`odds_dependency_analysis.json`)。
- **ルメール反証**: 騎手内で corr(JE値, 人気) ≈ −0.83。ルメール騎乗でも「その馬の JE がレース内最高」なのは 41% のみ → JE は騎手の腕でなく**人気を写しているだけ**。
- gain では騎手指数族が 137特徴中 **37.65%を支配** → 他の能力特徴を crowd out していた。
- 帰結: 「穴特化」を目指すなら人気代理族は**消す方が正当**(人気に織り込み済 = エッジにならない)。
- 詳細メモリ: `keiba-jockey-exp-anatomy`

### 1.2 `jrdb_ze_*` 4特徴のリーク(backtest/cache のみ)
- `tools/jrdb_features.py:788` が ZED(=過去走の**成績**=結果)を `blood_num` で **日付カットオフ無しの全期間平均** していた → 当該レース/未来の成績が特徴に混入。
- 該当4特徴: `jrdb_ze_idm_avg / jrdb_ze_ten_avg / jrdb_ze_agari_avg / jrdb_ze_furi_count`。
- 症状: override test で「市場本命を覆した馬が 44% 勝つ」「人気10〜18番手でも 38% 勝つ」= 明らかに**結果を見ていた**。`dam_top3r` / `SKB POST-RACE` と**同型**のリーク。
- **★重要: 本番 live は過去 ZED のみ参照で元々 leak-free。リークは backtest/cache 生成時のみ。実運用 ROI 98〜146% は本物★**。
- 除去: leak-free cache を生成(元 cache は不変、ze4特徴のみ「当該レース日付より前の ZED だけで expanding 平均」に再計算)。

### 1.3 V15 の真値(leak-free v2)
- 単勝 ROI **~105%**・WF AUC **~0.842**(2023-25 WF, 89.9%cov, N=10,350)。実運用 98% と整合。
- ★ リーク版の AUC 0.8678 / 単勝 156% は無効。CLAUDE.md 旧記載の「genuine WF 0.8678」も ze リークで嵩上げの疑い ★。

---

## 2. s2b の結論(正直に)

### 2.1 定義
- **V16能力137特徴 − 人気代理"族"13 + レース相対特徴**。
- 除去13 = `paci_jockey_exp_wr/_3rd` + 印4(`paci_jockey_mark/sogo_mark/train_mark/idm_mark`) + `jrdb_cid_idx/ls_idx/training_idx/stable_idx` + `paci_goal_rank/goal_diff/dochu_rank`。
- 追加 = 脚質one-hot/距離適性one-hot + レース相対(`n_nige/n_front/front_ratio/front_advantage/is_lone_nige/n_apt_match/inner_draw/front_x_inner/...`)。
- 実装: `tools/v16_anaba_s2_eval.py`。候補: `models/v16_anaba_s2b_candidate.pkl.gz`(検証専用・投票未使用)。

### 2.2 性能(leak-free v2, WF 2023-25, N=10,350)

| モデル | AUC | 単勝 | 三連複top4box | 馬連top3box | 反市場好走率 | spearman vs V15 |
|--------|-----|------|---------------|-------------|--------------|------------------|
| **V15** | 0.8418 | 105.0% | 154.4% | 126.6% | — | 1.000 |
| **s2b** | 0.8295 | **111.3%** | **207.3%** | **141.3%** | 16.8% | 0.928 |
| V24 | 0.8418 | 106.4% | 159.6% | 128.4% | 17.7% | 0.989 |
| V24b | 0.8417 | 105.9% | 153.8% | 127.5% | 17.1% | 0.989 |

- s2b は **AUC を犠牲(0.829 < V15 0.842)にして全券種で ROI を獲得**。

### 2.2b 統計的有意性 — ROI 95%CI(leak-free v2, race-level bootstrap n=5000)

| 券種 | s2b ROI [95%CI] | V15 ROI [95%CI] | 周辺CI | ペア差 s2b−V15 [95%CI] | 判定 |
|------|-----------------|-----------------|--------|------------------------|------|
| 単勝 | 111.3% [107.6, 115.1] | 105.0% [102.0, 108.1] | 僅かに重複 | **+6.2pt [+2.8, +9.8]** | 0跨がず=**有意**(小) |
| 三連複top4box | 207.3% [177.1, 251.7] | 154.4% [145.0, 164.1] | **非重複** | **+52.9pt [+22.9, +97.4]** | 0跨がず=**有意**(CI幅広) |

- ★ 周辺CIは単勝で僅かに重なるが、**ペア(同一レース)差のCITが0を跨がない** = ノイズを除いた s2b の優位は両券種で統計的に有意 ★。三連複は優位が大きいが CI幅が広い(高分散)。

### 2.2c 三連複top4box 高ROIの正体 — 年別分解(頑健性)

| 年 | s2b ROI | 的中率 | 最大配当(総払戻に占める比) | V15 ROI |
|----|---------|--------|-----------------------------|---------|
| 2023 | 227.1% | 28.3% | 728,220円(**23%**) | 152.6% |
| 2024 | 206.9% | 29.5% | 172,490円(6%) | 158.5% |
| 2025 | 188.0% | 29.1% | 106,740円(4%) | 152.2% |

- **全3年で V15 超・的中率 ~29% 安定 = 特定年の偶然ではない(頑健)**。
- ただし 2023 の 227% は**単一の 728,220円 配当が総払戻の 23%**を占めるファットテール依存。この1点を除いても 2023 は ~174%、2025 は大口依存なしで 188% → 高ROIは jackpot 頼みではない。

### 2.2d 資金管理 — モンテカルロ(20,000パス, horizon≈1年全レース, ★エッジ持続前提のi.i.d.bootstrap★)

| 戦略 | 年間stake | 平均最終損益 | 5%最悪 | 最大DD(中央/95%tile) | 破産確率(-5万到達) |
|------|-----------|--------------|--------|------------------------|---------------------|
| s2b 単勝 | ¥345,000 | +¥38,877 | +¥20,270 | ¥4,980 / ¥8,340 | **0.0%** |
| s2b 三連複t4 | ¥1,380,000 | +¥1,481,002 | +¥954,880 | ¥17,370 / ¥25,820 | **0.0%** |
| V15 単勝 | ¥345,000 | +¥17,232 | +¥1,790 | ¥6,210 / ¥11,190 | 0.0% |
| V15 三連複t4 | ¥1,380,000 | +¥750,659 | +¥562,577 | ¥17,850 / ¥27,020 | 0.0% |

- 撤退ライン(警戒-3万/一時停止-4万/撤退-5万)・余力 ¥43,080(現PnL -6,920→-50,000)に照合 → **最大DDの95%tile(s2b t4=¥25,820)でも撤退余力内・破産確率0.0%**。
- ★ 重大な前提: これは「backtest のエッジが本物で持続する」と仮定した i.i.d. bootstrap。本セッションの教訓どおり backtest エッジは過去にリークで幻だった前科がある。実運用 friction(オッズ変動・取得遅延)も未モデル化。**従ってこの安全性は前向き paper で未確証** ★。

### 2.3 s2b の正体
- **「穴特化」ではなく「配当効率寄りの全体ランキングモデル」**。
- ★ **反市場好走率 16.8% < base ~22%** = 市場本命を覆して当てる力は**無い**。ROI 優位は「穴を当てる」のではなく「全体ランキングの配当効率(人気薄も含めた拾い方)」由来 ★。
- 当初観測した **"反市場29%" はリーク版(ze が結果を見ていた)の幻**。leak-free では 16.8% に落ちた。

### 2.4 評価基準の教訓
- ★ **AUC でなく leak-free ROI で評価せよ** ★。s2b は AUC を下げて ROI を上げるモデル → 当時の **AUC基準 NO-GO 判定では穴特化(配当効率)の価値が見えなかった**。
- ただし「ROI で評価」も backtest リークに脆い(本件がまさにそれ)→ 最終確証は前向き paper のみ。

---

## 3. 効かなかったこと(再挑戦防止・正直に)

### 3.1 蘇生特徴 2回とも効かず(全 gain ≈ 0%)
- **① 脚質/距離適性 one-hot 化**(step1/s2): per-horse 単一コードでは死(gain≈0)。レース相対化(`front_advantage` 0.1→0.86%)で多少蘇生したが、**反市場好走率は改善せず**。
- **② 馬具変更/足元/バイアス×枠×脚質**(s3_revive): `bagu_change/is_bagu/ashimoto/front_inner_bias3/bagu_x_front` を追加。**全 gain < 0.02%**(下表)。反市場好走率はむしろ低下(16.8→16.3%)、ROI はほぼ不変(誤差内)。

| 蘇生特徴 | gain% |
|----------|-------|
| ashimoto | 0.017 |
| front_inner_bias3 | 0.018 |
| bagu_change | 0.012 |
| bagu_x_front | 0.005 |
| is_bagu | 0.003 |

- **理由 = 市場が織り込み済み**。馬具変更22%発生・足元・バイアスは「見えやすい」情報で、人気が既に反映している → エッジにならない。

### 3.2 過去モデル(V24/V24b)
- leak-free v2 でも **s2b 未満**(単勝・三連複とも)。AUC のみ V15 と僅差で上だが、ROI では s2b に及ばない。AUC優位 ≠ ROI優位、を再確認。

### 3.3 Optuna ハイパラ最適化(ROI目的)
- valid24 では複合ROI 138.3%→152.1% に見えたが、**held-out 2025 で頑健な改善なし**(三連複はむしろ悪化)。train複合ROI 398% = **過学習サイン**。最適 AUC(0.8306)も s2b(0.8295)とほぼ同じ。
- 結論: **s2b の現行 param は十分。Optuna 改善は本番で消える**(CLAUDE.md「Optuna過信」教訓の再現)。
- 注: ★ AUC で最適化すると人気をなぞる方向(AUC↑ROI↓)になるため ROI目的にしたが、それでも held-out で勝てなかった ★。

### 3.4 市場の効率性(繰り返し確認した結論)
- 見えやすい情報(脚質/馬具/枠/血統/騎手指数)で穴を当てるのは**ほぼ不可能**。これらは人気に織り込み済。
- 残る未実装エッジ候補は「人気が鈍い情報」のみ: ブリンカー(歴史 raw KYI 不在で実装不可)、A/Bコース替わり(データ無 + IP BAN リスク)。
- 詳細メモリ: `keiba-feature-inventory`

---

## 4. 運用面の成果と宿題

### 4.1 成果(今日まで)
- **per-race 完走サマリ + 夜間突合**: 取りこぼし検知。`tools/per_race_coverage_check.py`(JRDB KYI レース数 vs 通知ログを夜間照合 → 不足を UPDATES 警告)。race_auto_notify に完走サマリ「per-race通知 X/Y件完了」を追加。
- **s2b paper trading 仕組み**: `tools/paper_trade_s2b.py`。**V15 の特徴量ダンプを再利用**(`data/v15_feat_dump/{date}/*.parquet`)→ s2b は netkeiba/JRDB へ**新規アクセスせず IP BAN ゼロ**。通知は **DISCORD_WEBHOOK_TEST1 のみ**(本番 #買い目 BETS と完全分離)。
- **V15 DailyPredict に特徴量ダンプ追加**: try/except 完全保護・予測ロジック不変(承認済)。
- **スケジューラ登録**: paper predict/results + per-race 取りこぼし検知(RunLevel Limited・既存本番タスク不変)。

### 4.2 per-race console-kill 対策(2026-06-04)
- **真因 = 可視コンソール直起動**(「起動遅れ」は誤診)。5/30 は 8:45 定刻起動 → `^C`(コンソール制御イベント)で途中死亡 → 10:55 手動再起動で早い6R取りこぼし。スリープ/電源/トリガーは全て正常。
- **対策 = silent_runner 隠し窓化**: RaceAutoNotify_Sat/Sun の Action を `race_auto_notify.bat` 直起動 → `wscript.exe "tools\silent_runner.vbs" "race_auto_notify.bat"`(DailyPredict と同方式・console-kill 免疫)。
- 適用 script = `tools/apply_raceauto_silentrunner.ps1`、戻す = `tools/revert_raceauto_silentrunner.ps1`。
- ★ **管理者権限が必要(RunLevel=Highest)。管理者 PowerShell で適用 script を1行実行する運用。2026-06-04 時点 未適用(適用待ち・期限 土曜6/6)** ★。
- 保険: 朝 8:00 DailyPredict が全R買い目を #買い目(BETS)へ送信 → per-race 死亡でも買い目情報は届く。
- 詳細メモリ: `keiba-raceautonotify-console-kill`

### 4.3 宿題(急がない)
1. **silent_runner 化の管理者適用**(土曜6/6まで)。
2. **watchdog 安全再登録**: `process_watchdog_v2` の restart cap(per-day上限)実装 + path 確認後に kill-switch 削除。5/9 の再起動ループ(Discord spam)再発防止が前提。
3. **s2b paper 蓄積待ち**: 確証に必要なサンプル目安 = 単勝 ~300R / 三連複 ~1000R。前向きデータのみがリーク不可能な唯一の証拠。
4. **leak-free 100% は不可**: horse_name→blood_num 結合の実上限 89.9%(残9.6%は真の初出走で原理的に過去成績なし、100%不可能)。
5. (防御的・別件・要承認) 本番 `jrdb_features.py` の ze集計に日付フィルタ追加。live は既に安全なので緊急性なし。

---

## 5. 結論(一望)

- **V15 は本物**(実運用 98〜146%、live は leak-free)。
- **s2b は配当効率で V15 超だが穴力なし**(反市場 17% < base 22%)。本番昇格は **paper 確証待ち**。
- **市場は効率的**。見えやすい情報で穴は当てられない(蘇生2種・過去モデル・Optuna 全敗で確認)。
- **美味しい数字は疑って正解だった**: リーク(ze)・幻(反市場29%)・過最適化(Optuna)を全て訂正した。

### 5.1 s2b の信頼度 診断(統計的 + 実用的, leak-free v2)
- **統計的に有意か → YES(ペア検定)**: ROI 優位はペア差CIが0を跨がない(単勝 +6.2pt[+2.8,+9.8]、三連複t4 +52.9pt[+22.9,+97.4])。三連複は優位が大きいが CI 幅広(高分散)。
- **特定年の偶然か → NO(頑健)**: 三連複t4 は 2023/24/25 の全年で V15 超・的中率 ~29% 安定。2023 の jackpot 依存を除いても高水準。
- **ドローダウンは許容内か → YES(条件付き)**: MC で最大DD 95%tile ¥25,820 < 撤退余力 ¥43,080、破産確率 0.0%。**ただし backtest エッジ持続前提**。
- **総合判定**: leak-free v2 上では「有意・頑益・DD許容内」の3拍子。**しかし backtest エッジは過去にリークで幻だった前科がある** → 実運用昇格は前向き paper(リーク不可能な唯一の証拠)での確証が必須。**現時点では投票未使用のまま paper 観察継続が妥当(時期尚早)**。

---

## 6. 索引(再現用)

### 6.1 leak-free cache(.gitignore・ローカルのみ)
| ファイル | 内容 |
|----------|------|
| `data/_v15_optuna_df_cache_leakfree.pkl.gz` | v1(85.8%cov)・ze4特徴を当該日付前 expanding に再計算 |
| `data/_v15_optuna_df_cache_leakfree_v2.pkl.gz` | ★ v2(89.9%cov、NFKC正規化 + NaN-aware累積平均)。**評価は必ずこれを使う** ★ |

### 6.2 検証スクリプト(tools/)
| スクリプト | 役割 |
|------------|------|
| `v16_make_leakfree_cache.py` / `_v2.py` | leak-free cache 生成(v1 / v2=89.9%) |
| `v16_anaba_s2_eval.py` | s2b 定義(build_features / ODDS_REMOVE / PROXY_FAMILY / RAW_REPLACE / NEW) |
| `v16_leakfree_roi_grid.py` | 買い方別 ROI 関数群(S_tan/S_fuku1/S_trio4/S_umaren_t3box) + make_oof + LGB_P/XGB_P |
| `v16_leakfree_roi_ci.py` | ROI の bootstrap 95%CI |
| `v16_pastmodels_leakfree_v2.py` | V15/s2b/V24/V24b 横並び再評価(§2.2 表の出所) |
| `v16_anaba_s3_revive.py` | 蘇生特徴(馬具/足元/交互)検証(§3.1 表の出所) |
| `v16_anaba_s4_leakaudit.py` | リーク源の特定(override test / family ablation) |
| `v16_anaba_s4_optuna.py` | Optuna ROI最適化 + held-out 2025 検証(§3.3 の出所) |
| `v16_s2b_ci_montecarlo_v2.py` | ★ ROI 95%CI + モンテカルロ(DD/破産) + 三連複年別分解(§2.2b/c/d の出所) ★ |

### 6.3 候補モデル(.gitignore・検証専用・投票未使用)
| ファイル | 内容 |
|----------|------|
| `models/v16_anaba_s2b_candidate.pkl.gz` | ★ s2b 本命候補(LGB+XGB) ★ |
| `models/v16_anaba_s3_revive_candidate.pkl.gz` | s2b + 蘇生特徴(効かず) |
| `models/v16_anaba_c1/s1/s2_candidate.pkl.gz` | 途中段階(c1=sample-weight, s1=one-hot, s2=族除去前段) |

### 6.4 結果 JSON(data/・.gitignore)
| ファイル | 内容 |
|----------|------|
| `data/v16_pastmodels_leakfree_v2.json` | §2.2 横並び数値 |
| `data/v16_anaba_s3_revive.json` | §3.1 蘇生 gain・ROI |
| `data/v16_anaba_s4_optuna.json` | §3.3 Optuna best_params・held-out |
| `data/v16_anaba_s4_leakaudit.json` | リーク源特定の証跡 |
| `data/v16_s2b_ci_montecarlo_v2.json` | §2.2b/c/d ROI CI・モンテカルロ・年別分解 |

### 6.5 運用ツール(tools/)
| ファイル | 役割 |
|----------|------|
| `paper_trade_s2b.py` | s2b paper(V15ダンプ再利用・TEST1通知・IP BANゼロ) |
| `per_race_coverage_check.py` | 夜間 per-race 取りこぼし照合 |
| `race_day_check.py` | netkeiba 0件 + JRDB開催日 → リトライ+警告 |
| `apply_raceauto_silentrunner.ps1` / `revert_raceauto_silentrunner.ps1` | console-kill 対策の適用/復旧(要管理者) |

### 6.6 関連メモリ
- `keiba-v15-cache-backtest-leak` — ze リーク確定・leak-free 再生成・s2b 未証明
- `keiba-jockey-exp-anatomy` — 騎手指数=93%人気代理
- `keiba-feature-inventory` — 上級者要素の蘇生可能/追加/諦め分類
- `keiba-raceautonotify-console-kill` — per-race console-kill とその対策
- CLAUDE.md「🧪 leak-free 監査 + 評価基準」節 + リーク厳禁ルール表の `jrdb_ze_*` 行

---

## 7. Fable独立監査の採択 (2026-06-11、本番不変・netkeiba非アクセス)

> Fable 5 による初回独立監査(6/11)の所見を採択し、①補正後再CI ③paper定義整合 を実施。
> ②全特徴override総当たりは次セッション(重い)。★補正で数字が縮んでもそのまま記録(糊塗禁止)★

### 7.1 表現訂正(所見採択)
- ★「V15は本物」は言い過ぎ★ → 正確には「**予測はリークなし・市場平均超は有意**(leak-free 単勝 105.0% CI[102.0,108.1] > 控除後期待 ~80%)。**ただし実運用ROIのCI[66.3,138.05]は100%を含み、長期利益性は統計的に未証明**(12週再チェック継続)」。
- MC「破産確率0.0%」は**エッジ持続前提込み**の数字(i.i.d.bootstrap・friction未モデル化)。0%という表記自体が美味しい数字であることに注意。
- s2bの§2.2b有意性は **selection後推論**だった: 券種12×モデル4×レース選択グリッド総当たり後の最良セルにCIを当てており多重比較未補正、i.i.d. race bootstrapは同日相関(馬場バイアス日)を無視しCI過小の可能性 → **6/17判定は §7.2 の補正後CIを使う**。

### 7.2 ①三連複t4 207%・単勝111%の補正後再CI (`tools/fable_corrected_ci.py` → `data/fable_corrected_ci.json`)
- 整合検証: OOFをv2 cacheから決定論再生成し公表値(単勝111.3/105.0・t4 207.3/154.4)と**完全一致**を確認。N=10,350R/D=322日。
- jackpot = `20230109_中山_1_4_3` ¥728,220(2023総払戻の23%)。winsorize cap(両モデル的中配当プール99%tile)= t4 ¥17,227 / 単勝 ¥936。
- 補正後(日付クラスタbootstrap n=100k・Bonferroni m=102セル→99.951%CI):

| variant | s2b t4 | V15 t4 | t4ペア差 [95%CI] | [Bonf m=102] | 単勝ペア差 [95%CI] | [Bonf m=102] |
|---------|--------|--------|------------------|--------------|--------------------|--------------|
| raw | 207.3% | 154.4% | +52.9pt [+22.4,+97.7] | [+9.0,+144.3] 生存 | +6.2pt [+2.9,+9.7] | [+0.3,+12.7] 生存 |
| jackpot除外 | 189.8% | 154.4% | +35.3pt [+18.5,+54.2] | [+6.9,+70.0] 生存 | +6.2pt [+3.0,+9.7] | [+0.5,+12.5] 生存 |
| **winsorize** | **167.8%** | 147.7% | **+20.1pt [+10.8,+29.5]** | **[+3.2,+36.8] 生存** | +4.4pt [+1.4,+7.2] | **[-0.7,+9.6] ★0跨ぐ★** |

- m=204(v1再実行分も数えた感度)でも同じ: t4 [+3.2,+37.5] 生存 / 単勝 [-1.2,+9.6] 跨ぐ。
- ★結論★: **三連複t4のs2b優位は3段補正(winsorize+日付クラスタ+Bonferroni)後も有意に生存**。ただし規模は 207%→168〜190% に縮小。**単勝優位(+6.2pt)は最厳格条件で有意性消失** — 単勝を根拠にしない。
- **6/17基準の更新**: winsorize後ペア差を真値と仮定した paper必要N(片側α=0.05・検出力80%)= **三連複t4 ≈3,500R / 単勝 ≈7,600R**。6/17時点のpaper N(数百R)では確証不能 → **6/17は「方向確認のみ」(paper ROIがbacktest表と整合方向か・壊滅していないか)とし、昇格判定はN到達後**。旧目安「三連複~1000R」はペア差検出には不足(絶対ROI>100%検出の目安としてのみ有効)。

### 7.3 ワールズ型戦略 NO-GO (`tools/fable_worlds_audit.py` → `data/fable_worlds_audit.json`)
- 定量定義(閾値は2020-22分布から導出・look-ahead排除): 地力=`jrdb_ze_idm_avg`レース内上位25% × 近走=近3走複勝圏1回以上 × 人気=前日人気レース内下位50%。
- **好走率edgeは実在**: 同一前日人気順位の層別比較で複勝率 **+5.4pt CI[+4.0,+6.7]**(2023/24/25とも正で安定)。
- **だがROIにならない**: パリミュチュエルで最終オッズが織り込み、同人気帯でも的中時配当が25〜40%低い(例: 9番人気 W¥468 vs 非W¥689)→ 単勝/複勝ROIペア差はCIが0を跨ぐ・**絶対ROI 62〜84%でCI上限すら<100%**。
- 安田記念の「発見」= 実在する小さな好走率edgeの単発顕在化(生存者バイアス)。「見えやすい情報は織り込み済み」結論と整合(ze系地力もJRDB購読者に可視)。
- ze系はs2b特徴量に内包済 → **独立戦略化はしない(NO-GO)**。

### 7.4 初B(初ブリンカー)単独 見送り (2026-06-08集計)
- 2026年562頭: 初B複勝率 21.1% ≒ 全馬 21.8%。人気薄初B 7.5% ≒ ベース 7.7% = **穴にならず**。
- 「1-3人気×初B」+6.2pt(n=74で小)・「逃げ先行×初B」のみ低優先で保留。単独戦略化なし。

### 7.5 ③paper集計定義の backtest 整合点検 (paper_trade_s2b vs leak-free v2 backtest)

| # | 項目 | 判定 |
|---|------|------|
| 1 | 三連複top4box 組成(4C3=4点) | 一致 |
| 2 | 単勝top1 / 馬連top3box(3点) / 三連単form1-2-5 | 一致 |
| 3 | ベット額(100円/点) | 一致 |
| 4 | 見送り条件(5頭未満skip) | 一致 |
| 5 | 的中判定(frozenset/順序比較) | 一致 |
| 6 | 払戻ソース: backtest=jra_payouts.csv / paper=daily_results_full JSON(fallback同csv)。確定払戻同士 | 解釈注意(複勝・三連単は本番未取得時スキップ=券種で母数が異なる) |
| 7 | 対象母集団: backtest=v2 cache(89.9%cov) / paper=当日全レース | 解釈注意(paperが広い) |
| 8 | 特徴スナップショット: cache(前日確定相当) / 朝8:00ダンプ。s2bはオッズ系除去済で差小 | 解釈注意 |
| 9 | モデル: WF fold再学習 / candidate固定(≤2025学習) | 設計差(go-forwardとして正当) |
| 10 | ★不一致→修正★ KYI族24特徴がダンプでデフォルト化(idm=50/脚質=0) → s2bレース相対特徴(NEW)が全死 | **検証側修正済**: `_restore_collided_columns()` で `jrdb_*_x`(実値)から復元。**6/6-6/7の2日分pred(47R)は劣化特徴での記録 → 6/17評価では分離扱い(混ぜない)** |

### 7.6 JRDB 二重マージ (6/11発見 → 同日 daily_predict 修正済・★race_auto_notify 等は未修正・要承認★)
- **発見**: `predict_core.build_features` 内(predict_core.py:2008、3/31 381522a2 で追加)で JRDB マージ済みの df に **同じ merge_jrdb_predict_features を再適用** → pandas merge 衝突で実値が `jrdb_*_x`/`_y` に退避し、素列がデフォルト再充填(idm=50・脚質=0 等、KYI族24列)。
- **採点位置の確定 (6/11)**: daily_predict は build_features → merge#2 → `predict_race`(df直採点) の順 → **朝スコアは劣化dfで採点**。判別テスト(`tools/fable_dpfix_discriminate.py`)で実証: 6/6-7 ダンプの当日実スコアは **(a)劣化df再採点と一致 41/43R・(b)復元df一致 0R**。
- **★訂正(6/11)★: 「race_auto_notify は健全」は誤りだった**(初回調査の grep 表示25件制限による見落とし)。`race_auto_notify.py:344(build_features)→353(再マージ)` も**同型の二重マージ**(4/2 commit 8aa5c68a 起源)。→ ★**お金は劣化スコアに乗っていた = YES**(4/2以降、レース直前の実投票通知も KYI族デフォルトで採点)★。`predict_one_race.py:103` / `predict_one_race_v3.py:102` も同型。
- **影響量 (6/6-7 の43Rで劣化→復元の変化)**: top1変化 **3R** / top3変化 **14R** / 三連複formation変化 **23R**。朝仮想ROI(cumulative)は 〜4/3 76.5%(n=173) vs 4/4〜 94.8%(n=582) = **ROI上の劣化は識別不能**(ノイズに埋没。最近30日69.1%⚠の主因とも断定できない。糊塗せず記録)。
- **修正 (6/11・承認済・daily_predict のみ)**: `daily_predict.py` に `merge_jrdb_once()` ガード追加(jrdb_列が既にあれば再マージしない=「本来のmerge#1入力に戻す」のみ・予測ロジック不変)。**検証**: 6/6-7相当シミュレーションで ①ガードno-op 43/43 ②KYI族デフォルト率 **100%→15.3%**(残15.3%=真の欠損) ③スコア=predict_core直採点と完全一致 43/43。**回帰**: `tests/test_no_double_jrdb_merge.py` 追加(ガード動作+ソース検査+6/13以降ダンプの再発検知)全PASS。
- **★未修正(要承認・最優先=6/13土曜前)★**: `race_auto_notify.py:353`(**実投票経路**)と `predict_one_race(_v3).py` に同じ1行ガード適用が必要。本タスクの承認範囲は daily_predict のみのため未着手。
- **6/13実地確認の仕込み**: `tools/kyi_health_check.py`(ダンプの`_x`列有無+KYI族デフォルト率→ `data/paper_s2b/kyi_check_{date}.json`、NG時警告表示)。paper_trade_s2b predict(土曜朝スケジュール済)から自動実行。6/7データでNG_DEGRADED検知を確認済=検知器は機能する。

### 7.7 Fable監査TODO
- [x] ① 補正後再CI(jackpot/クラスタ/Bonferroni 3段) — 本セッション(§7.2)
- [x] ③ paper定義整合 + 検証側修正 — 本セッション(§7.5)
- [ ] ② 全特徴 override総当たり(zeを見つけた手法を残存145特徴の全族に適用) — 次セッション(重い)
- [ ] (要承認・本番) daily_predict 二重マージ修正(§7.6)
- [ ] (要承認・本番) jrdb_features.py ze集計への日付フィルタ(防御的・既存宿題)
