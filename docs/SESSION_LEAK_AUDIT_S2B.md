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

- s2b は **AUC を犠牲(0.829 < V15 0.842)にして全券種で ROI を獲得**。三連複t4 は高分散(v1で 95%CI 179-212%、V15 は 146-166% で CI 非重複だったが、t4 は当たり外れが大きく実運用の安定性は別問題)。

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
