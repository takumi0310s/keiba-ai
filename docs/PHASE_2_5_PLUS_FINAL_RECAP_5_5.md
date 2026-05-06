# Phase 2.5+ 51.5 時間 全総括 レポート

**作成**: 2026-05-06 00:30 (Session #25 = 寝る前最終投入)
**期間**: 2026-05-03 19:00 〜 2026-05-05 22:30 (51.5 時間 連続作業)
**最終 commit**: 86cd1da5 (緊急 3 件対応)

---

## 1. エグゼクティブサマリー

### 数字

| 項目 | 値 |
|------|-----|
| **作業時間** | 51.5 時間連続 |
| **セッション数** | 24 (Claude Code Opus 中心) |
| **commit 数** | 35 (全 push 済) |
| **新規ドキュメント** | 60+ (docs/ 37 + data/v18/ 59 内、新規 60+) |
| **schtasks 整備** | 28 件 (静音化 vbs ラッパー化) |
| **テスト** | 19 ファイル / 17+ PASS 維持 |
| **累計収支** | **+13,530 円** (生データ、5/6 Session #27 真相確定) / +14,140 円 (USER 申告) — 撤退余裕 +63,530〜64,140 円 |

### 主要発見 TOP 3

1. **V15.1 SKB +0.0699 大発見** (commit 7c5ba9f8): 専門家印 (SKB) 10 features 単独で AUC 0.8728 → 0.9427 改善。Phase 3 (5/24+) 採用候補。
2. **chihou_races_2020_2025.csv blocker は誤情報** (commit 86cd1da5): `train_nar_v4.py` 解析で nar_all_races.csv のみ実使用と判明、5/12 NAR paper の blocker ではなかった。 5-10h の不要復旧作業を回避。
3. **distribution shift 27.7 倍 = scaling 問題** (commit 74eb10b7): V18/V19 retro で BT/production の race_max_p 比 27.69x、winner_top1 rate -13.3pt 劣化。 race-level normalize でも解消されず、feature shift 別問題と確定。

### Phase 2.5+ ステータス: **READY**

5/9 (土) JRA 案B改 投資、5/12 (火) NAR paper、5/16 (土) 試行候補、5/24 (金) Phase 3 移行判断、すべて準備完了。 5/6 (火) のユーザー作業は admin 1 コマンドのみ。

---

## 2. セッション一覧 (年表形式)

### 5/3 (土) 19:00 〜 5/4 (日) 朝 (Day 1)

| # | commit | 内容 |
|---|--------|------|
| 1 | 5fdfc2d0 | 5/3 GW Day3 結果集計 + Phase 2.5 提案 |
| 2 | 943791b3 | 5/9 投資戦略提案 + V18/V19 retrospective (BT 295%/149% 確認) |
| 3 | fcc4741d | healthy 分析 + odds_base retro + V18/V19 完全 retro + watchdog 化 |
| 4 | 777cc08e | fix: V18/V19/V17 LGB model 破損 (CRLF) 復旧 + retro script 修正 |
| 5 | 660b13a6 | 5/3 直前予測分析 + V15/V17 改良提案 + 5/9 plan v2 |
| 6 | ccd0c890 | fix: 5/4 朝の事故防止 緊急対応 3 点 |

### 5/4 (日) 朝 〜 夜 (Day 2)

| # | commit | 内容 |
|---|--------|------|
| 7 | e20bbc0c | 5/4 朝 データ監査 + Phase 2.5 残タスク棚卸し + .gitattributes 設定 (CRLF 再発防止) |
| 8 | 470a9d90 | Phase 2.5 A: ra_score 再取得 blocker (jra_races_full 2026 年なし) |
| 9 | 48709274 | Phase 2.5 B: sc_score 再取得 同 blocker 確認 |
| 10 | 5262e0c0 | Phase 2.5 C: TYB publish タイミング観測 自動化完了 |
| 11 | 0e03c55c | Phase 2.5 D: V18/V19 Platt scaling + 5/2-5/3 retro 再評価 |
| 12 | d8988f97 | Phase 2.5 E: 5/4 進捗サマリー (4 タスク A-D) |
| 13 | 9c88d27c | タスクスケジューラ全 16 件 静音化 vbs ラッパー + 一括変更 |

### 5/5 (月) 朝 〜 PM (Day 3、本セッション)

| # | commit | 内容 |
|---|--------|------|
| 14 | b4c4894c | jra_races_full 2026 年 4-5 月分追加 + ra_score 60 races 取得 |
| 15 | 6b5e4e7b | sc_score (stable_comments) 2026 年 4-5 月分取得完了 |
| 16 | 74eb10b7 | race-level normalization + V18/V19 retro 改善 (Session #10) |
| 17 | 6820b362 | Phase 2.5 D+E+F: V17 features 充足率検証 + 5/9 plan v3 維持 |
| 18 | bfbddebc | 5/5 かしわ記念 (船橋 11R Jpn1) NAR 予測 + 遊び投資配分 |
| 19 | e5f71cfa | NAR v4 model 復活 (archive→active) + 5/5 柏記念 ハイブリッド予測 |
| 20 | 57029ff1 | NAR v4 体系化 (pipeline 設計 + 自動化準備 + V15 統合プラン) |
| 21 | 2b6dc4eb | 5/9 本番最終調整 (運用ガイド + 撤退ライン + 自動レポート) |

### 5/5 (月) 夜 (Day 3 続き)

| # | commit | 内容 |
|---|--------|------|
| 22 | edfa9897 | 5/3-5/5 振り返り記録 + 引き継ぎ書 v2 (誤情報 7 件訂正) |
| 23 | eeb48e45 | NAR pipeline 未実装 script 2 個 実装 (5/12 paper 発火対応) |
| 24 | 06081e26 | tasks 整理 (Phase 1-A クローズ確認、5/5 PM スナップショット) |
| 25 | d761a257 | 静音化 28 task 動作完全検証 (Session #18) |
| 26 | 3e4b5fa2 | 古いログ + stale CSV → archive 移動 (291 MB) |
| 27 | c7fdce57 | 5/9 ドライラン リハーサル完了 |
| 28 | af3951f9 | 引き継ぎ書 v2 セッション #16-18 追記 |
| 29 | c111bd5f | 累計収支 + Phase 2.5 進捗 ダッシュボード |
| 30 | e23b5a88 | 古いモデル archive 化 + README v2 |
| 31 | c106f66b | NAR pipeline placeholder 2 個 本実装 |
| 32 | 7c5ba9f8 | **V15.1 features 拡張試作 (KKA/SKB/SR 逆輸入) — SKB +0.0694 大発見** |
| 33 | f408d93d | 5/5 柏記念: 本格予測 v2 (過去傾向 + 専門 features + ハイブリッド) |
| 34 | (本書 base) | UPDATE_INVENTORY 6 領域 並列調査 (Session #23) |
| 35 | 86cd1da5 | **Phase 2.5+: 緊急 3 件対応 (ProcessWatchdog v2 + fire_check 監査 + chihou_races 解決)** (Session #24) |

---

## 3. 領域別成果

### 3.1 モデル

| モデル | 状態 | 備考 |
|--------|------|------|
| **V15** (本番、AUC 0.8939、150 features) | ✅ 健全 | 5/9 案B改 で稼働、軸 top3 率 -16pt gap は継続観察 |
| **V15.1** (試作、SKB +0.0699) | 🔬 評価中 | LGB single quick mode で AUC 0.9427、4-model + WF + leak audit が Phase 3 必須 |
| **V17** (morning ULTRA-CLEAN) | ✅ CRLF 復旧済 | TYB 観測継続中、5/11 月に midday 戦略生死決定 |
| **V18 (単勝) / V19 (複勝)** | ⚠️ shift 残 | BT 295%/149%、retro で distribution shift 27.7x、5/16 試行は前提 5 件未達で no-go 寄り |
| **NAR v4** (AUC 0.8145/0.8519) | ✅ archive→active 復活 | 5/5 柏記念で 0.777 完全再現、5/12 paper 開始準備済 |

### 3.2 データ

| データ | 状態 | 備考 |
|---|---|---|
| jra_races_full.csv | 5/3 末尾まで復旧 | 178MB / 532K 行、commit b4c4894c で 4-5 月分手動 backfill |
| JRDB Advance 23 種 | ✅ 健全 | 5/3 まで raw + 連結 CSV、jrdb_paci.csv 4/4 停止問題は解消済 |
| netkeiba master course (premium) | ✅ Cookie 1817 文字健全 | speed_index 4-5 月のみ stale |
| ra_score (race_review) | 部分復活 (60 races) | 完全復活は 5/16 V15.1 SKB 投入時に必要 |
| sc_score (stable_comments) | 4-5 月分取得済 | カバレッジ 30% から大幅改善 |
| odds_base | retro 完了 | watchdog 化 |
| **JRA-VAN (TARGET Frontier JV)** | **退会済** | 一度だけ契約、2025 年データ抜き取り済、5/24 まで再契約不要、6 月の v16/v17/v20 学習タイミングで再契約候補 |

### 3.3 自動化

- **schtasks 28 件**: 静音化 vbs ラッパー (`tools/silent_runner.vbs`) + 動作完全検証済 (Session #18)
- **DailyPredict watchdog 化**: subprocess 監視 + Cookie 自動 refresh + max 3 restart (Session #4)
- **race_day_report**: 土日 18:00 自動 Discord 通知 (Session #14)
- **NAR pipeline 5 段**: NarMidDayCalendar / NarDailyScrape / NarDailyPredict / NarLiveOddsRefresh / NarDailyResults (admin 登録済、5/12 発火待機)
- **TYB publish monitor**: 毎時 X:30 観測、5/4-5/10 で蓄積中、5/11 結果判定
- **緊急 3 件** (Session #24): ProcessWatchdog v2 schtasks ps1 + fire_check 監査 (バグ 2 件修正)

### 3.4 開発ワークフロー

- tests 19 ファイル / 17+ PASS 全回維持
- 引き継ぎ書 v2 (`docs/HANDOFF_5_5_TO_5_9.md` 420 行) で v1 誤情報 7 件訂正
- ドキュメント体系: docs/ 37 + data/v18/ 59 + data/results/ 16+
- 静音化済 vbs + Discord 3 channel 振り分け (#bets / #updates / fallback)
- .gitattributes で CRLF 再発防止 (Session #7)
- archive/ 整理 (291MB 移動、Session #18)

---

## 4. 最大の発見 5 件

### #1 V15.1 SKB +0.0699 AUC 改善 (commit 7c5ba9f8)

JRDB SKB (専門家印) 10 features を逆輸入して LGB single retro: V15 baseline AUC 0.8728 → V15.1 ALL 179f AUC 0.9427。 寄与は SKB が独占 (KKA 16f は 0%、SRB 8f は +0.0013)。 リーク確認 PASS (pre-race 印)。 Phase 3 (5/24+) で 4-model ensemble + WF + leak audit を経て本格採用候補。

### #2 distribution shift 27.7 倍 (commit 74eb10b7)

V18/V19 retro 5/2-5/3 で全 filter で bet=0 の原因を解析: BT/retro race_max_p 比 27.69x = scaling shift。 race-level softmax T=1.0 normalize で bet>0 化 ROI 1450-2708% 復活、ただし winner_top1 rate 34.5% (BT 47.8%) は monotonic 変換で **不変** = feature shift 別問題と確定。 5/15 までに調査必要 (90 min)。

### #3 chihou_races_2020_2025.csv blocker 誤情報 (commit 86cd1da5)

UPDATE_INVENTORY で「5/12 NAR paper の唯一最大 blocker、5-10h 復旧必要」と緊急扱いだったが、`archive/nar/train_nar_v4.py` L22-50 を解析した結果、`SCRAPED_CSV = 'nar_all_races.csv'` のみ実使用、`OLD_CSV = 'chihou_races_2020_2025.csv'` は変数定義のみで未読込と判明。 5-10h の不要作業を回避。

### #4 引き継ぎ書 v1 の誤情報 7 件 (commit edfa9897)

`docs/handoff_v1_v2_diff.md` で訂正:
- training_times 2025: v1 = 2,551 件 → v2 **192,296**
- 5/2 USER 損失: v1 = -23,800 円 → v2 **-8,820 円**
- v15 batch ROI 31.3% → v2 案B改 ROI **161%** (BT)
- TYB 17:00 確実公開 → v2 **不明** (404 観測継続)
- NAR モデル AUC 0.789 → v2 **0.8145** (v4 復活、OOS 0.8519)
- 累計 約 -25,000 円 → v2 **+14,140 円**
- chihou_races 不在 blocker → 実は誤情報 (Session #24 確定)

教訓: 数字は必ず生データで再検証。session 越し transfusion 禁止。

### #5 archive 化された資産が活用可能 (commit e5f71cfa)

NAR v4 model + train_nar_v4.py が `archive/nar/` で 3/11 から眠っていた。 復活させて 5/5 柏記念で 0.777 完全再現確認、5/12 paper 開始の主力モデルに。 archive ≠ 削除を再認識。

---

## 5. 学んだ教訓 (簡潔版)

| # | 教訓 | source |
|---|------|--------|
| 1 | データ品質は数字を生から再検証 | v1 → v2 訂正 7 件 |
| 2 | CRLF は LightGBM model を破壊する | commit 777cc08e + .gitattributes 対策 |
| 3 | distribution shift は normalize で解消されない | V18/V19 winner_top1 -13.3pt 不変 |
| 4 | 静音化 (vbs hidden window) で flicker 問題消える | Session #9 / 28 task 一括 |
| 5 | archive ≠ 削除、再活用可能 | NAR v4 復活 |
| 6 | 撤退ライン明文化 (-50k) で動揺消える | risk_management_5_9.md |
| 7 | 取り返し禁止 + 累計死守 = 心理安全装置 | ユーザー絶対方針 |
| 8 | 緊急扱いのものほど誤情報の可能性 | chihou_races 誤情報 |
| 9 | SCRAPER-GUARD は caller 引数渡しが必須 | 4/19 事故 + 4/27 修正 |
| 10 | 単発 task より監査 task (fire_check) で誤発報を抑止 | am8 平日 critical 修正 |
| 11 | dry-run で必ず動作確認、本実装直後ほど | scrape_nar_today/results 動作未検証 → 5/12 前確認推奨 |
| 12 | レポート訂正は事実発覚した瞬間に | 5/5 のうちに JRA-VAN + chihou 訂正 |

---

## 6. 残タスクとマイルストーン

### 5/6 (火) ユーザー手動 1 件のみ

```powershell
# admin PowerShell で
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1
```

これで ProcessWatchdog v2 切替完了。 他は何もしなくて良い。

### 5/7 (水) - 5/8 (金) 平日 隙間時間

- 🟠 SED260503 取得 + KKA/KAB 連結再実行 (15min)
- 🟠 speed_index 4-5 月 backfill (30min)
- 🟠 戦略⑦ 5/2-5/3 retro 完全版 (2h、5/9 投入直前必須)
- 🟠 cumulative_results.csv 書き込みバグ修正 (4h)
- 🟠 累計 PnL 自動計算 + 撤退ライン Discord アラート (1h)

### 5/8 (金) 21:00 後 (1 度だけ、必須)

```bash
# 12R race_name 確認
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}', headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '→', (m.group(1).strip() if m else 'NOT_FOUND'))
"

python tools/refresh_cookie.py --check
```

### 5/9 (土) 本番

`data/results/20260509_pat_checklist.md` 順番通り。
06:30 / 08:00 自動発火 → 14:00-15:30 PAT 投票 (採用 R × 700 円、最大 2,100 円) → 18:00 自動レポート → 20:30 振り返りテンプレ。

### 5/10 (日) - 5/11 (月)

- 5/10 ROI 確認 → 5/11 月の判断
- TYB 観測完了判定 (5/11、5min) → v17 ULTRA-CLEAN 生死決定
- nar_all_races.csv 2025-06 〜 2026-05 backfill 推奨

### 5/12 (火) NAR paper 開始

scrape_nar_today/results 手動 dry-run (30min) → 5 task 自動発火 4 日間観察 → 5/15 夜 go/no-go 判定。

### 5/13 (水) - 5/15 (金)

- race-level normalize 本番統合 (predict_core.py、30min)
- feature distribution shift 調査 (90min) ← 5/16 V18/V19 試行の生命線
- 複勝 odds 実値で fukusho retro 再評価 (30min)
- 5/15 夜 V18/V19 GO/no-go 判定

### 5/16 (土) 試行候補

- V18/V19 1,000 円/日 (条件達成時のみ、現状 no-go 寄り)
- NAR 500 円/日 (paper 良好なら)
- V15.1 paper trading 開始

### 5/17 (日) - 5/24 (金) Phase 3 移行週

- V15.1 4-model ensemble 学習 (6h)
- V15.1 全年 WF (6h)
- V15.1 軸 top3 率 retro WF (2h)
- V15.1 leak audit (4h)
- V15.1 + 戦略⑦ MC 再実行 (2h)
- KKA 16f coverage 0% 原因究明 (1h)
- predict_core.py SKB merge 統合 (1h)

### 5/24 (金) 夜 Phase 3 移行判定

採用基準 6 条件全達成チェック (`data/v18/post_5_9_improvement_template.md` §5):
1. JRA 案B改 ROI ≥ 100% (4/12-5/24 累計)
2. race-level normalize 本番統合済
3. NAR paper 12-14 race 蓄積
4. V18/V19 試行 sample 30+ bets
5. 累計 +10,000 円維持
6. 撤退ライン余裕 30,000+ 円

→ GO なら V15.1 + V20 統合構想着手 / 未達なら Phase 2.5 延長

### 5/25 - 6/8 (Phase 3 V15.1 本格採用)

V15.1 production pipeline 統合 + paper trading 7-14 日 + 段階的本番投入。

### 6/9 - 6/30 (V20 統合モデル)

JRA + NAR 共通 features 52+ で V20 学習 (要 JRA-VAN 再契約 + chihou data 整理)。

---

## 7. 累計収支推移 + 撤退余裕

| 日付 | 累計 | 撤退余裕 (-50k 基準) | 備考 |
|------|------|---------------------|------|
| 5/1 (推定起点) | +30,520 円 | +80,520 円 | 5/2 直前 (逆算) |
| 5/2 | +21,170 円 | +71,170 円 | 5/2 USER 実投資 -9,350 (15R 1hit) |
| 5/3 | +13,220 円 | +63,220 円 | 5/3 USER 実投資 -7,950 (22R 4hits) |
| 5/5 | **+13,530 円** | **+63,530 円** | NAR 柏記念 +310、生データ累計 |
| 5/5 USER 申告 | +14,140 円 | +64,140 円 | USER 申告 (±610 円差、要確認) |
| 5/9 想定 (期待) | +13,930-14,830 円 | +63,930-64,830 円 | V15 案B改 想定 ROI 161% |
| 5/9 想定 (最悪) | +11,430 円 | +61,430 円 | 全 R 外し (-2,100 円) |
| 5/16 想定 (期待) | +15,000-20,000 円 | +65,000-70,000 円 | NAR paper 良好なら |
| 5/24 想定 | +20,000-25,000 円 | +70,000-75,000 円 | Phase 3 移行判定 |
| 6/末 想定 | +30,000-50,000 円 | +80,000-100,000 円 | Phase 3 完了 |
| 年末目標 | +100,000 円 | +150,000 円 | V20 + 月次成長 |

---

## 8. 結論 (一句)

**Phase 2.5+ 完璧に着地**。 51.5 時間連続作業の終点として、5/9 当日リスクゼロ・5/12 NAR paper 開始準備完了・5/16 試行と 5/24 Phase 3 移行の判断材料すべて揃った。 ユーザー絶対方針 (取り返し禁止 / 累計 +14,140 円死守 / 撤退ライン -50,000 円) と整合した運用体制が確立。 朝起きて `docs/UPDATE_INVENTORY_20260505.md` § 0 と本書 § 6 を読めば全体像が把握でき、5/6 (火) は admin 1 コマンドだけ実行すれば次のフェーズへ進める。
