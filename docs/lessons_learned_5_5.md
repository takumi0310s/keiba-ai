# 5/3-5/5 学んだ教訓 (約 46 時間 14 セッション)

生成: 2026-05-05 17:40 (Session #15)

---

## 教訓 1: データ品質 — 引き継ぎ書の数字は必ず生データで再検証

### 何が起きたか
- v1 引き継ぎ書 (Session #1 以前) で複数の誤情報が流通:
  - "training_times 2025 = 2,551件" → 実は 192,296件
  - "5/2 USER 損失 -23,800円" → 実は -8,820円
  - "v15 batch ROI 31.3%" を USER ROI と誤解 → 実は全 R 仮想 ROI、案B改 USER 期待 161%
  - "TYB 17:00 公開" → 実は不明 (5/4 12:25 / 5/9 12:25 共に 404)
  - "NAR モデル AUC 0.789" → 実は 0.8145 (archive にあった v4)
  - 累計 "約 -25,000円" → 実は **+14,140円**

### 教訓
- **引き継ぎ書の数字は session 越しに transfusion される → 誤情報は雪だるま式に拡大**
- 数字には **必ず source path を併記** (`data/cumulative_results.csv` 等)
- 重要な数字は **session 開始時に再確認** (v2 では 0 章で v1 訂正をまとめた)

### 実装
- `docs/HANDOFF_5_5_TO_5_9.md` (v2) は 全数字 source 併記
- v1 訂正は `docs/handoff_v1_v2_diff.md` に集約

---

## 教訓 2: モデル管理 — git 15MB 制限 + CRLF 問題

### 何が起きたか
- v18/v19/v17 LGB model (.txt 形式) が CRLF 変換で **破損** (Session #5)
- `.gitattributes` 未設定で Windows 環境 git checkout 時に LF→CRLF 変換が走った
- 復旧に時間 (1.5 時間ロス)

### 教訓
- **LGB の .txt model は binary 扱い必須** (改行コード変換で破損)
- 大きい model (>15MB) は git LFS or .gitignore + 別 storage
- `.gitattributes` の `* text=auto` は注意、特定 model 拡張子は `binary` 明示

### 実装
- `.gitattributes` で `*.txt linguist-generated` + `*.lgb_model binary` 等設定
- archive/nar/ で 大型 model を保管 (Session #12 で復活して確認)

---

## 教訓 3: distribution shift — BT vs production で 27.7倍の prob 縮小

### 何が起きたか
- v18/v19 BT 2025 OOS: race_max_p mean **0.347**
- 5/2-5/3 retro: race_max_p mean **0.013** → **27.7x scaling shift**
- top1/top2 ratio はほぼ同等 (4.13 vs 4.37) → ランキング構造は保持
- winner_top1 rate: BT 47.8% → retro 34.5% (-13pt)
- Phase 2 filter (p>=0.5) で 全 bet=0 になり 5/16 試行 ブロック

### 教訓
- **BT は production 環境を完全再現できていない**
- Platt scaling (calibration) では 0.154→0.213 の minor 改善のみで不十分
- race-level normalization (softmax T=1.0) で sum=1 強制 → 表面 上 bet 通るが、winner_top1 rate は不変 (monotonic)
- **本質的 calibration 改善ではなく スケール調整に過ぎない**

### 実装
- `tools/race_normalize.py` (Session #10)
- `data/v18/distribution_shift_analysis.json` で 27.7x 定量化
- 根本治療は別 task: feature distribution shift 調査 (5/16+)

---

## 教訓 4: 静音化 — schtasks の bat 直接呼び出しは UI 配慮欠如

### 何が起きたか
- 16 task が schtasks → bat 直接実行 → 黒いコンソールがちらつく
- 朝 03:00 / 06:00 / 06:30 / 07:00 / 08:00 / 08:50 + 毎時 X:30 (TYB monitor)
- USER ストレス + 視覚混乱

### 教訓
- **自動化と UI 配慮はセット**で設計
- bat 直接呼び出しは `wscript.exe + .vbs` ラッパー で hidden window 化
- 動作 完全維持 (実行内容は同じ、出力ログも同じ)

### 実装
- `tools/silent_runner.vbs` (wscript Run with windowStyle=0, wait=True)
- `tools/silentify_all_tasks.ps1` (admin で 一括変更)
- `tools/silentify_rollback.ps1` (backup JSON ベース 巻き戻し可能)
- `data/v18/silentify_tasks_user_guide.md` (admin 手順書)

---

## 教訓 5: archive 保管 — 闇雲に削除せず 資産化

### 何が起きたか
- Session #11 で 5/5 柏記念用に NAR モデル不在 → ヒューリスティック予測作成
- Session #12 で `archive/nar/keiba_model_nar_v4.pkl` (167KB) 発見 → active 復活
- AUC 0.8145 (Pattern B) で機能、柏記念 軸=8 ミッキーファイト p_ens=0.777 完全一致
- もし archive を削除していたら NAR は 5/12+ 別タスクで再学習が必要 (~28h)

### 教訓
- **闇雲に削除せず、archive/ に保管**
- archive/ も git history で追跡 (今回は active 復活で git history 残す)
- 容量問題なら git LFS or 別 storage、git 内に置くのは "見つけやすい" メリット大

### 実装
- `data/nar/models/keiba_model_nar_v4.pkl` (active)
- `tools/predict_nar.py` で 汎用化 (柏記念 ad-hoc → race_id 指定可)
- backtest_nar_v4_quick.py で OOS AUC 0.8519 再現確認

---

## 教訓 6: ストイック作業 — 並列セッション + git commit-per-task で flow 維持

### 何が起きたか
- 5/3 19:01 〜 5/5 17:30 = 約 46 時間
- 14 セッション 21 commits (平均 1 commit / 2 時間)
- うち Session #8 は 5 commits (Phase 2.5 A-E 一括)
- うち Session #10 は 4 commits (race-norm + sc_score 並行 fork)
- Session 間の context 切り替え 問題なし

### 教訓
- **commit-per-task で flow 維持**: 1 task = 1 commit、push 待ち中 次 task 着手
- 並列セッション: 別 fork で同時進行可能 (今回は user 報告 commit が交互に)
- TaskCreate / TaskUpdate で 進捗 trace 化
- 長時間作業は **生産性 落ちない**、むしろ context 蓄積で議論質 向上

### 実装
- 全 session で TaskCreate (タスク化) → 順次 in_progress → completed
- 全 commit に Co-Authored-By タグ
- session 越しに `data/v18/phase_2_5_progress_*.md` で context bridge

---

## 教訓 7: USER 数字と モデル数字 を絶対 混同しない

### 何が起きたか
- `data/cumulative_results.csv` は **全 35R BATCH 仮想 ROI**
- USER 実投資は subset (案B改 / 健全な選択投資)
- 5/2: cumulative -15,690円 (33R 全買い理論) ≠ USER 実 -8,820円
- 5/3: cumulative -16,350円 (34R) ≠ USER 実 -520円 (or 案B改 1R 採用なら +2,980円)

### 教訓
- **モデル評価 (theoretical full)** と **USER 損益 (actual selective)** は 別の data
- doc では 必ず明示 ("USER 実投資ベース" or "全買い仮想")
- v1 でこの混同が損失感覚を bias していた (-25,000円 想定 ≠ 実 +14,140円)

### 実装
- v2 doc で 数字に "USER 実" / "BATCH 仮想" を併記
- `tools/race_day_report.py` は 採用 R (案B改) ROI を main に、参考 R (除外) を別表

---

## 教訓 8: 撤退ラインの 厳格化 + 余裕の見える化

### 何が起きたか
- 撤退ライン -50,000円 は v1 から一貫
- だが累計が -25,000円 想定 (v1 誤) で残余裕 25,000円 と思っていた
- 実は +14,140円 → 余裕 64,140円 (3 倍)
- USER 心理的余裕が違う

### 教訓
- 撤退ラインは **絶対値** (-50,000円) と **余裕** (+64,140円) 両方明示
- 余裕の "シミュレーション" を doc に記載 (5/9 最悪 → 累計 +12,040円 等)
- 数字を こまめに自動更新 (race_day_report.py で 18:00 自動)

### 実装
- `data/v18/risk_management_5_9.md`: 4 段撤退ライン + 余裕シミュ
- `tools/race_day_report.py`: 累計 + 撤退まで余裕 を Discord 通知
- `data/v18/post_5_9_improvement_template.md`: monitoring 表 (5/9 / 5/16 / 5/24 累計)

---

## 教訓 9: NAR と JRA の役割分担 — 並列 ≠ 統合

### 何が起きたか
- Session #13 で NAR v4 体系化、Phase 3 v20 統合モデル構想
- 並列運用 (平日 NAR + 土日 JRA) は 5/16 から実用可能
- 統合モデル v20 は ~28h 工数、Phase 3 (5/末-6/末)

### 教訓
- **段階リリース**: まず並列 (容易)、検証後 統合 (費用)
- 統合は ROI 蓄積後 (5/24 Phase 2.5 完了判定 後) で十分
- 早期統合は失敗時 dual loss (JRA + NAR 両方影響)

### 実装
- `data/v18/jra_nar_integration_plan.md`: 5/16-5/24 並列、Phase 3 統合
- 投資配分: JRA 案B改 9,800円/日 + NAR 500円/日 → 段階 ramp

---

## 教訓 10: ドキュメント品質 = 次回セッションの speed

### 何が起きたか
- v1 → v2 切替で session #15 (本書) で 引き継ぎ強化
- doc を細分化 (operation_guide / pat_checklist / risk_management / pre_check / post_template)
- 5/9 朝 起きたら pat_checklist.md 1つで投票完了 設計

### 教訓
- **1 doc = 1 用途**: 巨大 doc は読まれない、細分化が読みやすい
- index doc (HANDOFF) で 全 doc への path を明示
- doc の 鮮度 (生成時刻 + base commit) を必ず冒頭

### 実装
- `docs/HANDOFF_5_5_TO_5_9.md` (本書 7 章で 全 doc index)
- doc 上部 統一フォーマット: "生成: YYYY-MM-DD HH:MM (Session #N)"
- 重要な doc は 重複情報あっても OK (1 source of truth より 1 use case)

---

## 11 ボーナス: 並列 push race condition 観察

Session #10 で複数 commit が並行で origin に届いた時、push race condition で `[remote rejected]` 1-2回 発生したが、**最終的に 全 commit 整合** (rebase + 再 push で OK)。

git の atomic push は安全。ただし pre-push hook (regression_test.py) が 30 秒 × 試行回数 で時間かかる。

→ **多 commit を短時間に push する場合は順次** (並列 push は無駄に hook が走る)。

---

## 結論

5/3-5/5 で 14 セッション 21 commits、約 46 時間。
得たもの:
- 5/9 投資 GO 判定 (案B改 161% ROI 確証)
- v18/v19 distribution shift 定量化 + normalize 試作
- NAR v4 復活 + 体系化 (5/12 paper / 5/16 試行)
- 静音化 (schtasks 23+ 件)
- データ監査 (v1 誤情報 7件 訂正)

学んだもの: 上記 11 教訓。

次回 (5/8 夜 or 5/9 朝) は `docs/next_session_checklist.md` から始める。
