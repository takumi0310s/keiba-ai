# P4-1 push 不能 root cause + 解消 design (2026-05-16)

## 0. 結論

- **推奨案: 案 E (GitHub Release / external storage への移行 + 案 F 現状維持の hybrid)**
- 理由: user 絶対遵守 (filter-repo / force push 永久 NG / destructive op 禁止) で全 history 書換系 (A/C/D) は全滅。 案 B は history が残るので push 修復しない。 案 F 単独だと backup なし risk。 案 E で 未来は clean (将来生成される large data は GitHub Release / external storage)、 既存 60 commits の local backup は user の判断で 案 F 維持。
- 着手: 5/24+ 週末 (Phase 3 着手と並行、 半日工数)

---

## 1. 114MB CSV 特定 (git 実測)

### 1-1. 該当 blob
- **file path**: `data/v20_training_data_full.csv`
- **size**: 118,305,823 bytes (約 **112.8 MB**、 GitHub の hard limit 100MB 超過 → push 不可)
- **blob hash**: git ls-tree 上では既に削除済 (HEAD には存在しない)
- 確認: `git ls-files data/v20_training_data_full.csv` → 空 (現 index 不在)

### 1-2. add commit
- **commit**: `8dfb595f` (2026-05-12 05:32:47 JST)
- **title**: "Phase 26 task #5+#8 完了: v20 merge fix + 5/17 動的 features tool"
- **当該 file**: v20_training_data_full.csv が 190,342 rows × 103 cols で生成、 commit に含まれた

### 1-3. delete commit
- **commit**: `3f663d37` (2026-05-15 10:38:14 JST)
- **title**: "TOS review + .gitignore 強化 + 大 csv index 削除 (push 修復 Step A、 AI 自律 完了)"
- **動作**: `git rm --cached data/v20_training_data_full.csv` で **index からのみ削除**
- **filesystem は保持** (V20 学習 path に影響なし)
- **.gitignore line 135** に追加済 (`grep -n` 確認)

### 1-4. ★ history 残存 確認 ★
- 8dfb595f と 3f663d37 の間に存在する commit に該当 blob が含まれる
- `git rev-list --objects --all | git cat-file --batch-check ...` で 118,305,823 byte blob が依然 reachable
- **GitHub 100MB hard limit → push 不能の root cause = 8dfb595f に含まれる 118MB blob**
- `git status -uno` で local main は **origin/main から 60 commits ahead** (8dfb595f 以降の 60 commits が一切 push できていない)

### 1-5. 補足: 他の大 blob (HEAD tree、 ★ 100MB 未満 ★)
| size | file | 100MB 超? |
|---|---|---|
| 103,601,164 (98.8 MB) | `data/_v15_optuna_df_cache.pkl.gz` | ★ ギリギリ未満、 単独では push 可 |
| 84,560,456 | `data/jrdb_skb.csv` | NO |
| 82,716,497 | `data/jrdb_zk.csv` | NO |
| 77,534,239 | `data/jrdb_paci.csv` | NO |
| ... 残り | ... | すべて NO |

→ **100MB 超は v20_training_data_full.csv のみ**。 これさえ history から除去すれば push 復帰可能。 但し除去 = history 書換 = destructive op。

---

## 2. 解消 path 候補 6 案 (★ 4 案ではなく実際は 6 案 ★)

| 案 | 方法 | destructive? | user policy 遵守 | 効果 | 工数 |
|---|---|---|---|---|---|
| A | git LFS migration (`git lfs migrate import --include="data/v20_training_data_full.csv" --include-ref=refs/heads/main`) | ★ yes (history rewrite + force push 必須) | ★ NG | full (push 復帰) | 中 (30 min + LFS 課金 検討) |
| B | .gitignore + git rm --cached (★ 既に 5/15 実施済 ★) | no | OK | **なし (history 残存)** | 完了済 |
| C | git filter-repo --invert-paths | ★★ yes (history 完全書換 + force push 必須) | ★ 絶対 NG | full | 短 (5 min) |
| D | 新 branch + cherry-pick で 60 commits 移植 | ★ yes (主 history 切断 + force push 必須) | ★ NG | full (但し commit hash 全変更) | 大 (2-3 時間) |
| E | GitHub Release / external storage (S3 等) で 該当 file を別保管 + .gitignore + download script | no | OK | **partial (未来 clean、 過去 history 残るが push は別問題)** | 中 (半日) |
| F | 現状維持 (local commit のみ、 push 諦め) | no | OK | **なし (backup 不在 risk)** | 0 |

### 各案の詳細評価

**案 A (LFS migration)**: 
- LFS は file を hash で外部に置く GitHub native の仕組み。 但し history を書き換えるため `git push --force` が必須 = user 絶対遵守 (force push 永久 NG) 抵触。
- LFS 課金: 1GB/月 free、 超えると $5/月 + $0.05/GB transfer。 一度きりなら無料枠内だが、 force push policy 抵触で **NG**。

**案 B (.gitignore + git rm --cached)**:
- 5/15 3f663d37 で実施済。 index から消したが history は残るため push 修復には至らない。
- 唯一の効果は 「未来の commit に再混入しない」 こと。

**案 C (filter-repo)**:
- 最強の history 書換 tool。 `git filter-repo --invert-paths --path data/v20_training_data_full.csv --force` で 完全削除可能。
- 但し ★ user 絶対遵守: filter-repo / force push 永久禁止 ★ → **絶対 NG**。
- 5/15 GIT_PUSH_FIX_5_15.md でも user 認可待ち とされていた (実行されず)。

**案 D (新 branch + cherry-pick)**:
- 60 commits を 該当 file 抜きで 一個ずつ cherry-pick → 新 branch を main に rename。
- 主 history が切断され、 commit hash 全変更。 force push 必須。
- user 絶対遵守 抵触 + 工数大 + 既存 collaborator なし (個人 repo) でも commit hash 切断は **NG**。

**案 E (GitHub Release / external storage)**:
- 該当 file は **history に残る (push は依然 不可)** が、 **未来生成される large data** は GitHub Release (1 release あたり 2GB、 個別 file 100MB 上限なし) や S3 / Google Drive 等に upload。
- .gitignore で git tracked にしない。 download script (`tools/fetch_large_artifacts.py` 等) で 取得可能化。
- ★ 既存 60 commits の push は別問題 (案 F に委ねる) ★ だが、 **将来再発防止** + 必要なら external storage 経由で artifact 共有可能。
- 法務的にも OK (TOS_REVIEW_5_15.md 既存)、 destructive op なし。

**案 F (現状維持)**:
- 60 commits を origin に push せず local のみで保持。
- backup なし = HDD 故障 / OS 再インストール時 全消失 risk。
- 但し destructive op 不要、 user policy 完全遵守。

---

## 3. 推奨案 + 根拠

### 推奨: **案 E + 案 F の hybrid**

#### 短期 (5/24+ 即着手)
- **案 E 着手**: 
  - GitHub Release `v15-data-2026-05-16` を作成、 v20_training_data_full.csv (+ 必要なら _v15_optuna_df_cache.pkl.gz 等) を artifact upload
  - `tools/fetch_large_artifacts.py` を作成 (gh release download コマンド wrapper)
  - .gitignore で当該 file を tracked 化禁止 (5/15 適用済を継承)
- **案 F 継続**: 既存 60 commits は local のみで保持、 push せず

#### 中期 (Phase 3 着手後 6 月)
- 将来 V20 学習 data を再生成する際、 100MB 超 file は **必ず GitHub Release** に upload する運用ルール化
- README / CLAUDE.md に「100MB 超 file は GitHub Release へ」と明記
- Phase 4 動画解析 (DLC 訓練 weights 等) も Release 経由化 検討

#### 長期 (Phase 4 以降、 user 判断時)
- もし backup 必要性が緊急化したら、 user が手動で案 C / D を実行 (★ AI からは絶対実行しない ★)
- もしくは Anthropic 等 enterprise 用 git host へ移行 (private + 大 file OK の Bitbucket Cloud / GitLab Premium 等) も検討余地あり

### 根拠
1. **user 絶対遵守を完全に守る**: filter-repo / force push 一切なし
2. **未来は clean**: 5/24+ 以降生成される large file は Release / external で運用
3. **既存 60 commits の risk は user 判断**: AI からは案 F を提案するに留め、 実 backup 取得は user が判断
4. **destructive op ゼロ**: 既存 history / V15 production / data に一切影響なし
5. **工数 半日**: GitHub Release 作成 + fetch script で完結、 Phase 3 と並行可能

---

## 4. 実施 plan (5/24+ 週末)

### 4-1. 実施タイミング候補

| 候補 | 日 | 理由 |
|---|---|---|
| ★ 5/24 (土) AM | 5/24 09:00-12:00 | JV-Link 加入 + Phase 3 着手 plan の合間、 半日空きが取りやすい |
| 5/25 (日) AM | 5/25 09:00-12:00 | 予備 (5/24 で完遂しなかった時) |
| 5/31 (土) AM | 予備 | 念のため 1 週間 余裕 |

### 4-2. step (工数 半日、 4 hours)

1. **(15 min) gh CLI 認証確認** + GitHub Release UI 確認
   - `gh auth status` (既に認証済か確認、 未認証なら `gh auth login`)
   - https://github.com/takumi0310s/keiba-ai/releases へアクセス可能か

2. **(30 min) GitHub Release 作成**
   - tag: `v15-data-2026-05-16` (or 適切な version 名)
   - title: "V15 baseline + V20 training data snapshot (2026-05-16)"
   - body: 各 artifact の用途 / size / hash を記載
   - artifact upload:
     - data/v20_training_data_full.csv (112.8 MB)
     - data/_v15_optuna_df_cache.pkl.gz (98.8 MB、 100MB ギリギリだが念のため)
     - 他 大 csv (jrdb_*.csv) は サイズ的に push 可能なので Release 化は任意

3. **(60 min) `tools/fetch_large_artifacts.py` 作成**
   - 引数: `--release v15-data-2026-05-16` (or default)
   - 動作: `gh release download {tag} --pattern "*.csv" --pattern "*.pkl.gz" --dir data/`
   - 既存 file が存在する場合は skip / overwrite option
   - hash 検証 option (SHA256)

4. **(30 min) README / CLAUDE.md に運用ルール明記**
   - 「100MB 超 file は GitHub Release へ upload、 .gitignore で tracked 化禁止」
   - 「初回 clone 後は `python tools/fetch_large_artifacts.py` で artifact 取得」
   - 既存 large file の Release version を明記

5. **(45 min) verify**
   - 別 directory に clone → fetch_large_artifacts.py 実行 → 全 artifact 取得確認
   - V20 学習 script を試走 (data load まで確認)

6. **(60 min) buffer** (想定外 トラブル対応)

### 4-3. rollback plan

- **GitHub Release 作成は完全 read-only op** (既存 history / file に一切触らない)
- rollback 必要時: GitHub Release を `gh release delete v15-data-2026-05-16` で削除するだけ
- local の data/ filesystem には一切影響なし
- V15 production 完全不変
- 既存 60 commits の状態も完全不変

### 4-4. 既存 60 commits の扱い

- ★ 案 F 継続: local main に保持、 origin への push は諦める ★
- backup として: 
  - local の `.git/` directory を そのまま 別 drive にコピー (HDD 故障対策)
  - もしくは bundle 形式で export: `git bundle create keiba-ai-local-2026-05-16.bundle --all`
  - 上記 bundle file を GitHub Release artifact として upload する option もあり (private repo なので OK)

---

## 5. 補足 / 注意事項

### 5-1. 既存 5/15 doc との関連
- `data/v18/GIT_PUSH_FIX_5_15.md`: 既存 4 案 (filter-repo / BFG / LFS / Release) を列挙、 推奨は filter-repo
- 本 doc は filter-repo を ★ 絶対 NG ★ と再判定し、 Release + 現状維持 hybrid に方針転換
- 既存 doc は archive 扱い、 本 doc が最新方針

### 5-2. なぜ 5/15 git rm --cached だけで push 修復しなかったか
- `git rm --cached` は **index からのみ削除**、 過去の commit (8dfb595f) には依然 118MB blob が残存
- GitHub の push validation は **全 commit の objects** を check するため、 1 つでも 100MB 超 blob があれば reject
- history を完全に書換える (案 A/C/D) しか push 修復方法はない → user policy で全 NG

### 5-3. fabrication 防止 確認
- 全数値は `git rev-list --objects --all | git cat-file --batch-check` の実測
- 118,305,823 bytes / 8dfb595f / 3f663d37 は `git log` / `git show` で実確認
- 「destructive」判定は git の history rewrite 定義に基づく一般知識
- 案比較は GitHub 公式 doc + git filter-repo / LFS 公式 spec ベース

---

## 6. 完了 criteria

- [ ] 5/24+ で 案 E 着手 (GitHub Release 作成)
- [ ] tools/fetch_large_artifacts.py 作成 + verify
- [ ] CLAUDE.md / README に運用ルール記載
- [ ] local backup (案 F 継続) として bundle export (optional)
- [ ] 既存 60 commits の push は ★ 諦める ★ (user policy 最優先)
