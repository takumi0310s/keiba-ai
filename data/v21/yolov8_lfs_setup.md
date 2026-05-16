# YOLOv8 weight 用 Git LFS setup 手順

> Phase C (5/16) Terminal C 成果物
> 状態: **準備 doc のみ**、 actual `git add` / `git commit` は **親が判断** (5/18+ で 実行候補)

---

## 1. 対象 file (現状 untracked)

| file | size | 用途 |
|------|------|------|
| `yolov8n.pt` | **6,549,796 byte (約 6.2 MB)** | YOLOv8 nano、 軽量、 CPU 138ms/frame (Session #42 確認) |
| `yolov8s.pt` | **22,588,772 byte (約 21.5 MB)** | YOLOv8 small、 中精度、 GPU 推奨 |

両 file とも root に存在 (`C:/Users/takum/keiba-ai/yolov8n.pt`, `yolov8s.pt`)。

→ 合計 **約 28 MB**、 GitHub の 100 MB / file 制限内だが、 binary を git history に直接入れると repo 肥大化 → **LFS 推奨**。

---

## 2. 環境 確認 (5/16 完了)

| 項目 | 状態 |
|------|------|
| git-lfs install | ✅ `git-lfs/3.7.1 (GitHub; windows amd64; go 1.25.1; git b84b3384)` |
| repo の lfs 初期化 | 未確認 (`git lfs install` 実行履歴 要確認) |
| 既存 `.gitattributes` | ✅ 存在 (LightGBM/XGBoost model 用 `-text` rule のみ) |

---

## 3. .gitattributes 追加内容 (案、 親が commit 判断)

既存 `.gitattributes` に以下 2 行 追加:

```gitattributes
# YOLOv8 weight files: Git LFS tracked (Phase C 5/18+ PoC 用)
*.pt filter=lfs diff=lfs merge=lfs -text
```

→ 全 `.pt` file が LFS で扱われる (将来の YOLOv8m / YOLOv11 等にも適用)。

---

## 4. 親 が 実行する step (5/18+ 候補)

### Step 1: LFS 初期化 (1 回のみ)
```bash
cd C:/Users/takum/keiba-ai
git lfs install
```

### Step 2: .gitattributes 更新
- 上記 2 行を 既存 `.gitattributes` に append

### Step 3: track + add
```bash
git lfs track "*.pt"  # .gitattributes に自動 append される
git add .gitattributes
git add yolov8n.pt yolov8s.pt
```

### Step 4: 確認
```bash
git lfs ls-files
# 期待出力:
#   <oid> * yolov8n.pt
#   <oid> * yolov8s.pt
```

### Step 5: commit + push
```bash
git commit -m "Phase C: YOLOv8 weights を Git LFS で tracked 化 (yolov8n.pt 6 MB + yolov8s.pt 22 MB)"
git push origin main
```

---

## 5. 注意事項

### 5.1 既存 commit 履歴に `.pt` が 入っていないか確認
```bash
git log --all --full-history -- yolov8n.pt yolov8s.pt
```
- もし過去 commit に raw binary が含まれていれば、 `git filter-repo` で 履歴 cleanup 必要 (★ 親判断 ★、 destructive のため 注意)。
- Phase C 5/16 時点では **両 file untracked** = 履歴クリーン、 そのまま LFS 化で OK。

### 5.2 LFS quota (GitHub free)
- free tier: 1 GB storage + 1 GB bandwidth / 月
- 5/16 時点で 28 MB 追加 → 余裕あり
- ★ ただし将来の fine-tuned model (YOLOv8m 50 MB+ × 多バージョン) を見越して、 月次 quota monitor 必須 ★

### 5.3 Streamlit Cloud / 他環境
- Streamlit Cloud は LFS 非対応 → V21 動画 features 投入時 (9/1+) は LFS file を runtime download する仕組み 必要
- 5/16 時点では V15 production のみ Streamlit、 V21 / YOLO は ローカル only → 影響なし

---

## 6. 代替 案 (LFS NG の場合)

| 案 | 内容 | 採用判断 |
|----|------|---------|
| A. release artifact | GitHub Release に `.pt` を attach、 runtime download | 親が LFS quota 不安なら 候補 |
| B. ローカル download script | `tools/v21/download_yolov8_weights.py` で ultralytics 公式 URL から取得 (初回のみ) | ★ 最軽量、 推奨 fallback ★ |
| C. 外部 storage (S3 等) | 自前 bucket | コスト + 運用負担 大 |

★ 推奨優先順 ★: **LFS (主)** → **案 B (fallback、 LFS quota 超過時)**。

---

## 7. Phase C 5/16 完了 status

- [x] file size 確認 (6.2 MB + 21.5 MB)
- [x] git-lfs install 確認 (3.7.1)
- [x] .gitattributes 既存内容 確認
- [x] 追加 rule + 手順 doc 作成 (本 file)
- [ ] **actual `git lfs track` / `git add` / `git commit` は 親判断** (5/16 は実行せず)

---

## 8. 関連 doc

- [phase_c_patrol_8_features_spec.md](phase_c_patrol_8_features_spec.md) — 8 features 詳細
- [phase_c_patrol_yolo_poc_plan.md](phase_c_patrol_yolo_poc_plan.md) — 5/18-5/24 PoC plan
- [phase16_patrol_yolo.md](phase16_patrol_yolo.md) — 既存 Phase 16 C 設計 (前身)
