# 古いログ + stale CSV → archive 整理 (5/5 PM、Session #18)

---

## 1. 移動 内容

### 1.1 data/_backup_20260501/ → archive/old_backups/_backup_20260501/

- size: 291 MB
- contents: jrdb_skb.csv.old, jrdb_zk.csv.old (古い JRDB CSV backup)
- 用途: 完全 retain (削除しない、safety)

### 1.2 data/_emergency_backup/ → archive/old_backups/_emergency_backup/

- size: 56 KB
- contents: 緊急 backup ファイル
- 用途: 完全 retain

### 1.3 logs/ 30 日以上前 (3 files) → archive/old_logs/

| file | mtime |
|------|-------|
| logs/premium_scrape_20260404.log | 2026-04-04 |
| logs/race_auto_notify_20260404.log | 2026-04-04 |
| logs/roi_v135b.log | (古) |

→ logs/ は元々 .gitignore で git tracking 外、移動は disk hygiene 目的のみ。

---

## 2. ディスク使用量 before / after

| dir | before | after | delta |
|-----|------:|------:|------:|
| data/ (root level only) | 291 MB +α | (移動後 -291 MB) | -291 MB |
| archive/ | 1.0 MB | **292 MB** | +291 MB |
| logs/ | 8.3 MB | 8.3 MB | (3 file 移動 軽微) |

**実際 disk 上の data/ 全体は 15GB** (主に v162/v17 model dirs 各 1-2GB)。これらは生成物であり cleanup 対象ではない。

---

## 3. .gitignore 更新

新規追加 (line 87-93):

```gitignore
# Old backups / emergency snapshots (moved to archive/ on 5/5 PM, Session #18)
data/_backup_*/
data/_emergency_backup/

# Archive directory (old logs / backups / unused models)
archive/old_logs/
archive/old_backups/
```

→ 今後も `_backup_YYYYMMDD/` パターンで作成された backup dir は自動 ignore。

---

## 4. archive/ 構造

```
archive/
├── nar/                                 # NAR モデル archive (Session #12 で復活確認)
│   ├── keiba_model_nar_v4.pkl           ← active (data/nar/models/ にコピー済)
│   ├── keiba_model_v9_nar.pkl           # 旧版 (479 KB)
│   ├── keiba_model_v10_nar_ref.pkl      # JRA v10 参考 (620 KB)
│   ├── train_nar*.py                    # 学習 script
│   └── backtest_nar*.py / *.json        # 旧 backtest
├── old_backups/                         # 5/5 PM 追加
│   ├── _backup_20260501/                # JRDB CSV backup (291 MB)
│   └── _emergency_backup/               # 緊急 backup (56 KB)
└── old_logs/                            # 5/5 PM 追加
    ├── premium_scrape_20260404.log
    ├── race_auto_notify_20260404.log
    └── roi_v135b.log
```

---

## 5. 削除しなかった (注意)

| dir | size | 理由 |
|-----|-----:|------|
| data/v162/ | 2.0 GB | V16/V162 model artifacts (現存 reference 用) |
| data/v17/ | 1.4 GB | V17 ULTRA-CLEAN model artifacts (Phase 2.5 観測中) |
| data/v18/ | 139 MB | active development、5/16+ Phase 2.5+ で使用 |
| logs/ (5/4以降) | 8 MB | 直近 1 ヶ月、debug 用 retain |

**user 方針**: 「闇雲に削除せず、archive 保管」(Session #15 教訓 #5)。生成物 + 大型 model dirs は触らない。

---

## 6. 結論

✅ **291 MB の stale backup を archive 化**
✅ **.gitignore で 今後の untracked 増加を防止**
✅ **削除なし、すべて retain (safety)**

5/9 当日 + 5/12 NAR paper でも data/ 直下の clutter なし。
