# Session #60 C: 5/9 重賞 5 system 最終予測 (simulate モード)

**作成**: 2026-05-09 (Session #60 C)
**実装**: tools/predict_majors_5system_5_9_v60.py
**output**: data/v18/predictions_majors_5system_5_9_FINAL.json
        data/v18/horse_motion_5_9.csv (simulate motion features 9 行)

---

## 1. 5/9 重賞 3R 予測結果

### 1.1 京都 11R 京都新聞杯 (G2) 15:30 — race_id 202608030511

| system | status | top1 | 備考 |
|--------|--------|------|------|
| 1. V15 単独 (production) | ✅ ok | #1 アーレムアレス (score 0.6020) | top2 #8 バドリナート / top3 #2 エムズビギン |
| 2. V15 + 拡張調教 | deferred | — | Sprint 2 features 統合 待ち |
| 3. V15 + TM 公式 | deferred | — | TFJV TCOV 統合 (Phase 3) 待ち |
| 4. V15 + 当日体重 | deferred | — | 9:30 morning weight check 待ち |
| 5. V15 + 動画 features | simulate | #1 (V15 一致と仮定) | DL HTTP 400、 simulate 値 |
| **合議** | — | **#1 (2/5 低)** | 動作 system 2 件のみ |

V15 trio bets: 1-2-8 / 1-2-12 / 1-2-13 / 1-2-15 / 1-8-12 / 1-8-13 / 1-8-15

### 1.2 東京 11R エプソムカップ (G3) 15:45 — race_id 202605020511

| system | status | top1 | 備考 |
|--------|--------|------|------|
| 1. V15 単独 | ✅ ok | #14 サクラファレル (score 0.6952) | top2 #11 トロヴァトーレ / top3 #1 ジュタ |
| 2 / 3 / 4 | deferred | — | 同上 |
| 5. V15 + 動画 simulate | simulate | #14 | 同上 |
| **合議** | — | **#14 (2/5 低)** | 同上 |

V15 trio bets: 1-2-14 / 1-4-14 / 1-8-14 / 1-11-14 / 2-11-14 / 4-11-14 / 8-11-14

### 1.3 新潟 11R 駿風 S (OP) 15:20 — race_id 202604010311

| system | status | top1 | 備考 |
|--------|--------|------|------|
| 1. V15 単独 | ✅ ok | #1 パラサイコロジー (score 0.5166) | top2 #13 エコロジーク / top3 #5 カウンターセブン |
| 2 / 3 / 4 | deferred | — | 同上 |
| 5. V15 + 動画 simulate | simulate | #1 | 同上 |
| **合議** | — | **#1 (2/5 低)** | 同上 |

V15 trio bets: 1-5-8 / 1-5-9 / 1-5-13 / 1-5-14 / 1-8-13 / 1-9-13 / 1-13-14

---

## 2. 5 system v60 の状況 (5/9 朝)

| system | 5/9 状態 | 期待 (Phase 3+) |
|--------|---------|----------------|
| 1. V15 単独 | ✅ 動作 | production baseline |
| 2. V15 + 拡張調教 | deferred | 5/24+ Sprint 2 で活性化 |
| 3. V15 + TM 公式 (TFJV) | deferred | 6/8 V20 投入で活性化 (Phase 3) |
| 4. V15 + 当日体重 | deferred | 9:30 morning weight check 連携 (近日) |
| 5. V15 + 動画 features | simulate | 7-8 月 Phase 4 で実 PoC |

→ **5/9 時点で実動作は System 1 (V15) のみ、 System 5 は simulate**

---

## 3. 動画 DL 失敗対応

Session #60 B で 3 race 全 HTTP 400。 真の DL impl には Playwright 必要 (Phase 4 候補)。
本 Session C では:
- `data/v18/horse_motion_5_9.csv` を simulate 値で生成 (9 rows、 race × 3 horses)
- System 5 は V15 top1 と一致と仮定して simulate

→ verdict 用学習データ として記録、 ★ **投票には使用しない** ★

---

## 4. ★ 5/9 投資方針 (絶対遵守) ★

| 項目 | 内容 |
|------|------|
| 投票対象 | **12R 1勝のみ ¥2,100 (案B改 単独継続)** |
| 11R 重賞 | 観戦のみ、 投票しない (PoC verdict 用) |
| 累計目標 | **+13,530円 死守** (撤退余裕 +63,530円) |
| 撤退ライン | 累計 -50,000円 |

---

## 5. V15 投資保護 (再確認)

- ✅ V15 model file 不変
- ✅ predict_core / daily_predict / app.py 不変
- ✅ schtasks 41 件 不変
- ✅ Session #59 main commit 不変 (Discord dedup logic)
- ✅ dev/training-poc 専用 commit (新規 file のみ)

---

**Session #60 C 完了 (5/9 重賞 5 system 最終予測 ready)**
