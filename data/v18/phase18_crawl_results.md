# Phase 18 B: netkeiba マスター 過去 backfill (controlled)

**作成**: 2026-05-10 (Session #91 Phase 18 B、 ★ Opus 4.7 ★)
**前提**: Phase 13 「scope 限定: 当日開催 R のみ、 過去 backfill しない」を 限定的に拡張
**目的**: BAN risk を抑えた controlled backfill 基盤を整備、 過去 5 年 16,000 R を 16 日で取得可能にする

---

## 1. tools/netkeiba_master_backfill.py (新規)

### 1.1 規約 / safety 概要

| safety | 実装 |
|--------|------|
| 明示同意 flag | `--i-accept-tos-risk` 指定無し → 即 refuse |
| 1 セッション上限 | `--max-races` (デフォルト 100、 hard cap 5,000) |
| 24h 累計上限 | `--max-daily-quota` (デフォルト 1,000) |
| kill switch | `data/netkeiba_master/.disabled` 存在で全 fetch skip |
| Cookie 検証 | `.env NETKEIBA_COOKIE` 未設定で refuse |
| rate limit | 既存 scraper 3 sec interval 継承 (1 R = 12 sec、 4 page) |
| checkpoint | `data/netkeiba_master/backfill_progress.csv` に append |
| resume | `--resume` で seen race_id を skip |
| daily quota | `data/netkeiba_master/backfill_quota.json` で 24h 累計 trace |

### 1.2 race_id 列挙

`tools/netkeiba_master_backfill.py:discover_race_ids()` は
`data/jra_races_full.csv` の year/month/day から start_date〜end_date を抽出。
714 R / 1 週間 (2024 年 1 月 1-7 日) を確認済 (smoke test)。

外部リスト経由も可: `--race-ids-file path` (1 行 1 race_id)

### 1.3 使い方

```bash
# Step 1: dry-run で race_id 列挙
python tools/netkeiba_master_backfill.py \
    --i-accept-tos-risk \
    --start-date 20240101 --end-date 20240107 \
    --dry-run

# Step 2: 100 R 試行 (1 R 12 sec → 約 20 分)
python tools/netkeiba_master_backfill.py \
    --i-accept-tos-risk \
    --start-date 20240101 --end-date 20240107 \
    --max-races 100

# Step 3: 1 日 1000 R 上限で 5 年 backfill (16 日) ※段階的に
python tools/netkeiba_master_backfill.py \
    --i-accept-tos-risk \
    --start-date 20210101 --end-date 20260510 \
    --max-races 5000 --max-daily-quota 1000 \
    --resume
```

### 1.4 出力

```
data/netkeiba_master/
├── .disabled                          (kill switch、 touch で即停止)
├── backfill_progress.csv              (race_id, fetched_at, status, note)
├── backfill_quota.json                (date, count、 24h リセット)
└── backfill/
    ├── 2021/
    │   ├── 0521010101.json.gz
    │   ├── 0521010102.json.gz
    │   └── ...
    ├── 2022/
    └── ...
```

各 json.gz: `{race_id, umaban, fetched_at, fetch_status, features}` (25 features)

---

## 2. 規模見積 (rate limit ベース)

| scope | races | 時間 (1 R = 12 sec) | 備考 |
|-------|-------|---------------------|------|
| 当日 35 R | 35 | 7 min | Phase 13 既存 |
| 直近 1 週間 (約 240 R) | 240 | 48 min | smoke test 範囲 |
| 直近 1 ヶ月 (約 1,000 R) | 1,000 | 3.3 h | 1 日 quota 同等 |
| 直近 6 ヶ月 (約 6,000 R) | 6,000 | 20 h | quota 1000/day で 6 日 |
| 直近 1 年 (約 12,000 R) | 12,000 | 40 h | quota 1000/day で 12 日 |
| 直近 5 年 (約 60,000 R) | 60,000 | 200 h | quota 1000/day で **60 日** |
| 過去 24 年 (約 200,000 R) | 200,000 | 667 h | quota 1000/day で **200 日** |

→ 「過去 5 年 16,000 R を 16 日」 ユーザー想定は quota 1,000/day かつ
  rate limit 3 sec で **2 ヶ月相当**、 想定よりかなり長い。 1 ヶ月 (6,000 R)
  程度が現実的初期 scope。

→ ★ 24 年 200,000 R full backfill は **本 script で 実行非推奨** ★ —
  BAN risk + Cookie 期限切れ + 帯域制約で 200 日連続取得は破綻の可能性大

---

## 3. ★★★ 重要警告: 24 年 backfill 実施しない方針 ★★★

### 3.1 想定 risk

| risk | 影響 |
|------|------|
| **netkeiba BAN** | Cookie / アカウント無効化 → V15 production も停止 (cookie 共有のため) |
| **rate limit 強化** | 3 sec interval で不足 → 大幅低速化 or 502 連発 |
| **規約改訂** | 個人利用 自動 access 明示禁止 → 即停止必要 |
| **Cookie 期限切れ** | 200 日連続では数十回 refresh 必要、 自動更新も限界 |
| **帯域 / 電気代** | 200 日 24h 連続稼働、 マシン障害 risk |

### 3.2 推奨 scope (現実的)

| 期間 | 想定取得 | quota | 日数 |
|------|---------|-------|------|
| Phase 18 B (本 script) | 直近 6 ヶ月 6,000 R | 1,000/day | 6 日 |
| Phase 18 B 拡張 (V20 学習用) | 直近 12 ヶ月 12,000 R | 1,000/day | 12 日 |
| Phase 18 B 究極 (BAN risk 受容) | 直近 24 ヶ月 24,000 R | 1,000/day | 24 日 |

過去 24 年 200,000 R は **DataLab + JRDB + TFJV からデータ補完** が現実的。
netkeiba マスター は 直近データの「真値補完」用と割り切る。

---

## 4. 本 session で実施しないこと

| 項目 | 理由 |
|------|------|
| 実 fetch (1 R 含め) | DOM probe は別 session で user 同伴の元 実行推奨 |
| backfill 起動 | selector 真値化前に大量 fetch すると 25 feature が全 default で無価値 |
| 24 年 backfill | 上記 risk + 200 日 連続稼働 不現実 |

→ 本 session は **基盤整備のみ** (script + safety gate + doc)

---

## 5. 5/11+ 推奨 sequence

```
5/11   DOM probe で selector 真値化 (Phase 18 A)
5/12   parser 反映 (commit)
5/13   1 R fetch 試行 → 25 features 取得率確認
5/14   1 日分 (35 R) fetch 検証
5/15   1 週間 (240 R) fetch 検証
5/16-5/22  1 ヶ月 (6,000 R) backfill (1 日 1,000 R quota、 6 日)
5/23-6/15  6 ヶ月 (6,000 R 追加) backfill (V20 学習データ用)
5/24+  V20 4-model ensemble 学習着手 (TFJV + Phase 11/12/13 + master)
```

---

## 6. V15 投資保護 (絶対遵守)

✅ backfill script は predict_core / daily_predict / app.py 完全不変
✅ kill switch 1 ファイル touch で即停止可能
✅ 24h quota で暴走防止 (default 1,000/day)
✅ session 上限 5,000 R で長時間プロセス避ける
✅ V15 cookie BAN 顕在時は累計収支 +¥14,140 維持のために 即 fetch 停止

---

## 7. 結論

✅ tools/netkeiba_master_backfill.py 新規 (controlled、 6 重 safety gate)
✅ dry-run 確認済 (2024 年 1 週間 = 714 R 列挙)
✅ V15 投資保護完全
⚠ 24 年 full backfill は実施しない方針 (BAN risk 過大)
⚠ 推奨初期 scope: 直近 6 ヶ月 6,000 R / 6 日

---

**Phase 18 B 完了** (Opus 4.7)
