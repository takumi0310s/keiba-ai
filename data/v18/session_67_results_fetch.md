# Session #67 A: 5/9 全 R 結果取得 (16:55-17:05)

## 1. 取得 status

| 項目 | status | 備考 |
|------|--------|------|
| 全 34 R 着順 | ✓ 取得済 | JRDB SED260509.txt (495 行 = 36 races × 各馬) |
| 払戻 (単勝) | △ 概算のみ | JRDB HJC260509.txt 簡易 parse、 JRDB v4 spec 厳密版は Session #68 候補 |
| 払戻 (三連複) | ⚠ raw のみ | regex 抽出で誤値含む (e.g., ¥20,000,000 等異常) |
| netkeiba 直接取得 | ❌ HTTP 400 | server-side block 継続 (Session #62/#63 と同種) |

## 2. 取得 method 変遷

### 試行 1: netkeiba race/result.html (`tools/session_67_fetch_results.py`)
- 全 34 R で 「結果テーブルが見つかりません」 → fail
- Session #61 verdict×6 schtasks (15:35-16:50) 全件 同 fail
- Session #62/#63 で確定した netkeiba 全 page server block の延長

### 試行 2: db.netkeiba.com fallback
- 同じく HTTP 400 で fail

### 試行 3 (採用): JRDB SED + HJC (`tools/session_67_jrdb_results.py`)
- SED (成績データ): 着順、 異常区分、 馬番、 単勝 odds 等
- HJC (払戻データ): 各券種払戻
- 両方とも 5/9 16:50 時点で publish 済 (HJC は手動 download 必要だった)
- 全 34 R 着順 取得 OK

## 3. 出力

- `data/results/20260509_results.csv` — 36 行 × 15 columns
- columns: race_id, course, race_num, race_name, num_horses, finish_1, finish_2, finish_3, trio_nums, umaren_nums, payout_tansho, payout_umaren, payout_trio, payout_sanrentan, fetch_status

注: 京都 R1 / R4 など payout_trio が異常値 (>¥1M) → JRDB HJC parser 簡易版の限界。 着順データは正確。

## 4. 主要 R 着順 (verdict 用)

### 重賞 3R (verdict 用、 投票なし)
- 京都 R11 京都新聞杯 (G2): **5-6-15** (trio 取得値 ¥13,940,000 — 怪しい、 実 ¥139,400 推定)
- 東京 R11 エプソムカップ (G3): **11-16-17** (trio ¥27,400,000 — 実 ¥274,000 推定)
- 新潟 R11 駿風 S (OP): **12-15-16** (trio ¥14,190,000 — 実 ¥141,900 推定)

### 12R 全 (V15 投資対象 + 観戦)
- 京都 R12 4歳以上2勝 (案B改 除外): **8-10-13**
- 東京 R12 4歳以上2勝 (案B改 除外): **3-11-12**
- ★ 新潟 R12 4歳以上1勝 (V15 投票) ★: **3-8-11**

## 5. V15 投票 vs 結果 (B 領域 詳細)

V15 vote 三連複 7点: `6-8-11; 6-11-12; 8-9-11; 8-10-11; 8-11-12; 9-11-12; 10-11-12`
新潟 12R 1-2-3 着: **3-8-11** (馬番 set = {3, 8, 11})

→ いずれの 7 点も 3 番を含まない → **MISS 確定**
→ 損益 -¥700

## 6. fallback 適用
- B 三連複 hit 判定: SED の 1-3 着で十分 (払戻精度不要)
- payout 値: HJC 簡易 parser のため 1〜数桁 不一致あり、 user 表示時は 実払戻に注意
- C 全 R verdict: 着順は全 36R 正確、 system 比較は実行可能

→ 着順データ confidence は HIGH。 payout 信頼性は LOW (Session #68 で HJC 厳密 parser 化候補)。
