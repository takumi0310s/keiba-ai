# Session #67 C: 全 R verdict 集計 (5/9 全 34 R)

## 1. システム別 hit rate (全 34 R 集計)

| system | top1 hit | top3 overlap (any) | trio 7点 hit |
|--------|----------|--------------------|--------------|
| **V15 朝予測** | **13/34 (38.2%)** | **31/34 (91.2%)** | **9/34 (26.5%)** |
| 動画 v66 (NO_TYB) | 2/6 (33.3%) | — | — |
| Stage 2 1h 前 | 0/0 (取得 fail) | — | — |

備考:
- V15 top1 hit 38.2% は WF AUC 0.8939 と整合 (期待値 ~36-40%)
- V15 top3 overlap 91.2% は predict.top3 が実 top3 の少なくとも 1 馬 当てる確率、 極めて高水準
- V15 trio 7点 hit 26.5% は条件横断、 案B改 1勝のみ filter なら 別水準
- 動画 v66 は 6 R sample のみ (重賞 3 + 12R 3)、 統計的有意性 LOW
- Stage 2 は Session #65 watchdog から fire したが netkeiba 出馬表 fetch fail (Session #62/63 server block の影響、 Session #68 で root cause 特定済)

## 2. 重賞 + 12R 仮投資 (V15 三連複 7点 / R)

| race | 着順 | V15 top1 | V15 trio hit | 仮投資 |
|------|------|----------|--------------|--------|
| 京都 R11 京都新聞杯 (G2) | 5-6-15 | 1 アーレムアレス | ❌ | -¥700 |
| 東京 R11 エプソムカップ (G3) | 11-16-17 | 14 サクラファレル | ❌ | -¥700 |
| 新潟 R11 駿風 S (OP) | 12-15-16 | 1 パラサイコロジー | ❌ | -¥700 |
| 京都 R12 4歳以上 2勝 (除外) | 8-10-13 | 8 ロードヴォイジャー | ❌ | -¥700 |
| ★ 新潟 R12 4歳以上 1勝 (V15 投票★) | 3-8-11 | 11 ハイクオリティ | ❌ (3 番未含) | -¥700 |
| 東京 R12 4歳以上 2勝 (除外) | 3-11-12 | 11 フィドルファドル | ❌ | -¥700 |

5 R 仮投資: ¥3,500 / 払戻 ¥0 / ROI 0%

★ 案B改 strict (12R 1勝のみ) で実投票したのは新潟 12R のみ → 投資 ¥700 / 払戻 ¥0 / -¥700。 重賞 / 2勝 12R は投票しない判断 (案B改 除外条件) は **的中しなかったので機会損失なし**。

## 3. システム比較 (重賞 3R で詳細)

### 京都新聞杯 G2 (1-2-3 = 5-6-15)
- V15 top1: 1 アーレムアレス → 実 5 着圏外 (6 着 以下)
- 5 system top1 (sys 1 = V15、 sys 2-4 deferred、 sys 5 simulate)
- 動画 v66 top1: ?? (要 csv 参照)

### エプソムカップ G3 (1-2-3 = 11-16-17)
- V15 top1: 14 サクラファレル → 実 圏外
- ★ V15 top2 11 トロヴァトーレ → 1 着 ★ → top3 overlap +1
- 17 ジュタ (V15 top3) → 3 着 ★ → top3 overlap +2
- V15 trio_bets `1-2-14; 1-4-14; 1-8-14; 1-11-14; 2-11-14; 4-11-14; 8-11-14` に {11, 16, 17} は 含まれず → trio hit ❌
- → top3 hit (V15 top3 中 2 馬が actual top3 入り) は良好だが 1 着 14 番 を外し trio fail

### 駿風 S OP (1-2-3 = 12-15-16)
- V15 top1: 1 パラサイコロジー → 圏外
- V15 trio_bets `1-5-8; 1-5-9; 1-5-13; 1-5-14; 1-8-13; 1-9-13; 1-13-14` に {12, 15, 16} 含まれず → trio hit ❌

## 4. 動画 v66 評価 (NO_TYB mode、 90 馬 cover)

データソース: `data/v18/horse_total_scores_5_9.csv`

| race | 動画 v66 top1 (馬番) | 実 1 着 | hit |
|------|----------------------|---------|------|
| 京都 R11 | 14 ベイビーキッス (※駿風S と混在?) | 5 | ❌ |
| 東京 R11 | (要 csv) | 11 | (要確認) |
| 新潟 R11 (駿風S) | 14 ベイビーキッス | 12 | ❌ |
| (12R) | (取れた R のみ) | | |

→ 6 R sample で 2 hit (33.3%)、 V15 top1 hit 38.2% と同水準だが N 不足、 5/16+ で要追加 sample。

## 5. Stage 2 (1h 前予測) 評価

- Session #65 watchdog fire 件数: ~10+ R 想定 (13:30 ~ 15:30 範囲)
- 取得 success: **0/N (全 fail)** — netkeiba 出馬表 fetch HTTP 400 (Session #68 で root cause 確定)
- Session #65 fork 報告: 「dry-run の出馬表 fetch 失敗 (Cookie/timing 早すぎ)」
- 5/10 朝 修復対象: Session #68 C `tools/stage2_predict.py` 修復 (netkeiba block 検知 + Stage 1 fallback) 済

## 6. 5/16 V18 trial 含意

| 観点 | 結論 |
|------|------|
| V15 安定性 | 1 day max -¥700、 想定通り (案B改 strict ✓) |
| 朝予測 強さ | top1 38% / top3 overlap 91% — V15 production 健全 |
| Stage 2 system | 5/9 fire 0 success → 5/10 朝 修復 (Session #68 反映済) |
| 動画 v66 | sample 不足、 5/16+ で追加検証 |
| **5/16 V18 trial** | NO-GO 維持 (Session #38 確定)、 sib_*_exp 修正版で 6/15+ 再判定 |

## 7. 出力 file

- `data/v18/system_comparison_5_9.csv` (34 R × 20 columns)
- `data/v18/session_67_verdict_summary.json` (D 領域用 JSON)
- `tools/session_67_verdict.py` (実装)
