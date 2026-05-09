# 5/9 (土) 1 day summary (Session #67、 17:10 時点)

## 1. V15 投資結果 (案B改 strict)

| 項目 | 値 |
|------|----|
| 投票 R | 1 (新潟 12R 4歳以上1勝) |
| 軸馬 | 11 ハイクオリティ (V15 score 0.6483) |
| 三連複 7点 | `6-8-11; 6-11-12; 8-9-11; 8-10-11; 8-11-12; 9-11-12; 10-11-12` |
| 投資 | ¥700 |
| 結果 (1-2-3 着) | **3-8-11** (馬番 set {3, 8, 11}) |
| 軸馬 着順 | 11 番 → **3 着** |
| trio hit | ❌ MISS (全 7 点 が 3 番 を含まず) |
| 払戻 | ¥0 |
| 損益 | **-¥700** |
| 5/3 開始累計 | +¥13,530 |
| **5/9 終了 累計** | **+¥12,830** |
| 撤退余裕 (vs -¥50,000) | **+¥62,830** |

## 2. 重賞 verdict (3R、 観戦のみ)

| race | 1-2-3 着 | V15 top1 | V15 trio hit | 仮 ROI (¥700/R) |
|------|----------|----------|--------------|------------------|
| 京都新聞杯 G2 | 5-6-15 | 1 アーレムアレス | ❌ | -¥700 |
| エプソムカップ G3 | 11-16-17 | 14 サクラファレル | ❌ (top2/top3 of 1着・3着 = top3 overlap 2 馬!) | -¥700 |
| 駿風 S OP | 12-15-16 | 1 パラサイコロジー | ❌ | -¥700 |

3 R 仮投資 ¥2,100 / 払戻 ¥0 / **仮 ROI 0%** → 案B改 が 重賞除外したのは結果論的に正解。

## 3. 12R 全 (3 場)

| race | 1-2-3 着 | V15 top1 | trio hit |
|------|----------|----------|----------|
| 京都 12R 4歳以上 2勝 (除外) | 8-10-13 | 8 ロードヴォイジャー | ❌ |
| 東京 12R 4歳以上 2勝 (除外) | 3-11-12 | 11 フィドルファドル | ❌ |
| ★ 新潟 12R 4歳以上 1勝 (V15 投票★) | 3-8-11 | 11 ハイクオリティ | ❌ |

→ 12R 3R 全 miss、 1勝/2勝 関係なく V15 trio frame に当てはまらない結果。 案B改 strict (1勝のみ) が他 2 R 投票回避したのは結果論で同じ (機会損失なし)。

## 4. システム比較 (本日 全 34 R 集計)

| system | top1 hit | top3 overlap (any) | trio 7点 hit | 仮 ROI |
|--------|----------|--------------------|--------------|---------|
| **V15 朝予測** | **13/34 (38.2%)** | 31/34 (91.2%) | 9/34 (26.5%) | (要計算) |
| Stage 2 1h 前 | (取得 fail) | — | — | — |
| 動画 v66 NO_TYB | 2/6 (33.3%) | — | — | — |

注: Stage 2 は Session #65 watchdog から fire したが netkeiba 出馬表 fetch fail、 Session #68 C で修復済。 5/16 から 実 Stage 2 評価可能。

## 5. Stage 2 system 評価

- Session #65 watchdog 13:30 開始、 30 分毎 fire
- 取得 success: **0 R** (netkeiba HTTP 400 server block 影響、 出馬表 fetch fail)
- Session #68 で root cause 特定 + `tools/stage2_predict.py` 修復 (block 検知 + Stage 1 fallback 明示)
- 5/10 朝 watchdog 健全 fire 想定、 翌週末 5/16 で実評価

## 6. 動画代替 v66 評価 (NO_TYB mode、 90 馬 cover)

- 静止画 DL は netkeiba server block (Session #62/63 確定)
- JRDB 数値 only (training_idx + IDM + gekiso + stable + ninki) で fallback
- top1 hit 2/6 (33.3%) — V15 と同水準だが N=6 で統計的有意性 LOW
- TYB publish 後 (paddock 0.30 重み版) は本日未実施 (Session #66 A で 13:08 まで 4 回 fail、 publish 遅延)
- 5/16 で TYB 取得 後 paddock 重み付き再評価予定

## 7. 5/16 V18 trial 含意

| 観点 | 結論 |
|------|------|
| V15 案B改 strict 安定性 | 1 day max -¥700 (撤退余裕 1.1% のみ消費) ✓ |
| V15 朝予測 強さ | top1 38% / top3 overlap 91% — production 健全 ✓ |
| Stage 2 system | 5/9 fire 0 success、 Session #68 修復済、 5/16 で実評価 |
| 動画 v66 | sample 不足 (N=6)、 TYB publish 後 評価必要 |
| **5/16 V18 trial 判定** | NO-GO 維持 (Session #38 sib_top3_rate hybrid 確定)、 sib_*_exp 修正版で 6/15+ 再判定 |

## 8. 撤退余裕

| 項目 | 円 |
|------|-----|
| 撤退ライン | -50,000 |
| 5/9 終了 累計 | +12,830 |
| **撤退余裕** | **+62,830** |
| 5/16 max loss 想定 | -¥2,100 (新潟 12R + 案B改 上限) |
| → 5/16 後 撤退余裕 (worst case) | +¥60,730 |

## 9. 並行 Session 含意

5/9 当日 並行進行した sessions の status:
- **Session #61** (12:30): realtime_5_9.py + 9 件 schtasks ✓ (vote_candidates 14:00 動作 OK、 verdict×6 は netkeiba 取得 fail で empty)
- **Session #63** (12:45): 動画代替 全馬総合スコア (NO_TYB 90 馬) ✓
- **Session #64** (12:50): ProcessWatchdog spam bug 修正 + kill-switch ✓ (main +1 commit)
- **Session #65** (12:50): Stage 2 1h 前予測 watchdog ✓ (ただし 5/9 当日 fire は fetch fail)
- **Session #66** (13:00 〜 dev/training-poc 並行): TYB retry + paddock 統合 (publish 遅延で NO_TYB fallback)
- **Session #67** (16:55 〜): 本セッション、 全 R verdict + 1 day summary
- **Session #68** (並行): stage2_predict 修復 (5/10 朝 動作する状態)

## 10. 次 step (5/10 朝)

- [ ] daily_predict 8:00 fire 確認 (Session #64 watchdog 再発防止)
- [ ] morning_weight_check 9:30 fire 確認
- [ ] Stage 2 system 5/10 動作 確認 (Session #68 修復版)
- [ ] TYB publish 確認 (5/9 13:00 想定だったが遅延、 5/10 朝に publish 済か再 retry)
- [ ] 5/10 V15 投票判定 (12R 1勝 該当 R 確認)
