# Session #63 B: 静止画 DL 結果 (★ 失敗 確定 ★)

**作成**: 2026-05-09 11:XX (Session #63 B、 dev/training-poc)

## 結果: 全 fail (netkeiba 全 page block)

```
=== 京都 R11 京都新聞杯 (202608030511) ===  shutuba fetch failed (HTTP 400)
=== 京都 R12 4歳以上2勝クラス (202608030512) === shutuba fetch failed (HTTP 400)
=== 東京 R11 エプソムC (202605020511) === shutuba fetch failed (HTTP 400)
=== 東京 R12 4歳以上2勝クラス (202605020512) === shutuba fetch failed (HTTP 400)
=== 新潟 R11 駿風 S (202604010311) === shutuba fetch failed (HTTP 400)
=== 新潟 R12 4歳以上1勝クラス (202604010312) === shutuba fetch failed (HTTP 400)
合計: OK=0, FAIL=0 (fetch 自体不可)
```

## 診断

```python
# race.netkeiba.com/race/shutuba.html → HTTP 400 / len 0
# db.netkeiba.com/race/<id>/         → HTTP 400 / len 0
# db.netkeiba.com/horse/<id>/        → HTTP 400 / len 0
```

→ Session #62 で確認した動画 DL block と **同一 server-side block**。 netkeiba は
race.netkeiba.com / db.netkeiba.com への requests / curl client を全て 400 で拒否。
Cookie あり/なし、 Referer 設定、 User-Agent 詐称 全て無効。

## 結論

**5/9 静止画 DL 不可**。 Session #63 C (YOLOv8 features) は実行不能、
全馬 NaN 化。 Session #63 E (全馬統合スコア) は **数値 features (JRDB) のみ** で
構築する。 confidence は全馬 'mid' (静止画なし、 数値のみ)。

## fallback (実装済)

- C YOLOv8 → 全馬 body_size/pose/coat = NaN
- E 統合スコア → JRDB paddock_idx + training_idx + idm + (weight_diff) のみ
- 重み再正規化: paddock 0.40 / training 0.30 / idm 0.20 / weight 0.10

## 5/16 (土) 以降 の対策候補

1. **netkeiba server 復旧 待ち** (1 週間後 再試行)
2. **JRA 公式 DataLab (JV-Link) 動画 feed 加入** (5/24 計画)
3. **ユーザー manual download** (Premium account ブラウザで右クリック保存)
4. **Phase 4 Playwright stealth mode 強化** (7-8 月 PoC)

→ commit "Session #63 B: 静止画 DL"
