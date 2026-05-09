# Session #63 A: 動画代替 全馬総合スコア features 設計

**作成**: 2026-05-09 11:XX (Session #63 A、 dev/training-poc)

## 背景

Session #62 で動画 DL は **完全 server-side block** 確定:
- yt-dlp / curl / requests / Playwright real Chromium 全 client BLOCK
- JRA-VAN DataLab (JV-Link): data feed のみ、 動画なし
- JRDB: 数値 + text のみ、 動画なし

ユーザー要望「JRDB or jravan で取れない? なさそうなら 写真から全馬スコア出して全馬に順位付け」
→ **静止画 (パドック写真) + JRDB 数値 features** で全馬総合スコア + ranking 構築。

## features 一覧 + 重み

| # | feature | source | weight | 説明 |
|---|---------|--------|--------|------|
| 1 | paddock_idx | JRDB TYB padock_idx | **0.30** ★ | 動画代替の主軸。 ◎/○/▲/△ 数値化 |
| 2 | training_idx | JRDB KYI train_idx | 0.20 | 調教指数 |
| 3 | idm_score | JRDB KYI idm | 0.15 | IDM (基礎能力) |
| 4 | body_size_relative | 静止画 YOLOv8 (同 R 内 percentile) | 0.15 | 体格 (bbox area の同 R 内順位) |
| 5 | pose_score | 静止画 YOLOv8 (bbox 縦横比) | 0.10 | 立ち姿の安定性 |
| 6 | coat_score | 静止画 (色 saturation 平均) | 0.10 | 毛艶、 健康度 |
| 7 | weight_change | JV-Link SE / JRDB TYB weight_diff | optional +0.10 | 体調変化 |

## スコア計算

```
integrated_score = (
    0.30 * paddock_norm
    + 0.20 * training_norm
    + 0.15 * idm_norm
    + 0.15 * body_size_norm
    + 0.10 * pose_norm
    + 0.10 * coat_norm
)
+ 0.10 * weight_change_norm  (取得可能なら、 重み再正規化)
```

各 feature は同 R 内 min-max 正規化 (0-1)。 NaN は中央値 0.5 で fill。

confidence:
- **high**: 静止画 features 全 + 数値 features 全
- **mid**: 数値 features のみ (静止画 NG)
- **low**: 一部のみ (NaN 多い)

## 利用想定

- **観戦 / verdict 用**: 全 R 全馬で順位確認
- **V20 (6/8) 構築時**: features 統合候補 (動画 dependency なし)
- **9月 V21 投入**: 動画なしで実装可能

## 5/9 対象 race

| race_id | 場 | R | race_name | grade | 頭数 | 投票 |
|---------|----|---|-----------|-------|------|------|
| 202608030511 | 京都 | 11 | 京都新聞杯 | G2 | 16 | × verdict |
| 202608030512 | 京都 | 12 | 4歳以上2勝クラス | - | 13 | × verdict (案B改 除外) |
| 202604010311 | 新潟 | 11 | 駿風 S | OP | 16 | × verdict |
| 202604010312 | 新潟 | 12 | 4歳以上1勝クラス | - | 12 | **★ V15 ¥700 投票対象 ★** |
| 202605020511 | 東京 | 11 | エプソムカップ | G3 | 17 | × verdict |
| 202605020512 | 東京 | 12 | 4歳以上2勝クラス | - | 16 | × verdict (案B改 除外) |

## JRDB data 状況 (5/9 11:XX 時点)

- TYB: latest **5/2** (TYB260502.txt) — 5/9 file 未取得
- CYB: latest **5/3** (CYB260503.txt)
- KYI: latest **5/3** (KYI260503.txt)

→ 5/9 当日 JRDB feed 未配信。 **latest 利用 + 5/9 race_id 該当行なし → 全馬 NaN fallback** が想定される。

→ commit "Session #63 A: features 設計"
