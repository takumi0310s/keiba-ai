# Phase 21E A: JV-Link 動画 API 調査

> Session #88 (2026-05-11) Phase 21E A
> 結論: ★ JV-Link は data 取得 only、 動画 API 一切なし ★

---

## 1. 結論 (一行)

**JV-Link は競走 data (RA/SE/HR/O1...) 取得 専用 API、 動画 stream/download は仕様外。 V21 動画 source としては 0% 利用不可。**

---

## 2. 調査 source

| # | source | 確認結果 |
|---|--------|---------|
| 1 | [JV-Link 公式 forum](https://developer.jra-van.jp/c/jv-link/10) | 動画 / video / movie / mpeg / stream の topic 0 件 |
| 2 | JV-Link Interface Specification (Ver.4.9.0.1) | data 種別 RA/SE/HR/O1-O6/UM/SK/BR/DM... 全て tabular text data |
| 3 | [JRA-VAN SDK 一覧](https://jra-van.jp/dlb/sdv/sdk.html) | sample program 全て data parser、 動画関連 0 件 |
| 4 | JRA-VAN Wikipedia / 製品 list | 動画 service は **JRA-VAN NEXT (動画オプション)** + **JRA レーシングビュアー (RV)** の別 service |
| 5 | 既存 jvlink_fetcher.py 確認 | RA/SE/HR/O1/UM/SK のみ実装、 動画関連 method 一切なし |

---

## 3. JV-Link が扱う data 種別 (動画 NOT 含む)

```
RA  : レース詳細
SE  : 馬毎レース情報
HR  : 払戻
H1  : 票数
O1-O6: オッズ (単複/馬連/ワイド/馬単/三連複/三連単)
UM  : 馬マスタ
KS  : 騎手マスタ
CH  : 調教師マスタ
BR  : 繁殖馬マスタ
SK  : 産駒マスタ
HC  : 坂路調教
HS  : 木馬場調教
WF  : 重勝式
JG  : 競走除外
WE  : 天候馬場状態
WH  : 馬体重
DM  : デンマ (調教備考)
TCOV: コース変更
TK  : 特別登録
SLOP: 障害コース
```

→ **全て tabular ASCII / Shift-JIS data**。 binary stream 一切なし。

---

## 4. 公式 statement (探索結果から)

> 「動画配信 service は JRA-VAN NEXT モバイル / smartphone 版および JRAレーシングビュアー (JRA Racing Viewer) で提供」
> ([JRA-VAN Data Lab.開発者コミュニティ](https://developer.jra-van.jp/) より)

→ JV-Link API とは完全に分離された別 service。

---

## 5. 仮に JV-Link 経由 動画取得を試みる場合の risk

| risk | 詳細 |
|------|------|
| 仕様外利用 | 利用規約違反、 account ban 可能性 |
| 機能存在しない | 物理的に取得不可 (API method なし) |
| 工数 浪費 | 試行する value 0 |

→ **完全 NG。 試行禁止**。

---

## 6. V21 への影響

JV-Link は引き続き V20 の data 基盤として活用 (Phase 12 で 17 features 真値化中)。
動画 features (V21 +30) は **B (RV ソフト) または C (ブラウザ scraping)** で取得する必要あり。

---

## 7. 次 step

→ [phase21e_rv_software.md](phase21e_rv_software.md) (RV ソフト調査)
→ [phase21e_browser_scraping.md](phase21e_browser_scraping.md) (ブラウザ scraping 調査)
→ [phase21e_recommended_method.md](phase21e_recommended_method.md) (推奨 method)
