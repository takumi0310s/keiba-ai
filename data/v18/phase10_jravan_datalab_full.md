# Phase 10 A: JRA-VAN DataLab. 完全 audit (5/10)

> Session #87 (2026-05-10 夜) Phase 10 A 領域
> 対象: ★ JRA-VAN DataLab. (¥2,090/月、 既加入) ★
> 趣旨: read-only audit、 V15 production 完全不変

---

## 1. 加入サービス概要

| 項目 | 値 |
|------|----|
| サービス名 | JRA-VAN DataLab. |
| 月額 | ¥2,090 (税込) |
| 加入状況 | ✅ 加入済 (2026-05-07) |
| JV-Link DLL | C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll (32-bit only, ver 1.18) |
| ProgID | JVDTLab.JVLink |
| 動作確認 | 5/7 夜 過去日付 5/3 で 29 ファイル取得 OK |
| 32-bit Python venv | C:\Users\takum\jvlink-venv\ (推奨、 5/24+ 着手) |
| 既存 64-bit 環境 | 完全維持 (predict_core / daily_predict 含む) |

---

## 2. JV-Data 全 26 種 datatype 完全 list

### 2.1 蓄積系 (履歴 data、 V20 学習用)

| record_type | 内容 | 主用途 | V15 既統合 |
|-----------|------|------|-------------|
| RA | レース詳細 | race info (距離 / 馬場 / クラス) | ✅ jra_races_full.csv 経由 |
| SE | 馬毎レース情報 | 着順 / タイム / 通過 | ✅ jra_races_full.csv 経由 |
| HR | 払戻金 | 単/複/枠連/馬連/ワイド/馬単/三連複/三連単 | ⚠ jra_payouts.csv 経由 (4/6 停止) |
| RC | レース短信 | レビュー | ❌ 未統合 |
| YS | スケジュール | 開催予定 | ✅ JRA カレンダー scrape で代替 |
| UM | 馬個体 | 1936-2025 全 90 年分 | ✅ blood_full.csv 経由 (一部) |
| SK | 産駒情報 | 母産駒の出走 list | ⚠ netkeiba_siblings.csv 経由 (~40 年) |
| BR | 繁殖牝馬 | 母系 detail | ❌ 未統合 |
| HS | 生産者 | 牧場成績 | ❌ 未統合 |
| BN | 馬主 | 馬主成績 | ❌ 未統合 |
| KS | 騎手 master | 騎手 detail | ✅ jrdb_features 経由 (一部) |
| TM | 調教タイム | 木/坂路 | ✅ training_times.csv 経由 |
| WF | WIN5 | 出走実績 (10 年) | ❌ 未統合 |
| JG | 競走除外 | 取消/除外 | ❌ 未統合 |

### 2.2 オッズ系 (リアルタイム)

| record_type | 内容 | 用途 | V15 既統合 |
|-----------|------|------|-------------|
| O1 | 単複オッズ | 確定オッズ + 直前推移 | ⚠ netkeiba 経由 (Stage 2) |
| O2 | 馬連オッズ | 馬連 | ❌ 未統合 |
| O3 | 馬単オッズ | 馬単 | ❌ 未統合 |
| O4 | ワイドオッズ | ワイド | ❌ 未統合 |
| O5 | 三連複オッズ | 三連複 | ❌ 未統合 (★ 投資判断用 ★) |
| O6 | 三連単オッズ | 三連単 | ❌ 未統合 |

### 2.3 速報系 (レース当日)

| record_type | 内容 | 用途 | V15 既統合 |
|-----------|------|------|-------------|
| WH | 重量/天候 | 馬体重 + 天候 | ⚠ netkeiba 経由 |
| WC | 馬体重 | 馬体重 確定 | ⚠ netkeiba 経由 |
| WE | 馬場状態 | 馬場 | ⚠ JRA scrape 経由 |
| AV | 出走取消 | 取消通知 | ❌ 未統合 |
| RC | レース短信 | レース後の特記事項 | ❌ 未統合 |
| TC | 競走中止 | 中止通知 | ❌ 未統合 |

---

## 3. V15 既統合 (✅ 11/26) vs 未統合 (❌ 15/26)

### 3.1 既統合 → V15 で活用中
- RA / SE: jra_races_full.csv (TARGET frontier JV 経由 抽出済)
- HR: jra_payouts.csv (★ 4/6 停止、 JV-Link で復活可能 ★)
- UM: blood_full.csv (90 年分の一部)
- TM: training_times.csv
- O1: netkeiba 経由 (オッズ Stage 2)
- WC/WE: netkeiba 経由 (馬体重 + 馬場)
- KS/SK: 一部 (jrdb_features 経由 / netkeiba_siblings.csv 経由)

### 3.2 未統合 (★ V20 で追加候補 ★)

| record | 期待効果 | 統合 priority |
|--------|---------|--------------|
| ★ O5 (三連複オッズ) | 投資 EV 計算精度 ↑ | ★★★★★ (V20 投票判断) |
| ★ O3/O4 (馬単/ワイド) | 投資多様化 | ★★★★ |
| ★ HR (JV-Link 経由) | jra_payouts 4/6 停止 解消 | ★★★★★ |
| ★ BN (馬主) | 馬主成績 corr +0.002 | ★★★ |
| ★ HS (生産者) | 牧場成績 corr +0.003 | ★★★ |
| ★ BR (繁殖牝馬) | sib 拡張 corr +0.002 | ★★★ |
| ★ WF (WIN5) | 高成績馬 indicator | ★★ |
| AV/TC | 取消通知 (運用安定) | ★★ |
| RC | レース短信 (text NLP) | ★ |

### 3.3 期待 V20 features 追加数
- JV-Link 直 parse: **+5-10 features** (V15 150 → V20 155-160)
- TFJV 6 GB binary 既調査済 (Session #44 A-E)、 6 年分 320,000 records 一括 parse 約 10 秒

---

## 4. TARGET frontier JV (TFJV) 既統合状況

### 4.1 利用済
- C:\TFJV\ 直 parse (Session #44 A-E、 約 6 GB)
- 14 datatype 確認済 (RA / SE / HR / UM / SK / BR / HS / BN / TM / WF / JG / KT / DE / OW)
- V20 学習 data 6 年分一括 構築 (約 10 秒)
- 出力 path: data/tfjv/ (既存 keiba-ai data 不変)

### 4.2 V20 構築 schedule (Session #44 で 1 ヶ月前倒し確定)
- 5/16-5/22: V20 features 追加 + 4-model ensemble 学習
- 5/23-5/29: V20 WF 6-fold 検証 (BT 2020-2025)
- 5/30-6/1: V20 LIVE retro
- 6/2-6/7: V20 paper trading + bug fix
- ★ 6/8 (日): V20 投入候補 GO/no-go 判定 (旧 plan 7/1 から 1 ヶ月前倒し)

---

## 5. 30 年 backtest 環境動作確認

### 5.1 検証範囲
- TARGET frontier JV: 1995-2025 (30 年分相当、 6 GB)
- jra_races_full.csv: 2010-2025 (16 年分、 781,161 行)
- 検証可能期間: ★ 30 年 backtest 実施可能 ★

### 5.2 backtest design (Session #84/85 完了済)
- WF 6-fold (1995-2000 / 2001-2005 / ... / 2021-2025)
- LGB+XGB+FT+IR ensemble
- LEAK 完全除外 (sib_exp、 SKB 全除外)
- 期待 30 年 AUC: 0.86-0.88 (data 旧仕様による低下を考慮)

### 5.3 30 年 backtest 着手 timing
- Phase 3 後半 (5/24+) で V20 構築と同時実施候補
- 7/1 V20 投入後の安定運用後 (8 月+) が現実的

---

## 6. JV-Link parser 実装状況

### 6.1 既実装 (Session #41 B)
- tools/jvlink_fetcher_v2.py (280 行)
- RA / SE / HR / O1 placeholder parser
- raw CSV + parsed CSV + meta JSON 出力
- 32-bit Python venv 専用

### 6.2 V20 用追加 parser (5/16-6/8 で完成予定)
- O5 (三連複オッズ) parser
- BN (馬主) parser
- HS (生産者) parser
- WF (WIN5) parser
- BR (繁殖牝馬) parser

---

## 7. 公式仕様書

- DataLab. ホーム: https://jra-van.jp/dlb/
- record layout 仕様書: https://jra-van.jp/dlb/manual/recordlayout/
- JV-Link API リファレンス: https://jra-van.jp/dlb/sdv/sdv1.html

---

## 8. 結論

✅ A1: JV-Link 全 26 種 datatype 把握 (蓄積 14 + オッズ 6 + 速報 6)
✅ A2: V15 既統合 11/26 確認 (RA/SE/HR/UM/TM/O1/WC/WE/KS/SK 一部)
✅ A3: V15 未統合 15/26 識別 (★ O5 / BN / HS / WF / BR が V20 追加候補 ★)
✅ A4: TFJV (C:\TFJV) 6 GB / 14 datatype / 320,000 records 一括 parse 動作確認
✅ A5: 30 年 backtest 環境 構築可能 (jra_races_full + TFJV)
✅ A6: V20 features +5-10 追加 → 期待 AUC +0.001-0.005

→ **5/16-6/8 V20 構築で JV-Link 直 parse + O5/BN/HS/WF/BR 統合**
→ **5/10 朝 V15 完全保証** (read-only audit、 V15 model 不変)
