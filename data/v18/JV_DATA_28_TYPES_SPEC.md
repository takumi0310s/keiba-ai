# JV-Data 28 種 datatypes 仕様 (JRA-VAN DataLab、 V20 構築 用)

参考: JRA-VAN 公式 spec https://jra-van.jp/dlb/manual/recordlayout/

## 主要 datatypes (V20 構築 必須)

### 蓄積系 (option=1 / option=4 で 取得)

| record | 名称 | バイト長 | 主用途 | V15 利用 | V20 で活用 |
|--------|------|---------|--------|---------|----------|
| **RA** | レース詳細 | 1270 | コース / 距離 / 馬場 / 賞金 / 重賞番号 | ✅ | ✅ |
| **SE** | 馬毎レース情報 | 553 | 着順 / 走破タイム / 上がり 3F / 通過順位 | △ TFJV | ✅ 真値化 |
| **HR** | 払戻 | 1148 | 全 8 券種 配当 | △ TFJV | ✅ jra_payouts 代替 |
| **H1** | 単複オッズ確定 | 多 | 単勝 / 複勝 確定 オッズ | ❌ | ✅ |
| **H6** | 三連単 オッズ | 多 | 三連単 推移 | ❌ | ✅ |
| **WH** | 馬体重 | 70 | 当日朝 確定 体重 | △ netkeiba | ✅ 真値化 |
| **WE** | 天候・馬場 | 多 | 風速 / 気温 / 湿度 / 馬場状態 詳細 | △ 気象庁 | ✅ 真値化 |
| **AV** | 出走馬予定 | 多 | 翌週 出馬予定 (TOKU) | ❌ | ✅ TK 代替 |
| **JC** | 騎手変更 | 多 | 直前 騎手 変更 | ❌ | ✅ LIVE alert |
| **TC** | 調教師移動 | 多 | 移籍 / 開業 / 廃業 | ❌ | ✅ 厩舎期 feature |
| **CS** | コース情報 | 多 | 各 場 コース detail | ❌ | ✅ course_renovated 補強 |
| **UM** | 馬個体 master | 多 | 馬主 / 生産者 / 出生地 / 父 / 母 / bms | △ TFJV | ✅ jrdb_ukc 代替 |
| **HS** | 騎手 master | 多 | 騎手 詳細 (生年月日、 所属、 etc.) | ❌ | ✅ |
| **HN** | 調教師 master | 多 | 調教師 詳細 | ❌ | ✅ |
| **HC** | 馬主 master | 多 | 馬主 詳細 | ❌ | ✅ |
| **BR** | 生産者 master | 多 | 生産者 詳細 | ❌ | ✅ |
| **CK** | 系統 master | 多 | 系統 (SK 代替) | ❌ | ✅ |
| **BT** | 血統 5 代 | 多 | inbreeding 5代 (PEDE) | ❌ | ✅ ★ 5代 inbreeding ★ |

### オッズ 時系列系 (option=2 で 取得)

| record | 名称 | 取得 タイミング |
|--------|------|------------|
| **O1** | 単複オッズ 推移 | 投票終了 5-30 min ごと |
| **O2** | 馬連 推移 | 同上 |
| **O3** | ワイド 推移 | 同上 |
| **O4** | 馬単 推移 | 同上 |
| **O5** | 三連複 推移 | 同上 |
| **O6** | 三連単 推移 | 同上 |

→ ★ オッズ時系列 features (o1_change_3h_v18 等) の真値化 ★

### 速報系 (option=3、 開催当日)

| record | 名称 | timing |
|--------|------|--------|
| **DM** | デンマ (馬全成績 cumsum) | 速報 |
| **TM** | タイム (走破時計 速報) | レース直後 |
| **WF** | WIN5 | レース当日 |
| **WC** | 出馬表 速報 | 直前 |

## 全 28 種 (full list)

```
RA, SE, HR, H1, H6, WH, WE, AV, JC, TC, CS,
UM, HS, HN, HC, BR, CK, BT,
O1, O2, O3, O4, O5, O6,
DM, TM, WF, WC
```

## 17 features 真値化 path (V20 構築)

| feature | 必要 record |
|---------|-----------|
| race_name | RA ✅ (Phase 13 で TFJV 経由 完了) |
| race_class | RA (shubetsu_code 既取得) |
| race_grade | RA (tokubetsu_num 既取得) |
| race_distance_class | RA (race_dist_raw 既取得) |
| **se_pace_v18** | SE (通過順位 1-4 + タイム) ← JV-Link RT |
| **se_lap_3f_v18** | SE (上がり 3F field) ← JV-Link RT |
| **we_temperature_v18** | WE (気温 record) ← JV-Link RT |
| **wh_track_condition_v18** | WH (馬場状態 detail) ← JV-Link RT |
| **we_wind_v18** | WE (風速 / 方向) ← JV-Link RT |
| **wh_rainfall_v18** | WH (含水率 / 雨量) ← JV-Link RT |
| **o1_change_3h_v18** | O1 (時系列 複数) ← JV-Link option=2 |
| **o1_change_30m_v18** | O1 (時系列 複数) ← JV-Link option=2 |
| **o2_winrate_v18** | O2 (馬連 → tansho 推定) ← JV-Link option=2 |
| **o5_change_v18** | O5 (三連複 時系列) ← JV-Link option=2 |
| **um_sire_winrate_v18** | UM + 既存 SE history (expanding) ← JV-Link option=1 |
| **um_broodmare_winrate_v18** | UM + 既存 SE history (expanding) ← JV-Link option=1 |
| **sk_pedigree_class_v18** | CK (系統 master) ← JV-Link option=1 |

→ ★ JV-Link 4 種 (SE / WE+WH / O1-O6 / UM+CK) で **17 features 真値化 完了** ★

## V20 構築 fetch plan (user authorize 後 AI 自律)

```
Day 1 (AI 自律):
- JVOpen('RACE', 5/3, option=4) → RA + SE + HR  全 record dl
- parser 実装 + features 抽出
- features_jv_se.csv / features_jv_we.csv 生成

Day 2 (AI 自律):
- JVOpen('BLDN', ..., option=1) → 血統 master
- JVOpen('TOKU', ..., option=1) → 出馬予定
- inbreeding 5代 計算
- features_jv_bt.csv 生成

Day 3-5:
- 17 features 全 真値化
- V20 学習 (V15 cache + 17 真値 features + LGB top 100)
- 6-fold WF retrain
- 期待 AUC 0.91-0.93 (V15 0.8939 越え 候補)

Day 6:
- V20 vs V15 実 ROI backtest (本日 V22 比較 logic 流用)
- production switch 判定 user 報告
```

## V15 投資保護 完全 (全 fetch 中)

- 全 fetch read-only、 data/jvlink/raw/ 別 dir
- V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変
- 累計 +13,530 円 守る

## user authorize 待ち task

1. settings.local.json に `Bash(C:/Users/takum/python32/python.exe:*)` allow rule 追加
2. AI 自律 JV-Link fetch 着手 → 17 features 真値化 → V20 構築
3. 6/15+ V20 production 投入判定 (V15 比較 後)
