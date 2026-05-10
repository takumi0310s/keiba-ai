# Phase 13 — V20 features 統合 plan

**date**: 2026-05-10
**target**: netkeiba マスター 25 features を V20 学習 + 推論 pipeline に統合

---

## 1. 統合状態 (2026-05-10 22:00 完了)

### 新規 file

| file | 行数 | 役割 |
|------|------|------|
| `tools/netkeiba_master_scraper.py` | ~430 | scraper 本体 (kill switch + rate limit + 4 系統 parser) |
| `data/v18/phase13_netkeiba_terms.md` | — | 規約 + compliance 設計 |
| `data/v18/phase13_ai_tenkai_poc.md` | — | B. AI 展開予測 PoC |
| `data/v18/phase13_other_features_poc.md` | — | C. 波乱度 + ラップ + バイアス PoC |
| `data/v18/phase13_integration_plan.md` | — | 本書 |

### 既存 file 拡張 (V15 投資保護下、 V18 candidate のみ)

| file | 変更 |
|------|------|
| `tools/predict_core_v18.py` | Phase 13 25 features を build_features 内 V18 candidate END 後に追加 |

★ V15 production 完全不変 (predict_core.py / daily_predict.py / app.py / V15 model file 全て) ★

---

## 2. V18/V20 features 累積 状況

| Phase | features | 累計 | 状態 |
|-------|----------|------|------|
| V15 base | 150 | 150 | ★production★ |
| Phase 11 (V18 JRDB candidate) | +15 | 165 | scaffold |
| Phase 12 (V18 JV-Link candidate) | +17 | 182 | skeleton |
| **Phase 13 (V18 netkeiba master)** | **+25** | **207** | **skeleton** |

→ V18 candidate 207 features (V15 の 150 を完全包含)。
→ V20 投入時の features pool は ~210 の見込み (重複除去 / pruning 後)。

---

## 3. 25 features 内訳 (再掲)

### F. AI 展開予測 (7) — race/compatibility.html

```
master_pace_pred              (0/1/2)
master_pred_winner_score      (0-100)
master_pred_first3f_avg       (秒)
master_pred_last3f_avg        (秒)
master_pred_finish_time       (秒)
master_horse_aitenkai_score   (0-100、 馬別)
master_horse_pred_pos         (1-18、 馬別)
```

### G. AI 波乱度 (3) — race/upset.html (推定)

```
master_haran_score            (0-100)
master_top_pop_trust          (0-100)
master_haran_meter            (1-5)
```

### H. 個別ラップ (10) — race/lap.html (推定)

```
master_horse_lap_avg_first3f
master_horse_lap_avg_last3f
master_horse_lap_best_last3f
master_horse_lap_consistency  (std)
master_horse_lap_best_3f
master_horse_lap_pos_change_avg
master_horse_lap_finish_speed
master_horse_lap_acc_phase
master_horse_lap_dec_phase
master_horse_lap_distance_factor
```

### I. トラックバイアス (5) — race/track_bias.html (推定)

```
master_track_inner_outer_bias  (-1〜+1)
master_track_front_back_bias   (-1〜+1)
master_track_corner_bias       (-1〜+1)
master_track_pace_bias_score   (-1〜+1)
master_track_today_severity    (0-100)
```

---

## 4. data 取得 schedule 設計

### Phase 13.5 (5/11+ 実 DOM 検証)

1. user が master 加入後の browser session で 1 R 手動 access
2. browser DevTools で HTML 保存 → `tools/netkeiba_master_scraper.py` parser stub 検証
3. selector 真値で `_parse_*` 関数を更新
4. 単発 fetch test → 25 features 値妥当性確認

### Phase 13.6 (5/12+ 自動化)

| schtask | 時刻 | 内容 |
|---------|------|------|
| **NetkeibaMasterMorning** | 朝 07:00 | 当日全 R 一括 fetch (B/G/H/I) |
| **NetkeibaMasterRealtime** | 発走 30 min 前 (オプション) | I. トラックバイアス 再 fetch (馬場最終確定) |

実装場所: `tools/scrape_netkeiba_master_daily.py` (新規、 Phase 13.6 で作成)

### rate limit + 規約遵守

- 3 sec interval 必須 (`time.sleep(3.0)` after each fetch)
- 1 日 計 fetch: max 35 R × 4 endpoint = 140 fetch / 日 → 約 7 min/日
- 月間: ~4,200 fetch、 master 月額 ¥4,980 ÷ 4,200 = ¥1.18/fetch (合理的範囲)
- KILL_SWITCH (`data/netkeiba_master/.disabled`) で 即停止可能

---

## 5. V20 学習 pipeline 統合 (5/24+ Phase 3 後半)

```python
# train/v20_master_features.py (新規、 Phase 3 後半)
from tools.netkeiba_master_scraper import (
    ALL_PHASE13_FEATURES,
    fetch_race_master_features,
)

def merge_v20_features(df_v15: pd.DataFrame) -> pd.DataFrame:
    """V15 base 150 + Phase 11/12/13 を merge して V20 学習 data 構築."""
    df = df_v15.copy()
    # Phase 11 JRDB scaffold (15)
    df = merge_phase11_jrdb_features(df)
    # Phase 12 JV-Link skeleton (17)
    df = merge_phase12_jvlink_features(df)
    # Phase 13 netkeiba マスター (25)
    df = merge_phase13_master_features(df)
    return df  # 150 + 15 + 17 + 25 = 207 features
```

V20 4-model ensemble (LGB + XGB + FT + IR) は v13.5b の grid 設計を継承、 features 数のみ拡大。

---

## 6. 期待 ΔAUC (Phase 13 単独寄与)

| category | 期待 ΔAUC | 根拠 |
|----------|-----------|------|
| F. AI 展開予測 | +0.005 〜 +0.012 | netkeiba 内製 AI 流用 (相関高い可能性) |
| G. AI 波乱度 | +0.003 〜 +0.008 | race-level 投資判断 filter 寄与 |
| H. 個別ラップ | +0.015 〜 +0.030 | 既存 prev_last3f 系 superset、 std/距離 factor 完全新規 |
| I. トラックバイアス | +0.005 〜 +0.015 | 完全新規 race-level 信号 |
| **合計** | **+0.028 〜 +0.065** | conservative |

V18 (Phase 11+12) 期待 0.91-0.93 + Phase 13 → V20 期待 **0.94 視野**。

---

## 7. risk + mitigation

| risk | mitigation |
|------|------------|
| selector 推定外れ | parser stub の selector 候補を多重定義、 取れない field は default fill |
| netkeiba 規約改訂 | KILL_SWITCH (`.disabled` file) → 全 fetch 即停止 |
| Cookie 期限切れ | 既存 `tools/refresh_cookie.py` 自動更新 system 流用 |
| rate limit 違反 / IP BAN | 3 sec 厳守 + 1 日 7 min 程度に抑制 |
| AI 予測 features の二重計上 | V15 既存 features と相関 check、 高相関は drop |
| master 機能仕様変更 | parser stub に warning 出力、 異常値検知時 fallback |

---

## 8. 5/11+ TODO

| 期日 | task | 担当 |
|------|------|------|
| 5/11 | user による 1 R 実 fetch + DOM 検証 | れんはす |
| 5/11 | parser stub の selector 真値 化 | Claude |
| 5/12 | 当日 全 R fetch test | 両者 |
| 5/12 | NetkeibaMasterMorning schtask 登録 | Claude |
| 5/13-5/15 | data 蓄積 + 値分布検証 | 両者 |
| 5/24+ | Phase 3 後半 V20 学習に Phase 13 features 投入 | Claude |

---

## 9. V15 投資保護 (絶対 不変)

| 不変対象 | 状態 |
|----------|------|
| `predict_core.py` | ★完全不変★ |
| `daily_predict.py` | ★完全不変★ |
| `app.py` | ★完全不変★ |
| `keiba_model_v15_central.pkl.gz` | ★完全不変★ |
| 累計収支 +¥14,140 | ★維持★ |
| 戦略⑦ (06_特別 / 京都 / E / B 除外) | ★継続★ |
| 案 B 改 12R 1 勝 C 上限 ¥2,100 | ★継続★ |
