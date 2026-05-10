# Phase 12 D: 血統情報拡張 features (4 件) 実装 (5/10)

> Session #87 Phase 12 D 領域 (2026-05-10 18:00+)
> 出力: tools/predict_core_v18.py の BLOODLINE_FEATURES (4 件)

---

## 1. user 指示 vs 実 JV-Link record mapping

| user 指示 | 実 JV-Link record | 用途 |
|----------|-------------------|------|
| 「BR 血統情報」 | UM (馬個体 1936-2025) + SK (産駒情報) + BR (繁殖牝馬) | 4 features (父系/母父系 距離+馬場 適性) |

★ 注: JV-Link 仕様上 BR = 繁殖牝馬 record。 「血統情報拡張」 は UM (馬個体)、 SK (産駒)、 BR (繁殖牝馬) の **3 record 集約** が正規 source。 user の functional intent に対応。 ★

---

## 2. 実装 4 features

| feature | source | 計算 |
|---------|--------|------|
| jv_sire_dist_apt_score | UM + SK 集計 | sire の同距離帯 top3 率 / 全産駒 top3 率 (0-1) |
| jv_dam_sire_apt_score | UM + BR 集計 | 母父 同距離 + 同馬場 複合 score (0-1) |
| jv_sire_surface_apt_score | UM + SK 集計 | sire の同馬場 top3 率 / 全産駒 top3 率 (0-1) |
| jv_ped_score_blend | 上記 集約 | 0.5 * sire + 0.3 * dam_sire + 0.2 * bms_dist (0-1) |

default = 0.5 (中央値、 race 結果 中立)

---

## 3. V15 既統合 features との 差別化

V15 既統合 (血統):
- sire_enc, bms_enc (TOP100 encoding)
- sire_surface_wr, sire_dist_wr (expanding window)
- bms_surface_wr (expanding window)
- sire_shinba_top3r (新馬戦 父産駒 top3 率)

Phase 12 D 差別化:
- ★ jv_sire_dist_apt_score ★ は **正規化 score** (V15 は raw rate)
- ★ jv_dam_sire_apt_score ★ は **複合 score** (距離 + 馬場 同時、 V15 は単独)
- ★ jv_ped_score_blend ★ は **総合 1 score** (V15 は別々の features 4 個)

---

## 4. live activation 設計

### 4.1 Phase 12 (本日)
- skeleton 実装、 default fill 0.5 のみ
- blood_num (血統登録番号 8 桁) を caller から受け取る
- blood_num 不明 / JV-Link 未到着 → 0.5 fill

### 4.2 5/24+ Phase 3 後半
- TFJV UM_DATA (1936-2025、 90 年分) を直 parse (Session #44 A inventory 済)
- UM record から 馬個体 + 父 + 母 + 母父 fetch
- SK record から 種牡馬産駒成績 (距離別 / 馬場別) 集計
- BR record から 繁殖牝馬産駒成績 集計
- expanding window (date 順 cumsum-current) で leak free

### 4.3 計算 logic (5/24+ で実装)

```python
# sire_dist_apt_score 計算例
def compute_sire_dist_apt(sire_id: str, dist_cat: int, race_date: str):
    # sire の 全産駒 (race_date 以前) top3 率
    base = sire_all_runs.before(race_date).top3.mean()
    # 同距離帯 top3 率
    same_dist = sire_runs_at_dist(dist_cat).before(race_date).top3.mean()
    # 適性 score (1.0 = 全産駒平均、 > 1 で得意、 < 1 で苦手)
    return min(1.0, max(0.0, same_dist / max(base, 0.01)))
```

---

## 5. 動作 test

```
E. 血統拡張 (4): ['jv_sire_dist_apt_score', 'jv_dam_sire_apt_score',
                  'jv_sire_surface_apt_score', 'jv_ped_score_blend']
default fill 動作確認: 全 4 features 0.5 で OK
```

---

## 6. V15 投資保護

✅ tools/predict_core.py 不変
✅ V15 model 不変
✅ data/blood_full.csv 不変 (V15 経路維持)
✅ predict_core_v18.py 新規、 V15 と完全独立
✅ TFJV UM_DATA は read-only

---

## 7. 結論

✅ D1: UM + SK + BR records 由来 4 features 定義
✅ D2: V15 既統合 (sire/bms_enc / sire_dist_wr 等) と差別化 (正規化 score / 複合 / blend)
✅ D3: skeleton 動作 OK (default 0.5 fill 4 features)
✅ D4: 5/24+ で TFJV UM (90 年分) 経由 live 化、 expanding window で leak free
✅ D5: V15 完全保護
