# Phase 3 / B: 特徴量可視化 audit (5/10)

## 結論
⚠ **現状: 通知 message に「主要特徴量」 表示なし**。 V15 score / TOP3 / JRDB 指数 / Premium データ のみ。
📅 **5/16 V18 trial 直前 plan 化** (今 急ぐと bug risk)

## 現状 通知 content (`tools/notify.py:185 build_rich_bet_message`)

1. レース header (日付 / 場 / R / 発走時刻)
2. レース名 + surface + distance + condition + 頭数
3. 条件 + ★ + ROI + 的中率
4. 収益パターンマッチ ⭐
5. 買い目 (馬連 1 軸 2 流 / 三連複フォーメーション)
6. **TOP3 馬: 馬番 + 馬名 + スコア** (score のみ、 特徴量内訳なし)
7. 配当レンジ
8. Premium: タイム指数 / 調教ランク / 厩舎スコア
9. JRDB 指数: IDM / パドック / オッズ / 激走 / 総合 (TOP3)

## 不足

ユーザー要望:
- 「何のデータを参考に + どんな特徴量で予想?」
- 主要 5 特徴量 + その寄与 を見たい (例: "騎手成績 +0.12, 厩舎成績 +0.08")

## V15 model の feature_importance 取得可能性

**LGB**: `model.feature_importance(importance_type='gain')` - 既存 model から取得可
**XGB**: `model.feature_importances_` - 同上
**FT**: 取得困難 (attention weight)
**IR**: 取得困難 (intra-race attention)

→ LGB/XGB の gain importance を **学習時 1 回 計算 + キャッシュ** で十分。
→ 1 馬の predict 時 個別寄与は SHAP 必要だが 重い。 当面は **global 重要度 TOP5 + 馬の該当 feature 値** で十分。

## 工数評価

| 案 | 工数 | impact |
|----|------|--------|
| A. global TOP5 feature + 馬の値 表示 | 1-2h | ★★★ |
| B. SHAP per-horse 寄与 | 4-6h | ★★★★ (重いが正確) |
| C. predict_core から feature_dict 取得 + 通知 append | 2-3h | ★★★ |

## plan: 5/16 V18 trial 直前

### 実装 step
1. `train/v15_feature_importance.py` 新規 - LGB+XGB importance を export → `data/v15_feature_importance.json`
2. `tools/notify.py:build_rich_bet_message` 拡張:
   - param `feature_dict` (dict[馬番, dict[feature_name, value]]) 追加
   - TOP3 馬 末尾に `📊 主要特徴量: jockey_wr=0.12, JRDB_IDM=82.3, sib_top3_rate=0.45` 追加
3. `tools/race_auto_notify.py` `predict_and_notify`:
   - df から TOP5 importance feature 値を抜き出して dict 化
   - `build_rich_bet_message(..., feature_dict=fd)` 呼び出し
4. `tools/daily_predict.py` も同様 (但し朝予測は表示優先度低)

### 5/16 投入条件
- V18 trial 後 通知形式変更でも安全 確認後
- main commit 1 本 + 4-6h 検証
- ユーザー確認後 deploy

## 修正 (今回)
✅ **なし** (5/16 plan 化)
