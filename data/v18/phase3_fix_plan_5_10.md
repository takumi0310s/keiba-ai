# Phase 3 / D: 修正必要性 評価 + 自動修正 (5/10)

## 結論
✅ **今回の即時修正: なし** (V15 投資保護 + Phase 2 緊急 確定 logic 不変保証)
📅 **5/15 V18 trial 直前 改善 plan 確定**

## 評価 matrix

| 領域 | 即時修正 | 5/15 plan |
|------|----------|----------|
| A. 場 filter | なし (戦略⑦ 仕様通り) | 京都 filter 再評価 (5/11+ データ蓄積後) |
| B. 特徴量可視化 | なし (5/16 V18 trial 直前 plan) | LGB+XGB importance + 通知 append |
| C. 体重統合 | なし (case A 設計通り) | PreRacePredict_Watchdog re-enable + Stage 2 体重統合 復活 |

## 即時修正 しない 理由

### 🔴 NEVER (絶対遵守)
1. predict_core / daily_predict / app.py 変更 ★絶対不変★
2. V15 model 変更
3. schtasks 既存 50 件 変更
4. 既存 dev branch 変更
5. ★destructive git op (reset --hard / push --force)★
6. ★5/10 投票候補 logic 変更 (Phase 2 緊急で確定済)★

### Phase 2 緊急 で 確定した 5/10 投票 logic
- 案B改 strict 12R 1 勝クラス ¥2,100 投票 (上位 3 R)
- 期待 ROI: 125.1% (5月 実績、 体重なしでも実証)
- 体重情報なしでも V15 model は十分機能 (default 480/0 で 8 features 非情報化、 他 features で予測)

## 5/15 V18 trial 直前 改善 plan (確定)

### plan 1: PreRacePredict_Watchdog re-enable + Stage 2 体重統合
```powershell
# 5/15 朝 実行
schtasks /change /tn "Keiba-PreRacePredict_Watchdog_5_9" /enable
python tools/multi_stage_predict.py --stage race12_1545 --date 20260515 --dry-run
# 体重 default 480 → 実体重 反映確認
```

### plan 2: hardcode date 確認 (Session #78 修正後 再確認)
```powershell
Select-String -Path tools/multi_stage_predict.py,tools/pre_race_predict.py `
  -Pattern "20260509|5/9 hardcode" -SimpleMatch
```

### plan 3: 通知 主要 5 特徴量 追加 (5/16 V18 trial 直前)
- `train/v15_feature_importance.py` 新規作成
- LGB+XGB gain importance export → `data/v15_feature_importance.json`
- `tools/notify.py` `build_rich_bet_message` 拡張
- TOP3 馬 末尾に `📊 主要特徴量: jockey_wr=X, JRDB_IDM=Y, sib_top3_rate=Z` 追加

### plan 4: 体重 急変 (±10kg) アラート 追加検討
- Stage 2 体重統合 後、 weight_change_abs ≥ 10 で Discord アラート
- `tools/notify.py` に 警告 line 追加

### plan 5: 京都 filter 再評価 (5/11+)
- CLAUDE.md 記述: "京都 を除外 (データ蓄積待ち、5/11 以降に再評価)"
- 5/11 + 5/12 + 5/17 + 5/18 京都 ROI 蓄積後 戦略⑦ 再判定

## 即時修正 案 (検討した が 採用しない)

### 案 X1. 京都 filter 削除 → ❌
- 戦略⑦ 仕様通り、 ROI 検証 待ち
- 5/11+ データ蓄積後 別判定

### 案 X2. PreRacePredict 今すぐ re-enable → ❌
- 5/9 disable 経緯不明 (Session #78 に詳細あり と推察)
- 5/15 V18 trial 前に動作確認必要、 今 enable は risk
- 今日中 (5/10) 残時間で 全 R Stage 2 安定動作 検証 不可

### 案 X3. 通知に 主要特徴量 追加 → ❌
- 5/10 残 R 通知に bug 混入 risk
- V18 trial 直前 (5/16) で実装 + 検証 が安全

## 修正実施
✅ **即時修正なし** (audit + plan doc のみ)
✅ main +1 commit (4 audit doc 追加)
