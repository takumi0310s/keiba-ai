# JRDB TYB 直前情報 実装後 honest 評価 + 次 step 統合 (2026-05-16 evening 追記)

> **背景**: 5/16 evening session で JRDB TYB fetch 復旧 + LR predictor 学習完了。
> 5CV AUC: V15 only 0.4653 → V15+TYB 0.6082 (+0.1429、 n=348)。
>
> **本 doc 目的**: この成果を honest 評価、 過大評価 / 過小評価を避け、
> ALL_IN_PLAN_2026_05_16.md の P1/P2 に正しく統合する。

---

## 🎯 真の意味

### 良い面 ✅
1. **既存 548K rows TYB data が眠っていた bug 発見**
   - 4/18 daily_jrdb_kyi.bat で取得 schtask 既登録
   - merge logic 機能不全で V15 で 0% 結合だった
   - → 1 年以上の data が活用可能に
2. **fetch 復旧成功** (5/9-5/15 停止期間あり、 5/16 today 487 rows 投入)
3. **真の signal 確認** (padock_idx LR coef +0.44 は理論的に正しい方向)
4. **V15 production 完全不変** (predict_core / app.py / schtasks 全部 untouched)

### 慎重に見るべき面 ⚠
1. **baseline AUC 0.4653 が異常に低い**
   - top1_score alone で top3 hit predict task は不適切
   - random (0.50) 以下 = 測定 task 設計に問題
   - → +0.1429 という大きな改善幅は **task 設計バイアス** の可能性大
2. **train 0.696 vs CV 0.608、 gap 0.088 = over-fit 兆候**
3. **n=348 は production 判断には不十分**
4. **leak 監査未完了**
   - tansho_odds が -15 min snapshot か race 確定か audit 必須
   - V15.1 SKB leak (skb_kishi_code_3 +480bp) と同じ罠 risk
5. **TYB merge bug の root cause 未特定**
   - なぜ 1 年以上 0% 結合だったか
   - 再発防止 logic 必要

---

## 📋 ALL_IN_PLAN 改訂 (TYB 反映版)

### ★ P0 (5/17 G1 day、 read-only のみ)
- P0-1: ROI 乖離真値確定 (変更なし)
- P0-2: 京都/中京 戦略⑦再除外 (変更なし)
- ★ NEW ★ **P0-3: TYB calibrator leak 監査** (1-2h、 read-only)
  - tansho_odds が -15 min snapshot か race 確定か検証
  - daily_predictions/YYYYMMDD.csv の TYB merge timing 確認
  - padock_idx の release timing audit (JRDB CYB との関係)
  - over-fit risk 評価 (train vs CV gap 0.088)
  - 結果次第で P1 採用 / 巻き戻し判定

### ★ P1 (5/18-5/24、 paper eval)
- ★ 改訂 ★ **P1-0: TYB calibrator paper shadow eval ★最優先★**
  - 5/18 朝から strategy_layer_v2 --calibrator tyb shadow eval 開始
  - 既存 V15 投票には**一切影響なし** (shadow only)
  - 30R 蓄積後、 V15 単独 ROI vs V15+TYB calibrator ROI 比較
  - 統計的有意性 Welch's t-test (p<0.05) 判定
  - 採用判定: 上記 + 直前情報 (paddock release timing) audit PASS が条件
- P1-1: calibrator v1 (既存) paper eval (継続)
- P1-2: JRDB tyb/cha feature engineering (P1-0 結果次第)
- P1-3: netkeiba マスターコース評価 (変更なし)

### ★ P2 (5/25-5/31)
- P2-1: v15.2 再学習 (★ TYB features を含む candidate ★、 ただし P0-3 leak 監査 PASS が前提)
- P2-2 / P2-3 (変更なし)
- ★ NEW ★ **P2-4: TYB daily fetch schtask 登録 + monitor**
  - 5/9-5/15 停止再発防止
  - fetch 失敗時 Discord 警告
  - jrdb_tyb.csv 鮮度 daily check

---

## ⚠ 重大な 5/17 確認事項

### 5/17 V15 投票への影響: ★ ゼロ ★
```
✅ predict_core.py 不変
✅ daily_predict.py 不変
✅ race_auto_notify.py 不変
✅ app.py 不変
✅ V15 .pkl.gz 不変
✅ 既存 schtasks 不変
✅ data/jrdb_tyb.csv は merge されているが V15 model は使わない
   (LR predictor は shadow 専用、 production routing 未統合)
```

→ 5/17 ヴィクトリアマイル G1 day は完全に V15 単独運用。 TYB は shadow eval 専用。

### 5/17 朝 06:00 schtask の状況
```
DailyJrdbKyi (AM6:00、 4/18 から既登録):
  - KYI/SED/TYB/CYB/JOA/KAB/KTA/CHA/KKA/JO を fetch
  - 5/16 evening の jrdb_tyb_live_fetch.py で fetch 復旧確認済
  - 5/17 朝 06:00 は通常通り fire 想定
  - もし TYB が再度 publish なし → NO_TYB fallback (既存 logic)
```

---

## 🛡 V15 投資保護 状況 (5/16 evening 時点)

```
✅ V15 production 完全不変
✅ 累計 (要 P0-1 真値確定) 維持
✅ commit b4948d6a: 新規 file のみ (tools/v21/*, data/tyb_top3_predictor.pkl, docs)
✅ V15 routing 未変更 (LR predictor は shadow 専用、 strategy_layer_v2 統合は次 step)
✅ destructive op なし、 settings.local.json 維持 (filter-repo NG / force push NG)
```

---

## 🚀 5/17 G1 day 当日 行動 plan (TYB 反映版)

```
5/17 03:00: 通常 schtasks fire (既存)
5/17 06:00: ★ DailyJrdbKyi 復旧後の初回稼働 ★
   → jrdb_tyb.csv に 5/17 today rows 投入されるか確認
5/17 08:00: Daily predict (既存、 V15 単独)
5/17 09:30: Discord #買い目 通知 (既存)
5/17 14:00: 14:00 投票候補確定通知
5/17 15:40: ヴィクトリアマイル発走
5/17 17:00: 結果集計
5/17 20:00: 週次 report
5/17 21:00+: ★ P0-1 + P0-2 + P0-3 一斉起動 ★ (Option B 推奨)
```

---

## 🎯 結論: 「動画なし業界 frontier」 への一歩

```
直前 session で達成:
  ✅ JRDB TYB の真の活用 path 確立
  ✅ padock_idx という 真の signal 発見 (LR coef +0.44)
  ✅ 動画 (規約 NG) の代替として、 人間数値化 data 最大活用 path
  ✅ V15 投資保護 完全維持

ただし honest に:
  ⚠ baseline AUC 0.46 は task 設計問題、 +0.14 は過大評価リスク
  ⚠ n=348 + over-fit gap 0.088 = 30R 以上の shadow eval 必須
  ⚠ leak 監査 (P0-3) PASS が production 投入の前提

正しい path:
  5/17 G1 day: V15 単独完全守備
  5/18+: P0-3 leak 監査 + P1-0 shadow eval 開始
  5/24+: 30R 蓄積後、 statistical test 通過なら P2-1 v15.2 学習に統合
  6/1+: production 投入候補

★ 「動画なし限界点」 への path は確実に前進した ★
```
