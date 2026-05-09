# JRA-VAN NEXT 5/15 trial plan

Session #81 (2026-05-09 夜)。

## 目的

JRA-VAN NEXT の 自動分配 + 1 click PAT 送信を、
5/16 V18 trial と 同時に **1 ヶ月無料 trial** で 検証する。

## schedule

| 日付 | 内容 |
|------|------|
| 5/15 (木) | NEXT contract (1 ヶ月無料 trial 申込)、 動作確認 |
| 5/16 (土) | V18 trial と 同時 試用、 案B改 strict + 自動分配 |
| 5/17 (日) | 案B改 strict + 自動分配 (継続) |
| 5/22 (木) | 1 週間 評価 集計 |
| 5/23 (金) | 採用判定 |

## 5/15 (木) 作業項目

1. JRA-VAN NEXT 公式サイト で 1 ヶ月無料 trial 申込
2. TARGET frontier JV (NEXT 込み) install + login
3. スマホアプリ install + login (PC と 同期 確認)
4. PAT (or 即PAT) と NEXT の 連携設定
5. テスト 投票 (最小金額 100 円 で 動作確認)

## 5/16-5/22 試用項目

### 案B改 strict (V15) で 自動分配 試用

各レース で:
1. V15 daily_predict 候補 確認
2. 案B改 strict 適合 R のみ pick (06_特別/京都/条件E/条件B 除外)
3. NEXT に 馬番リスト 入力 (予算 2,100 円)
4. 自動分配 button push
5. 配分結果 を スクリーンショット 保存
6. 1 click PAT 送信
7. 結果 (的中/不的中、 配当) 記録

### 配分 algorithm 観察

5/16-5/22 で 10-20 R 試用 → 配分結果 を 集計:
- EV ベース? オッズ ベース? Kelly? 均等?
- 「全体プラス」mode と 通常 mode の 差分

### V18 trial との 並行運用

- V18 trial: pre_race_predict_v3 で 候補生成 (絞り込み目的)
- V15 案B改 strict: 投票実行
- NEXT 自動分配: 金額配分 + PAT 送信

3 系統 完全独立、 V15 投資保護 継続。

## 5/22 (木) 評価項目

| 項目 | metric | GO 条件 |
|------|--------|---------|
| ROI | 自動分配 vs 手動分配 ROI | 自動分配 >= 手動分配 - 5pt |
| 操作性 | 1 R あたり 入力時間 | 30 秒以内 |
| 連携精度 | PAT 送信 成功率 | 100% |
| 誤投票 | 件数 | 0 件 |

## 5/23 (金) 採用判定

### GO シナリオ (NEXT 採用、 月 2,090 円 継続)

- 1 R 30 秒 + 自動分配 ROI 維持 → 即採用
- 6/8 V20 投入時にも NEXT 連携 継続
- TARGET frontier カスタム指数 機能 調査 開始

### NO-GO シナリオ (NEXT 解約)

- 自動分配 ROI 大幅 negative (例: -10pt 以上)
- 操作性 悪化 (1 R 1 分以上)
- 誤投票 1 件以上 発生
- → 現状 PAT 手動入力 に戻る、 ロードマップ 見直し

## 予算

- NEXT 1 ヶ月無料 (5/15-6/14)
- 6/15+: 採用なら +2,090 円/月 (既存 JRA-VAN DataLab とは別 contract の可能性)
- 5/22 確認後 確定

## risk

- V15 投資保護: 完全 維持 (NEXT は UI レイヤーのみ)
- 累計 +14,140 円 / 撤退余裕 +64,140 円 死守
- 試用中 単日 ROI<50% / 累計 -10k / 累計 -50k で **trial 即中止**

## 関連 doc

- [JRA_VAN_NEXT_AUTO_ALLOCATION.md](JRA_VAN_NEXT_AUTO_ALLOCATION.md) — 機能詳細
- [V15_TO_JRA_VAN_INTEGRATION.md](V15_TO_JRA_VAN_INTEGRATION.md) — 連携 logic
- [AUTO_VOTING_ROADMAP.md](AUTO_VOTING_ROADMAP.md) — 完全自動化 ロードマップ
