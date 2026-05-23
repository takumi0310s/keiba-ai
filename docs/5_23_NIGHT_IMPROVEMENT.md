# 5/23 夜 改善実装 完了記録

**実装日**: 2026-05-23  
**方針**: 通知への情報追加のみ。V15 production 完全不変。

---

## 1. 人気除外順位 (PAR) 通知併記 — 実装完了

### 実装ファイル

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `tools/popularity_rank.py` | **新規作成** | PAR 計算 + 通知行フォーマット関数 |
| `tools/notify.py` | 追加 (後方互換) | `build_rich_bet_message(par_df=None)` 引数追加 + TOP3 行 PAR 表示 |
| `tools/race_auto_notify.py` | 追加のみ | PAR 計算呼び出し → `build_rich_bet_message` に `par_df` 渡す |

### 通知フォーマット (変更後)

```
軸: 5 オオタニサーン (0.85) [能力:#1 / 人気:3位]
2位: 10 ペッパーミル (0.72) [能力:#5 / 人気:1位] ★人気
3位: 8 サムライブルー (0.68) [能力:#2 / 人気:7位] ★能力
```

| フラグ | 条件 | 意味 |
|--------|------|------|
| `★人気` | PAR順位 - V15順位 ≥ 3 | 人気除くと大きく下がる (人気頼み) |
| `★能力` | V15順位 - PAR順位 ≥ 3 | 人気除いても上位 (純粋能力評価) |
| なし | 差 < 3 | V15順位と能力順位がほぼ一致 |

### PAR 計算方法 (方法 A: 再予測法)

1. `predict_race(df, model_data)` で V15 スコア確定 (不変)
2. df コピーを作り `paci_ninki_idx` を全馬レース内平均に置換
3. `predict_race(df_copy, model_data)` を再実行 → 「能力スコア」取得
4. 能力スコアで再ランク付け → PAR_rank / PAR_label を元 df に追加
5. V15 の df['スコア'] / bets は一切変更しない

### V15 production 不変の保証

- `predict_core.py` 変更ゼロ
- bets は V15 の df (元スコアソート済み) から生成 — PAR から生成しない
- 例外発生時は `par_df=None` にフォールバック → 既存フォーマットで通知

### JRDB データなし時の挙動

```python
std_ninki = df['paci_ninki_idx'].std()
if std_ninki < 10.0:  # 全馬ほぼ同値 (JRDB未取得)
    PAR_valid = False  # PAR ラベルを表示するが "(データ不足)" 付記
```

---

## 2. DISCORD_WEBHOOK_V21_PAPER 設定

### .env に追加 (プレースホルダ)

```
DISCORD_WEBHOOK_V21_PAPER=
```

現在は空 → `send_discord(..., channel="v21_paper")` は DISCORD_WEBHOOK_UPDATES にフォールバック。

### Discord で webhook を作成する手順

1. Discord サーバー → 対象チャンネル → ⚙️ 設定
2. 「連携サービス」→「ウェブフック」→「新しいウェブフック」
3. 名前: `V21 Paper Trading`
4. 「ウェブフック URL をコピー」
5. `.env` の `DISCORD_WEBHOOK_V21_PAPER=` の右側に URL を貼り付け

### notify.py のルーティング (実装済み)

```python
if channel == "v21_paper":
    url = env.get('DISCORD_WEBHOOK_V21_PAPER', '')
    if url.startswith('https://'):
        return url
    # fallback to updates
    url = env.get('DISCORD_WEBHOOK_UPDATES', '')
```

---

## 3. V15 不変確認

| チェック項目 | 結果 |
|------------|------|
| `predict_core.py` 変更 | **ゼロ** |
| `app.py` 変更 | **ゼロ** |
| `*.pkl.gz` 変更 | **ゼロ** |
| regression tests | **23/23 PASS** (214s) |
| 構文チェック (3ファイル) | **全 OK** |

---

## 4. 翌日確認事項 (5/24)

- [ ] Discord 通知で `[能力:#X / 人気:X位]` が表示されているか
- [ ] `[PAR] 人気除外順位計算完了` ログが race_auto_notify.log に出るか
- [ ] JRDB データあり/なし 両方のフォーマットを確認
- [ ] `DISCORD_WEBHOOK_V21_PAPER` を設定して V21 paper を専用 ch に送れるか
