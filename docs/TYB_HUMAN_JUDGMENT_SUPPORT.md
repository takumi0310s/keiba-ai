# TYB 人間判断支援設計 (Path 1 + Path 2)

**作成日**: 2026-05-22  
**前提**: V21 ablation NO-GO 確定 (全 TYB fields delta ≤ -0.0016)  
**方針**: TYB は AI 特徴量としては不採用。★ 人間 (れんはす) の発走前最終判断材料として活用 ★

---

## 設計方針

| 項目 | 決定 |
|------|------|
| V15 AI 予測への TYB 組み込み | **永久 NO** (ablation 確定、predict_core 変更なし) |
| TYB 活用方法 | Discord 補足表示 + 気配急変 alert のみ |
| 有効化タイミング | **6/1 観測成功後** に `TYB_SHADOW_ENABLED = True` (手動) |
| 5/23 実運用 | TYB_SHADOW_ENABLED=False のまま → 完全無影響 |
| 投票 formation | V15 + 戦略⑦ + C4 のまま **不変** |

---

## Path 1: Discord 補足表示

### 表示フォーマット

V15 が推す top-5 馬それぞれの直前情報を Discord メッセージ末尾に付記する。

```
━━ R06 直前情報 (TYB shadow) ━━
  #5 ホワイトオーキッド 1位: 体重:480kg(-2) パドック:◎(90) 単勝:3.5倍 / 気配:良
  #7 ブルーサファイア 2位: 体重:466kg(+4) パドック:○(72) 単勝:5.8倍 / 気配:普通
  #3 サンダーストーム 3位: 体重:512kg(-18)⚠ パドック:△(55) 単勝:9.2倍 / 気配:発汗
  #11 シルバーウインド 4位: 体重:458kg(0) パドック:?(45) 単勝:12.0倍
  #2 レッドフォックス 5位: 体重:494kg(+2) パドック:○(68) 単勝:8.5倍 / 気配:良
  発走18分前取得 | ★shadow only V15予測非影響★
```

### 気配コード表示

| kehai_code | 表示 |
|-----------|------|
| 1 | 良 |
| 2 | 普通 |
| 3 | 不振 |
| 4 | 発汗 |
| 5 | チャカ |

---

## Path 2: 気配急変 Alert

### アラート条件

| 条件 | 閾値 | 対象 |
|------|------|------|
| 馬体重大幅変化 | \|weight_diff\| ≥ 15kg | top 1-3 |
| 気配悪化 | kehai_code in {3, 4, 5} | top 1-3 |
| 取消 | cancel_flag == 1 | top 1-3 |

### アラート文字列例

```
⚠ top1 サンダーストーム(#3): 馬体重減18kg → 要確認
⚠ top2 ブルーサファイア(#7): 気配:発汗 → 要確認
⚠ top3 シルバーウインド(#11): 取消 → 要確認
```

---

## 実装済み関数 (tools/tyb_shadow_fetcher.py)

| 関数 | 役割 |
|------|------|
| `fetch_tyb_shadow(race_id, start_time, enabled)` | TYB fetch (default disabled) |
| `format_tyb_horse_line(row, horse_name, rank)` | 1馬分の Discord 行フォーマット |
| `check_tyb_anomalies(tyb_result, top_horses)` | top1-3 気配急変検出 → alert list |
| `build_tyb_discord_block(tyb_result, top_horses, race_num)` | 全体ブロック組み立て |
| `format_tyb_discord_supplement(tyb_result)` | 旧サマリー形式 (後方互換) |

### V15 非影響保証

```python
# tyb_result=None (disabled or fetch失敗) → 常に "" を返す
block = build_tyb_discord_block(None, top_horses)  # → ""
alerts = check_tyb_anomalies(None, top_horses)      # → []

# TYB は V15 predict_race() / 投票 formation に渡さない
# race_auto_notify.py の V15 予測ロジックは一切変更しない
```

---

## 6/1 有効化手順 (観測成功後)

```
前提: 6/1 shadow 観測で R01〜R12 全取得確認 (parse 成功率 ≥ 95%)
```

### Step 1: TYB_SHADOW_ENABLED を True に変更

```python
# tools/tyb_shadow_fetcher.py
TYB_SHADOW_ENABLED: bool = True   # ← False から変更 (6/1 観測成功後)
```

### Step 2: race_auto_notify.py の Discord 通知後に TYB ブロックを付記

```python
# (race_auto_notify.py の send_discord 直後に追加、V15予測ロジックは不変)
# import は import 部に追加: from tools.tyb_shadow_fetcher import fetch_tyb_shadow, build_tyb_discord_block, TYB_SHADOW_ENABLED

# 予測・formation 生成後、Discord 送信直後に追加:
tyb_result = fetch_tyb_shadow(race_id, start_time_str=rinfo.get('start_time',''), enabled=TYB_SHADOW_ENABLED)
tyb_block = build_tyb_discord_block(tyb_result, top5_horses, race_num=race_info['race_num'])
if tyb_block:
    from notify import send_discord as _sd
    _sd(f"直前情報 R{race_info['race_num']}", tyb_block, color="blue", channel="bets")
```

`top5_horses` の構築例:
```python
top5_horses = [
    {"umaban": int(row['馬番']), "horse_name": row.get('馬名', ''), "score": row.get('スコア', 0)}
    for _, row in df.head(5).iterrows()
]
```

### Step 3: 観測ログ確認

```
data/tyb_shadow_log.csv に R01〜R12 の status=OK が揃っていること
data/tyb_shadow/{date}/*.json が 12 件存在すること
```

---

## V15 非影響保証 (テスト)

```
tests/test_tyb_shadow_fetcher.py — 12 tests all PASS (0.08s)

Test 7:  format_horse_line ± 15kg → ⚠ 付き
Test 8:  format_horse_line ± 2kg → ⚠ なし
Test 9:  check_anomalies 体重-20kg → alert あり
Test 10: check_anomalies 正常 → alert []
Test 11: build_discord_block(None, ...) → "" (V15非影響)
Test 12: build_discord_block 正常データ → ブロック+⚠
```

---

## タイムライン

| 日付 | アクション |
|------|---------|
| 5/23 (SAT) | V15 本番運用。TYB 完全 disabled |
| 6/1 | shadow 観測 W1(R01)/W2(R06)/W3(R12) 取得確認 |
| 6/1 観測 OK | `TYB_SHADOW_ENABLED = True` + race_auto_notify 統合 |
| 6/17 | paper eval 結果 + TYB alert 実用性 評価 |

---

*TYB → AI feature: NO-GO (V21 ablation 確定 2026-05-22)*  
*TYB → 人間判断支援: ✅ Path 1+2 実装済み (6/1 有効化待ち)*
