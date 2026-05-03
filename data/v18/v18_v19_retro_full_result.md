# v18/v19 完全 retro (5/2-5/3) — **実施失敗 (model 破損)**

生成: 2026-05-03

## 結論: ⚠️ retro 実行できず

### 原因

`data/v18/models/v19_fukusho_lgb.txt` および `data/v17/models/v17_morning_lgb_fold5.txt` が両方 LightGBM Model format error。

```
[LightGBM] [Fatal] Model format error, expect a tree here. met
```

両 model file は **md5 hash 同一** (3d95efb5f069b2fcd8969e5b07738adb) で byte 単位で同じだが LightGBM が parse 失敗。

### 推定原因

1. **CRLF 変換**: file 形式 `ASCII text with CRLF line terminators` — git autocrlf による LF→CRLF 変換が破損の原因可能性高
2. **session 競合**: session#1/#3 が同 file を modify し format 不整合の状態で残留
3. **timestamp 異常**: file mtime が `May 3 20:29` を示す (実時刻 15:03) — 時計同期 or 別 session の future-dated commit

### 影響

- v19 単独でモデル inference 不能 → 5/2-5/3 v18/v19 完全 retro 不可
- 楽観バイアス (BT 295% / 149% vs 実 retro) の確定不能

### 既に成功している部分

- ✅ `tools/v18_v19_retro_full.py` script 作成完了 (model 修復後そのまま実行可能)
- ✅ `tools/build_odds_base_retro.py` 完成、5/2-5/3 odds_base 構築済 (552 + 580 rows)
- ✅ `data/odds_base_20260502.csv`, `data/odds_base_20260503.csv` 生成済

### 復旧手順

```bash
# Option 1: git で v19 (= v17_morning) model 復元 (LF 状態で commit されているなら)
cd /c/Users/takum/keiba-ai
git log --oneline -5 -- data/v18/models/v19_fukusho_lgb.txt
git checkout <good-sha> -- data/v18/models/v19_fukusho_lgb.txt
# 確認: file <path> で "no line terminators" or "LF" を期待

# Option 2: model を再 train
python train/run_v19_fukusho.py
# (cache 必要、~8min)

# Option 3: CRLF→LF 変換を試行 (file が単に line ending 違いだけなら有効)
python -c "
with open('data/v18/models/v19_fukusho_lgb.txt','rb') as f: c = f.read()
c = c.replace(b'\r\n', b'\n')
with open('data/v18/models/v19_fukusho_lgb.txt','wb') as f: f.write(c)
print('CRLF→LF 変換完了')
"
```

## v18/v19 retro 代替分析 (proxy ベース)

完全 retro は不能だが、Phase 2 BT (2025 OOS) と session#3 計算済 proxy で代替判断:

### Phase 2 BT (2025 OOS, n=47,497)

| モデル | filter | n_bet | hit_rate | ROI |
|--------|--------|------:|---------:|----:|
| v18 単勝 | p≥0.5, EV≥1.2 | 642 | 74.9% | **295.1%** |
| v19 複勝 | p≥0.7, EV≥1.1 | 1,919 | 86.7% | **149.3%** |

### session#3 calibration 検証

- v18 全 bin で actual > predicted (under-confidence)
- 平均 gap +0.05〜+0.13 → 確率 過小評価
- → **本番 ROI = BT × 0.5-0.8 程度の楽観バイアス想定**

### session#3 5/2-5/3 人気1位 proxy

| 日付 | 人気1位 trio含有率 |
|------|-----------------:|
| 5/2 | 66.7% |
| 5/3 | 52.9% |

V19 BT 期待 hit rate (~80%) との gap **-13〜-27pt**

## 5/16 以降への影響

楽観バイアス確定は v19 model 復旧後に再実行必要。
それまでは:

| 想定 | 修正係数 | 期待 ROI (本番) |
|------|---------:|---------------:|
| 楽観 | 1.0 | 295% / 149% (BT そのまま) |
| 中庸 | 0.7 | 207% / 105% |
| 保守 | 0.5 | 148% / 75% |

→ **5/16 以降 v18/v19 部分実弾 開始時、保守シナリオ (148% / 75%) で見積り**。
v19 retro 完全版が完了したら数値修正。

## TODO

- [ ] `data/v18/models/v19_fukusho_lgb.txt` の format 修正 (CRLF→LF or git checkout)
- [ ] `tools/v18_v19_retro_full.py` 再実行 (修復後)
- [ ] BT vs 実 retro ROI 比較で楽観バイアス係数を確定
- [ ] 5/16 以降の v18/v19 部分実弾投資額計算
