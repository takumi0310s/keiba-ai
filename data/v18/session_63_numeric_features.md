# Session #63 D: JRDB / JV-Link 数値 features 統合 結果

**作成**: 2026-05-09 11:XX (Session #63 D、 dev/training-poc)

対象: 5/9 重賞 3 + 12R 3 = 6 R / 90 馬

## カバレッジ

| feature | source | 取得率 | 備考 |
|---------|--------|--------|------|
| paddock_idx | TYB | 0/90 (0%) | TYB 5/9 未publish (HTTP 404)、 13:00 頃 publish 予定 |
| weight_diff | TYB | 0/90 (0%) | 同上 |
| training_idx | KYI | 90/90 (100%) | ★ |
| idm_score | KYI | 90/90 (100%) | ★ |
| stable_idx | KYI | 90/90 (100%) | 厩舎指数 |
| ninki_idx | KYI | 90/90 (100%) | 人気指数 |
| gekiso_idx | KYI | 90/90 (100%) | ★ 激走指数 (paddock 代替候補) |
| train_eval | CYB | 0/90 (0%) | byte 位置調査要 (5/9 file 取得済だが値 empty) |
| train_mark | CYB | 0/90 (0%) | 同上 |

## JRDB 5/9 feed 状況 (10:XX 確認)

```
TYB: HTTP 404 (未publish)
CYB: HTTP 200 (取得済、 5,401 bytes、 1 file)
KYI: HTTP 200 (取得済、 101,155 bytes、 1 file)
Bac: HTTP 200 (取得済、 1,170 bytes、 1 file)
Sed: HTTP 200 (取得済、 11,610 bytes、 2 files)
Paci: HTTP 200 (取得済、 597,825 bytes、 14 files)
```

→ TYB は当日 13:00 前後 publish のため、 後 retry。 13:00 後再実行で paddock_idx
取得可能、 当初設計通り重み 0.30 で integrated_score に反映可。

## JV-Link 当日体重 (SE)

5/9 当日 取得 skip (時間制約)。 後 retry の TYB 取得時 weight_diff も同時取得。

## 重み 再正規化 (TYB なし版、 E 段階で利用)

| feature | 重み (TYB なし) | 重み (TYB あり、 13:00+ 再実行時) |
|---------|----------------|----------------------------------|
| training_idx | 0.30 | 0.20 |
| idm_score | 0.25 | 0.15 |
| gekiso_idx | 0.20 (paddock 代替) | 0.10 |
| stable_idx | 0.15 | 0.10 |
| ninki_idx | 0.10 | 0.05 |
| paddock_idx | (0.00 NaN) | **0.30 ★** |
| weight_diff | (0.00) | 0.10 |
| 計 | 1.00 | 1.00 |

## 出力

- csv: `data/v18/horse_numeric_features_5_9.csv` (90 行)
- columns: race_id, course, race_num, race_name, umaban, horse_name,
  paddock_idx, padock_mark, weight_diff, horse_weight,
  training_idx, idm_score, stable_idx, ninki_idx, gekiso_idx,
  train_eval, train_mark, tansho_odds, *_source

## TYB 13:00 retry 手順 (任意)

```bash
# 13:00 以降に再実行で paddock_idx 取得
python -c "
import os, requests, zipfile, io
from requests.auth import HTTPBasicAuth
env = {l.split('=',1)[0]: l.split('=',1)[1].strip().strip('\"').strip(\"'\") for l in open('.env','r',encoding='utf-8') if '=' in l and not l.startswith('#')}
auth = HTTPBasicAuth(env['JRDB_ID'], env['JRDB_PASSWORD'])
r = requests.get('http://www.jrdb.com/member/datazip/Tyb/2026/TYB260509.zip', auth=auth, timeout=20)
print('TYB:', r.status_code)
if r.status_code == 200:
    with open('data/jrdb/raw/Tyb/TYB260509.zip','wb') as f: f.write(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z: z.extractall('data/jrdb/extracted/Tyb')
    print('extracted')
"
python tools/horse_numeric_features.py
python tools/horse_total_score_v63.py
```
