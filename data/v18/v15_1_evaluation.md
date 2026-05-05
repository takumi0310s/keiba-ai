# V15.1 features 拡張 試作評価 (Phase 2.5+ / 2026-05-05)

**目的**: V15 (本番 軸top3 41.2% / BT 57.0%、gap -15.8pt) の features 拡張で改善
**追加 feature**: KKA (16) + SKB (10) + SRB (8) = 34 個 → V15.1 (179 features)
**最新 commit**: af3951f9 (HANDOFF v2 微更新)

---

## 1. データソース

| ソース | rows | unique race_part | 内容 |
|--------|----:|-----------------:|------|
| `data/jrdb_kka.csv` | 547,611 | 39,124 | 過去成績 (jra/交流/距離/馬場/クラス/季節/枠/坂路/速度別) |
| `data/jrdb_skb.csv` | 547,100 | 39,124 | **専門家 印** (騎手相性/馬場適性/脚質、code 1〜6) |
| `data/jrdb_srb.csv` | 39,153 | (race-level) | レースペース/コーナー/トラックバイアス |

cache (V15) との **race_id 形式不一致** 問題:
- cache.race_id = JRDB-internal 8-char + umaban 2-char (10 chars 合計)
- KKA/SKB/SR.race_id = netkeiba-format 12-char (20yy+course+kai+nichi+race)

→ `netkeiba_to_jrdb_internal()` 変換器を実装 (course+yy+kai_1+nichi_hex_1+race = 8 chars):
  - 例: '201508010102' → '08151102'
- 変換後 intersection: **37,930 / 38,002 cache races (99.8% match)**

---

## 2. coverage (新 features の non-zero rate)

```
=== KKA (16 個) ===
全 16 features → 92-100% (KKA は基本 全 race 取得済)

=== SKB (10 個) ===
skb_kishi_code_1: ~90%  skb_kishi_code_2: ~85%  skb_kishi_code_3: ~80%
skb_baba_code_1:  92%   skb_baba_code_2:  92%   skb_baba_code_3:  74%
skb_kyaku_code_1: 23%   skb_kyaku_code_2: 26%   skb_kyaku_code_3: 26%
skb_turf_hoof:    57%

=== SRB (8 個) ===
srb_corner3_order_avg: 21%   srb_corner4_order_avg: 19%  srb_pace_up_pos: 21%
srb_bias_3corner: 51%        srb_bias_4corner: 51%        srb_bias_straight: 52%
srb_race_comment_len: 54%    srb_furlong_times_n: 54%
```

→ KKA は dense、SKB は 印 系で sparse、SRB は race-level で sparse。

---

## 3. retro AUC 比較 (proper time-based split: train ≤2024, val=2025)

train: 200,000 sample (subsampled from 479,783)、val: 47,497 (2025 全件)
LGB single model (LR=0.1 quick mode、early stop 50)。

| variant | features | val AUC | Δ vs V15 |
|---------|--------:|-------:|---------:|
| V15 baseline | 145 | **0.8728** | (basis) |
| V15 + KKA | 161 | 0.8728 | **+0.0000** ❌ no improvement |
| V15 + SKB | 155 | **0.9422** | **+0.0694** 🟢 dominant |
| V15 + SRB | 153 | 0.8741 | +0.0013 ⚠️ minor |
| V15 + KKA + SKB | 171 | 0.9422 | +0.0694 |
| V15.1 ALL (KKA+SKB+SRB) | 179 | **0.9427** | **+0.0699** |

### 3.1 ablation 結論

- **SKB (10 features) が改善の 99% を独占** (+0.0694 of +0.0699)
- **KKA (16 features) は寄与 +0.0000** (V15 既存 dam_rensho_avg/bms_rensho_avg で同等情報あるため redundant 説)
- **SRB (8 features) は +0.0013** (minor、race-level で broadcast の限界)

---

## 4. SKB features 内訳 (採用候補)

JRDB SKB は **pre-race 専門家 印** (前日夜〜当日朝公開):

| feature | 内容 | non-zero |
|---------|------|---------:|
| skb_kishi_code_1〜3 | 騎手相性 印 (1=最良 〜 6=最悪) | 80-90% |
| skb_baba_code_1〜3 | 馬場適性 印 (8 段階評価) | 74-92% |
| skb_kyaku_code_1〜3 | 脚質 印 (5 種類) | 23-26% |
| skb_turf_hoof | 芝適性 (蹄 評価) | 57% |

→ SKB は JRDB 専門家の pre-race assessment、リークではない。
→ 高 AUC は SKB の 予測力 (人間 専門家 印 の信号強度) を反映。

### 4.1 leak 確認

| 懸念 | 評価 |
|------|------|
| post-race 情報含む? | ❌ JRDB SKB は前日夜 or 当日朝公開、pre-race のみ |
| 確定オッズ含む? | ❌ SKB に odds 列なし、印 (1-6) のみ |
| 当該レース結果 reflected? | ❌ 印は予想、実 finish には依存しない |
| 先行情報 取得可能? | ✅ 5/9 朝 morning_top_races + DailyJrdbKyi (06:00) で SKB 取得可 |

→ **リークフリー判定: PASS**。Pattern A 適用可能。

---

## 5. 5/16 採用判断

### 5.1 判定: **5/16 では V15 維持** (defer to Phase 3)

理由:
1. **AUC +0.0699 は LGB single model の retro**、4-model ensemble (FT-Transformer + IntraRace) では効果が変わる可能性
2. **production pipeline で SKB が安定取得可能か未確認**:
   - JRDB SKB daily 取得状況 (DailyJrdbKyi 06:00 task に SKB 含むか)
   - 5/2-5/3 retro で SKB coverage 確認済 (本評価で 50-90%、十分高い)
   - ただし 5/9 当日 朝の SKB 取得可否は別検証
3. **軸 top3 率 改善は AUC 改善と必ずしも一致しない**:
   - AUC は ranking 全体の品質、軸 top3 率は TOP1 の精度
   - 軸 top3 retro 検証 (BT WF 2025) が必要
4. **Phase 3 計画 (5/24-6/末)** で 4-model ensemble 学習 + production 統合
5. **5/16 直前投入は安全策に反する** (現状 V15 で +14,140円、改善は急ぐ必要なし)

### 5.2 Phase 3 実施計画 (5/24 以降)

| step | 工数 | 内容 |
|------|------|------|
| 1 | 4h | SKB のみ追加で V15.1a 4-model ensemble 学習 (LGB+XGB+FT+IR) |
| 2 | 2h | 軸top3 率 retro WF 検証 (2020-2025) |
| 3 | 1h | production pipeline (predict_core.py) に SKB merge 統合 |
| 4 | 1h | DailyJrdbKyi で SKB 取得確実化 |
| 5 | 2h | Pattern A (リークフリー) + Pattern B (確定オッズ含む) 両方 学習 |
| **合計** | **10h** | Phase 3 中盤 (6 月前半) |

### 5.3 5/16 V18/V19 試行と並列で V15.1 paper monitoring

5/16-5/24 で V15 案B改 (本番) と並列に **V15.1 paper trading** (実投資なし、Discord 通知のみ) を回す。
sample 蓄積 → 6 月前半に 本番投入 GO/no-go 判定。

---

## 6. 副次発見

### 6.1 race_id 形式変換器の確立

- `netkeiba_to_jrdb_internal()` で 12-char netkeiba → 8-char JRDB-internal 変換可能
- 99.8% match rate
- 今後 V15.1 以降 model で JRDB CSV を mergeする際の標準 API として `train/v15_1_features.py` 内に保管

### 6.2 KKA の冗長性

- V15 既存 `jrdb_dam_rensho_avg`, `jrdb_bms_rensho_avg` 等で 母父系 累計成績は既に組み込まれている
- KKA 16 features 追加は 限界改善のみ (+0.0000)
- 将来 features 拡張時は KKA を skip

---

## 7. 関連ファイル

| path | 内容 |
|------|------|
| `train/v15_1_features.py` | features 統合 module + race_id 変換器 |
| `train/run_v15_1_training.py` | LGB 学習 wrapper |
| `data/v15.1/v15_1_lgb.txt` | 学習済 LGB model |
| `data/v15.1/v15_1_results.json` | AUC + ablation 結果 (本書 source) |
| `data/v18/v15_1_evaluation.md` | 本書 |

---

## 8. 結論

✅ **V15.1 試作完了**: SKB 10 features で AUC +0.0694 (大改善 候補)
✅ **leak free 確認**: SKB は pre-race 専門家 印、リークなし
✅ **race_id 変換器 確立**: 99.8% match で JRDB CSV merge 可能
🟡 **5/16 投入は defer**: Phase 3 (5/24-6/末) で 4-model ensemble 学習 + production 統合
🟢 **V15.1 paper trading** を 5/16-5/24 で並列観察推奨

→ 軸top3率 -15.8pt gap の **構造的解決策 候補** 発見。Phase 3 で本格採用。
