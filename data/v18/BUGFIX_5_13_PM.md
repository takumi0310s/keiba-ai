# 5/13 PM bug 一括修正 (Session #88 続き)

user 依頼: "障害レースを除く 足りないもの、 修正点をまとめて修正と実装、 テストまで自動で行って"

## 修正 4 件

### 1. nightly_sanity SCRAPER-GUARD 誤検知 (修正済)

**file**: `tools/nightly_sanity_check.py` L72-80

**症状**: PowerShell schtask query (UTF-8 forced) を Python が cp932 で 読んで UnicodeDecodeError → 0 件解析 → SCRAPER-GUARD section 空

**修正**:
```python
r = subprocess.run(
    cmd, capture_output=True, text=True,
    encoding="utf-8", errors="replace", timeout=60,
)
```

**test**: PowerShell utf-8 で 5/14 (木) + 5/16 (土) 共に [OK] 全チェック PASS。

### 2. cumulative_results.csv top1_num/score 95% 欠損 (修正済)

**file**: `tools/daily_results.py` L587-617 (result_row) + L348-368 (csv loader)

**症状**: result_row dict に top1_num / top1_name / top1_score / top2_* / top3_* が含まれず、 cumulative_results.csv の該当列が NaN。

**修正**: result_row に 9 列追加 (top1_num/name/score, top2_num/name/score, top3_num/name/score)。 csv loader も対応。 DB loader は score なし (DB schema 制約) のため 0 default 維持。

**test**: pandas concat sanity test で 既存行 NaN + 新行 値あり 動作確認。 次回 daily_results 実行で 累積 csv に反映。

### 3. predict_core.py FutureWarning 5 箇所 (修正済、 logic 不変)

**file**: `tools/predict_core.py` L1676, L1827, L1844-1846, L1989-1990, L2363

**症状**: pandas 2.2+ で `.replace(0, scalar)` の downcasting deprecation warning。

**修正**: 全て `.mask(s == 0, value)` に置換。 logic 数学的等価:
- `.replace(0, x)` = where s==0, set to x
- `.mask(s == 0, x)` = where mask True (s==0), set to x

`rank(method='min')` は 維持 (API 変更なし)。

**test**:
- `python -W error::FutureWarning -c "import predict_core"` → 警告 raise なし
- `python tools/predict_one_race.py 202605020211` → 予測 動作 OK、 TOP3 ラベルセーヌ (0.741) / ラフターラインズ (0.703) / エンネ (0.559)
- syntax compile OK

**V15 production 完全保護 確認**: keiba_model_v15_central_live.pkl.gz は 不変。 inference logic は 値の置換のみ で 出力 完全同一。

### 4. V22 IR fold 22 collapse 対策 (trainer 修正済、 retrain は 5/24+ で実施)

**file**: `train/train_v22_4ensemble.py` L147-180 (train_intra_race) + L342-345 (caller)

**症状**: V22 4-ens full WF で fold 22 のみ IR が 0.7765 に collapse (通常 0.87+)。 mean Grid 0.8800 が V15 0.8939 に -0.014 届かず。

**修正**:
- epochs 20 → 30 (early-stop 余裕)
- patience 5 → 10 (val noise robust)
- d_model 64 → 128 (representation 拡大、 GPU 16GB 余裕)
- explicit `torch.manual_seed(42 + y_lo)` (fold ごと seed 変更で 初期重み instability 軽減)
- np.random / Python random も seed 統一

**期待効果**: fold 22 IR collapse 回避 → mean Grid 0.880 → 0.888-0.892 想定 (V15 0.8939 接近)。

**test**: syntax check のみ (retrain 自体は 93min GPU 必須、 5/24+ で 実行判断)。

---

## 障害レース 対応

user 依頼 "障害レースを除く" → daily_results.py の csv loader は 既に `if '障' in surface: continue` で除外済 (L346-347)。 追加対応 不要。

## V20/V21 leak audit + Strategy 8 / V22 design memory

既に 5/13 朝 memory に 保存済:
- `horse_id_mapper.md`
- `v20_v21_leak_audit.md`
- `strategy_8_verified.md`
- `v22_design.md`

## 残課題 (Phase 3 / JV-Link 待ち)

- jrdb_sed.csv 2026 年分 未取得 (SED LZH 単日 endpoint なし、 JRA-VAN SE で代替予定 5/24+)
- jrdb_paci.csv 4/4 停止 (scrape_jrdb_paci.py で修復可能、 next run 待ち)
- jra_payouts.csv 4/6 停止 (JV-Link HR で代替予定)

これらは ネットワーク 取得問題で コード修正だけでは 解決しない。

## V15 production 完全保護 (本日も遵守)

- V15 .pkl.gz / app.py / daily_predict.py / race_auto_notify.py **完全不変**
- predict_core.py は FutureWarning suppression のみ、 inference 出力 完全同一
- V22 train script のみ修正、 V22 .pkl.gz は 別 file
- 累計収支 +13,530 円 守る

## 5/16 (土) 本番運用 (変更なし)

V15 戦略⑦ 案B改 単独継続 + Strategy 8 sidecar shadow eval (別 Discord channel)。
