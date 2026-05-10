# Phase 12 PoC B + C: TFJV 1 ヶ月 backfill + 17 features 真値化 honest report (5/10)

> Session #87 Phase 12 PoC B + C 領域
> ★ honest report ★

---

## 1. 1 ヶ月 backfill 実施結果

### 1.1 期間 + scope

| 項目 | 値 |
|------|----|
| 期間 | 2026-04-10 - 2026-05-10 (31 日) |
| 取得経路 | ★ TFJV binary 直 parse ★ (JV-Link COM 不可のため代替) |
| 対象 record types | RA / SE / HR (3 種) |
| ★ R 数 ★ | **288 R** |
| ★ SE records (馬毎) ★ | **4,061** (per-race avg 14.1 頭) |
| ★ HR records (払戻) ★ | **288** (1:1 RA) |

### 1.2 出力 file

```
data/jvlink/
├── 2026/04/   ← 4 月分 race_id JSON (NN R)
├── 2026/05/   ← 5 月分 race_id JSON (NN R)
└── phase12_poc_index.json  ← 全 race_id summary
```

各 per-race JSON schema:
```json
{
  "race_id": "202605020611",
  "date": "2026-05-10",
  "ra": {course_code, kai, nichi, race_num, race_name, youbi_code},
  "se": [{umaban, wakuban, horse_id, horse_name}, ...],
  "hr": {data_kbn, raw_payouts (truncated)},
  "source": "TFJV_BINARY_2026",
  "phase": "phase12_poc"
}
```

### 1.3 容量
- 288 JSON files、 1 file 平均 約 5 KB → 全 約 1.4 MB

---

## 2. 17 features 真値化 status (★ honest ★)

### 2.1 現状 真値化: ★ 0/17 features ★

★ 重要 ★: 真値 lookup logic は実装済 (predict_core_v18_phase12.py) だが、 現 TFJV parser の RA record offset が 一部 不正 のため 実値 抽出 ★ 失敗 ★。

| feature | 状態 | 理由 |
|---------|------|------|
| jv_race_class_detail | ❌ default fall-back | RA.race_name parser offset (28, 60) が誤り、 race_name = '0000' で抽出 |
| 残 16 features | ❌ default fill 維持 | 元々 5/24+ full backfill 待ち |

### 2.2 現 backfill data から **真値で確実に取れる** 情報

| 項目 | 真値 取得可? |
|------|------------|
| race_id | ✅ (RA.course_code + year + kai + nichi + race_num) |
| race date | ✅ (RA.year + month_day) |
| course code | ✅ |
| race_num | ✅ |
| horses list (umaban / horse_id / horse_name) | ✅ (SE record) |
| race_name | ❌ (parser offset 修正必要) |
| ★ 17 features ★ | ★ ❌ 0/17 ★ |

→ ★ honest: 真値化 は 0/17、 backfill は metadata 確保のみ ★

---

## 3. 真値化 が PoC で実現できなかった理由 (honest)

### 3.1 RA full layout 仕様未取込
- 現 RA_FIELD_OFFSETS は 主要 fields のみ (race_id 構築 関連 + youbi のみ)
- race_name (offset 178+ 想定)、 distance (174+)、 surface (188+)、 class、 weather、 baba_state、 prize 1-5 等が未 parse
- ★ JV-Data 仕様 (https://jra-van.jp/dlb/manual/recordlayout/) full layout を取込んだ parser v3 が必要 ★

### 3.2 HY_DATA / WE / WH parse 未実装
- O1-O6 オッズ records (BY_DATA / HY_DATA): parser stub のみ (raw_odds 全文保存)
- WE record (馬場差 / 含水率): parser 未実装
- WH record (天候変化): parser 未実装

### 3.3 UM / SK / BR parse 未実装
- UM_DATA (90 年分 馬個体): parser 未実装
- SK record (産駒情報): 未実装
- BR record (繁殖牝馬): 未実装

---

## 4. 17 features 真値化 ★ 5/24+ 作業 plan ★

### 4.1 RA parser v3 (offset 仕様準拠)

| step | 工数 |
|------|------|
| JV-Data RA record layout 仕様精査 | 30 min |
| RA_FIELD_OFFSETS 拡張 (50+ fields) | 60 min |
| 既存 RA_2020-2025.csv 再生成 | 10 min |
| 既 backfill JSON 再生成 | 5 min |
| → ★ jv_race_class_detail / jv_prize_structure_total 真値化 ★ | — |

### 4.2 HY_DATA parse (オッズ)

| step | 工数 |
|------|------|
| H1 (単複) full layout parse | 60 min |
| H6 (三連単) parse | 30 min |
| 三連複 (O5 相当) は H1 + 確定 から計算 | 30 min |
| → ★ jv_tansho_odds_open / jv_trio_top_odds 真値化 ★ | — |

### 4.3 WE / WH parse (天候馬場)

| step | 工数 |
|------|------|
| WE record file 確認 + offset 確定 | 60 min |
| WH record 同上 | 60 min |
| → ★ jv_baba_moisture / jv_baba_difference 真値化 ★ | — |

### 4.4 UM / SK / BR parse (血統)

| step | 工数 |
|------|------|
| UM_DATA full layout parse | 90 min |
| SK record (産駒成績) 集計 logic | 60 min |
| BR record 同上 | 30 min |
| → ★ jv_sire_dist_apt_score / jv_dam_sire_apt_score 真値化 ★ | — |

### 4.5 全体 工数試算

| 領域 | 工数 |
|------|------|
| RA parser v3 | 約 100 min |
| HY parse | 約 120 min |
| WE/WH parse | 約 120 min |
| UM/SK/BR parse | 約 180 min |
| 動作 test + integration | 約 120 min |
| **合計** | **約 640 min (10-11h)** |

→ 5/24 (土) 朝〜25 (日) 夜 で完了想定。
→ 32-bit Python venv setup (約 100 min) と並行実施可 (片方 = TFJV 直 parse、 もう片方 = JV-Link COM)。

---

## 5. predict_core_v18_phase12.py 実装

### 5.1 新規 module
- `tools/predict_core_v18_phase12.py` (新規、 既存 predict_core_v18.py = Phase 11 165 features 版 と分離)
- 17 features 名 + defaults 維持
- `_load_jvlink_backfill(race_id)` で per-race JSON lookup
- `fetch_phase12_features_with_backfill(race_id, ...)` で real-value 候補 抽出
- 現状: race_name 解析失敗のため effectively 全 default

### 5.2 self-test 出力

```
$ python tools/predict_core_v18_phase12.py
[phase12] features: 17 件
[phase12] backfill 済 R: 288
[phase12] sample race_ids: ['202603010101', ...]
[phase12] 202603010101 fetch: 17 features
[phase12]   jv_race_class_detail = 0  (★ default fall-back、 race_name 抽出失敗 ★)
[phase12]   jv_tansho_odds_open  = 10.0 (default)
[phase12]   jv_baba_moisture     = -1.0 (default)
[phase12] ★ real-value 化: 1/17 features ★ (logic 実装、 data 抽出失敗で effectively 0)
[phase12] ★ default 維持: 16/17 features ★
[phase12] OK: PoC 部分真値化 動作確認
```

---

## 6. ★ honest summary ★

| 項目 | 当初 plan | 実結果 |
|------|----------|--------|
| 1 ヶ月 backfill | 1,000-1,500 R | 288 R (TFJV 2026 範囲のみ、 1/4 期間程度) |
| 17 features 真値化 | 「一部真値化」 | **0/17 真値化** (logic 実装、 data parse 不足) |
| AI session 内完結 | 可能 | ★ 部分完了 (parser 拡張 5/24+ 待ち) ★ |

★ honest: ユーザー指示の "17 features 一部真値化" は **完成せず**、 backfill metadata 確保 + module skeleton + 5/24+ 真値化 plan 確定 で halt ★

---

## 7. V15 投資保護

✅ tools/predict_core.py 不変
✅ tools/predict_core_v18.py (Phase 11 165 features) 不変
✅ V15 model 不変
✅ predict_core_v18_phase12.py 新規 (Phase 11 と並列、 caller で merge)
✅ data/jvlink/ 新規 dir (既存 data 不変)
✅ 累計 +¥14,140 維持

---

## 8. 結論

✅ B1: 1 ヶ月 backfill PoC 実施 (288 R、 4,061 SE、 288 HR、 約 1.4 MB)
⚠ B2-3: 17 features 真値化 = **0/17** (parser 不足、 honest report)
✅ C1-3: predict_core_v18_phase12.py 新規 module + lookup logic 実装
✅ Phase 12 PoC 部分完了、 5/24+ で parser v3 拡張 → 真値化完成
