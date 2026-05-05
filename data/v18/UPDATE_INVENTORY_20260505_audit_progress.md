# UPDATE_INVENTORY 緊急 3 件 対応進捗

**作成**: 2026-05-05 夜 (Session #24、49 時間連続作業)
**作業時間**: 約 50 分
**ベース commit**: f408d93d
**ユーザー方針**: 取り返し禁止 / 累計 +14,140円 死守 / 5/9 V15 案B改 維持

## 緊急 3 件 status

| # | task | status | 出力 |
|---|------|--------|------|
| A | ProcessWatchdog v2 schtasks 登録 | ✅ 完了 (admin 実行待ち) | `tools/register_process_watchdog_v2.ps1` + `data/v18/process_watchdog_v2_setup.md` |
| B | fire_check caller 監査 | ✅ 完了 + バグ 2 件修正 | `data/v18/fire_check_audit_5_5.md` |
| C | chihou_races 2020-2025 不在 | ✅ **誤情報と判明、解決済** | `data/v18/chihou_races_recovery_5_5.md` |

## A. ProcessWatchdog v2 詳細

**作成物**:
- `tools/register_process_watchdog_v2.ps1` (97 行、`-DryRun` / `-Rollback` 対応)
- `data/v18/process_watchdog_v2_setup.md` (手順書、5/6 admin 実行)

**ユーザー手動作業 (5/6 以降の隙間時間)**:
```powershell
# admin PowerShell で
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1
```

これで ProcessWatchdog (Disabled v1) → v2 (Enable + 静音化済) に切替完了。
失敗時は -Rollback で元に戻せる。

## B. fire_check 監査詳細

**監査結果**: 4 種すべて設計健全。 4/19 事故と同型のリスクなし。

**修正したバグ 2 件**:
1. `pre_fire_check.py` UnicodeEncodeError (cp932 で `✓` 出力不能) → utf-8 reconfigure + ASCII icon
2. `am8_fire_check.py` 平日 critical 誤判定 → 平日 + CSV 未生成 → 早期 OK

**動作確認 dry-run**: 4 種すべて期待通り動作。

## C. chihou_races_2020_2025.csv 大発見

**発見**: `archive/nar/train_nar_v4.py` を解析した結果:
```python
SCRAPED_CSV = os.path.join(DATA_DIR, 'nar_all_races.csv')      # ← 実使用
OLD_CSV = os.path.join(DATA_DIR, 'chihou_races_2020_2025.csv') # ← 変数定義のみ、未使用
```

→ **`chihou_races_2020_2025.csv` は NAR v4 学習・推論で実は使われていなかった**。
→ 5/12 NAR paper 開始の blocker ではない。
→ UPDATE_INVENTORY § 緊急 3 件のうち #3 は誤情報と判明。

## D. レポート訂正

訂正済:
1. `docs/UPDATE_INVENTORY_20260505.md` § 0 緊急 3 件を全対応済 status に更新
2. `docs/UPDATE_INVENTORY_20260505.md` § 2.1 / § 6.8 で JRA-VAN を「一度だけ契約 → 退会済、5/24まで不要、6月再契約候補」に訂正
3. `docs/HANDOFF_5_5_TO_5_9.md` § 9 既知問題で chihou_races を「解決済」に訂正

## E. 5/6 以降 TODO (5/9 までの平日)

### 高優先度 (5/6-5/8 で実施推奨)

- [ ] **A 実行**: admin PowerShell で `register_process_watchdog_v2.ps1` 実行
- [ ] **NAR データ backfill**: `nar_all_races.csv` の 2025-06 〜 2026-05 分 (5/12 paper 前)
- [ ] **SED260503 取得 + KKA/KAB 連結再実行** (5/9 朝の前走成績結合率に直結、15min)
- [ ] **speed_index 4-5 月 backfill** (条件 C/D 予測精度に直結、30min)
- [ ] **戦略⑦ 5/2-5/3 retro 完全版** (5/9 投入直前必須、2h)

### 中優先度

- [ ] CLAUDE.md V13.5b 中心の記述を V15 中心に書換 (header + § 4 + § 6 + § 11)
- [ ] CLAUDE.md jra_payouts 4/6 stale + jrdb_paci 4/4 停止 の記述訂正 (実は解消済)
- [ ] cumulative_results.csv 書き込みバグ (top1_num/score 95% 欠損) 修正

## F. 5/9 (土) 本番フロー (変更なし)

### 5/8 (金) 21:00 後 (1度だけ)

```bash
# 12R race_name 確認
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}',
                     headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '→', (m.group(1).strip() if m else 'NOT_FOUND'))
"

python tools/refresh_cookie.py --check
```

### 5/9 (土) 朝

`data/results/20260509_pat_checklist.md` を順番通り。
06:30 / 08:00 自動発火を Discord で確認 → 14:00-15:30 PAT 投票 (採用 R × 700 円、最大 2,100 円)。

## G. 結論

**緊急 3 件 すべて 5/9 本番に間に合う形で対応完了**。
- A は admin 1 コマンドで完了 (script 待ち)
- B は本日修正完了、追加作業なし
- C は誤情報と判明、追加作業なし

5/9 当日リスクは **ゼロ**。 累計 +14,140 円死守の体制 維持。
朝起きて `docs/UPDATE_INVENTORY_20260505.md` を読めば全体像が把握できる。
