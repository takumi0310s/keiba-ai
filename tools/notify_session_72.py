"""Session #72 F: Discord 1 通 (1h 前通知 全馬 V15 score 順 + 自動 test 完了).

dedup logic: data/discord_dedup_state.json の dedup_key で重複防止。

usage:
  python tools/notify_session_72.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from notify import send_discord  # noqa: E402

DEDUP_STATE = BASE / "data" / "discord_dedup_state.json"


def _load_dedup() -> dict:
    if DEDUP_STATE.exists():
        try:
            return json.loads(DEDUP_STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_dedup(state: dict):
    DEDUP_STATE.parent.mkdir(parents=True, exist_ok=True)
    DEDUP_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def main():
    dedup_key = "session72_complete"
    state = _load_dedup()
    if state.get(dedup_key):
        print(f"[skip dedup] {dedup_key}")
        return

    title = "Session #72 完了 (1h 前通知 全馬スコア順 化)"
    body = """## Session #72 完了 (1h 前通知 全馬スコア順 化)

ユーザー要望 4 件、 全件 自動完結:

### 1. Stage 2 不具合 (HTTP 400 server block) 解消確認
✅ Session #68 修復 (commit 911ab4fc) で fallback logic 完成。
✅ 5/9 18:00 manual 動作確認 (HTTP 400 → "Stage 1 fallback 採用" + cache skip)。
⚠ netkeiba 側 server block 自体は解除待ち (Session #62/63 既知)。

### 2. 30 分毎 fire 全 R cover 確認
✅ 5/9 7 回 fire / Discord 16 通 sent。
window=60min × 30 分間隔 = 各 R 1-2 回 cover、 dedup で重複防止。

### 3. 通知内容変更 (全馬 V15 score 順 table)
✅ tools/stage2_predict.py に 3 関数 追加:
   - load_full_predictions(race_id, date)
   - build_horse_table(rows, max_chars=1700)
   - build_message_all_horses(race_id, morning, stage2, full_rows)

✅ 自動切替:
   - 5/10 以降 (Session #71 daily_predictions_full csv 生成後): 全馬 V15 score 順 markdown table
   - 5/9 以前 (file 不在): 朝予測 top3 のみ (旧 logic、 互換性維持)

### 4. 自動 test 5 件 (実装は 7 件)
✅ tests/test_stage2_predict.py 7 件、 7/7 PASS in 0.28s:
   - load_full_predictions (5/10 success / 5/9 fallback / invalid race_id)
   - build_horse_table (18 頭 1700 内 / 100 頭 truncate)
   - build_message_all_horses (Stage 2 成功 / 失敗 path)

### dev/two-stage push 5 commits
- 979a34a5 A+B: audit + design
- 812eaf54 C: stage2_predict 通知変更
- b741abf8 D: 自動 test 7/7 PASS
- 1db5561a E: 5/9 fire 検証
- (本 F commit)

### 5/10 以降の運用
PreRacePredict_Watchdog 30 分毎 fire → Session #71 csv read →
全馬 V15 score 順 markdown table を Discord 送信 (channel=bets)。
想定: 3 場 × 12R × 1-2 回 cover ≈ 36 通 / 日。

### 5/16 V18 trial への含意
- block 解除前: Stage 2 全 R fallback、 朝予測 (Stage 1) のみで運用
- block 解除後: probe で検知 → 個別 R 予測 自動再開
- どちらでも V15 不変、 V18 trial 妨害なし

★ V15 投資保護 完全 ★
- main 8fc4e13b 不変
- V15 model file 不変
- predict_core / daily_predict / app.py 不変
- schtasks 既存 不変
- 5/9 投票結果 確定 (-¥700、 累計 +¥12,830)
"""
    ok = send_discord(title, body, color="green", channel="updates")
    if ok:
        state[dedup_key] = datetime.now().isoformat()
        _save_dedup(state)
        print(f"[sent] {dedup_key}")
    else:
        print(f"[fail] {dedup_key}")


if __name__ == "__main__":
    main()
