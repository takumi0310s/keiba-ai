"""Session #73: 5/10 朝 dry run 完了通知 (Discord 1 通 + dedup).

dedup_key: session73_complete (重複防止)
channel: updates (システム通知)
color: green (成功) / yellow (要対応あり)

Session #73 主要 finding:
- A: 5/10 manual dry run → V15 model load OK、 SaveAllHorseScores_0930 動作見込み
- B: stage2_predict.py 5/9 hardcode → 5/10 動作不能 (要対応)
- C: schtasks 52 件、 5/10 朝 主要 task (DailyPredict/SaveAllHorseScores) Ready
- D: failure runbook (docs/RUNBOOK_5_10_DRY_RUN.md)

usage:
  python tools/notify_session_73.py
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
DEDUP_KEY = "session73_complete"


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


def main() -> int:
    state = _load_dedup()
    if state.get(DEDUP_KEY):
        print(f"[skip dedup] {DEDUP_KEY} already sent at {state[DEDUP_KEY]}")
        return 0

    title = "Session #73 完了 (5/10 朝 dry run)"
    body = """## Session #73 完了 (5/10 朝 dry run + manual fire test)

### A: 5/10 manual dry run
- V15 Pattern B model load OK (150 features)
- save_all_horse_scores --dry-run 動作確認 (graceful exit on no-csv)
- 5/9 1 R inference 試験 (parse_shutuba 既終了 R 仕様確認)
- doc: data/v18/session_73_dry_run.md

### B: stage2_predict 動作試験
- ★ stage2_predict.py 5/9 完全 hardcode 確認 (DATE/CACHE/RACE_TIMES/出力名)
- ★ pre_race_predict_runner.bat dev/two-stage 在中、 main 不在
- → 5/10 朝 PreRacePredict_Watchdog_5_9 silent fail 見込み
- 影響: Session #72 1h 前通知機能 完全停止 (V15 投資保護に影響なし)
- doc: data/v18/session_73_stage2_test.md

### C: schtasks 5/10 fire 確認
- 総 task 数: 52 件
- ★ DailyPredict 8:00 (Ready)
- ★ SaveAllHorseScores_0930 9:30 (Ready、 Session #71 動作見込み)
- RaceAutoNotify_Sun 8:45 (Ready)
- MorningWeightCheck_Sun 9:30 (Ready)
- doc: data/v18/session_73_schtasks_check.md

### D: failure runbook
- 6 case 対応手順 doc 化
  1. DailyPredict 8:00 失敗
  2. SaveAllHorseScores_0930 失敗
  3. PreRacePredict_Watchdog_5_9 (★ 確定 silent fail ★)
  4. Discord 通知届かない
  5. V15 model 読み込み失敗
  6. process_watchdog kill-switch 誤発動
- doc: docs/RUNBOOK_5_10_DRY_RUN.md

### dev/training-poc commits
- 5 commits (A/B/C/D/E)
- main 不変、 V15 model 不変

### 5/10 朝の状態 (要 attention)
- ✓ DailyPredict / SaveAllHorseScores / RaceAutoNotify 健在
- ⚠ PreRacePredict_Watchdog_5_9 silent fail (補助通知のみ、 投資保護 OK)
- 推奨: 5/9 内に PreRacePredict_Watchdog_5_9 disable
  `schtasks /Change /TN "\\Keiba-PreRacePredict_Watchdog_5_9" /DISABLE`

★ V15 投資保護 完全 ★
- main HEAD 5f5c3d43 不変
- V15 model 不変
- predict_core / daily_predict / app.py / race_auto_notify 不変
- 累計 +13,530 円 死守
"""

    color = "yellow"  # PreRacePredict 要対応 のため yellow
    ok = send_discord(title, body, color=color, channel="updates")
    if ok:
        state[DEDUP_KEY] = datetime.now().isoformat()
        _save_dedup(state)
        print(f"[sent] {DEDUP_KEY}")
        return 0
    else:
        print(f"[fail] {DEDUP_KEY}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
