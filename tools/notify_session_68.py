"""Session #68 E: Discord 1 通 (Stage 2 修復完了通知).

dedup logic: data/discord_dedup_state.json の dedup_key で重複防止。

usage:
  python tools/notify_session_68.py
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
    dedup_key = "session68_complete"
    state = _load_dedup()
    if state.get(dedup_key):
        print(f"[skip dedup] {dedup_key}")
        return

    title = "Session #68 完了 (Stage 2 エラー修復)"
    body = """## Session #68 完了 (Stage 2 エラー修復)

**root cause**: netkeiba HTTP 400 (server block) — Session #62/63 既知の block 継続。
parse_shutuba 全 endpoint で空 body → predict_one_race None return → 5/9 13:00-15:30
fire 17/17 全失敗 (成功 0%)。

### A: stage2_predict.py audit
log 384 行 解析、 race-specific bug ではなく全 R 失敗を確認。

### B: root cause
netkeiba HTTP 400 を全 endpoint × 全 header (User-Agent / Cookie / 完全 Chrome) で再現。
client 側修正 不可、 server block 解除待ち。

### C: 修復実装 (tools/stage2_predict.py 5 項目)
1. _probe_netkeiba(): HTTP 診断 関数 追加
2. error_kind 分類 (netkeiba_block / shutuba_empty / exception / None)
3. Discord body 改善: 「Stage 1 fallback 採用」 + 診断行
4. 失敗時 cache 書込みスキップ → 次 fire で再試行可能
5. fire 起動時 1 回 probe → block 検知時 全 R skip + 警告 1 通

manual 動作確認 済 (5/9 16:57)、 17/17 tests PASS。

### D: stage_compare 拡張
3 系統 hit rate 並記:
- 系統 1: morning_only (全 R baseline)
- 系統 2: stage2_success_only (Stage 2 効果測定)
- 系統 3: integrated (s2 成功は s2、 失敗は morning fallback、 ★実運用方針)

### E: dev/two-stage push
4 commits (140759ac / 3fb7893d / 911ab4fc / 892f865e) → push 済。
※ 3fb7893d は parallel agent 干渉により file が含まれず、 911ab4fc で再 commit。

### 5/16 V18 trial への含意
- block 解除前: Stage 2 全 R fallback、 朝予測のみ運用 (V15 不変)
- block 解除後: probe で検知 → 個別 R 予測 自動再開、 系統 2 で効果測定可
- どちらでも V15 投資 完全保護、 V18 trial 妨害なし

★ V15 投資保護 (絶対遵守) ★
- 累計 +13,530 円 死守
- 5/9 投票: 新潟 12R ¥700 既完了 (案B改 strict)
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
