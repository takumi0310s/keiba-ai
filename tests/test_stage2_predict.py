"""Session #72 D: tools/stage2_predict.py 自動 test (5 件).

target:
  - load_full_predictions: file 不在 / race_id 不在 / 正常 (Session #71 連携)
  - build_horse_table: size 上限以下 / 超過時 truncate
  - build_message_all_horses: 5/10+ 全馬 table 通知 path
  - build_message: 5/9 以前 top3 fallback path

usage:
  pytest tests/test_stage2_predict.py -v

V15 production / predict_core / app.py に依存しない (mock + tmp_path)。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import stage2_predict as s2p  # noqa: E402


# ---------- helpers ----------

def _mock_morning_row(race_id: str = "202604010312",
                     n_horses: int = 14,
                     top1_num: str = "11", top1_name: str = "ハイクオリティ",
                     top1_score: float = 0.648):
    return pd.Series({
        "race_id": race_id,
        "course": "新潟",
        "race_num": "12",
        "race_name": "4歳以上1勝クラス",
        "num_horses": str(n_horses),
        "distance": "1200",
        "surface": "ダ",
        "track_condition": "良",
        "top1_num": top1_num,
        "top1_name": top1_name,
        "top1_score": top1_score,
        "top2_num": "12",
        "top2_name": "マテンロウミラクル",
        "top3_num": "8",
        "top3_name": "カレンラップスター",
        "trio_bets": "8-11-12; 8-11-7",
    })


def _mock_full_csv(tmp_path: Path, race_id: str, n_horses: int = 14,
                   long_names: bool = False) -> Path:
    """tmp_path に daily_predictions_full csv を生成、 file path を返す."""
    rows = []
    for i in range(n_horses):
        rank = i + 1
        umaban = i + 1
        name = (f"超々超々超々超超超々ホース{rank:02d}A" if long_names
                else f"ホース{rank:02d}")
        score = round(0.7 - i * 0.03, 4)
        odds = round(1.5 + i * 0.8, 1)
        rows.append({
            "race_id": race_id,
            "course": "新潟",
            "race_num": 12,
            "race_name": "4歳以上1勝クラス",
            "num_horses": n_horses,
            "distance": 1200,
            "surface": "ダ",
            "track_condition": "良",
            "horse_rank": rank,
            "umaban": umaban,
            "horse_name": name,
            "score": score,
            "odds": odds,
        })
    csv_path = tmp_path / "20260510.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


# ---------- test 1: 5/10 想定 (full predictions あり) ----------

def test_load_full_predictions_5_10_success(tmp_path, monkeypatch):
    """5/10 想定: daily_predictions_full csv あり → 全馬 dict list 取得 OK."""
    race_id = "202605100501"
    _mock_full_csv(tmp_path, race_id, n_horses=14)
    monkeypatch.setattr(s2p, "DAILY_PRED_FULL_DIR", tmp_path)

    rows = s2p.load_full_predictions(race_id, date="20260510")
    assert rows is not None
    assert len(rows) == 14
    # rank 順 sorted
    ranks = [int(r["horse_rank"]) for r in rows]
    assert ranks == list(range(1, 15))
    # 必須 column 存在
    for r in rows:
        for col in ("umaban", "horse_name", "score", "odds"):
            assert col in r
    # score range OK (0-1)
    scores = [float(r["score"]) for r in rows]
    assert all(0.0 <= s <= 1.0 for s in scores)


# ---------- test 2: 5/9 以前 (full predictions 不在) ----------

def test_load_full_predictions_5_9_fallback(tmp_path, monkeypatch):
    """5/9 以前 想定: daily_predictions_full file 不在 → None 返却."""
    monkeypatch.setattr(s2p, "DAILY_PRED_FULL_DIR", tmp_path)
    rows = s2p.load_full_predictions("202604010312", date="20260509")
    assert rows is None


# ---------- test 3: 不正 race_id ----------

def test_load_full_predictions_invalid_race_id(tmp_path, monkeypatch):
    """csv あるが race_id 不在 → None (fallback へ)."""
    _mock_full_csv(tmp_path, race_id="202605100501", n_horses=14)
    monkeypatch.setattr(s2p, "DAILY_PRED_FULL_DIR", tmp_path)
    rows = s2p.load_full_predictions("999999999999", date="20260510")
    assert rows is None


# ---------- test 4: 全馬 table size 上限 (18 頭 + 2000 char 内) ----------

def test_build_horse_table_size_within_limit():
    """18 頭で markdown table が DISCORD_BODY_SAFE_LIMIT 内 (1700 char 以下)."""
    rows = [
        {"horse_rank": i, "umaban": i, "horse_name": f"ホース{i:02d}",
         "score": round(0.7 - i * 0.02, 4), "odds": round(1.5 + i * 1.5, 1)}
        for i in range(1, 19)  # 18 頭
    ]
    table, n_trunc = s2p.build_horse_table(rows)
    assert n_trunc == 0, f"18 頭で truncate 発生 (n_trunc={n_trunc})"
    assert len(table) <= s2p.DISCORD_BODY_SAFE_LIMIT
    # 全 18 頭 row 存在
    body_rows = [ln for ln in table.split("\n") if ln.startswith("| ") and "馬名" not in ln and "----" not in ln]
    assert len(body_rows) == 18


def test_build_horse_table_truncates_when_over_limit():
    """十分長い名前で size 超過時 truncate + (以下 N 頭省略)."""
    rows = [
        {"horse_rank": i, "umaban": i,
         "horse_name": f"超超々超超々超超々ホース{i:02d}",
         "score": 0.5, "odds": 99.9}
        for i in range(1, 100)  # 100 頭 (異常 多)
    ]
    table, n_trunc = s2p.build_horse_table(rows, max_chars=500)
    assert n_trunc > 0, "size 超過で truncate されない"
    assert "(以下" in table and "頭省略" in table


# ---------- test 5: build_message_all_horses E2E (5/10+ 通知) ----------

def test_build_message_all_horses_success_path(tmp_path, monkeypatch):
    """5/10+ 全馬 table 通知 (Stage 2 成功 path)."""
    race_id = "202605100501"
    _mock_full_csv(tmp_path, race_id, n_horses=12)
    monkeypatch.setattr(s2p, "DAILY_PRED_FULL_DIR", tmp_path)
    monkeypatch.setitem(s2p.RACE_START_TIMES, race_id, "15:30")

    full_rows = s2p.load_full_predictions(race_id, date="20260510")
    morning = _mock_morning_row(race_id=race_id, n_horses=12, top1_num="1")

    stage2_ok = {
        "race_name": "test",
        "rinfo": {},
        "top3": [{"umaban": "1", "name": "ホース01", "score": 0.7}],
        "n_horses": 12,
        "error": None,
        "error_kind": None,
    }
    title, body, color = s2p.build_message_all_horses(race_id, morning, stage2_ok, full_rows)
    assert "全馬 V15 score" in title
    assert "全馬 V15 score 順" in body
    assert "Stage 2 状況 (成功)" in body
    assert color == "blue"
    # Discord 2000 char 上限 (title + body)
    assert len(title) < 256
    assert len(body) < 2000


def test_build_message_all_horses_block_path(tmp_path, monkeypatch):
    """5/10+ 全馬 table 通知 (Stage 2 失敗 = netkeiba_block path)."""
    race_id = "202605100501"
    _mock_full_csv(tmp_path, race_id, n_horses=12)
    monkeypatch.setattr(s2p, "DAILY_PRED_FULL_DIR", tmp_path)
    monkeypatch.setitem(s2p.RACE_START_TIMES, race_id, "15:30")

    full_rows = s2p.load_full_predictions(race_id, date="20260510")
    morning = _mock_morning_row(race_id=race_id, n_horses=12, top1_num="1")

    stage2_block = {
        "error": "netkeiba HTTP 400 (server block) / len=0",
        "error_kind": "netkeiba_block",
        "diag": {"status_code": 400, "response_len": 0},
    }
    title, body, color = s2p.build_message_all_horses(race_id, morning, stage2_block, full_rows)
    assert "Stage 2 状況 (失敗: netkeiba_block)" in body
    assert "Stage 1 (朝予測) 採用" in body
    assert color == "yellow"
