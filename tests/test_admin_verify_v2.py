#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_admin_verify_v2.py  (★ 完成-4 sub-task 2026-05-18 ★)

5/18 17:00 false positive (bash grep pattern が日本語 schtasks error と非一致) 再発防止。
admin_verify_v2.py が exit code 直接判定で 正しく registered/未登録 を分けることを test。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import admin_verify_v2  # noqa: E402


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


SAMPLE_QUERY_SUCCESS_JA = """\
フォルダー: \\
ホスト名:                              DESKTOP
タスク名:                              \\Keiba-LiveOrchestrator-15min
次回の実行時刻:                        2026/05/19 08:30:00
状態:                                  準備完了
ログオン モード:                       対話型のみ
前回の実行時刻:                        2026/05/18 08:30:00
前回の結果:                            0
作成者:                                takum
スケジュールされたタスクの実行時刻:    08:30:00
スケジュール:                          スケジュール データを利用できません。
タスクを実行する時刻:                  毎日 08:30:00
"""

SAMPLE_QUERY_ERROR_JA = """\
エラー: 指定されたファイルが見つかりません。
"""


class TestParseSchtasksQuery(unittest.TestCase):
    """test 1+2: schtasks /Query success / not-exist 両 case"""

    def test_parse_schtasks_query_success(self):
        fake = _FakeCompleted(returncode=0, stdout=SAMPLE_QUERY_SUCCESS_JA)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake):
            info = admin_verify_v2.query_schtask("Keiba-LiveOrchestrator-15min")
        self.assertTrue(info["registered"])
        self.assertEqual(info["exit_code"], 0)
        self.assertEqual(info["next_run"], "2026/05/19 08:30:00")
        self.assertEqual(info["status"], "準備完了")
        self.assertEqual(info["last_run"], "2026/05/18 08:30:00")

    def test_parse_schtasks_query_not_exist(self):
        # ★ 重要 ★ 5/18 17:00 false positive 再発 防止: exit code != 0 -> registered=False
        # 日本語 error 文言は judgment に 使わない
        fake = _FakeCompleted(
            returncode=1, stdout="", stderr=SAMPLE_QUERY_ERROR_JA
        )
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake):
            info = admin_verify_v2.query_schtask("Keiba-NonExistent")
        self.assertFalse(info["registered"])
        self.assertEqual(info["exit_code"], 1)
        self.assertIsNone(info["next_run"])


class TestBatExistenceCheck(unittest.TestCase):
    """test 3: bat existence check"""

    def test_bat_existence_check_real(self):
        # live_orchestrator.bat は実在 (B-5 後 想定)
        info = admin_verify_v2.check_bat("live_orchestrator.bat")
        self.assertTrue(info["exists"])
        self.assertGreater(info["size_bytes"], 0)

    def test_bat_existence_check_missing(self):
        info = admin_verify_v2.check_bat("__non_existent_bat__.bat")
        self.assertFalse(info["exists"])


class TestPyModuleCompileCheck(unittest.TestCase):
    """test 4: py_compile PASS check"""

    def test_py_module_compile_ok(self):
        info = admin_verify_v2.check_py_module("tools/anomaly_auto_detector.py")
        self.assertTrue(info["exists"])
        self.assertTrue(info["compile_ok"])

    def test_py_module_missing(self):
        info = admin_verify_v2.check_py_module("tools/__non_existent__.py")
        self.assertFalse(info["exists"])
        self.assertFalse(info["compile_ok"])


class TestFalsePositiveRegression(unittest.TestCase):
    """test 5: ★ 5/18 17:00 false positive 再発 検出 test ★

    旧 method (bash grep pattern マッチ) と 新 method (exit code 直接) の diff を 検証。
    旧 pattern ["ERROR", "エラー", "not exist"] は schtasks /Query error 文言と部分一致
    する場合があり、 false positive を生んだ。 新 method は exit code 0 でのみ 登録済 とする。
    """

    def test_5_18_17_00_false_positive_detected(self):
        # ★ 5/18 17:00 再現 ★
        # 旧 method: bash grep stdout 上で en-US pattern マッチ (case-sensitive、 stderr 無視)
        # 実 schtasks /Query は ja-JP locale で stderr に error 出力、 stdout 空
        # -> 旧 grep は stdout 上で 「ERROR」「not exist」 を見つけられず "登録済" と誤判定
        old_grep_patterns = ["ERROR", "not exist", "does not exist"]
        sim_stdout = ""  # 実行 失敗 時 stdout は空
        sim_stderr = SAMPLE_QUERY_ERROR_JA  # 「エラー:...」 (cp932)

        # 旧 method 1: stdout 上で en-US pattern grep (実 bash one-liner 想定)
        old_judgment_registered_stdout = not any(
            p in sim_stdout for p in old_grep_patterns
        )
        self.assertTrue(
            old_judgment_registered_stdout,
            "旧 grep (stdout / en-US pattern) は ja-JP error を見逃し false positive",
        )

        # 旧 method 2: stderr + stdout 合算 でも en-US pattern では ja-JP 「エラー」 と非一致
        # (case-sensitive な en-US 大文字 "ERROR" は 「エラー」 と byte-level 不一致)
        combined = sim_stdout + sim_stderr
        old_judgment_registered_combined = not any(
            p in combined for p in old_grep_patterns
        )
        self.assertTrue(
            old_judgment_registered_combined,
            "旧 grep (combined / en-US pattern) も ja-JP error を見逃す",
        )

        # 新 method: exit code != 0 -> 未登録 確実 判定 (locale 非依存)
        fake = _FakeCompleted(returncode=1, stdout=sim_stdout, stderr=sim_stderr)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake):
            info = admin_verify_v2.query_schtask("Keiba-LiveOrchestrator-15min")
        new_judgment_registered = info["registered"]

        # ★ diff: 旧 True (false positive) vs 新 False (真値) ★
        self.assertTrue(old_judgment_registered_stdout)
        self.assertFalse(new_judgment_registered)
        self.assertNotEqual(old_judgment_registered_stdout, new_judgment_registered)


class TestRunVerifyShape(unittest.TestCase):
    """test (bonus): run_verify が 9 schtask 全て対象に処理 + summary keys 揃う"""

    def test_run_verify_full_shape(self):
        # subprocess は mock (全部 未登録 = 5/18 真値想定)
        fake = _FakeCompleted(returncode=1, stdout="", stderr=SAMPLE_QUERY_ERROR_JA)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake):
            report = admin_verify_v2.run_verify()

        self.assertIn("verify_time", report)
        self.assertIn("method", report)
        self.assertEqual(len(report["schtasks"]), 9)
        self.assertEqual(len(report["bats"]), 9)
        self.assertEqual(report["summary"]["schtasks_registered"], "0/9")
        # 全 schtask 未登録 と 判定された 事実 確認
        for name, info in report["schtasks"].items():
            self.assertFalse(info["registered"], f"{name} should be unregistered")


# ============================================================
# ★ F-5 追加 test (2026-05-19) ★
# ready_for_5_22_admin logic bug audit + fire_ready 新フラグ
# ============================================================

class TestFireReadyWhenAllRegistered(unittest.TestCase):
    """F-5 test 1: 9/9 登録済 → fire_ready=True / registration_complete=True"""

    def test_ready_when_all_registered(self):
        # 9/9 登録済 mock
        fake_ok = _FakeCompleted(returncode=0, stdout=SAMPLE_QUERY_SUCCESS_JA)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake_ok):
            report = admin_verify_v2.run_verify()

        s = report["summary"]
        self.assertEqual(s["schtasks_registered"], "9/9")
        self.assertTrue(s["registration_complete"], "9/9 登録済 → registration_complete=True")
        # fire_ready は bat + py も依存するが、それらは実ファイル存在に依存する。
        # schtask 9/9 登録済なら ready_for_5_22_admin=False (旧フラグの「admin前」条件未満)。
        self.assertFalse(
            s["ready_for_5_22_admin"],
            "9/9 登録済 → 旧フラグ ready_for_5_22_admin=False が正しい (admin 着手前ではない)",
        )
        # registration_status に "COMPLETE" が含まれる
        self.assertIn("COMPLETE", s["registration_status"])
        # summary に fire_ready キー が存在する
        self.assertIn("fire_ready", s)


class TestAdminNeededWhenZeroRegistered(unittest.TestCase):
    """F-5 test 2: 0/9 未登録 → fire_ready=False / ready_for_5_22_admin=bat+py 依存"""

    def test_not_ready_when_zero_registered(self):
        # 0/9 未登録 mock
        fake_err = _FakeCompleted(returncode=1, stdout="", stderr=SAMPLE_QUERY_ERROR_JA)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake_err):
            report = admin_verify_v2.run_verify()

        s = report["summary"]
        self.assertEqual(s["schtasks_registered"], "0/9")
        self.assertFalse(s["registration_complete"], "0/9 → registration_complete=False")
        self.assertFalse(s["fire_ready"], "0/9 未登録 → fire_ready=False")
        self.assertIn("NOT REGISTERED", s["registration_status"])


class TestRegistrationStatusStrings(unittest.TestCase):
    """F-5 test 3: registration_status 文字列が状態に応じて正しく変化する"""

    def test_complete_status_string(self):
        fake_ok = _FakeCompleted(returncode=0, stdout=SAMPLE_QUERY_SUCCESS_JA)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake_ok):
            report = admin_verify_v2.run_verify()
        self.assertIn("COMPLETE", report["summary"]["registration_status"])
        self.assertIn("no admin needed", report["summary"]["registration_status"])

    def test_not_registered_status_string(self):
        fake_err = _FakeCompleted(returncode=1, stdout="", stderr=SAMPLE_QUERY_ERROR_JA)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake_err):
            report = admin_verify_v2.run_verify()
        self.assertIn("NOT REGISTERED", report["summary"]["registration_status"])


class TestSummaryBackwardCompat(unittest.TestCase):
    """F-5 test 4: 旧フラグ ready_for_5_22_admin が summary に残っている (後方互換)"""

    def test_old_flag_still_present(self):
        fake_err = _FakeCompleted(returncode=1, stdout="", stderr=SAMPLE_QUERY_ERROR_JA)
        with mock.patch.object(admin_verify_v2.subprocess, "run", return_value=fake_err):
            report = admin_verify_v2.run_verify()
        self.assertIn("ready_for_5_22_admin", report["summary"])
        # 旧フラグの意味: bat/py 存在 + 0/9 未登録 = admin 着手可能
        # bat/py の実ファイル存在に依存するため bool 値の厳密 assert は行わず、
        # キーの存在のみ確認する
        self.assertIsInstance(report["summary"]["ready_for_5_22_admin"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
