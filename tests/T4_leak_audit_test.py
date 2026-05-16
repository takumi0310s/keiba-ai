#!/usr/bin/env python
"""T4: leak audit automation tests

学習前 gate logic 検証。 V15 / v15.2 候補に対し、 leak detect / suspect flag / verdict が
仕様 (CLAUDE.md Session #38 + Sub-task 5-1/6/11/18) どおりに機能するかを検査。

★ V15 production 完全不変 ★ — read-only test、 .pkl.gz / predict_core / app.py 変更なし。

Usage:
    python tests/T4_leak_audit_test.py
"""
from __future__ import annotations

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from leak_audit_automation import (
    JRDB_DATATYPE_RELEASE,
    EXTERNAL_CORR_PRIOR,
    HIGH_CORR_THRESHOLD,
    identify_datatype,
    audit_feature,
    audit_features,
    summarize,
    _load_v15_cache,
    _read_features_file,
)


# -------------------------------------------------------------------
# 1. release_timing mapping coverage
# -------------------------------------------------------------------
class TestReleaseMapping(unittest.TestCase):
    """JRDB 29 datatype の release timing が正しくマップされているか。"""

    def test_skb_is_leak(self):
        """SKB は POST-RACE LEAK (V15.1 SKB、 Session #38 確定)。"""
        info = JRDB_DATATYPE_RELEASE["skb"]
        self.assertEqual(info["safety"], "leak")
        self.assertEqual(info["timing"], "post_race")

    def test_tyb_is_leak(self):
        """TYB は POST-RACE LEAK (Sub-task 11、 17:00 JST publish)。"""
        info = JRDB_DATATYPE_RELEASE["tyb"]
        self.assertEqual(info["safety"], "leak")
        self.assertIn("17:00", info["timing"])

    def test_payout_datatypes_are_leak(self):
        """払戻系 (hjc / ot / ou / ov / ow / oz) 全部 leak。"""
        for dt in ("hjc", "ot", "ou", "ov", "ow", "oz"):
            with self.subTest(datatype=dt):
                self.assertEqual(JRDB_DATATYPE_RELEASE[dt]["safety"], "leak",
                                 f"{dt} should be leak")

    def test_safe_datatypes(self):
        """safe datatype 一覧。"""
        safe_dts = {"kyi", "bac", "cha", "kab", "paci", "kta", "jo", "joa"}
        for dt in safe_dts:
            with self.subTest(datatype=dt):
                self.assertEqual(JRDB_DATATYPE_RELEASE[dt]["safety"], "safe",
                                 f"{dt} should be safe")

    def test_prev_only_datatypes(self):
        """prev_only datatype (sed, sr, srb, ze, zk, kka, kka_v2 等)。"""
        for dt in ("sed", "sr", "srb", "ze", "zk", "kka", "kka_v2"):
            with self.subTest(datatype=dt):
                self.assertEqual(JRDB_DATATYPE_RELEASE[dt]["safety"], "prev_only",
                                 f"{dt} should be prev_only")

    def test_live_only_datatypes(self):
        """CYB は live_only。"""
        self.assertEqual(JRDB_DATATYPE_RELEASE["cyb"]["safety"], "live_only")


# -------------------------------------------------------------------
# 2. identify_datatype の判定 logic
# -------------------------------------------------------------------
class TestIdentifyDatatype(unittest.TestCase):
    def test_jrdb_skb_prefix(self):
        """jrdb_skb_* は datatype=skb と判定 (★ leak ★)。"""
        dt, _ = identify_datatype("jrdb_skb_kishi_code_3")
        self.assertEqual(dt, "skb")

    def test_jrdb_tyb_prefix(self):
        """jrdb_tyb_* は datatype=tyb と判定 (★ leak ★)。"""
        dt, _ = identify_datatype("jrdb_tyb_padock_idx")
        self.assertEqual(dt, "tyb")

    def test_jrdb_idm_inherits_kyi(self):
        """jrdb_idm は kyi 直系。"""
        dt, _ = identify_datatype("jrdb_idm")
        self.assertEqual(dt, "kyi")

    def test_jrdb_kta_prefix(self):
        """jrdb_kta_* は datatype=kta (safe morning_06)。"""
        dt, _ = identify_datatype("jrdb_kta_idm")
        self.assertEqual(dt, "kta")

    def test_jrdb_ze_prefix(self):
        """jrdb_ze_* は datatype=ze (prev_only、 過去 5 走集計)。"""
        dt, _ = identify_datatype("jrdb_ze_idm_avg")
        self.assertEqual(dt, "ze")

    def test_jrdb_prev_is_sed(self):
        """jrdb_prev_* は前走 ref = sed。"""
        dt, _ = identify_datatype("jrdb_prev_idm")
        self.assertEqual(dt, "sed")

    def test_oz_fe_is_kyi(self):
        """oz_tansho_base_log は JRDB oz (払戻系 LEAK) ではなく kyi 基準オッズ FE。"""
        dt, _ = identify_datatype("oz_tansho_base_log")
        self.assertEqual(dt, "kyi", "oz_* fe は kyi morning_06 base odds で safe")

    def test_paci_prefix(self):
        dt, _ = identify_datatype("paci_jockey_exp_wr")
        self.assertEqual(dt, "paci")

    def test_jrdb_cha_inherited(self):
        """jrdb_oikiri_idx / jrdb_ten_time_idx / jrdb_shimai_time_idx は cha 由来。"""
        for n in ("jrdb_oikiri_idx", "jrdb_ten_time_idx", "jrdb_shimai_time_idx"):
            with self.subTest(name=n):
                dt, _ = identify_datatype(n)
                self.assertEqual(dt, "cha")

    def test_jrdb_jo_inherited(self):
        """jrdb_cid_idx / jrdb_ls_idx は jo 由来。"""
        for n in ("jrdb_cid_idx", "jrdb_ls_idx"):
            with self.subTest(name=n):
                dt, _ = identify_datatype(n)
                self.assertEqual(dt, "jo")

    def test_v152_candidate_prefixes(self):
        """v15.2 候補 prefix (cha_, kta_, kab_, breeder_, srb_)。"""
        self.assertEqual(identify_datatype("cha_chukan_time_idx")[0], "cha")
        self.assertEqual(identify_datatype("kta_ichi_idx_pred")[0], "kta")
        self.assertEqual(identify_datatype("kab_renzoku_day")[0], "kab")
        self.assertEqual(identify_datatype("breeder_dist_1")[0], "kka_v2")
        self.assertEqual(identify_datatype("srb_bias_straight")[0], "srb")


# -------------------------------------------------------------------
# 3. audit_feature verdict
# -------------------------------------------------------------------
class TestAuditVerdict(unittest.TestCase):
    def test_jrdb_skb_blocked(self):
        """SKB datatype の features は全て LEAK 判定。"""
        for f in ("jrdb_skb_kishi_code_1", "jrdb_skb_kishi_code_3",
                  "jrdb_skb_baba_code_1", "jrdb_skb_kyaku_code_3",
                  "jrdb_skb_turf_hoof"):
            with self.subTest(feature=f):
                r = audit_feature(f)
                self.assertEqual(r.verdict, "LEAK", f"{f} must be LEAK")

    def test_jrdb_tyb_blocked(self):
        """TYB datatype の features は全て LEAK 判定 (post_race_17:00)。"""
        for f in ("jrdb_tyb_padock_idx", "jrdb_tyb_tansho_odds",
                  "jrdb_tyb_fukusho_odds", "jrdb_tyb_jockey_idx",
                  "jrdb_tyb_info_idx", "jrdb_tyb_odds_idx", "jrdb_tyb_sogo_idx",
                  "jrdb_tyb_cancel_flag", "jrdb_tyb_horse_weight"):
            with self.subTest(feature=f):
                r = audit_feature(f)
                self.assertEqual(r.verdict, "LEAK", f"{f} must be LEAK")
                self.assertEqual(r.timing, "post_race_17:00")

    def test_payout_blocked(self):
        for f in ("jrdb_hjc_first", "jrdb_ot_payout", "jrdb_ou_dividend"):
            with self.subTest(feature=f):
                r = audit_feature(f)
                self.assertEqual(r.verdict, "LEAK")

    def test_safe_features_ok(self):
        """kyi / paci / cha / kta / kab / jo は OK 判定。"""
        for f in ("jrdb_idm", "jrdb_kta_idm", "paci_ninki_idx",
                  "jrdb_oikiri_idx", "jrdb_cid_idx", "weight_carry"):
            with self.subTest(feature=f):
                r = audit_feature(f)
                self.assertEqual(r.verdict, "OK", f"{f} must be OK")

    def test_prev_ref_ok(self):
        """prev_* / jrdb_prev_* / jrdb_ze_* / jrdb_tb_* は前走 ref で OK。"""
        for f in ("prev_finish", "prev_odds_log", "jrdb_prev_idm",
                  "jrdb_ze_idm_avg", "jrdb_tb_homestr_inner"):
            with self.subTest(feature=f):
                r = audit_feature(f)
                self.assertEqual(r.verdict, "OK")

    def test_breeder_kka_v2_ok(self):
        """breeder_* (kka_v2 静的 集計) は OK。"""
        for f in ("breeder_dist_1", "breeder_track_1",
                  "jrdb_dam_rensho_avg", "jrdb_bms_rensho_avg",
                  "jrdb_ranch_rank"):
            with self.subTest(feature=f):
                r = audit_feature(f)
                self.assertEqual(r.verdict, "OK")


# -------------------------------------------------------------------
# 4. corr_target HIGH_CORR_LEAK_SUSPECT detection
# -------------------------------------------------------------------
class TestHighCorrSuspect(unittest.TestCase):
    def test_paci_info_idx_flagged(self):
        """paci_info_idx は corr +0.41 (Sub-task 18 §3-2)、 SUSPECT flag。"""
        r = audit_feature("paci_info_idx")
        self.assertIsNotNone(r.corr_target)
        self.assertGreater(abs(r.corr_target), HIGH_CORR_THRESHOLD,
                           "paci_info_idx corr must exceed 0.40")
        self.assertEqual(r.corr_flag, "HIGH_CORR_LEAK_SUSPECT")
        self.assertEqual(r.verdict, "SUSPECT")

    def test_known_safe_high_corr_not_flagged(self):
        """paci_jockey_exp_wr 等 既知 safe は high-corr でも flag されない。"""
        # cache 由来 (実測 corr) を強制したいので V15 cache を load
        try:
            df, _ = _load_v15_cache()
        except FileNotFoundError:
            self.skipTest("V15 cache not available")
        for f in ("paci_jockey_exp_wr", "paci_ninki_idx", "jrdb_training_idx",
                  "oz_tansho_base_log"):
            with self.subTest(feature=f):
                r = audit_feature(f, cache_df=df)
                self.assertIsNotNone(r.corr_target, f"{f} corr missing")
                if abs(r.corr_target) > HIGH_CORR_THRESHOLD:
                    self.assertEqual(r.corr_flag, "ok_known_safe",
                                     f"{f} should be ok_known_safe")
                    self.assertEqual(r.verdict, "OK")

    def test_low_corr_features_ok(self):
        """corr が threshold 以下なら verdict は datatype-baseline まま。"""
        for f in ("kab_renzoku_day", "cha_chukan_time_idx"):
            with self.subTest(feature=f):
                r = audit_feature(f)
                if r.corr_target is not None:
                    self.assertLess(abs(r.corr_target), HIGH_CORR_THRESHOLD)
                self.assertEqual(r.verdict, "OK")


# -------------------------------------------------------------------
# 5. V15 (150 features) 全体 audit
# -------------------------------------------------------------------
class TestV15FullAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.df, cls.features = _load_v15_cache()
        except FileNotFoundError:
            cls.df = None
            cls.features = None

    def setUp(self):
        if self.df is None:
            self.skipTest("V15 cache not available")

    def test_v15_no_leak_features(self):
        """V15 145 features で LEAK 判定 0 件 (★ 期待 ★)。

        もし leak が出たら V15 自体に致命的問題、 緊急 alert。
        """
        results = audit_features(self.features, cache_df=self.df)
        s = summarize(results)
        self.assertEqual(s["by_verdict"].get("LEAK", 0), 0,
                         f"V15 にあるはずのない leak features 検出: {s['leaks']}")

    def test_v15_no_suspect_features(self):
        """V15 145 features で HIGH_CORR_LEAK_SUSPECT 0 件。"""
        results = audit_features(self.features, cache_df=self.df)
        s = summarize(results)
        self.assertEqual(len(s["suspects"]), 0,
                         f"V15 で suspect 検出: {s['suspects']}")

    def test_v15_high_corr_all_safe(self):
        """V15 で corr > 0.40 features は全て safe datatype。"""
        results = audit_features(self.features, cache_df=self.df)
        high_corr = [r for r in results
                     if r.corr_target is not None
                     and abs(r.corr_target) > HIGH_CORR_THRESHOLD]
        for r in high_corr:
            with self.subTest(feature=r.feature):
                self.assertIn(r.safety, ("safe", "fe_internal"),
                              f"{r.feature} corr={r.corr_target:.4f} but safety={r.safety}")


# -------------------------------------------------------------------
# 6. v15.2 candidates pre-audit
# -------------------------------------------------------------------
class TestV152Candidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates_path = os.path.join(BASE_DIR, "data", "v15_2",
                                           "features_v152_candidates.txt")
        if not os.path.exists(cls.candidates_path):
            cls.features = None
        else:
            cls.features = _read_features_file(cls.candidates_path)

    def setUp(self):
        if self.features is None:
            self.skipTest("v15.2 candidates file not available")

    def test_v152_candidates_no_leak(self):
        """v15.2 候補 22 features で LEAK 0 件 (TYB/SKB は永久除外)。"""
        results = audit_features(self.features)
        s = summarize(results)
        self.assertEqual(s["by_verdict"].get("LEAK", 0), 0,
                         f"v15.2 候補に LEAK 検出: {s['leaks']}")

    def test_v152_paci_info_idx_is_suspect(self):
        """paci_info_idx (corr +0.41) は SUSPECT 検出される。"""
        results = audit_features(self.features)
        s = summarize(results)
        self.assertIn("paci_info_idx", s["suspects"],
                      "paci_info_idx must be HIGH_CORR_LEAK_SUSPECT")

    def test_v152_gate_block_on_suspect(self):
        """gate logic: SUSPECT 検出時は exit 1 相当 (FAIL)。"""
        results = audit_features(self.features)
        s = summarize(results)
        # leak 0 件 + suspect 1 件 → gate FAIL
        self.assertEqual(s["by_verdict"].get("LEAK", 0), 0)
        self.assertGreater(len(s["suspects"]), 0)
        # 実際の gate は tools/v15_2_train_gate.py で exit 1 (詳細別 test)


# -------------------------------------------------------------------
# 7. hypothetical leak inject test (★ regression 強化 ★)
# -------------------------------------------------------------------
class TestLeakInjection(unittest.TestCase):
    """悪意の leak feature を inject しても確実に block されるか。"""

    def test_inject_skb_into_v15(self):
        """V15 features list に skb feature を inject → LEAK 検出。"""
        features = ["weight_carry", "age", "jrdb_skb_kishi_code_3", "jrdb_idm"]
        results = audit_features(features)
        s = summarize(results)
        self.assertEqual(s["by_verdict"].get("LEAK", 0), 1)
        self.assertIn("jrdb_skb_kishi_code_3", s["leaks"])

    def test_inject_tyb_padock(self):
        """jrdb_tyb_padock_idx (5/16 P0-3 で確定) は LEAK 検出。"""
        results = audit_features(["jrdb_tyb_padock_idx", "weight_carry"])
        s = summarize(results)
        self.assertEqual(s["by_verdict"].get("LEAK", 0), 1)
        self.assertIn("jrdb_tyb_padock_idx", s["leaks"])

    def test_inject_hjc_payout(self):
        """jrdb_hjc_* (払戻) は LEAK。"""
        results = audit_features(["jrdb_hjc_first_payout"])
        s = summarize(results)
        self.assertEqual(s["by_verdict"].get("LEAK", 0), 1)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
def run_all_tests() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestReleaseMapping, TestIdentifyDatatype, TestAuditVerdict,
                TestHighCorrSuspect, TestV15FullAudit, TestV152Candidates,
                TestLeakInjection]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
