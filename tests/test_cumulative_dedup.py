#!/usr/bin/env python3
"""cumulative_results.csv 二重計上の再発防止テスト (2026-06-11 Fable sweep)。
背景: pandas が date を float 読み ('20260607.0') して文字列キー比較がすり抜け、
同日2回実行 (土日18:00+20:00) で全レース二重計上 (6/7 23R / 5/24 34R 実測)。
"""
import os, sys, tempfile, unittest
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'tools'))


class TestCumulativeDedup(unittest.TestCase):

    def test_norm_key(self):
        from daily_results import _norm_key
        self.assertEqual(_norm_key('20260607.0'), '20260607')
        self.assertEqual(_norm_key(20260607.0), '20260607')
        self.assertEqual(_norm_key('20260607'), '20260607')
        self.assertEqual(_norm_key(202605030201), '202605030201')

    def test_upsert_no_duplicate_on_float_date(self):
        """float化した date を含む既存CSVに同日を再upsertしても二重計上しない。"""
        import daily_results as dr
        results = [{'race_id': '202605030201', 'course': '東京', 'race_num': 1,
                    'status': 'settled', 'profit': -700, 'investment': 700,
                    'actual_payout': 0, 'trio_hit': 0, 'umaren_hit': 0,
                    'trio_payout': 0, 'umaren_payout': 0}]
        with tempfile.TemporaryDirectory() as td:
            tmp_csv = os.path.join(td, 'cumul.csv')
            orig = dr.CUMUL_CSV
            dr.CUMUL_CSV = tmp_csv
            try:
                # 1回目: 正常書込み → pandas が float 読みする状況を再現 (date を float で保存)
                df1 = dr._upsert_cumulative(results, '20260607')
                df_f = pd.read_csv(tmp_csv, encoding='utf-8-sig')
                df_f['date'] = df_f['date'].astype(float)  # float化を強制再現
                df_f.to_csv(tmp_csv, index=False, encoding='utf-8-sig')
                # 2回目: 同日再実行
                df2 = dr._upsert_cumulative(results, '20260607')
                self.assertEqual(len(df2), 1, f'二重計上が再発: {len(df2)}行')
                self.assertEqual(df2['date'].astype(str).tolist(), ['20260607'])
            finally:
                dr.CUMUL_CSV = orig

    def test_live_ledger_clean(self):
        """実台帳: 重複キーゼロ + .0 date ゼロ + 行内収支整合。"""
        p = os.path.join(BASE, 'data', 'cumulative_results.csv')
        if not os.path.exists(p):
            self.skipTest('台帳なし')
        df = pd.read_csv(p, encoding='utf-8-sig', low_memory=False)
        self.assertEqual(int(df['date'].astype(str).str.endswith('.0').sum()), 0, 'float date 残存')
        self.assertEqual(int(df.duplicated(subset=['date', 'race_id']).sum()), 0, '重複キー再発')
        s = df[df['status'] == 'settled']
        prof = pd.to_numeric(s['profit'], errors='coerce')
        ap = pd.to_numeric(s['actual_payout'], errors='coerce')
        inv = pd.to_numeric(s['investment'], errors='coerce')
        self.assertEqual(int((prof != ap - inv).sum()), 0, '行内収支不整合')


if __name__ == '__main__':
    unittest.main()
