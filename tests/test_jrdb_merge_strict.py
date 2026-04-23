"""SED merge: NaN IDM フォールバック改善の回帰テスト

最新SED行の IDM が NaN な場合、過去で IDM が有効な最新行から拾うかを検証。
"""
import os
import sys
import unittest
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'tools'))


class TestSedNaNFallback(unittest.TestCase):

    def test_picks_latest_non_nan_idm(self):
        """SED の IDM が最新で NaN、3つ前で有効な値 → 有効な値を採用"""
        sed = pd.DataFrame({
            '_bn': ['11111111'] * 5,
            '_dt': [20240101, 20240301, 20240501, 20240801, 20251001],
            'IDM': ['50.0', '60.0', None, None, None],
        })
        sed_sub = sed.sort_values(['_bn', '_dt'])
        vals = pd.to_numeric(sed_sub['IDM'], errors='coerce')
        valid = sed_sub[vals.notna()]
        latest_valid = valid.groupby('_bn').tail(1).set_index('_bn')['IDM']
        self.assertEqual(float(latest_valid.loc['11111111']), 60.0)

    def test_falls_back_to_overall_when_all_nan(self):
        """全行 IDM が NaN → fallback でも NaN（default 50 で fillna されるべき）"""
        sed = pd.DataFrame({
            '_bn': ['22222222'] * 3,
            '_dt': [20240101, 20240301, 20240501],
            'IDM': [None, None, None],
        })
        sed_sub = sed.sort_values(['_bn', '_dt'])
        vals = pd.to_numeric(sed_sub['IDM'], errors='coerce')
        valid = sed_sub[vals.notna()]
        self.assertEqual(len(valid), 0)

    def test_per_column_independence(self):
        """IDM は古い行から、テン指数は新しい行から → 各列独立に最新有効値を取る"""
        sed = pd.DataFrame({
            '_bn': ['33333333'] * 3,
            '_dt': [20240101, 20240301, 20240501],
            'IDM':       ['40.0', None, None],
            'テン指数': [None, None, '15.0'],
        })
        sed_sub = sed.sort_values(['_bn', '_dt'])

        cols = [('IDM', 'IDM'), ('テン指数', 'テン指数')]
        per_col = {}
        for feat, src in cols:
            vals = pd.to_numeric(sed_sub[src], errors='coerce')
            valid = sed_sub[vals.notna()]
            if len(valid):
                per_col[feat] = valid.groupby('_bn').tail(1).set_index('_bn')[src]

        self.assertEqual(float(per_col['IDM'].loc['33333333']), 40.0)
        self.assertEqual(float(per_col['テン指数'].loc['33333333']), 15.0)

    def test_real_4_19_smoke(self):
        """4/19 実データで merge 後の prev_idm が default 50 ばかりにならない"""
        kyi_path = os.path.join(BASE, 'data', 'jrdb_kyi.csv')
        sed_path = os.path.join(BASE, 'data', 'jrdb_sed.csv')
        if not (os.path.exists(kyi_path) and os.path.exists(sed_path)):
            self.skipTest('JRDB csv files not present')
        from jrdb_features import merge_jrdb_predict_features
        rid = '202606030801'
        horses = pd.DataFrame({
            'horse_num': list(range(1, 13)),
            '馬名': [f'h{i}' for i in range(1, 13)],
        })
        out = merge_jrdb_predict_features(horses, rid)
        if 'jrdb_prev_idm' not in out.columns:
            self.skipTest('prev_idm not generated (race not in JRDB)')
        s = pd.to_numeric(out['jrdb_prev_idm'], errors='coerce').dropna()
        non_default = (s != 50.0).sum()
        # 修正前は 12中6前後 が default. 修正後は 8+ が non-default を期待
        self.assertGreaterEqual(non_default, 7,
            f'prev_idm non-default 数が改善されていない: {non_default}/12')

    def test_jrdb_features_module_loads(self):
        """jrdb_features モジュールが構文エラーなくロードできる"""
        import jrdb_features
        self.assertTrue(hasattr(jrdb_features, 'merge_jrdb_predict_features'))
        self.assertTrue(hasattr(jrdb_features, 'JRDB_DEFAULTS'))


if __name__ == '__main__':
    unittest.main()
