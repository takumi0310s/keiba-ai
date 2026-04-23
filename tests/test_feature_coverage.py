"""特徴量カバレッジツールの回帰テスト"""
import os
import sys
import unittest
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'tools'))


class TestFeatureCoverage(unittest.TestCase):

    def test_categorize_feature_known_columns(self):
        from feature_coverage_check import categorize_feature
        self.assertEqual(categorize_feature('jrdb_prev_idm'), 'jrdb_sed_prev')
        self.assertEqual(categorize_feature('jrdb_kta_idm'), 'jrdb_kta')
        self.assertEqual(categorize_feature('jrdb_oikiri_idx'), 'jrdb_cha')
        self.assertEqual(categorize_feature('jrdb_cid_idx'), 'jrdb_jo')
        self.assertEqual(categorize_feature('jrdb_paddock_idx'), 'jrdb_tyb')
        self.assertEqual(categorize_feature('jrdb_idm'), 'jrdb_kyi_basic')

    def test_jrdb_defaults_complete(self):
        from jrdb_features import JRDB_DEFAULTS
        # 全 prev_* が default 定義に含まれること
        required = ['jrdb_prev_idm', 'jrdb_prev_track_bias', 'jrdb_prev_interference',
                    'jrdb_prev_late_start', 'jrdb_prev_ten_idx', 'jrdb_prev_agari_idx',
                    'jrdb_prev_pace_idx', 'jrdb_prev_rise_code']
        for k in required:
            self.assertIn(k, JRDB_DEFAULTS, f'{k} が JRDB_DEFAULTS にない')

    def test_coverage_snapshot_exists_and_valid(self):
        snap = os.path.join(BASE, 'report', 'feature_coverage_20260419_after.json')
        if not os.path.exists(snap):
            self.skipTest('snapshot not yet generated')
        df = pd.read_json(snap)
        # 必須カラム
        for c in ['feature', 'category', 'non_default_rate', 'status']:
            self.assertIn(c, df.columns)
        # 改善判定: prev_idm 非default率 80%+
        idm = df[df['feature'] == 'jrdb_prev_idm']
        if len(idm) == 1:
            self.assertGreaterEqual(float(idm['non_default_rate'].iloc[0]), 0.80)

    def test_jrdb_coverage_detailed_module_loads(self):
        import jrdb_coverage_detailed
        self.assertTrue(hasattr(jrdb_coverage_detailed, 'JRDB_DEFS'))
        self.assertTrue(hasattr(jrdb_coverage_detailed, 'measure_one'))

    def test_coverage_categories_exhaustive(self):
        """全 jrdb_* 特徴量に対して categorize_feature が known カテゴリを返す"""
        from feature_coverage_check import categorize_feature
        from jrdb_features import JRDB_DEFAULTS
        known_cats = {
            'jrdb_sed_prev', 'jrdb_kta', 'jrdb_ze', 'jrdb_cha', 'jrdb_jo',
            'jrdb_kab_sr', 'jrdb_tyb', 'jrdb_blood', 'jrdb_skb', 'jrdb_kyi_basic',
        }
        for feat in JRDB_DEFAULTS.keys():
            cat = categorize_feature(feat)
            self.assertIn(cat, known_cats, f'{feat} のカテゴリ {cat} が未知')


if __name__ == '__main__':
    unittest.main()
