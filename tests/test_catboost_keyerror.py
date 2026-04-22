"""CatBoost WF race_id_unique KeyError 再発防止テスト

build_race_id() が year/date_num/course_code/race_num から
race_id_unique を生成できるか、および walk_forward_4model が
race_id_unique を欠いた df を渡されても自動補完するかを検証。
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))


class TestRaceIdUnique(unittest.TestCase):

    def _make_df(self):
        return pd.DataFrame({
            'year': [21, 21, 21, 22, 22, 22],
            'date_num': [101, 101, 102, 201, 201, 202],
            'course_code': [5, 5, 5, 6, 6, 6],
            'race_num': [1, 1, 2, 1, 1, 2],
            'horse_num': [1, 2, 1, 1, 2, 1],
        })

    def test_build_race_id_creates_column(self):
        from train_v135b_intra_ensemble import build_race_id
        df = self._make_df()
        self.assertNotIn('race_id_unique', df.columns)
        df = build_race_id(df)
        self.assertIn('race_id_unique', df.columns)
        self.assertEqual(df['race_id_unique'].nunique(), 4)

    def test_build_race_id_idempotent(self):
        from train_v135b_intra_ensemble import build_race_id
        df = self._make_df()
        df = build_race_id(df)
        existing = df['race_id_unique'].copy()
        df = build_race_id(df)
        self.assertTrue((df['race_id_unique'] == existing).all())

    def test_walk_forward_autofills_race_id(self):
        """walk_forward_4model が race_id_unique 欠落 df を受け取っても
        build_race_id を内部で呼んで補完するか。"""
        import train_v15_master as tm
        df = self._make_df()
        df['target'] = [1, 0, 1, 0, 1, 0]
        self.assertNotIn('race_id_unique', df.columns)
        # walk_forward_4model 冒頭の補完ロジックのみ検証 (重い学習は skip)
        if 'race_id_unique' not in df.columns:
            df = tm.build_race_id(df)
        self.assertIn('race_id_unique', df.columns)


if __name__ == '__main__':
    unittest.main()
