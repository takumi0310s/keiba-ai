#!/usr/bin/env python3
"""JRDB二重マージ再発防止テスト (2026-06-11 Fable監査③修正)。
背景: predict_core.build_features 内の merge#1 の後に再マージすると pandas merge 衝突で
実値が jrdb_*_x に退避し素列がデフォルト再充填(idm=50/脚質=0) → 4/2-6/10 の予測スコア劣化。
"""
import os, sys, glob, unittest
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'tools'))


class TestNoDoubleJrdbMerge(unittest.TestCase):

    def test_guard_skips_when_jrdb_present(self):
        """jrdb_列が既にある df は再マージしない(値不変・衝突列なし)。"""
        import daily_predict as dp
        df = pd.DataFrame({'horse_num': [1, 2], 'jrdb_idm': [31.1, 23.0]})
        out = dp.merge_jrdb_once(df.copy(), '202605030201')
        self.assertEqual(list(out.columns), list(df.columns))
        self.assertFalse(any(str(c).endswith('_x') or str(c).endswith('_y') for c in out.columns))
        self.assertEqual(out['jrdb_idm'].tolist(), [31.1, 23.0])

    def test_guard_merges_when_missing(self):
        """jrdb_列が無い df にはマージを呼ぶ(ガードが過剰スキップしない)。"""
        import daily_predict as dp
        called = {}
        orig = dp.merge_jrdb_predict_features
        def stub(df, rid):
            called['rid'] = rid
            return df
        dp.merge_jrdb_predict_features = stub
        try:
            df = pd.DataFrame({'horse_num': [1, 2]})
            dp.merge_jrdb_once(df, '202605030201')
        finally:
            dp.merge_jrdb_predict_features = orig
        self.assertEqual(called.get('rid'), '202605030201')

    def test_daily_predict_source_uses_guard(self):
        """per-raceループの直接再マージが復活していない(ガード経由のみ)。"""
        src = open(os.path.join(BASE, 'tools', 'daily_predict.py'), encoding='utf-8').read()
        self.assertIn('merge_jrdb_once(df, race_id)', src)
        self.assertNotIn('df = merge_jrdb_predict_features(df, race_id)', src)

    def test_dump_kyi_default_rate_after_fix(self):
        """修正後(6/13+)のダンプ再発検知: _x衝突列が無い + KYI族デフォルト率<90%。"""
        dirs = sorted(glob.glob(os.path.join(BASE, 'data', 'v15_feat_dump', '*')))
        recent = [d for d in dirs if os.path.basename(d) >= '20260613']
        if not recent:
            self.skipTest('修正後(6/13以降)のダンプ未生成')
        defaults = {'jrdb_idm': 50.0, 'jrdb_running_style': 0, 'jrdb_dist_apt': 0}
        tot = hit = 0
        for pq in glob.glob(os.path.join(recent[-1], '*.parquet')):
            try:
                df = pd.read_parquet(pq)
            except Exception:
                continue
            self.assertFalse(any(str(c).endswith('_x') for c in df.columns),
                             f'二重マージ衝突列(_x)が再発: {pq}')
            for c, dv in defaults.items():
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors='coerce').fillna(dv)
                    tot += len(v); hit += int((v == dv).sum())
        if tot:
            rate = hit / tot
            self.assertLess(rate, 0.9, f'KYI族デフォルト率 {rate:.1%} = 劣化再発の疑い')


if __name__ == '__main__':
    unittest.main()
