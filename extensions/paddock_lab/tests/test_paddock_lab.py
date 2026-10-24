import copy
import math
import tempfile
import unittest
from pathlib import Path
from paddock_lab import (Store, compare, evaluate, parse_race_url, select_past,
                         chronological_split, validate_observation, scores, observer_agreement, acquisition_plan)
RID='202608040811'
URL='https://race.sp.netkeiba.com/?pid=paddock&race_id='+RID

def race():
    return {'race_id':RID,'scheduled_start':'2026-10-24T15:30:00+09:00','entry_source_url':URL,
            'runners':[{'horse_id':str(i),'number':i,'name':'馬'+str(i)} for i in range(1,5)]}

def observation(h='1',kind='current'):
    return {'horse_id':h,'horse_name':'馬'+h,'kind':kind,'identity_verified':True,
            'video_race_id':RID if kind=='current' else '202609040311',
            'video_horse_number':int(h) if kind=='current' else 8,
            'video_race_start':'2026-10-24T15:30:00+09:00' if kind=='current' else '2026-09-12T15:30:00+09:00',
            'observed_at':'2026-10-24T15:10:00+09:00','source_url':URL,
            'playback_confirmed':True,'mode':'continuous','view':'side','visibility':'clear',
            'continuous_seconds':10,'segments':[[10,20]],'equipment':'same_recorded_equipment',
            'weather':'cloudy','observer_id':'analyst-1','facts':['side walk observed'],'interpretations':[],
            'features':{'outline':2,'walk_rhythm':2},
            'feature_evidence':{'outline':'human_ordinal','walk_rhythm':'human_ordinal'}}

class TestComparison(unittest.TestCase):
    def test_url(self):
        self.assertEqual(parse_race_url(URL),RID)
        with self.assertRaises(ValueError):parse_race_url('https://example.com/?race_id='+RID)
    def test_identity(self):
        p=observation(kind='past');p['horse_id']='2'
        with self.assertRaises(ValueError):compare(observation(),p)
    def test_wrong_current_number(self):
        o=observation();o['video_horse_number']=9
        with self.assertRaises(ValueError):validate_observation(o,race())
    def test_stills_do_not_become_gait(self):
        o=observation();o.update(mode='stills',continuous_seconds=0)
        c=compare(o,observation(kind='past'))
        self.assertEqual(c['features']['walk_rhythm']['status'],'hold')
        self.assertEqual(c['features']['outline']['status'],'comparable')
    def test_missing_not_no_change(self):
        c=compare(observation(),observation(kind='past'))
        self.assertIsNone(c['features']['sweat']['ordinal_delta'])
    def test_angle_equipment_and_age(self):
        for key,val in [('view','front'),('equipment',None),('video_race_start','2025-09-13T15:30:00+09:00')]:
            p=observation(kind='past');p[key]=val
            self.assertEqual(compare(observation(),p)['features']['outline']['status'],'hold')
    def test_selection_ignores_result_and_future(self):
        old=observation(kind='past');old['finish_position']=1
        new=copy.deepcopy(old);new.update(video_race_id='new',video_race_start='2026-10-10T15:30:00+09:00',finish_position=15)
        future=copy.deepcopy(new);future.update(video_race_id='future',video_race_start='2026-10-25T15:30:00+09:00')
        self.assertEqual(select_past([old,new,future],'2026-10-24T15:20:00+09:00',1)[0]['video_race_id'],'new')
    def test_bad_duration(self):
        o=observation();o['continuous_seconds']=100
        with self.assertRaises(ValueError):validate_observation(o,race())
    def test_score_provenance(self):
        o=observation();o['feature_evidence']={}
        with self.assertRaises(ValueError):validate_observation(o,race())
    def test_symmetry_never_measured_from_stills(self):
        c=compare(observation(),observation(kind='past'))
        self.assertIn('validated_multiview_measurement_not_implemented',c['features']['left_right_symmetry']['reasons'])

class TestStore(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.now='2026-10-24T14:00:00+09:00'
        self.store=Store(str(Path(self.temp.name)/'test.sqlite'),lambda:self.now)
        self.store.register(race())
    def tearDown(self):
        self.store.close();self.temp.cleanup()
    def freeze(self):
        self.now='2026-10-24T15:00:00+09:00'
        self.store.forecast(RID,'before',{str(i):.75 for i in range(1,5)}, {'version':'base1','target':'finish_top3','baseline_input_sha256':'b'*64})
    def after(self,**extra):
        self.now='2026-10-24T15:20:00+09:00'
        m={'version':'pad1','target':'finish_top3','baseline_version':'base1','artifact_sha256':'a'*64,'baseline_input_sha256':'b'*64,'feature_names':['outline'],
           'trained_through':'2026-09-01T00:00:00+09:00','frozen_at':'2026-10-01T00:00:00+09:00'}
        m.update(extra)
        self.store.forecast(RID,'after',{'1':.9,'2':.8,'3':.8,'4':.5},m)
    def coverage(self):
        self.now='2026-10-24T15:11:00+09:00'
        for i in range(1,5):
            for k in ['current','past']:self.store.observation(RID,observation(str(i),k))
    def test_no_overwrite_and_no_partial_field(self):
        self.freeze()
        with self.assertRaises(ValueError):self.freeze()
        with self.assertRaises(ValueError):self.store.forecast(RID,'after',{'1':1},{'version':'x','target':'finish_top3','baseline_input_sha256':'b'*64})
    def test_nonfinite_and_wrong_target(self):
        for p in [float('nan'),float('inf'),-1,True]:
            with self.assertRaises(ValueError):self.store.forecast(RID,'before',{'1':p,'2':.75,'3':.75,'4':.75},{'version':'b','target':'finish_top3','baseline_input_sha256':'b'*64})
    def test_incomplete_excluded(self):
        self.freeze();self.after()
        self.assertIn('full_field_comparison_incomplete',self.store.audit(RID)['exclusions'])
    def test_prospective_pair_and_metrics(self):
        self.freeze();self.coverage();self.after()
        self.assertTrue(self.store.audit(RID)['eligible_paired_evaluation'])
        self.now='2026-10-24T15:40:00+09:00'
        self.store.result(RID,{'1':1,'2':2,'3':3,'4':4},'2026-10-24T15:39:00+09:00',URL)
        e=evaluate(self.store)
        self.assertEqual(e['eligible_races'],1)
        self.assertLess(e['metrics_race_equal']['after']['brier'],e['metrics_race_equal']['before']['brier'])
        self.assertFalse(e['production_approved']);self.assertIsNone(e['actual_roi'])
        self.assertIsNone(e['date_equal_delta_95ci']['brier'])
    def test_baseline_leak_recorded_later(self):
        self.freeze();self.coverage();self.after()
        self.store.exposure(RID,'expert_comment','2026-10-24T14:59:00+09:00',URL)
        self.assertIn('baseline_exposed_before_freeze',self.store.audit(RID)['exclusions'])
    def test_late_freeze(self):
        self.now='2026-10-24T16:00:00+09:00'
        self.store.forecast(RID,'before',{str(i):.75 for i in range(1,5)}, {'version':'b','target':'finish_top3','baseline_input_sha256':'b'*64})
        self.assertIn('before_saved_after_start',self.store.audit(RID)['exclusions'])
    def test_future_model_excluded(self):
        self.freeze();self.coverage();self.after(trained_through='2026-10-24T15:01:00+09:00')
        self.assertIn('paddock_training_cutoff_missing_or_leaked',self.store.audit(RID)['exclusions'])
    def test_model_cannot_be_frozen_before_training_end(self):
        self.freeze();self.coverage();self.after(frozen_at='2026-08-31T00:00:00+09:00')
        self.assertIn('model_frozen_before_training_finished',self.store.audit(RID)['exclusions'])
    def test_tamper_detection(self):
        self.store.db.execute("UPDATE events SET payload='{}'");self.store.db.commit()
        with self.assertRaises(ValueError):self.store.verify()
    def test_result_must_be_confirmed(self):
        self.now='2026-10-24T16:00:00+09:00'
        with self.assertRaises(ValueError):self.store.result(RID,{'1':1,'2':2,'3':3,'4':4},self.now,URL,'flash')
    def test_input_changes_excluded(self):
        self.freeze();self.coverage();self.after(baseline_input_sha256='c'*64)
        self.assertIn('baseline_inputs_changed_or_hash_missing',self.store.audit(RID)['exclusions'])
    def test_feature_contract_excluded(self):
        self.freeze();self.coverage();self.after(feature_names=['sweat'])
        self.assertIn('required_model_features_unavailable',self.store.audit(RID)['exclusions'])
    def test_cancellation_invalidates(self):
        self.freeze();self.coverage();self.after()
        self.store.invalidate(RID,'scratch confirmed',URL)
        self.assertIn('field_or_experiment_invalidated',self.store.audit(RID)['exclusions'])
    def test_clock_backwards(self):
        self.now='2026-10-23T00:00:00+09:00'
        with self.assertRaises(ValueError):self.store.exposure(RID,'current_video',self.now,URL)

class TestStats(unittest.TestCase):
    def test_observer_agreement(self):
        self.assertEqual(observer_agreement([(0,0),(2,2),(4,4)])['weighted_kappa'],1)
        self.assertIsNone(observer_agreement([(2,2)])['weighted_kappa'])
    def test_full_field_plan(self):
        p=acquisition_plan(race(),[],'2026-10-24T15:00:00+09:00')
        self.assertEqual(len(p['tasks']),4)
        self.assertTrue(all(t['past_missing'] for t in p['tasks']))
        self.assertFalse(p['unattended_collection_running'])
    def test_known_scores(self):
        self.assertAlmostEqual(scores([.5,.5],[1,0])['brier'],.25)
        self.assertAlmostEqual(scores([.5,.5],[1,0])['log_loss'],math.log(2))
    def test_chronological_split(self):
        rows=[{'race_id':str(i),'scheduled_start':f'2026-10-{i:02}T15:00:00+09:00',
               'features_available_at':f'2026-10-{i:02}T14:00:00+09:00',
               'result_confirmed_at':f'2026-10-{i:02}T16:00:00+09:00'} for i in [1,2,3]]
        s=chronological_split(rows,'2026-10-01T23:00:00+09:00','2026-10-02T23:00:00+09:00')
        self.assertEqual([len(s[k]) for k in ['train','validation','test']],[1,1,1])
        rows[0]['features_available_at']='2026-10-01T17:00:00+09:00'
        with self.assertRaises(ValueError):chronological_split(rows,'2026-10-01T23:00:00+09:00','2026-10-02T23:00:00+09:00')

if __name__=='__main__':unittest.main()
