"""Synthetic engine test only: invented horses/probabilities/results, never a racing forecast."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paddock_lab import Store,evaluate,acquisition_plan

def main():
    out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('demo-output')
    out.mkdir(parents=True,exist_ok=True)
    db=out/'synthetic.sqlite'
    if db.exists():
        raise SystemExit('Use a fresh output directory; demo never overwrites a ledger.')
    now=['2026-10-01T09:00:00+09:00']
    s=Store(str(db),lambda:now[0])
    rid='202608040811';url='https://example.org/SYNTHETIC'
    r={'race_id':rid,'scheduled_start':'2026-10-01T15:00:00+09:00','entry_source_url':url,
       'runners':[{'horse_id':str(i),'name':'架空馬'+str(i),'number':i} for i in range(1,5)]}
    s.register(r)
    now[0]='2026-10-01T14:00:00+09:00'
    s.forecast(rid,'before',{str(i):.75 for i in range(1,5)},{'version':'SYNTHETIC-base','target':'finish_top3','baseline_input_sha256':'b'*64})
    now[0]='2026-10-01T14:30:00+09:00'
    for h in r['runners']:
        for kind in ['current','past']:
            s.observation(rid,{'horse_id':h['horse_id'],'horse_name':h['name'],'kind':kind,
              'video_race_id':rid if kind=='current' else 'SYNTHETIC-past','video_horse_number':h['number'],
              'video_race_start':r['scheduled_start'] if kind=='current' else '2026-09-01T15:00:00+09:00',
              'observed_at':'2026-10-01T14:25:00+09:00','source_url':url,'identity_verified':True,
              'playback_confirmed':True,'mode':'continuous','continuous_seconds':10,'segments':[[0,10]],
              'view':'side','visibility':'clear','equipment':'fixture','weather':'fixture',
              'observer_id':'SYNTHETIC','facts':['SYNTHETIC observation'],'interpretations':[],
              'features':{'outline':2},'feature_evidence':{'outline':'human_ordinal'}})
    now[0]='2026-10-01T14:50:00+09:00'
    s.forecast(rid,'after',{'1':.9,'2':.8,'3':.8,'4':.5},
         {'version':'SYNTHETIC-pad','target':'finish_top3','baseline_version':'SYNTHETIC-base',
          'artifact_sha256':'a'*64,'baseline_input_sha256':'b'*64,'feature_names':['outline'],'trained_through':'2026-08-01T00:00:00+09:00',
          'frozen_at':'2026-08-31T00:00:00+09:00'})
    now[0]='2026-10-01T15:10:00+09:00'
    s.result(rid,{'1':1,'2':2,'3':3,'4':4},'2026-10-01T15:09:00+09:00',url)
    for name,data in [('audit',s.audit(rid)),('evaluation',evaluate(s)),
                      ('plan',acquisition_plan(r,[],'2026-10-01T14:00:00+09:00'))]:
        data['synthetic_fixture_only']=True
        (out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'fixture_only':True,'eligible_races':evaluate(s)['eligible_races'],
                      'production_approved':False,'head_hash':s.verify()},indent=2))
    s.close()

if __name__=='__main__':main()
