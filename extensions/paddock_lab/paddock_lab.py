"""Paddock evidence, fixed forecasts and paired evaluation. Python stdlib only.
No scraping, credentials, video downloads, medical diagnosis or automatic betting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

VERSION = '0.1.0'
FEATURES = ('outline', 'neck_use', 'walk_rhythm', 'stride_consistency',
            'left_right_symmetry', 'sweat', 'settledness')
DYNAMIC = {'walk_rhythm', 'stride_consistency', 'left_right_symmetry'}
# Engineering defaults, NOT empirically validated performance thresholds.
POLICY = {'version': 'quality-0.1', 'minimum_continuous_seconds': 8,
          'maximum_past_days': 180, 'minimum_races_for_review': 100,
          'minimum_dates_for_review': 20}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def moment(s):
    d = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('timezone required')
    return d.astimezone(timezone.utc)


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def source(url):
    p = urlsplit(url)
    require(p.scheme == 'https' and bool(p.netloc) and not p.username and not p.password,
            'public HTTPS source URL required; never put credentials in URLs')


def parse_race_url(url):
    """Only decode explicit netkeiba IDs. RV/UUID identity must be resolved in UI."""
    p = urlsplit(url)
    require(p.hostname in {'race.sp.netkeiba.com', 'race.netkeiba.com'}, 'unsupported race URL')
    ids = parse_qs(p.query).get('race_id', [])
    require(len(ids) == 1 and re.fullmatch(r'\d{12}', ids[0]), 'explicit 12 digit race_id required')
    return ids[0]


def validate_race(r):
    require(re.fullmatch(r'\d{12}', r['race_id']), 'invalid race_id')
    moment(r['scheduled_start'])
    require(r['entry_source_url'], 'entry source required')
    source(r['entry_source_url'])
    runners = r['runners']
    require(len(runners) >= 3, 'at least three active runners required')
    ids = [h['horse_id'] for h in runners]
    nums = [h['number'] for h in runners]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), 'unique horse IDs required')
    require(all(isinstance(n, int) and n > 0 for n in nums) and len(nums) == len(set(nums)), 'unique positive runner numbers required')
    require(all(h['name'] for h in runners), 'horse names required')
    return r


def validate_observation(o, r):
    h = next((h for h in r['runners'] if h['horse_id'] == o['horse_id']), None)
    require(h is not None, 'horse not in race registry')
    require(o['kind'] in {'current', 'past'}, 'kind must be current/past')
    require(o['horse_name'] == h['name'], 'horse name/ID mismatch')
    require(isinstance(o['identity_verified'], bool), 'identity_verified must be boolean')
    source(o['source_url'])
    moment(o['observed_at'])
    moment(o['video_race_start'])
    require(o['video_race_id'] and o['video_horse_number'] > 0, 'video identity required')
    if o['kind'] == 'current':
        require(o['video_race_id'] == r['race_id'], 'wrong current race')
        require(o['video_horse_number'] == h['number'], 'wrong current horse number')
    else:
        require(moment(o['video_race_start']) < moment(r['scheduled_start']), 'future video rejected')
    require(isinstance(o['playback_confirmed'], bool), 'playback flag must be boolean')
    require(o['mode'] in {'stills', 'continuous'}, 'mode must be stills/continuous')
    require(o['view'] in {'side', 'front', 'rear', 'mixed', 'unknown'}, 'invalid view')
    require(o['visibility'] in {'clear', 'partial', 'poor'}, 'invalid visibility')
    require(number(o['continuous_seconds']) and o['continuous_seconds'] >= 0, 'invalid observed duration')
    require(o['mode'] != 'stills' or o['continuous_seconds'] == 0, 'stills cannot claim continuous gait observation')
    require(o['segments'] and all(len(s) == 2 and all(number(t) for t in s) and 0 <= s[0] <= s[1]
                                for s in o['segments']), 'video timestamp segments required')
    require(o['continuous_seconds'] <= max(s[1]-s[0] for s in o['segments']) + 0.01,
            'observed continuous duration exceeds timestamp segment')
    require(isinstance(o['facts'], list) and isinstance(o['interpretations'], list), 'facts and interpretations must be separate lists')
    require(not o.get('features') or o.get('observer_id'), 'scored observations need observer_id')
    for f in o.get('features', {}):
        require(f in FEATURES, 'unknown feature')
        val = o['features'][f]
        require(val is None or (number(val) and 0 <= val <= 4), 'feature must be ordinal 0..4 or null')
        require(val is None or o.get('feature_evidence', {}).get(f) == 'human_ordinal',
                'each score needs explicit human_ordinal provenance; no invented measured values')
    # Missing stays null. Ordinals are human observations, not medical or metric measurements.
    return o


def select_past(observations, as_of, count=3):
    """Latest available prior starts, independent of finish position. No cherry-picking winners."""
    cutoff = moment(as_of)
    require(isinstance(count, int) and count > 0, 'positive past count required')
    eligible = [o for o in observations if o['kind'] == 'past'
                and moment(o['observed_at']) <= cutoff and moment(o['video_race_start']) < cutoff]
    ordered = sorted(eligible, key=lambda o: (moment(o['video_race_start']), moment(o['observed_at'])), reverse=True)
    result, seen = [], set()
    for o in ordered:
        if o['video_race_id'] not in seen:
            result.append(o)
            seen.add(o['video_race_id'])
        if len(result) == count:
            break
    return result


def acquisition_plan(race, catalog, as_of):
    """Full-field task queue, not a claim that browser collection has run.
    Catalog entries are observed site links, never invented RV UUIDs or video paths.
    """
    validate_race(race)
    cutoff, start = moment(as_of), moment(race['scheduled_start'])
    tasks = []
    for horse in race['runners']:
        candidates = []
        for c in catalog:
            if c['horse_id'] != horse['horse_id']:
                continue
            source(c['source_url'])
            if moment(c['link_seen_at']) <= cutoff and moment(c['video_race_start']) < start:
                candidates.append(c)
        candidates.sort(key=lambda c:moment(c['video_race_start']), reverse=True)
        unique = {}
        for c in candidates:
            unique.setdefault(c['video_race_id'], c)
        chosen = list(unique.values())[:3]
        tasks.append({'horse_id':horse['horse_id'], 'number':horse['number'], 'name':horse['name'],
                      'current':'resolve_from_visible_official_calendar', 'past_links':chosen,
                      'past_missing':len(chosen) == 0,
                      'capture':'verify identity; time progression; matched side view; record actual watched segments',
                      'expert_text':'do_not_read_before_independent_observation'})
    return {'race_id':race['race_id'], 'as_of':as_of, 'late':cutoff >= start,
            'baseline_freeze_deadline':(start-timedelta(minutes=30)).isoformat(),
            'comparison_deadline':(start-timedelta(minutes=10)).isoformat(),
            'refresh_deadline':(start-timedelta(minutes=5)).isoformat(), 'tasks':tasks,
            'unattended_collection_running':False}


def compare(current, past, policy=None):
    """Return feature-level eligibility; missing/low quality NEVER becomes no change."""
    p = policy or POLICY
    reasons = []
    require(current['horse_id'] == past['horse_id'], 'comparison must use same horse ID')
    age = (moment(current['video_race_start']) - moment(past['video_race_start'])).total_seconds()/86400
    if not 0 < age <= p['maximum_past_days']:
        reasons.append('past_video_too_old_or_not_prior')
    for tag, o in [('current', current), ('past', past)]:
        if not o['identity_verified']:
            reasons.append(tag + '_identity_unverified')
        if not o['playback_confirmed']:
            reasons.append(tag + '_playback_unconfirmed')
        if o['visibility'] != 'clear':
            reasons.append(tag + '_visibility_insufficient')
    if current['view'] != past['view'] or current['view'] not in {'side', 'front', 'rear'}:
        reasons.append('view_mismatch_or_unknown')
    if current.get('equipment') is None or past.get('equipment') is None:
        reasons.append('equipment_unknown')
    elif current['equipment'] != past['equipment']:
        reasons.append('equipment_changed')
    rows = {}
    for f in FEATURES:
        blocked = list(reasons)
        if f == 'left_right_symmetry':
            blocked.append('validated_multiview_measurement_not_implemented')
        if f in DYNAMIC:
            for tag, o in [('current', current), ('past', past)]:
                if o['mode'] != 'continuous' or o['continuous_seconds'] < p['minimum_continuous_seconds']:
                    blocked.append(tag + '_continuous_gait_insufficient')
            if f in {'stride_consistency', 'left_right_symmetry'} and current['view'] != 'side':
                blocked.append('side_view_required')
        if f == 'sweat' and (not current.get('weather') or current.get('weather') != past.get('weather')):
            blocked.append('weather_not_matched')
        cv, pv = current.get('features', {}).get(f), past.get('features', {}).get(f)
        if cv is None or pv is None:
            blocked.append('feature_unscored')
        rows[f] = {'status': 'hold' if blocked else 'comparable', 'reasons': sorted(set(blocked)),
                   'current': cv, 'past': pv, 'ordinal_delta': None if blocked else cv-pv}
    return {'horse_id': current['horse_id'], 'past_days': round(age, 1), 'policy': p['version'],
            'features': rows, 'numeric_probability_adjustment': None,
            'note': 'Ordinal deltas are not calibrated probability changes. No global good/bad diagnosis.'}


class Store:
    """SQLite event ledger. Append-only via API; hashes detect edits, not full-file replacement.
    Protect external copies of the head hash for stronger auditability.
    """
    def __init__(self, path, clock=stamp):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA busy_timeout=10000')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL, race_id TEXT NOT NULL, payload TEXT NOT NULL, prev TEXT NOT NULL, hash TEXT NOT NULL)')
        self.db.commit()
        self.clock = clock

    def close(self):
        self.db.close()

    def events(self, race_id=None):
        self.verify()
        sql = 'SELECT seq,at,kind,race_id,payload,prev,hash FROM events'
        rows = self.db.execute(sql + (' WHERE race_id=?' if race_id else '') + ' ORDER BY seq',
                               (race_id,) if race_id else ()).fetchall()
        return [dict(seq=s, at=a, kind=k, race_id=r, payload=json.loads(p), prev=v, hash=h)
                for s,a,k,r,p,v,h in rows]

    def verify(self):
        prev = '0'*64
        for seq, at, kind, rid, payload, before, hashed in self.db.execute('SELECT * FROM events ORDER BY seq'):
            expected = digest({'at': at, 'kind': kind, 'race_id': rid, 'payload': json.loads(payload), 'prev': prev})
            require(before == prev and expected == hashed, f'ledger integrity failure at {seq}')
            prev = hashed
        return prev

    def _append(self, kind, rid, payload):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prev = self.verify()
            at = self.clock()
            moment(at)
            # Application clock may not move backward relative to ledger.
            last = self.db.execute('SELECT at FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            require(not last or moment(last[0]) <= moment(at), 'clock moved backward')
            require(not any(e['kind'] == kind and kind in {'race', 'before', 'after', 'result'} for e in self.events(rid)),
                    f'{kind} already fixed; register a new field version/race key for scratches before forecasts')
            d = {'at': at, 'kind': kind, 'race_id': rid, 'payload': payload, 'prev': prev}
            hashed = digest(d)
            self.db.execute('INSERT INTO events(at,kind,race_id,payload,prev,hash) VALUES(?,?,?,?,?,?)',
                            (at,kind,rid,canonical(payload),prev,hashed))
            self.db.commit()
            return hashed
        except Exception:
            self.db.rollback()
            raise

    def register(self, race):
        validate_race(race)
        return self._append('race', race['race_id'], race)

    def race(self, rid):
        rows = [e['payload'] for e in self.events(rid) if e['kind'] == 'race']
        require(len(rows) == 1, 'race not registered')
        return rows[0]

    def observation(self, rid, o):
        validate_observation(o, self.race(rid))
        require(moment(o['observed_at']) <= moment(self.clock()), 'future observation time')
        return self._append('observation', rid, o)

    def exposure(self, rid, kind, at, source_url):
        self.race(rid)
        require(kind in {'current_video', 'expert_comment', 'current_result'}, 'invalid exposure')
        source(source_url)
        require(moment(at) <= moment(self.clock()), 'future exposure')
        return self._append('exposure', rid, {'kind': kind, 'exposed_at': at, 'source_url': source_url})

    def invalidate(self, rid, reason, source_url):
        """Scratches/field changes do not silently renormalize a frozen experiment."""
        self.race(rid)
        require(isinstance(reason, str) and bool(reason.strip()), 'invalidation reason required')
        source(source_url)
        return self._append('invalidated', rid, {'reason':reason,'source_url':source_url})

    def forecast(self, rid, stage, probabilities, model):
        r = self.race(rid)
        require(stage in {'before', 'after'}, 'invalid forecast stage')
        expected = {h['horse_id'] for h in r['runners']}
        require(set(probabilities) == expected, 'full active field probabilities required')
        require(all(number(p) and 0 <= p <= 1 for p in probabilities.values()), 'invalid probabilities')
        require(abs(sum(probabilities.values()) - 3) <= 0.05, 'top3 marginal probabilities must sum to 3 (no dead-heat assumption)')
        require(model.get('version') and model.get('target') == 'finish_top3', 'model version and explicit top3 target required')
        if stage == 'after':
            require(any(e['kind'] == 'before' for e in self.events(rid)), 'before forecast must be fixed first')
        return self._append(stage, rid, {'probabilities': probabilities, 'model': model,
                                         'target': 'finish_top3', 'field_hash': digest(r['runners'])})

    def result(self, rid, finish, confirmed_at, source_url, status='confirmed'):
        r = self.race(rid)
        require(status == 'confirmed', '速報 cannot enter evaluation')
        require(set(finish) == {h['horse_id'] for h in r['runners']}, 'result must cover full active field')
        require(all(v is None or (isinstance(v,int) and not isinstance(v,bool) and v > 0) for v in finish.values()), 'invalid placing; DNF is null')
        require(moment(r['scheduled_start']) < moment(confirmed_at) <= moment(self.clock()), 'invalid result confirmation time')
        source(source_url)
        return self._append('result', rid, {'finish': finish, 'confirmed_at': confirmed_at, 'source_url': source_url})

    def audit(self, rid):
        r, es = self.race(rid), self.events(rid)
        start = moment(r['scheduled_start'])
        before = next((e for e in es if e['kind'] == 'before'), None)
        after = next((e for e in es if e['kind'] == 'after'), None)
        reasons = []
        if any(e['kind'] == 'invalidated' for e in es):
            reasons.append('field_or_experiment_invalidated')
        for name, e in [('before', before), ('after', after)]:
            if not e:
                reasons.append(name + '_missing')
            elif moment(e['at']) >= start:
                reasons.append(name + '_saved_after_start')
        exposures = [(moment(e['payload']['exposed_at']), e['payload']['kind']) for e in es if e['kind'] == 'exposure']
        obs = [e for e in es if e['kind'] == 'observation']
        exposures += [(moment(e['payload']['observed_at']), 'current_video') for e in obs if e['payload']['kind'] == 'current']
        if before and any(t <= moment(before['at']) and k in {'current_video', 'expert_comment', 'current_result'} for t,k in exposures):
            reasons.append('baseline_exposed_before_freeze')
        if after and any(t <= moment(after['at']) and k in {'expert_comment', 'current_result'} for t,k in exposures):
            reasons.append('after_expert_or_result_contaminated')
        # Full field coverage is a strict gate, not selectively evaluated favourites only.
        coverage = {}
        for h in r['runners']:
            available = [e['payload'] for e in obs if e['payload']['horse_id'] == h['horse_id']
                         and after and moment(e['at']) < moment(after['at'])
                         and moment(e['payload']['observed_at']) < moment(after['at'])]
            current = sorted([o for o in available if o['kind'] == 'current'], key=lambda o:moment(o['observed_at']))
            past = select_past(available, after['at']) if after else []
            comparisons = [compare(current[-1], p) for p in past] if current else []
            usable = any(any(v['status'] == 'comparable' for v in c['features'].values()) for c in comparisons)
            coverage[h['horse_id']] = {'current_seen': bool(current), 'past_seen': len(past),
                                      'usable_comparison': usable, 'comparisons': comparisons}
        if not all(v['usable_comparison'] for v in coverage.values()):
            reasons.append('full_field_comparison_incomplete')
        if before and after:
            if before['payload']['model'].get('version') != after['payload']['model'].get('baseline_version'):
                reasons.append('baseline_model_changed')
            model = after['payload']['model']
            baseline_input = before['payload']['model'].get('baseline_input_sha256')
            if (not isinstance(baseline_input,str) or not re.fullmatch('[0-9a-f]{64}',baseline_input)
                    or baseline_input != model.get('baseline_input_sha256')):
                reasons.append('baseline_inputs_changed_or_hash_missing')
            required_features = model.get('feature_names')
            if (not isinstance(required_features,list) or not required_features
                    or not all(f in FEATURES for f in required_features)):
                reasons.append('paddock_feature_contract_missing')
            elif not all(any(all(c['features'][f]['status'] == 'comparable' for f in required_features)
                             for c in v['comparisons']) for v in coverage.values()):
                reasons.append('required_model_features_unavailable')
            if not model.get('artifact_sha256') or not re.fullmatch('[0-9a-f]{64}', model['artifact_sha256']):
                reasons.append('paddock_model_artifact_missing')
            if not model.get('trained_through') or moment(model['trained_through']) >= moment(before['at']):
                reasons.append('paddock_training_cutoff_missing_or_leaked')
            if not model.get('frozen_at') or moment(model['frozen_at']) >= moment(before['at']):
                reasons.append('paddock_model_not_frozen_before_baseline')
            if model.get('trained_through') and model.get('frozen_at') and moment(model['trained_through']) > moment(model['frozen_at']):
                reasons.append('model_frozen_before_training_finished')
        return {'race_id': rid, 'eligible_paired_evaluation': not reasons, 'exclusions': reasons,
                'coverage': coverage, 'head_hash': self.verify(),
                'trust_limit': 'self-declared exposures/identity/model provenance; local hash chain is not external timestamp attestation'}


def mean(xs):
    return sum(xs)/len(xs) if xs else None


def scores(p, y):
    q = [min(1-1e-15,max(1e-15,v)) for v in p]
    return {'brier': mean([(v-t)**2 for v,t in zip(p,y)]),
            'log_loss': mean([-t*math.log(v)-(1-t)*math.log1p(-v) for v,t in zip(q,y)])}


def bootstrap_clusters(values, iterations=2000, seed=31415):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(mean([rng.choice(values) for _ in values]) for _ in range(iterations))
    return [samples[int(0.025*iterations)], samples[min(iterations-1,int(0.975*iterations))]]


def evaluate(store):
    """Race-equal metrics; date-cluster paired CI. Never infers exotic bet joint odds."""
    rids = [e['race_id'] for e in store.events() if e['kind'] == 'race']
    paired, excluded = [], {}
    before_all, after_all, labels_all = [], [], []
    for rid in rids:
        a = store.audit(rid)
        es = store.events(rid)
        result = next((e for e in es if e['kind'] == 'result'), None)
        reasons = a['exclusions'] + ([] if result else ['confirmed_result_missing'])
        if reasons:
            excluded[rid] = reasons
            continue
        b = next(e for e in es if e['kind'] == 'before')['payload']['probabilities']
        p = next(e for e in es if e['kind'] == 'after')['payload']['probabilities']
        keys = sorted(b)
        y = [int(result['payload']['finish'][h] is not None and result['payload']['finish'][h] <= 3) for h in keys]
        bs, ps = scores([b[k] for k in keys],y), scores([p[k] for k in keys],y)
        paired.append({'race_id': rid, 'date': moment(store.race(rid)['scheduled_start']).astimezone(
            timezone(timedelta(hours=9))).date().isoformat(), 'before': bs, 'after': ps,
            'delta_brier': ps['brier']-bs['brier'], 'delta_log_loss': ps['log_loss']-bs['log_loss']})
        before_all += [b[k] for k in keys]
        after_all += [p[k] for k in keys]
        labels_all += y
    dates = defaultdict(list)
    for x in paired:
        dates[x['date']].append(x)
    cis = {m:bootstrap_clusters([mean([r['delta_'+m] for r in rs]) for rs in dates.values()])
           for m in ['brier', 'log_loss']}
    def calibration(probs):
        bins = []
        for i in range(10):
            ix = [j for j,p in enumerate(probs) if min(int(p*10),9) == i]
            bins.append({'lower':i/10,'upper':(i+1)/10,'n':len(ix),
                         'mean_p':mean([probs[j] for j in ix]), 'observed_rate':mean([labels_all[j] for j in ix])})
        return bins
    n, nd = len(paired), len(dates)
    enough = n >= POLICY['minimum_races_for_review'] and nd >= POLICY['minimum_dates_for_review']
    signal = enough and all(v is not None and v[1] < 0 for v in cis.values())
    return {'version': VERSION, 'eligible_races': n, 'eligible_dates': nd, 'excluded_races': excluded,
            'metrics_race_equal': {s:{m:mean([r[s][m] for r in paired]) for m in ['brier','log_loss']}
                                   for s in ['before','after']},
            'date_equal_delta_95ci':cis, 'negative_delta_favors_paddock':True,
            'calibration_runner_weighted':{'before':calibration(before_all),'after':calibration(after_all)},
            'races':paired, 'review_signal':bool(signal), 'production_approved':False,
            'actual_roi':None, 'notes':['Engineering sample thresholds are provisional, not proof of accuracy.',
             'No automatic promotion: untouched-period validation and probability calibration still required.',
             'CI clusters by race date; repeated horses across dates may add dependence.',
             'Marginal top3 probabilities do not determine wide/trio joint probabilities.',
             'No real purchase log; actual ROI stays null.']}


def chronological_split(rows, train_end, validation_end):
    """Full races stay together; boundary-day leakage rejected by timestamp cutoff."""
    a,b = moment(train_end), moment(validation_end)
    require(a < b, 'invalid split boundaries')
    require(len({r['race_id'] for r in rows}) == len(rows), 'one row per complete race required')
    out = {'train':[], 'validation':[], 'test':[]}
    for r in sorted(rows,key=lambda r:moment(r['scheduled_start'])):
        start = moment(r['scheduled_start'])
        require(moment(r['features_available_at']) < start, 'poststart features rejected')
        result_at = moment(r['result_confirmed_at'])
        require(result_at > start, 'invalid label timestamp')
        if start <= a:
            require(result_at <= a, 'train label not known at cutoff')
            out['train'].append(r)
        elif start <= b:
            require(result_at <= b, 'validation label not known at cutoff')
            out['validation'].append(r)
        else:
            out['test'].append(r)
    return out


def observer_agreement(pairs):
    """Quadratic weighted kappa for independent blinded paired ordinal ratings.
    Callers pair same horse/video/feature and exclude non-blinded duplicates.
    """
    require(all(len(p) == 2 and all(isinstance(v,int) and not isinstance(v,bool) and 0<=v<=4 for v in p)
                for p in pairs), 'paired integer ordinal scores 0..4 required')
    n = len(pairs)
    if not n:
        return {'n':0,'exact_agreement':None,'weighted_kappa':None}
    left,right = Counter(a for a,b in pairs),Counter(b for a,b in pairs)
    observed = sum((a-b)**2/16 for a,b in pairs)/n
    expected = sum(left[a]*right[b]*(a-b)**2/16 for a in range(5) for b in range(5))/(n*n)
    return {'n':n, 'exact_agreement':sum(a==b for a,b in pairs)/n,
            'weighted_kappa':1-observed/expected if expected else None,
            'note':'One-valued ratings make kappa undefined. Does not prove predictive validity.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', default='paddock.sqlite')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['register','observe','forecast','exposure','result','plan','invalidate']:
        q = sub.add_parser(name)
        q.add_argument('file', help='UTF-8 JSON input')
    q = sub.add_parser('audit'); q.add_argument('race_id')
    sub.add_parser('evaluate'); sub.add_parser('verify')
    args = p.parse_args()
    s = Store(args.db)
    try:
        if args.command in {'audit','evaluate','verify'}:
            out = s.audit(args.race_id) if args.command == 'audit' else evaluate(s) if args.command == 'evaluate' else {'head_hash':s.verify()}
        else:
            d = json.loads(Path(args.file).read_text(encoding='utf-8'))
            if args.command == 'register': out = s.register(d)
            elif args.command == 'observe': out = s.observation(d['race_id'], d['observation'])
            elif args.command == 'forecast': out = s.forecast(d['race_id'],d['stage'],d['probabilities'],d['model'])
            elif args.command == 'exposure': out = s.exposure(d['race_id'],d['kind'],d['at'],d['source_url'])
            elif args.command == 'plan': out = acquisition_plan(d['race'],d.get('catalog',[]),d.get('as_of',stamp()))
            elif args.command == 'invalidate': out = s.invalidate(d['race_id'],d['reason'],d['source_url'])
            else: out = s.result(d['race_id'],d['finish'],d['confirmed_at'],d['source_url'],d.get('status','confirmed'))
        print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    finally:
        s.close()


if __name__ == '__main__':
    main()
