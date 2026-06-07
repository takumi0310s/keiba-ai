#!/usr/bin/env python3
"""前向き paper trading: s2b(穴特化候補)の予測を発走前に記録 → 結果照合 → ROI継続測定。

★完全分離・本番非影響★:
  - 本番 V15/V16 .pkl.gz / predict_core / daily_predict / app.py / race_auto_notify は一切改変しない。
  - 投票・買い目通知・Discord(買い目)には一切出さない。記録のみ。
  - 発走前データ(本番liveが使えるデータ)のみで予測 = リーク構造的に不可能。
  - 出力は新ログ data/paper_s2b/ (.gitignore対象)。

モード:
  predict  --date YYYYMMDD : 当日全レースを s2b で予測 → top6+人気を記録 (発走前に実行)
  results  --date YYYYMMDD : JRA払戻と照合 → 券種別ROIを計算・追記
  report                    : 累積ROI/的中率/N を券種別に表示 (leak-freeバックテストと乖離確認)
  from-oof --date YYYYMMDD : (検証用) leak-free OOF から擬似pred-logを生成しresults配線をテスト

s2b予測は predict_core.build_features(本番と同じ発走前特徴) → s2b特徴(人気代理族除去+レース相対) → s2b候補。
ze4特徴は live では過去ZEDのみ=元々 leak-free。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, argparse, time
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
LOG_DIR = os.path.join(DATA, 'paper_s2b')
S2B_PATH = os.path.join(BASE, 'models', 'v16_anaba_s2b_candidate.pkl.gz')

ODDS_REMOVE = ['paci_ninki_idx','odds_change_rate','odds_sharp_drop','oz_base_pop_rank','oz_fukusho_base_log','oz_tansho_base_log','pop_rank_change','prev_odds_log']
PROXY_FAMILY = ['paci_jockey_exp_wr','paci_jockey_exp_3rd','paci_jockey_mark','paci_sogo_mark','paci_train_mark','paci_idm_mark','jrdb_cid_idx','jrdb_ls_idx','jrdb_training_idx','jrdb_stable_idx']
EXTRA = ['paci_goal_rank','paci_goal_diff','paci_dochu_rank']
RAW_REPLACE = ['jrdb_running_style','jrdb_dist_apt']

_S2B = None
def _load_s2b():
    global _S2B
    if _S2B is None:
        with gzip.open(S2B_PATH,'rb') as f: _S2B = pickle.load(f)
    return _S2B


def build_s2b_features(race_df: pd.DataFrame) -> pd.DataFrame:
    """1レース分の V15特徴df(全馬) から s2b 特徴(one-hot脚質/距離適性 + レース相対 + 交互)を構築。
    v16_anaba_s2_eval.build_features を流用(race_id_unique を定数化=1レース内で集計)。"""
    from v16_anaba_s2_eval import build_features as _bf
    df = race_df.copy()
    if 'race_id_unique' not in df.columns: df['race_id_unique'] = 'PAPER_RACE'
    # 必須列の存在保証(無ければ0)
    for c in ['jrdb_running_style','jrdb_dist_apt','distance','num_horses_val','horse_num','jrdb_tb_homestr_inner']:
        if c not in df.columns: df[c] = 0
    return _bf(df)


def score_s2b(race_df: pd.DataFrame) -> dict:
    """race_df(V15特徴・全馬) → {horse_num: s2b_score}。"""
    m = _load_s2b(); feats = m['features']
    df = build_s2b_features(race_df)
    for f in feats:
        if f not in df.columns: df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    import xgboost as xgb
    X = df[feats].values
    p = 0.5*m['model'].predict(X) + 0.5*m['xgb_model'].predict(xgb.DMatrix(X))
    hn = pd.to_numeric(df['horse_num'], errors='coerce').fillna(0).astype(int).values
    return {int(h): float(s) for h, s in zip(hn, p)}


# ============ 払戻 index (date, course, race_num) ============
def _to_i(x):
    try: return int(float(x))
    except Exception: return None

def _payout_index_daily_full(date):
    """本番 daily_results が保存した全払戻JSON(data/daily_results_full/{date}.json)から payout index を構築。
    ★go-forward の正規ソース★: 本番が既に netkeiba から取得済みの結果を読むだけ=新規アクセス無し。
    複勝/三連単は本番未取得 → None(=その券種はスキップ)。 単勝/馬連/三連複は実払戻で照合可。"""
    p = os.path.join(DATA, 'daily_results_full', f'{date}.json')
    if not os.path.exists(p): return {}
    try:
        recs = json.load(open(p, encoding='utf-8'))
    except Exception:
        return {}
    idx = {}
    for r in recs:
        key = f"{date}_{r.get('course','')}_{int(r.get('race_num',0))}"
        fo = {int(k): int(v) for k, v in (r.get('finish_order') or {}).items()}
        winner = next((h for h, pos in fo.items() if pos == 1), None)
        pay = r.get('payouts') or {}
        tn = [int(x) for x in (r.get('trio_nums') or [])]
        un = [int(x) for x in (r.get('umaren_nums') or [])]
        # 複勝: {馬番: 払戻}(本番 daily_results が複勝parse追加後に格納)。 無ければ None でスキップ。
        fk = pay.get('fukusho')
        fuk = {int(a): int(b) for a, b in fk.items()} if isinstance(fk, dict) and fk else None
        # 三連単: 着順(tierce_nums優先 / 無ければ finish_order から1-2-3着) + 払戻。 払戻0/着順不明なら None。
        ten = [int(x) for x in (r.get('tierce_nums') or [])]
        if len(ten) < 3:
            ten = [h for h, _ in sorted(fo.items(), key=lambda kv: kv[1])][:3] if len(fo) >= 3 else []
        tierce_pay = int(pay.get('tierce', 0) or 0)
        idx[key] = {
            'tan': (winner, int(pay.get('tansho', 0) or 0)),
            'fuk': fuk,
            'umaren': (frozenset(un), int(pay.get('umaren', 0) or 0)) if len(un) == 2 else None,
            'trio': (frozenset(tn), int(pay.get('trio', 0) or 0)) if len(tn) == 3 else None,
            'tierce': (tuple(ten), tierce_pay) if (len(ten) == 3 and tierce_pay > 0) else None,
        }
    return idx


def load_payout_index():
    """★fallback(historical)★: jra_payouts.csv(現在5/17で停止)から全券種の payout index。"""
    p = pd.read_csv(os.path.join(DATA,'jra_payouts.csv'), low_memory=False)
    idx = {}
    for _, r in p.iterrows():
        key = f"{int(r['race_date'])}_{r['course']}_{int(r['race_num'])}"
        fuk = {}
        try:
            for n,pay in zip(str(r['fukusho_nums']).replace(' ','').split('/'), str(r['fukusho_payouts']).replace(' ','').split('/')):
                if _to_i(n) is not None: fuk[_to_i(n)] = _to_i(pay) or 0
        except Exception: pass
        def pset(nums,pay):
            try:
                a=str(nums).replace(' ','').split('-')
                if all(_to_i(x) is not None for x in a): return (frozenset(_to_i(x) for x in a), _to_i(pay) or 0)
            except Exception: pass
            return None
        def pord(nums,pay):
            try:
                a=str(nums).replace(' ','').split('-')
                if len(a)==3 and all(_to_i(x) is not None for x in a): return (tuple(_to_i(x) for x in a), _to_i(pay) or 0)
            except Exception: pass
            return None
        idx[key] = {'tan':(_to_i(r['tansho_nums']),_to_i(r['tansho_payout']) or 0),'fuk':fuk,
                    'umaren':pset(r['umaren_nums'],r['umaren_payout']),'trio':pset(r['trio_nums'],r['trio_payout']),
                    'tierce':pord(r['tierce_nums'],r['tierce_payout'])}
    return idx


# ============ 券種(top6順位 o から (return, points)) ============
import itertools
def bt_tan(o,pm): return (pm['tan'][1] if pm['tan'][0]==o[0] else 0, 1)
def bt_fuku1(o,pm):
    if pm.get('fuk') is None: return (0,0)   # 払戻源に複勝なし → スキップ(pts=0)
    return (pm['fuk'].get(o[0],0), 1)
def bt_umaren_t3box(o,pm):
    r=sum(pm['umaren'][1] for a,b in itertools.combinations(o[:3],2) if pm['umaren'] and pm['umaren'][0]==frozenset((a,b))); return (r,3)
def bt_trio4(o,pm):
    if not pm['trio']: return (0,0)
    won=pm['trio'][0]; combos=list(itertools.combinations(o[:4],3))
    return (pm['trio'][1] if any(frozenset(c)==won for c in combos) else 0, len(combos))
def bt_tierce125(o,pm):
    if not pm['tierce']: return (0,0)
    bets=set()
    for b in o[1:3]:
        for c in o[1:6]:
            if len({o[0],b,c})==3: bets.add((o[0],b,c))
    return (pm['tierce'][1] if pm['tierce'][0] in bets else 0, len(bets))
BETS = [('単勝',bt_tan),('複勝top1',bt_fuku1),('馬連top3box',bt_umaren_t3box),('三連複top4box',bt_trio4),('三連単form1-2-5',bt_tierce125)]


def make_bets(top6: list) -> dict:
    """s2b top6(馬番) から buy targets を生成(leak-free ROIで良かった順)。"""
    o = [int(x) for x in top6]
    import itertools
    trio4 = [tuple(sorted(c)) for c in itertools.combinations(o[:4], 3)] if len(o) >= 4 else []
    # 三連単 1-2-5: 1着=top1 / 2着∈top2-3 / 3着∈top2-6
    t125 = []
    for b in o[1:3]:
        for c in o[1:6]:
            if len({o[0], b, c}) == 3:
                t125.append((o[0], b, c))
    return {'tansho': o[0] if o else None, 'trio4box': trio4, 'tierce125': t125}


def _name_map(df: pd.DataFrame) -> dict:
    """ダンプdfから {馬番: 馬名}。馬名列が無ければ空(=馬番のみ表示)。netkeiba非アクセス。"""
    try:
        col = '馬名' if '馬名' in df.columns else next((c for c in df.columns if c in ('horse_name', 'name')), None)
        if col is None: return {}
        hn = pd.to_numeric(df['horse_num'], errors='coerce')
        return {int(n): str(v) for n, v in zip(hn, df[col]) if pd.notna(n)}
    except Exception:
        return {}


def _n_features() -> int:
    """s2bモデルの実特徴数を動的取得(ハードコード回避・モデル差し替え時も嘘にならない)。失敗時0。"""
    try:
        m = _load_s2b()
        return int(m.get('n_features') or len(m['features']))
    except Exception:
        return 0


_GAIN_FEATS = None
def _top_gain_feats(n: int = 6) -> list:
    """s2bモデルの gain上位特徴(LGB)を取得。失敗しても通知は続行。"""
    global _GAIN_FEATS
    if _GAIN_FEATS is None:
        try:
            m = _load_s2b(); feats = m['features']
            g = m['model'].feature_importance(importance_type='gain')
            _GAIN_FEATS = [feats[i] for i in np.argsort(g)[::-1]]
        except Exception:
            _GAIN_FEATS = []
    return _GAIN_FEATS[:n]


def _confidence(scores: dict, o: list):
    """確信度: top1スコア・2位との差・★段階を返す。"""
    if not scores or not o: return 0.0, 0.0, '★'
    s1 = float(scores.get(o[0], 0.0))
    s2 = float(scores.get(o[1], 0.0)) if len(o) > 1 else 0.0
    gap = s1 - s2
    stars = '★★★' if gap >= 0.10 else ('★★' if gap >= 0.05 else '★')
    return s1, gap, stars


def format_test_embed(rec: dict, resend: bool = False) -> dict:
    """TEST1 用 Discord Embed(色帯+構造化)。検証用と明記・実投票とは無関係。"""
    o = [int(x) for x in rec.get('s2b_top6', [])]
    b = make_bets(o)
    scores = {int(k): float(v) for k, v in (rec.get('s2b_scores') or {}).items()}
    names = {int(k): v for k, v in (rec.get('s2b_names') or {}).items()}
    course = rec.get('course', ''); rn = rec.get('race_num', '')
    rname = rec.get('race_name', ''); st = rec.get('start_time', '')
    s1, gap, stars = _confidence(scores, o)
    # 色: 検証用(紫基調) を確信度で変える。★★★=緑 / ★★=青 / ★=灰紫
    color = {'★★★': 0x2ECC71, '★★': 0x3498DB, '★': 0x9B59B6}[stars]

    medals = ['①', '②', '③', '④', '⑤', '⑥']
    rows = []
    for i, h in enumerate(o[:6]):
        nm = names.get(h, '')
        sc = scores.get(h, None)
        sc_s = f"{sc:.3f}" if sc is not None else '-'
        rows.append(f"{medals[i]} `{h:>2}` {nm[:10]:　<10} `{sc_s}`")
    top_block = "\n".join(rows) if rows else "(データなし)"

    trio = " / ".join('-'.join(map(str, c)) for c in b['trio4box']) or '-'
    t125 = (f"1着 **{o[0]}** → 2着 {','.join(map(str, o[1:3]))} → 3着 {','.join(map(str, o[1:6]))} "
            f"({len(b['tierce125'])}点)") if o else '-'
    gfeats = _top_gain_feats(6)
    gain_s = " / ".join(gfeats) if gfeats else "(取得不可)"

    title = f"🧪 s2bテスト {course}{rn}R {rname}".strip()
    if resend: title = "🔁 " + title

    embed = {
        "title": title,
        "description": (f"発走 **{st}**　確信度 {stars}（top1 `{s1:.3f}` / 2位差 `+{gap:.3f}`）\n"
                        f"⚠️ 穴特化候補の**観察用**。実際の投票(#買い目)とは**無関係**・実投票ではない。"),
        "color": color,
        "fields": [
            {"name": "🏇 s2b 上位6頭（馬番 / 馬名 / score）", "value": top_block, "inline": False},
            {"name": "◆ 単勝", "value": f"**{b['tansho']}**" if b['tansho'] else '-', "inline": True},
            {"name": "◆ 三連複 top4 BOX (4点)", "value": trio, "inline": True},
            {"name": "◆ 三連単 form 1-2-5", "value": t125, "inline": False},
            {"name": "📈 効いた特徴（gain上位）", "value": gain_s, "inline": False},
        ],
        "footer": {"text": (f"特徴量{_n_features()}・人気代理族除去の穴特化候補(leak-free v2) ｜ 検証用・実投票ではない")},
    }
    return embed


def notify_test1_date(date, resend: bool = False, limit: int = 0):
    """当日の paper pred-log を読み、s2b買い目を TEST1 へ Embed送信(検証用)。BETS/UPDATESには絶対出さない。
    limit>0 のときは先頭limit件のみ送信(デザイン確認用)。"""
    from dotenv import dotenv_values
    import requests
    url = dotenv_values(os.path.join(BASE, '.env')).get('DISCORD_WEBHOOK_TEST1')
    if not url:
        print("[test1] DISCORD_WEBHOOK_TEST1 未設定"); return
    pp = _pred_path(date)
    if not os.path.exists(pp):
        print(f"[test1] {pp} なし(先に predict を実行)"); return
    n = 0
    for line in open(pp, encoding='utf-8'):
        rec = json.loads(line)
        if len(rec.get('s2b_top6', [])) < 5: continue
        if limit and n >= limit: break
        embed = format_test_embed(rec, resend=resend)
        try:
            resp = requests.post(url, json={'embeds': [embed]}, timeout=15)
            if resp.status_code in (200, 204): n += 1
            else: print(f"[test1] {rec.get('rk')} HTTP {resp.status_code} {resp.text[:120]}")
            time.sleep(0.5)  # Discord rate-limit回避
        except Exception as e:
            print(f"[test1] 送信例外 {e}")
    print(f"[test1] {date}: {n}R を TEST1 に Embed送信(検証用・BETS非混入)")


def _pred_path(date): return os.path.join(LOG_DIR, f'{date}_pred.jsonl')
def _res_path(date):  return os.path.join(LOG_DIR, f'{date}_results.jsonl')


# ============ predict (★IP BAN回避: V15特徴ダンプを読むだけ・新規スクレイプしない★) ============
# V15(DailyPredict 8:00)が build_features した特徴量df を race毎に parquet ダンプする前提:
#   data/v15_feat_dump/{date}/{race_id}.parquet  (各馬1行・V15の145特徴 + race_id/horse_num/course/race_num)
# このダンプを s2b が読んで再スコア → netkeiba/JRDBへ二度アクセスしない = IP BANリスクゼロ・当日高精度。
# ★ get_horse_stats は db.netkeiba.com に馬毎アクセスするため、s2bが build_features を呼ぶと
#   ~360リクエストのフル再スクレイプ=IP BAN高リスク。 よって本モードは新規スクレイプを一切しない。★
FEAT_DUMP_DIR = os.path.join(DATA, 'v15_feat_dump')

import re as _re
def _parse_race_num(v):
    """ダンプの race_num は int(2) でも文字列('2R'/'2R '/'02') でもあり得る → 数字だけ抽出。"""
    try:
        m = _re.search(r'\d+', str(v))
        return int(m.group()) if m else 0
    except Exception:
        return 0

def predict_date(date, allow_scrape=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    import glob as _glob
    dump_dir = os.path.join(FEAT_DUMP_DIR, date)
    parquets = sorted(_glob.glob(os.path.join(dump_dir, '*.parquet')))
    if not parquets:
        print(f"[paper_s2b] V15特徴ダンプ無し ({dump_dir})。")
        print("  ★IP BAN回避のため新規スクレイプはしません★。 次のいずれか:")
        print("   (A推奨) DailyPredict に特徴ダンプ追加 → s2bが読むだけ=二度アクセスなし。")
        print("   (B) --allow-scrape で9時以降に独自フェッチ(get_horse_stats~360req=IP BANリスク・要承認)。")
        if not allow_scrape:
            return
        print("  [allow_scrape] 独自フェッチを許可。 V15(8:00)と十分離れた時刻で実行してください。")
        return _predict_via_scrape(date)
    out = open(_pred_path(date), 'w', encoding='utf-8'); n_ok = 0
    for pq in parquets:
        try:
            df = pd.read_parquet(pq)
            scores = score_s2b(df)
            order = [h for h, _ in sorted(scores.items(), key=lambda x: -x[1])]
            names = _name_map(df)  # {馬番: 馬名} (ダンプ内のみ・netkeiba非アクセス)
            r0 = df.iloc[0]
            rn = _parse_race_num(r0.get('race_num', 0))  # ダンプは '2R' 等の文字列のことがある → 数値抽出
            rec = {'date': date, 'race_id': str(r0.get('race_id', '')),
                   'course': str(r0.get('course', '')), 'race_num': rn,
                   'race_name': str(r0.get('race_name', '')), 'start_time': str(r0.get('start_time', '')),
                   'rk': f"{date}_{r0.get('course','')}_{rn}",
                   's2b_top6': order[:6],
                   's2b_scores': {int(h): round(float(s), 4) for h, s in scores.items()},
                   's2b_names': names,
                   'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
            out.write(json.dumps(rec, ensure_ascii=False) + '\n'); n_ok += 1
            print(f"  {rec['course']}{rec['race_num']}R s2b top6={order[:6]} (V15特徴ダンプ再利用)")
        except Exception as e:
            print(f"  [skip] {pq}: {e}")
    out.close()
    print(f"[paper_s2b] {date}: {n_ok}R 記録(V15特徴ダンプ再利用=新規アクセスなし) → {_pred_path(date)}")


def _predict_via_scrape(date):
    """(要承認・IP BANリスク) V15特徴ダンプが無い場合の独自フェッチ。default無効。"""
    from predict_core import load_models, parse_shutuba, build_features
    from jrdb_features import merge_jrdb_predict_features
    import daily_predict as dp
    model_data = load_models(); races = dp.fetch_race_list(date)
    if not races: print(f"[paper_s2b] {date} レースなし"); return
    out = open(_pred_path(date), 'w', encoding='utf-8'); n_ok = 0
    for r in races:
        rid = r['race_id']
        try:
            horses = parse_shutuba(rid)
            if not horses: continue
            df = build_features(horses, r, model_data, race_id=rid)
            try: df = merge_jrdb_predict_features(df, rid)
            except Exception: pass
            scores = score_s2b(df); order = [h for h, _ in sorted(scores.items(), key=lambda x: -x[1])]
            rec = {'date': date, 'race_id': rid, 'course': r.get('course', ''), 'race_num': r.get('race_num', 0),
                   'rk': f"{date}_{r.get('course','')}_{r.get('race_num',0)}", 's2b_top6': order[:6]}
            out.write(json.dumps(rec, ensure_ascii=False) + '\n'); n_ok += 1
        except Exception as e:
            print(f"  [skip] {rid}: {e}")
    out.close(); print(f"[paper_s2b] {date}: {n_ok}R (独自フェッチ)")


# ============ results 照合 ============
def results_date(date):
    pp = _pred_path(date)
    if not os.path.exists(pp): print(f"[paper_s2b] {pp} なし"); return
    # ★払戻源: 本番 daily_results が保存した全払戻JSONを最優先(go-forward・netkeiba非アクセス)。
    #   無ければ jra_payouts.csv(historical・5/17で停止)へfallback★
    pay = _payout_index_daily_full(date)
    src = 'daily_results_full'
    if not pay:
        pay = load_payout_index(); src = 'jra_payouts.csv(fallback)'
    print(f"[paper_s2b] payout source = {src} ({len(pay)} races indexed)")
    out = open(_res_path(date),'w',encoding='utf-8'); n=0
    for line in open(pp,encoding='utf-8'):
        rec = json.loads(line); key = rec['rk']
        if key not in pay: continue
        o = [int(x) for x in rec['s2b_top6']]
        if len(o) < 5: continue
        pm = pay[key]; bets = {}
        for name, fn in BETS:
            ret, pts = fn(o, pm); bets[name] = {'ret':ret, 'pts':pts}
        out.write(json.dumps({'rk':key,'bets':bets}, ensure_ascii=False)+'\n'); n+=1
    out.close()
    print(f"[paper_s2b] {date}: {n}R 照合 → {_res_path(date)}")


# ============ report 累積 ============
def report():
    files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_results.jsonl')]) if os.path.isdir(LOG_DIR) else []
    agg = {name:{'ret':0,'stake':0,'hit':0,'n':0} for name,_ in BETS}
    for fn in files:
        for line in open(os.path.join(LOG_DIR,fn),encoding='utf-8'):
            rec = json.loads(line)
            for name,_ in BETS:
                b = rec['bets'].get(name)
                if not b or b['pts']==0: continue
                a = agg[name]; a['ret']+=b['ret']; a['stake']+=100*b['pts']; a['hit']+=(b['ret']>0); a['n']+=1
    print(f"=== paper s2b 累積ROI (照合日数={len(files)}) ===")
    print(f"  leak-freeバックテスト参照: 単勝111.6% / 三連複top4box194.3% / 三連単1-2-5 229% / 馬連top3box146%")
    print(f"  {'券種':16s}{'ROI':>8s}{'的中率':>8s}{'N':>7s}{'収支':>12s}")
    for name,_ in BETS:
        a = agg[name]
        if a['n']==0: print(f"  {name:16s}{'--':>8s}{'--':>8s}{0:>7d}"); continue
        roi=a['ret']/a['stake'] if a['stake'] else 0; pnl=a['ret']-a['stake']
        warn = ' ★N<100信用不可' if a['n']<100 else ''
        print(f"  {name:16s}{roi*100:7.1f}%{a['hit']/a['n']*100:7.1f}%{a['n']:7d}{pnl:>11,}円{warn}")


# ============ from-oof (検証配線) ============
def from_oof(date):
    """検証用: leak-free OOF(s_s2b)から擬似pred-log生成(過去レースで results/report 配線テスト)。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    oof = pd.read_parquet(os.path.join(DATA,'v16_leakfree_s2b_oof.parquet'))  # leak-free s_s2b, _rk, horse_num, finish
    # _rk = date_course_kai_nichi_racenum → paper rk = date_course_racenum へ変換
    def conv(rk):
        p = rk.split('_'); return f"{p[0]}_{p[1]}_{p[4]}" if len(p)>=5 else rk
    oof = oof.copy(); oof['prk'] = oof['_rk'].map(conv)
    if date != 'ALL':
        oof = oof[oof['prk'].str.startswith(date)]
    out = open(_pred_path(date),'w',encoding='utf-8'); n=0
    for prk,g in oof.groupby('prk'):
        order=[int(x) for x in g.sort_values('s_s2b',ascending=False)['horse_num'].tolist()]
        out.write(json.dumps({'date':date,'rk':prk,'s2b_top6':order[:6]},ensure_ascii=False)+'\n'); n+=1
    out.close(); print(f"[from-oof] {n}R 擬似pred生成({date})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['predict','results','report','from-oof','notify-test1'])
    ap.add_argument('--date', default='')
    ap.add_argument('--allow-scrape', action='store_true', help='V15特徴ダンプ無時に独自フェッチ(IP BANリスク・要承認)')
    ap.add_argument('--resend', action='store_true', help='再送(タイトルに🔁を付け重複と区別)')
    ap.add_argument('--limit', type=int, default=0, help='送信レース数の上限(デザイン確認用・0=全件)')
    a = ap.parse_args()
    if a.mode=='predict': predict_date(a.date or time.strftime('%Y%m%d'), allow_scrape=a.allow_scrape)
    elif a.mode=='results': results_date(a.date or time.strftime('%Y%m%d'))
    elif a.mode=='report': report()
    elif a.mode=='from-oof': from_oof(a.date or 'ALL')
    elif a.mode=='notify-test1': notify_test1_date(a.date or time.strftime('%Y%m%d'), resend=a.resend, limit=a.limit)

if __name__=='__main__':
    main()
