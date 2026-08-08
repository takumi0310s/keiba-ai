# -*- coding: utf-8 -*-
"""
SCI ツールのフィルタ/類似度エンジンの忠実 Python 移植。
analysis.html の JS (calcSim/buildSim/styleOfRun/rowPassF/passF/firstStart) を
1:1 で移植。research/ruiji 専用・検証用。

使い方:
    import json
    from tool_filters import load_race, build_sim, filter_horses, DEFAULT_F
    race, horses = load_race("raw/20260808/scatter_....json")
    F = dict(DEFAULT_F); F["good"] = True
    passed = filter_horses(race, horses, F)   # 条件を満たす馬(num)のリスト
"""
import json, io

# JS: const R_DR=0,...,R_MO=14
R_DR, R_VEN, R_RN, R_DIST, R_SURF, R_RES, R_NH, R_TD, R_WIN, R_PASS, \
    R_MAE, R_AG3, R_AGR, R_CV, R_MO = range(15)

CLS_ORD = {'G1':1,'G2':2,'G3':3,'L':4,'OP':5,'3':6,'2':7,'1':8,'未':9,'新':10}

# フィルタ状態 F の初期値（JS resetF 準拠）
DEFAULT_F = {
    "good": False, "ky": False, "agari": 0, "grades": {},  # grades: {'G1':1,...}
    "sim": 0, "rank": 0, "td": 0, "td0": False,
    "oth": {},  # {'scls':1,'sdist':1,'sven':1,'y1':1,'turf':1,'dirt':1,'first':1}
}


def load_race(path):
    d = json.load(io.open(path, encoding="utf-8"))
    return d["race"], d["horses"]


# ── クラス判定（JS: clsOf） ────────────────────────────────────────
def cls_of(race_name):
    """レース名からクラスキー(G1/G2/G3/L/OP/3/2/1/未/新)を返す。"""
    s = race_name or ""
    # JS clsOf の実装に準拠（グレード/クラス語の照合）
    if "新馬" in s: return "新"
    if "未勝利" in s: return "未"
    if "ＧＩ" in s or "GI" in s or "(G1)" in s or "G1" in s: pass
    # グレード表記ゆれ
    for tok, key in [("GIII","G3"),("GII","G2"),("GI","G1"),
                     ("G3","G3"),("G2","G2"),("G1","G1"),
                     ("Ｇ３","G3"),("Ｇ２","G2"),("Ｇ１","G1")]:
        if tok in s: return key
    if "リステッド" in s or "(L)" in s or "Ｌ" in s: return "L"
    if "オープン" in s or "OP" in s or "オープン特別" in s: return "OP"
    if "3勝" in s or "１６００万" in s or "1600万" in s: return "3"
    if "2勝" in s or "１０００万" in s or "1000万" in s: return "2"
    if "1勝" in s or "５００万" in s or "500万" in s: return "1"
    return ""


# ── 類似度（JS: calcSim）──────────────────────────────────────────
def calc_sim_turf(tcv, tm, rcv, rm):
    """芝: max(0, 100*(1 - (|Δクッション| + |Δ含水|*0.5)/6.0))"""
    if rcv is None or rm is None:
        return 0.0
    return max(0.0, 100.0 * (1 - (abs(rcv - tcv) + abs(rm - tm) * 0.5) / 6.0))


def calc_sim_dirt(ty, rm):
    """ダ: max(0, 100*(1 - |Δ含水|/15.0))"""
    return max(0.0, 100.0 * (1 - abs((rm or 0) - ty) / 15.0))


# ── 脚質判定（JS: styleOfRun）─────────────────────────────────────
def style_of_run(passing, n_horses):
    """通過順(最終コーナー位置)/頭数 → 逃げ/先行/好位/中団/後方"""
    ps = str(passing or "").split("-")
    c1 = 0
    for v in reversed(ps):
        try:
            iv = int(v)
        except ValueError:
            iv = 0
        if iv > 0:
            c1 = iv
            break
    try:
        n = int(n_horses)
    except (ValueError, TypeError):
        n = 0
    if not c1 or not n:
        return ""
    if c1 == 1:
        return "逃げ"
    r = c1 / n
    if r <= 0.30: return "先行"
    if r <= 0.50: return "好位"
    if r <= 0.75: return "中団"
    return "後方"


# ── 類似レース構築（JS: buildSim）─────────────────────────────────
def build_sim(race, horses, top_n=10):
    """
    各馬の直近 top_n 走を race単位でまとめ、当該レース馬場との類似度 pct を付与。
    返り値: simRaces = [{dr,ven,rn,surf,dist,cv,mo,win,pct,runs:[{num,hn,wk,res,td,agr,ps,nh}]}]
    """
    is_turf = race["s"] == "芝"
    tx, ty = race.get("tx"), race.get("ty")
    date = int(race["date"])
    m = {}
    for h in horses:
        for r in (h.get("runs") or [])[:top_n]:
            if len(r) < 15:
                continue
            if r[R_MO] is None:
                continue
            if is_turf and r[R_CV] is None:
                continue
            if int(r[R_DR]) >= date:
                continue
            if is_turf:
                pct = calc_sim_turf(tx, ty, r[R_CV], r[R_MO])
            else:
                pct = calc_sim_dirt(ty, r[R_MO])
            key = f"{r[R_DR]}|{r[R_VEN]}|{r[R_RN]}|{r[R_SURF]}|{r[R_DIST]}"
            if key not in m:
                m[key] = {"dr": r[R_DR], "ven": r[R_VEN], "rn": r[R_RN],
                          "surf": r[R_SURF], "dist": r[R_DIST], "cv": r[R_CV],
                          "mo": r[R_MO], "win": r[R_WIN] or "", "pct": pct, "runs": []}
            m[key]["runs"].append({"num": h["num"], "hn": h["name"], "wk": h["waku"],
                                   "res": r[R_RES], "td": r[R_TD], "agr": r[R_AGR],
                                   "ps": r[R_PASS], "nh": r[R_NH]})
            if pct > m[key]["pct"]:
                m[key]["pct"] = pct
    return list(m.values())


# ── 行(走)単位フィルタ（JS: rowPassF）─────────────────────────────
def row_pass_f(q, race, F):
    if F["oth"].get("first") and q["res"] != 1:
        return False
    if F["rank"] > 0 and not (q["res"] is not None and q["res"] <= F["rank"]):
        return False
    if F["td"] > 0 and not (q["res"] == 1 or (q["td"] is not None and q["td"] <= F["td"])):
        return False
    if F["td0"] and not (q["res"] != 1 and q["td"] is not None and q["td"] < 0.05):
        return False
    if F["good"] and not ((q["res"] is not None and q["res"] <= 3)
                          or (q["td"] is not None and q["td"] <= 0.6)):
        return False
    if F["ky"]:
        kyt = race.get("kyt") or []
        if kyt and style_of_run(q["ps"], q["nh"]) not in kyt:
            return False
    if F["agari"] > 0 and not (q["agr"] is not None and q["agr"] <= F["agari"]):
        return False
    return True


# ── 馬単位フィルタ（JS: passF）。x=simRaces要素 ───────────────────
def race_pass_f(x, race, F, sim_horse=None):
    rs = ([q for q in x["runs"] if q["num"] == sim_horse] if sim_horse else x["runs"])
    if not rs:
        return False
    if F["sim"] > 0 and x["pct"] < F["sim"]:
        return False
    gk = list(F["grades"].keys())
    if gk and not F["grades"].get(cls_of(x["rn"])):
        return False
    oth = F["oth"]
    if oth.get("scls") and cls_of(x["rn"]) != cls_of(race.get("nfull") or ""):
        return False
    if oth.get("sdist") and not (x["surf"] == race["s"] and x["dist"] == race["dist"]):
        return False
    if oth.get("sven") and x["ven"] != race["v"]:
        return False
    if oth.get("y1") and int(x["dr"]) < int(race["date"]) - 10000:
        return False
    if oth.get("turf") and x["surf"] != "芝":
        return False
    if oth.get("dirt") and x["surf"] != "ダ":
        return False
    if not any(row_pass_f(q, race, F) for q in rs):
        return False
    return True


def filter_horses(race, horses, F, sim_horse=None):
    """
    フィルタ F を満たす『行』の馬番集合を返す（JS updateFunoDim の vis 準拠）。
    表示に出る = passF を通った simRace の rowPassF を通った行の馬番。
    """
    simraces = build_sim(race, horses)
    vis = set()
    for x in simraces:
        if race_pass_f(x, race, F, sim_horse):
            for q in x["runs"]:
                if row_pass_f(q, race, F):
                    vis.add(q["num"])
    return sorted(vis, key=lambda s: int(s) if str(s).isdigit() else 999)


# ── 昇級初戦（JS: firstStart / firstWinOrd）───────────────────────
def first_win_ord(horse, race):
    co = CLS_ORD.get(cls_of(race.get("nfull") or ""), 99)
    if co >= 9:
        return None
    has_cur = False
    w = None
    for r in (horse.get("runs") or []):
        if len(r) < 15:
            continue
        o = CLS_ORD.get(cls_of(r[R_RN]), 99)
        if (o <= 3) if co <= 3 else (o <= co):
            has_cur = True
            continue
        if r[R_RES] == 1 and (w is None or o < w):
            w = o
    return None if has_cur else w


def first_start(horse, race):
    return first_win_ord(horse, race) is not None
