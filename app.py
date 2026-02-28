import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, re
from bs4 import BeautifulSoup
import requests

MODEL_PATH = "keiba_model.pkl"
JOCKEY_PATH = "jockey_wr.json"

FEATURES = [
    "馬体重","場体重増減","斤量","馬齢","距離(m)",
    "競馬場コード_enc","芝ダート_enc","馬場状態_enc",
    "性別_enc","騎手勝率","前走着順"
]

COURSE_MAP = {
    "札幌":0,"函館":1,"福島":2,"新潟":3,"東京":4,
    "中山":5,"中京":6,"京都":7,"阪神":8,"小倉":9
}
SURFACE_MAP = {"芝":0, "ダート":1}
CONDITN_MAP = {"良":0, "稍重":1, "重":2, "不良":3}
SEX_MAP = {"牡":0, "牝":1, "セ":2}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_jockey():
    with open(JOCKEY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
jockey_dict = load_jockey()

def normalize_url(url):
    url = url.replace("race.sp.netkeiba.com", "race.netkeiba.com")
    url = re.sub(r"&rf=.*", "", url)
    return url

def scrape(race_url):
    race_url = normalize_url(race_url)
    res = requests.get(race_url, headers=HEADERS, timeout=10)
    res.encoding = "EUC-JP"
    soup = BeautifulSoup(res.content, "html.parser")
    race_name = soup.select_one("h1.RaceName")
    race_name = race_name.text.strip() if race_name else ""
    race_data_el = soup.select_one("div.RaceData01")
    race_data_text = race_data_el.text.strip() if race_data_el else ""
    dist_m = re.search(r"(\d{4})m", race_data_text)
    distance = int(dist_m.group(1)) if dist_m else 0
    if "ダート" in race_data_text:
        surface = "ダート"
    else:
        surface = "芝"
    condition = "良"
    for cond in ["不良","重","稍重","良"]:
        if cond in race_data_text:
            condition = cond
            break
    course_name = ""
    race_data2 = soup.select_one("div.RaceData02")
    race_data2_text = race_data2.text.strip() if race_data2 else ""
    for name in COURSE_MAP:
        if name in race_data2_text or name in race_name:
            course_name = name
            break
    if not course_name:
        for name in COURSE_MAP:
            if name in race_url:
                course_name = name
                break
    horses = []
    rows = soup.select("tr.HorseList")
    for row in rows:
        row_class = " ".join(row.get("class", []))
        if "Cancel" in row_class:
            continue
        tds = row.select("td")
        if len(tds) < 8:
            continue
        try:
            umaban = tds[1].text.strip()
            horse_name_el = row.select_one("span.HorseName a")
            horse_name = horse_name_el.text.strip() if horse_name_el else ""
            if not horse_name:
                continue
            barei = row.select_one("td.Barei")
            sex_age = barei.text.strip() if barei else ""
            sex = sex_age[0] if sex_age else "牡"
            age = int(sex_age[1:]) if len(sex_age) > 1 and sex_age[1:].isdigit() else 3
            kinryo_tds = row.select("td.Txt_C")
            kinryo = 55.0
            for kt in kinryo_tds:
                try:
                    kinryo = float(kt.text.strip())
                    if 40 < kinryo < 70:
                        break
                except:
                    continue
            jockey_el = row.select_one("td.Jockey a")
            jockey = jockey_el.text.strip() if jockey_el else ""
            weight_td = row.select_one("td.Weight")
            weight_text = weight_td.text.strip() if weight_td else ""
            bw_match = re.search(r"(\d+)\(([\+\-]?\d+)\)", weight_text)
            if bw_match:
                body_weight = int(bw_match.group(1))
                weight_diff = int(bw_match.group(2))
            else:
                bw_only = re.search(r"(\d+)", weight_text)
                body_weight = int(bw_only.group(1)) if bw_only else 480
                weight_diff = 0
            horses.append({"馬番": umaban, "馬名": horse_name, "性別": sex, "馬齢": age, "斤量": kinryo, "騎手": jockey, "馬体重": body_weight, "場体重増減": weight_diff})
        except Exception:
            continue
    return {"race_name": race_name, "distance": distance, "surface": surface, "condition": condition, "course": course_name, "horses": horses}

def predict(race_info):
    rows = []
    for h in race_info["horses"]:
        jwr = jockey_dict.get(h["騎手"], 0.05)
        row = {"馬体重": h["馬体重"], "場体重増減": h["場体重増減"], "斤量": h["斤量"], "馬齢": h["馬齢"], "距離(m)": race_info["distance"], "競馬場コード_enc": COURSE_MAP.get(race_info["course"], 4), "芝ダート_enc": SURFACE_MAP.get(race_info["surface"], 0), "馬場状態_enc": CONDITN_MAP.get(race_info["condition"], 0), "性別_enc": SEX_MAP.get(h["性別"], 0), "騎手勝率": jwr, "前走着順": 5}
        rows.append(row)
    df = pd.DataFrame(rows)[FEATURES]
    proba = model.predict_proba(df)[:, 1]
    return proba

st.set_page_config(page_title="競馬AI予想", layout="wide")
st.title("🏇 競馬AI予想アプリ")
url = st.text_input("netkeibaのレースURL（出馬表）を入力してください")
if st.button("予測する") and url:
    with st.spinner("データ取得中..."):
        try:
            race = scrape(url)
        except Exception as e:
            st.error(f"スクレイピングに失敗しました: {e}")
            st.stop()
    if not race["horses"]:
        st.warning("出馬データを取得できませんでした。URLを確認してください。")
        st.stop()
    st.subheader(race["race_name"])
    st.write(f"{race['course']}  {race['surface']}{race['distance']}m  馬場: {race['condition']}")
    with st.spinner("AI予測中..."):
        proba = predict(race)
    result = pd.DataFrame({"馬番": [h["馬番"] for h in race["horses"]], "馬名": [h["馬名"] for h in race["horses"]], "騎手": [h["騎手"] for h in race["horses"]], "AI勝率(%)": np.round(proba * 100, 2)})
    result = result.sort_values("AI勝率(%)", ascending=False).reset_index(drop=True)
    result.index = result.index + 1
    result.index.name = "AI順位"
    st.dataframe(result, use_container_width=True)
