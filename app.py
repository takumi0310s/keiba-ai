import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import re
import time

@st.cache_resource
def load_model():
    with open("keiba_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_jockey_wr():
    with open("jockey_wr.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
jockey_wr = load_jockey_wr()

FEATURES = [
    '馬体重', '場体重増減', '斤量', '馬齢', '距離(m)',
    '競馬場コード_enc', '芝ダート_enc', '馬場状態_enc',
    '性別_enc', '騎手勝率', '前走着順'
]

COURSE_MAP = {
    '札幌': 0, '函館': 1, '福島': 2, '新潟': 3, '東京': 4,
    '中山': 5, '中京': 6, '京都': 7, '阪神': 8, '小倉': 9
}
SURFACE_MAP = {'芝': 0, 'ダ': 1, '障': 2}
COND_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
SEX_MAP = {'牡': 0, '牝': 1, 'セ': 2, '騸': 2}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                  "Mobile/15E148 Safari/604.1"
}

def get_last_finish(horse_id):
    try:
        url = f"https://db.netkeiba.com/horse/{horse_id}/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="db_h_race_results")
        if table is None:
            return 5
        tbody = table.find("tbody")
        if tbody is None:
            return 5
        first_row = tbody.find("tr")
        if first_row is None:
            return 5
        tds = first_row.find_all("td")
        if len(tds) < 1:
            return 5
        thead = table.find("thead")
        finish_idx = 11
        if thead:
            ths = thead.find_all("th")
            for idx, th in enumerate(ths):
                if th.get_text(strip=True) == "着順":
                    finish_idx = idx
                    break
        if len(tds) > finish_idx:
            finish_text = tds[finish_idx].get_text(strip=True)
        else:
            finish_text = tds[0].get_text(strip=True)
        if finish_text.isdigit():
            return int(finish_text)
        else:
            return 5
    except Exception:
        return 5

def parse_shutuba(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")
    race_name_tag = soup.find("div", class_="RaceName")
    race_name = race_name_tag.get_text(strip=True) if race_name_tag else "レース名取得失敗"
    race_data01 = soup.find("div", class_="RaceData01")
    data01_text = race_data01.get_text(strip=True) if race_data01 else ""
    dist_match = re.search(r'(\d{4})m', data01_text)
    distance = int(dist_match.group(1)) if dist_match else 0
    if '芝' in data01_text:
        surface = '芝'
    elif 'ダ' in data01_text:
        surface = 'ダ'
    else:
        surface = '芝'
    cond_match = re.search(r'馬場:(\S+)', data01_text)
    condition = cond_match.group(1) if cond_match else '良'
    race_data02 = soup.find("div", class_="RaceData02")
    data02_text = race_data02.get_text(strip=True) if race_data02 else ""
    course_name = ""
    for name in COURSE_MAP.keys():
        if name in data02_text:
            course_name = name
            break
    rows = soup.select("tr.HorseList")
    horses = []
    horse_ids = []
    for row in rows:
        row_class = row.get("class", [])
        if "Cancel" in row_class:
            continue
        name_tag = row.select_one("span.HorseName a")
        if name_tag is None:
            continue
        horse_name = name_tag.get_text(strip=True)
        href = name_tag.get("href", "")
        hid_match = re.search(r'/horse/(\d+)', href)
        horse_id = hid_match.group(1) if hid_match else None
        info_tag = row.select_one("td.Barei") or row.select_one("span.Barei")
        if info_tag:
            sex_age_text = info_tag.get_text(strip=True)
        else:
            tds_all = row.find_all("td")
            sex_age_text = ""
            for td in tds_all:
                t = td.get_text(strip=True)
                if re.match(r'^[牡牝セ騸]\d+$', t):
                    sex_age_text = t
                    break
        sex = sex_age_text[0] if sex_age_text else '牡'
        age = int(sex_age_text[1:]) if sex_age_text and sex_age_text[1:].isdigit() else 3
        kinryo = 55.0
        tds = row.find_all("td")
        for td in tds:
            t = td.get_text(strip=True)
            try:
                val = float(t)
                if 48.0 <= val <= 62.0:
                    kinryo = val
                    break
            except ValueError:
                continue
        jockey_tag = row.select_one("td.Jockey a") or row.select_one("a[href*='jockey']")
        jockey_name = jockey_tag.get_text(strip=True) if jockey_tag else ""
        horse_weight = 480
        weight_diff = 0
        for td in tds:
            bw_text = td.get_text(strip=True)
            bw_match = re.search(r'(\d{3,})\(([\+\-]?\d+)\)', bw_text)
            if bw_match:
                w = int(bw_match.group(1))
                if 350 <= w <= 600:
                    horse_weight = w
                    weight_diff = int(bw_match.group(2))
                    break
        horses.append({
            '馬名': horse_name,
            '馬体重': horse_weight,
            '場体重増減': weight_diff,
            '斤量': kinryo,
            '馬齢': age,
            '距離(m)': distance,
            '競馬場コード_enc': COURSE_MAP.get(course_name, 4),
            '芝ダート_enc': SURFACE_MAP.get(surface, 0),
            '馬場状態_enc': COND_MAP.get(condition, 0),
            '性別_enc': SEX_MAP.get(sex, 0),
            '騎手勝率': jockey_wr.get(jockey_name, 0.05),
            '騎手名': jockey_name,
        })
        horse_ids.append(horse_id)
    return race_name, horses, horse_ids

st.title("🏇 競馬AI予想")
url_input = st.text_input("netkeibaの出馬表URLを入力")

if st.button("予想する") and url_input:
    url_input = url_input.replace("race.sp.netkeiba.com", "race.netkeiba.com")
    rid_match = re.search(r'race_id=(\d+)', url_input)
    if not rid_match:
        st.error("URLからrace_idを取得できませんでした")
        st.stop()
    race_id = rid_match.group(1)
    with st.spinner("出馬表を取得中..."):
        race_name, horses, horse_ids = parse_shutuba(race_id)
    if not horses:
        st.error("馬データを取得できませんでした。URLを確認してください。")
        st.stop()
    st.subheader(race_name)
    with st.spinner(f"前走着順を取得中...（{len(horses)}頭）"):
        progress_bar = st.progress(0)
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                last_finish = get_last_finish(hid)
                horse['前走着順'] = last_finish
            else:
                horse['前走着順'] = 5
            progress_bar.progress((i + 1) / len(horses))
            if i < len(horses) - 1:
                time.sleep(0.5)
        progress_bar.empty()
    df = pd.DataFrame(horses)
    X = df[FEATURES].values
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        scores = proba[:, 1]
    else:
        scores = proba[:, :3].sum(axis=1) if proba.shape[1] >= 3 else proba[:, 0]
    df['スコア'] = scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    display_cols = ['AI順位', '馬名', '騎手名', '前走着順', '騎手勝率', 'スコア']
    result_df = df[display_cols].copy()
    result_df['スコア'] = result_df['スコア'].map(lambda x: f"{x:.3f}")
    result_df['騎手勝率'] = result_df['騎手勝率'].map(lambda x: f"{x:.3f}")
    result_df = result_df.reset_index(drop=True)
    st.dataframe(result_df, use_container_width=True)
    top3 = df.head(3)
    st.markdown("### 🏆 AI推奨 TOP3")
    for _, row in top3.iterrows():
        st.write(f"**{int(row['AI順位'])}位: {row['馬名']}** "
                 f"（騎手: {row['騎手名']}｜前走{int(row['前走着順'])}着｜"
                 f"スコア: {row['スコア']:.3f}）")
